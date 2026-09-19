"""§6.1 loss.

    L = CE(y, p_T) + (1/|M|) * sum_{t in M} CE(l_t, p_t)
    M = unmasked steps, last step T excluded.  If M is empty only the first term.

p_t is the positive-class probability of the 2-class logits at the SEP token of
step t.  The step term is averaged *inside* a solution so that long solutions and
long zero tails do not dominate; solutions are then averaged over the batch.

The soft variant (§6.2 ablation) replaces the hard step labels l_t by the
teacher's raw probabilities sigma(z_t); the outcome term stays hard, because y is
observed, not estimated.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


@dataclass
class LossParts:
    loss: torch.Tensor
    outcome: torch.Tensor
    steps: torch.Tensor
    n_step_terms: int


def step_probs(step_logits: torch.Tensor) -> torch.Tensor:
    """P(positive) from 2-class logits, in float32."""
    return torch.softmax(step_logits.float(), dim=-1)[..., 1]


def _soft_ce(logits: torch.Tensor, p_pos: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits.float(), dim=-1)
    return -(p_pos * logp[..., 1] + (1.0 - p_pos) * logp[..., 0])


def prm_loss(
    step_logits: torch.Tensor,
    targets: torch.Tensor,
    n_steps: torch.Tensor,
    soft_targets: torch.Tensor | None = None,
    outcome_only: bool = False,
    ignore_index: int = IGNORE_INDEX,
    return_parts: bool = False,
) -> torch.Tensor | LossParts:
    """
    step_logits   [B, S, 2]  logits at the SEP positions (padded)
    targets       [B, S]     0/1 labels, `ignore_index` for masked steps and padding.
                             targets[b, n_steps[b]-1] must hold the outcome y.
    n_steps       [B]        number of real steps per solution (>=1)
    soft_targets  [B, S]     optional probabilities in [0, 1]; non-finite entries
                             (NaN) are treated as masked.  Used for the step term
                             only.
    """
    if step_logits.dim() != 3 or step_logits.shape[-1] != 2:
        raise ValueError(f"step_logits must be [B, S, 2], got {tuple(step_logits.shape)}")
    logits = step_logits.float()
    B, S, _ = logits.shape
    device = logits.device
    n_steps = n_steps.to(device=device, dtype=torch.long)
    targets = targets.to(device=device, dtype=torch.long)
    if torch.any(n_steps < 1):
        raise ValueError("every solution needs at least one step")
    b_idx = torch.arange(B, device=device)
    last = n_steps - 1

    # --- outcome term: CE(y, p_T)
    outcome_logits = logits[b_idx, last]                     # [B, 2]
    outcome_targets = targets[b_idx, last]                   # [B]
    if torch.any(outcome_targets == ignore_index):
        raise ValueError("the last step target (the outcome y) must never be masked")
    outcome_ce = F.cross_entropy(outcome_logits, outcome_targets, reduction="none")  # [B]

    if outcome_only:
        loss = outcome_ce.mean()
        zero = torch.zeros((), device=device, dtype=loss.dtype)
        return LossParts(loss, outcome_ce.mean().detach(), zero, 0) if return_parts else loss

    # --- step term over M (unmasked, last step excluded)
    pos = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
    in_m = pos < last.unsqueeze(1)
    if soft_targets is None:
        in_m = in_m & (targets != ignore_index)
        # positions outside M contribute nothing; neutralise their (padded) targets
        safe = torch.where(in_m, targets, torch.zeros_like(targets))
        ce = F.cross_entropy(
            logits.reshape(-1, 2), safe.reshape(-1), reduction="none"
        ).reshape(B, S)
    else:
        soft = soft_targets.to(device=device, dtype=torch.float32)
        in_m = in_m & torch.isfinite(soft)
        ce = _soft_ce(logits, torch.nan_to_num(soft, nan=0.0))

    m = in_m.to(ce.dtype)
    counts = m.sum(dim=1)                                    # |M| per solution
    step_sum = (ce * m).sum(dim=1)
    step_term = torch.where(counts > 0, step_sum / counts.clamp_min(1.0),
                            torch.zeros_like(step_sum))
    per_solution = outcome_ce + step_term
    loss = per_solution.mean()
    if return_parts:
        n_terms = int(counts.sum().item())
        denom = max(int((counts > 0).sum().item()), 1)
        return LossParts(
            loss,
            outcome_ce.mean().detach(),
            (step_term.sum() / denom).detach(),
            n_terms,
        )
    return loss
