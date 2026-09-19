import math

import pytest
import torch

from koprm.train.loss import IGNORE_INDEX, prm_loss, step_probs


def _logits(pairs):
    return torch.tensor(pairs, dtype=torch.float32)


def ce(logit_pair, target):
    l0, l1 = logit_pair
    z = [l0, l1]
    m = max(z)
    lse = m + math.log(sum(math.exp(v - m) for v in z))
    return lse - z[target]


def test_masking_and_batch_mean():
    """L = CE(y, p_T) + mean over unmasked non-last steps; averaged over the batch."""
    logits = _logits([
        [[0.0, 1.0], [2.0, -1.0], [0.5, 0.5], [0.0, 0.0]],   # 3 real steps
        [[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0], [0.3, -0.7]],  # 4 real steps
    ])
    targets = torch.tensor([
        [1, IGNORE_INDEX, 1, IGNORE_INDEX],   # step0 kept, step1 masked, step2 = outcome
        [1, 1, 0, 0],                         # steps 0..2 kept, step3 = outcome
    ])
    n_steps = torch.tensor([3, 4])

    a = ce(logits[0, 2].tolist(), 1) + ce(logits[0, 0].tolist(), 1)          # |M| = 1
    b = ce(logits[1, 3].tolist(), 0) + (
        ce(logits[1, 0].tolist(), 1) + ce(logits[1, 1].tolist(), 1)
        + ce(logits[1, 2].tolist(), 0)) / 3                                   # |M| = 3
    expected = (a + b) / 2

    got = prm_loss(logits, targets, n_steps)
    assert got.item() == pytest.approx(expected, rel=1e-6)


def test_padding_after_n_steps_is_ignored():
    logits = _logits([[[0.0, 1.0], [2.0, -1.0], [50.0, -50.0]]])
    targets = torch.tensor([[1, 1, 7]])  # position 2 is padding with junk
    base = prm_loss(logits, targets, torch.tensor([2]))
    logits2 = logits.clone()
    logits2[0, 2] = torch.tensor([-99.0, 99.0])
    got = prm_loss(logits2, targets, torch.tensor([2]))
    assert got.item() == pytest.approx(base.item(), rel=1e-9)


def test_m_empty_is_outcome_only():
    """One-step solution, and a solution whose only non-last step is masked."""
    logits = _logits([
        [[0.0, 1.0], [5.0, -5.0]],
        [[0.0, 1.0], [1.0, -1.0]],
    ])
    targets = torch.tensor([[1, IGNORE_INDEX], [IGNORE_INDEX, 0]])
    n_steps = torch.tensor([1, 2])
    expected = (ce(logits[0, 0].tolist(), 1) + ce(logits[1, 1].tolist(), 0)) / 2
    parts = prm_loss(logits, targets, n_steps, return_parts=True)
    assert parts.loss.item() == pytest.approx(expected, rel=1e-6)
    assert parts.n_step_terms == 0
    assert parts.steps.item() == pytest.approx(0.0, abs=1e-9)
    # identical to the outcome-only ablation on this batch
    assert prm_loss(logits, targets, n_steps, outcome_only=True).item() == pytest.approx(
        expected, rel=1e-6)


def test_outcome_only_ignores_step_terms():
    logits = _logits([[[0.0, 3.0], [1.0, -1.0], [0.0, 0.0]]])
    t_a = torch.tensor([[0, 1, IGNORE_INDEX]])
    t_b = torch.tensor([[1, 1, IGNORE_INDEX]])
    n = torch.tensor([2])
    assert prm_loss(logits, t_a, n, outcome_only=True).item() == pytest.approx(
        prm_loss(logits, t_b, n, outcome_only=True).item())
    assert prm_loss(logits, t_a, n).item() != pytest.approx(prm_loss(logits, t_b, n).item())


def test_soft_targets_match_hard_at_zero_one():
    logits = _logits([[[0.0, 1.0], [2.0, -1.0], [0.4, 0.1], [0.0, 0.0]]])
    targets = torch.tensor([[1, 0, 1, 0]])
    n_steps = torch.tensor([4])
    soft = torch.tensor([[1.0, 0.0, 1.0, float("nan")]])
    hard = prm_loss(logits, targets, n_steps)
    got = prm_loss(logits, targets, n_steps, soft_targets=soft)
    assert got.item() == pytest.approx(hard.item(), rel=1e-5)


def test_soft_targets_nan_is_masked():
    logits = _logits([[[0.0, 1.0], [2.0, -1.0], [0.4, 0.1]]])
    targets = torch.tensor([[1, 1, 1]])
    n = torch.tensor([3])
    soft_all = torch.tensor([[0.7, 0.3, float("nan")]])
    soft_one = torch.tensor([[0.7, float("nan"), float("nan")]])
    outcome = ce(logits[0, 2].tolist(), 1)

    def soft_ce(pair, p):
        lp = torch.log_softmax(torch.tensor(pair), dim=-1).tolist()
        return -(p * lp[1] + (1 - p) * lp[0])

    exp_all = outcome + (soft_ce(logits[0, 0].tolist(), 0.7)
                         + soft_ce(logits[0, 1].tolist(), 0.3)) / 2
    exp_one = outcome + soft_ce(logits[0, 0].tolist(), 0.7)
    assert prm_loss(logits, targets, n, soft_targets=soft_all).item() == pytest.approx(
        exp_all, rel=1e-6)
    assert prm_loss(logits, targets, n, soft_targets=soft_one).item() == pytest.approx(
        exp_one, rel=1e-6)


def test_masked_outcome_raises():
    logits = _logits([[[0.0, 1.0], [1.0, 0.0]]])
    with pytest.raises(ValueError):
        prm_loss(logits, torch.tensor([[1, IGNORE_INDEX]]), torch.tensor([2]))


def test_step_probs():
    p = step_probs(_logits([[[0.0, 0.0], [-10.0, 10.0]]]))
    assert p.shape == (1, 2)
    assert p[0, 0].item() == pytest.approx(0.5)
    assert p[0, 1].item() > 0.999
