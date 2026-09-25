"""Final-report extras mapped to the proposal (연구계획서): CPU only, no new generation.

Three analyses, all re-reading files that earlier runs already saved.

  adaptive     연구 내용 ②·④. Does the PRM carry an allocation signal? Completions are drawn
               one at a time from the stored 64 and sampling stops once a stop signal
               reaches tau. PRM stop signals (PRM-weighted vote share, best single score)
               are compared with the PRM-free one (unweighted agreement share,
               adaptive-consistency) at equal average budget, with the answer rule held
               fixed, plus fixed N. Averaged over several completion orders; the cost of
               matching fixed@64 is estimated with tau picked on held-out problem halves.
  levels       연구 내용 ① (고난도). naive/weighted/maj@N broken down by MATH level 1-5.
               (`adaptive` also prints it; this subcommand is the cheap one.)
  error-types  연구 내용 ① (오류 유형). ProcessBench first-error position as a proxy for the
               three error types of the proposal: first step (문제 이해·초기 설정), middle
               steps (다단계 논리 전개), last step (최종 답안 도출). Distribution per split,
               and per scorer how often each type is located exactly / missed.

Examples (paths follow HANDOFF.md; adjust to what is on the server). `koprm.eval.bon`
sets HF_HUB_OFFLINE=1, so export HF_HUB_OFFLINE=0 for the --dataset commands:

    # 8B student step scores already exist for EXAONE and Qwen2.5-3B:
    #   data/eval/math500_7b/qwen3-8b_B_24k_soft_ep3_<generator>.scores.jsonl
    # for another generator, save them first (GPU, one pass)
    python -m koprm.eval.bon --dataset ENSEONG/ko-ko-math-500-test-Qwen2.5-1.5B-Instruct-bon \\
        --dataset-config ENSEONG_ko-math-500-test--T-0.8--top_p-1.0--n-64--seed-0--agg_strategy-last \\
        --scorer data/ckpt/qwen3-8b_B_24k_soft/epoch3 --device cuda:0 --agg last \\
        --save-scores data/eval/scores/qwen15b_qwen3-8b_24k_soft.jsonl \\
        --out data/eval/qwen15b_qwen3-8b_24k_soft.json

    python scripts/plan_extras.py adaptive \\
        --dataset ENSEONG/ko-ko-math-500-test-EXAONE-4.0-1.2B-bon \\
        --scorer existing=existing \\
        --scorer qwen3-8b=data/eval/math500_7b/qwen3-8b_B_24k_soft_ep3_EXAONE-4.0-1.2B.scores.jsonl \\
        --out data/reports/adaptive_exaone.json

    python scripts/plan_extras.py levels --dataset ... --scorer ... --out data/reports/levels_exaone.json

    python scripts/plan_extras.py error-types --rows data/shift/pb_rows.jsonl \\
        --scorer prm72b=data/teacher/pb_ko_prm72b.jsonl \\
        --scorer qwen3-8b=data/shift/pb_probs/qwen3-8b_B_48k_soft_ko.jsonl \\
        --out data/reports/error_types_ko.json

`--scorer NAME=PATH`: PATH is `existing` (scores stored in the dataset), a
`koprm.eval.bon --save-scores` jsonl (adaptive/levels), or for error-types either a
`koprm.shift first-error --save-probs` jsonl ({id, probs}) or a teacher jsonl
({id, teacher_logodds}).
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

N_MAX = 64
SEED0_CONFIG = "ENSEONG_ko-math-500-test--T-0.8--top_p-1.0--n-64--seed-0--agg_strategy-last"


# =================================================================== pure helpers
def aggregate(scores, agg="last"):
    if not scores:
        return 0.0
    if agg == "last":
        return float(scores[-1])
    if agg == "min":
        return float(min(scores))
    if agg == "mean":
        return float(sum(scores) / len(scores))
    if agg == "prod":
        p = 1.0
        for s in scores:
            p *= float(s)
        return p
    raise ValueError(agg)


def vote(gids, sc):
    """Return (weighted winner, naive winner index, majority winner, weighted share)."""
    wsum, cnt = {}, {}
    for g, s in zip(gids, sc):
        wsum[g] = wsum.get(g, 0.0) + s
        cnt[g] = cnt.get(g, 0) + 1
    first = {}
    for i, g in enumerate(gids):
        first.setdefault(g, i)
    gw = max(wsum, key=lambda g: (wsum[g], -first[g]))
    gm = max(cnt, key=lambda g: (cnt[g], -first[g]))
    best = max(range(len(sc)), key=lambda j: (sc[j], -j))
    tot = sum(wsum.values())
    share = wsum[gw] / tot if tot > 0 else 0.0
    return gw, best, gm, share


def fixed_n(problems, n):
    """problems: list of {gids, correct, agg}. Accuracy of weighted/naive/maj@n."""
    w = nv = m = 0
    for p in problems:
        g, sc = p["gids"][:n], p["agg"][:n]
        gw, best, gm, _ = vote(g, sc)
        w += p["correct"][gw]
        nv += p["correct"][g[best]]
        m += p["correct"][gm]
    k = len(problems)
    return {"weighted": w / k, "naive": nv / k, "maj": m / k}


# ------------------------------------------------------------------ adaptive allocation
# Each rule is (stop signal, answer). Every PRM rule has a PRM-free partner:
#   agree>maj  stop on the unweighted agreement share, majority answer (PRM 없음,
#              adaptive-consistency)
#   agree>prm  same stop, PRM-weighted answer (PRM only in the answer)
#   prm>prm    stop on the PRM-weighted share, weighted answer (PRM in the stop too)
#   top>prm    stop once the best single PRM score reaches tau, weighted answer
# prm>prm vs agree>prm (and top>prm vs agree>prm) differ only in the stop signal, so their
# gap at equal budget is the PRM's allocation signal; prm>prm vs agree>maj is the PRM's
# whole contribution.
RULES = {"agree>maj": ("agree", "maj"), "agree>prm": ("agree", "w"),
         "prm>prm": ("wshare", "w"), "top>prm": ("top", "w")}
CONTRASTS = [("prm>prm", "agree>prm"), ("top>prm", "agree>prm"), ("prm>prm", "agree>maj"),
             ("top>prm", "agree>maj"), ("prm>prm", "fixed-w"), ("top>prm", "fixed-w"),
             ("agree>maj", "fixed-maj")]
BUDGETS = (6, 8, 12, 16, 24)


def prefix_stats(gids, sc, correct):
    """Per prefix length k=1..n: correctness of the weighted / majority / naive winner and
    the three stop signals. Ties break toward the earliest completion, as in vote()."""
    wsum, cnt, first = {}, {}, {}
    tot, best = 0.0, 0
    out = {k: [] for k in ("w", "maj", "naive", "wshare", "agree", "top")}
    for i, (g, s) in enumerate(zip(gids, sc)):
        wsum[g] = wsum.get(g, 0.0) + s
        cnt[g] = cnt.get(g, 0) + 1
        first.setdefault(g, i)
        tot += s
        if s > sc[best]:
            best = i
        gw = max(wsum, key=lambda x: (wsum[x], -first[x]))
        gm = max(cnt, key=lambda x: (cnt[x], -first[x]))
        out["w"].append(correct[gw])
        out["maj"].append(correct[gm])
        out["naive"].append(correct[gids[best]])
        out["wshare"].append(wsum[gw] / tot if tot > 0 else 0.0)
        out["agree"].append(cnt[gm] / (i + 1))
        out["top"].append(sc[best])
    return out


def build_tensors(problems, n_orders, seed=0):
    """Arrays [problem, order, k-1]. Order 0 is the stored order, the others random
    permutations of the completions, so one lucky draw order does not decide the result."""
    import numpy as np

    n = len(problems[0]["gids"])
    if any(len(p["gids"]) != n for p in problems):
        raise SystemExit("adaptive: every problem needs the same number of completions")
    rng = random.Random(seed)
    keys = ("w", "maj", "naive", "wshare", "agree", "top")
    T = {k: np.zeros((len(problems), n_orders, n)) for k in keys}
    for pi, p in enumerate(problems):
        for r in range(n_orders):
            order = list(range(n))
            if r:
                rng.shuffle(order)
            st = prefix_stats([p["gids"][i] for i in order], [p["agg"][i] for i in order],
                              p["correct"])
            for k in keys:
                T[k][pi, r] = st[k]
    return T


def stop_points(signal, tau, k_min):
    """First k >= k_min whose signal reaches tau (else n), per [problem, order]."""
    import numpy as np

    n = signal.shape[-1]
    hit = signal[..., k_min - 1:] >= tau
    return np.where(hit.any(-1), hit.argmax(-1) + k_min, n)


def rule_runs(T, rule, k_min):
    """One run per tau: per-problem mean samples and correctness over orders. Taus are the
    signal's own quantiles because PRM scores of different scorers live on different
    scales; runs are compared at equal budget, never at equal tau."""
    import numpy as np

    sig, ans = RULES[rule]
    S = T[sig]
    taus = sorted(set(np.quantile(S[..., k_min - 1:], np.linspace(0, 1, 201)).tolist()))
    runs = []
    for tau in taus + [math.inf]:
        k = stop_points(S, tau, k_min)
        ok = np.take_along_axis(T[ans], (k - 1)[..., None], -1)[..., 0]
        runs.append({"tau": tau, "k": k.mean(1), "ok": ok.mean(1)})
    return runs


def fixed_runs(T, ans):
    import numpy as np

    P, _, n = T[ans].shape
    return [{"tau": None, "k": np.full(P, float(m)), "ok": T[ans][:, :, m - 1].mean(1)}
            for m in range(1, n + 1)]


def boot_weights(n, iters=1000, seed=0):
    """[iters, n] resampling weights (counts / n) for a problem-level bootstrap."""
    import numpy as np

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iters, n))
    W = np.zeros((iters, n))
    np.add.at(W, (np.arange(iters)[:, None], idx), 1.0)
    return W / n


def acc_at_budget(runs, budget, W=None):
    """Accuracy at an average budget, linearly interpolated along the (avg samples, acc)
    curve. Avg samples never decreases with tau, so the curve is ordered. With W, one
    value per bootstrap resample (the curve is rebuilt on each resample)."""
    import numpy as np

    K = np.stack([r["k"] for r in runs])
    O = np.stack([r["ok"] for r in runs])
    if W is None:
        return float(np.interp(budget, K.mean(1), O.mean(1)))
    xs, ys = K @ W.T, O @ W.T
    return np.array([np.interp(budget, xs[:, b], ys[:, b]) for b in range(W.shape[0])])


def cv_match_fixed(runs, fixed_ok, margin, reps=50, seed=0):
    """Samples needed to stay within `margin` of fixed@64, without picking tau on the
    problems it is scored on: repeated 2-fold split, cheapest qualifying tau chosen on one
    half, average samples and accuracy gap read on the other."""
    import numpy as np

    rng = np.random.default_rng(seed)
    K = np.stack([r["k"] for r in runs])
    O = np.stack([r["ok"] for r in runs])
    P = len(fixed_ok)
    ks, gaps = [], []
    for _ in range(reps):
        perm = rng.permutation(P)
        for tr, te in ((perm[:P // 2], perm[P // 2:]), (perm[P // 2:], perm[:P // 2])):
            good = np.flatnonzero(O[:, tr].mean(1) >= fixed_ok[tr].mean() - margin)
            j = good[np.argmin(K[good][:, tr].mean(1))]   # tau=inf always qualifies
            ks.append(K[j, te].mean())
            gaps.append(O[j, te].mean() - fixed_ok[te].mean())
    gaps = np.sort(gaps)
    return {"avg_samples": float(np.mean(ks)), "acc_gap": float(np.mean(gaps)),
            "acc_gap_p5_p95": [float(gaps[int(0.05 * len(gaps))]),
                               float(gaps[int(0.95 * len(gaps)) - 1])]}


def bootstrap_diff(a, b, iters=1000, seed=0):
    """Paired problem-level bootstrap CI for mean(a) - mean(b)."""
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] - b[i] for i in idx) / n)
    diffs.sort()
    return (sum(x - y for x, y in zip(a, b)) / n, diffs[int(0.025 * iters)], diffs[int(0.975 * iters) - 1])


def fixed_n_per_problem(problems, n, kind="naive"):
    """Per-problem 0/1 correctness of the naive or weighted pick among the first n."""
    out = []
    for p in problems:
        g = p["gids"][:n]
        gw, best, _, _ = vote(g, p["agg"][:n])
        out.append(p["correct"][g[best]] if kind == "naive" else p["correct"][gw])
    return out


def error_type(label, n_steps):
    if label < 0:
        return None
    if label == 0:
        return "first"   # 문제 이해·초기 설정
    if label == n_steps - 1:
        return "last"    # 최종 답안 도출
    return "middle"      # 다단계 논리 전개


TYPE_KO = {"first": "첫 단계(문제 이해·초기 설정)", "middle": "중간 단계(다단계 논리 전개)",
           "last": "마지막 단계(최종 답안 도출)"}


def predict_first_error(probs, threshold=0.5):
    for i, p in enumerate(probs):
        if float(p) < threshold:
            return i
    return -1


def error_type_table(rows, preds_by_scorer):
    """rows: [{id, split, label, n_steps}], preds_by_scorer: {name: {id: pred}}."""
    out = {"distribution": {}, "by_split": {}, "scorers": {}}
    wrong = [r for r in rows if r["label"] >= 0]
    for t in ("first", "middle", "last"):
        out["distribution"][t] = sum(1 for r in wrong if error_type(r["label"], r["n_steps"]) == t)
    for sp in sorted({r["split"] for r in wrong}):
        rs = [r for r in wrong if r["split"] == sp]
        out["by_split"][sp] = {t: sum(1 for r in rs if error_type(r["label"], r["n_steps"]) == t)
                               for t in ("first", "middle", "last")}
    for name, preds in preds_by_scorer.items():
        res = {}
        for t in ("first", "middle", "last"):
            rs = [r for r in wrong if error_type(r["label"], r["n_steps"]) == t and r["id"] in preds]
            n = len(rs)
            exact = sum(1 for r in rs if preds[r["id"]] == r["label"])
            early = sum(1 for r in rs if 0 <= preds[r["id"]] < r["label"])
            late = sum(1 for r in rs if preds[r["id"]] > r["label"])
            miss = sum(1 for r in rs if preds[r["id"]] == -1)
            res[t] = {"n": n, "exact": exact / n if n else None, "early": early / n if n else None,
                      "late": late / n if n else None, "missed": miss / n if n else None}
        out["scorers"][name] = res
    return out


# =================================================================== loading (koprm)
def load_problems(args, scorer_path):
    from koprm.eval.bon import (
        answer_groups,
        existing_scores,
        load_hf_rows,
        load_jsonl_rows,
        load_scores,
    )
    from koprm.paths import EVAL

    rows = (load_hf_rows(args.dataset, args.dataset_config, args.split, args.limit)
            if args.dataset else load_jsonl_rows(args.jsonl, args.limit))
    scores = existing_scores(rows) if scorer_path == "existing" else load_scores(scorer_path, rows)
    groups = answer_groups(rows, timeout=3.0, cache_dir=EVAL / "cache", verbose=False)
    levels = load_levels(args, rows)
    probs = []
    for r, s, g, lv in zip(rows, scores, groups, levels):
        probs.append({"id": r["problem_id"], "level": lv, "gids": g["gids"],
                      "correct": [1 if c else 0 for c in g["correct"]],
                      "agg": [aggregate(x, args.agg) for x in s]})
    return probs


def load_levels(args, rows):
    lv = [r.get("level") for r in rows]
    if any(v is not None for v in lv) or not args.levels_from:
        return [norm_level(v) for v in lv]
    by_id, seq = {}, []
    with open(args.levels_from, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                key = d.get("unique_id") or d.get("id") or d.get("problem_id")
                by_id[key] = d.get("level")
                seq.append(d.get("level"))
    out = []
    for i, r in enumerate(rows):
        v = by_id.get(r["problem_id"])
        if v is None and len(seq) == len(rows):
            v = seq[i]
        out.append(norm_level(v))
    return out


def norm_level(v):
    if v is None:
        return None
    s = str(v).lower().replace("level", "").strip()
    try:
        return int(s)
    except ValueError:
        return None


def parse_scorers(items):
    out = []
    for it in items:
        name, _, path = it.partition("=")
        if not path:
            raise SystemExit(f"--scorer expects NAME=PATH, got {it!r}")
        out.append((name, path))
    return out


# =================================================================== commands
def level_table(problems, n_list=(1, 16, 64)):
    lv = sorted({p["level"] for p in problems if p["level"] is not None})
    out = {}
    for L in lv:
        ps = [p for p in problems if p["level"] == L]
        out[L] = {"n_problems": len(ps),
                  "pass@64": sum(1 for p in ps if any(p["correct"][g] for g in p["gids"])) / len(ps)}
        for n in n_list:
            out[L][f"@{n}"] = fixed_n(ps, n)
    return out


def level_diffs(base, other, n=N_MAX):
    """Per level, paired bootstrap of (other - base) for naive and weighted @n. Both
    problem lists come from the same dataset rows, so they align by position."""
    out = {}
    for L in sorted({p["level"] for p in base if p["level"] is not None}):
        ix = [i for i, p in enumerate(base) if p["level"] == L]
        out[L] = {}
        for kind in ("naive", "weighted"):
            a = fixed_n_per_problem([other[i] for i in ix], n, kind)
            b = fixed_n_per_problem([base[i] for i in ix], n, kind)
            out[L][kind] = bootstrap_diff(a, b)
    return out


def cmd_levels(args):
    probs = {name: load_problems(args, path) for name, path in parse_scorers(args.scorer)}
    res = {name: level_table(ps) for name, ps in probs.items()}
    names = list(probs)
    diffs = {f"{o}-{names[0]}": level_diffs(probs[names[0]], probs[o]) for o in names[1:]}
    write(args.out, {"levels": res, f"diff@{N_MAX}": diffs})
    for name, tab in res.items():
        print(f"\n## {name}  (weighted / naive / maj, pass@64)")
        print("| level | n | @1 naive | @16 naive | @64 naive | @64 weighted | @64 maj | pass@64 |")
        print("|---|---|---|---|---|---|---|---|")
        for L, d in tab.items():
            print(f"| {L} | {d['n_problems']} | {d['@1']['naive']:.3f} | {d['@16']['naive']:.3f} | "
                  f"{d['@64']['naive']:.3f} | {d['@64']['weighted']:.3f} | {d['@64']['maj']:.3f} | "
                  f"{d['pass@64']:.3f} |")
    for key, dd in diffs.items():
        print(f"\n## {key} @{N_MAX}, paired bootstrap 95% CI")
        print("| level | naive diff | weighted diff |\n|---|---|---|")
        for L, d in dd.items():
            print("| {} | {:+.3f} [{:+.3f}, {:+.3f}] | {:+.3f} [{:+.3f}, {:+.3f}] |".format(
                L, *d["naive"], *d["weighted"]))


def adaptive_one(probs, args):
    import numpy as np

    T = build_tensors(probs, args.orders)
    runs = {r: rule_runs(T, r, args.k_min) for r in RULES}
    runs["fixed-w"] = fixed_runs(T, "w")
    runs["fixed-maj"] = fixed_runs(T, "maj")
    W = boot_weights(len(probs))
    at_budget = {}
    for B in BUDGETS:
        point = {r: acc_at_budget(rs, B) for r, rs in runs.items()}
        boot = {r: acc_at_budget(rs, B, W) for r, rs in runs.items()}
        con = {}
        for a, b in CONTRASTS:
            d = np.sort(boot[a] - boot[b])
            con[f"{a} - {b}"] = [point[a] - point[b], float(d[int(0.025 * len(d))]),
                                 float(d[int(0.975 * len(d)) - 1])]
        at_budget[B] = {"acc": point, "contrasts": con}
    fixed64 = runs["fixed-w"][-1]["ok"]
    # each rule against fixed@64 with its own answer rule, so tau=inf always qualifies
    own = {"w": fixed64, "maj": runs["fixed-maj"][-1]["ok"]}
    answer = {**{r: a for r, (_, a) in RULES.items()}, "fixed-w": "w", "fixed-maj": "maj"}
    cv = {r: cv_match_fixed(runs[r], own[a], args.margin) for r, a in answer.items()}
    # where the budget goes: per level, at the run whose average is closest to 16
    levels = sorted({p["level"] for p in probs if p["level"] is not None})
    alloc = {}
    for r in RULES:
        run = min(runs[r], key=lambda x: abs(x["k"].mean() - 16))
        alloc[r] = {"avg_samples": float(run["k"].mean()), "acc": float(run["ok"].mean()),
                    "per_level": {L: {"avg_samples": float(np.mean([k for k, p in zip(run["k"], probs)
                                                                     if p["level"] == L])),
                                      "acc": float(np.mean([o for o, p in zip(run["ok"], probs)
                                                            if p["level"] == L]))}
                                  for L in levels}}
    curves = {r: [{"tau": x["tau"], "avg_samples": float(x["k"].mean()), "acc": float(x["ok"].mean())}
                  for x in rs] for r, rs in runs.items()}
    return {"orders": args.orders, "k_min": args.k_min, "margin": args.margin,
            "fixed64": {"weighted": float(fixed64.mean()),
                        "maj": float(runs["fixed-maj"][-1]["ok"].mean())},
            "at_budget": at_budget, "cv_match_fixed64": cv, "alloc_at_16": alloc,
            "curves": curves, "levels": level_table(probs)}


def cmd_adaptive(args):
    res = {name: adaptive_one(load_problems(args, path), args)
           for name, path in parse_scorers(args.scorer)}
    write(args.out, res)
    cols = list(RULES) + ["fixed-w", "fixed-maj"]
    for name, r in res.items():
        f = r["fixed64"]
        print(f"\n## {name}: fixed@64 weighted {f['weighted']:.3f}, maj {f['maj']:.3f} "
              f"({r['orders']} completion orders, k_min {r['k_min']})")
        print("\naccuracy at equal average budget\n| budget | " + " | ".join(cols) + " |")
        print("|---" * (len(cols) + 1) + "|")
        for B, d in r["at_budget"].items():
            print(f"| {B} | " + " | ".join(f"{d['acc'][c]:.3f}" for c in cols) + " |")
        keys = list(next(iter(r["at_budget"].values()))["contrasts"])
        print("\ncontrasts at equal budget, paired bootstrap 95% CI\n| budget | "
              + " | ".join(keys) + " |")
        print("|---" * (len(keys) + 1) + "|")
        for B, d in r["at_budget"].items():
            print(f"| {B} | " + " | ".join("{:+.3f} [{:+.3f}, {:+.3f}]".format(*d["contrasts"][k])
                                           for k in keys) + " |")
        print(f"\nsamples to stay within {r['margin']} of fixed@64 with the same answer rule "
              "(tau picked on the other half, repeated 2-fold)")
        print("| rule | avg samples | held-out acc gap | gap p5..p95 |\n|---|---|---|---|")
        for rule, c in r["cv_match_fixed64"].items():
            lo, hi = c["acc_gap_p5_p95"]
            print(f"| {rule} | {c['avg_samples']:.1f} | {c['acc_gap']:+.3f} | {lo:+.3f}..{hi:+.3f} |")
        print("\nwhere the budget goes (run closest to 16 samples on average): avg samples / acc")
        levels = list(next(iter(r["alloc_at_16"].values()))["per_level"])
        print("| rule | avg | " + " | ".join(f"L{L}" for L in levels) + " |")
        print("|---" * (len(levels) + 2) + "|")
        for rule, a in r["alloc_at_16"].items():
            print(f"| {rule} | {a['avg_samples']:.1f} / {a['acc']:.3f} | "
                  + " | ".join(f"{a['per_level'][L]['avg_samples']:.1f} / {a['per_level'][L]['acc']:.3f}"
                               for L in levels) + " |")


def cmd_error_types(args):
    rows = []
    with open(args.rows, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                steps = d.get(args.steps_field) or d.get("steps_ko") or d.get("steps_en") or []
                rows.append({"id": d["id"], "split": d.get("split", "?"), "label": int(d["label"]),
                             "n_steps": int(d.get("n_steps") or len(steps))})
    preds = {}
    for name, path in parse_scorers(args.scorer):
        p = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("probs") is not None:
                    pr = [float(x) for x in d["probs"]]
                elif d.get("teacher_logodds") is not None:
                    pr = [1.0 / (1.0 + math.exp(-float(z))) for z in d["teacher_logodds"]]
                else:
                    continue
                p[d["id"]] = predict_first_error(pr, args.threshold)
        preds[name] = p
    res = error_type_table(rows, preds)
    write(args.out, res)
    tot = sum(res["distribution"].values())
    print(f"\n## 오류 풀이 {tot}개의 첫 오류 위치 분포")
    for t, n in res["distribution"].items():
        print(f"  {TYPE_KO[t]}: {n} ({n / tot:.1%})")
    print("\n| split | first | middle | last |\n|---|---|---|---|")
    for sp, d in res["by_split"].items():
        print(f"| {sp} | {d['first']} | {d['middle']} | {d['last']} |")
    print("\n| scorer | type | n | exact | early flag | late | missed |\n|---|---|---|---|---|---|---|")
    for name, d in res["scorers"].items():
        for t, x in d.items():
            if x["n"]:
                print(f"| {name} | {t} | {x['n']} | {x['exact']:.3f} | {x['early']:.3f} | "
                      f"{x['late']:.3f} | {x['missed']:.3f} |")


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[plan_extras] wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("adaptive", "levels"):
        s = sub.add_parser(name)
        s.add_argument("--dataset", default=None)
        s.add_argument("--dataset-config", default=SEED0_CONFIG,
                       help="the stored KO MATH500 BoN datasets have no 'default' config")
        s.add_argument("--split", default="train")
        s.add_argument("--jsonl", default=None)
        s.add_argument("--limit", type=int, default=None)
        s.add_argument("--scorer", action="append", required=True)
        s.add_argument("--agg", default="last")
        s.add_argument("--levels-from", default=None,
                       help="jsonl with id/unique_id and level, used when the dataset has no level")
        s.add_argument("--k-min", type=int, default=4)
        s.add_argument("--orders", type=int, default=10,
                       help="completion orders averaged over (0 = stored order, rest shuffled)")
        s.add_argument("--margin", type=float, default=0.01,
                       help="accuracy slack vs fixed@64 when counting samples to match it")
        s.add_argument("--out", required=True)
    e = sub.add_parser("error-types")
    e.add_argument("--rows", required=True)
    e.add_argument("--steps-field", default="steps_ko")
    e.add_argument("--scorer", action="append", required=True)
    e.add_argument("--threshold", type=float, default=0.5)
    e.add_argument("--out", required=True)
    args = ap.parse_args()
    {"adaptive": cmd_adaptive, "levels": cmd_levels, "error-types": cmd_error_types}[args.cmd](args)


if __name__ == "__main__":
    main()
