#!/usr/bin/env python3
"""Statistical analysis for the TinyToolCaller evaluation (README §18 / §19-§21).

Two modes:

1. ``--report`` — reproduces the Wilson 95% confidence intervals and Cohen's h
   effect sizes for the *reported* aggregate results (n=200 eval split, n=50
   GSM8K). Runs with no data dependencies and prints the table used in the
   publication's Statistical Analysis section.

2. ``--mcnemar predictions.jsonl`` — the rigorous paired analysis. Reads the
   per-example predictions dumped by ``train_tool_caller.py --eval-dump`` and
   computes, for each metric:
     * Wilson 95% CI for base and fine-tuned accuracy;
     * McNemar's exact test on the discordant pairs (base-vs-fine-tuned on the
       SAME examples — the correct paired test);
     * a bootstrap 95% CI for the paired accuracy difference;
     * Cohen's h effect size.

The per-example dump is required because McNemar's test and paired bootstrap
CIs cannot be computed from aggregate percentages alone — they need the
contingency table of (base wrong, fine-tuned right) vs (base right, fine-tuned
wrong) transitions.

Usage:
    python scripts/statistical_analysis.py --report
    python scripts/statistical_analysis.py --mcnemar outputs/eval_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

Z = 1.959963984540054  # 95% two-sided


def wilson_ci(p: float, n: int) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    denom = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / denom
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denom
    return center - half, center + half


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size between two proportions."""
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def h_label(h: float) -> str:
    a = abs(h)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


# Reported aggregate results: (metric, n, base_p, finetuned_p)
REPORTED = [
    ("JSON validity", 200, 0.785, 0.980),
    ("Tool-name accuracy", 200, 0.650, 0.925),
    ("Argument exact match", 200, 0.420, 0.840),
    ("GSM8K retention", 50, 0.520, 0.500),
]


def print_report():
    print("Statistical analysis of the reported aggregate results")
    print("(95% Wilson confidence intervals; effect size Cohen's h)\n")
    print(f"{'Metric':22} {'Base':>8} {'95% CI (base)':>24} "
          f"{'Fine-tuned':>11} {'95% CI (ft)':>24} {'Cohen h':>9} {'Effect':>10}")
    print("-" * 112)
    for name, n, b, f in REPORTED:
        lo_b, hi_b = wilson_ci(b, n)
        lo_f, hi_f = wilson_ci(f, n)
        h = cohens_h(f, b)
        print(f"{name:22} {b*100:6.1f}%  [{lo_b*100:5.1f}%,{hi_b*100:5.1f}%]  "
              f"{f*100:6.1f}%  [{lo_f*100:5.1f}%,{hi_f*100:5.1f}%]  "
              f"{h:+7.3f}  {h_label(h)}")
    print("\nNote: Wilson CIs treat each metric as a binomial proportion on the")
    print("evaluation split. Confidence intervals on the paired difference")
    print("(base vs fine-tuned on the SAME examples) require the per-example")
    print("dump -> run with --mcnemar.")


def metric_bits(row: dict, metric: str):
    """Return (base_ok, ft_ok) booleans for a metric on one example row."""
    gt = row.get("gt") or {}
    base = row.get("base") or {}
    ft = row.get("ft") or {}

    def ok(pred):
        if pred is None:
            return False
        if metric == "json_valid":
            return True
        if metric == "tool":
            return pred.get("name") == gt.get("name")
        if metric == "args":
            return pred.get("arguments") == gt.get("arguments")
        raise ValueError(metric)

    return ok(base.get("pred")), ok(ft.get("pred"))


def mcnemar(rows, metric: str):
    """McNemar's test on discordant pairs + bootstrap CI on the difference."""
    import random

    n = len(rows)
    b = c = 0  # b: base wrong & ft right ; c: base right & ft wrong
    for row in rows:
        base_ok, ft_ok = metric_bits(row, metric)
        if not base_ok and ft_ok:
            b += 1
        elif base_ok and not ft_ok:
            c += 1

    if b + c == 0:
        exact_p = 1.0
        stat = 0.0
    else:
        # exact binomial two-sided p-value on discordant pairs
        from math import comb

        m = b + c
        p_half = 0.5
        exact_p = sum(
            comb(m, k) * (p_half ** m)
            for k in range(0, m + 1)
            if abs(k - m / 2.0) >= abs(b - m / 2.0)
        )
        # continuity-corrected chi-square statistic
        stat = (abs(b - c) - 1) ** 2 / m if m > 0 else 0.0

    # bootstrap paired difference CI
    rng = random.Random(42)
    diffs = []
    for _ in range(10_000):
        idx = [rng.randrange(n) for _ in range(n)]
        base_acc = sum(metric_bits(rows[i], metric)[0] for i in idx) / n
        ft_acc = sum(metric_bits(rows[i], metric)[1] for i in idx) / n
        diffs.append(ft_acc - base_acc)
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs))]

    base_acc = sum(metric_bits(r, metric)[0] for r in rows) / n
    ft_acc = sum(metric_bits(r, metric)[1] for r in rows) / n
    return {
        "n": n,
        "base_acc": base_acc,
        "ft_acc": ft_acc,
        "base_ci": wilson_ci(base_acc, n),
        "ft_ci": wilson_ci(ft_acc, n),
        "discordant_b": b,
        "discordant_c": c,
        "mcnemar_stat": stat,
        "mcnemar_p": exact_p,
        "bootstrap_ci": (lo, hi),
        "cohens_h": cohens_h(ft_acc, base_acc),
    }


def print_mcnemar(path: str):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    print(f"Paired analysis of {len(rows)} examples from {path}\n")
    for metric in ("json_valid", "tool", "args"):
        r = mcnemar(rows, metric)
        name = {"json_valid": "JSON validity",
                "tool": "Tool-name accuracy",
                "args": "Argument exact match"}[metric]
        print(f"== {name} ==")
        print(f"  base       : {r['base_acc']*100:5.1f}%  "
              f"95% CI [{r['base_ci'][0]*100:5.1f}%, {r['base_ci'][1]*100:5.1f}%]")
        print(f"  fine-tuned : {r['ft_acc']*100:5.1f}%  "
              f"95% CI [{r['ft_ci'][0]*100:5.1f}%, {r['ft_ci'][1]*100:5.1f}%]")
        print(f"  discordant pairs: base-wrong/ft-right = {r['discordant_b']}, "
              f"base-right/ft-wrong = {r['discordant_c']}")
        print(f"  McNemar chi2 = {r['mcnemar_stat']:.3f} (exact two-sided "
              f"p = {r['mcnemar_p']:.4g})")
        print(f"  bootstrap 95% CI on paired difference: "
              f"[{r['bootstrap_ci'][0]*100:+.1f} pp, {r['bootstrap_ci'][1]*100:+.1f} pp]")
        print(f"  Cohen's h = {r['cohens_h']:+.3f} ({h_label(r['cohens_h'])})\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--report", action="store_true",
                       help="Print Wilson CIs + Cohen's h for the reported aggregates.")
    group.add_argument("--mcnemar", metavar="PREDICTIONS.jsonl",
                       help="Paired McNemar + bootstrap analysis from the per-example dump.")
    args = parser.parse_args()

    if args.report:
        print_report()
    else:
        print_mcnemar(args.mcnemar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
