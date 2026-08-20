#!/usr/bin/env python3
"""§8.1 — Tool-distribution profiling for the TinyToolCaller 5,200-example subset.

Computes the three quantities that must be reported before the headline
function-calling metrics are quoted without qualification:

  (a) the number of UNIQUE tool names (ground-truth ``name``) appearing in the
      5,200-example subset;
  (b) the TOP-10 most frequent tools and their share of the subset;
  (c) whether the 200-example VALIDATION split's tool distribution matches the
      TRAINING split's (chi-square test + Jensen-Shannon divergence +
      validation→training coverage).

Sampling reproduces the documented recipe exactly:

    shuffle(seed=42) -> select(5_200) -> first 5_000 = train, last 200 = val

using the Hugging Face ``datasets`` shuffle, so the membership is identical to
what the training pipeline produced (for the same ``datasets`` version; the
shuffle RNG is version-sensitive, and the script prints the version it used).

The source dataset is GATED on the Hugging Face Hub — you must be logged in and
have accepted the dataset terms:

    https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k

Usage:
    export HF_TOKEN=<your_token>            # needs gated-dataset + read access
    python scripts/profile_tool_distribution.py
    python scripts/profile_tool_distribution.py --seed 42 --n-sample 5200 --n-train 5000

    # Or point at a local copy of the derived subset (JSON or Parquet):
    python scripts/profile_tool_distribution.py --path data/subset.json
    python scripts/profile_tool_distribution.py --path data/subset.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

import numpy as np

try:
    from scipy.stats import chi2_contingency

    HAS_SCIPY = True
except Exception:  # pragma: no cover - scipy is optional for the chi-square part
    HAS_SCIPY = False


# --------------------------------------------------------------------------- #
# Extraction helpers
# --------------------------------------------------------------------------- #
def _as_obj(value):
    """Coerce a value that may be a JSON string into a Python object."""
    if isinstance(value, str):
        return json.loads(value)
    return value


def _answer_name(example: dict):
    """Return the ground-truth tool name for one example.

    xLAM examples look like::

        {"query": str, "tools": [...], "answers": [{"name": ..., "arguments": ...}, ...]}

    The evaluation pipeline scores a single expected call, so we take the FIRST
    answer. (The report also counts how many examples had multiple answers, so
    the single-answer assumption is visible rather than silent.)
    """
    answers = example.get("answers", example.get("answer"))
    answers = _as_obj(answers)
    if isinstance(answers, list):
        if not answers:
            return None, 0
        first, n = answers[0], len(answers)
    elif isinstance(answers, dict):
        first, n = answers, 1
    else:
        return None, 0

    first = _as_obj(first)
    if isinstance(first, dict):
        return first.get("name"), n
    return None, n


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _rows_from_path(path: str):
    if path.endswith((".json", ".jsonl")):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            # some exports wrap rows under a "rows"/"data" key
            data = data.get("rows", data.get("data", list(data.values())))
        return list(data)
    if path.endswith(".parquet"):
        import pandas as pd

        return pd.read_parquet(path).to_dict("records")
    raise ValueError(f"Unsupported file type for --path: {path}")


def _rows_from_hub():
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "`datasets` is not installed. Run: pip install datasets"
        ) from exc

    print("Loading Salesforce/xlam-function-calling-60k (gated dataset)...")
    try:
        ds = load_dataset(
            "Salesforce/xlam-function-calling-60k", split="train"
        )
    except Exception as exc:
        raise SystemExit(
            "\nFailed to load the source dataset.\n"
            "The dataset is gated: you must be logged in and have accepted its "
            "terms at https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k\n"
            "Then re-run with `export HF_TOKEN=<your_token>`.\n\n"
            f"Underlying error: {exc}"
        )
    return ds.to_list()


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (natural log base) between two distributions."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    m = (p + q) / 2.0

    def kl(a, b):
        a_safe = np.where(a > 0, a, 1.0)
        b_safe = np.where(b > 0, b, 1.0)
        return float(np.sum(a * np.log(a_safe / b_safe)))

    return 0.5 * (kl(p, m) + kl(q, m))


def chi2_train_vs_val(train_counter: Counter, val_counter: Counter, top_n: int = 10):
    """Chi-square test of homogeneity on the top-N tools (rest pooled to 'other').

    Returns (chi2, p_value, dof, categories, contingency) or None if scipy is absent.
    """
    if not HAS_SCIPY:
        return None

    pooled = train_counter + val_counter
    top_tools = [t for t, _ in pooled.most_common(top_n)]
    categories = top_tools + ["<other>"]

    train_row = [train_counter.get(t, 0) for t in top_tools]
    train_row.append(sum(c for t, c in train_counter.items() if t not in top_tools))
    val_row = [val_counter.get(t, 0) for t in top_tools]
    val_row.append(sum(c for t, c in val_counter.items() if t not in top_tools))

    table = np.array([train_row, val_row], dtype=int)
    chi2, p_value, dof, _ = chi2_contingency(table, correction=False)
    return chi2, p_value, dof, categories, table


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _pct(count: int, total: int) -> str:
    return f"{100.0 * count / total:.2f}%"


def build_report(names_subset, names_train, names_val, multi_answer_count, seed):
    total = len(names_subset)
    known = [n for n in names_subset if n]
    missing = total - len(known)

    unique = sorted({n for n in known})
    counter = Counter(known)
    train_counter = Counter(n for n in names_train if n)
    val_counter = Counter(n for n in names_val if n)

    top10 = counter.most_common(10)

    # (c) train/val match
    train_vocab = set(train_counter)
    val_in_train = sum(1 for n in names_val if n in train_vocab)
    coverage = val_in_train / len(names_val) if names_val else 0.0

    # JS divergence over the pooled vocabulary
    vocab = sorted(set(train_counter) | set(val_counter))
    p = np.array([train_counter.get(t, 0) for t in vocab], dtype=float)
    q = np.array([val_counter.get(t, 0) for t in vocab], dtype=float)
    jsd = js_divergence(p, q)

    chi2_result = chi2_train_vs_val(train_counter, val_counter, top_n=10)

    lines = []
    A = lines.append
    A("")
    A("=" * 72)
    A("TOOL-DISTRIBUTION PROFILE  (paste into §8.1 of the publication)")
    A("=" * 72)
    A(f"Sampling          : shuffle(seed={seed}) -> {total} examples "
      f"({len(names_train)} train / {len(names_val)} val)")
    A(f"Examples parsed   : {total}")
    A(f"Missing tool name : {missing}")
    A(f"Multi-answer rows : {multi_answer_count} "
      "(first answer used; see §8.1 note)")
    A("")
    A("(a) UNIQUE TOOL NAMES")
    A("-" * 72)
    A(f"Unique ground-truth tool names in the {total}-example subset: {len(unique)}")
    A("")
    A("(b) TOP-10 TOOLS")
    A("-" * 72)
    A("| Rank | Tool name | Count | Share of subset |")
    A("| ---- | --------- | ----: | --------------: |")
    for rank, (tool, count) in enumerate(top10, start=1):
        A(f"| {rank} | `{tool}` | {count} | {_pct(count, total)} |")
    if len(counter) > 10:
        rest = sum(c for t, c in counter.items() if t not in dict(top10))
        A(f"| 11+ | _remaining {len(counter) - 10} tools_ | {rest} | "
          f"{_pct(rest, total)} |")
    A("")
    A("(c) TRAIN vs VALIDATION DISTRIBUTION MATCH")
    A("-" * 72)
    A(f"Tools present in BOTH splits          : "
      f"{len(set(train_counter) & set(val_counter))}")
    A(f"Validation tools absent from training : "
      f"{len(set(val_counter) - set(train_counter))}")
    A(f"Validation examples covered by train  : "
      f"{val_in_train}/{len(names_val)} = {100.0 * coverage:.2f}%")
    A(f"Jensen-Shannon divergence (train,val) : {jsd:.6f}")
    if chi2_result is not None:
        chi2, p_value, dof, cats, table = chi2_result
        A(f"Chi-square (top-10 + '<other>')      : "
          f"chi2={chi2:.3f}, dof={dof}, p={p_value:.4f}")
        A("")
        A("Contingency table (rows: train, val; columns: top-10 tools + <other>):")
        A("  " + " ".join(f"{c:>12}" for c in cats))
        for label, row in zip(("train", "val  "), table):
            A("  " + label + " " + " ".join(f"{v:>12}" for v in row))
        if p_value < 0.05:
            A("")
            A("  -> p < 0.05: the validation split's tool distribution differs")
            A("     from the training split's. Interpret tool-accuracy numbers")
            A("     with extra care (see §8.1 interpretation notes).")
        else:
            A("")
            A("  -> p >= 0.05: no statistically significant difference detected")
            A("     between the training and validation tool distributions.")
    else:
        A("Chi-square skipped (scipy not installed; `pip install scipy`).")
    A("=" * 72)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tool-distribution profiling for the TinyToolCaller subset."
    )
    parser.add_argument("--path", default=None,
                        help="Optional local JSON/Parquet copy of the dataset. "
                             "If omitted, load from the Hub.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-sample", type=int, default=5200)
    parser.add_argument("--n-train", type=int, default=5000)
    args = parser.parse_args()

    if args.path:
        rows = _rows_from_path(args.path)
    else:
        rows = _rows_from_hub()

    if len(rows) < args.n_sample:
        raise SystemExit(
            f"Only {len(rows)} rows available; need at least {args.n_sample} "
            f"for the documented subset."
        )

    # Reproduce the documented sampling with the `datasets` shuffle so the
    # membership matches the training pipeline (same seed, same RNG).
    try:
        import datasets
        from datasets import Dataset

        ds = Dataset.from_list(rows)
        subset = ds.shuffle(seed=args.seed).select(range(args.n_sample))
        train = subset.select(range(args.n_train))
        val = subset.select(range(args.n_train, args.n_sample))
        ds_version = datasets.__version__
    except Exception:  # pragma: no cover - fallback to numpy if datasets missing
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(rows))[: args.n_sample]
        subset = [rows[i] for i in idx]
        train = subset[: args.n_train]
        val = subset[args.n_train:]
        ds_version = "numpy fallback"

    names_subset = []
    names_train = []
    names_val = []
    multi_answer = 0
    for ex in subset:
        name, n_ans = _answer_name(ex)
        names_subset.append(name)
        multi_answer += int(n_ans > 1)
    for ex in train:
        name, _ = _answer_name(ex)
        names_train.append(name)
    for ex in val:
        name, _ = _answer_name(ex)
        names_val.append(name)

    print(f"datasets version used for shuffle: {ds_version}")
    print(build_report(names_subset, names_train, names_val,
                       multi_answer, args.seed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
