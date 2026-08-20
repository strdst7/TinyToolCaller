#!/usr/bin/env python3
"""Basic dataset statistics for the TinyToolCaller subset (publication §8.2).

Computes, for the documented seed-42 subset (5,200 -> 5,000 train / 200 val):

  * row counts per split;
  * unique ground-truth tool names (overlap with §8.1);
  * multi-answer rows (``answers`` lists longer than 1);
  * tools-per-example distribution (mean / median / max);
  * prompt length in tokens (system + user turn, via the Qwen tokenizer):
    mean / median / p95 / max;
  * the number of examples whose prompt exceeds ``max_seq_length`` (1024) —
    the truncation count that quantifies the §13 implementation concern.

Prints a Markdown table ready to paste into §8.2.

The source dataset is GATED: log in and accept its terms, then export HF_TOKEN.

Usage:
    export HF_TOKEN=<token>
    python scripts/dataset_stats.py                       # load from the Hub
    python scripts/dataset_stats.py --path data/subset.json
    python scripts/dataset_stats.py --max-seq-length 1024 --tokenizer Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinytoolcaller.data import load_source_dataset, load_tokenizer, sample_and_split  # noqa: E402
from tinytoolcaller.formatting import build_messages, ground_truth  # noqa: E402


def _rows_from_path(path: str):
    if path.endswith(".parquet"):
        import pandas as pd

        return pd.read_parquet(path).to_dict("records")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = data.get("rows", data.get("data", list(data.values())))
    return list(data)


def _fmt(x) -> str:
    return f"{x:.0f}" if float(x).is_integer() else f"{x:.1f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", default=None,
                        help="Optional local JSON/Parquet copy; default: Hub.")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-sample", type=int, default=5200)
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    args = parser.parse_args()

    if args.path:
        rows = _rows_from_path(args.path)
        from datasets import Dataset

        ds = Dataset.from_list(rows)
    else:
        ds = load_source_dataset("Salesforce/xlam-function-calling-60k")

    train, val = sample_and_split(ds, args.n_sample, args.n_train, args.seed)
    print(f"[stats] Loading tokenizer {args.tokenizer} ...")
    tok = load_tokenizer(args.tokenizer)

    train_gt = [ground_truth(e) for e in train]
    val_gt = [ground_truth(e) for e in val]

    unique = Counter(g["name"] for g in train_gt + val_gt if g)
    multi = sum(1 for e in list(train) + list(val)
                if isinstance(e.get("answers", e.get("answer")), list)
                and len(e.get("answers", [])) > 1)
    tools_per = [len(e["tools"]) for e in list(train) + list(val)]
    lens = []
    for e in list(train) + list(val):
        msgs = build_messages(e)
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
        lens.append(len(tok(text)["input_ids"]))
    lens = np.array(lens)
    truncated = int((lens > args.max_seq_length).sum())

    print("\n## §8.2 Basic Dataset Statistics (paste into the publication)\n")
    print("| Statistic | Value |")
    print("| --- | --- |")
    print(f"| Examples (train / validation) | {len(train)} / {len(val)} |")
    print(f"| Unique tool names in subset | {len(unique)} |")
    print(f"| Multi-answer rows (>1 ground-truth answer) | {multi} |")
    print(f"| Tools per example — mean / median / max | "
          f"{_fmt(np.mean(tools_per))} / {_fmt(np.median(tools_per))} / {int(np.max(tools_per))} |")
    print(f"| Prompt tokens (system+user) — mean / median / p95 / max | "
          f"{_fmt(np.mean(lens))} / {_fmt(np.median(lens))} / "
          f"{_fmt(np.percentile(lens, 95))} / {int(np.max(lens))} |")
    print(f"| Examples truncated at max_seq_length={args.max_seq_length} | {truncated} "
          f"({100.0 * truncated / len(lens):.2f}%) |")
    print("\nTop-5 tools by frequency:")
    for name, c in unique.most_common(5):
        print(f"  {name}: {c} ({100.0 * c / len(lens):.2f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
