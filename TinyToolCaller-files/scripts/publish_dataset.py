#!/usr/bin/env python3
"""Publish the derived TinyToolCaller dataset + card to the Hugging Face Hub.

Reproduces the documented sampling — shuffle(seed=42) → select(5,200) →
first 5,000 = train / last 200 = validation — saves it as Parquet, and uploads
the data and the dataset card (README.md) to `strdst77/TinyToolCaller`.

Required access:
  - read access + accepted terms for `Salesforce/xlam-function-calling-60k` (gated)
  - write access for `strdst77/TinyToolCaller`

Usage:
  export HF_TOKEN=<your_token>
  python scripts/publish_dataset.py --push                # build + upload
  python scripts/publish_dataset.py                       # build locally only
  python scripts/publish_dataset.py --repo-id strdst77/TinyToolCaller --push
"""

from __future__ import annotations

import argparse
import os

from datasets import Dataset, load_dataset

SOURCE_DATASET = "Salesforce/xlam-function-calling-60k"
DEFAULT_REPO_ID = "strdst77/TinyToolCaller"


def build_subset(seed: int = 42, n_sample: int = 5200, n_train: int = 5000):
    print(f"Loading {SOURCE_DATASET} (gated — requires auth)...")
    ds = load_dataset(SOURCE_DATASET, split="train")
    subset = ds.shuffle(seed=seed).select(range(n_sample))

    train = subset.select(range(n_train)).add_column(
        "split", ["train"] * n_train
    )
    val = subset.select(range(n_train, n_sample)).add_column(
        "split", ["validation"] * (n_sample - n_train)
    )
    return train, val


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-sample", type=int, default=5200)
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--push", action="store_true",
                        help="Upload to the Hub (default: write locally only)")
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        print("Warning: HF_TOKEN not set. Loading the gated source dataset will fail.")

    train, val = build_subset(args.seed, args.n_sample, args.n_train)

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train.parquet")
    val_path = os.path.join(args.out_dir, "validation.parquet")
    train.to_parquet(train_path)
    val.to_parquet(val_path)
    print(f"Wrote {train_path} ({len(train)} rows) and {val_path} ({len(val)} rows).")

    if not args.push:
        print("Dry run complete. Re-run with --push (and HF_TOKEN) to upload.")
        return 0

    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    create_repo(args.repo_id, repo_type="dataset", exist_ok=True)

    readme = os.path.join(os.path.dirname(__file__), "..", "README.md")
    if not os.path.exists(readme):
        readme = os.path.join(os.path.dirname(__file__), "..", "README.hf.md")
    if os.path.exists(readme):
        api.upload_file(
            path_or_fileobj=readme,
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print(f"Uploaded dataset card from {readme}.")

    for local, name in ((train_path, "train.parquet"),
                        (val_path, "validation.parquet")):
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=name,
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print(f"Uploaded {name}.")

    print(f"Done. View at https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
