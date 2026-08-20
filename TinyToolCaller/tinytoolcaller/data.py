"""Tokenizer and dataset loading + deterministic sampling (publication §9)."""

from __future__ import annotations


def load_tokenizer(model_id: str):
    """Load the tokenizer and guarantee a pad token.

    Qwen2.5-1.5B-Instruct reports ``<|endoftext|>`` as pad; for SFT we set
    pad = eos (``<|im_end|>``) so padding is not attended to and is masked in
    the loss (label = -100 on pad positions, handled by the trainer).
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_source_dataset(dataset_id: str):
    """Load the (gated) source dataset, with an actionable error message."""
    from datasets import load_dataset

    try:
        return load_dataset(dataset_id, split="train")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "\nFailed to load the source dataset. It is GATED: log in and "
            f"accept the terms at https://huggingface.co/datasets/{dataset_id}, "
            "then set `export HF_TOKEN=<your_token>` and re-run.\n\n"
            f"Underlying error: {exc}"
        ) from exc


def sample_and_split(ds, n_sample: int, n_train: int, seed: int):
    """Deterministic subset: shuffle(seed) -> select(n_sample) -> train/val.

    Uses the ``datasets`` shuffle so the membership matches the documented
    seed-42 recipe (publication §9). Note the shuffle RNG is
    ``datasets``-version-sensitive; record the version (§12).
    """
    subset = ds.shuffle(seed=seed).select(range(n_sample))
    train = subset.select(range(n_train))
    val = subset.select(range(n_train, n_sample))
    return train, val
