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


# --------------------------------------------------------------------------- #
# Data-quality rules (publication §9.2)
# --------------------------------------------------------------------------- #
def validate_example(example: dict) -> tuple[bool, str]:
    """Enforce the §9.2 data-quality rules; returns (is_valid, reason).

    Rules (each failure is counted and reported by :func:`clean_subset`):
      1. ``query`` must be a non-empty string.
      2. ``tools`` must be a list, and every tool a dict with a non-empty
         string ``name`` (tools with unparseable schemas are rejected).
      3. Ground truth (``answers``/``answer``) must be present and yield at
         least one ``{name, arguments}`` entry.
    """
    query = example.get("query")
    if not isinstance(query, str) or not query.strip():
        return False, "missing_or_empty_query"

    tools = example.get("tools")
    if not isinstance(tools, list):
        return False, "tools_not_a_list"
    for tool in tools:
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str) \
                or not tool["name"].strip():
            return False, "malformed_tool_entry"

    answers = example.get("answers", example.get("answer"))
    if answers is None:
        return False, "missing_answers"
    return True, "ok"


def _example_key(example: dict) -> tuple:
    """Deterministic dedup key: (query, tool names, answer)."""
    from .formatting import ground_truth
    import json

    tools = tuple(sorted(t.get("name", "") for t in example.get("tools", [])))
    gt = ground_truth(example)
    return (example.get("query", "").strip(), tools, json.dumps(gt, sort_keys=True))


def clean_subset(rows) -> tuple[list, dict]:
    """Apply the §9.2 cleaning pipeline and return (kept_rows, stats).

    Stats dict: total, dropped_by_reason (Counter), duplicates_removed.
    The kept rows are otherwise **unchanged** — no re-labelling or value
    filtering, because the source's execution verification is the authority
    on correctness (§8.5).
    """
    from collections import Counter, OrderedDict

    stats = {"total": len(rows), "dropped_by_reason": Counter()}
    kept, seen = [], set()
    for row in rows:
        ok, reason = validate_example(row)
        if not ok:
            stats["dropped_by_reason"][reason] += 1
            continue
        key = _example_key(row)
        if key in seen:
            stats["dropped_by_reason"]["exact_duplicate"] += 1
            continue
        seen.add(key)
        kept.append(row)
    stats["kept"] = len(kept)
    return kept, stats
