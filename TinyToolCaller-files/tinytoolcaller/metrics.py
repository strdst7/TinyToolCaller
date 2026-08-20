"""Evaluation metrics and scorers (publication §14).

`ToolCallingMetrics` is import-safe without torch; the generation helpers
import torch lazily so this module can be unit-tested on CPU/CI.
"""

from __future__ import annotations

from dataclasses import dataclass

from .formatting import (
    extract_gsm8k_answer,
    extract_json,
    format_for_inference,
    ground_truth,
    normalise_number,
)


@dataclass
class ToolCallingMetrics:
    n: int = 0
    json_valid: int = 0
    tool_correct: int = 0
    args_correct: int = 0

    @property
    def json_validity(self) -> float:
        return 100.0 * self.json_valid / self.n if self.n else 0.0

    @property
    def tool_accuracy(self) -> float:
        return 100.0 * self.tool_correct / self.n if self.n else 0.0

    @property
    def argument_exact_match(self) -> float:
        return 100.0 * self.args_correct / self.n if self.n else 0.0

    def as_table(self) -> str:
        return (
            f"    JSON validity         : {self.json_validity:5.1f}% "
            f"({self.json_valid}/{self.n})\n"
            f"    Tool-name accuracy    : {self.tool_accuracy:5.1f}% "
            f"({self.tool_correct}/{self.n})\n"
            f"    Argument exact match  : {self.argument_exact_match:5.1f}% "
            f"({self.args_correct}/{self.n})"
        )


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    """Deterministic greedy generation; returns only the NEW tokens."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )


def evaluate_tool_calling(model, tokenizer, val_ds, max_new_tokens: int,
                          return_details: bool = False):
    """Score the evaluation split; optionally return per-example predictions.

    ``return_details=True`` yields ``(metrics, details)`` where ``details`` is
    a list of ``{"gt": ..., "raw": ..., "pred": ...}`` dicts for the paired
    significance test (``scripts/statistical_analysis.py --mcnemar``).
    """
    metrics = ToolCallingMetrics(n=len(val_ds))
    details = []
    for example in val_ds:
        prompt = format_for_inference(example, tokenizer)
        raw = generate(model, tokenizer, prompt, max_new_tokens)
        pred = extract_json(raw)
        gt = ground_truth(example)

        if pred is not None:
            metrics.json_valid += 1
        if pred is not None and pred.get("name") == gt.get("name"):
            metrics.tool_correct += 1
        if pred is not None and pred.get("arguments") == gt.get("arguments"):
            metrics.args_correct += 1
        if return_details:
            details.append({"gt": gt, "raw": raw, "pred": pred})

    if return_details:
        return metrics, details
    return metrics


def evaluate_gsm8k(model, tokenizer, n: int, dataset_id: str, config: str,
                   split: str, max_new_tokens: int) -> float:
    """GSM8K retention probe (publication §20). Shared harness for both models."""
    from datasets import load_dataset

    try:
        ds = load_dataset(dataset_id, config, split=split)
    except Exception:  # noqa: BLE001 - gsm8k "test" can be gated; fall back
        print(f"    [gsm8k] split '{split}' unavailable (gated?); trying 'train'")
        ds = load_dataset(dataset_id, config, split="train")

    ds = ds.shuffle(seed=42).select(range(n))
    correct = 0
    for example in ds:
        prompt = (f"Question: {example['question']}\n"
                  "Let's think step by step.\nAnswer:")
        raw = generate(model, tokenizer, prompt, max_new_tokens)
        pred = normalise_number(extract_gsm8k_answer(raw))
        gold = normalise_number(extract_gsm8k_answer(example["answer"]))
        correct += int(pred == gold)
    return 100.0 * correct / n
