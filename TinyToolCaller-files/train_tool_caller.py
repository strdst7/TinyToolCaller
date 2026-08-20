#!/usr/bin/env python3
"""TinyToolCaller — end-to-end QLoRA fine-tuning and evaluation pipeline.

Specializes `Qwen/Qwen2.5-1.5B-Instruct` for reliable structured function
calling: given a natural-language request and a set of tool schemas, the model
returns a single JSON tool call ``{"name": ..., "arguments": {...}}``.

The pipeline mirrors the 14 stages documented in the publication (§13):

     1. Load tokenizer              8.  Attach LoRA
     2. Load dataset                9.  Train SFT model (TRL SFTTrainer)
     3. Shuffle / split (seed=42)  10.  Save adapter
     4. Format ChatML              11.  Evaluate fine-tuned model
     5. Evaluate baseline          12.  Evaluate GSM8K retention
     6. Load 4-bit model           13.  Merge adapter
     7. Prepare k-bit training     14.  Publish model to the Hub

Requirements are in requirements.txt. The source dataset is GATED on the Hub:
log in and accept its terms first, then export HF_TOKEN.

Usage:
    export HF_TOKEN=<huggingface_token>       # required (gated source dataset)
    export WANDB_API_KEY=<wandb_key>          # optional experiment tracking

    python train_tool_caller.py                       # full pipeline
    python train_tool_caller.py --no-baseline         # skip base-model eval
    python train_tool_caller.py --skip-gsm8k          # skip GSM8K retention
    python train_tool_caller.py --no-push             # don't publish to Hub
    python train_tool_caller.py --max-seq-length 512  # override a config value
    python train_tool_caller.py --eval-dump outputs/eval_predictions.jsonl
            # write per-example predictions for scripts/statistical_analysis.py

Faithfulness to the documented configuration (README §11-§14):
    * the base model is evaluated in 4-bit NF4 (``eval_load_in_4bit=True``) so
      baseline and fine-tuned are scored under identical quantization;
    * JSON validity is computed AFTER extraction from the raw output (§21.7);
    * GSM8K accuracy uses one shared extraction harness for both models (§20).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from tinytoolcaller import data, formatting, metrics, model, train
from tinytoolcaller.config import CONFIG


def _dump_eval(base_details, ft_details, path: str) -> None:
    """Write paired per-example predictions to JSONL for McNemar analysis."""
    rows = [
        {"gt": b["gt"],
         "base": {"raw": b["raw"], "pred": b["pred"]},
         "ft": {"raw": f["raw"], "pred": f["pred"]}}
        for b, f in zip(base_details, ft_details)
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"\n    Per-example predictions written to {path}")
    print(f"    Run: python scripts/statistical_analysis.py --mcnemar {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--skip-gsm8k", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--eval-dump", default=None,
                        help="Write per-example base+ft predictions as JSONL "
                             "(for scripts/statistical_analysis.py --mcnemar).")
    args = parser.parse_args()

    if args.max_seq_length is not None:
        CONFIG["max_seq_length"] = args.max_seq_length
    if args.output_dir is not None:
        CONFIG["output_dir"] = args.output_dir
    if args.hub_model_id is not None:
        CONFIG["hub_model_id"] = args.hub_model_id

    if not os.environ.get("HF_TOKEN"):
        print("Warning: HF_TOKEN is not set. Loading the gated source dataset "
              "will fail unless you are already logged in via `huggingface-cli login`.")
    if os.environ.get("WANDB_API_KEY"):
        import wandb

        wandb.login(key=os.environ["WANDB_API_KEY"])

    print("=" * 72)
    print("TinyToolCaller — QLoRA function-calling pipeline")
    print(f"  base model : {CONFIG['model_id']}")
    print(f"  data       : {CONFIG['source_dataset_id']} "
          f"(seed={CONFIG['seed']}, {CONFIG['n_train']} train / {CONFIG['n_val']} val)")
    print("=" * 72)

    # --- 1. tokenizer ------------------------------------------------------ #
    print("\n[1] Loading tokenizer ...")
    tokenizer = data.load_tokenizer(CONFIG["model_id"])

    # --- 2-4. dataset, split, ChatML --------------------------------------- #
    print("[2] Loading source dataset (gated) ...")
    ds = data.load_source_dataset(CONFIG["source_dataset_id"])
    print("[3] Shuffling (seed=42) and splitting ...")
    train_ds, val_ds = data.sample_and_split(
        ds, CONFIG["n_sample"], CONFIG["n_train"], CONFIG["seed"]
    )
    print("[4] ChatML formatting ...")
    train_ds = train_ds.map(
        lambda ex: formatting.format_for_training(ex, tokenizer),
        remove_columns=train_ds.column_names,
    )
    print(f"    train={len(train_ds)}  val={len(val_ds)}")

    # --- 5. baseline evaluation ------------------------------------------- #
    baseline = None
    base_details = None
    if not args.no_baseline:
        print("\n[5] Evaluating baseline (4-bit base model) ...")
        base_model = model.load_quantized_model(
            CONFIG["model_id"], CONFIG["eval_load_in_4bit"],
            {"bnb_4bit_quant_type": CONFIG["bnb_4bit_quant_type"],
             "bnb_4bit_use_double_quant": CONFIG["bnb_4bit_use_double_quant"]},
        )
        result = metrics.evaluate_tool_calling(
            base_model, tokenizer, val_ds, CONFIG["max_new_tokens"],
            return_details=bool(args.eval_dump),
        )
        baseline, base_details = result if args.eval_dump else (result, None)
        print("    BASELINE")
        print(baseline.as_table())
        del base_model
        import torch  # noqa: F401 - lazily imported; only needed on GPU runs

        torch.cuda.empty_cache()

    # --- 6-9. QLoRA fine-tuning -------------------------------------------- #
    if not args.no_train:
        print("\n[6] Loading 4-bit NF4 model ...")
        ft_model = model.load_quantized_model(
            CONFIG["model_id"], CONFIG["load_in_4bit"],
            {"bnb_4bit_quant_type": CONFIG["bnb_4bit_quant_type"],
             "bnb_4bit_use_double_quant": CONFIG["bnb_4bit_use_double_quant"]},
        )
        print("[7] Preparing k-bit training ...")
        print("[8] Attaching LoRA adapters ...")
        ft_model = model.attach_lora(ft_model, CONFIG)
        trainable = sum(p.numel() for p in ft_model.parameters() if p.requires_grad)
        print("[9] Training with SFTTrainer ...")
        trainer = train.train(ft_model, tokenizer, train_ds, CONFIG, trainable)

        # --- 11. fine-tuned evaluation ------------------------------------- #
        if not args.no_eval:
            print("\n[11] Evaluating fine-tuned model ...")
            result = metrics.evaluate_tool_calling(
                trainer.model, tokenizer, val_ds, CONFIG["max_new_tokens"],
                return_details=bool(args.eval_dump),
            )
            ft, ft_details = result if args.eval_dump else (result, None)
            print("    FINE-TUNED")
            print(ft.as_table())
            if baseline is not None:
                print("\n    COMPARISON (base -> fine-tuned)")
                print(f"    JSON validity         : {baseline.json_validity:5.1f}% -> {ft.json_validity:5.1f}%")
                print(f"    Tool-name accuracy    : {baseline.tool_accuracy:5.1f}% -> {ft.tool_accuracy:5.1f}%")
                print(f"    Argument exact match  : {baseline.argument_exact_match:5.1f}% -> {ft.argument_exact_match:5.1f}%")

            if args.eval_dump and base_details is not None and ft_details is not None:
                _dump_eval(base_details, ft_details, args.eval_dump)

        # --- 12. GSM8K retention ------------------------------------------- #
        if not args.skip_gsm8k:
            print("\n[12] GSM8K retention check ...")
            ft_gsm8k = metrics.evaluate_gsm8k(
                trainer.model, tokenizer, CONFIG["gsm8k_n"],
                CONFIG["gsm8k_dataset_id"], CONFIG["gsm8k_config"],
                CONFIG["gsm8k_split"], CONFIG["max_new_tokens"],
            )
            print(f"    fine-tuned GSM8K (n={CONFIG['gsm8k_n']}): {ft_gsm8k:.1f}%")

        # --- 13-14. merge & publish ---------------------------------------- #
        train.save_and_publish(trainer, tokenizer, CONFIG, push=not args.no_push)

    print("\nDone. See README §16-§21 before quoting results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
