"""SFTTrainer wrapper and adapter/merge publication (publication §11, §28)."""

from __future__ import annotations

import os


def train(model, tokenizer, train_ds, config: dict, trainable_params: int):
    """Run supervised fine-tuning with TRL's SFTTrainer.

    Handles the TRL API split: TRL >= 0.12 uses ``SFTConfig`` (which carries
    ``max_seq_length`` and ``packing``); older TRL passes ``max_seq_length``
    directly to ``SFTTrainer``. Packing is disabled so each example maps to a
    clean ChatML sequence (publication §9).
    """
    import torch
    from trl import SFTTrainer

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    common = dict(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_epochs"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        optim=config["optim"],
        bf16=bf16,
        fp16=(not bf16),
        logging_steps=config["logging_steps"],
        save_strategy=config["save_strategy"],
        gradient_checkpointing=config["gradient_checkpointing"],
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
        run_name=os.environ.get("WANDB_RUN_NAME", "tinytoolcaller-qlora"),
    )

    def formatting_func(examples):
        return [ex["text"] for ex in examples]

    try:
        from trl import SFTConfig

        args = SFTConfig(max_seq_length=config["max_seq_length"],
                         packing=False, **common)
        trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds,
                             tokenizer=tokenizer, formatting_func=formatting_func)
    except ImportError:
        from transformers import TrainingArguments

        args = TrainingArguments(**common)
        trainer = SFTTrainer(
            model=model, args=args, train_dataset=train_ds, tokenizer=tokenizer,
            max_seq_length=config["max_seq_length"],
            formatting_func=formatting_func,
        )

    print(f"\n    Trainable LoRA parameters : {trainable_params:,}")
    print(f"    Training examples         : {len(train_ds):,}")
    trainer.train()
    return trainer


def save_and_publish(trainer, tokenizer, config: dict, push: bool):
    """Save the adapter, merge it into the base, and publish both to the Hub."""
    adapter_dir = os.path.join(config["output_dir"], "adapter")
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\n[13] LoRA adapter saved to {adapter_dir}")

    merged_dir = os.path.join(config["output_dir"], "merged")
    model = trainer.model.merge_and_unload()
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"[13] Merged model saved to {merged_dir}")

    if not push:
        print("[14] Skipping Hub publication (--no-push).")
        return
    if not os.environ.get("HF_TOKEN"):
        print("[14] Skipping Hub publication: HF_TOKEN not set.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    for local, repo in (
        (adapter_dir, f"{config['hub_model_id']}-adapter"),
        (merged_dir, config["hub_model_id"]),
    ):
        try:
            api.create_repo(repo, exist_ok=True)
            api.upload_folder(folder_path=local, repo_id=repo, repo_type="model")
            print(f"[14] Published {repo}")
        except Exception as exc:  # noqa: BLE001
            print(f"[14] Failed to publish {repo}: {exc}")
