"""Central experiment configuration (publication §11) and system prompt."""

CONFIG = {
    # -- data (publication §8-§9) ------------------------------------------ #
    "source_dataset_id": "Salesforce/xlam-function-calling-60k",  # gated
    "seed": 42,
    "n_sample": 5200,
    "n_train": 5000,
    "n_val": 200,
    "max_seq_length": 1024,
    # -- model / publishing ------------------------------------------------ #
    "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
    "hub_model_id": "strdst7/TinyToolCaller",      # override via --hub-model-id
    "output_dir": "outputs/tinytoolcaller",
    # -- quantization (QLoRA, publication §10) ----------------------------- #
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    # Base and fine-tuned models are scored under the SAME quantization so
    # the comparison isolates fine-tuning, not precision (publication §14).
    "eval_load_in_4bit": True,
    # -- LoRA (publication §11) -------------------------------------------- #
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_bias": "none",
    "lora_task_type": "CAUSAL_LM",
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    # -- training (publication §11) ---------------------------------------- #
    "learning_rate": 2e-4,
    "num_epochs": 2,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,               # effective batch = 16
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "optim": "paged_adamw_8bit",
    "gradient_checkpointing": True,
    "logging_steps": 10,
    "save_strategy": "epoch",
    # -- evaluation (publication §14) -------------------------------------- #
    "max_new_tokens": 256,
    "gsm8k_n": 50,
    "gsm8k_dataset_id": "gsm8k",
    "gsm8k_config": "main",
    "gsm8k_split": "test",                          # gated; falls back to train
}

SYSTEM_PROMPT = (
    "You are a function-calling assistant. Given the user's request and the "
    "available tools, select the correct tool and construct the correct "
    "arguments. Respond with ONLY a JSON object containing exactly two keys: "
    '"name" (the tool name) and "arguments" (an object of argument values). '
    "Do not include markdown, explanations, or any other text."
)
