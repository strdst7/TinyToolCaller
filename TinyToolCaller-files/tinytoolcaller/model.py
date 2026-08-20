"""Quantized model loading and LoRA attachment (publication §10-§11)."""

from __future__ import annotations


def load_quantized_model(model_id: str, load_in_4bit: bool,
                         bnb_config_kwargs: dict):
    """Load the base model, optionally 4-bit NF4 quantized.

    ``load_in_4bit=False`` loads bf16 instead — used by the bf16-baseline
    ablation in publication §25.4.
    """
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    kwargs = {}
    if load_in_4bit:
        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            **bnb_config_kwargs,
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16
    kwargs["device_map"] = "auto"
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


def attach_lora(model, config: dict):
    """Prepare the quantized model for k-bit training and attach LoRA."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if config["load_in_4bit"]:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config["gradient_checkpointing"]
        )
    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias=config["lora_bias"],
        task_type=config["lora_task_type"],
        target_modules=config["target_modules"],
    )
    return get_peft_model(model, lora_config)
