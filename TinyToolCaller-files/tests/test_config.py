"""Invariant checks: the code's CONFIG must match the documented configuration."""

from tinytoolcaller.config import CONFIG


def test_documented_seed_and_splits():
    assert CONFIG["seed"] == 42
    assert CONFIG["n_train"] == 5000
    assert CONFIG["n_val"] == 200
    assert CONFIG["n_sample"] == CONFIG["n_train"] + CONFIG["n_val"]


def test_lora_hyperparameters_match_publication():
    assert CONFIG["lora_rank"] == 16
    assert CONFIG["lora_alpha"] == 32
    assert CONFIG["lora_dropout"] == 0.05
    assert CONFIG["lora_bias"] == "none"
    assert CONFIG["lora_task_type"] == "CAUSAL_LM"


def test_target_modules_cover_attention_and_mlp():
    assert set(CONFIG["target_modules"]) == {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    }


def test_training_config_matches_publication():
    assert CONFIG["learning_rate"] == 2e-4
    assert CONFIG["num_epochs"] == 2
    assert CONFIG["per_device_train_batch_size"] == 2
    assert CONFIG["gradient_accumulation_steps"] == 8
    assert (CONFIG["per_device_train_batch_size"]
            * CONFIG["gradient_accumulation_steps"]) == 16
    assert CONFIG["lr_scheduler_type"] == "cosine"
    assert CONFIG["optim"] == "paged_adamw_8bit"
    assert CONFIG["max_seq_length"] == 1024


def test_quantization_config():
    assert CONFIG["load_in_4bit"] is True
    assert CONFIG["bnb_4bit_quant_type"] == "nf4"
    assert CONFIG["bnb_4bit_use_double_quant"] is True
