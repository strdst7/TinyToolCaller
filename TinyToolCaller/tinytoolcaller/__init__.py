"""TinyToolCaller — QLoRA fine-tuning of Qwen2.5-1.5B-Instruct for function calling.

Package layout (mirrors the publication §13):

    config.py       Central experiment configuration + system prompt.
    formatting.py   Pure prompt/JSON/answer helpers (no heavy dependencies).
    data.py         Tokenizer/dataset loading and deterministic sampling.
    model.py        Quantized model loading and LoRA attachment.
    metrics.py      Evaluation metrics and scorers.
    train.py        SFTTrainer wrapper and adapter/merge publication.
    repair.py       One-shot JSON repair loop (publication §3.1, §22.1).

Only `formatting`, `config`, and the dataclass part of `metrics` are safe to
import in a CPU/CI environment without torch, trl, peft, or bitsandbytes; the
heavy dependencies are imported lazily inside the functions that need them.
"""

__version__ = "0.1.0"
