# Technology Stack

**Analysis Date:** 2026-08-21

## Languages

**Primary:**
- Python 3.10+ — All source code, training pipeline, evaluation scripts, and tests

**Secondary:**
- Not detected (no TypeScript, Go, Rust, or other compiled languages in the codebase)

## Runtime

**Environment:**
- Python 3.10+ (tested on 3.9, 3.10, 3.11 via CI; analyzed on 3.14.5 on macOS 15.6 arm64)
- No nvm, nodenv, or `.python-version` file detected

**Package Manager:**
- pip (standard Python pip)
- Lockfile: Not present (no `requirements.lock` or `Pipfile.lock`)

## Frameworks

**Core:**
- PyTorch 2.x (CUDA-compatible) — Tensor backend, model execution, GPU compute
- Hugging Face Transformers 4.x/5.x — Model/tokenizer loading, ChatML template (`AutoModelForCausalLM`, `AutoTokenizer`, `BitsAndBytesConfig`, `TrainingArguments`)
- Hugging Face PEFT 0.x — LoRA adapter attachment (`LoraConfig`, `get_peft_model`, `prepare_model_for_kbit_training`)
- Hugging Face TRL 0.8+ / 0.12+ — SFTTrainer for supervised fine-tuning (supports both API versions via shim)
- Hugging Face Datasets 2.14+/5.0.1 — Dataset loading, shuffling, splitting, profiling
- Hugging Face Accelerate — Device management for multi-GPU/CPU
- Hugging Face Hub (`huggingface_hub`) — Model/dataset publication, API interaction

**Testing:**
- pytest — Test runner (41 tests across 4 test files)
  - Configuration: `pytest.ini` — Not present (relies on `python -m pytest`)
  - Fixtures in `tests/conftest.py` — `FakeTokenizer`, `weather_example`

**Build/Dev:**
- flake8 — Linting (enforced in CI: 4-space indentation, 88-char lines, double-quoted strings, max-complexity=10)
- No build system (no setuptools/pyproject.toml — package installed via `PYTHONPATH` manipulation in `conftest.py`)

## Key Dependencies

**Critical:**
- `torch` — Core tensor computation; loaded lazily inside functions to keep modules importable on CPU
- `transformers` — All Hugging Face model/tokenizer operations; version 4.x/5.x tested
- `datasets` — Dataset I/O; version 5.0.1 used for profiling (shuffle RNG is version-sensitive — documented in publication §12)
- `peft` — LoRA adapter attachment; loaded lazily
- `trl` — SFTTrainer; loaded lazily; version shim handles `SFTConfig` (>=0.12) vs older API
- `bitsandbytes` — 4-bit NF4 quantization (Linux CUDA only; CPU fallback handled)
- `accelerate` — Device map management

**Infrastructure:**
- `numpy` — Numerical operations in scripts
- `scipy` — Statistical analysis (Wilson confidence intervals, McNemar test, bootstrap)
- `pandas` — Data profiling and stats
- `jsonschema` — JSON Schema validation for production safety gate (`scripts/build_architecture.py` references, `tinytoolcaller/metrics.py` import mention)
- `wandb` — Experiment tracking (optional; only initialized when `WANDB_API_KEY` environment variable is set)

## Configuration

**Environment:**
- Configuration via `tinytoolcaller/config.py` (`CONFIG` dict — central experiment configuration):
  - `source_dataset_id`: `"Salesforce/xlam-function-calling-60k"` (gated Hugging Face dataset)
  - `model_id`: `"Qwen/Qwen2.5-1.5B-Instruct"` (base model)
  - `hub_model_id`: `"strdst7/TinyToolCaller"` (publication target)
  - Training hyperparameters: LoRA rank 16, alpha 32, dropout 0.05, learning rate 2e-4, cosine scheduler, 2 epochs
  - Quantization: 4-bit NF4, double quantization, `eval_load_in_4bit=True`
  - Sequence length: 1024 max
  - GSM8K dataset: `"gsm8k"`, config `"main"`, split `"test"` (falls back to `"train"`)
- Environment variables (never stored in `.env` files):
  - `HF_TOKEN` — Required for gated dataset access
  - `WANDB_API_KEY` — Optional experiment tracking
  - `WANDB_RUN_NAME` — Optional run naming (default: `"tinytoolcaller-qlora"`)
- `.env` listed in `.gitignore` — no env files committed

**Build:**
- No build config files (no `pyproject.toml`, `setup.py`, `setup.cfg`, `Makefile`)

## Platform Requirements

**Development:**
- macOS or Linux (analysis/profiling tested on macOS 15.6 arm64)
- Linux with NVIDIA GPU + CUDA for training
- Python 3.9–3.11 (CI matrix); 3.10+ recommended
- 8 GB RAM minimum for analysis; 16 GB recommended
- Optional: NVIDIA GPU with 8 GB VRAM minimum, 16 GB recommended for training

**Production:**
- Inference on CPU: 8 GB RAM minimum
- Inference on GPU: 4 GB VRAM minimum, 8 GB recommended
- Deployment target: Hugging Face Hub (model and dataset repositories)

---

*Stack analysis: 2026-08-21*