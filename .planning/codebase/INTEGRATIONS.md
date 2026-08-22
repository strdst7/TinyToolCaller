# External Integrations

**Analysis Date:** 2026-08-21

## APIs & External Services

**Hugging Face Hub:**
- Model base: `Qwen/Qwen2.5-1.5B-Instruct` — Apache-2.0 licensed base model used as foundation
  - SDK/Client: `transformers` (`AutoModelForCausalLM.from_pretrained`, `AutoTokenizer.from_pretrained`)
  - Auth: Via `HF_TOKEN` env var (also supports `huggingface-cli login`)
- Model publication target: `strdst7/TinyToolCaller` (merged model) and `strdst7/TinyToolCaller-adapter` (LoRA adapter)
  - SDK/Client: `huggingface_hub` (`HfApi.create_repo`, `HfApi.upload_folder`)
  - Auth: `HF_TOKEN` env var required for write access
- Dataset source: `Salesforce/xlam-function-calling-60k` — CC-BY-4.0 licensed, **gated** (must accept terms)
  - SDK/Client: `datasets` (`load_dataset`)
  - Auth: `HF_TOKEN` env var required for gated access
- Project dataset: `strdst77/TinyToolCaller` — Apache-2.0 licensed derived subset
  - SDK/Client: `huggingface_hub` (`HfApi.upload_file`)
  - Format: Parquet files (`train.parquet`, `validation.parquet`)

**Weights & Biases (W&B):**
- Service: Experiment tracking and logging
  - SDK/Client: `wandb`
  - Auth: `WANDB_API_KEY` env var (optional)
  - Usage: `wandb.login()` in `train_tool_caller.py`, `report_to=["wandb"]` in SFTTrainer
  - Optional: Pipeline continues without W&B if key is missing

## Data Storage

**Databases:**
- Not detected (no SQLite, PostgreSQL, or other database connections)
- Dataset is loaded entirely in-memory via `datasets.load_dataset()` from Hugging Face Hub

**File Storage:**
- Local filesystem only (no cloud storage like S3, GCS, or Azure Blob)
- Artifacts written locally:
  - `outputs/tinytoolcaller/adapter/` — LoRA adapter weights
  - `outputs/tinytoolcaller/merged/` — Merged model weights
  - `outputs/eval_predictions.jsonl` — Per-example evaluation dump (optional, via `--eval-dump`)
  - `data/train.parquet` / `data/validation.parquet` — Local dataset artifacts

**Caching:**
- Not applicable (no Redis, Memcached, or other caching systems)

## Authentication & Identity

**Auth Provider:**
- Hugging Face token-based authentication (not OAuth or SSO)
  - Implementation: `HF_TOKEN` env var passed to `huggingface-cli` / `datasets` / `huggingface_hub`
  - No user identity management — this is an ML training pipeline, not a web application

## Monitoring & Observability

**Error Tracking:**
- Not detected (no Sentry, DataDog, or similar)

**Logs:**
- Console prints and W&B experiment tracking (optional)
- Training metrics logged via SFTTrainer's built-in `logging_steps=10`
- No structured logging framework (no `structlog`, `loguru`, etc.)

## CI/CD & Deployment

**Hosting:**
- Hugging Face Hub (model and dataset artifacts published via `huggingface_hub`)
- GitHub Actions (CI pipeline)

**CI Pipeline:**
- GitHub Actions — `.github/workflows/python-package.yml`
  - Trigger: Push or PR to `main` branch
  - Runner: `ubuntu-latest`
  - Python versions: 3.9, 3.10, 3.11 (matrix)
  - Steps:
    1. `actions/checkout@v4`
    2. `actions/setup-python@v3`
    3. Install deps: `pip install flake8 pytest` + `pip install -r requirements.txt`
    4. Lint: `flake8 . --count --select=E9,F63,F7,F82` + `--max-complexity=10 --max-line-length=127`
    5. Test: `pytest` (runs 41 tests on CPU)
  - Notes: CI runs `pytest` without GPU stack; heavy deps (torch, trl, peft) are installed but the lazy-import pattern means pure-unit tests pass on CPU

## Environment Configuration

**Required env vars:**
- `HF_TOKEN` — Required for loading gated source dataset `Salesforce/xlam-function-calling-60k` and publishing to Hub

**Optional env vars:**
- `WANDB_API_KEY` — Enables experiment tracking with Weights & Biases
- `WANDB_RUN_NAME` — Custom run name for W&B (defaults to `"tinytoolcaller-qlora"`)

**Secrets location:**
- Environment variables only (not stored in files)
- `.env` file pattern is gitignored but not used by the pipeline
- Documentation warns user to `export HF_TOKEN=<token>` manually

## Webhooks & Callbacks

**Incoming:**
- Not detected

**Outgoing:**
- Not detected (pipeline is CLI-driven, no server endpoints)

---

*Integration audit: 2026-08-21*