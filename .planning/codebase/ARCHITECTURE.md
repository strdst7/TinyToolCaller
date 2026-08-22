<!-- refreshed: 2026-08-21 -->
# Architecture

**Analysis Date:** 2026-08-21

## System Overview

```text
┌─────────────────────────────────────────────────────────────────┐
│                      Command-Line Entry Point                     │
│              `train_tool_caller.py` (199 lines)                   │
│    argparse-based CLI wiring 14 pipeline stages (publication §13) │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│                      tinytoolcaller/ Package                      │
│                                                                   │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ config   │  │ formatting   │  │ data     │  │ repair       │  │
│  │ config.py│  │ formatting.py│  │ data.py  │  │ repair.py    │  │
│  │ (pure)   │  │ (pure)       │  │ (lazy)   │  │ (pure)       │  │
│  └────┬─────┘  └──────┬───────┘  └────┬─────┘  └──────┬───────┘  │
│       │               │               │               │          │
│       ▼               ▼               ▼               ▼          │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  model.py  (lazy: torch, transformers, peft, bitsandbytes)   │ │
│  │  train.py  (lazy: torch, trl, SFTConfig/TrainingArguments)   │ │
│  │  metrics.py (lazy: torch inside generate(); end-to-end CPU)  │ │
│  └──────────────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────────────┤
│                        Support Scripts                            │
│  scripts/                                                         │
│  ├── profile_tool_distribution.py   §8.1 — tool distribution      │
│  ├── dataset_stats.py               §8.2 — dataset statistics     │
│  ├── statistical_analysis.py        §18  — CIs, McNemar           │
│  ├── capture_environment.py         §12  — env capture            │
│  ├── publish_dataset.py                  — HF Hub publication     │
│  └── build_preprint.py                   — README → PDF render    │
└──────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Outputs                                                        │
│  outputs/tinytoolcaller/                                        │
│  ├── adapter/   (LoRA adapter weights + tokenizer)              │
│  └── merged/    (merged full model + tokenizer)                 │
│                                                                  │
│  Hugging Face Hub: strdst7/TinyToolCaller                       │
└─────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Entry point | Orchestrates 14-stage pipeline: load → split → eval baseline → quantize → train → eval tuned → GSM8K → merge → publish | `train_tool_caller.py` |
| Config | Central `CONFIG` dict (all tunable hyperparameters) + `SYSTEM_PROMPT` | `tinytoolcaller/config.py` |
| Formatting | ChatML construction, JSON extraction, GSM8K answer extraction (pure Python, no heavy deps) | `tinytoolcaller/formatting.py` |
| Data | Tokenizer loading, gated dataset loading, deterministic seed-42 split | `tinytoolcaller/data.py` |
| Model | 4-bit NF4 quantized model loading + LoRA adapter attachment | `tinytoolcaller/model.py` |
| Metrics | O-FME evaluation (JSON validity / tool accuracy / argument exact match), GSM8K probe | `tinytoolcaller/metrics.py` |
| Repair | One-shot JSON repair loop with injected `generate_fn` | `tinytoolcaller/repair.py` |
| Train | SFTTrainer wrapper (TRL API shim), adapter save + merge + Hub publication | `tinytoolcaller/train.py` |

## Pattern Overview

**Overall:** Pipeline-architected ML experiment — a single monolithic CLI script (`train_tool_caller.py`) that drives the `tinytoolcaller/` package through 14 numbered stages.

**Key Characteristics:**
- **Lazy import boundary** — `tinytoolcaller/` modules that require `torch`, `trl`, `peft`, or `bitsandbytes` import them inside function bodies, not at module top-level. This allows the pure submodules (`config.py`, `formatting.py`, and the `ToolCallingMetrics` dataclass in `metrics.py`) to be imported and unit-tested on CPU/CI without any GPU stack.
- **Deterministic by design** — greedy decoding (`do_sample=False`), fixed seed 42 for data splits, `datasets.shuffle(seed=seed)`. All evaluation is reproducible.
- **Dependency injection** — `repair.py`'s `repair()` accepts a `generate_fn` callable rather than a model/tokenizer pair, enabling unit tests without GPU.
- **Shared parser contract** — `formatting.extract_json` is used both at evaluation time (`metrics.py`) and in the repair loop (`repair.py`), ensuring evaluation and mitigation share one definition of "valid JSON".

## Layers

**CLI Layer:**
- Purpose: Parses flags, wires the pipeline, handles user overrides
- Location: `train_tool_caller.py` (root of `TinyToolCaller/TinyToolCaller/`)
- Contains: `main()`, `_dump_eval()`, argparse definitions
- Depends on: All seven `tinytoolcaller.*` modules
- Used by: End user via `python train_tool_caller.py`

**Core Logic Layer (tinytoolcaller/):**
- Purpose: All experiment logic — configuration, data processing, model loading, training, evaluation
- Location: `tinytoolcaller/`
- Contains: 7 modules organized by concern
- Depends on: External libraries (torch, transformers, datasets, peft, trl, bitsandbytes) — imported lazily
- Used by: `train_tool_caller.py`

**Scripts Layer:**
- Purpose: Auxiliary analysis, profiling, dataset publication, and publication PDF generation
- Location: `scripts/`
- Contains: 6 standalone scripts (see table above)
- Used by: Researcher for profiling, analysis, and artifact publishing

**Tests Layer:**
- Purpose: Unit tests for pure-logic modules (formatting, metrics, config)
- Location: `tests/`
- Contains: 3 test files + conftest.py
- Depends on: pytest, `tinytoolcaller.formatting`, `tinytoolcaller.metrics`, `tinytoolcaller.config`

## Data Flow

### Primary Pipeline Path (train_tool_caller.py main())

1. **[1] Load tokenizer** — `data.load_tokenizer(model_id)` → HuggingFace `AutoTokenizer` with pad=eos fallback (`data.py:6-18`)
2. **[2-3] Load + split** — `data.load_source_dataset(dataset_id)` → `datasets.Dataset` → `data.sample_and_split(ds, 5200, 5000, 42)` → `(train_ds, val_ds)` (`data.py:21-46`)
3. **[4] Format ChatML** — `train_ds.map(formatting.format_for_training)` applies tokenizer chat template with system+user+assistant turns (`formatting.py:52-63`)
4. **[5] Baseline eval** — `model.load_quantized_model(...)` → `metrics.evaluate_tool_calling(base_model, tokenizer, val_ds, ...)` → `ToolCallingMetrics` dataclass (`metrics.py:67-94`)
5. **[6-8] QLoRA setup** — `model.load_quantized_model(...)` → `model.attach_lora(model, config)` → trainable PEFT model (`model.py:6-50`)
6. **[9] Training** — `train.train(model, tokenizer, train_ds, config, trainable_params)` → TRL `SFTTrainer.train()` (`train.py:8-61`)
7. **[11] Fine-tuned eval** — `metrics.evaluate_tool_calling(trainer.model, ...)` → `ToolCallingMetrics` + optional per-example detail dump (`metrics.py:67-94`)
8. **[12] GSM8K retention** — `metrics.evaluate_gsm8k(model, tokenizer, 50, ...)` → accuracy % (`metrics.py:97-117`)
9. **[13-14] Save & publish** — `train.save_and_publish(trainer, tokenizer, config, push)` → adapter dir + merged dir + optional Hub upload (`train.py:64-96`)

### Evaluation Data Flow

1. `formatting.format_for_inference(example, tokenizer)` → ChatML string (system+user, no assistant) (`formatting.py:66-70`)
2. `metrics.generate(model, tokenizer, prompt, max_new_tokens)` → greedy-decoded raw string (`metrics.py:50-64`)
3. `formatting.extract_json(raw)` → parsed dict or None (`formatting.py:76-111`)
4. `formatting.ground_truth(example)` → expected `{"name": ..., "arguments": ...}` (`formatting.py:38-49`)
5. `ToolCallingMetrics` increments counters for validity, tool match, argument match (`metrics.py:20-47`)

### Repair Data Flow

1. `repair.repair(raw, generate_fn, prompt, max_attempts=1)` — loop: if `extract_json(current)` is None, re-prompt with `REPAIR_INSTRUCTION + current` (`repair.py:28-39`)
2. Returns `(raw_text, attempts)` — same `extract_json` parser as evaluation

**State Management:**
- No global mutable state — all state is local to the `main()` function in `train_tool_caller.py`
- `CONFIG` dict in `config.py` is a module-level constant (frozen dict-like, mutated only by CLI overrides in `main()` before any heavy work)
- Model state (weights, LoRA adapters) is held in PyTorch `nn.Module` instances and serialized via `trainer.save_model()` / `model.save_pretrained()`

## Key Abstractions

**ToolCallingMetrics dataclass:**
- Purpose: Container for three O-FME axes — JSON validity, tool-name accuracy, argument exact match — with computed percentage properties and a table formatting method
- File: `tinytoolcaller/metrics.py:20-47`
- Pattern: Immutable after construction (all fields set at init); properties for derived percentages; `as_table()` for display

**fake_tokenizer (test fixture):**
- Purpose: Minimal chat-template tokenizer for CPU/CI unit tests
- File: `tests/conftest.py:14-24`
- Pattern: Implements only `apply_chat_template()` — the single method used by the formatting functions

**One-shot repair loop:**
- Purpose: Injectable retry-with-repair that re-prompts the model with its own faulty output
- File: `tinytoolcaller/repair.py:28-39`
- Pattern: Dependency injection — `generate_fn` is a callable parameter, keeping the function pure and testable without GPU

## Entry Points

**train_tool_caller.py:**
- Location: `TinyToolCaller/TinyToolCaller/train_tool_caller.py`
- Triggers: Direct invocation `python train_tool_caller.py [--flags]`
- Responsibilities: Parse CLI args, apply config overrides, orchestrate all 14 pipeline stages, print comparison results, handle Hub publication

**Individual scripts/:**
- Location: `TinyToolCaller/TinyToolCaller/scripts/*.py`
- Triggers: Direct `python scripts/<script_name>.py`
- Responsibilities: Standalone analysis tasks (tool distribution profiling, dataset statistics, statistical analysis, environment capture, dataset publication, preprint PDF generation)

## Architectural Constraints

- **Lazy imports boundary:** Only `config.py`, `formatting.py`, and the `ToolCallingMetrics` dataclass in `metrics.py` are safe to import without `torch`, `trl`, `peft`, or `bitsandbytes`. All other modules import heavy deps inside function bodies. This enables the 41-test suite to run on CPU/CI.
- **Single-threaded:** No threading or multiprocessing. Training uses PyTorch's single-process GPU training loop.
- **No global state:** Beyond the immutable `CONFIG` dict, there is no module-level mutable state. Model instances are passed explicitly.
- **Deterministic random seed:** Seed 42 is used for data splitting and GSM8K sampling. Greedy decoding ensures deterministic generation.
- **No circular imports:** The package is a strict DAG: `config.py` → `formatting.py` → `metrics.py` and `repair.py`; `data.py`, `model.py`, `train.py` are independent consumers of `config.py`.

## Anti-Patterns

### Monolithic Entry Point

**What happens:** `train_tool_caller.py` (199 lines) contains the entire pipeline orchestration in a single `main()` function. All 14 pipeline stages are sequential blocks with no modular decomposition between stages.

**Why it's wrong:** Adding new stages, making stages conditional on intermediate results, or testing individual stages in isolation requires modifying this single function. The `--no-baseline`, `--no-train`, `--no-eval`, `--skip-gsm8k` flags create combinatorial complexity.

**Do this instead:** Decompose into a `Pipeline` class or separate orchestration layer with composable stage objects. See `tinytoolcaller/train.py:save_and_publish` for an example of a well-encapsulated sub-pipeline.

### Single mutable CONFIG dict

**What happens:** `config.py` exports a bare `dict` that is mutated by CLI overrides in `main()` (`train_tool_caller.py:84-89`). Downstream functions receive `config: dict` — any key can be accessed anywhere with no type safety.

**Why it's wrong:** Typo in a key name → silent `KeyError` at runtime. No IDE autocompletion. No validation that required keys exist.

**Do this instead:** Use a `@dataclass` or `pydantic.BaseModel` for `Config`. See `ToolCallingMetrics` (`metrics.py:20-47`) for the dataclass pattern used elsewhere in the project.

### Hardcoded seed 42

**What happens:** The seed value 42 is defined in `CONFIG["seed"]` and used in `data.sample_and_split()` and `metrics.evaluate_gsm8k()`. It is configurable only via the `CONFIG` dict, not via CLI flag.

**Why it's wrong:** Multi-seed robustness checks (proposed in §15, Step B2) require editing config values or code. The `--max-seq-length` and `--output-dir` flags show the precedent for CLI-overridable config values exists.

**Do this instead:** Add `--seed` CLI argument to `train_tool_caller.py`, similar to `--max-seq-length`.

## Error Handling

**Strategy:** Defensive with actionable error messages. The gated dataset helper (`data.py:27-33`) raises `SystemExit` with instructions for HF login. Tokenizer pad-token fallback (`data.py:16-17`) handles a common silent failure. The TRL API shim (`train.py:41-56`) catches `ImportError` to support two API versions.

**Patterns:**
- `SystemExit` with instructions — `data.py:27-33`
- `except Exception: pass` only where the fallback is well-defined (TRL API version check, GSM8K split fallback)
- No custom exception hierarchy — all errors use built-in or library exceptions

## Cross-Cutting Concerns

**Logging:** Print-based logging throughout `train_tool_caller.py` and `train.py`. Weights & Biases integration is optional (controlled by `WANDB_API_KEY` env var, `train.py:34-35`). No Python `logging` module usage.

**Validation:** Config invariant tests in `tests/test_config.py` (43 lines) assert every documented hyperparameter value matches the code. Data validation is structural only (see `validate_example` docstring in README §9.2).

**Authentication:** HF Hub token via `HF_TOKEN` environment variable for gated dataset access and model publication. No OAuth, no API keys in code.

---

*Architecture analysis: 2026-08-21*