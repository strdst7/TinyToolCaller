# Codebase Structure

**Analysis Date:** 2026-08-21

## Directory Layout

```
TinyToolCaller/                         # Repository root
├── TinyToolCaller/                     # Primary project directory
│   ├── train_tool_caller.py            # CLI entry point (14-stage pipeline)
│   ├── requirements.txt                # Python dependencies
│   ├── README.md                       # Full publication (readthedocs-style)
│   ├── README.hf.md                    # Hugging Face dataset card
│   ├── LICENSE                         # Apache 2.0
│   ├── .gitignore                      # Ignores: __pycache__, outputs/, wandb/, .env
│   │
│   ├── tinytoolcaller/                 # ★ Core Python package (7 modules)
│   │   ├── __init__.py                 # Package docstring + version = "0.1.0"
│   │   ├── config.py                   # CONFIG dict + SYSTEM_PROMPT string
│   │   ├── formatting.py               # ChatML construction, JSON/GSM8K extraction (pure)
│   │   ├── data.py                     # Tokenizer loading, dataset loading, seed-42 split
│   │   ├── model.py                    # 4-bit NF4 loading, LoRA attachment
│   │   ├── metrics.py                  # ToolCallingMetrics dataclass, generate(), eval functions
│   │   ├── repair.py                   # One-shot JSON repair loop
│   │   └── train.py                    # SFTTrainer wrapper, save/merge/publish
│   │
│   ├── scripts/                        # Auxiliary analysis and publication tools
│   │   ├── profile_tool_distribution.py  # §8.1 — tool-distribution profiling (335 lines)
│   │   ├── dataset_stats.py              # §8.2 — basic dataset statistics (122 lines)
│   │   ├── statistical_analysis.py       # §18 — Wilson CI, McNemar test, bootstrap (211 lines)
│   │   ├── capture_environment.py        # §12 — environment capture (86 lines)
│   │   ├── publish_dataset.py            # Build & upload dataset to HF Hub (103 lines)
│   │   └── build_preprint.py             # README → PDF renderer via Markdown+WeasyPrint (184 lines)
│   │
│   ├── tests/                          # pytest test suite (41 tests)
│   │   ├── conftest.py                 # FakeTokenizer fixture, weather_example fixture (50 lines)
│   │   ├── test_config.py              # Config invariant checks (43 lines, 5 tests)
│   │   ├── test_formatting.py          # extract_json, GSM8K, ChatML formatting (94 lines, 7 tests)
│   │   └── test_metrics.py             # ToolCallingMetrics dataclass tests (30 lines, 4 tests)
│   │
│   ├── preprint/                       # Auto-generated publication PDF
│   │   └── TinyToolCaller_Publication.pdf
│   │
│   └── outputs/                        # Training artifacts (git-ignored)
│       └── environment.json            # Only committed env snapshot
│
├── TinyToolCaller-files/               # Published-version snapshot (same structure)
│   ├── (identical layout to TinyToolCaller/)
│   ├── outputs/
│   │   └── environment.json
│
├── hf-dataset/                         # Hugging Face dataset card only
│   └── README.md
│
├── .github/                            # CI workflows (empty — no .github/workflows detected)
└── README.md                           # Repo-level README
```

## Directory Purposes

**tinytoolcaller/ (Core Package):**
- Purpose: All experiment logic — configuration, data processing, model loading, training, evaluation, repair
- Contains: 7 Python modules, each with a single responsibility (see table below)
- Key files: `__init__.py` (package docstring), `config.py` (central config)

**tests/:**
- Purpose: pytest unit tests covering the pure-logic modules (no GPU required)
- Contains: `conftest.py` (shared fixtures: `FakeTokenizer`, `weather_example`), 3 test modules (16 tests total for formatting, 5 for config, 4 for metrics)
- Key files: `conftest.py` (fixture factory), `test_formatting.py` (largest — tests JSON extraction, GSM8K, ChatML)

**scripts/:**
- Purpose: Standalone analysis tasks (tool profiling, dataset statistics, statistical significance testing, environment capture, Hub publishing, preprint rendering)
- Contains: 6 standalone CLI scripts, independent from the training pipeline

**preprint/:**
- Purpose: Auto-generated PDF of the publication (`scripts/build_preprint.py`)
- Contains: `TinyToolCaller_Publication.pdf` (build artifact, committed)

## Key Locations

| Artifact | Path | Notes |
|---|---|---|
| CLI entry point | `TinyToolCaller/train_tool_caller.py` | `#!/usr/bin/env python3`, 199 lines |
| Central config | `TinyToolCaller/tinytoolcaller/config.py` | All tunable values in one dict (57 lines) |
| System prompt | `TinyToolCaller/tinytoolcaller/config.py:51-57` | `SYSTEM_PROMPT` constant |
| Package init + version | `TinyToolCaller/tinytoolcaller/__init__.py` | `__version__ = "0.1.0"` |
| Dependencies | `TinyToolCaller/requirements.txt` | 19 lines, 17 packages |
| License | `TinyToolCaller/LICENSE` | Apache 2.0 |
| .gitignore | `TinyToolCaller/.gitignore` | Ignores `outputs/`, `wandb/`, `__pycache__/`, `.env` |

## Naming Conventions

**Python style:** PEP 8 consistent across all files.

- **Snake_case** for all functions, variables, and file names:
  - Functions: `load_tokenizer`, `sample_and_split`, `format_for_training`, `extract_json`, `evaluate_gsm8k`, `save_and_publish`
  - Variables: `train_ds`, `val_ds`, `max_new_tokens`, `bnb_config_kwargs`
  - Files: `train_tool_caller.py`, `build_preprint.py`, `dataset_stats.py`
  - Test files: `test_config.py`, `test_formatting.py`, `test_metrics.py`

- **UPPER_CASE** for module-level constants:
  - `CONFIG` (`config.py`)
  - `SYSTEM_PROMPT` (`config.py`)
  - `REPAIR_INSTRUCTION` (`repair.py`)
  - `SOURCE_DATASET`, `DEFAULT_REPO_ID` (`scripts/publish_dataset.py`)

- **CamelCase** for classes and dataclasses:
  - `ToolCallingMetrics` (`metrics.py`)
  - `FakeTokenizer` (`tests/conftest.py`)
  - `SFTConfig`, `SFTTrainer`, `LoraConfig` (external library types)

**Package structure conventions:**
- `__init__.py` contains a module-level docstring describing each submodule's role (publication §13 reference) and the `__version__` string
- One submodule per concern — none exceed 135 lines
- Pure vs. heavy modules documented in `__init__.py` docstring for import-safety guidance
- All module docstrings reference the publication section (§) for traceability
- Lazy imports are documented via inline comments: `# lazy: inside function, not at module top`

**Test conventions:**
- `conftest.py` at `tests/` level (no per-module `conftest.py`)
- `test_<module>.py` naming matches `tinytoolcaller/<module>.py` (e.g., `test_formatting.py` tests `tinytoolcaller/formatting.py`)
- `test_` prefix on all test functions
- Parametrized tests via `@pytest.mark.parametrize` for table-driven cases (JSON extraction, GSM8K extraction)
- `FakeTokenizer` fixture avoids real HF tokenizer (no network)
- `weather_example` fixture provides a realistic tool-calling example dict

**File size distribution:**

| File | Lines | Role |
|---|---|---|
| `scripts/profile_tool_distribution.py` | 335 | Largest standalone script |
| `tinytoolcaller/formatting.py` | 131 | Largest core module (pure) |
| `tinytoolcaller/metrics.py` | 117 | Evaluation logic |
| `tinytoolcaller/train.py` | 96 | Training + publish |
| `tests/test_formatting.py` | 94 | Largest test module |
| `tinytoolcaller/config.py` | 57 | Config dict + prompt |
| `tinytoolcaller/model.py` | 50 | Model loading |
| `tinytoolcaller/data.py` | 46 | Data loading |
| `tests/test_config.py` | 43 | Config invariant tests |
| `tinytoolcaller/repair.py` | 39 | JSON repair loop |
| `tests/test_metrics.py` | 30 | Metrics tests |
| `tinytoolcaller/__init__.py` | 18 | Package docstring + version |