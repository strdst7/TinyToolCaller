# Coding Conventions

**Analysis Date:** 2026-08-21

## Naming Patterns

**Files:**
- Python source modules in `tinytoolcaller/` use `snake_case.py`: `config.py`, `formatting.py`, `data.py`, `model.py`, `train.py`, `metrics.py`, `repair.py`
- Test files use `test_<module>.py`: `test_config.py`, `test_formatting.py`, `test_metrics.py`
- Scripts in `scripts/` use `snake_case.py`: `statistical_analysis.py`, `publish_dataset.py`, `profile_tool_distribution.py`, `capture_environment.py`, `build_preprint.py`, `dataset_stats.py`
- Entry point: `train_tool_caller.py` (at repo root, lowercase with underscores)

**Functions:**
- All functions use `snake_case`: `build_messages`, `extract_json`, `load_tokenizer`, `sample_and_split`, `evaluate_tool_calling`, `save_and_publish`
- Private/helper functions prefixed with underscore: `_as_obj`, `_answer_name`, `_rows_from_path`, `_pct`, `_fmt`, `_dump_eval`, `_generator`
- Property-like accessors on dataclasses use `@property` decorated methods: `json_validity`, `tool_accuracy`, `argument_exact_match`
- Generator closures used in tests: `_generator(responses)` at `tests/test_repair.py`

**Variables:**
- Local variables use `snake_case`: `tools_json`, `raw`, `pred`, `gt`, `train_ds`, `val_ds`
- Loop variables are short: `ex`, `m`, `b`, `f`, `row`, `ch`
- Boolean flags in config are `snake_case`: `load_in_4bit`, `gradient_checkpointing`, `eval_load_in_4bit`
- Config dictionary keys use `snake_case`: `"n_sample"`, `"max_seq_length"`, `"lora_rank"`

**Types:**
- Dataclasses follow PascalCase: `ToolCallingMetrics`, `FakeTokenizer` (test stub)
- Type hints from `from __future__ import annotations` used throughout (all modules use this import)
- `dict` and `list` used as type annotations (no `TypedDict` usage observed)
- `callable` used in function signatures for injected dependencies: `generate_fn: callable(prompt) -> raw_text`

**Constants:**
- Module-level constants are `UPPER_SNAKE_CASE`: `CONFIG`, `SYSTEM_PROMPT`, `REPAIR_INSTRUCTION`, `Z` (in `statistical_analysis.py`), `REPORTED`, `SOURCE_DATASET`, `DEFAULT_REPO_ID`, `HAS_SCIPY`
- Minor exceptions: `ROOT` (Path object) is uppercase, `HAS_SCIPY` is mixed case

## Code Style

**Formatting:**
- No explicit formatter configuration detected (no `.prettierrc`, `pyproject.toml`, or `setup.cfg` with formatter settings)
- Code is manually formatted with consistent style: 4-space indentation, double-quoted strings preferred
- Maximum line length ~100 chars; some lines up to ~120 chars exist in format strings
- Blank lines between sections marked with `# ---- comment ----` section dividers (used in `formatting.py`, `test_formatting.py`, `profile_tool_distribution.py`)
- `import torch` is done lazily inside functions (see `metrics.py:52`, `train.py:16`, `train_tool_caller.py:142`) to allow CPU-only testing

**Linting:**
- GitHub Actions CI uses `flake8` with `--select=E9,F63,F7,F82 --show-source --statistics` for errors and `--exit-zero --max-complexity=10 --max-line-length=127 --statistics` for warnings
- Lint config is at `.github/workflows/python-package.yml` (inline flags, no `.flake8` file)
- `# noqa: E402` used for import-before-path-setup in scripts (e.g., `dataset_stats.py:41-42`)
- `# noqa: BLE001` used for broad exception catches in evaluation/training loops (`metrics.py:104`, `train.py:95`)
- `# noqa: F401` used for lazy torch import (`train_tool_caller.py:142`)

## Import Organization

**Order:**
1. `from __future__ import annotations` — first import in every module (present in 8/9 source modules)
2. Standard library: `import json`, `import re`, `import os`, `import sys`, `import math`, `from pathlib import Path`, `from collections import Counter`, `from dataclasses import dataclass`
3. Third-party: `import torch`, `import pytest`, `import numpy as np`, `from transformers import ...`, `from datasets import ...`, etc.
4. Local: `from .config import ...`, `from .formatting import ...`, `from tinytoolcaller import ...`

**Path Aliases:**
- No aliases or path rewriting for imports; package is importable as `tinytoolcaller`
- `conftest.py` at `tests/conftest.py` inserts the repo root into `sys.path` to enable `from tinytoolcaller import ...` during test runs
- Private intra-package imports use relative form: `from .formatting import extract_json` (`repair.py:19`, `metrics.py:11`)

## Error Handling

**Patterns:**
- **Broad exception handling in evaluation/training loops:** `except Exception as exc:  # noqa: BLE001` — used in `metrics.py:104` (GSM8K dataset fallback) and `train.py:95` (Hub publication failure). These are intentional: non-critical failures in loops should not abort the full run.
- **Actionable error messages:** `data.py:28-33` raises `SystemExit` with a user-facing message including exact instructions for fixing the issue (auth token, dataset URL).
- **Pair of `try/except` for API detection:** `train.py:41-56` uses `try: from trl import SFTConfig` to handle TRL version differences between >= 0.12 and older versions. The `except ImportError` branch provides backward compatibility.
- **Conditional capability detection:** `profile_tool_distribution.py:46-51` uses `try: from scipy.stats import chi2_contingency` with `HAS_SCIPY` flag, skipping chi-square when scipy is absent. Similarly `capture_environment.py:28-41` uses optional imports for version collection.
- **No custom exception classes** defined anywhere in the codebase.

**Return types for errors:**
- Pure functions return `None` for failure: `extract_json(text)` returns `None` if JSON cannot be extracted
- Tuple `(ok_bool, reason_string)` pattern used in `TinyToolCaller-files/tests/test_data.py`: `validate_example()` returns `(True, "ok")` or `(False, "missing_or_empty_query")`
- `repair()` returns the raw text even if repair fails (the caller inspects the result)

## Logging

**Framework:** `builtins.print` — no logging framework (no `import logging` anywhere in the codebase)

**Patterns:**
- Stage-numbered output with brackets: `print("\n[1] Loading tokenizer ...")` in `train_tool_caller.py`
- Section separators: `print("=" * 72)` for major section boundaries
- Tab-prefixed metric output: `print("    BASELINE")` then `print(baseline.as_table())`
- Conditionally reported: `print(f"[14] Skipping Hub publication (--no-push).")` when skipped; warnings printed when env vars are missing
- Scripts produce Markdown tables printed to stdout (e.g., `capture_environment.py:65-76`, `dataset_stats.py:103-117`)
- No structured logging (no JSON logs, no log levels)

## Comments

**When to Comment:**
- Module-level docstrings (`""" ... """`) are present in every source file, describing the module purpose and linking to publication sections (e.g., `"""Pure prompt-construction and extraction helpers (no heavy dependencies)."""` at `formatting.py:1-6`)
- Function docstrings present for all public functions, describing purpose, parameters, return values, and design rationale
- Inline comments explain non-obvious decisions: `# Balanced-brace scan of the first '{...}' region, tolerant of wrappers.` (`formatting.py:94`)
- Section markers: `# ---- ChatML construction (publication §9) ----` (`formatting.py:16-18`) and similar
- Design decision comments preserved in code (e.g., `repair.py:8-14`: "ONE repair attempt by default — each attempt costs latency and tokens")

**JSDoc/TSDoc:**
- Not applicable (Python project). Python docstrings are Google-style: brief summary line, optional blank line, then detailed description with Args/Returns sections where needed.

## Function Design

**Size:**
- Most functions are concise (10–30 lines). Largest: `evaluate_tool_calling` (~30 lines), `train()` (~60 lines including imports), `build_report()` (~60 lines), `main()` in `train_tool_caller.py` (~140 lines)
- `main()` functions in scripts tend to be longer (CLI parsing + orchestration)
- Pure data helpers are very short: `ground_truth()` is 6 lines, `normalise_number()` is 7 lines

**Parameters:**
- Config dictionaries passed as `config: dict` rather than individual parameters (e.g., `model.attach_lora(ft_model, CONFIG)` at `train_tool_caller.py:156`)
- Tokenizer, model, dataset objects passed directly
- Boolean flags for optional behavior: `return_details: bool = False` (`metrics.py:67`), `push: bool` (`train.py:64`)
- Default parameter values used where sensible: `max_attempts: int = 1` (`repair.py:28`)

**Return Values:**
- Pure functions return the transformed value directly
- Functions with multiple modes return `(metrics, details)` tuple when `return_details=True` (pattern: result = evaluate_tool_calling(...); metrics, details = result if return_details else (result, None))
- `repair()` returns `(raw_text, attempts)` tuple
- `ToolCallingMetrics.as_table()` returns a formatted `str`
- `main()` returns `int` (exit code) in scripts

## Module Design

**Exports:**
- No `__all__` defined in any module
- All public functions are imported explicitly: `from tinytoolcaller.formatting import build_messages` — no star imports
- Heavy dependencies (torch, trl, peft, bitsandbytes) are imported lazily inside functions that need them, making the `formatting`, `config`, and `metrics` (dataclass only) modules safe for CPU/CI

**Barrel Files:**
- `tinytoolcaller/__init__.py` serves as package documentation only (docstring and `__version__`). It does NOT re-export any symbols. All imports are explicit from submodules.

**Design Philosophy:**
- **Separation of pure and impure:** Pure prompt-construction and parsing lives in `formatting.py`; side-effect-heavy model/training code lives in `model.py`, `train.py`, `data.py`
- **Injectable dependencies:** `repair.py` takes `generate_fn` as a callable so it is testable without a GPU
- **Lazy heavy imports:** torch is imported only inside functions that use CUDA, allowing unit tests to run on CPU-only CI

---

*Convention analysis: 2026-08-21*