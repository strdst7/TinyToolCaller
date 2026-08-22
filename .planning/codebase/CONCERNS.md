# Codebase Concerns

**Analysis Date:** 2026-08-21

## Tech Debt

### Orphaned duplicate codebase (`TinyToolCaller-files/`)

- **Issue:** The repository root contains two near-identical copies of the same package: `TinyToolCaller/` (the active, developed copy) and `TinyToolCaller-files/` (an older snapshot that diverged). Several files exist only in `TinyToolCaller-files/` (e.g., `tests/test_data.py`, `tests/test_repair.py`) and the main `TinyToolCaller/` copy does NOT have these tests. Some source files differ in content as well.
- **Files:**
  - `TinyToolCaller/` — active codebase
  - `TinyToolCaller-files/` — stale duplicate
  - `TinyToolCaller-files/tests/test_data.py` — contains tests for `data.validate_example()` and `data.clean_subset()` that would fail in `TinyToolCaller/` because those functions do not exist there
  - `TinyToolCaller-files/tests/test_repair.py` — contains tests for `repair.repair()` that would pass in `TinyToolCaller/`
  - `TinyToolCaller-files/tinytoolcaller/data.py` — has `validate_example()` and `clean_subset()` (lines 47-114), which the active copy lacks
- **Impact:** Confusion about which copy is authoritative. Extra tests in the `-files` copy represent untapped test coverage invisible from the main codebase. Developers may accidentally edit the wrong copy.
- **Fix approach:** Delete `TinyToolCaller-files/` from the repository. Merge its missing tests (`test_data.py`, `test_repair.py`) into the active `TinyToolCaller/tests/` directory. Port the missing data-quality functions (`validate_example`, `clean_subset`) from the `-files` copy into `TinyToolCaller/tinytoolcaller/data.py`.

### Stub-only data-quality functions in the active copy

- **Issue:** The active `TinyToolCaller/tinytoolcaller/data.py` provides `load_tokenizer()`, `load_source_dataset()`, and `sample_and_split()` — but it does NOT have the `validate_example()` or `clean_subset()` data-quality functions that exist in the orphaned `TinyToolCaller-files/tinytoolcaller/data.py`. These implement the documented §9.2 data-quality rules (validating query/tools/answers fields, deduplication) and have corresponding tests. The active codebase skips this quality gate entirely.
- **Files:**
  - `TinyToolCaller/tinytoolcaller/data.py` — missing `validate_example()` and `clean_subset()`
  - `TinyToolCaller-files/tinytoolcaller/data.py` — has implementations (lines 47-114)
- **Impact:** Training data is not validated or deduplicated before training. Malformed or duplicate examples can silently enter the fine-tuning pipeline, degrading model quality.
- **Fix approach:** Port `validate_example()` and `clean_subset()` (with `_example_key()` helper) from `TinyToolCaller-files/tinytoolcaller/data.py` into the active `TinyToolCaller/tinytoolcaller/data.py`.

### Missing tests in active copy

- **Issue:** The active `TinyToolCaller/tests/` directory has 4 test files (31 tests based on the README claim of 41). The `TinyToolCaller-files/tests/` has 6 test files. Tests for `data.py` and `repair.py` are absent from the active copy.
- **Files:**
  - Missing: `TinyToolCaller/tests/test_data.py` (exists in `TinyToolCaller-files/tests/test_data.py`)
  - Missing: `TinyToolCaller/tests/test_repair.py` (exists in `TinyToolCaller-files/tests/test_repair.py`)
- **Impact:** Approximately 24 fewer tests running in CI. The repair loop has zero test coverage in the active copy. The `data.py` quality functions have zero coverage.
- **Fix approach:** Copy `test_repair.py` to the active tests directory (it passes as-is against `TinyToolCaller/tinytoolcaller/repair.py`). Copy and adapt `test_data.py` once the functions it tests are ported.

### Unpinned dependency versions

- **Issue:** `TinyToolCaller/requirements.txt` pins NO versions — every dependency is listed bare (`torch`, `transformers`, `datasets`, `peft`, `trl`, `accelerate`, `bitsandbytes`, `wandb`, `huggingface_hub`, `numpy`, `scipy`, `pandas`, `jsonschema`, `pytest`). This means builds are non-reproducible. The README itself warns that the `datasets` shuffle RNG is version-sensitive ("record the version"), but no version is enforced.
- **Files:** `TinyToolCaller/requirements.txt`
- **Impact:** The documented seed-42 subset membership will differ across `datasets` versions, breaking the train/val split reproducibility claimed in the publication. Training results (loss curves, metrics) may not reproduce across library versions.
- **Fix approach:** Pin all direct dependencies with `>=` lower bounds and add a `requirements-lock.txt` with exact versions. At minimum pin `datasets`, `transformers`, `torch`, `trl`, and `peft`.

### Weak `.gitignore` coverage

- **Issue:** The `.gitignore` at `TinyToolCaller/.gitignore` is incomplete:
  - `*.pyc` not covered (only `*.py[cod]`)
  - `.pytest_cache/` not covered (exists and untracked in root)
  - `.DS_Store` not covered
  - IDE files (`.idea/`, `.vscode/`) not covered
  - `data/` directory (output of `scripts/publish_dataset.py`) not covered
  - `environment.json` (output of `scripts/capture_environment.py --save`) not covered
- **Files:** `TinyToolCaller/.gitignore`
- **Impact:** Generated artifacts can accidentally be committed.
- **Fix approach:** Extend `.gitignore` to include `.pytest_cache/`, `.DS_Store`, `data/`, `environment.json`, `*.pyc`, IDE directories.

## Known Bugs

### No inline annotations for known issues

- **Issue:** A systematic grep for `BUG`, `FIXME`, `HACK`, `XXX`, `WORKAROUND`, `TEMP` across all Python files returned zero results. No known bugs are annotated in the codebase.
- **Files:** All `.py` files under `TinyToolCaller/`
- **Impact:** Code contains no markers for known issues, making it difficult for new contributors to identify problem areas.
- **Workaround:** N/A — this is a process gap rather than a specific bug.

## Security Considerations

### Bare `except` clauses with blanket suppression

- **Issue:** Multiple locations use bare `except Exception` (aliased `BLE001`) that catch and silently suppress errors:
  - `TinyToolCaller/tinytoolcaller/data.py`, line 27: `except Exception as exc:` — catches all errors during dataset loading, including KeyboardInterrupt
  - `TinyToolCaller/tinytoolcaller/metrics.py`, line 104: `except Exception:` — fallback fallback for gated GSM8K split
  - `TinyToolCaller/tinytoolcaller/train.py`, line 95: `except Exception as exc:` — suppresses Hub publication errors
  - `TinyToolCaller/scripts/profile_tool_distribution.py`, line 50: `except Exception:` — falls back silently if scipy missing
  - `TinyToolCaller/scripts/profile_tool_distribution.py`, line 305: `except Exception:` — falls back if datasets missing
- **Files:**
  - `TinyToolCaller/tinytoolcaller/data.py:27`
  - `TinyToolCaller/tinytoolcaller/metrics.py:104`
  - `TinyToolCaller/tinytoolcaller/train.py:95`
  - `TinyToolCaller/scripts/profile_tool_distribution.py:50,305`
- **Impact:** KeyboardInterrupt or SystemExit during these blocks would be caught and silently handled, causing confusing hangs. Hub publication failures are logged but never raised, leaving the user with no exit code to detect failure.
- **Recommendations:** Change `except Exception` to `except Exception` (still broad but explicit) and log/re-raise critical errors. For `train.py:95`, consider re-raising after logging so the process exits with a non-zero code.

### HF_TOKEN env var warning only

- **Issue:** `TinyToolCaller/train_tool_caller.py` (line 91-93) checks for `HF_TOKEN` but only prints a warning if missing. The actual failure occurs later with a different error message from the dataset loader.
- **Files:** `TinyToolCaller/train_tool_caller.py:91-93`
- **Impact:** User confusion — the warning is printed but execution continues, then fails at the dataset loading step with a different error. The validation is partially redundant with `load_source_dataset()` which has its own error message.
- **Recommendations:** Either make the `HF_TOKEN` check an early exit (raise `SystemExit` with instructions) or remove it entirely since `load_source_dataset()` already handles the failure with an actionable message.

## Performance Bottlenecks

### Sequential per-example generation in evaluation

- **Issue:** `TinyToolCaller/tinytoolcaller/metrics.py` function `evaluate_tool_calling()` (line 67) iterates sequentially over the validation set with `for example in val_ds`, calling `generate()` for each. The generation itself is not batched. With `n_val=200` this is acceptable, but for larger evaluations it would be slow.
- **Files:** `TinyToolCaller/tinytoolcaller/metrics.py:77`
- **Cause:** Each example requires a full forward pass through the model — no token-level batching across examples.
- **Improvement path:** If the evaluation set grows, add a batched generation path using `tokenizer()` with padding and `model.generate()` with batched inputs.

### Deterministic greedy generation always used

- **Issue:** `TinyToolCaller/tinytoolcaller/metrics.py` function `generate()` (line 50) always uses `do_sample=False` (greedy). This is correct for reproducible evaluation, but means there is no way to get a diversity of outputs or temperature-based sampling for qualitative analysis.
- **Files:** `TinyToolCaller/tinytoolcaller/metrics.py:50-61`
- **Impact:** No mechanism to probe model confidence or explore alternate outputs for ambiguous tool calls.
- **Improvement path:** Add a `do_sample` parameter to `generate()` (defaulting to `False` for backward compatibility).

## Fragile Areas

### TRL API version divergence handling

- **Issue:** `TinyToolCaller/tinytoolcaller/train.py` (lines 41-56) has a `try/except ImportError` block that handles two different TRL APIs (pre-0.12 and 0.12+). The `SFTConfig` approach supplies `max_seq_length` to `SFTConfig`; the fallback passes it to `SFTTrainer` directly. Both branches also use `formatting_func` which accepts a list of dicts.
- **Files:** `TinyToolCaller/tinytoolcaller/train.py:41-56`
- **Why fragile:** This pattern relies on import-failing to detect TRL version. If `SFTConfig` is importable but has a different signature, the code silently uses the wrong set of arguments. The try/except has no version logging to know which path was taken.
- **Test coverage:** None — neither branch is tested in CI.
- **Safe modification:** Add a version-print after the path resolves (e.g., `print(f"Using TRL {trl.__version__} with {'SFTConfig' if use_sftconfig else 'TrainingArguments'}")`). Add a regression test that mocks both import paths.

### Dataset shuffle reproducibility depends on `datasets` version

- **Issue:** `TinyToolCaller/tinytoolcaller/data.py` function `sample_and_split()` uses `ds.shuffle(seed=seed).select(range(n_sample))`. The README (line 20 of `data.py`) notes: "the shuffle RNG is datasets-version-sensitive". With unpinned `datasets` version, the exact same seed produces different subsets across versions.
- **Files:** `TinyToolCaller/tinytoolcaller/data.py:43-45`
- **Why fragile:** Any re-running of the pipeline on a different `datasets` version will produce a different train/val split, invalidating comparisons to published results.
- **Test coverage:** Only tests that config values match (`test_config.py`), not that the actual split contents are deterministic.
- **Safe modification:** Pin `datasets` version. Alternatively, save the sampled subset as Parquet once and re-load it instead of re-shuffling.

### GSM8K test split fallback

- **Issue:** `TinyToolCaller/tinytoolcaller/metrics.py` lines 102-106 catch any exception when loading the GSM8K test split and silently fall back to the train split. This means GSM8K evaluation may be training on the test data if the test split is unavailable, invalidating the metric.
- **Files:** `TinyToolCaller/tinytoolcaller/metrics.py:102-106`
- **Why fragile:** A silent fallback from "test" to "train" means numbers reported as "GSM8K retention" might include data leakage. The comment says "gated; falls back to train" in `config.py` (line 48), but this is a design weakness.
- **Test coverage:** None.
- **Safe modification:** Log a clear warning when the fallback occurs, and consider exiting with an error instead of silently using the wrong split.

### `_dump_eval` assumes equal-length paired details

- **Issue:** `TinyToolCaller/train_tool_caller.py` function `_dump_eval()` (line 51) uses `zip(base_details, ft_details)` without asserting the lists have equal length. If base evaluation and fine-tuned evaluation produce different numbers of predictions (e.g., due to different validation splits or errors), the ZIP silently truncates to the shorter list.
- **Files:** `TinyToolCaller/train_tool_caller.py:51-64`
- **Why fragile:** Silent data loss. The statistical analysis that depends on this file (`scripts/statistical_analysis.py --mcnemar`) would produce incorrect results from mismatched data.
- **Safe modification:** Add `assert len(base_details) == len(ft_details)` before the zip, or use `zip(..., strict=True)` (Python 3.10+).

## Scaling Limits

### Evaluation set size (n=200)

- **Issue:** The validation split is only 200 examples. The README itself acknowledges this caveat: "The results come from the same 200 examples used during development." A separate held-out test set is listed as the next step.
- **Files:** `TinyToolCaller/tinytoolcaller/config.py:9` (`n_val: 200`)
- **Current capacity:** 200 examples.
- **Limit:** Statistical power is low — a difference of a few correct/wrong examples can shift percentages by multiple points.
- **Scaling path:** Reserve a separate held-out test set (e.g., 1,000 examples) not used during development. Update the pipeline to evaluate against it.

### GSM8K probe size (n=50)

- **Issue:** The GSM8K retention probe uses only 50 examples (`gsm8k_n: 50` in `config.py:45`), giving extremely wide confidence intervals. The difference reported (52% → 50%) is well within noise.
- **Files:** `TinyToolCaller/tinytoolcaller/config.py:45`
- **Limit:** 95% Wilson CI width for n=50 at ~50% accuracy is approximately ±14 percentage points — too wide to detect any meaningful degradation.
- **Scaling path:** Increase to n=500+.

## Dependencies at Risk

### `bitsandbytes` — quantization dependency

- **Risk:** `bitsandbytes` is a core dependency for 4-bit QLoRA quantization. It has had compatibility issues across CUDA versions, PyTorch versions, and platforms. On macOS / MPS, it is unsupported, meaning the pipeline cannot run locally at all.
- **Impact:** Training and evaluation of the 4-bit quantized model fails entirely without a CUDA-capable GPU and matching `bitsandbytes` build.
- **Migration plan:** Consider `torchao` or native PyTorch quantization as an alternative for cross-platform support. However, for the documented methodology, `bitsandbytes` is intentionally used for reproducibility with the publication.

### No lockfile for reproducible environments

- **Risk:** Without pinned dependencies or a lockfile, any environment rebuild may silently produce a different software stack.
- **Impact:** Reproducibility of the published results is at risk.
- **Migration plan:** Generate a `requirements-lock.txt` with exact versions from a known-good run. Document the `pip freeze` output alongside the publication.

## Missing Critical Features

### No held-out test set

- **Problem:** The pipeline uses the same 200-example validation split for both development and reporting. There is no held-out test set. The README explicitly identifies this as the next step.
- **Files:** Entire evaluation pipeline via `TinyToolCaller/tinytoolcaller/metrics.py`
- **Blocks:** Credible claims about generalization to unseen data. All reported numbers are potentially overfitted to the 200-example split.

### No CI test runner for heavy-dependency tests

- **Problem:** The GitHub Actions workflow (`.github/workflows/python-package.yml`) runs `pytest` in the root, but `TinyToolCaller/` (not the root) is where the tests and package live. The CI would likely fail because the import paths assume execution from the `TinyToolCaller/` directory. Also, several modules require torch/trl/peft which are heavy dependencies not efficiently installed in CI.
- **Files:** `.github/workflows/python-package.yml`
- **Blocks:** Reliable CI signal for PR validation. Currently the CI workflow may run zero actual tests.

### No type hints enforcement or static analysis

- **Problem:** The code uses `from __future__ import annotations` in most files but does not run `mypy`, `pyright`, or any static type checker. There is no `py.typed` marker. The active codebase has no linting configuration file (no `.flake8`, `pyproject.toml`, `ruff.toml`, or `setup.cfg`).
- **Files:** All `.py` files in `TinyToolCaller/`
- **Blocks:** Type errors may silently exist. New contributors have no style guide to follow.

### No `conftest.py` `weather_example` fixture usable for repair tests

- **Problem:** The `test_repair.py` in `TinyToolCaller-files/` does not import `weather_example` fixture from `conftest.py`. It could benefit from using it for test data instead of constructing inline test cases. However, the existing inline tests are self-contained.

## Test Coverage Gaps

### `data.py` — zero test coverage

- **What's not tested:** `load_tokenizer()`, `load_source_dataset()`, `sample_and_split()` are entirely untested.
- **Files:** `TinyToolCaller/tinytoolcaller/data.py`
- **Risk:** Changes to these functions (which control the entire training data pipeline) could silently change the data composition, invalidating published results.
- **Priority:** High

### `repair.py` — zero test coverage in active copy

- **What's not tested:** `repair()` function — the one-shot JSON repair loop (publication §3.1). Tests exist in the orphaned `TinyToolCaller-files/` copy.
- **Files:** `TinyToolCaller/tinytoolcaller/repair.py`
- **Risk:** Changes to the repair logic could break the retry-with-repair mechanism without detection.
- **Priority:** Medium

### `model.py` — zero test coverage

- **What's not tested:** `load_quantized_model()`, `attach_lora()` — these require GPU and heavy dependencies so unit testing is difficult, but no mock-based tests exist.
- **Files:** `TinyToolCaller/tinytoolcaller/model.py`
- **Risk:** Medium (mitigated by heavy deps requirement).
- **Priority:** Low

### `train.py` — zero test coverage

- **What's not tested:** `train()`, `save_and_publish()` — the SFTTrainer wrapper and merge/publish functions.
- **Files:** `TinyToolCaller/tinytoolcaller/train.py`
- **Risk:** The fragile TRL API version branching (lines 41-56) has no test coverage.
- **Priority:** Medium

### `scripts/` directory — zero test coverage

- **What's not tested:** All 6 scripts (`profile_tool_distribution.py`, `dataset_stats.py`, `statistical_analysis.py`, `capture_environment.py`, `publish_dataset.py`, `build_preprint.py`).
- **Files:** `TinyToolCaller/scripts/*.py`
- **Risk:** Scripts may break across dependency upgrades without detection.
- **Priority:** Low (scripts are run infrequently).

---

*Concerns audit: 2026-08-21*
