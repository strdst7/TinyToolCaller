# Testing Patterns

**Analysis Date:** 2026-08-21

## Test Framework

**Runner:**
- **pytest** (listed in `requirements.txt`)
- Config: no `pytest.ini`, `pyproject.toml`, or `setup.cfg` detected — runs with defaults
- Cache present at `TinyToolCaller/.pytest_cache/`

**Assertion Library:**
- Standard `assert` statements (pytest native assertion rewriting)

**Run Commands:**
```bash
pytest                            # Run all tests (from repo root)
pytest -v                         # Verbose mode
pytest tests/                     # From within TinyToolCaller/ subdirectory
pytest tests/test_formatting.py   # Single test file
pytest -k "extract"               # Filter by test name
```

**CI Integration:**
- GitHub Actions workflow at `.github/workflows/python-package.yml` runs `pytest` on Python 3.9, 3.10, and 3.11
- Also runs `flake8` linting in the same CI step

## Test File Organization

**Location:**
- Tests are located in a **separate `tests/` directory** at `TinyToolCaller/tests/` — NOT co-located with source files
- Source code is at `TinyToolCaller/tinytoolcaller/`; tests are one directory level up

**Naming:**
- Test files mirror source module names: `test_config.py` tests `config.py`, `test_formatting.py` tests `formatting.py`, `test_metrics.py` tests `metrics.py`
- A second copy of the test suite exists at `TinyToolCaller-files/tests/` with two additional test files: `test_data.py` and `test_repair.py`

**Structure:**
```
TinyToolCaller/
├── tinytoolcaller/
│   ├── config.py
│   ├── formatting.py
│   ├── data.py
│   ├── model.py
│   ├── metrics.py
│   ├── train.py
│   └── repair.py
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_formatting.py
│   └── test_metrics.py
└── scripts/
    ├── statistical_analysis.py
    ├── build_preprint.py
    └── ... (no test files for scripts)
```

**Current test count:** 3 test files with ~30 test functions across `TinyToolCaller/tests/`, plus 2 more test files (~10 tests) in `TinyToolCaller-files/tests/`. The README badge claims "41 tests passed" — suggesting additional tests exist across the full tree.

**Untested modules:** `data.py`, `model.py`, `train.py`, `repair.py` (no test files in `TinyToolCaller/tests/`). Test coverage for `data.py` and `repair.py` exists only in `TinyToolCaller-files/tests/`.

## Test Structure

**Suite Organization:**
```python
"""Unit tests for the pure prompt/JSON/answer helpers (tinytoolcaller.formatting)."""

import pytest

from tinytoolcaller.formatting import (
    build_messages,
    extract_gsm8k_answer,
    extract_json,
    format_for_inference,
    format_for_training,
    ground_truth,
    normalise_number,
)


# --------------------------------------------------------------------------- #
# extract_json — the documented §14 / §21.7 parsing behaviour
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,expected", [
    ('{"name":"get_weather","arguments":{"location":"Tokyo"}}',
     {"name": "get_weather", "arguments": {"location": "Tokyo"}}),
    ('```json\n{"name":"x","arguments":{}}\n```',
     {"name": "x", "arguments": {}}),
    ('', None),
    (None, None),
])
def test_extract_json(raw, expected):
    assert extract_json(raw) == expected
```

**Patterns:**
- **`conftest.py`** at `tests/conftest.py` provides shared fixtures (`fake_tokenizer`, `weather_example`)
- Tests use `@pytest.mark.parametrize` extensively for data-driven testing
- Section comments (`# ---- ChatML construction (§9) ----`) separate test groups
- Plain `assert` statements — no `self.assertEqual` or `pytest.assert` wrappers
- No `unittest.TestCase` usage anywhere; tests are pure pytest functions

## Fixtures and Factories

**conftest.py** (`TinyToolCaller/tests/conftest.py`):
```python
"""Shared fixtures for the TinyToolCaller test suite."""

import sys
from pathlib import Path

import pytest

# Make the package importable when running `pytest` from the repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeTokenizer:
    """Minimal stand-in for a chat-template tokenizer (pure Python)."""

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False):
        parts = []
        for m in messages:
            parts.append(f"[{m['role']}]{m['content']}[/{m['role']}]")
        if add_generation_prompt:
            parts.append("[assistant]")
        return "".join(parts)


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


@pytest.fixture
def weather_example():
    return {
        "query": "What's the weather in Tokyo?",
        "tools": [{
            "name": "get_weather",
            ...
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }],
        "answers": [{"name": "get_weather",
                     "arguments": {"location": "Tokyo", "unit": "celsius"}}],
    }
```

**Test Data:**
- Fixtures live in `tests/conftest.py` — shared across all test files
- `fake_tokenizer` — a pure-Python stand-in that produces a `[role]content[/role]` format, avoids needing to load an actual Hugging Face tokenizer
- `weather_example` — a canonical example dict matching the xLAM dataset schema, used across `test_formatting.py` and `test_data.py`
- `_generator(responses)` closure at `test_repair.py` creates a scripted generator for testing the repair loop without a real model

## Mocking

**Framework:** No mocking framework used (no `unittest.mock` or `pytest-mock`). The codebase uses **dependency injection** instead:

**Patterns:**
```python
# repair.py — generate_fn is injected
def repair(raw: str, generate_fn, prompt: str, max_attempts: int = 1):
    ...

# test_repair.py — a scripted generator replaces the real model
def _generator(responses):
    """A scripted generator that returns the next canned response."""
    calls = {"n": 0}

    def gen(prompt):
        idx = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        return responses[idx]

    return gen

def test_invalid_output_repaired_in_one_attempt():
    bad = "Sure, here you go!"
    fixed = '{"name": "x", "arguments": {}}'
    gen = _generator([fixed])
    out, attempts = repair(bad, gen, prompt="p", max_attempts=1)
    assert extract(out) is not None
    assert attempts == 1
```

**What to Mock:**
- Stateless pure functions are tested directly (no mocking needed)
- Model/tokenizer dependencies are replaced with test doubles:
  - `FakeTokenizer` for tokenizer-dependent formatting functions
  - `_generator` for model generation in repair tests
- Heavy dependencies (torch, trl, peft) are imported lazily so they are never loaded during unit tests

**What NOT to Mock:**
- No `unittest.mock.MagicMock` or `patch` decorators anywhere in the codebase
- Integration tests for `evaluate_tool_calling` and `evaluate_gsm8k` require real GPU hardware and are not mocked

**Note on `TinyToolCaller-files/tests/`:**
- `test_repair.py` imports `from tinytoolcaller.repair import repair` directly (no conftest usage)
- `test_data.py` imports `from tinytoolcaller.data import clean_subset, validate_example` — but these functions do NOT exist in `TinyToolCaller/tinytoolcaller/data.py`, suggesting `TinyToolCaller-files/` represents a different development branch

## Fixtures and Factories

**Test Data:**
```python
# Shared example from conftest.py — used across test files
@pytest.fixture
def weather_example():
    return {
        "query": "What's the weather in Tokyo?",
        "tools": [{"name": "get_weather", ...}],
        "answers": [{"name": "get_weather",
                     "arguments": {"location": "Tokyo", "unit": "celsius"}}],
    }
```

**Location:**
- All shared fixtures are in `tests/conftest.py`
- Inline test data defined within test functions for single-use cases (e.g., `_generator` closure, parametrize tuples)

## Coverage

**Requirements:** None enforced. No coverage config file, no `--cov` flag in CI, no coverage badge.

**View Coverage:**
```bash
pip install pytest-cov
pytest --cov=tinytoolcaller tests/
```

## Test Types

**Unit Tests:**
- The entire test suite is unit tests — no integration or E2E tests found
- Scope: pure functions only (`formatting.py`, `config.py`, `metrics.py` dataclass, `repair.py`)
- Modules with GPU dependencies (`model.py`, `train.py`, `data.py` [in the main branch], `metrics.py` generation functions) have no tests in `TinyToolCaller/tests/`

**Integration Tests:**
- None detected. The `train_tool_caller.py` script serves as an integration test (runs the full pipeline), but is NOT part of the pytest suite

**E2E Tests:**
- Not used

## Common Patterns

**Parameterized Testing Pattern:**
```python
@pytest.mark.parametrize("raw,expected", [
    ('{"name":"get_weather","arguments":{"location":"Tokyo"}}',
     {"name": "get_weather", "arguments": {"location": "Tokyo"}}),
    ('```json\n{"name":"x","arguments":{}}\n```',
     {"name": "x", "arguments": {}}),
    ('no json here at all', None),
    ('', None),
    (None, None),
])
def test_extract_json(raw, expected):
    assert extract_json(raw) == expected
```

**Config Invariant Pattern:**
```python
def test_documented_seed_and_splits():
    assert CONFIG["seed"] == 42
    assert CONFIG["n_train"] == 5000
    assert CONFIG["n_val"] == 200
    assert CONFIG["n_sample"] == CONFIG["n_train"] + CONFIG["n_val"]
```

**Indexed Lookup + Slice Pattern for List/Array:**
```python
def test_as_table_contains_counts():
    m = ToolCallingMetrics(n=200, json_valid=196, tool_correct=185,
                           args_correct=168)
    table = m.as_table()
    assert "98.0%" in table and "(196/200)" in table
```

**Decimal comparison pattern:**
```python
assert abs(m.json_validity - 98.0) < 1e-9
```

**Edge Case Pattern:**
```python
def test_zero_denominator_safe():
    m = ToolCallingMetrics(n=0, json_valid=0)
    assert m.json_validity == 0.0


def test_extract_json_rejects_bare_list():
    # A JSON list is not a valid tool call object.
    assert extract_json("[1, 2, 3]") is None
```

**Assertion Patterns (from `TinyToolCaller-files/tests/`):**
```python
# Tuple unpacking validation
def test_valid_example_passes(weather_example):
    ok, reason = validate_example(weather_example)
    assert ok and reason == "ok"

# Dict mutation pattern for incremental test cases
def test_missing_query_rejected(weather_example):
    ex = {**weather_example, "query": ""}
    ok, reason = validate_example(ex)
    assert not ok and reason == "missing_or_empty_query"

# Scripted generator with capture dict
def test_repair_prompt_contains_previous_output():
    captured = {}
    def gen(prompt):
        captured["prompt"] = prompt
        return '{"name": "ok", "arguments": {}}'
    repair("ORIGINAL GARBAGE", gen, prompt="base-prompt", max_attempts=1)
    assert "ORIGINAL GARBAGE" in captured["prompt"]
    assert "not valid JSON" in captured["prompt"]
```

## Key Testing Insights

1. **Testable design enforced by architecture:** Pure functions (`formatting.py`, `config.py`) are isolated from heavy ML dependencies. Lazy imports ensure CI tests without GPU.
2. **No mocking library needed:** The design uses dependency injection (`generate_fn` parameter in `repair()`) and scripted closures rather than `unittest.mock`.
3. **`conftest.py` enables importability:** The `sys.path.insert(0, str(ROOT))` trick allows `pytest` to be invoked from the repo root or from `TinyToolCaller/` without package installation.
4. **Coverage gap in `TinyToolCaller/tests/`:** `data.py`, `model.py`, `train.py`, `repair.py` have no test coverage in the primary test directory. Coverage for `data.py` and `repair.py` exists only in `TinyToolCaller-files/tests/`.
5. **No script tests:** The 6 scripts in `scripts/` have no corresponding test files.
6. **CI matrix:** Tests run on Python 3.9, 3.10, 3.11 via GitHub Actions but NOT on 3.12/3.13.
7. **No `pyproject.toml` test configuration:** All test configuration is implicit pytest defaults.

---

*Testing analysis: 2026-08-21*