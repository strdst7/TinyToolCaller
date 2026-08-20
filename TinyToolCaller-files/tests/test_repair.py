"""Unit tests for the one-shot repair loop (tinytoolcaller.repair)."""

from tinytoolcaller.repair import repair


def _generator(responses):
    """A scripted generator that returns the next canned response."""
    calls = {"n": 0}

    def gen(prompt):
        idx = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        return responses[idx]

    return gen


def test_valid_output_not_retried():
    raw = '{"name": "get_weather", "arguments": {"location": "Tokyo"}}'
    gen = _generator(["SHOULD NOT BE CALLED"])
    out, attempts = repair(raw, gen, prompt="p", max_attempts=1)
    assert out == raw
    assert attempts == 0


def test_invalid_output_repaired_in_one_attempt():
    bad = "Sure, here you go!"
    fixed = '{"name": "x", "arguments": {}}'
    gen = _generator([fixed])
    out, attempts = repair(bad, gen, prompt="p", max_attempts=1)
    assert extract(out) is not None
    assert attempts == 1


def test_unrepairable_exhausts_attempts():
    gen = _generator(["still not json", "also not json"])
    out, attempts = repair("not json", gen, prompt="p", max_attempts=2)
    assert attempts == 2
    assert extract(out) is None


def test_repair_prompt_contains_previous_output():
    captured = {}

    def gen(prompt):
        captured["prompt"] = prompt
        return '{"name": "ok", "arguments": {}}'

    repair("ORIGINAL GARBAGE", gen, prompt="base-prompt", max_attempts=1)
    assert "ORIGINAL GARBAGE" in captured["prompt"]
    assert "not valid JSON" in captured["prompt"]


def extract(raw):
    from tinytoolcaller.formatting import extract_json

    return extract_json(raw)
