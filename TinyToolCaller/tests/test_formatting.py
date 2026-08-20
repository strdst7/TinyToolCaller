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
    ('Sure! Here you go:\n{"name":"y","arguments":{"a":1}} hope that helps',
     {"name": "y", "arguments": {"a": 1}}),
    ('no json here at all', None),
    ('nested {"a": {"b": [1,2,{"c":3}]}} tail',
     {"a": {"b": [1, 2, {"c": 3}]}}),
    ('', None),
    (None, None),
])
def test_extract_json(raw, expected):
    assert extract_json(raw) == expected


def test_extract_json_rejects_bare_list():
    # A JSON list is not a valid tool call object.
    assert extract_json("[1, 2, 3]") is None


# --------------------------------------------------------------------------- #
# GSM8K answer extraction — shared harness for both models (§20)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,expected", [
    ("The answer is 42.", "42"),
    ("#### 1,234", "1234"),
    ("Therefore the total is 7 apples", "7"),
    ("#### -12.5", "-12.5"),
])
def test_extract_gsm8k_answer(raw, expected):
    assert extract_gsm8k_answer(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("1234", "1234"),
    ("12.0", "12"),
    ("1,234", "1234"),
    ("12.50", "12.5"),
])
def test_normalise_number(raw, expected):
    assert normalise_number(raw) == expected


# --------------------------------------------------------------------------- #
# ChatML construction (§9)
# --------------------------------------------------------------------------- #
def test_build_messages_structure(weather_example):
    msgs = build_messages(weather_example)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "Available Tools:" in msgs[1]["content"]
    assert "get_weather" in msgs[1]["content"]
    assert "User Request:" in msgs[1]["content"]
    assert "Tokyo" in msgs[1]["content"]


def test_ground_truth_list_and_dict():
    ex_list = {"answers": [{"name": "a", "arguments": {}}]}
    ex_dict = {"answers": {"name": "b", "arguments": {}}}
    assert ground_truth(ex_list) == {"name": "a", "arguments": {}}
    assert ground_truth(ex_dict) == {"name": "b", "arguments": {}}


def test_format_for_training_includes_assistant_turn(weather_example, fake_tokenizer):
    out = format_for_training(weather_example, fake_tokenizer)
    assert "[system]" in out["text"]
    assert "[user]" in out["text"]
    assert "[assistant]" in out["text"]
    assert "get_weather" in out["text"]


def test_format_for_inference_has_generation_prompt(weather_example, fake_tokenizer):
    out = format_for_inference(weather_example, fake_tokenizer)
    assert "[assistant]" in out
    assert "[assistant]get_weather" not in out  # no ground-truth leak
