"""Unit tests for the §9.2 data-quality rules (tinytoolcaller.data)."""

from tinytoolcaller.data import clean_subset, validate_example


def test_valid_example_passes(weather_example):
    ok, reason = validate_example(weather_example)
    assert ok and reason == "ok"


def test_missing_query_rejected(weather_example):
    ex = {**weather_example, "query": ""}
    ok, reason = validate_example(ex)
    assert not ok and reason == "missing_or_empty_query"


def test_query_not_string_rejected(weather_example):
    ex = {**weather_example, "query": None}
    ok, _ = validate_example(ex)
    assert not ok


def test_tools_not_list_rejected(weather_example):
    ex = {**weather_example, "tools": {"name": "get_weather"}}
    ok, reason = validate_example(ex)
    assert not ok and reason == "tools_not_a_list"


def test_malformed_tool_rejected(weather_example):
    ex = {**weather_example, "tools": [{"description": "no name"}]}
    ok, reason = validate_example(ex)
    assert not ok and reason == "malformed_tool_entry"


def test_missing_answers_rejected(weather_example):
    ex = {**weather_example, "answers": None}
    ok, reason = validate_example(ex)
    assert not ok and reason == "missing_answers"


def test_clean_subset_counts_and_dedups(weather_example):
    bad_query = {**weather_example, "query": ""}
    dup = {**weather_example}  # identical to weather_example
    rows = [weather_example, bad_query, dup]
    kept, stats = clean_subset(rows)
    assert len(kept) == 1
    assert stats["total"] == 3
    assert stats["kept"] == 1
    assert stats["dropped_by_reason"]["missing_or_empty_query"] == 1
    assert stats["dropped_by_reason"]["exact_duplicate"] == 1


def test_clean_subset_does_not_alter_valid_rows(weather_example):
    kept, _ = clean_subset([weather_example])
    assert kept[0] == weather_example  # no relabelling / filtering
