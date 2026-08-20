"""Unit tests for ToolCallingMetrics (pure part of tinytoolcaller.metrics)."""

from tinytoolcaller.metrics import ToolCallingMetrics


def test_empty_metrics_are_zero():
    m = ToolCallingMetrics()
    assert m.json_validity == 0.0
    assert m.tool_accuracy == 0.0
    assert m.argument_exact_match == 0.0


def test_percentages():
    m = ToolCallingMetrics(n=200, json_valid=196, tool_correct=185,
                           args_correct=168)
    assert abs(m.json_validity - 98.0) < 1e-9
    assert abs(m.tool_accuracy - 92.5) < 1e-9
    assert abs(m.argument_exact_match - 84.0) < 1e-9


def test_zero_denominator_safe():
    m = ToolCallingMetrics(n=0, json_valid=0)
    assert m.json_validity == 0.0


def test_as_table_contains_counts():
    m = ToolCallingMetrics(n=200, json_valid=196, tool_correct=185,
                           args_correct=168)
    table = m.as_table()
    assert "98.0%" in table and "(196/200)" in table
