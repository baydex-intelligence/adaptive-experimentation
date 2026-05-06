"""Tests for markdown table rendering helpers."""

from adaptive_experiments.plotting.tables import markdown_table


def test_markdown_table_renders_expected_grid() -> None:
    rows = [
        {"name": "A", "score": 1.5},
        {"name": "B", "score": 2.0},
    ]
    columns = [("name", "Name"), ("score", "Score")]

    observed = markdown_table(rows, columns)
    expected = "\n".join(
        [
            "| Name | Score |",
            "|---|---|",
            "| A | 1.5 |",
            "| B | 2.0 |",
        ]
    )

    assert observed == expected
