"""Markdown table helpers for notebook reporting."""


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    """Build a markdown table string from row dicts and column definitions."""
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, divider]
    for row in rows:
        values = [str(row[key]) for key, _ in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)