"""
Text utilities for cleaning and processing Confluence content.
"""

import re


def clean_whitespace(text: str) -> str:
    """Collapse multiple blank lines and strip trailing whitespace."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to max_length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(" ", 1)[0] + "..."


def extract_plain_text(text: str) -> str:
    """Remove any remaining HTML tags from text."""
    clean = re.sub(r"<[^>]+>", "", text)
    return clean_whitespace(clean)
