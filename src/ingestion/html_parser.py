"""
Confluence XHTML → clean markdown/text parser.
Strips Confluence macros, scripts, styles, and converts to readable text.
"""

import re
from bs4 import BeautifulSoup, Comment

try:
    from markdownify import markdownify as md
except ImportError:
    md = None

from src.utils.text_utils import clean_whitespace


# Confluence-specific tags to remove entirely
TAGS_TO_REMOVE = [
    "script",
    "style",
    "ac:structured-macro",
    "ac:parameter",
    "ac:rich-text-body",
    "ac:image",
    "ac:emoticon",
    "ac:inline-comment-marker",
    "ac:task-list",
    "ac:task",
    "ac:task-body",
    "ac:task-status",
    "ri:attachment",
    "ri:url",
    "ri:page",
    "ri:user",
]


def parse_confluence_html(html: str) -> str:
    """
    Parse Confluence storage-format XHTML and return clean readable text.

    Steps:
    1. Parse with BeautifulSoup
    2. Remove Confluence macros, scripts, styles
    3. Remove HTML comments
    4. Convert to markdown (if markdownify available) or plain text
    5. Clean up whitespace
    """
    if not html or not html.strip():
        return ""

    soup = BeautifulSoup(html, "lxml")

    # Remove Confluence-specific macro tags
    for tag_name in TAGS_TO_REMOVE:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove HTML comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Remove empty divs/spans
    for tag in soup.find_all(["div", "span"]):
        if not tag.get_text(strip=True):
            tag.decompose()

    # Convert to markdown if markdownify is available, otherwise plain text
    if md:
        text = md(str(soup), heading_style="ATX", strip=["img"])
    else:
        text = soup.get_text(separator="\n")

    # Remove image references that markdownify might leave
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

    # Clean up excessive whitespace
    text = clean_whitespace(text)

    return text
