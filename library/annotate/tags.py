"""
library/annotate/tags.py - Tag building utilities for annotation.

Provides functions for constructing XML-style semantic tags for the
annotated output format.
"""

from html import escape


def escape_xml(text: str) -> str:
    """Escape text for safe inclusion in XML content."""
    return escape(text, quote=False)


def build_endmatter_tag(content: str, endmatter_type: str) -> str:
    """
    Build an endmatter tag for frontmatter/backmatter pages.

    Args:
        content: The page text content.
        endmatter_type: One of TOC_INDEX, BIBLIO, OTHERENDMATTER.

    Returns:
        Tagged string like: <idi-endmatter type="TOC_INDEX">content</idi-endmatter>
    """
    return f'<idi-endmatter type="{endmatter_type}">{escape_xml(content)}</idi-endmatter>'


def build_paragraph_tag(text: str, perplexity: float | None = None) -> str:
    """
    Build a paragraph tag.

    Args:
        text: The paragraph text content.
        perplexity: Optional perplexity score for the paragraph.

    Returns:
        Tagged string like: <idi-paragraph perplexity="8.7">text</idi-paragraph>
    """
    if perplexity is not None:
        return f'<idi-paragraph perplexity="{perplexity:.1f}">{escape_xml(text)}</idi-paragraph>'
    return f"<idi-paragraph>{escape_xml(text)}</idi-paragraph>"


def build_section_tag(content: str, perplexity: float | None = None) -> str:
    """
    Build a section tag wrapping paragraph content.

    Args:
        content: Already-tagged paragraph content (not escaped again).
        perplexity: Optional mean perplexity of enclosed paragraphs.

    Returns:
        Tagged string like: <idi-section perplexity="10.5">...</idi-section>
    """
    if perplexity is not None:
        return f'<idi-section perplexity="{perplexity:.1f}">\n{content}\n</idi-section>'
    return f"<idi-section>\n{content}\n</idi-section>"


def build_duplicate_tag(content: str, cluster: str) -> str:
    """
    Build a duplicate wrapper tag.

    Args:
        content: Already-tagged paragraph content.
        cluster: Cluster identifier like "BARCODE:p:N" or "BARCODE:p:N-M".

    Returns:
        Tagged string like: <idi-duplicate cluster="...">...</idi-duplicate>
    """
    return f'<idi-duplicate cluster="{cluster}">\n{content}\n</idi-duplicate>'


def build_representative_tag(content: str, cluster: str) -> str:
    """
    Build a representative wrapper tag.

    Args:
        content: Already-tagged paragraph content.
        cluster: Cluster identifier like "BARCODE:p:N".

    Returns:
        Tagged string like: <idi-representative cluster="...">...</idi-representative>
    """
    return f'<idi-representative cluster="{cluster}">\n{content}\n</idi-representative>'
