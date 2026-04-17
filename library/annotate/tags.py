"""
library/annotate/tags.py - Tag building utilities for annotation.

Provides functions for constructing HTML semantic tags for the
annotated output format.
"""

from html import escape


def escape_html(text: str) -> str:
    """Escape text for safe inclusion in HTML content."""
    return escape(text, quote=False)


def build_endmatter_page_tag(content: str, endmatter_type: str) -> str:
    """
    Build a div tag for a single endmatter page.

    Args:
        content: The page text content.
        endmatter_type: One of TOC_INDEX, BIBLIO, OTHERENDMATTER.

    Returns:
        Tagged string like: <div class="toc_index">content</div>
    """
    css_class = endmatter_type.lower()
    return f'<div class="{css_class}">{escape_html(content)}</div>'


def build_frontmatter_tag(content: str) -> str:
    """
    Wrap endmatter page divs in a header tag for frontmatter.

    Args:
        content: Already-tagged endmatter page content.

    Returns:
        Tagged string like: <header>...</header>
    """
    return f"<header>\n{content}\n</header>"


def build_backmatter_tag(content: str) -> str:
    """
    Wrap endmatter page divs in a footer tag for backmatter.

    Args:
        content: Already-tagged endmatter page content.

    Returns:
        Tagged string like: <footer>...</footer>
    """
    return f"<footer>\n{content}\n</footer>"


def build_paragraph_tag(
    text: str,
    perplexity: float | None = None,
    language: str | None = None,
    representative_cluster: str | None = None,
) -> str:
    """
    Build a paragraph tag.

    Args:
        text: The paragraph text content.
        perplexity: Optional perplexity score for the paragraph.
        language: Optional language code for the paragraph.
        representative_cluster: Optional cluster ID if this is a representative
            paragraph (e.g., "BARCODE:10"). Adds data-representative and
            data-clusterid attributes.

    Returns:
        Tagged string like: <p data-perplexity="8.7" data-language="en">text</p>
    """
    attrs = []
    if perplexity is not None:
        attrs.append(f'data-perplexity="{perplexity:.1f}"')
    if language is not None:
        attrs.append(f'data-language="{language}"')
    if representative_cluster is not None:
        attrs.append("data-representative")
        attrs.append(f'data-clusterid="{representative_cluster}"')

    if attrs:
        attr_str = " " + " ".join(attrs)
        return f"<p{attr_str}>{escape_html(text)}</p>"
    return f"<p>{escape_html(text)}</p>"


def build_section_tag(content: str, perplexity: float | None = None) -> str:
    """
    Build a section tag wrapping paragraph content.

    Args:
        content: Already-tagged paragraph content (not escaped again).
        perplexity: Optional mean perplexity of enclosed paragraphs.

    Returns:
        Tagged string like: <section data-perplexity="10.5">...</section>
    """
    if perplexity is not None:
        return f'<section data-perplexity="{perplexity:.1f}">\n{content}\n</section>'
    return f"<section>\n{content}\n</section>"


def build_duplicate_tag(content: str, cluster: str) -> str:
    """
    Build a duplicate wrapper tag.

    Args:
        content: Already-tagged paragraph content.
        cluster: Cluster identifier like "BARCODE:N" or "BARCODE:N-M".

    Returns:
        Tagged string like: <aside data-cluster="...">...</aside>
    """
    return f'<aside data-cluster="{cluster}">\n{content}\n</aside>'
