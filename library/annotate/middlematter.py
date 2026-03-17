"""
library/annotate/middlematter.py - middlematter annotation for main content

Handles annotation of

- Sections (groups of paragraphs)
- Paragraphs (with perplexity scores)
- Duplicate/representative markers (with cluster references)
"""

from const.types import BookJSON
from library.annotate.tags import (
    build_duplicate_tag,
    build_paragraph_tag,
    build_representative_tag,
    build_section_tag,
)


def get_paragraph_text(
    sentences: list[str],
    para_starts: list[int],
    para_idx: int,
) -> str:
    """
    Extract paragraph text from sentences given paragraph boundaries.

    Args:
        sentences: List of all sentences.
        para_starts: Paragraph start indices into sentences.
        para_idx: Index of the paragraph to extract.

    Returns:
        Concatenated paragraph text with sentences joined by spaces.
    """
    start = para_starts[para_idx]
    end = para_starts[para_idx + 1] if para_idx + 1 < len(para_starts) else len(sentences)
    return " ".join(sentences[start:end])


def get_section_para_range(
    section_idx: int,
    section_starts: list[int],
    para_starts: list[int],
) -> tuple[int, int]:
    """
    Get the paragraph index range for a section.

    Args:
        section_idx: Index of the section.
        section_starts: Section start indices (into paragraph indices).
        para_starts: Paragraph start indices.

    Returns:
        Tuple of (start_para_idx, end_para_idx) - end is exclusive.
    """
    start_para = section_starts[section_idx]
    if section_idx + 1 < len(section_starts):
        end_para = section_starts[section_idx + 1]
    else:
        end_para = len(para_starts)
    return start_para, end_para


def annotate_middlematter(
    book: BookJSON,
    perplexities: list[float] | None = None,
) -> str:
    """
    Annotate middlematter content with semantic tags.

    Args:
        book: Book dictionary with middlematter_sentences and indices.
        perplexities: Optional list of perplexity values per paragraph.

    Returns:
        Single annotated string containing all middlematter content.
    """
    sentences = book.get("middlematter_sentences", [])
    para_starts = book.get("subtopic_paragraph_start_indices", [])
    section_starts = book.get("subtopic_section_start_indices", [])
    book_id = book.get("barcode_src", "UNKNOWN")
    if book_id == "UNKNOWN":
        raise ValueError("Unkown book_id in middlematter annotation")

    # Get duplicate/representative info
    duplicate_paras = book.get("duplicate_paragraphs", {})
    representative_paras = book.get("representative_paragraphs", {})

    if not sentences or not para_starts:
        return ""

    # Default to one section containing all paragraphs
    if not section_starts:
        section_starts = [0]

    num_paras = len(para_starts)
    num_sections = len(section_starts)

    # Build annotated sections
    annotated_sections = []

    for section_idx in range(num_sections):
        start_para, end_para = get_section_para_range(section_idx, section_starts, para_starts)

        section_content = []
        section_perplexities = []
        para_idx = start_para

        while para_idx < end_para:
            para_text = get_paragraph_text(sentences, para_starts, para_idx)
            para_perp = (
                perplexities[para_idx] if perplexities and para_idx < len(perplexities) else None
            )

            # Skip invalid perplexities (-1 means too short)
            if para_perp is not None and para_perp > 0:
                section_perplexities.append(para_perp)

            str_idx = str(para_idx)

            if str_idx in representative_paras:
                # Representative paragraph - NOT merged with others
                para_tag = build_paragraph_tag(
                    para_text, para_perp if para_perp and para_perp > 0 else None
                )
                cluster = f"{book_id}:p:{para_idx}"
                section_content.append(build_representative_tag(para_tag, cluster))
                para_idx += 1

            elif str_idx in duplicate_paras:
                # Start of duplicate sequence - check for consecutive duplicates
                dup_start = para_idx
                dup_paras = []

                # Collect consecutive duplicate paragraphs within same section
                while para_idx < end_para and str(para_idx) in duplicate_paras:
                    d_text = get_paragraph_text(sentences, para_starts, para_idx)
                    d_perp = (
                        perplexities[para_idx]
                        if perplexities and para_idx < len(perplexities)
                        else None
                    )
                    d_perp_valid = d_perp if d_perp and d_perp > 0 else None
                    dup_paras.append(build_paragraph_tag(d_text, d_perp_valid))

                    # Track perplexity for section mean
                    if d_perp is not None and d_perp > 0:
                        section_perplexities.append(d_perp)

                    para_idx += 1

                dup_end = para_idx - 1

                # Build cluster reference
                if dup_start == dup_end:
                    cluster = f"{book_id}:p:{dup_start}"
                else:
                    cluster = f"{book_id}:p:{dup_start}-{dup_end}"

                dup_content = "\n".join(dup_paras)
                section_content.append(build_duplicate_tag(dup_content, cluster))

            else:
                # Regular paragraph
                para_perp_valid = para_perp if para_perp and para_perp > 0 else None
                section_content.append(build_paragraph_tag(para_text, para_perp_valid))
                para_idx += 1

        # Compute section mean perplexity
        section_perp = None
        if section_perplexities:
            section_perp = sum(section_perplexities) / len(section_perplexities)

        # Build section tag
        section_inner = "\n".join(section_content)
        annotated_sections.append(build_section_tag(section_inner, section_perp))

    return "\n".join(annotated_sections)
