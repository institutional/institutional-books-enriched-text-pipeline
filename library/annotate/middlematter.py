"""
library/annotate/middlematter.py - middlematter annotation for main content

Handles annotation of

- Sections (groups of paragraphs)
- Paragraphs (with BPB scores)
- Duplicate/representative markers (with cluster references)
"""

from const.types import BookJSON
from library.annotate.language import detect_paragraph_languages
from library.annotate.tags import (
    build_duplicate_tag,
    build_paragraph_tag,
    build_section_tag,
)
from library.metadata.language_stats import compute_language_distribution


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
    bpb_values: list[float] | None = None,
) -> tuple[str, dict[str, list[str] | list[float]]]:
    """
    Annotate middlematter content with semantic tags.

    Args:
        book: Book dictionary with middlematter_sentences and indices.
        bpb_values: Optional list of bits-per-byte values per paragraph.

    Returns:
        Tuple of (annotated string, language distribution dict).
    """
    sentences = book.get("middlematter_sentences", [])
    para_starts = book.get("subtopic_paragraph_start_indices", [])
    section_starts = book.get("subtopic_section_start_indices", [])
    book_id = book.get("barcode_src", "UNKNOWN")
    if book_id == "UNKNOWN":
        raise ValueError("Unknown book_id in middlematter annotation")

    # Get duplicate/representative/removed info
    duplicate_paras = book.get("duplicate_paragraphs", {})
    representative_paras = book.get("representative_paragraphs", {})
    removed_paras = book.get("removed_paragraphs", {})

    if not sentences or not para_starts:
        return "", {"languages": [], "proportion": []}

    # Default to one section containing all paragraphs
    if not section_starts:
        section_starts = [0]

    num_sections = len(section_starts)

    # Extract all paragraph texts for language detection
    num_paras = len(para_starts)
    all_para_texts = [
        get_paragraph_text(sentences, para_starts, i) for i in range(num_paras)
    ]

    # Detect languages with propagation
    languages = detect_paragraph_languages(all_para_texts)

    # Build annotated sections
    annotated_sections = []

    for section_idx in range(num_sections):
        start_para, end_para = get_section_para_range(section_idx, section_starts, para_starts)

        section_content = []
        section_bpbs: list[float] = []
        para_idx = start_para

        while para_idx < end_para:
            para_text = get_paragraph_text(sentences, para_starts, para_idx)
            para_bpb = (
                bpb_values[para_idx] if bpb_values and para_idx < len(bpb_values) else None
            )

            if para_bpb is not None and para_bpb > 0:
                section_bpbs.append(para_bpb)

            str_idx = str(para_idx)

            if str_idx in removed_paras:
                para_idx += 1

            elif str_idx in representative_paras:
                para_lang = languages[para_idx] if para_idx < len(languages) else None
                cluster = f"{book_id}:{para_idx}"
                para_tag = build_paragraph_tag(
                    para_text,
                    para_bpb if para_bpb and para_bpb > 0 else None,
                    para_lang,
                    representative_cluster=cluster,
                )
                section_content.append(para_tag)
                para_idx += 1

            elif str_idx in duplicate_paras:
                dup_paras_tags = []
                start_dup = para_idx
                first_ref = duplicate_paras[str_idx]
                first_ref_book, first_ref_para = first_ref.rsplit(":", 1)
                ref_para = int(first_ref_para)

                while para_idx < end_para and str(para_idx) in duplicate_paras and str(para_idx) not in removed_paras:
                    ref = duplicate_paras[str(para_idx)]
                    ref_book, ref_p = ref.rsplit(":", 1)
                    expected_para = ref_para + (para_idx - start_dup)

                    if ref_book != first_ref_book or int(ref_p) != expected_para:
                        break

                    d_text = get_paragraph_text(sentences, para_starts, para_idx)
                    d_bpb = (
                        bpb_values[para_idx]
                        if bpb_values and para_idx < len(bpb_values)
                        else None
                    )
                    d_bpb_valid = d_bpb if d_bpb and d_bpb > 0 else None
                    if d_bpb is not None and d_bpb > 0:
                        section_bpbs.append(d_bpb)
                    d_lang = languages[para_idx] if para_idx < len(languages) else None
                    dup_paras_tags.append(
                        build_paragraph_tag(d_text, d_bpb_valid, d_lang)
                    )
                    para_idx += 1

                last_ref_para = ref_para + len(dup_paras_tags) - 1
                if ref_para == last_ref_para:
                    cluster = f"{first_ref_book}:{ref_para}"
                else:
                    cluster = f"{first_ref_book}:{ref_para}-{last_ref_para}"

                dup_content = "\n".join(dup_paras_tags)
                section_content.append(build_duplicate_tag(dup_content, cluster))

            else:
                para_bpb_valid = para_bpb if para_bpb and para_bpb > 0 else None
                para_lang = languages[para_idx] if para_idx < len(languages) else None
                section_content.append(build_paragraph_tag(para_text, para_bpb_valid, para_lang))
                para_idx += 1

        section_bpb = None
        if section_bpbs:
            section_bpb = sum(section_bpbs) / len(section_bpbs)

        section_inner = "\n".join(section_content)
        annotated_sections.append(build_section_tag(section_inner, section_bpb))

    included_languages = [
        languages[i] for i in range(len(languages)) if str(i) not in removed_paras
    ]
    lang_dist = compute_language_distribution(included_languages)
    return "\n".join(annotated_sections), lang_dist
