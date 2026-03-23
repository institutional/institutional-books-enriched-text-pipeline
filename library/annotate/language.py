"""
library/annotate/language.py - Language detection for paragraphs.

Uses polyglot for language detection with fallback propagation for
short or ambiguous paragraphs. Returns ISO-639-3 (3-letter) codes.
"""

import iso639

UNKNOWN = "UNKNOWN"


def iso639_1_to_3(code: str) -> str:
    """Convert ISO-639-1 (2-letter) code to ISO-639-3 (3-letter) code."""
    try:
        lang_info = iso639.find(code)
        # Prefer terminological code (ISO 639-2/T = ISO 639-3), fall back to bibliographic
        return lang_info.get("iso639_2_t") or lang_info.get("iso639_2_b") or code
    except iso639.NonExistentLanguageError:
        return code


def detect_language(text: str) -> str:
    """
    Detect language of text using polyglot.

    Args:
        text: Input text to detect language for.

    Returns:
        ISO-639-3 language code (e.g., 'eng', 'deu') or 'UNKNOWN' if detection fails.
    """
    import polyglot.text

    try:
        clean_text = text.replace("\u200b", "").replace("\n", " ")
        nlp_text = polyglot.text.Text(clean_text)
        lang = nlp_text.language
        if lang.code and lang.code != "un":
            return iso639_1_to_3(lang.code)
    except Exception:
        pass

    return UNKNOWN


def count_tokens(text: str) -> int:
    """
    Count tokens in text using tiktoken o200k_base.

    Args:
        text: Input text.

    Returns:
        Token count.
    """
    import tiktoken

    encoder = tiktoken.get_encoding("o200k_base")
    return len(encoder.encode(text))


def detect_paragraph_languages(
    paragraphs: list[str],
    token_threshold: int = 30,
) -> list[str]:
    """
    Detect languages for a list of paragraphs with propagation for unknowns.

    Rules:
    1. If polyglot detects a language, use it
    2. If detection fails, mark as UNKNOWN
    3. Post-processing: If a paragraph is UNKNOWN AND both surrounding
       paragraphs have the same known language AND the paragraph has
       fewer than `token_threshold` tokens, inherit the surrounding language

    Args:
        paragraphs: List of paragraph texts.
        token_threshold: Maximum token count for propagation (default 30).

    Returns:
        List of language codes, one per paragraph.
    """
    if not paragraphs:
        return []

    # Phase 1: Initial detection
    languages = [detect_language(p) for p in paragraphs]

    # Phase 2: Propagation for short UNKNOWN paragraphs
    for i in range(1, len(languages) - 1):
        if languages[i] != UNKNOWN:
            continue

        prev_lang = languages[i - 1]
        next_lang = languages[i + 1]

        # Both neighbors must have same known language
        if prev_lang == UNKNOWN or next_lang == UNKNOWN:
            continue
        if prev_lang != next_lang:
            continue

        # Check token count
        if count_tokens(paragraphs[i]) < token_threshold:
            languages[i] = prev_lang

    return languages
