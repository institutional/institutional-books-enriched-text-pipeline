"""
library/metadata/text_stats.py - Text statistics computation.

Computes various text metrics for books using:
- tiktoken (o200k_base) for token counting
- polyglot for word/sentence segmentation
- Basic n-gram statistics

Institutional Books - Enriched Text - 2026
"""

from const.types import BookJSON, TextStats


def get_tiktoken_encoder():
    """Get the o200k_base tiktoken encoder (lazy load)."""
    import tiktoken

    return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str, encoder=None) -> int:
    """
    Count o200k_base tokens in text.

    Args:
        text: Input text to tokenize.
        encoder: Optional pre-loaded encoder (for efficiency).

    Returns:
        Token count.
    """
    if encoder is None:
        encoder = get_tiktoken_encoder()
    return len(encoder.encode(text))


def get_words_and_sentences(text: str, language_code: str) -> tuple[list[str], list[str]]:
    """
    Get word and sentence lists using polyglot.

    Args:
        text: Input text.
        language_code: ISO 639-1 language code (e.g., 'en', 'de').

    Returns:
        Tuple of (words, sentences) as lowercase strings.
    """
    import polyglot.text

    # Clean text for polyglot (remove zero-width spaces and newlines)
    clean_text = text.replace("\u200b", "").replace("\n", " ")

    nlp_text = polyglot.text.Text(clean_text, hint_language_code=language_code)

    words = [str(word).lower() for word in nlp_text.words]
    sentences = [str(sentence).lower() for sentence in nlp_text.sentences]

    return words, sentences


def compute_word_ngrams(
    words: list[str],
    n: int,
) -> list[str]:
    """
    Compute word-level n-grams.

    Args:
        words: List of words.
        n: N-gram size (2 for bigrams, 3 for trigrams).

    Returns:
        List of n-grams as space-joined strings.
    """
    if len(words) < n:
        return []
    return [" ".join(words[i : i + n]).lower() for i in range(len(words) - n + 1)]


def compute_tokenizability(words: list[str], encoder=None) -> float:
    """
    Compute tokenizability score.

    Formula: min((word_count * 1.25 / total_word_tokens) * 100, 100.0)

    Higher scores indicate more tokenization-friendly text (fewer subword splits).

    Args:
        words: List of words.
        encoder: Optional pre-loaded tiktoken encoder.

    Returns:
        Tokenizability percentage (0-100).
    """
    if not words:
        return 0.0

    if encoder is None:
        encoder = get_tiktoken_encoder()

    total_word_tokens = sum(len(encoder.encode(word)) for word in words)

    if total_word_tokens == 0:
        return 0.0

    return min((len(words) * 1.25 / total_word_tokens) * 100, 100.0)


def compute_text_stats(book: BookJSON) -> TextStats:
    """
    Compute all text statistics for a book's middlematter.

    Args:
        book: Book dictionary with middlematter_sentences.

    Returns:
        TextStats dictionary containing all text statistics.
    """
    sentences = book.get("middlematter_sentences", [])
    para_starts = book.get("subtopic_paragraph_start_indices", [])
    section_starts = book.get("subtopic_section_start_indices", [])
    language = book.get("language_gen", "unknown")

    # Join all sentences into text
    text = " ".join(sentences)

    if not text.strip():
        return {
            "token_count": 0,
            "char_count": 0,
            "word_count": 0,
            "sentence_count": 0,
            "paragraph_count": 0,
            "section_count": 0,
            "bigram_count": 0,
            "bigram_count_unique": 0,
            "trigram_count": 0,
            "trigram_count_unique": 0,
            "tokenizability_o200k_base_ratio": 0.0,
        }

    # Get encoder once for efficiency
    encoder = get_tiktoken_encoder()

    # Basic counts
    token_count = count_tokens(text, encoder)
    char_count = len(text)

    # Use polyglot for word/sentence segmentation
    words, polyglot_sentences = get_words_and_sentences(text, language)

    word_count = len(words)
    sentence_count = len(polyglot_sentences) if polyglot_sentences else len(sentences)
    paragraph_count = len(para_starts) if para_starts else 0
    section_count = len(section_starts) if section_starts else 0

    # N-gram statistics (word-based)
    bigrams = compute_word_ngrams(words, 2)
    trigrams = compute_word_ngrams(words, 3)

    bigram_count = len(bigrams)
    bigram_count_unique = len(set(bigrams))
    trigram_count = len(trigrams)
    trigram_count_unique = len(set(trigrams))

    # Tokenizability
    tokenizability = compute_tokenizability(words, encoder)

    return {
        "token_count": token_count,
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "section_count": section_count,
        "bigram_count": bigram_count,
        "bigram_count_unique": bigram_count_unique,
        "trigram_count": trigram_count,
        "trigram_count_unique": trigram_count_unique,
        "tokenizability_o200k_base_ratio": tokenizability,
    }
