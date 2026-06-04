"""Tests for library/denoise/stray_numbers.py

Institutional Books - Enriched Text - 2026
"""

from library.denoise.stray_numbers import (
    detect_and_remove_stray_number_fragments,
    is_probable_stray_number_fragment,
    remove_stray_numbers_book,
)
from utils.text import numeric_count


class TestNumericCount:
    def test_all_numeric(self):
        numeric, nonnumeric = numeric_count("12345")
        assert numeric == 5
        assert nonnumeric == 0

    def test_all_letters(self):
        numeric, nonnumeric = numeric_count("hello")
        assert numeric == 0
        assert nonnumeric == 5

    def test_mixed(self):
        numeric, nonnumeric = numeric_count("page 42")
        assert numeric == 2
        assert nonnumeric == 4  # 'p', 'a', 'g', 'e'

    def test_unicode_numeric(self):
        # Unicode numeric characters should be counted
        numeric, nonnumeric = numeric_count("①②③")
        assert numeric == 3
        assert nonnumeric == 0

    def test_whitespace_ignored(self):
        numeric, nonnumeric = numeric_count("1 2 3")
        assert numeric == 3
        assert nonnumeric == 0

    def test_empty_string(self):
        numeric, nonnumeric = numeric_count("")
        assert numeric == 0
        assert nonnumeric == 0


class TestIsProbableStrayNumberFragment:
    def test_pure_number(self):
        assert is_probable_stray_number_fragment("123")
        assert is_probable_stray_number_fragment("456789")

    def test_number_with_punctuation(self):
        # 90% threshold: "123." has 3 numeric, 1 non-numeric = 75% numeric
        # but only 1 non-numeric
        assert is_probable_stray_number_fragment("123.")
        # "12345." has 5 numeric, 1 non-numeric = 83% numeric
        assert not is_probable_stray_number_fragment("1122334455..")
        # "123456789." has 9 numeric, 1 non-numeric = 90% numeric
        assert is_probable_stray_number_fragment("123456789.")

    def test_mostly_text(self):
        assert not is_probable_stray_number_fragment("This is sentence 1.")
        assert not is_probable_stray_number_fragment("Chapter 42")

    def test_too_short(self):
        # Default min_length is 2
        assert not is_probable_stray_number_fragment("1")

    def test_at_min_length(self):
        assert is_probable_stray_number_fragment("123")

    def test_empty_string(self):
        assert not is_probable_stray_number_fragment("")

    def test_whitespace_only(self):
        assert not is_probable_stray_number_fragment("   ")

    def test_custom_threshold(self):
        assert not is_probable_stray_number_fragment("112233aa", threshold=0.9)
        assert is_probable_stray_number_fragment("123a", threshold=0.7)

    def test_custom_min_length(self):
        assert not is_probable_stray_number_fragment("12", min_length=3)
        assert is_probable_stray_number_fragment("12", min_length=2)


class TestDetectAndRemoveStrayNumberFragments:
    def test_removes_pure_numbers(self):
        sentences = ["Hello world.", "123", "Goodbye."]
        result = detect_and_remove_stray_number_fragments(sentences)
        assert result == ["Hello world.", "Goodbye."]

    def test_preserves_normal_sentences(self):
        sentences = ["This is a test.", "Another sentence.", "Final one."]
        result = detect_and_remove_stray_number_fragments(sentences)
        assert result == sentences

    def test_preserves_sentences_with_numbers(self):
        sentences = ["There are 42 items.", "Chapter 1 begins here."]
        result = detect_and_remove_stray_number_fragments(sentences)
        assert result == sentences

    def test_removes_multiple_stray_numbers(self):
        sentences = ["Text.", "456", "More text.", "789", "End."]
        result = detect_and_remove_stray_number_fragments(sentences)
        assert result == ["Text.", "More text.", "End."]

    def test_empty_list(self):
        result = detect_and_remove_stray_number_fragments([])
        assert result == []

    def test_all_stray_numbers(self):
        sentences = ["123", "456", "789"]
        result = detect_and_remove_stray_number_fragments(sentences)
        assert result == []


class TestRemoveStrayNumbersBook:
    def test_book_processing(self):
        book = {
            "barcode_src": "test123",
            "middlematter_sentences": [
                "First sentence.",
                "420",
                "Second sentence.",
            ],
        }
        from const.config import PipelineConfig

        config = PipelineConfig()
        result = remove_stray_numbers_book(book, config)
        assert result["middlematter_sentences"] == ["First sentence.", "Second sentence."]

    def test_preserves_other_fields(self):
        book = {
            "barcode_src": "test123",
            "language_gen": "eng",
            "middlematter_sentences": ["Content.", "999", "More content."],
        }
        from const.config import PipelineConfig

        config = PipelineConfig()
        result = remove_stray_numbers_book(book, config)
        assert result["barcode_src"] == "test123"
        assert result["language_gen"] == "eng"

    def test_raises_on_missing_sentences(self):
        book = {"barcode_src": "test123"}
        from const.config import PipelineConfig

        config = PipelineConfig()
        import pytest

        with pytest.raises(ValueError, match="No 'sentences' found"):
            remove_stray_numbers_book(book, config)

    def test_raises_on_empty_sentences(self):
        book = {"barcode_src": "test123", "middlematter_sentences": []}
        from const.config import PipelineConfig

        config = PipelineConfig()
        import pytest

        with pytest.raises(ValueError, match="No 'sentences' found"):
            remove_stray_numbers_book(book, config)
