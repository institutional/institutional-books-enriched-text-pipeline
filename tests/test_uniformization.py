"""Tests for library/denoise/uniformize.py"""

import pytest

from const.types import BookJSON
from library.denoise.uniformize import normalize_unicode_in_page, uniformize_book


class TestNormalizeUnicodeInPage:
    def test_basic_normalization(self):
        # NFKC should normalize full-width chars
        result = normalize_unicode_in_page("Ｈｅｌｌｏ")
        assert result == "Hello"

    def test_preserves_newlines(self):
        result = normalize_unicode_in_page("line1\nline2\nline3")
        assert result == "line1\nline2\nline3"

    def test_empty_page(self):
        result = normalize_unicode_in_page("")
        assert result == ""


class TestUniformizeBook:
    def test_basic_uniformization(self):
        book = {
            "barcode_src": "test123",
            "text_by_page_src": ["page1", "page2"],
        }
        result = uniformize_book(book)

        assert "uniformized_text" in result
        assert len(result["uniformized_text"]) == 2
        assert "text_by_page_src" not in result

    def test_removes_old_fields(self):
        book = {
            "barcode_src": "test123",
            "text_by_page_src": ["page1"],
            "text_by_page_gen": ["generated"],
            "text_analysis_gen": {"some": "analysis"},
        }
        result = uniformize_book(book)

        assert "text_by_page_src" not in result
        assert "text_by_page_gen" not in result
        assert "text_analysis_gen" not in result

    def test_raises_on_missing_text(self):
        book = {"barcode_src": "test123"}
        with pytest.raises(ValueError, match="No 'text_by_page_src'"):
            uniformize_book(book)

    def test_raises_on_empty_text(self):
        book: BookJSON = {"barcode_src": "test123", "text_by_page_src": []}
        with pytest.raises(ValueError, match="No 'text_by_page_src'"):
            uniformize_book(book)
