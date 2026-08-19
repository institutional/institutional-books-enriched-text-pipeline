"""
Tests for library/denoise/pagenumber.py

Institutional Books - Enriched Text - 2026
"""

from library.denoise.pagenumber import (
    detect_and_remove_page_numbers,
    is_probable_page_number_line,
    remove_page_numbers_book,
)


class TestIsProbablePageNumberLine:
    def test_simple_number(self):
        assert is_probable_page_number_line("42")
        assert is_probable_page_number_line("1")
        assert is_probable_page_number_line("999")

    def test_number_with_one_char(self):
        # At most one non-numeric character allowed
        assert is_probable_page_number_line("-42")
        assert is_probable_page_number_line("42-")

    def test_too_many_chars(self):
        assert not is_probable_page_number_line("page 42")
        assert not is_probable_page_number_line("Chapter 1")

    def test_too_long(self):
        assert not is_probable_page_number_line("123456789")  # > 8 chars

    def test_empty_or_whitespace(self):
        assert not is_probable_page_number_line("")
        assert not is_probable_page_number_line("   ")


class TestDetectAndRemovePageNumbers:
    def test_removes_header_numbers(self):
        pages = [
            "42\nThis is the actual content.",
            "43\nMore content here.",
        ]
        result, count = detect_and_remove_page_numbers(pages, header_lines=1, footer_lines=0)
        assert "42" not in result[0]
        assert "43" not in result[1]
        assert count == 2

    def test_removes_footer_numbers(self):
        pages = [
            "Content here.\n99",
            "More content.\n100",
        ]
        result, count = detect_and_remove_page_numbers(pages, header_lines=0, footer_lines=1)
        assert "99" not in result[0]
        assert "100" not in result[1]
        assert count == 2

    def test_preserves_content(self):
        pages = [
            "42\nImportant content about 42 things.\n43",
        ]
        result, count = detect_and_remove_page_numbers(pages, header_lines=1, footer_lines=1)
        # Content in middle should be preserved
        assert "Important content about 42 things." in result[0]
        assert count == 2


class TestRemovePageNumbersBook:
    def test_book_processing(self):
        book = {
            "barcode_src": "test123",
            "middlematter": [
                "1\nPage content.",
                "2\nMore content.",
            ],
        }
        result = remove_page_numbers_book(book, header_lines=1, footer_lines=0)
        assert "1" not in result["middlematter"][0]
        assert "2" not in result["middlematter"][1]
