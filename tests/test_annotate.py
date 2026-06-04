"""Tests for library/annotate modules."""


from library.annotate.language import (
    UNKNOWN,
    detect_language,
    detect_paragraph_languages,
)
from library.annotate.middlematter import (
    annotate_middlematter,
    get_paragraph_text,
    get_section_para_range,
)
from library.annotate.tags import (
    build_duplicate_tag,
    build_endmatter_page_tag,
    build_paragraph_tag,
    build_section_tag,
    escape_html,
)


class TestEscapeHtml:
    def test_escapes_ampersand(self):
        assert escape_html("Tom & Jerry") == "Tom &amp; Jerry"

    def test_escapes_less_than(self):
        assert escape_html("a < b") == "a &lt; b"

    def test_escapes_greater_than(self):
        assert escape_html("a > b") == "a &gt; b"

    def test_preserves_normal_text(self):
        assert escape_html("Hello World") == "Hello World"

    def test_handles_empty_string(self):
        assert escape_html("") == ""


class TestBuildEndmatterPageTag:
    def test_toc_index_tag(self):
        result = build_endmatter_page_tag("Table of Contents", "TOC_INDEX")
        assert result == '<div class="toc_index">Table of Contents</div>'

    def test_biblio_tag(self):
        result = build_endmatter_page_tag("References", "BIBLIO")
        assert result == '<div class="biblio">References</div>'

    def test_otherendmatter_tag(self):
        result = build_endmatter_page_tag("Appendix", "OTHERENDMATTER")
        assert result == '<div class="otherendmatter">Appendix</div>'

    def test_escapes_content(self):
        result = build_endmatter_page_tag("A & B", "TOC_INDEX")
        assert "&amp;" in result


class TestBuildParagraphTag:
    def test_paragraph_with_bpb(self):
        result = build_paragraph_tag("Some text.", 12.5)
        assert result == '<p data-bpb="12.5">Some text.</p>'

    def test_paragraph_without_bpb(self):
        result = build_paragraph_tag("Some text.")
        assert result == "<p>Some text.</p>"

    def test_bpb_formatting(self):
        result = build_paragraph_tag("Text", 8.123456)
        assert 'data-bpb="8.1"' in result

    def test_escapes_content(self):
        result = build_paragraph_tag("A < B", 5.0)
        assert "&lt;" in result

    def test_representative_attributes(self):
        result = build_paragraph_tag("Rep text.", 9.2, "eng", representative_cluster="book1:10")
        assert "data-representative" in result
        assert 'data-clusterid="book1:10"' in result
        assert 'data-bpb="9.2"' in result
        assert 'data-language="eng"' in result


class TestBuildSectionTag:
    def test_section_with_bpb(self):
        content = "<p>P1</p>"
        result = build_section_tag(content, 10.5)
        assert '<section data-bpb="10.5">' in result
        assert "</section>" in result
        assert content in result

    def test_section_without_bpb(self):
        content = "<p>P1</p>"
        result = build_section_tag(content)
        assert "<section>" in result
        assert "</section>" in result


class TestBuildDuplicateTag:
    def test_duplicate_single_para(self):
        content = "<p>Dup</p>"
        result = build_duplicate_tag(content, "book1:5")
        assert '<aside data-cluster="book1:5">' in result
        assert "</aside>" in result

    def test_duplicate_range(self):
        content = "<p>Dup1</p>\n<p>Dup2</p>"
        result = build_duplicate_tag(content, "book1:5-7")
        assert 'data-cluster="book1:5-7"' in result


class TestGetParagraphText:
    def test_single_sentence_paragraph(self):
        sentences = ["First.", "Second.", "Third."]
        para_starts = [0, 1, 2]
        result = get_paragraph_text(sentences, para_starts, 0)
        assert result == "First."

    def test_multi_sentence_paragraph(self):
        sentences = ["First.", "Second.", "Third."]
        para_starts = [0, 2]  # Para 0: sentences 0-1, Para 1: sentence 2
        result = get_paragraph_text(sentences, para_starts, 0)
        assert result == "First. Second."

    def test_last_paragraph(self):
        sentences = ["A.", "B.", "C."]
        para_starts = [0, 2]
        result = get_paragraph_text(sentences, para_starts, 1)
        assert result == "C."


class TestGetSectionParaRange:
    def test_single_section(self):
        section_starts = [0]
        para_starts = [0, 1, 2]
        start, end = get_section_para_range(0, section_starts, para_starts)
        assert start == 0
        assert end == 3

    def test_multiple_sections(self):
        section_starts = [0, 2]  # Section 0: paras 0-1, Section 1: para 2
        para_starts = [0, 1, 2]

        start0, end0 = get_section_para_range(0, section_starts, para_starts)
        assert start0 == 0
        assert end0 == 2

        start1, end1 = get_section_para_range(1, section_starts, para_starts)
        assert start1 == 2
        assert end1 == 3


class TestAnnotateMiddlematter:
    def test_empty_book(self):
        book = {
            "barcode_src": "test",
            "middlematter_sentences": [],
            "subtopic_paragraph_start_indices": [],
        }
        result, lang_dist = annotate_middlematter(book)
        assert result == ""
        assert lang_dist == {"languages": [], "proportion": []}

    def test_single_paragraph(self):
        book = {
            "barcode_src": "test",
            "middlematter_sentences": ["Hello world."],
            "subtopic_paragraph_start_indices": [0],
            "subtopic_section_start_indices": [0],
        }
        result, _ = annotate_middlematter(book)
        assert "<section>" in result
        assert "<p" in result
        assert "Hello world.</p>" in result

    def test_with_bpb(self):
        book = {
            "barcode_src": "test",
            "middlematter_sentences": ["Content here."],
            "subtopic_paragraph_start_indices": [0],
            "subtopic_section_start_indices": [0],
        }
        result, _ = annotate_middlematter(book, bpb_values=[12.5])
        assert 'data-bpb="12.5"' in result

    def test_invalid_bpb_excluded(self):
        book = {
            "barcode_src": "test",
            "middlematter_sentences": ["Short."],
            "subtopic_paragraph_start_indices": [0],
            "subtopic_section_start_indices": [0],
        }
        # -1 indicates bpb couldn't be computed
        result, _ = annotate_middlematter(book, bpb_values=[-1.0])
        assert "data-bpb=" not in result

    def test_representative_paragraph(self):
        book = {
            "barcode_src": "book1",
            "middlematter_sentences": ["Representative content."],
            "subtopic_paragraph_start_indices": [0],
            "subtopic_section_start_indices": [0],
            "representative_paragraphs": {"0": True},
        }
        result, _ = annotate_middlematter(book)
        assert "data-representative" in result
        assert 'data-clusterid="book1:0"' in result

    def test_duplicate_paragraph(self):
        book = {
            "barcode_src": "book1",
            "middlematter_sentences": ["Duplicate content."],
            "subtopic_paragraph_start_indices": [0],
            "subtopic_section_start_indices": [0],
            "duplicate_paragraphs": {"0": "book2:5"},
        }
        result, _ = annotate_middlematter(book)
        assert "<aside" in result
        assert 'data-cluster="book2:5"' in result

    def test_consecutive_duplicates_merged_by_representative(self):
        """Consecutive duplicates referencing consecutive paragraphs in the same
        representative book are merged into a single tag with a range reference."""
        book = {
            "barcode_src": "book1",
            "middlematter_sentences": ["Dup1.", "Dup2.", "Dup3."],
            "subtopic_paragraph_start_indices": [0, 1, 2],
            "subtopic_section_start_indices": [0],
            "duplicate_paragraphs": {"0": "bookX:5", "1": "bookX:6", "2": "bookX:7"},
        }
        result, _ = annotate_middlematter(book)
        # Should reference the representative's range
        assert 'data-cluster="bookX:5-7"' in result
        # Should be a single aside wrapping all three
        assert result.count("<aside") == 1

    def test_consecutive_duplicates_not_merged_different_books(self):
        """Consecutive duplicates referencing different representative books
        should NOT be merged."""
        book = {
            "barcode_src": "book1",
            "middlematter_sentences": ["Dup1.", "Dup2."],
            "subtopic_paragraph_start_indices": [0, 1],
            "subtopic_section_start_indices": [0],
            "duplicate_paragraphs": {"0": "bookX:5", "1": "bookY:3"},
        }
        result, _ = annotate_middlematter(book)
        assert 'data-cluster="bookX:5"' in result
        assert 'data-cluster="bookY:3"' in result
        assert result.count("<aside") == 2

    def test_consecutive_duplicates_not_merged_nonconsecutive_refs(self):
        """Consecutive duplicates referencing non-consecutive paragraphs in the
        same book should NOT be merged."""
        book = {
            "barcode_src": "book1",
            "middlematter_sentences": ["Dup1.", "Dup2."],
            "subtopic_paragraph_start_indices": [0, 1],
            "subtopic_section_start_indices": [0],
            "duplicate_paragraphs": {"0": "bookX:5", "1": "bookX:10"},
        }
        result, _ = annotate_middlematter(book)
        assert 'data-cluster="bookX:5"' in result
        assert 'data-cluster="bookX:10"' in result
        assert result.count("<aside") == 2

    def test_multiple_sections(self):
        book = {
            "barcode_src": "test",
            "middlematter_sentences": ["Sec1 Para1.", "Sec2 Para1."],
            "subtopic_paragraph_start_indices": [0, 1],
            "subtopic_section_start_indices": [0, 1],  # Each para in own section
        }
        result, _ = annotate_middlematter(book)
        # Should have two sections
        assert result.count("<section>") == 2


class TestDetectLanguage:
    def test_detects_english(self):
        text = "This is a long English sentence with many words for reliable detection."
        result = detect_language(text)
        assert result == "eng"

    def test_detects_german(self):
        text = "Das ist ein langer deutscher Satz mit vielen Wörtern."
        result = detect_language(text)
        assert result == "deu"

    def test_returns_unknown_for_short_text(self):
        result = detect_language("123")
        assert result == UNKNOWN

    def test_returns_unknown_for_empty_text(self):
        result = detect_language("")
        assert result == UNKNOWN


class TestDetectParagraphLanguages:
    def test_detects_multiple_languages(self):
        paragraphs = [
            "This is a long English paragraph with enough words.",
            "Das ist ein langer deutscher Absatz mit vielen Wörtern.",
        ]
        langs = detect_paragraph_languages(paragraphs)
        assert langs == ["eng", "deu"]

    def test_propagates_to_short_unknown(self):
        paragraphs = [
            "This is a long English paragraph with enough words.",
            "42",  # Short, should inherit 'eng'
            "Another long English paragraph with enough words.",
        ]
        langs = detect_paragraph_languages(paragraphs)
        assert langs[0] == "eng"
        assert langs[1] == "eng"  # Inherited
        assert langs[2] == "eng"

    def test_no_propagation_when_neighbors_differ(self):
        paragraphs = [
            "This is a long English paragraph with enough words.",
            "42",
            "Das ist ein langer deutscher Absatz mit vielen Wörtern.",
        ]
        langs = detect_paragraph_languages(paragraphs)
        assert langs[0] == "eng"
        assert langs[1] == UNKNOWN  # Not propagated
        assert langs[2] == "deu"

    def test_no_propagation_for_long_paragraph(self):
        paragraphs = [
            "This is a long English paragraph with enough words.",
            "12345 " * 50,  # Long but undetectable (just numbers)
            "Another long English paragraph with enough words.",
        ]
        langs = detect_paragraph_languages(paragraphs, token_threshold=30)
        # Long paragraph should not be propagated even with same neighbors
        assert langs[1] == UNKNOWN

    def test_empty_list(self):
        assert detect_paragraph_languages([]) == []

    def test_single_paragraph(self):
        langs = detect_paragraph_languages(["This is English text with many words."])
        assert len(langs) == 1


class TestBuildParagraphTagWithLanguage:
    def test_paragraph_with_language(self):
        result = build_paragraph_tag("Text", language="eng")
        assert result == '<p data-language="eng">Text</p>'

    def test_paragraph_with_bpb_and_language(self):
        result = build_paragraph_tag("Text", bpb=10.5, language="deu")
        assert result == '<p data-bpb="10.5" data-language="deu">Text</p>'


class TestAnnotateMiddlematterWithLanguage:
    def test_includes_language_attribute(self):
        book = {
            "barcode_src": "test",
            "middlematter_sentences": [
                "This is a long English sentence with many words for detection."
            ],
            "subtopic_paragraph_start_indices": [0],
            "subtopic_section_start_indices": [0],
        }
        result, _ = annotate_middlematter(book)
        assert 'data-language="eng"' in result
