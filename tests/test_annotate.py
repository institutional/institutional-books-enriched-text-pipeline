"""Tests for library/annotate modules."""


from library.annotate.middlematter import (
    annotate_middlematter,
    get_paragraph_text,
    get_section_para_range,
)
from library.annotate.tags import (
    build_duplicate_tag,
    build_endmatter_tag,
    build_paragraph_tag,
    build_representative_tag,
    build_section_tag,
    escape_xml,
)


class TestEscapeXml:
    def test_escapes_ampersand(self):
        assert escape_xml("Tom & Jerry") == "Tom &amp; Jerry"

    def test_escapes_less_than(self):
        assert escape_xml("a < b") == "a &lt; b"

    def test_escapes_greater_than(self):
        assert escape_xml("a > b") == "a &gt; b"

    def test_preserves_normal_text(self):
        assert escape_xml("Hello World") == "Hello World"

    def test_handles_empty_string(self):
        assert escape_xml("") == ""


class TestBuildEndmatterTag:
    def test_toc_index_tag(self):
        result = build_endmatter_tag("Table of Contents", "TOC_INDEX")
        assert result == '<idi-endmatter type="TOC_INDEX">Table of Contents</idi-endmatter>'

    def test_biblio_tag(self):
        result = build_endmatter_tag("References", "BIBLIO")
        assert result == '<idi-endmatter type="BIBLIO">References</idi-endmatter>'

    def test_otherendmatter_tag(self):
        result = build_endmatter_tag("Appendix", "OTHERENDMATTER")
        assert result == '<idi-endmatter type="OTHERENDMATTER">Appendix</idi-endmatter>'

    def test_escapes_content(self):
        result = build_endmatter_tag("A & B", "TOC_INDEX")
        assert "&amp;" in result


class TestBuildParagraphTag:
    def test_paragraph_with_perplexity(self):
        result = build_paragraph_tag("Some text.", 12.5)
        assert result == '<idi-paragraph perplexity="12.5">Some text.</idi-paragraph>'

    def test_paragraph_without_perplexity(self):
        result = build_paragraph_tag("Some text.")
        assert result == "<idi-paragraph>Some text.</idi-paragraph>"

    def test_perplexity_formatting(self):
        result = build_paragraph_tag("Text", 8.123456)
        assert 'perplexity="8.1"' in result

    def test_escapes_content(self):
        result = build_paragraph_tag("A < B", 5.0)
        assert "&lt;" in result


class TestBuildSectionTag:
    def test_section_with_perplexity(self):
        content = "<idi-paragraph>P1</idi-paragraph>"
        result = build_section_tag(content, 10.5)
        assert '<idi-section perplexity="10.5">' in result
        assert "</idi-section>" in result
        assert content in result

    def test_section_without_perplexity(self):
        content = "<idi-paragraph>P1</idi-paragraph>"
        result = build_section_tag(content)
        assert "<idi-section>" in result
        assert "</idi-section>" in result


class TestBuildDuplicateTag:
    def test_duplicate_single_para(self):
        content = "<idi-paragraph>Dup</idi-paragraph>"
        result = build_duplicate_tag(content, "book1:p:5")
        assert '<idi-duplicate cluster="book1:p:5">' in result
        assert "</idi-duplicate>" in result

    def test_duplicate_range(self):
        content = "<idi-paragraph>Dup1</idi-paragraph>\n<idi-paragraph>Dup2</idi-paragraph>"
        result = build_duplicate_tag(content, "book1:p:5-7")
        assert 'cluster="book1:p:5-7"' in result


class TestBuildRepresentativeTag:
    def test_representative_tag(self):
        content = "<idi-paragraph>Rep</idi-paragraph>"
        result = build_representative_tag(content, "book1:p:3")
        assert '<idi-representative cluster="book1:p:3">' in result
        assert "</idi-representative>" in result


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
        result = annotate_middlematter(book)
        assert result == ""

    def test_single_paragraph(self):
        book = {
            "barcode_src": "test",
            "middlematter_sentences": ["Hello world."],
            "subtopic_paragraph_start_indices": [0],
            "subtopic_section_start_indices": [0],
        }
        result = annotate_middlematter(book)
        assert "<idi-section>" in result
        assert "<idi-paragraph>Hello world.</idi-paragraph>" in result

    def test_with_perplexity(self):
        book = {
            "barcode_src": "test",
            "middlematter_sentences": ["Content here."],
            "subtopic_paragraph_start_indices": [0],
            "subtopic_section_start_indices": [0],
        }
        result = annotate_middlematter(book, perplexities=[12.5])
        assert 'perplexity="12.5"' in result

    def test_invalid_perplexity_excluded(self):
        book = {
            "barcode_src": "test",
            "middlematter_sentences": ["Short."],
            "subtopic_paragraph_start_indices": [0],
            "subtopic_section_start_indices": [0],
        }
        # -1 indicates perplexity couldn't be computed
        result = annotate_middlematter(book, perplexities=[-1.0])
        assert "perplexity=" not in result

    def test_representative_paragraph(self):
        book = {
            "barcode_src": "book1",
            "middlematter_sentences": ["Representative content."],
            "subtopic_paragraph_start_indices": [0],
            "subtopic_section_start_indices": [0],
            "representative_paragraphs": {"0": True},
        }
        result = annotate_middlematter(book)
        assert "<idi-representative" in result
        assert 'cluster="book1:p:0"' in result

    def test_duplicate_paragraph(self):
        book = {
            "barcode_src": "book1",
            "middlematter_sentences": ["Duplicate content."],
            "subtopic_paragraph_start_indices": [0],
            "subtopic_section_start_indices": [0],
            "duplicate_paragraphs": {"0": "book2:p:5"},
        }
        result = annotate_middlematter(book)
        assert "<idi-duplicate" in result

    def test_consecutive_duplicates_merged(self):
        book = {
            "barcode_src": "book1",
            "middlematter_sentences": ["Dup1.", "Dup2.", "Dup3."],
            "subtopic_paragraph_start_indices": [0, 1, 2],
            "subtopic_section_start_indices": [0],
            "duplicate_paragraphs": {"0": "x:p:0", "1": "x:p:1", "2": "x:p:2"},
        }
        result = annotate_middlematter(book)
        # Should have a range cluster
        assert 'cluster="book1:p:0-2"' in result

    def test_multiple_sections(self):
        book = {
            "barcode_src": "test",
            "middlematter_sentences": ["Sec1 Para1.", "Sec2 Para1."],
            "subtopic_paragraph_start_indices": [0, 1],
            "subtopic_section_start_indices": [0, 1],  # Each para in own section
        }
        result = annotate_middlematter(book)
        # Should have two sections
        assert result.count("<idi-section>") == 2
