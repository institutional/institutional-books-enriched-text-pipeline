"""
Tests for unicode uniformization
"""

from library.denoise.uniformize import (
    HYPHEN_LIKE,
    QUOTE_MAP,
    SOFT_HYPHEN,
    SPACE_LIKE,
    ZERO_WIDTH,
    hard_normalize_unicode,
    light_normalize_text,
    normalize_hyphens,
    normalize_quotes,
    normalize_spaces,
    normalize_text,
    remove_zero_width,
    to_char_tokens,
)


class TestNormalizeHyphens:
    """Test hyphen normalization."""

    def test_regular_hyphen_unchanged(self):
        """ASCII hyphen-minus should remain unchanged."""
        assert normalize_hyphens("well-known") == "well-known"

    def test_en_dash(self):
        """EN DASH should be converted to ASCII hyphen."""
        assert normalize_hyphens("pages 1–10") == "pages 1-10"

    def test_em_dash(self):
        """EM DASH should be converted to ASCII hyphen."""
        assert normalize_hyphens("word—another") == "word-another"

    def test_all_hyphen_types(self):
        """All HYPHEN_LIKE characters should be normalized."""
        for hyphen_char in HYPHEN_LIKE:
            result = normalize_hyphens(f"a{hyphen_char}b")
            assert result == "a-b", f"Failed for U+{ord(hyphen_char):04X}"

    def test_soft_hyphen_removed(self):
        """Soft hyphen should be removed entirely."""
        text_with_soft_hyphen = f"con{SOFT_HYPHEN}tin{SOFT_HYPHEN}ue"
        assert normalize_hyphens(text_with_soft_hyphen) == "continue"

    def test_mixed_hyphens(self):
        """Multiple hyphen types in same text."""
        # EN DASH and EM DASH
        text = "well\u2013known\u2014fact"
        assert normalize_hyphens(text) == "well-known-fact"

    def test_empty_string(self):
        """Empty string should return empty."""
        assert normalize_hyphens("") == ""

    def test_no_hyphens(self):
        """Text without hyphens should be unchanged."""
        assert normalize_hyphens("hello world") == "hello world"


class TestNormalizeSpaces:
    """Test space normalization."""

    def test_regular_space_unchanged(self):
        """ASCII space should remain unchanged."""
        assert normalize_spaces("hello world") == "hello world"

    def test_non_breaking_space(self):
        """Non-breaking space should be converted."""
        assert normalize_spaces("hello\u00a0world") == "hello world"

    def test_ideographic_space(self):
        """Ideographic space (CJK) should be converted."""
        assert normalize_spaces("hello\u3000world") == "hello world"

    def test_all_space_types(self):
        """All SPACE_LIKE characters should be normalized."""
        for space_char in SPACE_LIKE:
            result = normalize_spaces(f"a{space_char}b")
            assert result == "a b", f"Failed for U+{ord(space_char):04X}"

    def test_mixed_spaces(self):
        """Multiple space types in same text."""
        # NO-BREAK SPACE and IDEOGRAPHIC SPACE
        text = "hello\u00a0world\u3000test"
        assert normalize_spaces(text) == "hello world test"

    def test_empty_string(self):
        """Empty string should return empty."""
        assert normalize_spaces("") == ""


class TestRemoveZeroWidth:
    """Test zero-width character removal."""

    def test_zero_width_space(self):
        """Zero-width space should be removed."""
        assert remove_zero_width("hello\u200bworld") == "helloworld"

    def test_zero_width_joiner(self):
        """Zero-width joiner should be removed."""
        assert remove_zero_width("a\u200db") == "ab"

    def test_bom(self):
        """BOM (zero-width no-break space) should be removed."""
        assert remove_zero_width("\ufeffhello") == "hello"

    def test_all_zero_width_types(self):
        """All ZERO_WIDTH characters should be removed."""
        for zw_char in ZERO_WIDTH:
            result = remove_zero_width(f"a{zw_char}b")
            assert result == "ab", f"Failed for U+{ord(zw_char):04X}"

    def test_multiple_zero_width(self):
        """Multiple zero-width characters should all be removed."""
        text = "\ufeffhello\u200bworld\u200c!"
        assert remove_zero_width(text) == "helloworld!"

    def test_empty_string(self):
        """Empty string should return empty."""
        assert remove_zero_width("") == ""

    def test_no_zero_width(self):
        """Text without zero-width chars should be unchanged."""
        assert remove_zero_width("hello world") == "hello world"


class TestNormalizeQuotes:
    """Test quote normalization."""

    def test_curly_single_quotes(self):
        """Curly single quotes should become straight."""
        assert normalize_quotes("\u2018hello\u2019") == "'hello'"

    def test_curly_double_quotes(self):
        """Curly double quotes should become straight."""
        assert normalize_quotes("\u201chello\u201d") == '"hello"'

    def test_backtick(self):
        """Backtick should become straight single quote."""
        assert normalize_quotes("`hello`") == "'hello'"

    def test_low_9_quotes(self):
        """Low-9 quotation marks should be normalized."""
        assert normalize_quotes("\u201ehello\u201d") == '"hello"'
        assert normalize_quotes("\u201ahello\u2019") == "'hello'"

    def test_all_quote_types(self):
        """All mapped quote characters should be normalized."""
        for code_point, replacement in QUOTE_MAP.items():
            char = chr(code_point)
            result = normalize_quotes(f"a{char}b")
            assert result == f"a{replacement}b", f"Failed for U+{code_point:04X}"

    def test_mixed_quotes(self):
        """Multiple quote types in same text."""
        text = "\u2018Hello,\u201d she said, \u201cworld!\u201d"
        expected = '\'Hello," she said, "world!"'
        assert normalize_quotes(text) == expected

    def test_empty_string(self):
        """Empty string should return empty."""
        assert normalize_quotes("") == ""


class TestHardNormalizeUnicode:
    """Test the hard_normalize_unicode (full NFKC + all normalizations)."""

    def test_basic_normalization(self):
        """Basic text should be cleaned up."""
        assert hard_normalize_unicode("  hello   world  ") == "hello world"

    def test_newlines_become_spaces(self):
        """Newlines should become spaces."""
        assert hard_normalize_unicode("hello\nworld") == "hello world"

    def test_tabs_become_spaces(self):
        """Tabs should become spaces."""
        assert hard_normalize_unicode("hello\tworld") == "hello world"

    def test_crlf_normalization(self):
        """CRLF should be handled."""
        assert hard_normalize_unicode("hello\r\nworld") == "hello world"

    def test_combined_normalization(self):
        """Test that all normalizations work together."""
        text = "\u201cHello\u201d\u2013world\u00a0test\u200b!"
        expected = '"Hello"-world test!'
        assert hard_normalize_unicode(text) == expected

    def test_unicode_nfkc(self):
        """NFKC normalization should be applied."""
        assert hard_normalize_unicode("\uff21") == "A"
        assert hard_normalize_unicode("x\u00b2") == "x2"

    def test_multiple_spaces_collapsed(self):
        """Multiple consecutive spaces should collapse to one."""
        assert hard_normalize_unicode("hello    world") == "hello world"
        assert hard_normalize_unicode("a  b   c    d") == "a b c d"

    def test_empty_string(self):
        """Empty string should return empty."""
        assert hard_normalize_unicode("") == ""

    def test_whitespace_only(self):
        """Whitespace-only string should return empty."""
        assert hard_normalize_unicode("   \n\t  ") == ""

    def test_normalize_text_is_alias(self):
        """normalize_text should be an alias for hard_normalize_unicode."""
        assert normalize_text is hard_normalize_unicode


class TestLightNormalizeText:
    """Test light_normalize_text (NFC + zero-width + spaces only)."""

    def test_nfc_normalization(self):
        """NFC composes decomposed characters."""
        # e + combining acute -> precomposed \u00e9
        assert light_normalize_text("e\u0301") == "\u00e9"

    def test_preserves_fullwidth(self):
        """NFC does not decompose compatibility characters."""
        assert light_normalize_text("\uff21") == "\uff21"

    def test_removes_zero_width(self):
        """Zero-width characters should be removed."""
        assert light_normalize_text("hello\u200bworld") == "helloworld"

    def test_normalizes_spaces(self):
        """Non-breaking and other spaces should become regular spaces."""
        assert light_normalize_text("hello\u00a0world") == "hello world"

    def test_collapses_multiple_spaces(self):
        """Multiple spaces should collapse to one."""
        assert light_normalize_text("hello   world") == "hello world"

    def test_strips_whitespace(self):
        """Leading/trailing spaces should be stripped."""
        assert light_normalize_text("  hello  ") == "hello"

    def test_preserves_hyphens(self):
        """Hyphens should NOT be normalized."""
        assert light_normalize_text("word\u2014another") == "word\u2014another"
        assert light_normalize_text("pages 1\u2013 10") == "pages 1\u2013 10"

    def test_preserves_quotes(self):
        """Quotes should NOT be normalized."""
        assert light_normalize_text("\u201cHello\u201d") == "\u201cHello\u201d"

    def test_preserves_newlines_tabs(self):
        """Newlines and tabs should NOT be converted to spaces."""
        # light_normalize_text doesn't do tab/newline -> space
        assert light_normalize_text("hello\tworld") == "hello\tworld"

    def test_empty_string(self):
        """Empty string should return empty."""
        assert light_normalize_text("") == ""


class TestToCharTokens:
    """Test character tokenization."""

    def test_basic_tokenization(self):
        """Characters should be space-separated."""
        assert to_char_tokens("abc") == "a b c"

    def test_space_becomes_special_token(self):
        """Spaces should become <sp> token."""
        assert to_char_tokens("a b") == "a <sp> b"

    def test_multiple_spaces(self):
        """Multiple spaces should each become <sp>."""
        assert to_char_tokens("a  b") == "a <sp> <sp> b"

    def test_sentence(self):
        """Full sentence tokenization."""
        result = to_char_tokens("Hi there")
        expected = "H i <sp> t h e r e"
        assert result == expected

    def test_empty_string(self):
        """Empty string should return empty."""
        assert to_char_tokens("") == ""

    def test_single_char(self):
        """Single character should return itself."""
        assert to_char_tokens("x") == "x"

    def test_only_space(self):
        """Single space should become <sp>."""
        assert to_char_tokens(" ") == "<sp>"

    def test_special_characters(self):
        """Special characters should be tokenized normally."""
        assert to_char_tokens("a!b") == "a ! b"
        assert to_char_tokens("1+2") == "1 + 2"
