"""
Basic tests for const/languages.py

Institutional Books - Enriched Text - 2026
"""

from const.languages import NUPUNKT_LANGUAGES, is_nupunkt_language


class TestNupunktLanguages:
    def test_english_in_list(self):
        assert "eng" in NUPUNKT_LANGUAGES

    def test_is_list(self):
        assert isinstance(NUPUNKT_LANGUAGES, list)

    def test_has_entries(self):
        assert len(NUPUNKT_LANGUAGES) > 0


class TestIsNupunktLanguage:
    def test_english(self):
        assert is_nupunkt_language("eng") is True

    def test_unknown_language(self):
        assert is_nupunkt_language("xyz") is False

    def test_empty_string(self):
        assert is_nupunkt_language("") is False

    def test_case_sensitive(self):
        # Language codes should be lowercase
        assert is_nupunkt_language("ENG") is False
