"""
Tests for simhash implementations (pure Python and C++ extension)

Tests both the pure Python implementation (utils/simhash.py) and the
C++ extension wrapper (utils/simhash_fast.py).
"""

import pytest

from utils.simhash import (
    BAND_BITS,
    NUM_BANDS,
    has_sufficient_entropy,
    ngrams,
)
from utils.simhash import (
    are_near_duplicates as py_are_near_duplicates,
)
from utils.simhash import (
    extract_bands as py_extract_bands,
)
from utils.simhash import (
    hamming_distance as py_hamming_distance,
)
from utils.simhash import (
    simhash128 as py_simhash128,
)

try:
    from utils.simhash_fast import (
        are_near_duplicates as fast_are_near_duplicates,
    )
    from utils.simhash_fast import (
        extract_bands as fast_extract_bands,
    )
    from utils.simhash_fast import (
        get_backend,
        is_cpp_available,
    )
    from utils.simhash_fast import (
        hamming_distance as fast_hamming_distance,
    )
    from utils.simhash_fast import (
        simhash128 as fast_simhash128,
    )
except ImportError:
    fast_simhash128 = None
    fast_hamming_distance = None
    fast_are_near_duplicates = None
    fast_extract_bands = None

    def is_cpp_available():
        return False


# For backwards compatibility in tests that don't care about backend
simhash128 = py_simhash128
hamming_distance = py_hamming_distance
are_near_duplicates = py_are_near_duplicates
extract_bands = py_extract_bands

# Skip C++ tests if not available
cpp_available = is_cpp_available()
requires_cpp = pytest.mark.skipif(not cpp_available, reason="C++ backend not available")


class TestTokenization:
    """Test basic tokenization functionality."""

    def test_5gram_basic(self):
        """Trivial 5gram tokenization."""
        text = "abcdef"
        expected_ngrams = ["abcde", "bcdef"]
        actual_ngrams = ngrams(text, 5)
        assert actual_ngrams == expected_ngrams

    def test_3gram_basic(self):
        """Trivial 3gram tokenization."""
        text = "abcdef"
        expected_ngrams = ["abc", "bcd", "cde", "def"]
        actual_ngrams = ngrams(text, 3)
        assert actual_ngrams == expected_ngrams

    def test_spaces_and_case(self):
        """Tokenization with spaces and mixed case (lowercased)."""
        text = "Ab cde F"
        expected_ngrams = ["ab cd", "b cde", " cde ", "cde f"]
        actual_ngrams = ngrams(text, 5)
        assert actual_ngrams == expected_ngrams

    def test_empty_string(self):
        """Empty string produces no ngrams."""
        assert ngrams("", 5) == []

    def test_short_string(self):
        """String shorter than n produces no ngrams."""
        assert ngrams("abc", 5) == []


class TestHasSufficientEntropy:
    """Test entropy filtering for ngrams."""

    def test_high_entropy(self):
        """Diverse characters should pass."""
        assert has_sufficient_entropy("abcde") is True

    def test_low_entropy(self):
        """Repeating characters should fail."""
        assert has_sufficient_entropy("aaaaa") is False  # 1 unique char < 2 (threshold)
        # Note: "aabba" has 2 unique chars = 40%, exactly at threshold, so it passes
        # Test with only 1 unique char to ensure failure
        assert has_sufficient_entropy("bbbbb") is False

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        # "aabbc" has 3 unique chars out of 5 = 60%
        assert has_sufficient_entropy("aabbc", threshold=0.5) is True
        assert has_sufficient_entropy("aabbc", threshold=0.7) is False


class TestSimhash:
    """Test the simhash implementation."""

    def test_deterministic(self):
        """Same input should always produce same hash."""
        text = "Welcome to Harvard Law."
        h1 = simhash128(text)
        h2 = simhash128(text)
        assert h1 == h2

    def test_case_insensitive(self):
        """Different cases should produce same hash after normalization."""
        h1 = simhash128("Hello World")
        h2 = simhash128("hello world")
        h3 = simhash128("HELLO WORLD")
        assert h1 == h2 == h3

    def test_whitespace_sensitive(self):
        """Different whitespace should produce different hashes."""
        h1 = simhash128("hello world")
        h2 = simhash128("hello wor ld")
        h3 = simhash128("helloworld")
        assert h1 != h2
        assert h2 != h3
        assert h1 != h3

    def test_different_texts_different_hashes(self):
        """Different texts should produce different hashes."""
        h1 = simhash128("Welcome to Harvard Law.")
        h2 = simhash128("What, like it's hard?")
        assert h1 != h2

    def test_short_text(self):
        """Short texts (< ngram_size) should still produce valid hashes."""
        h = simhash128("abc")
        assert isinstance(h, int)
        assert h > 0

    def test_empty_string(self):
        """Empty string should produce a valid hash."""
        h = simhash128("")
        assert isinstance(h, int)


class TestDeterministicHashes:
    """
    Test that specific inputs produce specific hash values.

    These ensure simhash computation is consistent across runs.
    """

    def test_exact_hash_ngram5_quick_brown_fox(self):
        """Verify exact hash for a well-known test string (ngram=5)."""
        text = "The quick brown fox jumps over the lazy dog."
        expected = 0xCCCBD9E22B1A88D17FFF69FFD17F693D
        actual = simhash128(text, ngram_size=5)
        assert actual == expected, f"Expected 0x{expected:032x}, got 0x{actual:032x}"

    def test_exact_hash_ngram5_hello_world(self):
        """Verify exact hash for Hello World (ngram=5)."""
        text = "Hello, World!"
        expected = 0xC62F9C60EAA70E3ACB892D82D0BE82EA
        actual = simhash128(text, ngram_size=5)
        assert actual == expected, f"Expected 0x{expected:032x}, got 0x{actual:032x}"

    def test_exact_hash_ngram5_emergency_broadcast(self):
        """Verify exact hash for emergency broadcast (ngram=5)."""
        text = "This is a test of the emergency broadcast system."
        expected = 0x0B7BD06501133B277AD276CB6E7EC76F
        actual = simhash128(text, ngram_size=5)
        assert actual == expected, f"Expected 0x{expected:032x}, got 0x{actual:032x}"

    def test_exact_hash_ngram5_lorem_ipsum(self):
        """Verify exact hash for Lorem ipsum (ngram=5)."""
        text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        expected = 0x2ED9F6C11F16B13D5CF7DEA3476EA9AF
        actual = simhash128(text, ngram_size=5)
        assert actual == expected, f"Expected 0x{expected:032x}, got 0x{actual:032x}"

    def test_exact_hash_ngram5_alphabet(self):
        """Verify exact hash for alphabet string (ngram=5)."""
        text = "abcdefghijklmnopqrstuvwxyz"
        expected = 0x6DC7DE9B7BBB64A42B7FCB95FE167A9F
        actual = simhash128(text, ngram_size=5)
        assert actual == expected, f"Expected 0x{expected:032x}, got 0x{actual:032x}"

    def test_exact_hash_ngram9_quick_brown_fox(self):
        """Verify exact hash with ngram=9 (our default for deduplication)."""
        text = "The quick brown fox jumps over the lazy dog."
        expected = 0x2D0EB6DDF4B2EC8D07867737819530FA
        actual = simhash128(text, ngram_size=9)
        assert actual == expected, f"Expected 0x{expected:032x}, got 0x{actual:032x}"

    def test_exact_hash_ngram9_paragraph(self):
        """Verify exact hash for a longer paragraph with ngram=9."""
        text = "This is a longer paragraph that should produce consistent hashes across machines. It contains enough text to generate meaningful ngrams."
        expected = 0x2431F026FE534DE765BECB58BFE4BE59
        actual = simhash128(text, ngram_size=9)
        assert actual == expected, f"Expected 0x{expected:032x}, got 0x{actual:032x}"


class TestHammingDistance:
    """Test Hamming distance calculation."""

    def test_identical_hashes(self):
        """Identical hashes should have distance 0."""
        h = simhash128("test")
        assert hamming_distance(h, h) == 0

    def test_single_bit_difference(self):
        """Flipping one bit should give distance 1."""
        h = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        h_flip = h ^ 1
        assert hamming_distance(h, h_flip) == 1
        assert are_near_duplicates(h, h_flip, threshold=1) is True
        assert are_near_duplicates(h, h_flip, threshold=0) is False

    def test_multiple_bit_difference(self):
        """Flipping multiple bits."""
        h = 0
        h_flip = 0xFF
        assert hamming_distance(h, h_flip) == 8

    def test_max_distance(self):
        """Maximum distance is 128 bits."""
        h1 = 0
        h2 = (1 << 128) - 1
        assert hamming_distance(h1, h2) == 128


class TestAreNearDuplicates:
    """Test near-duplicate detection."""

    def test_identical_texts(self):
        """Identical texts should be near duplicates."""
        h = simhash128("The quick brown fox")
        assert are_near_duplicates(h, h, threshold=0) is True

    def test_similar_texts(self):
        """Similar texts should be near duplicates with reasonable threshold."""
        h1 = simhash128("The quick brown fox jumps over the lazy dog")
        h2 = simhash128("The quick brown fox jumps over the lazy cat")
        dist = hamming_distance(h1, h2)
        assert are_near_duplicates(h1, h2, threshold=dist) is True
        assert are_near_duplicates(h1, h2, threshold=dist - 1) is False

    def test_different_texts(self):
        """Very different texts should not be near duplicates."""
        h1 = simhash128("The quick brown fox")
        h2 = simhash128("Lorem ipsum dolor sit amet")
        assert are_near_duplicates(h1, h2, threshold=5) is False


class TestMultilingual:
    """Test simhash with various languages and scripts."""

    MULTILINGUAL_TEXTS = [
        ("english", "The quick brown fox jumps over the lazy dog."),
        ("german", "Größe und Äpfel sind schön und wunderbar."),
        ("french", "Où est la bibliothèque? C'est très important."),
        ("spanish", "¿Cómo estás? El niño está bien."),
        ("russian", "Привет мир, как дела сегодня?"),
        ("chinese_simplified", "这是一个测试文本，用于测试多语言支持。"),
        ("japanese", "これはテストです。"),
        ("korean", "안녕하세요, 세계!"),
        ("arabic", "مرحبا بالعالم، هذا نص اختبار"),
        ("hebrew", "שלום עולם, זה טקסט בדיקה"),
        ("greek", "Γειά σου κόσμε, πώς είσαι;"),
        ("emoji", "Hello 👋 World 🌍!"),
        ("mixed", "Hello мир 世界 🌍"),
    ]

    @pytest.mark.parametrize("lang,text", MULTILINGUAL_TEXTS)
    def test_handles_language(self, lang: str, text: str):
        """Implementation should handle all languages."""
        h = simhash128(text)
        assert isinstance(h, int)
        assert h > 0

    @pytest.mark.parametrize("lang,text", MULTILINGUAL_TEXTS)
    def test_deterministic_multilingual(self, lang: str, text: str):
        """Same multilingual text should produce same hash."""
        h1 = simhash128(text)
        h2 = simhash128(text)
        assert h1 == h2


class TestExtractBands:
    """Test LSH band extraction."""

    def test_band_count(self):
        """Should extract correct number of bands."""
        h = simhash128("test text")
        bands = extract_bands(h)
        assert len(bands) == NUM_BANDS

    def test_band_bits_sum(self):
        """Band bits should sum to 128."""
        assert sum(BAND_BITS) == 128

    def test_zero_hash(self):
        """Zero hash should produce zero bands."""
        bands = extract_bands(0)
        assert all(b == 0 for b in bands)

    def test_deterministic(self):
        """Same hash should produce same bands."""
        h = simhash128("test text")
        bands1 = extract_bands(h)
        bands2 = extract_bands(h)
        assert bands1 == bands2


# =============================================================================
# C++ Extension Tests
# =============================================================================


@requires_cpp
class TestCppBackendActive:
    """Verify C++ backend is available and active."""

    def test_cpp_is_available(self):
        """C++ backend should be available."""
        assert is_cpp_available() is True

    def test_backend_is_cpp(self):
        """Backend should report as 'cpp'."""
        assert get_backend() == "cpp"


@requires_cpp
class TestCppPythonParity:
    """Test that C++ and Python implementations produce identical results."""

    PARITY_TEXTS = [
        # ASCII
        "The quick brown fox jumps over the lazy dog.",
        "UPPERCASE TEXT HERE",
        "mixed Case Text",
        "   whitespace   around   ",
        # Short texts
        "a",
        "ab",
        "abc",
        "abcd",
        "abcde",
        # European languages
        "Größe und Äpfel sind schön.",
        "Où est la bibliothèque?",
        "¿Cómo estás?",
        # Cyrillic
        "Привет мир, как дела?",
        "ПРИВЕТ МИР",
        # CJK
        "这是一个测试文本。",
        "これはテストです。",
        "안녕하세요",
        # Arabic/Hebrew (RTL)
        "مرحبا بالعالم",
        "שלום עולם",
        # Greek
        "Γειά σου κόσμε",
        "ΓΕΙΑ ΣΟΥ ΚΟΣΜΕ",
        # Mixed scripts
        "Hello мир 世界",
        "Test тест テスト",
        # Unicode whitespace
        "hello\u00a0world",  # NO-BREAK SPACE
        "hello\u3000world",  # IDEOGRAPHIC SPACE
        "hello\u2003world",  # EM SPACE
        # Edge cases
        "",
        " ",
        "\t\n\r",
    ]

    @pytest.mark.parametrize("text", PARITY_TEXTS)
    def test_simhash_parity(self, text: str):
        """C++ and Python should produce identical simhashes."""
        py_hash = py_simhash128(text)
        cpp_hash = fast_simhash128(text)
        assert py_hash == cpp_hash, f"Mismatch for text: {text!r}"

    @pytest.mark.parametrize("text", PARITY_TEXTS)
    def test_simhash_ngram_parity(self, text: str):
        """C++ and Python should produce identical simhashes for different ngram sizes."""
        for ngram_size in [3, 5, 7, 9]:
            py_hash = py_simhash128(text, ngram_size=ngram_size)
            cpp_hash = fast_simhash128(text, ngram_size=ngram_size)
            assert py_hash == cpp_hash, f"Mismatch for text: {text!r} with ngram_size={ngram_size}"

    @pytest.mark.parametrize("text", PARITY_TEXTS)
    def test_hamming_distance_parity(self, text: str):
        """C++ and Python hamming distance should match."""
        h1 = py_simhash128(text)
        h2 = py_simhash128(text + "x")  # Slightly different

        py_dist = py_hamming_distance(h1, h2)
        cpp_dist = fast_hamming_distance(h1, h2)
        assert py_dist == cpp_dist

    def test_hamming_distance_zero(self):
        """Identical hashes should have distance 0 in both implementations."""
        h = fast_simhash128("test text")
        assert fast_hamming_distance(h, h) == 0
        assert py_hamming_distance(h, h) == 0

    def test_hamming_distance_known_values(self):
        """Test with known bit patterns."""
        h1 = 0
        h2 = 0xFF  # 8 bits

        assert fast_hamming_distance(h1, h2) == 8
        assert py_hamming_distance(h1, h2) == 8

        h3 = (1 << 64) - 1  # 64 bits in low part
        assert fast_hamming_distance(0, h3) == 64
        assert py_hamming_distance(0, h3) == 64

    def test_are_near_duplicates_parity(self):
        """C++ and Python are_near_duplicates should match."""
        h1 = fast_simhash128("The quick brown fox")
        h2 = fast_simhash128("The quick brown cat")

        for threshold in [0, 5, 10, 20, 50]:
            py_result = py_are_near_duplicates(h1, h2, threshold=threshold)
            cpp_result = fast_are_near_duplicates(h1, h2, threshold=threshold)
            assert py_result == cpp_result, f"Mismatch at threshold={threshold}"

    def test_extract_bands_parity(self):
        """C++ and Python extract_bands should produce identical results."""
        test_hashes = [
            0,
            1,
            (1 << 128) - 1,
            fast_simhash128("test text"),
            fast_simhash128("another test"),
            0xDEADBEEFCAFEBABE1234567890ABCDEF,
        ]
        for h in test_hashes:
            py_bands = py_extract_bands(h)
            cpp_bands = fast_extract_bands(h)
            assert py_bands == cpp_bands, f"Mismatch for hash 0x{h:032x}"

    def test_multilingual_parity(self):
        """All multilingual texts should produce matching hashes."""
        for lang, text in TestMultilingual.MULTILINGUAL_TEXTS:
            py_hash = py_simhash128(text)
            cpp_hash = fast_simhash128(text)
            assert py_hash == cpp_hash, f"Mismatch for {lang}: {text!r}"


@requires_cpp
class TestCppLowEntropyFiltering:
    """Test that low-entropy ngrams are filtered identically in C++ and Python."""

    REPEATING_CHAR_TEXTS = [
        ("all_a", "aaaaaaaaaaaaaaaaaaaaaa"),
        ("all_space", "                      "),
        ("mostly_a", "aaaaabaaaaaaacaaaaaaa"),
        ("mostly_x", "xxxxxyxxxxxzxxxxx"),
        ("ab_pattern", "abababababababababab"),
        ("abc_pattern", "abcabcabcabcabcabcabc"),
        ("dashes", "----------------------"),
        ("dots", "......................"),
        ("trailing_zeros", "430000000000000000000"),
        ("underscores", "______________________"),
        ("mixed_low_entropy", "...---...---...---..."),
        ("text_with_dashes", "Hello--------------------World"),
        ("text_with_spaces", "Hello                    World"),
        ("single_char_long", "x" * 100),
        ("two_chars_alternating", "xyxyxyxyxyxyxyxyxyxy"),
    ]

    @pytest.mark.parametrize("name,text", REPEATING_CHAR_TEXTS)
    def test_cpp_python_parity_repeating_chars(self, name: str, text: str):
        """C++ and Python should produce identical results for repeating char texts."""
        py_hash = py_simhash128(text)
        cpp_hash = fast_simhash128(text)
        assert py_hash == cpp_hash, f"Mismatch for {name}: {text!r}"

    @pytest.mark.parametrize("name,text", REPEATING_CHAR_TEXTS)
    def test_deterministic_repeating_chars(self, name: str, text: str):
        """Same repeating char text should always produce same hash."""
        h1 = fast_simhash128(text)
        h2 = fast_simhash128(text)
        assert h1 == h2, f"Non-deterministic hash for {name}"

    def test_low_entropy_vs_high_entropy_different(self):
        """Low-entropy and high-entropy texts of same length should differ."""
        low_entropy = "aaaaabbbbbccccc"
        high_entropy = "abcdefghijklmno"

        h_low = fast_simhash128(low_entropy)
        h_high = fast_simhash128(high_entropy)

        assert h_low != h_high

    @pytest.mark.parametrize("ngram_size", [3, 5, 7])
    def test_parity_across_ngram_sizes_repeating(self, ngram_size: int):
        """Test C++/Python parity for repeating chars across ngram sizes."""
        texts = [
            "aaaaaaaaaaaaa",
            "abababababab",
            "Hello" + "x" * 20 + "World",
        ]
        for text in texts:
            py_hash = py_simhash128(text, ngram_size=ngram_size)
            cpp_hash = fast_simhash128(text, ngram_size=ngram_size)
            assert py_hash == cpp_hash, f"Mismatch for {text!r} with ngram_size={ngram_size}"


@requires_cpp
class TestCppPerformance:
    """Basic sanity checks that C++ is faster than Python."""

    def test_cpp_faster_than_python(self):
        """C++ should be significantly faster than Python."""
        import time

        text = "The quick brown fox jumps over the lazy dog. " * 10
        iterations = 100

        # Python timing
        start = time.perf_counter()
        for _ in range(iterations):
            py_simhash128(text)
        py_time = time.perf_counter() - start

        # C++ timing
        start = time.perf_counter()
        for _ in range(iterations):
            fast_simhash128(text)
        cpp_time = time.perf_counter() - start

        # C++ should be at least 10x faster
        speedup = py_time / cpp_time
        assert speedup > 10, f"Expected >10x speedup, got {speedup:.1f}x"
