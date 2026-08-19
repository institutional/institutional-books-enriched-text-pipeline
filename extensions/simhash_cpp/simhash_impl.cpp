/**
 * simhash_impl.cpp - high-performance SimHash implementation in C++
 *
 * This implements a 128-bit simhash using MurmurHash3 and character-based
 * n-grams with entropy filtering. It handles UTF-8 text and provides
 * functions for computing simhashes, Hamming distances, and LSH bands.
 */
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>
#include <utility>

// GCC/Clang popcount intrinsic (Windows/MSVC not supported)
inline uint64_t POPCOUNT64(uint64_t x) { return __builtin_popcountll(x); }

namespace idi_simhash {

// ----------------------------- MurmurHash3 ---------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the
// public domain. This is based on
// https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
// with simplifications by assuming we're working with 64-bit platforms.
// ---------------------------------------------------------------------

    inline uint64_t rotl64(uint64_t value, int8_t r) {
        return (value << r) | (value >> (64 - r));
    }

    inline uint64_t fmix64(uint64_t k) {
        k ^= k >> 33;
        k *= 0xff51afd7ed558ccdULL;
        k ^= k >> 33;
        k *= 0xc4ceb9fe1a85ec53ULL;
        k ^= k >> 33;
        return k;
    }

    inline uint64_t getblock64(const uint64_t* p, int i) {
        return p[i];
    }

    // Note that the length of the text is an int, so the maximum length is at least 2^15-1, and more likely
    // 2^31-1 on modern platforms.
    void murmurhash3_x64_128(const void* key, int len, uint32_t seed, void* out) {
        const uint8_t* data = (const uint8_t*)key;
        const int nblocks = len / 16;

        uint64_t h1 = seed;
        uint64_t h2 = seed;

        const uint64_t c1 = 0x87c37b91114253d5ULL;
        const uint64_t c2 = 0x4cf5ad432745937fULL;

        // Body
        const uint64_t* blocks = (const uint64_t*)(data);

        for (int i = 0; i < nblocks; i++) {
            uint64_t k1 = getblock64(blocks, i * 2 + 0);
            uint64_t k2 = getblock64(blocks, i * 2 + 1);

            k1 *= c1; k1 = rotl64(k1, 31); k1 *= c2; h1 ^= k1;
            h1 = rotl64(h1, 27); h1 += h2; h1 = h1 * 5 + 0x52dce729;

            k2 *= c2; k2 = rotl64(k2, 33); k2 *= c1; h2 ^= k2;
            h2 = rotl64(h2, 31); h2 += h1; h2 = h2 * 5 + 0x38495ab5;
        }

        // Tail
        const uint8_t* tail = (const uint8_t*)(data + nblocks * 16);

        uint64_t k1 = 0;
        uint64_t k2 = 0;

        switch (len & 15) {
            case 15: k2 ^= ((uint64_t)tail[14]) << 48; [[fallthrough]];
            case 14: k2 ^= ((uint64_t)tail[13]) << 40; [[fallthrough]];
            case 13: k2 ^= ((uint64_t)tail[12]) << 32; [[fallthrough]];
            case 12: k2 ^= ((uint64_t)tail[11]) << 24; [[fallthrough]];
            case 11: k2 ^= ((uint64_t)tail[10]) << 16; [[fallthrough]];
            case 10: k2 ^= ((uint64_t)tail[9]) << 8; [[fallthrough]];
            case 9:  k2 ^= ((uint64_t)tail[8]) << 0;
                     k2 *= c2; k2 = rotl64(k2, 33); k2 *= c1; h2 ^= k2;
                     [[fallthrough]];

            case 8:  k1 ^= ((uint64_t)tail[7]) << 56; [[fallthrough]];
            case 7:  k1 ^= ((uint64_t)tail[6]) << 48; [[fallthrough]];
            case 6:  k1 ^= ((uint64_t)tail[5]) << 40; [[fallthrough]];
            case 5:  k1 ^= ((uint64_t)tail[4]) << 32; [[fallthrough]];
            case 4:  k1 ^= ((uint64_t)tail[3]) << 24; [[fallthrough]];
            case 3:  k1 ^= ((uint64_t)tail[2]) << 16; [[fallthrough]];
            case 2:  k1 ^= ((uint64_t)tail[1]) << 8; [[fallthrough]];
            case 1:  k1 ^= ((uint64_t)tail[0]) << 0;
                     k1 *= c1; k1 = rotl64(k1, 31); k1 *= c2; h1 ^= k1;
        }

        // Finalization
        h1 ^= len; h2 ^= len;

        h1 += h2;
        h2 += h1;

        h1 = fmix64(h1);
        h2 = fmix64(h2);

        h1 += h2;
        h2 += h1;

        ((uint64_t*)out)[0] = h1;
        ((uint64_t*)out)[1] = h2;
    }

// ---------------------------------------------------------------------
//                      UTF-8 handling utilities
//         As we're working with multilingual text, we include
//         simple UTF-8 decoding and normalization.
// ---------------------------------------------------------------------

    /**
     * Decode a single UTF-8 code point from a string.
     * Returns the code point and advances the pointer.
     * Returns 0xFFFD (replacement character) on invalid sequences.
     *
     * NOTE: This is a permissive decoder. We do not strictly enforce
     * all UTF-8 validity rules.
     */
    inline uint32_t decode_utf8(const char*& ptr, const char* end) {
        if (ptr >= end) return 0;

        unsigned char c = static_cast<unsigned char>(*ptr++);

        // ASCII (0xxxxxxx)
        if (c < 0x80) {
            return c;
        }

        // Determine byte count and initial bits
        uint32_t codepoint;
        int remaining;

        if ((c & 0xE0) == 0xC0) {
            // 2-byte sequence (110xxxxx)
            codepoint = c & 0x1F;
            remaining = 1;
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte sequence (1110xxxx)
            codepoint = c & 0x0F;
            remaining = 2;
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte sequence (11110xxx)
            codepoint = c & 0x07;
            remaining = 3;
        } else {
            // Invalid start byte
            return 0xFFFD;
        }

        // Read continuation bytes (10xxxxxx)
        while (remaining-- > 0) {
            if (ptr >= end) return 0xFFFD;
            c = static_cast<unsigned char>(*ptr++);
            if ((c & 0xC0) != 0x80) {
                --ptr; // Put back the invalid byte
                return 0xFFFD;
            }
            codepoint = (codepoint << 6) | (c & 0x3F);
        }

        return codepoint;
    }

    /**
     * Encode a Unicode code point to UTF-8 and append to string.
     */
    inline void encode_utf8(uint32_t codepoint, std::string& out) {
        if (codepoint < 0x80) {
            out.push_back(static_cast<char>(codepoint));
        } else if (codepoint < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        } else if (codepoint < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        }
    }

    /**
     * Check if a Unicode code point is whitespace.
     * Covers ASCII whitespace and common Unicode whitespace characters.
     */
    inline bool is_unicode_whitespace(uint32_t cp) {
        // ASCII whitespace
        if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' ||
            cp == '\f' || cp == '\v') {
            return true;
        }
        // Unicode whitespace characters
        switch (cp) {
            case 0x00A0: // NO-BREAK SPACE
            case 0x2000: // EN QUAD
            case 0x2001: // EM QUAD
            case 0x2002: // EN SPACE
            case 0x2003: // EM SPACE
            case 0x2004: // THREE-PER-EM SPACE
            case 0x2005: // FOUR-PER-EM SPACE
            case 0x2006: // SIX-PER-EM SPACE
            case 0x2007: // FIGURE SPACE
            case 0x2008: // PUNCTUATION SPACE
            case 0x2009: // THIN SPACE
            case 0x200A: // HAIR SPACE
            case 0x200B: // ZERO WIDTH SPACE
            case 0x200C: // ZERO WIDTH NON-JOINER
            case 0x200D: // ZERO WIDTH JOINER
            case 0x202F: // NARROW NO-BREAK SPACE
            case 0x205F: // MEDIUM MATHEMATICAL SPACE
            case 0x3000: // IDEOGRAPHIC SPACE
            case 0xFEFF: // ZERO WIDTH NO-BREAK SPACE (BOM)
                return true;
            default:
                return false;
        }
    }

    /**
     * Simple Unicode lowercase for common languages.
     */
    inline uint32_t unicode_tolower(uint32_t cp) {
        // ASCII uppercase
        if (cp >= 'A' && cp <= 'Z') {
            return cp + 32;
        }
        // Latin-1 Supplement uppercase (À-Þ except × at 0xD7)
        if (cp >= 0xC0 && cp <= 0xDE && cp != 0xD7) {
            return cp + 32;
        }
        // Cyrillic uppercase (А-Я)
        if (cp >= 0x410 && cp <= 0x42F) {
            return cp + 32;
        }
        // Greek uppercase (Α-Ω, with gaps)
        if (cp >= 0x391 && cp <= 0x3A9 && cp != 0x3A2) {
            return cp + 32;
        }
        return cp;
    }

    /**
     * Decode UTF-8 string to vector of code points.
     */
    std::vector<uint32_t> utf8_to_codepoints(const std::string& text) {
        std::vector<uint32_t> result;
        result.reserve(text.size()); // Upper bound

        const char* ptr = text.data();
        const char* end = ptr + text.size();

        while (ptr < end) {
            result.push_back(decode_utf8(ptr, end));
        }
        return result;
    }


// ---------------------------------------------------------------------
//                      Entropy filtering
// ---------------------------------------------------------------------

    /**
     * Check if an ngram (represented as a span of codepoints) has sufficient entropy.
     * Returns true if the number of unique characters is at least threshold * length.
     */
    inline bool has_sufficient_entropy(const std::vector<uint32_t>& codepoints,
                                       int start, int length,
                                       float threshold = 0.4f) {
        // Use a simple set to count unique codepoints
        std::vector<uint32_t> seen;
        seen.reserve(length);

        // In the prototypical case, we look at 9-grams and this loop runs
        // at most 9 times. A simple vector with linear search is probably
        // faster than more complicated data structures.
        //
        // But if larger ngrams are used, this is a potential bottleneck.
        for (int i = 0; i < length; i++) {
            uint32_t cp = codepoints[start + i];
            bool found = false;
            for (uint32_t s : seen) {
                if (s == cp) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                seen.push_back(cp);
            }
        }

        return static_cast<float>(seen.size()) >= threshold * static_cast<float>(length);
    }

// ---------------------------------------------------------------------
//                      Simhash implementation
// ---------------------------------------------------------------------

    /**
     * Normalize text by lowercasing.
     * Returns the normalized string.
     *
     * NOTE: UTF8 Encoding is handled consistently but not strictly.
     */
    std::string normalize_text(const std::string& text) {
        std::vector<uint32_t> codepoints = utf8_to_codepoints(text);

        std::string result;
        result.reserve(text.size());

        for (uint32_t cp : codepoints) {
            encode_utf8(unicode_tolower(cp), result);
        }
        return result;
    }

    /**
     * Normalize text by lowercasing and removing all whitespace.
     * Returns the normalized string.
     *
     * NOTE: UTF8 Encoding is handled consistently but not strictly.
     */
    std::string normalize_text_whitespace_insensitive(const std::string& text) {
        std::vector<uint32_t> codepoints = utf8_to_codepoints(text);

        std::string result;
        result.reserve(text.size());

        for (uint32_t cp : codepoints) {
            if (!is_unicode_whitespace(cp)) {
                encode_utf8(unicode_tolower(cp), result);
            }
        }
        return result;
    }

    /**
     * Compute 128-bit simhash for the given text.
     * Returns (low_64_bits, high_64_bits).
     *
     * This uses character-based n-grams (not byte-based) and should agree with the
     * reference python implementation.
     */
    std::pair<uint64_t, uint64_t> simhash_128(const std::string& text, int ngram_size = 9) {
        std::string normalized = normalize_text(text);

        // Decode normalized text to code points for character-based n-grams
        std::vector<uint32_t> codepoints = utf8_to_codepoints(normalized);
        int num_chars = static_cast<int>(codepoints.size());

        // If the text is too short, hash the text directly
        if (num_chars < 2 * ngram_size) {
            uint64_t hash[2];
            murmurhash3_x64_128(normalized.data(), static_cast<int>(normalized.size()), 0, hash);
            return {hash[0], hash[1]};
        }

        // Weight accumulators for each bit position
        int32_t weights_lo[64] = {0};
        int32_t weights_hi[64] = {0};

        // Process each character n-gram
        int num_ngrams = num_chars - ngram_size + 1;

        for (int i = 0; i < num_ngrams; i++) {
            // Skip low-entropy ngrams (e.g., runs of the same character)
            if (!has_sufficient_entropy(codepoints, i, ngram_size)) {
                continue;
            }

            // Encode the n-gram back to UTF-8 for hashing
            std::string ngram_utf8;
            ngram_utf8.reserve(ngram_size * 4); // Max 4 bytes per char
            for (int j = 0; j < ngram_size; j++) {
                encode_utf8(codepoints[i + j], ngram_utf8);
            }

            uint64_t hash[2];
            murmurhash3_x64_128(ngram_utf8.data(), static_cast<int>(ngram_utf8.size()), 0, hash);

            uint64_t h_lo = hash[0];
            uint64_t h_hi = hash[1];

            // low bits
            for (int bit = 0; bit < 64; bit++) {
                if ((h_lo >> bit) & 1) {
                    weights_lo[bit]++;
                } else {
                    weights_lo[bit]--;
                }
            }
            // high bits
            for (int bit = 0; bit < 64; bit++) {
                if ((h_hi >> bit) & 1) {
                    weights_hi[bit]++;
                } else {
                    weights_hi[bit]--;
                }
            }
        }

        uint64_t result_lo = 0;
        uint64_t result_hi = 0;

        for (int bit = 0; bit < 64; bit++) {
            if (weights_lo[bit] >= 0) {
                result_lo |= (1ULL << bit);
            }
            if (weights_hi[bit] >= 0) {
                result_hi |= (1ULL << bit);
            }
        }

        return {result_lo, result_hi};
    }

    std::vector<std::pair<uint64_t, uint64_t>> simhash_128_batch(
        const std::vector<std::string>& texts,
        int ngram_size = 9
    ) {
        std::vector<std::pair<uint64_t, uint64_t>> results;
        results.reserve(texts.size());

        for (const auto& text : texts) {
            results.push_back(simhash_128(text, ngram_size));
        }

        return results;
    }

    //-----------------------------------------------------------------------------
    // Hamming distance
    //-----------------------------------------------------------------------------

    /**
     * Compute Hamming distance between two 128-bit hashes.
     * Uses POPCNT intrinsic for fast bit counting.
     */
    int hamming_distance(uint64_t h1_lo, uint64_t h1_hi, uint64_t h2_lo, uint64_t h2_hi) {
        return static_cast<int>(POPCOUNT64(h1_lo ^ h2_lo) + POPCOUNT64(h1_hi ^ h2_hi));
    }

    /**
     * Check if two hashes are within the specified Hamming distance threshold.
     */
    bool are_near_duplicates(uint64_t h1_lo, uint64_t h1_hi,
                             uint64_t h2_lo, uint64_t h2_hi,
                             int threshold = 5) {
        return hamming_distance(h1_lo, h1_hi, h2_lo, h2_hi) <= threshold;
    }

    //-----------------------------------------------------------------------------
    // Band extraction for LSH
    //-----------------------------------------------------------------------------

    // Split the 128 bit hash into 6 bands of approx similar size
    constexpr int BAND_BITS[6] = {22, 21, 21, 21, 21, 22};
    constexpr int NUM_BANDS = 6;

    /**
     * Extract band values from a 128-bit simhash.
     *
     * Returns array of 6 band values.
     */
    std::vector<uint32_t> extract_bands(uint64_t hash_lo, uint64_t hash_hi) {
        std::vector<uint32_t> bands(NUM_BANDS);

        int offset = 0;

        for (int i = 0; i < NUM_BANDS; i++) {
            int bits = BAND_BITS[i];
            uint32_t mask = (1U << bits) - 1;

            if (offset < 64) {
                if (offset + bits <= 64) {
                    // Entirely in low part
                    bands[i] = static_cast<uint32_t>((hash_lo >> offset) & mask);
                } else {
                    // Spans low and high
                    int bits_from_lo = 64 - offset;
                    int bits_from_hi = bits - bits_from_lo;
                    uint32_t lo_part = static_cast<uint32_t>(hash_lo >> offset);
                    uint32_t hi_part = static_cast<uint32_t>(hash_hi & ((1ULL << bits_from_hi) - 1));
                    bands[i] = (lo_part | (hi_part << bits_from_lo)) & mask;
                }
            } else {
                // Entirely in high part
                int hi_offset = offset - 64;
                bands[i] = static_cast<uint32_t>((hash_hi >> hi_offset) & mask);
            }
            offset += bits;
        }

        return bands;
    }

    //-----------------------------------------------------------------------------
    // Bucket processing for deduplication
    //-----------------------------------------------------------------------------

    /**
     * Process a single bucket: generate all pairs and return those within threshold.
     *
     * @param doc_indices  Vector of document indices in this bucket
     * @param hashes       Pointer to hash array (2 uint64_t values per doc: lo, hi)
     * @param threshold    Maximum Hamming distance to consider a duplicate
     * @param max_bucket_size  Skip buckets larger than this
     * @return Vector of (d1, d2) pairs that are duplicates
     */
    std::vector<std::pair<int64_t, int64_t>> process_bucket(
        const std::vector<int64_t>& doc_indices,
        const uint64_t* hashes,
        int threshold,
        int max_bucket_size
    ) {
        std::vector<std::pair<int64_t, int64_t>> verified;

        if (static_cast<int>(doc_indices.size()) > max_bucket_size) {
            return verified;  // Skip oversized buckets
        }

        // Sort indices for consistent ordering
        std::vector<int64_t> sorted_indices = doc_indices;
        std::sort(sorted_indices.begin(), sorted_indices.end());

        int n = static_cast<int>(sorted_indices.size());
        for (int i = 0; i < n; i++) {
            int64_t d1 = sorted_indices[i];
            uint64_t h1_lo = hashes[d1 * 2];
            uint64_t h1_hi = hashes[d1 * 2 + 1];

            for (int j = i + 1; j < n; j++) {
                int64_t d2 = sorted_indices[j];
                uint64_t h2_lo = hashes[d2 * 2];
                uint64_t h2_hi = hashes[d2 * 2 + 1];

                if (hamming_distance(h1_lo, h1_hi, h2_lo, h2_hi) <= threshold) {
                    verified.emplace_back(d1, d2);
                }
            }
        }

        return verified;
    }

    /**
     * Process a batch of buckets.
     *
     * @param buckets      Vector of buckets, each bucket is a vector of doc indices
     * @param hashes       Pointer to hash array (2 uint64_t values per doc: lo, hi)
     * @param threshold    Maximum Hamming distance to consider a duplicate
     * @param max_bucket_size  Skip buckets larger than this
     * @return Vector of (d1, d2) pairs that are duplicates
     */
    std::vector<std::pair<int64_t, int64_t>> process_bucket_batch(
        const std::vector<std::vector<int64_t>>& buckets,
        const uint64_t* hashes,
        int threshold,
        int max_bucket_size
    ) {
        std::vector<std::pair<int64_t, int64_t>> all_verified;

        for (const auto& bucket : buckets) {
            auto verified = process_bucket(bucket, hashes, threshold, max_bucket_size);
            all_verified.insert(all_verified.end(), verified.begin(), verified.end());
        }

        return all_verified;
    }

} // namespace idi_simhash
