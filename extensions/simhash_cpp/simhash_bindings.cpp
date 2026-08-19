/**
 * simhash_bindings.cpp - pybind11 bindings for simhash_impl.cpp
 *
 * Exposes the C++ simhash implementation to Python with proper
 * handling of 128-bit integers.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <string>
#include <vector>
#include <utility>

namespace py = pybind11;

namespace idi_simhash {
   std::string normalize_text(const std::string& text);
   std::pair<uint64_t, uint64_t> simhash_128(const std::string& text, int ngram_size);
   std::vector<std::pair<uint64_t, uint64_t>> simhash_128_batch(const std::vector<std::string>& texts, int ngram_size);
   int hamming_distance(uint64_t h1_lo, uint64_t h1_hi, uint64_t h2_lo, uint64_t h2_hi);
   bool are_near_duplicates(uint64_t h1_lo, uint64_t h1_hi, uint64_t h2_lo, uint64_t h2_hi, int threshold);
   std::vector<uint32_t> extract_bands(uint64_t hash_lo, uint64_t hash_hi);
   std::vector<std::pair<int64_t, int64_t>> process_bucket(const std::vector<int64_t>& doc_indices, const uint64_t* hashes, int threshold, int max_bucket_size);
   std::vector<std::pair<int64_t, int64_t>> process_bucket_batch(const std::vector<std::vector<int64_t>>& buckets, const uint64_t* hashes, int threshold, int max_bucket_size);
}

/**
 * Convert (lo, hi) pair to single python int
 */
py::object pair_to_int128(uint64_t lo, uint64_t hi) {
   py::object lo_obj = py::cast(lo);
   py::object hi_obj = py::cast(hi);
   py::object shift = py::int_(64);
   return (hi_obj << shift) | lo_obj;
}

/**
 * Convert single python int to (lo, hi) pair
 */
std::pair<uint64_t, uint64_t> int128_to_pair(py::object value) {
   // Use python's builtin int operations
   py::module_ builtins = py::module_::import("builtins");
   py::object mask = builtins.attr("int")("0xFFFFFFFFFFFFFFFF", 16);

   py::object lo_obj = value & mask;
   py::object hi_obj = value >> py::int_(64);

   uint64_t lo = PyLong_AsUnsignedLongLong(lo_obj.ptr());
   uint64_t hi = PyLong_AsUnsignedLongLong(hi_obj.ptr());

   return {lo, hi};
}

/**
 * Compute simhash and return as python int
 */
py::object py_simhash_128(const std::string & text, int ngram_size = 5) {
   auto [lo, hi] = idi_simhash::simhash_128(text, ngram_size);
   return pair_to_int128(lo, hi);
}

/**
 * Batch compute simhashes
 */
py::list py_simhash_128_batch(const std::vector<std::string>& texts, int ngram_size = 5) {
   auto hash_pairs = idi_simhash::simhash_128_batch(texts, ngram_size);
   py::list result;
   for (const auto& [lo, hi] : hash_pairs) {
       result.append(pair_to_int128(lo, hi));
   }
   return result;
}

int py_hamming_distance(py::object h1, py::object h2) {
   auto [h1_lo, h1_hi] = int128_to_pair(h1);
   auto [h2_lo, h2_hi] = int128_to_pair(h2);
   return idi_simhash::hamming_distance(h1_lo, h1_hi, h2_lo, h2_hi);
}

bool py_are_near_duplicates(py::object h1, py::object h2, int threshold = 5) {
   auto [h1_lo, h1_hi] = int128_to_pair(h1);
   auto [h2_lo, h2_hi] = int128_to_pair(h2);
   return idi_simhash::are_near_duplicates(h1_lo, h1_hi, h2_lo, h2_hi, threshold);
}

std::vector<uint32_t> py_extract_bands(py::object hash_value) {
   auto [hash_lo, hash_hi] = int128_to_pair(hash_value);
   return idi_simhash::extract_bands(hash_lo, hash_hi);
}

/**
 * Process a single bucket using a buffer of hashes.
 * The buffer should be an array of uint64_t values (2 per document: lo, hi).
 */
py::list py_process_bucket(
    const std::vector<int64_t>& doc_indices,
    py::buffer hashes_buffer,
    int threshold,
    int max_bucket_size
) {
    py::buffer_info info = hashes_buffer.request();

    if (info.format != py::format_descriptor<uint64_t>::format()) {
        throw std::runtime_error("Hash buffer must be uint64 array");
    }

    const uint64_t* hashes = static_cast<const uint64_t*>(info.ptr);
    auto verified = idi_simhash::process_bucket(doc_indices, hashes, threshold, max_bucket_size);

    py::list result;
    for (const auto& [d1, d2] : verified) {
        result.append(py::make_tuple(d1, d2));
    }
    return result;
}

/**
 * Process a batch of buckets using a buffer of hashes.
 */
py::list py_process_bucket_batch(
    const std::vector<std::vector<int64_t>>& buckets,
    py::buffer hashes_buffer,
    int threshold,
    int max_bucket_size
) {
    py::buffer_info info = hashes_buffer.request();

    if (info.format != py::format_descriptor<uint64_t>::format()) {
        throw std::runtime_error("Hash buffer must be uint64 array");
    }

    const uint64_t* hashes = static_cast<const uint64_t*>(info.ptr);
    auto verified = idi_simhash::process_bucket_batch(buckets, hashes, threshold, max_bucket_size);

    py::list result;
    for (const auto& [d1, d2] : verified) {
        result.append(py::make_tuple(d1, d2));
    }
    return result;
}

PYBIND11_MODULE(_simhash_cpp, m) {
   m.doc() = "High-performance simhash implementation in C++";

   m.def("simhash_128", &py_simhash_128,
         py::arg("text"),
         py::arg("ngram_size") = 5,
         "Compute 128-bit simhash for text, returns Python int. The text has a max length of at least 32767 characters.");

   m.def("simhash_128_batch", &py_simhash_128_batch,
         py::arg("texts"),
         py::arg("ngram_size") = 5,
         "Batch compute simhashes for multiple texts");

   m.def("hamming_distance", &py_hamming_distance,
         py::arg("h1"),
         py::arg("h2"),
         "Compute Hamming distance between two 128-bit simhashes");

   m.def("are_near_duplicates", &py_are_near_duplicates,
         py::arg("h1"),
         py::arg("h2"),
         py::arg("threshold") = 5,
         "Check if two simhashes are within Hamming distance threshold");

   m.def("extract_bands", &py_extract_bands,
         py::arg("hash_value"),
         "Extract 6 band values from a 128-bit simhash");

   m.def("normalize_text", &idi_simhash::normalize_text,
         py::arg("text"),
         "Normalize text by lowercasing");

   m.def("process_bucket", &py_process_bucket,
         py::arg("doc_indices"),
         py::arg("hashes_buffer"),
         py::arg("threshold"),
         py::arg("max_bucket_size"),
         "Process a single bucket and return duplicate pairs");

   m.def("process_bucket_batch", &py_process_bucket_batch,
         py::arg("buckets"),
         py::arg("hashes_buffer"),
         py::arg("threshold"),
         py::arg("max_bucket_size"),
         "Process a batch of buckets and return duplicate pairs");
}
