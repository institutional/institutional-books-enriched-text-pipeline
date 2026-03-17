/**
 * c99_banded.cpp - C99-style topic segmentation with banded similarity
 *
 * Implements the C99 algorithm from "Advances in domain independent linear
 * text segmentation" (Choi, 1999) with optimizations for banded similarity
 * matrices.
 *
 * Algorithm:
 * 1. Compute embeddings for each sentence
 * 2. Compute banded similarity matrix (only near-diagonal entries)
 * 3. Replace similarity values with ranks in local neighborhoods
 * 4. Identify subsegments with high local density via greedy splitting
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace py = pybind11;

namespace c99 {

struct BandRow {
    int j_start;
    std::vector<float> sims;
};

static inline float dot_product(const float* a, const float* b, int dim) {
    float s = 0.0f;
    for (int k = 0; k < dim; ++k) {
        s += a[k] * b[k];
    }
    return s;
}

static inline void normalize_rows(std::vector<float>& V, int N, int dim) {
    for (int i = 0; i < N; ++i) {
        float* row = &V[i * dim];
        double ss = 0.0;
        for (int k = 0; k < dim; ++k) {
            double v = row[k];
            ss += v * v;
        }
        float norm = static_cast<float>(std::sqrt(ss) + 1e-12);
        float inv = 1.0f / norm;
        for (int k = 0; k < dim; ++k) {
            row[k] *= inv;
        }
    }
}

static std::vector<BandRow> build_banded_similarity(
    const std::vector<float>& V,
    int N,
    int dim,
    int band_width
) {
    std::vector<BandRow> band_sims;
    band_sims.resize(N);

    for (int i = 0; i < N; ++i) {
        int j_start = std::max(0, i - band_width);
        int j_end   = std::min(N - 1, i + band_width);
        int width   = j_end - j_start + 1;

        BandRow& br = band_sims[i];
        br.j_start = j_start;
        br.sims.resize(width);

        const float* vi = &V[i * dim];
        for (int offset = 0; offset < width; ++offset) {
            int j = j_start + offset;
            const float* vj = &V[j * dim];
            br.sims[offset] = dot_product(vi, vj, dim);
        }
    }
    return band_sims;
}

static void compute_banded_rank_matrix(
    const std::vector<BandRow>& band_sims,
    int N,
    int mask_size,
    std::vector<float>& R_full,
    std::vector<float>& valid_full
) {
    const int half = mask_size / 2;

    auto idx = [N](int i, int j) {
        return i * N + j;
    };

    for (int i = 0; i < N; ++i) {
        int j_start_i = band_sims[i].j_start;
        const std::vector<float>& sims_i = band_sims[i].sims;
        int width_i = static_cast<int>(sims_i.size());
        int j_end_i = j_start_i + width_i - 1;

        for (int offset = 0; offset < width_i; ++offset) {
            int j = j_start_i + offset;
            float center_val = sims_i[offset];

            int i_min = std::max(0, i - half);
            int i_max = std::min(N - 1, i + half);
            int j_min = j - half;
            int j_max = j + half;

            int neigh_count = 0;
            int lower_count = 0;

            for (int ni = i_min; ni <= i_max; ++ni) {
                int j_start_n = band_sims[ni].j_start;
                const std::vector<float>& sims_n = band_sims[ni].sims;
                int width_n = static_cast<int>(sims_n.size());
                int j_end_n = j_start_n + width_n - 1;

                int col_min = std::max(j_min, j_start_n);
                int col_max = std::min(j_max, j_end_n);
                if (col_min > col_max) continue;

                for (int nj = col_min; nj <= col_max; ++nj) {
                    if (ni == i && nj == j) continue;
                    ++neigh_count;
                    float sim_val = sims_n[nj - j_start_n];
                    if (sim_val < center_val) {
                        ++lower_count;
                    }
                }
            }

            float rank_val = 0.0f;
            if (neigh_count > 0) {
                rank_val = static_cast<float>(lower_count) / neigh_count;
            }

            R_full[idx(i, j)]    = rank_val;
            valid_full[idx(i, j)] = 1.0f;
        }
    }
}

static void prefix_sum_2d_inplace(std::vector<float>& A, int N) {
    // cumulative over rows
    for (int i = 1; i < N; ++i) {
        float* cur  = &A[i * N];
        float* prev = &A[(i - 1) * N];
        for (int j = 0; j < N; ++j) {
            cur[j] += prev[j];
        }
    }
    // cumulative over cols
    for (int i = 0; i < N; ++i) {
        float* row = &A[i * N];
        for (int j = 1; j < N; ++j) {
            row[j] += row[j - 1];
        }
    }
}

static inline double block_sum_prefix(
    const std::vector<float>& prefix,
    int N,
    int s,
    int e
) {
    auto at = [N, &prefix](int i, int j) -> double {
        return static_cast<double>(prefix[i * N + j]);
    };

    int i1 = s, j1 = s;
    int i2 = e, j2 = e;

    double total = at(i2, j2);
    if (i1 > 0) total -= at(i1 - 1, j2);
    if (j1 > 0) total -= at(i2, j1 - 1);
    if (i1 > 0 && j1 > 0) total += at(i1 - 1, j1 - 1);
    return total;
}

/**
 * Segment sentences using C99 banded algorithm.
 *
 * @param embeddings Numpy array of shape (N, dim), row-major float32
 * @param band_width Width of band around diagonal for similarity
 * @param mask_size Size of neighborhood mask for rank computation
 * @param min_seg_len Minimum sentences per segment
 * @param max_segments Maximum number of segments (0 = no limit)
 * @param density_gain_threshold Minimum gain to continue splitting
 *
 * @return List of (start, end) tuples for each segment
 */
std::vector<std::pair<int, int>> c99_segment_banded(
    py::array_t<float> embeddings,
    int band_width = 80,
    int mask_size = 9,
    int min_seg_len = 3,
    int max_segments = 0,
    float density_gain_threshold = 0.0f
) {
    // Directly share C++ memory buffer with python. This is speed.
    // This is also my first time using this - DLD
    py::buffer_info buf = embeddings.request();

    if (buf.ndim != 2) {
        throw std::runtime_error("Embeddings must be 2D array");
    }

    int N = static_cast<int>(buf.shape[0]);
    int dim = static_cast<int>(buf.shape[1]);
    const float* data = static_cast<const float*>(buf.ptr);

    // Handle edge cases
    if (N <= 0 || dim <= 0) {
        return {};
    }
    if (mask_size <= 0 || mask_size % 2 == 0) {
        throw std::runtime_error("mask_size must be positive odd integer");
    }
    if (band_width < 0) {
        band_width = 0;
    }
    if (min_seg_len < 1) {
        min_seg_len = 1;
    }
    int maxSegs = (max_segments <= 0 ? N : max_segments);

    // 1. Copy embeddings and L2-normalize rows
    std::vector<float> V(N * dim);
    std::memcpy(V.data(), data, sizeof(float) * N * dim);
    normalize_rows(V, N, dim);

    // 2. Build band-limited similarity representation
    std::vector<BandRow> band_sims = build_banded_similarity(V, N, dim, band_width);

    // 3. Rank matrix and validity mask
    std::vector<float> R_full(N * N, 0.0f);
    std::vector<float> valid_full(N * N, 0.0f);
    compute_banded_rank_matrix(band_sims, N, mask_size, R_full, valid_full);

    // 4. Prefix sums for fast block queries
    // (Another very very very leetcode problem)
    prefix_sum_2d_inplace(R_full, N);
    prefix_sum_2d_inplace(valid_full, N);

    /*
     * Modern CPP compiler is required almost exclusively for this lambda
     * treatment.
     */
    auto block_sum = [&](int s, int e) -> double {
        return block_sum_prefix(R_full, N, s, e);
    };
    auto block_count = [&](int s, int e) -> double {
        return block_sum_prefix(valid_full, N, s, e);
    };

    // 5. Initial segmentation: one segment [0, N-1]
    std::vector<std::pair<int,int>> segments;
    segments.emplace_back(0, N - 1);

    double total_sum   = block_sum(0, N - 1);
    double total_count = block_count(0, N - 1);
    if (total_count <= 0.0) {
        return {{0, N - 1}};
    }
    double density_current = total_sum / total_count;

    // 6. Greedy divisive clustering
    while (true) {
        if (static_cast<int>(segments.size()) >= maxSegs) {
            break;
        }

        double best_gain = 0.0;
        int best_seg_idx = -1;
        int best_k       = -1;
        double best_new_total_sum   = 0.0;
        double best_new_total_count = 0.0;

        for (int seg_idx = 0; seg_idx < (int)segments.size(); ++seg_idx) {
            int s = segments[seg_idx].first;
            int e = segments[seg_idx].second;
            int seg_len = e - s + 1;
            if (seg_len < 2 * min_seg_len) {
                continue;
            }

            double old_sum   = block_sum(s, e);
            double old_count = block_count(s, e);

            int k_start = s + min_seg_len - 1;
            int k_end   = e - min_seg_len;
            for (int k = k_start; k <= k_end; ++k) {
                double sum_left   = block_sum(s, k);
                double cnt_left   = block_count(s, k);
                double sum_right  = block_sum(k + 1, e);
                double cnt_right  = block_count(k + 1, e);

                double new_total_sum   = total_sum - old_sum + sum_left + sum_right;
                double new_total_count = total_count - old_count + cnt_left + cnt_right;
                if (new_total_count <= 0.0) continue;

                double new_density = new_total_sum / new_total_count;
                double gain = new_density - density_current;
                if (gain > best_gain) {
                    best_gain = gain;
                    best_seg_idx = seg_idx;
                    best_k = k;
                    best_new_total_sum   = new_total_sum;
                    best_new_total_count = new_total_count;
                }
            }
        }

        if (best_seg_idx < 0 || best_gain <= static_cast<double>(density_gain_threshold)) {
            break;
        }

        // Commit best split
        int s = segments[best_seg_idx].first;
        int e = segments[best_seg_idx].second;
        segments[best_seg_idx] = std::make_pair(s, best_k);
        segments.insert(segments.begin() + best_seg_idx + 1,
                        std::make_pair(best_k + 1, e));

        total_sum   = best_new_total_sum;
        total_count = best_new_total_count;
        density_current = total_sum / total_count;
    }

    return segments;
}

} // namespace c99

PYBIND11_MODULE(_c99_cpp, m) {
    m.doc() = "C99 topic segmentation algorithm with banded similarity";

    m.def("c99_segment_banded", &c99::c99_segment_banded,
          py::arg("embeddings"),
          py::arg("band_width") = 80,
          py::arg("mask_size") = 9,
          py::arg("min_seg_len") = 3,
          py::arg("max_segments") = 0,
          py::arg("density_gain_threshold") = 0.0f,
          R"doc(
Segment sentences using C99 banded algorithm.

Args:
    embeddings: Numpy array of shape (N, dim), float32
    band_width: Width of band around diagonal for similarity (default: 80)
    mask_size: Size of neighborhood mask for rank computation (default: 9)
    min_seg_len: Minimum sentences per segment (default: 3)
    max_segments: Maximum number of segments, 0 = no limit (default: 0)
    density_gain_threshold: Minimum gain to continue splitting (default: 0.0)

Returns:
    List of (start, end) tuples for each segment
)doc");
}
