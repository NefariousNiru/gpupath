// file: multi_source_lengths.cpp

#include "gpupath/multi_source_lengths.hpp"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

#include "gpupath/bfs.hpp"
#include "gpupath/sssp.hpp"
#include "gpupath/types.hpp"

namespace gpupath {
    namespace {
        /**
         * @brief Resolve an execution thread count from a user request.
         *
         * @param requested Requested thread count. `<= 0` means use a default.
         * @param work_items Number of independent work items.
         *
         * @return Thread count in `[1, work_items]` when work is non-empty, otherwise 1.
         */
        int resolve_thread_count(const int requested, const std::size_t work_items) {
            if (work_items == 0U) {
                return 1;
            }

            if (requested > 0) {
                return std::min<int>(requested, static_cast<int>(work_items));
            }

            const unsigned int hw = std::thread::hardware_concurrency();
            const int fallback = (hw == 0U) ? 1 : static_cast<int>(hw);
            return std::min<int>(fallback, static_cast<int>(work_items));
        }

        /**
         * @brief Validate vertex ids against graph bounds.
         *
         * @param num_vertices Number of vertices in the graph.
         * @param vertices Vertex ids to validate.
         * @param what Human-readable label for exception messages.
         *
         * @throws std::out_of_range If any vertex id is outside the valid range.
         */
        void validate_vertices(
            const std::size_t num_vertices,
            const std::vector<int> &vertices,
            const char *what
        ) {
            for (const int vertex: vertices) {
                if (vertex < 0 || static_cast<std::size_t>(vertex) >= num_vertices) {
                    throw std::out_of_range(std::string(what) + " vertex out of range");
                }
            }
        }

        /**
         * @brief Materialize a filtered row from a full distance vector.
         *
         * @tparam T Distance scalar type.
         * @param distances Full per-vertex distance vector.
         * @param targets Optional target subset.
         *
         * @return Row containing either all distances or the requested target subset.
         */
        template<typename T>
        std::vector<T> select_targets(
            const std::vector<T> &distances,
            const std::optional<std::vector<int> > &targets
        ) {
            if (!targets.has_value()) {
                return distances;
            }

            std::vector<T> row;
            row.reserve(targets->size());
            for (const int target: *targets) {
                row.push_back(distances[static_cast<std::size_t>(target)]);
            }
            return row;
        }

        /**
         * @brief Parallel-for helper over contiguous source index ranges.
         *
         * Executes @p fn over contiguous half-open ranges `[begin, end)`. Any
         * exception thrown by a worker thread is captured and rethrown on the
         * calling thread after all workers have been joined.
         *
         * @tparam Fn Callable type taking `(std::size_t begin, std::size_t end)`.
         * @param num_items Number of work items.
         * @param num_threads Requested thread count.
         * @param fn Range worker.
         *
         * @throws Re-throws the first exception captured from any worker.
         */
        template<typename Fn>
        void parallel_for_ranges(
            const std::size_t num_items,
            const int num_threads,
            Fn &&fn
        ) {
            const int threads = resolve_thread_count(num_threads, num_items);

            if (threads <= 1 || num_items <= 1U) {
                fn(0U, num_items);
                return;
            }

            std::vector<std::thread> workers;
            workers.reserve(static_cast<std::size_t>(threads - 1));

            std::exception_ptr first_exception = nullptr;
            std::mutex exception_mutex;

            const std::size_t base = num_items / static_cast<std::size_t>(threads);
            const std::size_t extra = num_items % static_cast<std::size_t>(threads);

            auto guarded_fn = [&](const std::size_t begin, const std::size_t end) {
                try {
                    fn(begin, end);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(exception_mutex);
                    if (!first_exception) {
                        first_exception = std::current_exception();
                    }
                }
            };

            std::size_t begin = 0U;
            for (int tid = 0; tid < threads - 1; ++tid) {
                const std::size_t span =
                        base + (static_cast<std::size_t>(tid) < extra ? 1U : 0U);
                const std::size_t end = begin + span;

                workers.emplace_back([begin, end, &guarded_fn]() {
                    guarded_fn(begin, end);
                });

                begin = end;
            }

            guarded_fn(begin, num_items);

            for (auto &worker: workers) {
                worker.join();
            }

            if (first_exception) {
                std::rethrow_exception(first_exception);
            }
        }
    } // namespace

    std::vector<std::vector<int> > multi_source_bfs_lengths(
        const NativeCsrGraph &graph,
        const std::vector<int> &sources,
        const std::optional<std::vector<int> > &targets,
        const int num_threads
    ) {
        if (sources.empty()) {
            return {};
        }

        validate_vertices(graph.num_vertices(), sources, "Source");
        if (targets.has_value()) {
            validate_vertices(graph.num_vertices(), *targets, "Target");
        }

        std::vector<std::vector<int> > matrix(sources.size());

        parallel_for_ranges(
            sources.size(),
            num_threads,
            [&](const std::size_t begin, const std::size_t end) {
                for (std::size_t i = begin; i < end; ++i) {
                    const BfsResult result = bfs_unweighted(graph, sources[i]);
                    matrix[i] = select_targets(result.distances, targets);
                }
            }
        );

        return matrix;
    }

    std::vector<std::vector<double> > multi_source_sssp_lengths(
        const NativeCsrGraph &graph,
        const std::vector<int> &sources,
        const std::optional<std::vector<int> > &targets,
        const int num_threads
    ) {
        if (sources.empty()) {
            return {};
        }

        validate_vertices(graph.num_vertices(), sources, "Source");
        if (targets.has_value()) {
            validate_vertices(graph.num_vertices(), *targets, "Target");
        }

        std::vector<std::vector<double> > matrix(sources.size());

        parallel_for_ranges(
            sources.size(),
            num_threads,
            [&](const std::size_t begin, const std::size_t end) {
                for (std::size_t i = begin; i < end; ++i) {
                    const SsspResult result = sssp(graph, sources[i]);
                    matrix[i] = select_targets(result.distances, targets);
                }
            }
        );

        return matrix;
    }

    std::vector<std::vector<int> > multi_source_bfs_lengths(
        const std::size_t num_vertices,
        const std::vector<int> &indptr,
        const std::vector<int> &indices,
        const std::vector<int> &sources,
        const std::optional<std::vector<int> > &targets,
        const int num_threads
    ) {
        const NativeCsrGraph graph(num_vertices, indptr, indices);
        return multi_source_bfs_lengths(graph, sources, targets, num_threads);
    }

    std::vector<std::vector<double> > multi_source_sssp_lengths(
        const std::size_t num_vertices,
        const std::vector<int> &indptr,
        const std::vector<int> &indices,
        const std::optional<std::vector<double> > &weights,
        const std::vector<int> &sources,
        const std::optional<std::vector<int> > &targets,
        const int num_threads
    ) {
        const NativeCsrGraph graph =
                weights.has_value()
                    ? NativeCsrGraph(num_vertices, indptr, indices, *weights)
                    : NativeCsrGraph(num_vertices, indptr, indices);

        return multi_source_sssp_lengths(graph, sources, targets, num_threads);
    }
} // namespace gpupath
