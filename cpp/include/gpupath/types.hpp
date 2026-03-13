#pragma once
// file: cpp/include/gpupath/types.hpp

#include <vector>

namespace gpupath {
    /**
     * @brief Stores the result of a Breadth-First Search (BFS) traversal.
     *
     * Produced by BFS-based path engines. Both arrays are indexed by vertex id
     * and have length equal to the number of vertices in the traversed graph.
     *
     * Sentinel semantics:
     * - `distances[v] == -1` if vertex @p v was not reached.
     * - `predecessors[v] == -1` if vertex @p v is the source or was not reached.
     */
    struct BfsResult {
        /**
         * @brief Shortest-hop distances from the BFS source.
         *
         * `distances[v]` holds the number of edges on the shortest path
         * from the source to vertex @p v, or `-1` if @p v is unreachable.
         */
        std::vector<int> distances;

        /**
         * @brief Predecessor vertices along each shortest path.
         *
         * `predecessors[v]` holds the vertex that first discovered @p v during
         * traversal, or `-1` if @p v is the source itself or was not reached.
         */
        std::vector<int> predecessors;
    };
} // namespace gpupath
