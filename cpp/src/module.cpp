// file: cpp/src/module.cpp

#include <optional>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gpupath/bfs.hpp"
#include "gpupath/sssp.hpp"
#include "gpupath/types.hpp"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Internal bootstrap helpers
// ---------------------------------------------------------------------------

/**
 * @brief Return a human-readable version string for the native module.
 *
 * @return Version string identifying the current native bootstrap build.
 */
static std::string version() {
    return "gpupath native bootstrap v1";
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(_native, m) {
    m.doc() =
            "Native C++ backend for gpupath.\n\n"
            "This module is not intended to be imported directly by end users. "
            "Use the public Python API in ``gpupath`` instead, which dispatches "
            "to this backend internally.";

    // --- BfsResult --------------------------------------------------------

    py::class_<gpupath::BfsResult>(
                m,
                "BfsResult",
                "Result of a Breadth-First Search traversal.\n\n"
                "Both arrays are indexed by vertex id and have length equal to the "
                "number of vertices in the traversed graph. Unreachable vertices "
                "retain ``-1`` in both arrays."
            )
            .def_readonly(
                "distances",
                &gpupath::BfsResult::distances,
                "Shortest-hop distances from the BFS source. "
                "``distances[v]`` is ``-1`` if vertex ``v`` was not reached."
            )
            .def_readonly(
                "predecessors",
                &gpupath::BfsResult::predecessors,
                "Predecessor vertices along each shortest path. "
                "``predecessors[v]`` is ``-1`` if ``v`` is the source or "
                "was not reached."
            );

    // --- SsspResult ------------------------------------------------------

    py::class_<gpupath::SsspResult>(
                m,
                "SsspResult",
                "Result of a Single-Source Shortest Path traversal.\n\n"
                "Both arrays are indexed by vertex id and have length equal to the "
                "number of vertices in the traversed graph. Unreachable vertices "
                "retain ``inf`` in ``distances`` and ``-1`` in ``predecessors``."
            )
            .def_readonly(
                "distances",
                &gpupath::SsspResult::distances,
                "Minimum-cost distances from the source. "
                "``distances[v]`` is ``inf`` if vertex ``v`` was not reached."
            )
            .def_readonly(
                "predecessors",
                &gpupath::SsspResult::predecessors,
                "Predecessor vertices along each minimum-cost path. "
                "``predecessors[v]`` is ``-1`` if ``v`` is the source or "
                "was not reached."
            );

    // --- Bootstrap helpers ------------------------------------------------

    m.def(
        "version",
        &version,
        "Return the native module version string."
    );

    // --- BFS --------------------------------------------------------------

    m.def(
        "bfs_unweighted",
        &gpupath::bfs_unweighted,
        py::arg("num_vertices"),
        py::arg("indptr"),
        py::arg("indices"),
        py::arg("source"),
        "Run BFS from ``source`` on an unweighted CSR graph.\n\n"
        "Args:\n"
        "    num_vertices: Total number of vertices. Must be non-negative.\n"
        "    indptr: CSR row-pointer array of length ``num_vertices + 1``.\n"
        "    indices: Flat neighbor array. Every entry must be in "
        "``[0, num_vertices)``.\n"
        "    source: Source vertex. Must be in ``[0, num_vertices)``.\n\n"
        "Returns:\n"
        "    A :class:`BfsResult` with ``distances`` and ``predecessors`` "
        "arrays of length ``num_vertices``.\n\n"
        "Raises:\n"
        "    ValueError: If ``num_vertices`` is negative or if the CSR "
        "arrays are malformed.\n"
        "    IndexError: If ``source`` or any neighbor index is out of range."
    );

    // --- SSSP ------------------------------------------------------------

    m.def(
        "sssp",
        &gpupath::sssp,
        py::arg("num_vertices"),
        py::arg("indptr"),
        py::arg("indices"),
        py::arg("weights") = std::nullopt,
        py::arg("source"),
        "Run single-source shortest path from ``source`` on a CSR graph.\n\n"
        "Args:\n"
        "    num_vertices: Total number of vertices. Must be non-negative.\n"
        "    indptr: CSR row-pointer array of length ``num_vertices + 1``.\n"
        "    indices: Flat neighbor array. Every entry must be in "
        "``[0, num_vertices)``.\n"
        "    weights: Optional per-edge weights parallel to ``indices``. "
        "If omitted, every edge is treated as weight ``1.0``.\n"
        "    source: Source vertex. Must be in ``[0, num_vertices)``.\n\n"
        "Returns:\n"
        "    A :class:`SsspResult` with ``distances`` and ``predecessors`` "
        "arrays of length ``num_vertices``.\n\n"
        "Raises:\n"
        "    ValueError: If the CSR structure is malformed or if any weight "
        "is negative.\n"
        "    IndexError: If ``source`` or any neighbor index is out of range."
    );
}
