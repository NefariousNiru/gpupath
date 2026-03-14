// file: cpp/src/module.cpp

#include <optional>
#include <string>
#include <exception>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gpupath/bfs.hpp"
#include "gpupath/native_csr_graph.hpp"
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

/**
 * @brief Register exception translators used by the native module.
 *
 * This keeps Python-visible exception types aligned with the documented API:
 * - std::invalid_argument -> ValueError
 * - std::out_of_range -> IndexError
 *
 * Any exception types not handled here fall back to pybind11's default
 * behavior.
 *
 * @param m Pybind11 module handle.
 */
static void register_exception_translators(const py::module_ &m) {
    (void) m;

    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const std::invalid_argument &e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        } catch (const std::out_of_range &e) {
            PyErr_SetString(PyExc_IndexError, e.what());
        }
    });
}

/**
 * @brief Bind the native prepared CSR graph type.
 *
 * This type is primarily an internal/native boundary object. It is exposed to
 * Python so higher-level engine code can prepare a graph once and reuse it
 * across repeated native BFS/SSSP queries.
 *
 * @param m Pybind11 module handle.
 */
static void bind_native_csr_graph(py::module_ &m) {
    py::class_<gpupath::NativeCsrGraph>(m, "NativeCsrGraph")
            .def(
                py::init<std::size_t, std::vector<int>, std::vector<int> >(),
                py::arg("num_vertices"),
                py::arg("indptr"),
                py::arg("indices"),
                R"pbdoc(
                Construct an unweighted native CSR graph.

                Validation is performed once at construction.
                Raises ValueError if the CSR structure is invalid.
                )pbdoc"
            )
            .def(
                py::init<std::size_t, std::vector<int>, std::vector<int>, std::vector<double> >(),
                py::arg("num_vertices"),
                py::arg("indptr"),
                py::arg("indices"),
                py::arg("weights"),
                R"pbdoc(
                Construct a weighted native CSR graph.

                Validation is performed once at construction.
                Raises ValueError if the CSR structure is invalid.
                )pbdoc"
            )
            .def_property_readonly(
                "num_vertices",
                &gpupath::NativeCsrGraph::num_vertices,
                "Number of vertices in the graph."
            )
            .def_property_readonly(
                "num_edges",
                &gpupath::NativeCsrGraph::num_edges,
                "Number of edges in the graph."
            )
            .def_property_readonly(
                "is_weighted",
                &gpupath::NativeCsrGraph::is_weighted,
                "Whether the graph stores explicit edge weights."
            )
            .def_property_readonly(
                "indptr",
                &gpupath::NativeCsrGraph::indptr,
                "CSR row-pointer array."
            )
            .def_property_readonly(
                "indices",
                &gpupath::NativeCsrGraph::indices,
                "CSR column-index array."
            )
            .def_property_readonly(
                "weights",
                [](const gpupath::NativeCsrGraph &graph) -> py::object {
                    if (!graph.weights().has_value()) {
                        return py::none();
                    }
                    return py::cast(*graph.weights());
                },
                "Optional CSR edge-weight array. Returns None for unweighted graphs."
            );
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

    // --- Exception translation --------------------------------------------

    register_exception_translators(m);

    // --- Result types -----------------------------------------------------

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

    // --- Native graph boundary type --------------------------------------

    bind_native_csr_graph(m);

    // --- Bootstrap helpers ------------------------------------------------

    m.def(
        "version",
        &version,
        "Return the native module version string."
    );

    // --- BFS --------------------------------------------------------------

    m.def(
        "bfs_unweighted",
        py::overload_cast<std::size_t, const std::vector<int> &, const std::vector<int> &, int>(
            &gpupath::bfs_unweighted
        ),
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

    m.def(
        "bfs_unweighted",
        py::overload_cast<const gpupath::NativeCsrGraph &, int>(&gpupath::bfs_unweighted),
        py::arg("graph"),
        py::arg("source"),
        "Run BFS from ``source`` on a prepared native CSR graph.\n\n"
        "Args:\n"
        "    graph: Prepared native CSR graph.\n"
        "    source: Source vertex. Must be in ``[0, graph.num_vertices)``.\n\n"
        "Returns:\n"
        "    A :class:`BfsResult` with ``distances`` and ``predecessors`` arrays.\n\n"
        "Raises:\n"
        "    IndexError: If ``source`` is out of range."
    );

    // --- SSSP -------------------------------------------------------------

    m.def(
        "sssp",
        py::overload_cast<
            std::size_t,
            const std::vector<int> &,
            const std::vector<int> &,
            const std::optional<std::vector<double> > &,
            int>(&gpupath::sssp),
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

    m.def(
        "sssp",
        py::overload_cast<const gpupath::NativeCsrGraph &, int>(&gpupath::sssp),
        py::arg("graph"),
        py::arg("source"),
        "Run single-source shortest path from ``source`` on a prepared native CSR graph.\n\n"
        "Args:\n"
        "    graph: Prepared native CSR graph.\n"
        "    source: Source vertex. Must be in ``[0, graph.num_vertices)``.\n\n"
        "Returns:\n"
        "    A :class:`SsspResult` with ``distances`` and ``predecessors`` arrays.\n\n"
        "Raises:\n"
        "    ValueError: If any explicit edge weight is negative.\n"
        "    IndexError: If ``source`` is out of range."
    );
}
