/*
 * Python bindings for OpenOrbitalOptimizer using nanobind.
 *
 * Wraps SCFSolver<double, double> for use with PySCF.
 * Handles numpy <-> Armadillo conversion and GIL management.
 */

#include <memory>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/function.h>

#include <openorbitaloptimizer/scfsolver.hpp>

namespace nb = nanobind;
using namespace nb::literals;

using NpArray1D = nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using NpArray2D = nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

// --- Conversion helpers ---

static arma::mat numpy_to_arma_mat(NpArray2D arr) {
    size_t rows = arr.shape(0);
    size_t cols = arr.shape(1);
    // Armadillo is column-major, numpy (C-contiguous) is row-major
    arma::mat result(rows, cols);
    const double* data = arr.data();
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            result(i, j) = data[i * cols + j];
    return result;
}

static nb::ndarray<nb::numpy, double, nb::ndim<2>> arma_mat_to_numpy(const arma::mat& mat) {
    size_t rows = mat.n_rows;
    size_t cols = mat.n_cols;
    // Copy to row-major (C-contiguous) layout
    double* data = new double[rows * cols];
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            data[i * cols + j] = mat(i, j);

    nb::capsule owner(data, [](void* ptr) noexcept {
        delete[] static_cast<double*>(ptr);
    });
    size_t shape[2] = {rows, cols};
    return nb::ndarray<nb::numpy, double, nb::ndim<2>>(data, 2, shape, owner);
}

static arma::vec numpy_to_arma_vec(NpArray1D arr) {
    size_t n = arr.shape(0);
    arma::vec result(n);
    const double* data = arr.data();
    for (size_t i = 0; i < n; i++)
        result(i) = data[i];
    return result;
}

static nb::ndarray<nb::numpy, double, nb::ndim<1>> arma_vec_to_numpy(const arma::vec& vec) {
    size_t n = vec.n_elem;
    double* data = new double[n];
    std::memcpy(data, vec.memptr(), n * sizeof(double));

    nb::capsule owner(data, [](void* ptr) noexcept {
        delete[] static_cast<double*>(ptr);
    });
    size_t shape[1] = {n};
    return nb::ndarray<nb::numpy, double, nb::ndim<1>>(data, 1, shape, owner);
}

static arma::uvec numpy_to_arma_uvec(nb::ndarray<uint64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> arr) {
    size_t n = arr.shape(0);
    arma::uvec result(n);
    const uint64_t* data = arr.data();
    for (size_t i = 0; i < n; i++)
        result(i) = data[i];
    return result;
}

// Convert Python list of numpy arrays to vector of arma::mat
static std::vector<arma::mat> py_list_to_arma_mats(nb::list lst) {
    std::vector<arma::mat> result;
    result.reserve(nb::len(lst));
    for (size_t i = 0; i < nb::len(lst); i++) {
        auto arr = nb::cast<NpArray2D>(lst[i]);
        result.push_back(numpy_to_arma_mat(arr));
    }
    return result;
}

// Convert vector of arma::mat to Python list of numpy arrays
static nb::list arma_mats_to_py_list(const std::vector<arma::mat>& mats) {
    nb::list result;
    for (const auto& mat : mats)
        result.append(arma_mat_to_numpy(mat));
    return result;
}

// Convert Python list of numpy 1D arrays to vector of arma::vec
static std::vector<arma::vec> py_list_to_arma_vecs(nb::list lst) {
    std::vector<arma::vec> result;
    result.reserve(nb::len(lst));
    for (size_t i = 0; i < nb::len(lst); i++) {
        auto arr = nb::cast<NpArray1D>(lst[i]);
        result.push_back(numpy_to_arma_vec(arr));
    }
    return result;
}

// Convert vector of arma::vec to Python list of numpy 1D arrays
static nb::list arma_vecs_to_py_list(const std::vector<arma::vec>& vecs) {
    nb::list result;
    for (const auto& vec : vecs)
        result.append(arma_vec_to_numpy(vec));
    return result;
}

// --- Wrapper class ---

class SCFSolverWrapper {
    std::unique_ptr<OpenOrbitalOptimizer::SCFSolver<double, double>> solver_;
    nb::callable py_fock_builder_;
    nb::callable py_callback_;
    nb::callable py_convergence_callback_;

    // Wrap the Python Fock builder into a C++ std::function
    OpenOrbitalOptimizer::FockBuilder<double, double> make_fock_builder() {
        nb::callable& fb = py_fock_builder_;
        return [&fb](const OpenOrbitalOptimizer::DensityMatrix<double, double>& dm)
            -> OpenOrbitalOptimizer::FockBuilderReturn<double, double>
        {
            nb::gil_scoped_acquire gil;

            // Convert orbitals (vector<arma::mat>) to list of numpy arrays
            nb::list orbitals_py = arma_mats_to_py_list(dm.first);

            // Convert occupations (vector<arma::vec>) to list of numpy arrays
            nb::list occupations_py = arma_vecs_to_py_list(dm.second);

            // Call the Python function: returns (energy, [fock_matrices])
            nb::object result = fb(orbitals_py, occupations_py);

            // Parse the result
            nb::tuple result_tuple = nb::cast<nb::tuple>(result);
            double energy = nb::cast<double>(result_tuple[0]);
            nb::list fock_py = nb::cast<nb::list>(result_tuple[1]);
            auto fock_matrices = py_list_to_arma_mats(fock_py);

            return std::make_pair(energy, fock_matrices);
        };
    }

public:
    SCFSolverWrapper(
        nb::ndarray<uint64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> nblocks_per_type,
        NpArray1D max_occupation,
        NpArray1D nparticles,
        nb::callable fock_builder,
        std::vector<std::string> block_descriptions
    ) : py_fock_builder_(std::move(fock_builder))
    {
        auto nblocks = numpy_to_arma_uvec(nblocks_per_type);
        auto max_occ = numpy_to_arma_vec(max_occupation);
        auto nparts = numpy_to_arma_vec(nparticles);

        solver_ = std::make_unique<OpenOrbitalOptimizer::SCFSolver<double, double>>(
            nblocks, max_occ, nparts, make_fock_builder(), block_descriptions
        );
    }

    void initialize_with_fock(nb::list fock_guess_py) {
        auto fock_guess = py_list_to_arma_mats(fock_guess_py);
        solver_->initialize_with_fock(fock_guess);
    }

    void initialize_with_orbitals(nb::list orbitals_py, nb::list occupations_py) {
        auto orbitals = py_list_to_arma_mats(orbitals_py);
        auto occupations = py_list_to_arma_vecs(occupations_py);
        solver_->initialize_with_orbitals(orbitals, occupations);
    }

    void run() {
        nb::gil_scoped_release release;
        solver_->run();
    }

    double get_energy() const {
        return solver_->get_energy();
    }

    nb::tuple get_solution() const {
        auto solution = solver_->get_solution();
        return nb::make_tuple(
            arma_mats_to_py_list(solution.first),
            arma_vecs_to_py_list(solution.second)
        );
    }

    nb::list get_orbitals() const {
        return arma_mats_to_py_list(solver_->get_orbitals());
    }

    nb::list get_orbital_occupations() const {
        return arma_vecs_to_py_list(solver_->get_orbital_occupations());
    }

    nb::list get_fock_matrix() const {
        return arma_mats_to_py_list(solver_->get_fock_matrix());
    }

    // --- Setters / getters ---
    void set_convergence_threshold(double val) { solver_->convergence_threshold(val); }
    double get_convergence_threshold() const { return solver_->convergence_threshold(); }

    void set_maximum_iterations(size_t val) { solver_->maximum_iterations(val); }
    size_t get_maximum_iterations() const { return solver_->maximum_iterations(); }

    void set_verbosity(int val) { solver_->verbosity(val); }
    int get_verbosity() const { return solver_->verbosity(); }

    void set_diis_epsilon(double val) { solver_->diis_epsilon(val); }
    double get_diis_epsilon() const { return solver_->diis_epsilon(); }

    void set_diis_threshold(double val) { solver_->diis_threshold(val); }
    double get_diis_threshold() const { return solver_->diis_threshold(); }

    void set_maximum_history_length(int val) { solver_->maximum_history_length(val); }
    int get_maximum_history_length() const { return solver_->maximum_history_length(); }

    void set_frozen_occupations(bool val) { solver_->frozen_occupations(val); }
    bool get_frozen_occupations() const { return solver_->frozen_occupations(); }

    void set_error_norm(const std::string& val) { solver_->error_norm(val); }
    std::string get_error_norm() const { return solver_->error_norm(); }

    void set_callback(nb::callable cb) {
        py_callback_ = std::move(cb);
        nb::callable& ref = py_callback_;
        solver_->callback_function(
            [&ref](const std::map<std::string, std::any>& data) {
                nb::gil_scoped_acquire gil;
                nb::dict py_data;
                // Extract known keys
                for (const auto& [key, val] : data) {
                    if (val.type() == typeid(double))
                        py_data[nb::str(key.c_str())] = std::any_cast<double>(val);
                    else if (val.type() == typeid(size_t))
                        py_data[nb::str(key.c_str())] = std::any_cast<size_t>(val);
                    else if (val.type() == typeid(int))
                        py_data[nb::str(key.c_str())] = std::any_cast<int>(val);
                    else if (val.type() == typeid(std::string))
                        py_data[nb::str(key.c_str())] = nb::str(std::any_cast<std::string>(val).c_str());
                }
                ref(py_data);
            }
        );
    }

    void set_convergence_callback(nb::callable cb) {
        py_convergence_callback_ = std::move(cb);
        nb::callable& ref = py_convergence_callback_;
        solver_->callback_convergence_function(
            [&ref](const std::map<std::string, std::any>& data) -> bool {
                nb::gil_scoped_acquire gil;
                nb::dict py_data;
                for (const auto& [key, val] : data) {
                    if (val.type() == typeid(double))
                        py_data[nb::str(key.c_str())] = std::any_cast<double>(val);
                    else if (val.type() == typeid(size_t))
                        py_data[nb::str(key.c_str())] = std::any_cast<size_t>(val);
                }
                return nb::cast<bool>(ref(py_data));
            }
        );
    }
};


NB_MODULE(_core, m) {
    m.doc() = "Python bindings for OpenOrbitalOptimizer SCF library";

    nb::class_<SCFSolverWrapper>(m, "SCFSolver",
        "Self-consistent field solver using ADIIS/EDIIS/DIIS convergence acceleration.\n\n"
        "Works in the orthonormal basis (unit overlap)."
    )
    .def(nb::init<
            nb::ndarray<uint64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>,
            NpArray1D, NpArray1D, nb::callable, std::vector<std::string>
         >(),
         "nblocks_per_type"_a, "max_occupation"_a, "nparticles"_a,
         "fock_builder"_a, "block_descriptions"_a,
         "Create an SCF solver.\n\n"
         "Parameters\n"
         "----------\n"
         "nblocks_per_type : ndarray[uint64]\n"
         "    Number of symmetry blocks per particle type.\n"
         "max_occupation : ndarray[float64]\n"
         "    Maximum occupation per orbital in each block.\n"
         "nparticles : ndarray[float64]\n"
         "    Number of particles of each type.\n"
         "fock_builder : callable\n"
         "    Function(orbitals, occupations) -> (energy, fock_matrices).\n"
         "    orbitals: list of 2D numpy arrays (orbital coefficients per block)\n"
         "    occupations: list of 1D numpy arrays (occupations per block)\n"
         "    Returns: (total_energy, list_of_fock_matrices)\n"
         "block_descriptions : list[str]\n"
         "    Human-readable descriptions of each block.\n"
    )
    .def("initialize_with_fock", &SCFSolverWrapper::initialize_with_fock,
         "fock_guess"_a,
         "Initialize solver by diagonalizing a guess Fock matrix.")
    .def("initialize_with_orbitals", &SCFSolverWrapper::initialize_with_orbitals,
         "orbitals"_a, "occupations"_a,
         "Initialize solver with given orbitals and occupations.")
    .def("run", &SCFSolverWrapper::run,
         "Run the SCF solver to convergence.")
    .def("get_energy", &SCFSolverWrapper::get_energy,
         "Get the converged total energy.")
    .def("get_solution", &SCFSolverWrapper::get_solution,
         "Get (orbitals, occupations) tuple.")
    .def("get_orbitals", &SCFSolverWrapper::get_orbitals,
         "Get list of orbital coefficient matrices.")
    .def("get_orbital_occupations", &SCFSolverWrapper::get_orbital_occupations,
         "Get list of orbital occupation vectors.")
    .def("get_fock_matrix", &SCFSolverWrapper::get_fock_matrix,
         "Get list of Fock matrices.")
    .def_prop_rw("convergence_threshold",
         &SCFSolverWrapper::get_convergence_threshold,
         &SCFSolverWrapper::set_convergence_threshold,
         "Convergence threshold for orbital gradient norm.")
    .def_prop_rw("maximum_iterations",
         &SCFSolverWrapper::get_maximum_iterations,
         &SCFSolverWrapper::set_maximum_iterations,
         "Maximum number of SCF iterations.")
    .def_prop_rw("verbosity",
         &SCFSolverWrapper::get_verbosity,
         &SCFSolverWrapper::set_verbosity,
         "Verbosity level (0=silent).")
    .def_prop_rw("diis_epsilon",
         &SCFSolverWrapper::get_diis_epsilon,
         &SCFSolverWrapper::set_diis_epsilon,
         "Error threshold to start mixing DIIS.")
    .def_prop_rw("diis_threshold",
         &SCFSolverWrapper::get_diis_threshold,
         &SCFSolverWrapper::set_diis_threshold,
         "Error threshold for pure DIIS.")
    .def_prop_rw("maximum_history_length",
         &SCFSolverWrapper::get_maximum_history_length,
         &SCFSolverWrapper::set_maximum_history_length,
         "Maximum DIIS history length.")
    .def_prop_rw("frozen_occupations",
         &SCFSolverWrapper::get_frozen_occupations,
         &SCFSolverWrapper::set_frozen_occupations,
         "Whether to freeze occupations.")
    .def_prop_rw("error_norm",
         &SCFSolverWrapper::get_error_norm,
         &SCFSolverWrapper::set_error_norm,
         "Norm used for error evaluation ('rms', 'fro', 'inf').")
    .def("set_callback", &SCFSolverWrapper::set_callback,
         "callback"_a,
         "Set iteration callback function(dict).")
    .def("set_convergence_callback", &SCFSolverWrapper::set_convergence_callback,
         "callback"_a,
         "Set convergence check callback function(dict) -> bool.");
}
