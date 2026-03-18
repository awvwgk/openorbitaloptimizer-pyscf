"""
PySCF integration driver for OpenOrbitalOptimizer.

This module wraps a PySCF SCF object (RHF, RKS, UHF, UKS, etc.) so that
OpenOrbitalOptimizer performs the orbital optimisation while PySCF provides
the Fock builder (integrals, XC, …) and infrastructure (grids, etc.).

The driver works entirely in the **orthonormal basis**: it orthogonalises the
overlap matrix once and converts between the AO and orthonormal bases as
needed.  This matches the OpenOrbitalOptimizer API, which assumes unit overlap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TypeVar

import numpy as np
import scipy.linalg
from numpy.typing import NDArray
from pyscf import lib

from openorbitaloptimizer._core import SCFSolver


_SCF = TypeVar("_SCF")


class MolClass(Protocol):
    """Protocol for PySCF Mole objects."""

    nao: int
    nelectron: int
    nelec: tuple[int, int]


class SCFClass(Protocol):
    """Protocol for PySCF SCF objects."""

    mol: MolClass
    conv_tol: float
    max_cycle: int
    diis_space: int
    verbose: int

    def build(self) -> None: ...

    def get_hcore(self) -> NDArray: ...

    def get_ovlp(self) -> NDArray: ...

    def get_init_guess(self, key: str) -> NDArray: ...

    def get_veff(self, dm: NDArray | list[NDArray]) -> NDArray | list[NDArray]: ...

    def energy_tot(
        self, dm: NDArray | list[NDArray], h1e: NDArray, vhf: NDArray | list[NDArray]
    ) -> float: ...


def _orthogonalise(S: NDArray) -> tuple[NDArray, NDArray]:
    """Cholesky orthogonalisation of the overlap matrix.

    Decomposes S = L L^T and returns X = L^{-T} (the orthogonalising
    transformation) together with the Cholesky factor L (needed for the
    inverse transformation).
    """
    L = np.linalg.cholesky(S)  # lower-triangular, S = L @ L.T
    X = scipy.linalg.solve_triangular(L, np.eye(S.shape[0]), lower=True).T
    return X, L


def _to_ortho(mat: NDArray, X: NDArray) -> NDArray:
    """AO → orthonormal basis:  F' = X^T F X."""
    return X.T @ mat @ X


def _from_ortho(mat: NDArray, L: NDArray) -> NDArray:
    """Orthonormal → AO basis using the Cholesky factor:  F = L^T F' L."""
    return L.T @ mat @ L


@dataclass
class SCFState:
    """Lightweight recording of the SCF convergence trajectory."""

    ntries: int = 0
    cycles: list[int] = field(default_factory=list)
    e_tot_per_cycle: list[float] = field(default_factory=list)
    gradient_norm_per_cycle: list[float | None] = field(default_factory=list)
    dm_change_per_cycle: list[float | None] = field(default_factory=list)
    homo_lumo_gap_up_per_cycle: list[float | None] = field(default_factory=list)
    homo_lumo_gap_down_per_cycle: list[float | None] = field(default_factory=list)
    wall_time: float | None = None
    nfock: int = 0


@dataclass(frozen=True, eq=True)
class SCFConfig:
    """Configuration options for the SCF solver."""

    maximum_iterations: int = 128
    """Maximum number of iterations"""

    diis_epsilon: float = 1e-1
    """Start to mix in DIIS at this error threshold (Garza and Scuseria, 2012)"""
    diis_threshold: float = 1e-4
    """Threshold for pure DIIS (Garza and Scuseria, 2012)"""
    diis_diagonal_damping: float = 0.02
    """Damping factor for DIIS diagonal (Hamilton and Pulay, 1986)"""
    diis_restart_factor: float = 1e-4
    """DIIS restart criterion (Chupin et al, 2021)"""

    optimal_damping_threshold: float = 1.0
    """Criterion for max error for which to use optimal damping"""

    convergence_threshold: float = 1.0e-7
    """Convergence threshold"""
    maximum_history_length: int = 10
    """History length"""
    error_norm: str = "rms"
    """Norm to use by default: root-mean-square error"""

    def set_solver(self, solver: SCFSolver) -> None:
        """Apply the configuration to the given solver."""
        solver.convergence_threshold = self.convergence_threshold
        solver.maximum_iterations = self.maximum_iterations
        solver.maximum_history_length = self.maximum_history_length
        solver.diis_threshold = self.diis_threshold
        solver.diis_diagonal_damping = self.diis_diagonal_damping
        solver.diis_restart_factor = self.diis_restart_factor
        solver.optimal_damping_threshold = self.optimal_damping_threshold
        solver.error_norm = self.error_norm

    @classmethod
    def from_mf(cls, mf: SCFClass) -> SCFConfig:
        """Create a configuration from a PySCF mean-field object."""
        return cls(
            convergence_threshold=mf.conv_tol,
            maximum_iterations=mf.max_cycle,
            maximum_history_length=mf.diis_space,
            verbosity=mf.verbose,
        )

    @classmethod
    def from_solver(cls, solver: SCFSolver) -> SCFConfig:
        """Create a configuration from an SCFSolver."""
        return cls(
            convergence_threshold=solver.convergence_threshold,
            maximum_iterations=solver.maximum_iterations,
            maximum_history_length=solver.maximum_history_length,
            diis_threshold=solver.diis_threshold,
            diis_diagonal_damping=solver.diis_diagonal_damping,
            diis_restart_factor=solver.diis_restart_factor,
            optimal_damping_threshold=solver.optimal_damping_threshold,
            density_restart_factor=solver.density_restart_factor,
            convergence_threshold_orbital_gradient=solver.convergence_threshold_orbital_gradient,
            error_norm=solver.error_norm,
            minimal_gradient_projection=solver.minimal_gradient_projection,
            occupied_threshold=solver.occupied_threshold,
            initial_level_shift=solver.initial_level_shift,
            level_shift_factor=solver.level_shift_factor,
        )


def _make_fock_builder(
    mf: Any, X: NDArray, unrestricted: bool, iter_data: dict[str, float | None]
) -> Callable:
    """Return a Python callable matching the OpenOrbitalOptimizer FockBuilder signature.

    The closure captures *mf* and *X* and converts between orthonormal
    and AO representations on every call.  It also populates *iter_data*
    with the density-matrix change and HOMO-LUMO gap(s) for each call so
    that the iteration callback can record them.
    """

    _prev_dm: list[NDArray | None] = [None]

    def _builder(
        orbitals: list[NDArray], occupations: list[NDArray]
    ) -> tuple[float, list[NDArray]]:
        if unrestricted:
            Ca_ao = X @ orbitals[0]
            Cb_ao = X @ orbitals[1]
            Da = Ca_ao @ np.diag(occupations[0]) @ Ca_ao.T
            Db = Cb_ao @ np.diag(occupations[1]) @ Cb_ao.T
            dm = np.array([Da, Db])
            vhf = mf.get_veff(dm=dm)
            hcore = mf.get_hcore()
            fock_ao = [hcore + vhf[0], hcore + vhf[1]]
            energy = mf.energy_tot(dm=dm, h1e=hcore, vhf=vhf)
            fock_ortho = [_to_ortho(f, X) for f in fock_ao]
        else:
            C_ao = X @ orbitals[0]
            D = C_ao @ np.diag(occupations[0]) @ C_ao.T
            dm = D
            vhf = mf.get_veff(dm=dm)
            hcore = mf.get_hcore()
            fock_ao = hcore + vhf
            energy = mf.energy_tot(dm=dm, h1e=hcore, vhf=vhf)
            fock_ortho = _to_ortho(fock_ao, X)
            fock_ortho = [fock_ortho]

        # ---- per-iteration diagnostics ----
        # Density-matrix change (Frobenius norm)
        if _prev_dm[0] is not None:
            iter_data["dm_change"] = float(np.linalg.norm(dm - _prev_dm[0]))
        else:
            iter_data["dm_change"] = None
        _prev_dm[0] = dm.copy()

        # HOMO–LUMO gap from Fock eigenvalues in the orthonormal basis
        if unrestricted:
            for spin, key in enumerate(["gap_up", "gap_down"]):
                eigvals = np.linalg.eigvalsh(fock_ortho[spin])
                nocc = int(np.sum(occupations[spin] > 0))
                if 0 < nocc < len(eigvals):
                    iter_data[key] = float(eigvals[nocc] - eigvals[nocc - 1])
                else:
                    iter_data[key] = None
        else:
            eigvals = np.linalg.eigvalsh(fock_ortho[0])
            nocc = int(np.sum(occupations[0] > 0))
            if 0 < nocc < len(eigvals):
                gap = float(eigvals[nocc] - eigvals[nocc - 1])
            else:
                gap = None
            iter_data["gap_up"] = gap
            iter_data["gap_down"] = gap

        return (float(energy), fock_ortho)

    return _builder


def run_open_orbital_optimizer(
    mf: SCFClass,
    dm0: NDArray | list[NDArray] | None = None,
    config: SCFConfig | None = None,
) -> tuple[float, SCFState]:
    """Run a PySCF SCF calculation using the OpenOrbitalOptimizer back-end.

    Parameters
    ----------
    mf : SCFClass
        Any PySCF mean-field object (RHF, UHF, RKS, UKS, …).
        The object **must** have been built (``mf.build()`` or ``mf.get_hcore()``
        should work) but need *not* have run ``mf.kernel()`` yet.
        The object is modified in-place with the converged solution
        (``mo_coeff``, ``mo_energy``, ``mo_occ``, ``e_tot``, ``converged``).
    dm0 : NDArray or list[NDArray], optional
        Initial guess density matrix in the AO basis.  If not provided, the
        driver uses PySCF's default initial guess (``mf.get_init_guess()``).
    config : SCFConfig, optional
        Configuration options for the SCF solver.  If not provided, defaults are
        used.

    Returns
    -------
    (float, SCFState)
        The converged total energy and an SCFState recording the trajectory.
    """
    t0 = time.perf_counter()

    # 1. Ensure PySCF object is built
    if mf.mol.nao == 0:
        mf.build()
    if config is None:
        config = SCFConfig()

    is_unrestricted = _is_unrestricted(mf)

    # 2. Overlap and orthogonalisation
    S = mf.get_ovlp()
    X, L = _orthogonalise(S)
    nmo = X.shape[1]

    # 3. Determine blocks / particle info
    if is_unrestricted:
        nblocks_per_type = np.array([1, 1], dtype=np.uint64)
        nelec = mf.mol.nelec  # (nalpha, nbeta)
        max_occ = np.array([1.0, 1.0])
        nparticles = np.array([float(nelec[0]), float(nelec[1])])
        block_desc = ["alpha", "beta"]
    else:
        nblocks_per_type = np.array([1], dtype=np.uint64)
        nelec = mf.mol.nelectron
        max_occ = np.array([2.0])
        nparticles = np.array([float(nelec)])
        block_desc = ["spatial"]

    # 4. Build the Fock builder closure
    iter_data: dict[str, float | None] = {}
    fb = _make_fock_builder(mf, X, is_unrestricted, iter_data)

    # 5. Create the C++ solver
    solver = SCFSolver(nblocks_per_type, max_occ, nparticles, fb, block_desc)
    config.set_solver(solver)
    solver.verbosity = mf.verbose

    # 6. Logging state
    state = SCFState()
    state.ntries = 1

    def _callback(data: dict) -> None:
        it = data.get("iter", 0)
        state.cycles.append(int(it))
        state.e_tot_per_cycle.append(data.get("E", float("nan")))
        state.gradient_norm_per_cycle.append(data.get("diis_error", None))
        state.nfock = int(data.get("nfock", state.nfock))
        state.dm_change_per_cycle.append(iter_data.get("dm_change"))
        state.homo_lumo_gap_up_per_cycle.append(iter_data.get("gap_up"))
        state.homo_lumo_gap_down_per_cycle.append(iter_data.get("gap_down"))

    solver.set_callback(_callback)

    # 7. Initial guess (Fock from PySCF's init guess density)
    if dm0 is None:
        dm0 = mf.get_init_guess(key=mf.init_guess)
    hcore = mf.get_hcore()
    vhf0 = mf.get_veff(dm=dm0)
    if is_unrestricted:
        fock0_ao = [hcore + vhf0[0], hcore + vhf0[1]]
        fock0_ortho = [_to_ortho(f, X) for f in fock0_ao]
    else:
        fock0_ao = hcore + vhf0
        fock0_ortho = [_to_ortho(fock0_ao, X)]

    solver.initialize_with_fock(fock0_ortho)

    # 8. Run
    solver.run()

    # 9. Retrieve solution and map back to AO
    converged = solver.is_converged()
    energy = solver.get_energy()
    orbitals_ortho = solver.get_orbitals()
    occupations = solver.get_orbital_occupations()
    fock_ortho = solver.get_fock_matrix()

    if is_unrestricted:
        mo_coeff = [X @ orbitals_ortho[0], X @ orbitals_ortho[1]]
        mo_occ = [occupations[0], occupations[1]]
        mo_energy_list = []
        for i in range(2):
            f_ao = _from_ortho(fock_ortho[i], L)
            e, _ = scipy.linalg.eigh(f_ao, S)
            mo_energy_list.append(e[:nmo])
        mf.mo_energy = mo_energy_list
        mf.mo_coeff = mo_coeff
        mf.mo_occ = mo_occ
    else:
        mo_coeff = X @ orbitals_ortho[0]
        mo_occ = occupations[0]
        f_ao = _from_ortho(fock_ortho[0], L)
        eigvals, _ = scipy.linalg.eigh(f_ao, S)
        mf.mo_energy = eigvals[:nmo]
        mf.mo_coeff = mo_coeff
        mf.mo_occ = mo_occ

    mf.converged = converged
    mf.e_tot = mf.energy_tot()

    state.wall_time = time.perf_counter() - t0
    return energy, state


def _is_unrestricted(mf: Any) -> bool:
    """Detect whether the PySCF object is unrestricted / generalised."""
    from pyscf import scf

    return isinstance(mf, (scf.uhf.UHF,))


class _OpenOrbitalOptimizer:
    """Mixin class that replaces PySCF's SCF solver with OpenOrbitalOptimizer.

    This follows the standard PySCF addon pattern (like ``newton()``,
    ``density_fit()``, ``smearing()``, ...).  A dynamic class inheriting from
    both ``_OpenOrbitalOptimizer`` and the original mean-field class is created by :func:`open_orbital_optimizer`,
    so that all PySCF methods not overridden here fall through unchanged.
    """

    __name_mixin__ = "OpenOrbitalOptimizer"

    _keys = frozenset({"open_orbital_optimizer_config", "open_orbital_optimizer_state"})

    def __init__(self, mf: Any, config: SCFConfig | dict | None = None):
        self.__dict__.update(mf.__dict__)
        self._scf = mf
        if isinstance(config, dict):
            config = SCFConfig(**config)
        self.open_orbital_optimizer_config = config or SCFConfig()
        self.open_orbital_optimizer_state: SCFState | None = None

    def undo_open_orbital_optimizer(self):
        """Remove the OpenOrbitalOptimizer mixin and restore the original SCF class."""

        obj = lib.view(self, lib.drop_class(self.__class__, _OpenOrbitalOptimizer))
        del (
            obj.open_orbital_optimizer_config,
            obj.open_orbital_optimizer_state,
            obj._scf,
        )
        return obj

    def kernel(self, h1e=None, s1e=None, dm0=None, **kwargs):
        """Run the SCF using OpenOrbitalOptimizer.

        The signature mirrors ``pyscf.scf.hf.SCF.kernel`` so that this
        object is a true drop-in replacement.  *h1e* and *s1e* are
        accepted for compatibility but currently ignored (the driver
        always obtains them from the mean-field object).

        Parameters
        ----------
        h1e, s1e : ignored
            Accepted for API compatibility with PySCF's ``kernel``.
        dm0 : NDArray or list[NDArray], optional
            Initial guess density matrix in the AO basis.
        **kwargs
            Silently absorbed for forward-compatibility.

        Returns
        -------
        float
            The converged total energy.
        """
        if kwargs:
            import warnings

            warnings.warn(
                f"open_orbital_optimizer.kernel() got unexpected keyword arguments: {kwargs}",
                stacklevel=2,
            )

        self.build(self.mol)
        self.dump_flags()

        _, state = run_open_orbital_optimizer(
            self, dm0=dm0, config=self.open_orbital_optimizer_config
        )
        self.open_orbital_optimizer_state = state

        # PySCF convention: call _finalize (logs energy, etc.)
        self._finalize()

        return self.e_tot


def open_orbital_optimizer(mf: _SCF, config: SCFConfig | dict | None = None) -> _SCF:
    """Apply OpenOrbitalOptimizer as the SCF solver for a PySCF mean-field object.

    Returns a new SCF object whose ``kernel()`` method uses
    OpenOrbitalOptimizer for orbital optimisation instead of PySCF's
    built-in DIIS.  All other PySCF functionality (integrals, grids, …)
    is untouched.

    Parameters
    ----------
    mf : pyscf.scf.hf.SCF
        Any PySCF mean-field object (RHF, UHF, RKS, UKS, …).
    config : SCFConfig, dict, optional
        Configuration for the OpenOrbitalOptimizer solver.  If a dictionary is provided,
        it is converted to an SCFConfig object.  If ``None``, default ``SCFConfig()`` values are used.

    Returns
    -------
    mf : SCF object
        A copy of *mf* with ``kernel()`` overridden to use OpenOrbitalOptimizer.
        The original object is **not** modified.

    Examples
    --------
    >>> from pyscf import gto, scf
    >>> from openorbitaloptimizer.pyscf import open_orbital_optimizer
    >>> mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    >>> mf = open_orbital_optimizer(scf.RHF(mol))
    >>> mf.kernel()
    -1.1174...
    """

    if isinstance(mf, _OpenOrbitalOptimizer):
        # Already wrapped — just update the config if requested.
        if config is not None:
            if isinstance(config, dict):
                config = SCFConfig(**config)
            mf.open_orbital_optimizer_config = config
        return mf

    open_orbital_optimizer_mf = _OpenOrbitalOptimizer(mf, config)
    return lib.set_class(
        open_orbital_optimizer_mf, (_OpenOrbitalOptimizer, mf.__class__)
    )
