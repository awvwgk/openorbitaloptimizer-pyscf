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
from typing import Any, Callable, Protocol

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from openorbitaloptimizer._core import SCFSolver


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


def _make_fock_builder(mf: Any, X: NDArray, unrestricted: bool) -> Callable:
    """Return a Python callable matching the OOO FockBuilder signature.

    The closure captures *mf* and *X* and converts between orthonormal
    and AO representations on every call.
    """

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
            return (float(energy), fock_ortho)
        else:
            C_ao = X @ orbitals[0]
            D = C_ao @ np.diag(occupations[0]) @ C_ao.T
            dm = D
            vhf = mf.get_veff(dm=dm)
            hcore = mf.get_hcore()
            fock_ao = hcore + vhf
            energy = mf.energy_tot(dm=dm, h1e=hcore, vhf=vhf)
            fock_ortho = _to_ortho(fock_ao, X)
            return (float(energy), [fock_ortho])

    return _builder


def run_ooo_scf(
    mf: SCFClass,
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

    Returns
    -------
    (float, SCFState)
        The converged total energy and an SCFState recording the trajectory.
    """
    t0 = time.perf_counter()

    # 1. Ensure PySCF object is built
    if mf.mol.nao == 0:
        mf.build()

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
    fb = _make_fock_builder(mf, X, is_unrestricted)

    # 5. Create the C++ solver
    solver = SCFSolver(nblocks_per_type, max_occ, nparticles, fb, block_desc)
    solver.convergence_threshold = mf.conv_tol
    solver.maximum_iterations = mf.max_cycle
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

    solver.set_callback(_callback)

    # 7. Initial guess (Fock from PySCF's init guess density)
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

    mf.e_tot = energy
    mf.converged = True

    _fill_gap(state, mf, is_unrestricted)

    state.wall_time = time.perf_counter() - t0
    return energy, state


def _is_unrestricted(mf: Any) -> bool:
    """Detect whether the PySCF object is unrestricted / generalised."""
    from pyscf import scf

    return isinstance(mf, (scf.uhf.UHF,))


def _fill_gap(state: SCFState, mf: Any, unrestricted: bool) -> None:
    """Fill HOMO-LUMO gap info in *state* from the final MO energies/occupations."""
    try:
        if unrestricted:
            for spin, attr in enumerate(
                ["homo_lumo_gap_up_per_cycle", "homo_lumo_gap_down_per_cycle"]
            ):
                mo_e = np.asarray(mf.mo_energy[spin])
                mo_o = np.asarray(mf.mo_occ[spin])
                occ = mo_e[mo_o > 0]
                vir = mo_e[mo_o == 0]
                if len(occ) > 0 and len(vir) > 0:
                    getattr(state, attr).append(float(np.min(vir) - np.max(occ)))
                else:
                    getattr(state, attr).append(None)
        else:
            mo_e = np.asarray(mf.mo_energy)
            mo_o = np.asarray(mf.mo_occ)
            occ = mo_e[mo_o > 0]
            vir = mo_e[mo_o == 0]
            if len(occ) > 0 and len(vir) > 0:
                gap = float(np.min(vir) - np.max(occ))
            else:
                gap = None
            state.homo_lumo_gap_up_per_cycle.append(gap)
            state.homo_lumo_gap_down_per_cycle.append(gap)
    except Exception:
        state.homo_lumo_gap_up_per_cycle.append(None)
        state.homo_lumo_gap_down_per_cycle.append(None)
