"""
Benchmark: OpenOrbitalOptimizer  vs.  PySCF default  vs.  skala retry.

Runs a collection of molecules/basis-set combinations with all three solvers
and reports a table with the metrics recommended for judging solver quality.

Usage
-----
    python -m pytest tests/test_benchmark.py -v -s

or as a standalone script:

    python tests/test_benchmark.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pytest
from pyscf import gto, scf

from openorbitaloptimizer.pyscf import open_orbital_optimizer
from ._molecules import get_mole
from skala.pyscf.retry import retry_scf


BENCHMARK_TESTS = [
    "H2O,6-31g*",
    "H2O,def2-tzvp",
    "H2O,aug-pc-2",
    "N2,cc-pvdz",
    "N2,def2-tzvp",
    "C2H4,6-31g",
    "C2H4,def2-tzvp",
    "PCl3,sto-3g",
    "Fe(CO)5,sto-3g",
    "ClO2,def2-tzvp",
    "C2H,def2-tzvpd",
    "C5H3INO2P,sto-3g",
    "C3H9Pb,def2-svp",
    "H3+,aug-pc-2",
    "(H2O)2+,def2-tzvp",
    "C7H6IP,sto-3g",
    "F2H,def2-svp",
    "F2H,def2-tzvp",
    "CHO,cc-pvdz",
    "CHO,def2-tzvpd",
    "CH3Li3,def2-svp",
    "CH3Li3,def2-tzvp",
    "Li2,cc-pvdz",
    "Li2,def2-tzvp",
    "N2+,cc-pvdz",
    "N2+,def2-tzvp",
    "FO,def2-svp",
    "FO,aug-pc-2",
    "N,def2-svp",
    "N,def2-tzvpd",
]


@dataclass
class BenchmarkResult:
    name: str
    solver: str
    converged: bool
    energy: float
    n_iterations: int
    n_fock_builds: int
    wall_time: float
    final_gradient_norm: float | None


def _make_mf(mol: gto.Mole):
    """Return RHF or UHF depending on mol.spin."""
    return scf.UHF(mol) if mol.spin > 0 else scf.RHF(mol)


def run_pyscf_default(mol: gto.Mole) -> BenchmarkResult:
    """Run PySCF's default SCF solver (DIIS)."""
    mf = _make_mf(mol)
    mf.verbose = 0
    mf.max_cycle = 200

    # Track Fock builds via callback
    nfock = [0]
    grad_norms: list[float] = []

    def _cb(envs):
        nfock[0] += 1
        grad_norms.append(envs.get("norm_gorb", None))

    mf.callback = _cb
    t0 = time.perf_counter()
    mf.kernel()
    wall = time.perf_counter() - t0

    return BenchmarkResult(
        name=mol._basis if isinstance(mol._basis, str) else str(mol._basis),
        solver="PySCF-default",
        converged=bool(mf.converged),
        energy=float(mf.e_tot),
        n_iterations=nfock[0],
        n_fock_builds=nfock[0],
        wall_time=wall,
        final_gradient_norm=grad_norms[-1] if grad_norms else None,
    )


def run_pyscf_retry(mol: gto.Mole) -> BenchmarkResult:
    """Run PySCF via the skala retry mechanism."""
    mf = _make_mf(mol)
    mf.verbose = 0
    mf.max_cycle = 200

    t0 = time.perf_counter()
    mf, state = retry_scf(mf)
    wall = time.perf_counter() - t0

    final_grad = None
    if state.gradient_norm_per_cycle:
        finite = [g for g in state.gradient_norm_per_cycle if g is not None]
        if finite:
            final_grad = finite[-1]

    return BenchmarkResult(
        name="",
        solver="PySCF-retry",
        converged=bool(mf.converged),
        energy=float(mf.e_tot),
        n_iterations=len(state.e_tot_per_cycle),
        n_fock_builds=len(state.e_tot_per_cycle),
        wall_time=wall,
        final_gradient_norm=final_grad,
    )


def run_ooo(mol: gto.Mole) -> BenchmarkResult:
    """Run OpenOrbitalOptimizer via the PySCF driver."""
    mf = _make_mf(mol)
    config = dict(
        maximum_iterations=200,
    )

    t0 = time.perf_counter()
    mf = open_orbital_optimizer(mf, config=config)
    mf.kernel()
    wall = time.perf_counter() - t0

    state = mf.open_orbital_optimizer_state
    final_grad = None
    if state.gradient_norm_per_cycle:
        finite = [g for g in state.gradient_norm_per_cycle if g is not None]
        if finite:
            final_grad = finite[-1]

    return BenchmarkResult(
        name="",
        solver="OpenOrbitalOptimizer",
        converged=np.isfinite(mf.e_tot) and bool(mf.converged),
        energy=float(mf.e_tot) if np.isfinite(mf.e_tot) else float("nan"),
        n_iterations=len(state.e_tot_per_cycle),
        n_fock_builds=state.nfock,
        wall_time=wall,
        final_gradient_norm=final_grad,
    )


@pytest.mark.parametrize("name", BENCHMARK_TESTS)
def test_benchmark(name):
    """Compare all three solvers on each molecule."""
    mol_key, basis = name.split(",")
    mol = get_mole(mol_key, basis=basis)

    res_pyscf = run_pyscf_default(mol)
    res_retry = run_pyscf_retry(mol)
    res_ooo = run_ooo(mol)

    # Print comparison table
    header = f"\n{'=' * 86}\n  {name}\n{'=' * 86}"
    print(header)
    fmt = "{:<22s} {:>12s} {:>16s} {:>8s} {:>10s} {:>12s}"
    print(
        fmt.format(
            "Solver", "Converged", "Energy / Eh", "Iters", "Fock evals", "Wall / s"
        )
    )
    print("-" * 86)
    for r in [res_pyscf, res_retry, res_ooo]:
        print(
            f"{r.solver:<22s} {'yes' if r.converged else 'NO':>12s} "
            f"{r.energy:>16.10f} {r.n_iterations:>8d} {r.n_fock_builds:>10d} "
            f"{r.wall_time:>12.3f}"
        )

    # Energy agreement: OpenOrbitalOptimizer energy should be ≤ PySCF (it may find a lower
    # minimum, as happens for PCl3).  When close, they should agree to 1e-6.
    if res_pyscf.converged and res_ooo.converged:
        if res_ooo.energy > res_pyscf.energy + 1e-6:
            # OpenOrbitalOptimizer is higher — that would be a regression
            np.testing.assert_allclose(
                res_ooo.energy,
                res_pyscf.energy,
                atol=1e-6,
                err_msg=f"OpenOrbitalOptimizer energy is higher than PySCF on {name}",
            )
        elif res_ooo.energy < res_pyscf.energy - 1e-6:
            # OpenOrbitalOptimizer found a lower minimum — this is a win, just log it
            print(
                f"\n  ** OpenOrbitalOptimizer found a LOWER minimum by "
                f"{res_pyscf.energy - res_ooo.energy:.6f} Eh **"
            )


INIT_GUESS_OPTIONS = ["minao", "1e", "atom", "huckel", "mod_huckel", "sap"]


@pytest.mark.parametrize("init_guess", INIT_GUESS_OPTIONS)
def test_pcl3_init_guess(init_guess):
    """Compare OpenOrbitalOptimizer vs PySCF on PCl3/sto-3g with different initial guesses.

    The default (minao) initial guess leads to 144 OpenOrbitalOptimizer iterations;
    better starting points like mod_huckel or sap converge much faster.
    """
    mol = get_mole("PCl3", basis="sto-3g")

    # --- PySCF default solver ---
    mf_pyscf = scf.RHF(mol)
    mf_pyscf.verbose = 0
    mf_pyscf.max_cycle = 200
    mf_pyscf.init_guess = init_guess
    nfock_pyscf = [0]

    def _cb(envs):
        nfock_pyscf[0] += 1

    mf_pyscf.callback = _cb
    t0 = time.perf_counter()
    mf_pyscf.kernel()
    wall_pyscf = time.perf_counter() - t0
    assert mf_pyscf.converged, f"PySCF did not converge with init_guess={init_guess}"

    # --- OpenOrbitalOptimizer solver ---
    mf_ooo = scf.RHF(mol)
    mf_ooo.init_guess = init_guess
    t0 = time.perf_counter()
    mf_ooo = open_orbital_optimizer(mf_ooo, config={"maximum_iterations": 200})
    mf_ooo.kernel()
    wall_ooo = time.perf_counter() - t0
    assert mf_ooo.converged, (
        f"OpenOrbitalOptimizer did not converge with init_guess={init_guess}"
    )

    n_iter_ooo = len(mf_ooo.open_orbital_optimizer_state.e_tot_per_cycle)
    nfock_ooo = mf_ooo.open_orbital_optimizer_state.nfock

    # Print comparison table
    print(f"\n{'=' * 86}")
    print(f"  PCl3/sto-3g  init_guess={init_guess}")
    print(f"{'=' * 86}")
    fmt = "{:<22s} {:>12s} {:>16s} {:>8s} {:>10s} {:>12s}"
    print(
        fmt.format(
            "Solver", "Converged", "Energy / Eh", "Iters", "Fock evals", "Wall / s"
        )
    )
    print("-" * 86)
    print(
        f"{'PySCF':<22s} {'yes' if mf_pyscf.converged else 'NO':>12s} "
        f"{mf_pyscf.e_tot:>16.10f} {nfock_pyscf[0]:>8d} {nfock_pyscf[0]:>10d} "
        f"{wall_pyscf:>12.3f}"
    )
    print(
        f"{'OpenOrbitalOptimizer':<22s} {'yes' if np.isfinite(mf_ooo.e_tot) and mf_ooo.converged else 'NO':>12s} "
        f"{mf_ooo.e_tot:>16.10f} {n_iter_ooo:>8d} {nfock_ooo:>10d} "
        f"{wall_ooo:>12.3f}"
    )

    assert np.isfinite(mf_ooo.e_tot), (
        f"OpenOrbitalOptimizer did not converge with init_guess={init_guess}"
    )

    # With a good initial guess OpenOrbitalOptimizer should need far fewer than 144 iterations
    if init_guess in ("mod_huckel", "sap"):
        assert n_iter_ooo < 144, (
            f"OpenOrbitalOptimizer with init_guess={init_guess} took {n_iter_ooo} iterations, "
            f"expected < 144"
        )


if __name__ == "__main__":
    print("\nOpenOrbitalOptimizer benchmark")
    print("=" * 86)

    all_results: list[tuple[str, list[BenchmarkResult]]] = []

    for name in BENCHMARK_TESTS:
        mol_key, basis = name.split(",")
        mol = get_mole(mol_key, basis=basis)
        results = [run_pyscf_default(mol), run_pyscf_retry(mol), run_ooo(mol)]
        all_results.append((name, results))

    # ---- Summary table ----
    print(f"\n{'=' * 106}")
    print("  SUMMARY")
    print(f"{'=' * 106}")
    fmt = "{:<22s} {:<22s} {:>10s} {:>16s} {:>8s} {:>10s} {:>10s}"
    print(
        fmt.format(
            "System", "Solver", "Conv?", "Energy / Eh", "Iters", "Fock", "Wall/s"
        )
    )
    print("-" * 106)
    for name, results in all_results:
        for r in results:
            print(
                f"{name:<22s} {r.solver:<22s} {'yes' if r.converged else 'NO':>10s} "
                f"{r.energy:>16.10f} {r.n_iterations:>8d} {r.n_fock_builds:>10d} "
                f"{r.wall_time:>10.3f}"
            )
        print("-" * 106)
