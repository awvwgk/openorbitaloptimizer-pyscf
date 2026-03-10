"""
Challenging molecule tests where OpenOrbitalOptimizer converges to a *lower*
SCF energy than PySCF's default solver.

These systems expose saddle-point convergence, self-interaction error, or
multi-minimum PES landscapes where OpenOrbitalOptimizer's ADIIS/EDIIS strategy is more robust
than PySCF's DIIS.
"""

import numpy as np
import pytest
from pyscf import dft, scf

from openorbitaloptimizer.pyscf import open_orbital_optimizer

from ._molecules import get_mole

CLOSED_SHELL_TESTS = ["PCl3,sto-3g"]
OPEN_SHELL_TESTS = [
    "ClO2,sto-3g",
    "C2H,sto-3g",
    "(H2O)2+,sto-3g",
    "N2+,sto-3g",
    "N2+,cc-pvdz",
    "FO,sto-3g",
    "FO,def2-svp",
    "N,sto-3g",
    "N,def2-svp",
]


class TestRHF:
    """Restricted Hartree-Fock on challenging closed-shell systems.

    PySCF's default DIIS may converge to a higher local minimum.
    OpenOrbitalOptimizer should find an energy at least as low as PySCF's default.
    """

    @pytest.mark.parametrize("name", CLOSED_SHELL_TESTS)
    def test_energy(self, name):
        """OpenOrbitalOptimizer RHF energy is at least as low as PySCF's default."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf_ref = scf.RHF(mol)
        mf_ref.kernel()
        assert mf_ref.converged, f"PySCF did not converge on {name}"

        mf_ooo = scf.RHF(mol)
        mf_ooo = open_orbital_optimizer(mf_ooo, config={"maximum_iterations": 200})
        mf_ooo.kernel()

        assert mf_ooo.converged, f"OpenOrbitalOptimizer did not converge on {name}"
        assert np.isfinite(mf_ooo.e_tot)
        assert mf_ooo.e_tot <= mf_ref.e_tot + 1e-7


class TestUHF:
    """Unrestricted Hartree-Fock on challenging open-shell systems.

    With PySCF's initial guess, both solvers start from the same density.
    OpenOrbitalOptimizer should converge to an energy at least as low as PySCF's default.
    """

    @pytest.mark.parametrize("name", OPEN_SHELL_TESTS)
    def test_energy(self, name):
        """OpenOrbitalOptimizer UHF energy is at least as low as PySCF's default."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf_ref = scf.UHF(mol)
        mf_ref.kernel()
        assert mf_ref.converged, f"PySCF did not converge on {name}"

        mf_ooo = scf.UHF(mol)
        mf_ooo = open_orbital_optimizer(mf_ooo)
        mf_ooo.kernel()

        assert mf_ooo.converged, f"OpenOrbitalOptimizer did not converge on {name}"
        assert np.isfinite(mf_ooo.e_tot)
        assert mf_ooo.e_tot <= mf_ref.e_tot + 1e-7


class TestUKS:
    """Unrestricted Kohn-Sham DFT on challenging open-shell systems."""

    @pytest.mark.parametrize("name", OPEN_SHELL_TESTS[:1])
    def test_energy(self, name):
        """OpenOrbitalOptimizer UKS/TPSS energy is at least as low as PySCF's default."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf_ref = dft.UKS(mol, xc="tpss")
        mf_ref.kernel()
        assert mf_ref.converged, f"PySCF did not converge on {name}"

        mf_ooo = dft.UKS(mol, xc="tpss")
        mf_ooo = open_orbital_optimizer(mf_ooo)
        mf_ooo.kernel()

        assert mf_ooo.converged, f"OpenOrbitalOptimizer did not converge on {name}"
        assert np.isfinite(mf_ooo.e_tot)
        assert mf_ooo.e_tot <= mf_ref.e_tot + 1e-7


class TestHcoreGuessRHF:
    """RHF with hcore initial guess on challenging systems.

    Starting from the bare one-electron Hamiltonian avoids bias toward a
    particular basin of attraction.  OpenOrbitalOptimizer's ADIIS/EDIIS strategy should
    find an energy at least as low as PySCF's DIIS from the same start.
    """

    @pytest.mark.parametrize("name", CLOSED_SHELL_TESTS)
    def test_energy(self, name):
        """OpenOrbitalOptimizer RHF/hcore energy is at least as low as PySCF's."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf_ref = scf.RHF(mol)(init_guess="hcore")
        mf_ref.kernel()
        assert mf_ref.converged, (
            f"PySCF did not converge with init_guess=hcore on {name}"
        )

        mf_ooo = scf.RHF(mol)(init_guess="hcore")
        mf_ooo = open_orbital_optimizer(mf_ooo, config={"maximum_iterations": 200})
        mf_ooo.kernel()

        assert mf_ooo.converged, (
            f"OpenOrbitalOptimizer did not converge with init_guess=hcore on {name}"
        )
        assert np.isfinite(mf_ooo.e_tot)
        assert mf_ooo.e_tot <= mf_ref.e_tot + 1e-7


class TestHcoreGuessUHF:
    """UHF with hcore initial guess on challenging open-shell systems.

    Starting from the bare one-electron Hamiltonian avoids bias toward a
    particular basin of attraction.  OpenOrbitalOptimizer's ADIIS/EDIIS strategy should
    find an energy at least as low as PySCF's DIIS from the same start.
    """

    @pytest.mark.parametrize("name", OPEN_SHELL_TESTS)
    def test_energy(self, name):
        """OpenOrbitalOptimizer UHF/hcore energy is at least as low as PySCF's."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf_ref = scf.UHF(mol)(init_guess="hcore")
        mf_ref.kernel()
        assert mf_ref.converged, (
            f"PySCF did not converge with init_guess=hcore on {name}"
        )

        mf_ooo = scf.UHF(mol)(init_guess="hcore")
        mf_ooo = open_orbital_optimizer(mf_ooo, config={"maximum_iterations": 200})
        mf_ooo.kernel()

        assert mf_ooo.converged, (
            f"OpenOrbitalOptimizer did not converge with init_guess=hcore on {name}"
        )
        assert np.isfinite(mf_ooo.e_tot)
        assert mf_ooo.e_tot <= mf_ref.e_tot + 1e-7
