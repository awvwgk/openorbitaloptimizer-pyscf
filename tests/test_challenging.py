"""
Challenging molecule tests where OpenOrbitalOptimizer converges to a *lower*
SCF energy than PySCF's default solver.

These systems expose saddle-point convergence, self-interaction error, or
multi-minimum PES landscapes where OOO's ADIIS/EDIIS strategy is more robust
than PySCF's DIIS.
"""

import numpy as np
import pytest
from pyscf import dft, scf

from openorbitaloptimizer import run_ooo_scf

from ._molecules import get_mole

CLOSED_SHELL_TESTS = ["PCl3,sto-3g"]
OPEN_SHELL_TESTS = ["ClO2,sto-3g", "C2H,sto-3g", "(H2O)2+,sto-3g"]


class TestRHF:
    """Restricted Hartree-Fock on challenging closed-shell systems.

    PySCF's default DIIS may converge to a higher local minimum.
    OOO should find an energy at least as low as PySCF's default.
    """

    @pytest.mark.parametrize("name", CLOSED_SHELL_TESTS)
    def test_energy(self, name):
        """OOO RHF energy is at least as low as PySCF's default."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf_ref = scf.RHF(mol)
        mf_ref.kernel()
        assert mf_ref.converged

        mf_ooo = scf.RHF(mol)
        energy_ooo, state = run_ooo_scf(mf_ooo)

        assert np.isfinite(energy_ooo)
        assert energy_ooo <= mf_ref.e_tot + 1e-7


class TestUHF:
    """Unrestricted Hartree–Fock on challenging open-shell systems.

    With PySCF's initial guess, both solvers start from the same density.
    OOO should converge to an energy at least as low as PySCF's default.
    """

    @pytest.mark.parametrize("name", OPEN_SHELL_TESTS)
    def test_energy(self, name):
        """OOO UHF energy is at least as low as PySCF's default."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf_ref = scf.UHF(mol)
        mf_ref.kernel()
        assert mf_ref.converged

        mf_ooo = scf.UHF(mol)
        energy_ooo, state = run_ooo_scf(mf_ooo)

        assert np.isfinite(energy_ooo)
        assert energy_ooo <= mf_ref.e_tot + 1e-7


class TestUKS:
    """Unrestricted Kohn-Sham DFT on challenging open-shell systems."""

    @pytest.mark.parametrize("name", OPEN_SHELL_TESTS[:1])
    def test_energy(self, name):
        """OOO UKS/TPSS energy is at least as low as PySCF's default."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf_ref = dft.UKS(mol, xc="tpss")
        mf_ref.kernel()
        assert mf_ref.converged

        mf_ooo = dft.UKS(mol, xc="tpss")
        energy_ooo, state = run_ooo_scf(mf_ooo)

        assert np.isfinite(energy_ooo)
        assert energy_ooo <= mf_ref.e_tot + 1e-7


class TestHcoreGuessRHF:
    """RHF with hcore initial guess on challenging systems.

    Starting from the bare one-electron Hamiltonian avoids bias toward a
    particular basin of attraction.  OOO's ADIIS/EDIIS strategy should
    find an energy at least as low as PySCF's DIIS from the same start.
    """

    @pytest.mark.parametrize("name", CLOSED_SHELL_TESTS)
    def test_energy(self, name):
        """OOO RHF/hcore energy is at least as low as PySCF's."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf_ref = scf.RHF(mol)(init_guess="hcore")
        mf_ref.kernel()
        assert mf_ref.converged

        mf_ooo = scf.RHF(mol)(init_guess="hcore")
        energy_ooo, state = run_ooo_scf(mf_ooo)

        assert np.isfinite(energy_ooo)
        assert energy_ooo <= mf_ref.e_tot + 1e-7


class TestHcoreGuessUHF:
    """UHF with hcore initial guess on challenging open-shell systems.

    Starting from the bare one-electron Hamiltonian avoids bias toward a
    particular basin of attraction.  OOO's ADIIS/EDIIS strategy should
    find an energy at least as low as PySCF's DIIS from the same start.
    """

    @pytest.mark.parametrize("name", OPEN_SHELL_TESTS)
    def test_energy(self, name):
        """OOO UHF/hcore energy is at least as low as PySCF's."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf_ref = scf.UHF(mol)(init_guess="hcore")
        mf_ref.kernel()
        assert mf_ref.converged

        mf_ooo = scf.UHF(mol)(init_guess="hcore")
        energy_ooo, state = run_ooo_scf(mf_ooo)

        assert np.isfinite(energy_ooo)
        assert energy_ooo <= mf_ref.e_tot + 1e-7
