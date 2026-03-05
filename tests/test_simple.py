"""
Simple / small molecule tests where OpenOrbitalOptimizer and PySCF converge
to the same SCF energy.

Covers RHF, UHF, RKS (TPSS), UKS (TPSS), MO orthonormality, and SCFState
diagnostics on easy-to-converge systems.
"""

import numpy as np
import pytest
from pyscf import dft, scf

from openorbitaloptimizer import run_ooo_scf

from ._molecules import get_mole

CLOSED_SHELL_TESTS = ["H2O,sto-3g", "N2,cc-pvdz", "C2H4,6-31g*", "H3+,def2-tzvp"]
OPEN_SHELL_TESTS = ["C2H,def2-svp"]


class TestRHF:
    """Restricted Hartree-Fock via OpenOrbitalOptimizer vs. PySCF."""

    @pytest.mark.parametrize("name", CLOSED_SHELL_TESTS)
    def test_energy(self, name):
        """OOO energy matches PySCF RHF."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf_ref = scf.RHF(mol)
        mf_ref.kernel()
        assert mf_ref.converged

        mf_ooo = scf.RHF(mol)
        energy_ooo, state = run_ooo_scf(mf_ooo)

        np.testing.assert_allclose(energy_ooo, mf_ref.e_tot, atol=1e-7)

    @pytest.mark.parametrize("name", CLOSED_SHELL_TESTS[:1])
    def test_mo_coeff_orthonormal(self, name):
        """After OOO solve, C^T S C should be identity."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf = scf.RHF(mol)
        run_ooo_scf(mf)

        S = mf.get_ovlp()
        C = mf.mo_coeff
        CtSC = C.T @ S @ C
        np.testing.assert_allclose(CtSC, np.eye(CtSC.shape[0]), atol=1e-10)


class TestUHF:
    """Unrestricted Hartree-Fock on simple open-shell systems."""

    @pytest.mark.parametrize("name", OPEN_SHELL_TESTS)
    def test_energy(self, name: str) -> None:
        """OOO UHF energy matches PySCF UHF."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf_ref = scf.UHF(mol)
        mf_ref.kernel()
        assert mf_ref.converged

        mf_ooo = scf.UHF(mol)
        energy_ooo, state = run_ooo_scf(mf_ooo)

        np.testing.assert_allclose(energy_ooo, mf_ref.e_tot, atol=1e-6)


class TestRKS:
    """Restricted Kohn-Sham DFT via OOO vs. PySCF."""

    @pytest.mark.parametrize("name", CLOSED_SHELL_TESTS)
    def test_energy(self, name):
        """OOO RKS energy matches PySCF RKS."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf_ref = dft.RKS(mol, xc="pbe")
        mf_ref.grids.level = 1
        mf_ref.kernel()
        assert mf_ref.converged

        mf_ooo = dft.RKS(mol, xc="pbe")
        mf_ooo.grids.level = 1
        energy_ooo, state = run_ooo_scf(mf_ooo)

        np.testing.assert_allclose(energy_ooo, mf_ref.e_tot, atol=1e-7)


class TestUKS:
    """Unrestricted Kohn-Sham DFT on simple systems."""

    @pytest.mark.parametrize("name", OPEN_SHELL_TESTS)
    def test_energy(self, name):
        """OOO UKS energy matches PySCF UKS."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf_ref = dft.UKS(mol, xc="pbe")
        mf_ref.grids.level = 1
        mf_ref.kernel()
        assert mf_ref.converged

        mf_ooo = dft.UKS(mol, xc="pbe")
        mf_ooo.grids.level = 1
        energy_ooo, state = run_ooo_scf(mf_ooo)

        np.testing.assert_allclose(energy_ooo, mf_ref.e_tot, atol=1e-6)


class TestSCFState:
    """Verify the SCFState object is populated correctly."""

    @pytest.mark.parametrize("name", CLOSED_SHELL_TESTS[:1])
    def test_state_populated(self, name):
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf = scf.RHF(mol)
        _, state = run_ooo_scf(mf)

        assert state.ntries == 1
        assert len(state.e_tot_per_cycle) >= 2
        assert state.wall_time is not None and state.wall_time > 0
        assert state.nfock > 0
