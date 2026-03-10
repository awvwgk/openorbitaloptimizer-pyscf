"""
Simple / small molecule tests where OpenOrbitalOptimizer and PySCF converge
to the same SCF energy.

Covers RHF, UHF, RKS (TPSS), UKS (TPSS), MO orthonormality, and SCFState
diagnostics on easy-to-converge systems.
"""

import numpy as np
import pytest
from pyscf import dft, scf

from openorbitaloptimizer.pyscf import open_orbital_optimizer

from ._molecules import get_mole

CLOSED_SHELL_TESTS = [
    "H2O,sto-3g",
    "N2,cc-pvdz",
    "C2H4,6-31g*",
    "H3+,def2-tzvp",
    "CH3Li3,sto-3g",
    "CH3Li3,def2-svp",
    "Li2,sto-3g",
    "Li2,cc-pvdz",
]
OPEN_SHELL_TESTS = [
    "C2H,def2-svp",
    "F2H,sto-3g",
    "F2H,def2-svp",
    "CHO,sto-3g",
    "CHO,cc-pvdz",
]


class TestRHF:
    """Restricted Hartree-Fock via OpenOrbitalOptimizer vs. PySCF."""

    @pytest.mark.parametrize("name", CLOSED_SHELL_TESTS)
    def test_energy(self, name):
        """OpenOrbitalOptimizer energy matches PySCF RHF."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf_ref = scf.RHF(mol)
        mf_ref.kernel()
        assert mf_ref.converged, f"PySCF did not converge on {name}"

        mf_ooo = scf.RHF(mol)
        mf_ooo = open_orbital_optimizer(mf_ooo)
        mf_ooo.kernel()

        assert mf_ooo.converged, f"OpenOrbitalOptimizer did not converge on {name}"
        assert np.isfinite(mf_ooo.e_tot), (
            f"OpenOrbitalOptimizer returned non-finite energy on {name}"
        )
        np.testing.assert_allclose(mf_ooo.e_tot, mf_ref.e_tot, atol=1e-7)

    @pytest.mark.parametrize("name", CLOSED_SHELL_TESTS[:1])
    def test_mo_coeff_orthonormal(self, name):
        """After OpenOrbitalOptimizer solve, C^T S C should be identity."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf = scf.RHF(mol)
        mf = open_orbital_optimizer(mf)
        mf.kernel()

        S = mf.get_ovlp()
        C = mf.mo_coeff
        CtSC = C.T @ S @ C
        np.testing.assert_allclose(CtSC, np.eye(CtSC.shape[0]), atol=1e-10)


class TestUHF:
    """Unrestricted Hartree-Fock on simple open-shell systems."""

    @pytest.mark.parametrize("name", OPEN_SHELL_TESTS)
    def test_energy(self, name: str) -> None:
        """OpenOrbitalOptimizer UHF energy matches PySCF UHF."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf_ref = scf.UHF(mol)
        mf_ref.kernel()
        assert mf_ref.converged, f"PySCF did not converge on {name}"

        mf_ooo = scf.UHF(mol)
        mf_ooo = open_orbital_optimizer(mf_ooo)
        mf_ooo.kernel()

        assert mf_ooo.converged, f"OpenOrbitalOptimizer did not converge on {name}"
        assert np.isfinite(mf_ooo.e_tot), (
            f"OpenOrbitalOptimizer returned non-finite energy on {name}"
        )
        np.testing.assert_allclose(mf_ooo.e_tot, mf_ref.e_tot, atol=1e-6)


class TestRKS:
    """Restricted Kohn-Sham DFT via OpenOrbitalOptimizer vs. PySCF."""

    @pytest.mark.parametrize("name", CLOSED_SHELL_TESTS)
    def test_energy(self, name):
        """OpenOrbitalOptimizer RKS energy matches PySCF RKS."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf_ref = dft.RKS(mol, xc="pbe")
        mf_ref.grids.level = 1
        mf_ref.kernel()
        assert mf_ref.converged, f"PySCF did not converge on {name}"

        mf_ooo = dft.RKS(mol, xc="pbe")
        mf_ooo.grids.level = 1
        mf_ooo = open_orbital_optimizer(mf_ooo)
        mf_ooo.kernel()

        assert mf_ooo.converged, f"OpenOrbitalOptimizer did not converge on {name}"
        assert np.isfinite(mf_ooo.e_tot), (
            f"OpenOrbitalOptimizer returned non-finite energy on {name}"
        )
        np.testing.assert_allclose(mf_ooo.e_tot, mf_ref.e_tot, atol=1e-7)


class TestUKS:
    """Unrestricted Kohn-Sham DFT on simple systems."""

    @pytest.mark.parametrize("name", OPEN_SHELL_TESTS)
    def test_energy(self, name):
        """OpenOrbitalOptimizer UKS energy matches PySCF UKS."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf_ref = dft.UKS(mol, xc="pbe")
        mf_ref.grids.level = 1
        mf_ref.kernel()
        assert mf_ref.converged, f"PySCF did not converge on {name}"

        mf_ooo = dft.UKS(mol, xc="pbe")
        mf_ooo.grids.level = 1
        mf_ooo = open_orbital_optimizer(mf_ooo)
        mf_ooo.kernel()

        assert mf_ooo.converged, f"OpenOrbitalOptimizer did not converge on {name}"
        assert np.isfinite(mf_ooo.e_tot), (
            f"OpenOrbitalOptimizer returned non-finite energy on {name}"
        )
        np.testing.assert_allclose(mf_ooo.e_tot, mf_ref.e_tot, atol=1e-6)


class TestSCFState:
    """Verify the SCFState object is populated correctly."""

    @pytest.mark.parametrize("name", CLOSED_SHELL_TESTS[:1])
    def test_state_populated(self, name):
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf = scf.RHF(mol)
        mf = open_orbital_optimizer(mf)
        mf.kernel()
        state = mf.open_orbital_optimizer_state

        assert state.ntries == 1
        assert len(state.e_tot_per_cycle) >= 2
        assert state.wall_time is not None and state.wall_time > 0
        assert state.nfock > 0

    @pytest.mark.parametrize("name", CLOSED_SHELL_TESTS[:1])
    def test_homo_lumo_gap_per_cycle(self, name):
        """HOMO-LUMO gap is recorded for every SCF cycle (restricted)."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf = scf.RHF(mol)
        mf = open_orbital_optimizer(mf)
        mf.kernel()
        state = mf.open_orbital_optimizer_state

        ncycles = len(state.cycles)
        assert ncycles >= 2

        # One gap entry per cycle
        assert len(state.homo_lumo_gap_up_per_cycle) == ncycles
        assert len(state.homo_lumo_gap_down_per_cycle) == ncycles

        # All entries are finite positive floats (closed-shell should have a gap)
        for gap in state.homo_lumo_gap_up_per_cycle:
            assert gap is not None
            assert gap > 0.0
        for gap in state.homo_lumo_gap_down_per_cycle:
            assert gap is not None
            assert gap > 0.0

    @pytest.mark.parametrize("name", OPEN_SHELL_TESTS[:1])
    def test_homo_lumo_gap_per_cycle_unrestricted(self, name):
        """HOMO-LUMO gap is recorded for every SCF cycle (unrestricted)."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf = scf.UHF(mol)
        mf = open_orbital_optimizer(mf)
        mf.kernel()
        state = mf.open_orbital_optimizer_state

        ncycles = len(state.cycles)
        assert ncycles >= 2

        assert len(state.homo_lumo_gap_up_per_cycle) == ncycles
        assert len(state.homo_lumo_gap_down_per_cycle) == ncycles

        # At least the last entry should be a finite positive float
        assert state.homo_lumo_gap_up_per_cycle[-1] is not None
        assert state.homo_lumo_gap_up_per_cycle[-1] > 0.0
        assert state.homo_lumo_gap_down_per_cycle[-1] is not None
        assert state.homo_lumo_gap_down_per_cycle[-1] > 0.0

    @pytest.mark.parametrize("name", CLOSED_SHELL_TESTS[:1])
    def test_dm_change_per_cycle(self, name):
        """Density-matrix change is recorded for every SCF cycle."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf = scf.RHF(mol)
        mf = open_orbital_optimizer(mf)
        mf.kernel()
        state = mf.open_orbital_optimizer_state

        ncycles = len(state.cycles)
        assert ncycles >= 2

        # One dm_change entry per cycle
        assert len(state.dm_change_per_cycle) == ncycles

        # First entry may be None (no previous DM), rest must be non-negative
        for dm_change in state.dm_change_per_cycle:
            if dm_change is not None:
                assert dm_change >= 0.0

        # Converged SCF should show decreasing dm_change towards the end
        non_none = [v for v in state.dm_change_per_cycle if v is not None]
        assert len(non_none) >= 1
        assert non_none[-1] < non_none[0]
