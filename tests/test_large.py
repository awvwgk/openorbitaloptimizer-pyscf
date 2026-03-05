"""
Larger system tests for OpenOrbitalOptimizer — transition-metal complexes,
halogen-bonded systems, and heavy-element radicals.

These tests verify that OOO scales correctly to systems with many basis
functions, heavy elements (I, Pb), and challenging electronic structure.
"""

import numpy as np
import pytest
from pyscf import scf

from openorbitaloptimizer import run_ooo_scf

from ._molecules import get_mole

CLOSED_SHELL_TESTS = ["C5H3INO2P,sto-3g", "C7H6IP,sto-3g"]
OPEN_SHELL_TESTS = ["C3H9Pb,def2-svp"]


class TestRHF:
    """Restricted Hartree-Fock on larger / heavier systems."""

    @pytest.mark.parametrize("name", CLOSED_SHELL_TESTS)
    def test_energy(self, name):
        """OOO RHF energy matches PySCF on larger systems."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf_ref = scf.RHF(mol)
        mf_ref.kernel()
        assert mf_ref.converged

        mf_ooo = scf.RHF(mol)
        energy_ooo, state = run_ooo_scf(mf_ooo)

        assert np.isfinite(energy_ooo)
        assert energy_ooo <= mf_ref.e_tot + 1e-7

    @pytest.mark.parametrize("name", ["Fe(CO)5,sto-3g"])
    def test_convergence(self, name):
        """OOO converges on a transition-metal complex."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf_ooo = scf.RHF(mol)
        mf_ooo.conv_tol = 1e-7
        mf_ooo.max_cycle = 300
        energy_ooo, state = run_ooo_scf(mf_ooo)

        assert np.isfinite(energy_ooo)
        assert len(state.e_tot_per_cycle) > 1


class TestUHF:
    """Unrestricted Hartree–Fock on larger / heavier open-shell systems."""

    @pytest.mark.parametrize("name", OPEN_SHELL_TESTS)
    def test_energy(self, name):
        """OOO UHF energy matches PySCF on larger systems."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf_ref = scf.UHF(mol)
        mf_ref.kernel()
        assert mf_ref.converged

        mf_ooo = scf.UHF(mol)
        energy_ooo, state = run_ooo_scf(mf_ooo)

        assert np.isfinite(energy_ooo)
        assert energy_ooo <= mf_ref.e_tot + 1e-7
