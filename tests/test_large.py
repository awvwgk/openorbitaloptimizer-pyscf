"""
Larger system tests for OpenOrbitalOptimizer — transition-metal complexes,
halogen-bonded systems, and heavy-element radicals.

These tests verify that OpenOrbitalOptimizer scales correctly to systems with many basis
functions, heavy elements (I, Pb), and challenging electronic structure.
"""

import numpy as np
import pytest
from pyscf import scf

from openorbitaloptimizer.pyscf import open_orbital_optimizer

from ._molecules import get_mole

CLOSED_SHELL_TESTS = ["C5H3INO2P,sto-3g", "C7H6IP,sto-3g"]
OPEN_SHELL_TESTS = ["C3H9Pb,def2-svp"]


class TestRHF:
    """Restricted Hartree-Fock on larger / heavier systems."""

    @pytest.mark.parametrize("name", CLOSED_SHELL_TESTS)
    def test_energy(self, name):
        """OpenOrbitalOptimizer RHF energy matches PySCF on larger systems."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf_ref = scf.RHF(mol)
        mf_ref.kernel()
        assert mf_ref.converged, f"PySCF did not converge on {name}"

        mf_ooo = scf.RHF(mol)
        mf_ooo = open_orbital_optimizer(mf_ooo)
        mf_ooo.kernel()

        assert mf_ooo.converged, f"OpenOrbitalOptimizer did not converge on {name}"
        assert np.isfinite(mf_ooo.e_tot)
        assert mf_ooo.e_tot <= mf_ref.e_tot + 1e-7

    @pytest.mark.parametrize("name", ["Fe(CO)5,sto-3g"])
    def test_convergence(self, name):
        """OpenOrbitalOptimizer converges on a transition-metal complex."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf_ooo = scf.RHF(mol)
        mf_ooo = open_orbital_optimizer(
            mf_ooo, config={"maximum_iterations": 300, "convergence_threshold": 1e-7}
        )
        mf_ooo.kernel()

        assert mf_ooo.converged, f"OpenOrbitalOptimizer did not converge on {name}"
        assert np.isfinite(mf_ooo.e_tot)
        assert len(mf_ooo.open_orbital_optimizer_state.e_tot_per_cycle) > 1


class TestUHF:
    """Unrestricted Hartree-Fock on larger / heavier open-shell systems."""

    @pytest.mark.parametrize("name", OPEN_SHELL_TESTS)
    def test_energy(self, name):
        """OpenOrbitalOptimizer UHF energy matches PySCF on larger systems."""
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
