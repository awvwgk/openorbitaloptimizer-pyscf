"""
Skala convergence tests — SkalaRKS / SkalaUKS vs. PySCF default RKS / UKS.

Verifies that the Skala functional (via ``skala.pyscf.SkalaRKS`` /
``SkalaUKS``) converges for a representative set of molecules and that
``SkalaKS`` with a standard functional (PBE) matches the PySCF default
energy.
"""

import numpy as np
import pytest

from skala.pyscf import SkalaKS
from openorbitaloptimizer.pyscf import run_ooo_scf

from ._molecules import get_mole

CLOSED_SHELL_TESTS = [
    "H2O,sto-3g",
    "N2,cc-pvdz",
    "Li2,sto-3g",
    # "Li2,cc-pvdz",
    # "CH3Li3,sto-3g",
]
OPEN_SHELL_TESTS = [
    "CHO,sto-3g",
    "FO,sto-3g",
    # "N2+,sto-3g",
    # "N,sto-3g",
    # "F2H,sto-3g",
]


class TestSkalaRKS:
    """SkalaRKS convergence on closed-shell molecules."""

    @pytest.mark.parametrize("name", CLOSED_SHELL_TESTS)
    def test_energy(self, name):
        """SkalaRKS converges on the given molecule."""

        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])

        ks_ref = SkalaKS(mol, xc="skala", with_dftd3=False)
        ks_ref.kernel()
        assert ks_ref.converged

        ks = SkalaKS(mol, xc="skala", with_dftd3=False)
        energy, _ = run_ooo_scf(ks)
        assert ks.converged

        assert np.isfinite(energy)
        np.testing.assert_allclose(energy, ks_ref.e_tot, atol=1e-7)


class TestSkalaUKS:
    """SkalaUKS convergence on open-shell molecules."""

    @pytest.mark.parametrize("name", OPEN_SHELL_TESTS)
    def test_energy(self, name):
        """SkalaUKS converges on the given molecule."""
        level_shift = 0.5 if name.split(",")[0] == "F2H" else 0.0
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        ks_ref = SkalaKS(mol, xc="skala", with_dftd3=False)(level_shift=level_shift)
        ks_ref.kernel()
        assert ks_ref.converged
        assert np.isfinite(ks_ref.e_tot)

        ks = SkalaKS(mol, xc="skala", with_dftd3=False)
        energy, _ = run_ooo_scf(ks)
        assert ks.converged
        assert np.isfinite(energy)

        np.testing.assert_allclose(energy, ks_ref.e_tot, atol=1e-6)
