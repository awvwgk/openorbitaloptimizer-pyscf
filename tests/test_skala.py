"""
Skala convergence tests — SkalaRKS / SkalaUKS vs. PySCF default RKS / UKS.

Verifies that the Skala functional (via ``skala.pyscf.SkalaKS``) converges
for a representative set of closed- and open-shell molecules and that the
OpenOrbitalOptimizer energy matches the PySCF default SCF energy.

.. note::

   Every Fock build with the Skala neural-network functional is
   significantly more expensive than a conventional DFT Fock build, so
   these tests have a longer runtime than the other convergence tests.
   The OpenOrbitalOptimizer convergence criterion is the orbital gradient
   norm.  With the Skala NN functional the gradient has a numerical noise
   floor of roughly 1e-5, so the OOO convergence threshold is set to 1e-4
   to guarantee convergence while still being tight enough for a
   meaningful test.  PySCF's own SCF uses *energy-change* convergence
   (``conv_tol``) which is less affected by NN noise.
"""

import numpy as np
import pytest

from skala.pyscf import SkalaKS
from openorbitaloptimizer.pyscf import open_orbital_optimizer

from ._molecules import get_mole

# PySCF reference uses energy-change convergence – 1e-6 is fine.
REF_CONV_TOL = 1e-6

# OOO uses orbital-gradient convergence – NN noise floor is ~1e-5,
# so we must be above that.
OOO_CONV_TOL = 1e-4

# Energy comparison tolerance (atol) – at the 1e-4 gradient level the
# two energies typically agree to ~1e-4 Ha or better.
ENERGY_ATOL = 1e-4

CLOSED_SHELL_TESTS = [
    "H2O,sto-3g",
    "N2,cc-pvdz",
    "Li2,sto-3g",
]
OPEN_SHELL_TESTS = [
    "CHO,sto-3g",
    "FO,sto-3g",
]


class TestSkalaRKS:
    """SkalaRKS convergence on closed-shell molecules."""

    @pytest.mark.parametrize("name", CLOSED_SHELL_TESTS)
    def test_energy(self, name):
        """SkalaRKS converges on the given molecule."""

        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])

        ks_ref = SkalaKS(mol, xc="skala", with_dftd3=False)(conv_tol=REF_CONV_TOL)
        ks_ref.kernel()
        assert ks_ref.converged, f"PySCF SkalaRKS did not converge on {name}"

        ks = SkalaKS(mol, xc="skala", with_dftd3=False)
        ks = open_orbital_optimizer(ks, config={"convergence_threshold": OOO_CONV_TOL})
        ks.kernel()

        assert ks.converged, f"OpenOrbitalOptimizer SkalaRKS did not converge on {name}"
        assert np.isfinite(ks.e_tot), (
            f"OpenOrbitalOptimizer SkalaRKS returned non-finite energy on {name}"
        )
        np.testing.assert_allclose(ks.e_tot, ks_ref.e_tot, atol=ENERGY_ATOL)


class TestSkalaUKS:
    """SkalaUKS convergence on open-shell molecules."""

    @pytest.mark.parametrize("name", OPEN_SHELL_TESTS)
    def test_energy(self, name):
        """SkalaUKS converges on the given molecule."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])

        ks_ref = SkalaKS(mol, xc="skala", with_dftd3=False)(conv_tol=REF_CONV_TOL)
        ks_ref.kernel()
        assert ks_ref.converged, f"PySCF SkalaUKS did not converge on {name}"
        assert np.isfinite(ks_ref.e_tot), (
            f"PySCF SkalaUKS returned non-finite energy on {name}"
        )

        ks = SkalaKS(mol, xc="skala", with_dftd3=False)
        ks = open_orbital_optimizer(ks, config={"convergence_threshold": OOO_CONV_TOL})
        ks.kernel()

        assert ks.converged, f"OpenOrbitalOptimizer SkalaUKS did not converge on {name}"
        assert np.isfinite(ks.e_tot), (
            f"OpenOrbitalOptimizer SkalaUKS returned non-finite energy on {name}"
        )
        np.testing.assert_allclose(ks.e_tot, ks_ref.e_tot, atol=ENERGY_ATOL)
