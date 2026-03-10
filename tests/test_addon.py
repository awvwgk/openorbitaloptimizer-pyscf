"""
Tests for the ``open_orbital_optimizer()`` PySCF addon that overrides
``mf.kernel()``.

Verifies that the addon integrates cleanly with PySCF's dynamic class
mechanism: correct MRO, idempotent wrapping, ``undo_open_orbital_optimizer()``, and energy
agreement with both the raw ``run_open_orbital_optimizer`` driver and PySCF's own solver.
"""

import numpy as np
import pytest
from pyscf import dft, scf

from openorbitaloptimizer.pyscf import (
    SCFConfig,
    SCFState,
    open_orbital_optimizer,
    run_open_orbital_optimizer,
)

from ._molecules import get_mole

CLOSED_SHELL_TESTS = ["H2O,sto-3g", "N2,cc-pvdz"]
OPEN_SHELL_TESTS = ["C2H,def2-svp", "F2H,sto-3g"]


class TestAddonMechanism:
    """Structural tests for the ``open_orbital_optimizer()`` addon."""

    def test_class_name(self):
        """Dynamic class name contains 'OpenOrbitalOptimizer' mixin tag."""
        mol = get_mole("H2O", basis="sto-3g")
        mf = open_orbital_optimizer(scf.RHF(mol))
        assert "OpenOrbitalOptimizer" in type(mf).__name__

    def test_isinstance_original(self):
        """Wrapped object is still an instance of the original SCF class."""
        mol = get_mole("H2O", basis="sto-3g")
        mf_orig = scf.RHF(mol)
        mf = open_orbital_optimizer(mf_orig)
        assert isinstance(mf, type(mf_orig))

    def test_isinstance_open_orbital_optimizer_mixin(self):
        """Wrapped object is an instance of the _OpenOrbitalOptimizer mixin."""
        from openorbitaloptimizer.pyscf import _OpenOrbitalOptimizer

        mol = get_mole("H2O", basis="sto-3g")
        mf = open_orbital_optimizer(scf.RHF(mol))
        assert isinstance(mf, _OpenOrbitalOptimizer)

    def test_idempotent(self):
        """Wrapping twice returns the same object."""
        mol = get_mole("H2O", basis="sto-3g")
        mf1 = open_orbital_optimizer(scf.RHF(mol))
        mf2 = open_orbital_optimizer(mf1)
        assert mf1 is mf2

    def test_idempotent_config_update(self):
        """Wrapping twice with a new config updates the config."""
        mol = get_mole("H2O", basis="sto-3g")
        cfg1 = SCFConfig(maximum_iterations=50)
        cfg2 = SCFConfig(maximum_iterations=200)
        mf = open_orbital_optimizer(scf.RHF(mol), config=cfg1)
        assert mf.open_orbital_optimizer_config.maximum_iterations == 50
        open_orbital_optimizer(mf, config=cfg2)
        assert mf.open_orbital_optimizer_config.maximum_iterations == 200

    def test_undo_open_orbital_optimizer(self):
        """undo_open_orbital_optimizer() restores the original SCF class."""
        mol = get_mole("H2O", basis="sto-3g")
        mf_orig = scf.RHF(mol)
        mf_ooo = open_orbital_optimizer(mf_orig)
        mf_restored = mf_ooo.undo_open_orbital_optimizer()
        assert type(mf_restored) is scf.rhf.RHF
        assert not hasattr(mf_restored, "open_orbital_optimizer_config")
        assert not hasattr(mf_restored, "open_orbital_optimizer_state")

    def test_default_config(self):
        """Without explicit config, default SCFConfig is used."""
        mol = get_mole("H2O", basis="sto-3g")
        mf = open_orbital_optimizer(scf.RHF(mol))
        assert mf.open_orbital_optimizer_config == SCFConfig()

    def test_custom_config(self):
        """Custom config is stored on the object."""
        mol = get_mole("H2O", basis="sto-3g")
        cfg = SCFConfig(maximum_iterations=200, convergence_threshold=1e-8)
        mf = open_orbital_optimizer(scf.RHF(mol), config=cfg)
        assert mf.open_orbital_optimizer_config is cfg


class TestAddonRHF:
    """RHF energy tests through the addon's ``kernel()`` method."""

    @pytest.mark.parametrize("name", CLOSED_SHELL_TESTS)
    def test_energy_matches_pyscf(self, name):
        """open_orbital_optimizer(RHF).kernel() matches PySCF RHF energy."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])

        mf_ref = scf.RHF(mol)
        mf_ref.kernel()
        assert mf_ref.converged

        mf_ooo = open_orbital_optimizer(scf.RHF(mol))
        e_ooo = mf_ooo.kernel()

        assert mf_ooo.converged
        assert np.isfinite(e_ooo)
        np.testing.assert_allclose(e_ooo, mf_ref.e_tot, atol=1e-7)

    @pytest.mark.parametrize("name", CLOSED_SHELL_TESTS[:1])
    def test_energy_matches_run_open_orbital_optimizer(self, name):
        """open_orbital_optimizer(RHF).kernel() gives the same result as run_open_orbital_optimizer()."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])

        mf1 = scf.RHF(mol)
        e1, _ = run_open_orbital_optimizer(mf1)

        mf2 = open_orbital_optimizer(scf.RHF(mol))
        e2 = mf2.kernel()

        np.testing.assert_allclose(e2, e1, atol=1e-10)

    @pytest.mark.parametrize("name", CLOSED_SHELL_TESTS[:1])
    def test_state_recorded(self, name):
        """open_orbital_optimizer_state is populated after kernel()."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf = open_orbital_optimizer(scf.RHF(mol))
        mf.kernel()

        assert isinstance(mf.open_orbital_optimizer_state, SCFState)
        assert len(mf.open_orbital_optimizer_state.e_tot_per_cycle) > 0
        assert mf.open_orbital_optimizer_state.nfock > 0

    @pytest.mark.parametrize("name", CLOSED_SHELL_TESTS[:1])
    def test_mo_coeff_orthonormal(self, name):
        """C^T S C = I after kernel()."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])
        mf = open_orbital_optimizer(scf.RHF(mol))
        mf.kernel()

        S = mf.get_ovlp()
        CtSC = mf.mo_coeff.T @ S @ mf.mo_coeff
        np.testing.assert_allclose(CtSC, np.eye(CtSC.shape[0]), atol=1e-10)


class TestAddonUHF:
    """UHF energy tests through the addon's ``kernel()`` method."""

    @pytest.mark.parametrize("name", OPEN_SHELL_TESTS)
    def test_energy_matches_pyscf(self, name):
        """open_orbital_optimizer(UHF).kernel() matches PySCF UHF energy."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])

        mf_ref = scf.UHF(mol)
        mf_ref.kernel()
        assert mf_ref.converged

        mf_ooo = open_orbital_optimizer(scf.UHF(mol))
        e_ooo = mf_ooo.kernel()

        assert mf_ooo.converged
        assert np.isfinite(e_ooo)
        np.testing.assert_allclose(e_ooo, mf_ref.e_tot, atol=1e-6)


class TestAddonDFT:
    """DFT energy tests through the addon's ``kernel()`` method."""

    @pytest.mark.parametrize("name", CLOSED_SHELL_TESTS[:1])
    def test_rks_energy(self, name):
        """open_orbital_optimizer(RKS).kernel() matches PySCF RKS/PBE energy."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])

        mf_ref = dft.RKS(mol, xc="pbe")
        mf_ref.kernel()
        assert mf_ref.converged

        mf_ooo = open_orbital_optimizer(dft.RKS(mol, xc="pbe"))
        e_ooo = mf_ooo.kernel()

        assert mf_ooo.converged
        np.testing.assert_allclose(e_ooo, mf_ref.e_tot, atol=1e-7)

    @pytest.mark.parametrize("name", OPEN_SHELL_TESTS[:1])
    def test_uks_energy(self, name):
        """open_orbital_optimizer(UKS).kernel() matches PySCF UKS/TPSS energy."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])

        mf_ref = dft.UKS(mol, xc="tpss")
        mf_ref.kernel()
        assert mf_ref.converged

        mf_ooo = open_orbital_optimizer(dft.UKS(mol, xc="tpss"))
        e_ooo = mf_ooo.kernel()

        assert mf_ooo.converged
        np.testing.assert_allclose(e_ooo, mf_ref.e_tot, atol=1e-6)


class TestAddonComposition:
    """Test composing the OpenOrbitalOptimizer addon with other PySCF addons."""

    @pytest.mark.parametrize("name", CLOSED_SHELL_TESTS[:1])
    def test_density_fit_ooo(self, name):
        """open_orbital_optimizer(density_fit(RHF)) composes correctly."""
        mol = get_mole(name.split(",")[0], basis=name.split(",")[1])

        mf = scf.RHF(mol).density_fit()
        mf_ooo = open_orbital_optimizer(mf)

        # Should still be a density-fitted object
        assert hasattr(mf_ooo, "with_df")

        e = mf_ooo.kernel()
        assert mf_ooo.converged
        assert np.isfinite(e)
