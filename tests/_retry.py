"""
Retry mechanism for SCF calculations in PySCF.

Vendored (simplified) from https://github.com/microsoft/skala/blob/main/src/skala/pyscf/retry.py
Copyright (c) Microsoft Corporation — MIT License.

Only the parts needed for benchmarking are included here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pyscf.scf.hf import SCF

SMALL_GAP = 0.1


def _min_gap(gaps: list[float | None]) -> float | None:
    finite_gaps = [gap for gap in gaps[2:-1] if gap is not None]
    return min(finite_gaps) if finite_gaps else None


@dataclass
class SCFState:
    ntries: int = 0
    cycles: list[int] = field(default_factory=list)
    grid_size: int | None = None
    e_tot_per_cycle: list[float] = field(default_factory=list)
    gradient_norm_per_cycle: list[float | None] = field(default_factory=list)
    dm_change_per_cycle: list[float | None] = field(default_factory=list)
    homo_lumo_gap_up_per_cycle: list[float | None] = field(default_factory=list)
    homo_lumo_gap_down_per_cycle: list[float | None] = field(default_factory=list)

    def add_callback(self, scf: SCF) -> None:
        scf.pre_kernel = self.pre_kernel_callback
        scf.callback = self.post_cycle_callback
        scf.post_kernel = self.post_kernel_callback

    def pre_kernel_callback(self, envs: dict[str, Any]) -> None:
        scf: SCF = envs["mf"]
        e_tot = envs["e_tot"]
        self.ntries += 1
        self.cycles = [0]
        self.grid_size = scf.grids.size if hasattr(scf, "grids") else None
        self.e_tot_per_cycle = [e_tot]
        self.gradient_norm_per_cycle = [None]
        self.dm_change_per_cycle = [None]
        self.homo_lumo_gap_up_per_cycle = [None]
        self.homo_lumo_gap_down_per_cycle = [None]

    def post_cycle_callback(self, envs: dict[str, Any]) -> None:
        e_tot = envs["e_tot"]
        norm_gorb = envs["norm_gorb"]
        scf: SCF = envs["mf"]

        if "cycle" in envs:
            self.cycles.append(int(envs["cycle"] + 1))
            mo_energy = envs["mo_energy"]
        elif "imacro" in envs:
            self.cycles.append(envs["imacro"] + 1)
            mo_energy, _ = scf._scf.canonicalize(
                envs["mo_coeff"], envs["mo_occ"], envs["fock"]
            )
        else:
            mo_energy = envs["mo_energy"]

        self.e_tot_per_cycle.append(e_tot)
        self.gradient_norm_per_cycle.append(norm_gorb)
        if "norm_ddm" not in envs:
            envs["norm_ddm"] = np.linalg.norm(envs["dm"] - envs["dm_last"])
        self.dm_change_per_cycle.append(envs["norm_ddm"])

        if not isinstance(mo_energy, list) and len(mo_energy.shape) == 1:
            occ_mask = envs["mo_occ"] > 0
            vir_mask = envs["mo_occ"] == 0
            if np.any(occ_mask) and np.any(vir_mask):
                gap = float(np.min(mo_energy[vir_mask]) - np.max(mo_energy[occ_mask]))
            else:
                gap = None
            self.homo_lumo_gap_up_per_cycle.append(gap)
            self.homo_lumo_gap_down_per_cycle.append(gap)
        else:
            for spin in (0, 1):
                occ_mask = envs["mo_occ"][spin] > 0
                vir_mask = envs["mo_occ"][spin] == 0
                if np.any(occ_mask) and np.any(vir_mask):
                    gap = float(
                        np.min(mo_energy[spin][vir_mask])
                        - np.max(mo_energy[spin][occ_mask])
                    )
                else:
                    gap = None
                if spin == 0:
                    self.homo_lumo_gap_up_per_cycle.append(gap)
                else:
                    self.homo_lumo_gap_down_per_cycle.append(gap)

    def post_kernel_callback(self, envs: dict[str, Any]) -> None:
        scf: SCF = envs["mf"]
        if scf.conv_check:
            envs["cycle"] += 1
            scf.callback(envs)

    def get_gap(self) -> tuple[float | None, float | None]:
        return (
            _min_gap(self.homo_lumo_gap_up_per_cycle),
            _min_gap(self.homo_lumo_gap_down_per_cycle),
        )


def increment_level_shift(
    level_shift: float,
    max_level_shift: float = 0.5,
    level_shift_init: float = 0.1,
    level_shift_increment: float = 0.2,
) -> float:
    return (
        min(level_shift + level_shift_increment, max_level_shift)
        if level_shift > 0
        else level_shift_init
    )


def retry_scf(scf: SCF) -> tuple[SCF, SCFState]:
    """Retry the SCF calculation if it fails due to convergence issues."""
    from pyscf.soscf.newton_ah import _CIAH_SOSCF

    state = SCFState()
    state.add_callback(scf)

    scf.kernel()
    if scf.converged:
        return scf, state

    if isinstance(scf, _CIAH_SOSCF):
        return scf, state

    init_config: dict[str, float | int] = {
        "damp": scf.damp,
        "diis_start_cycle": scf.diis_start_cycle,
    }
    scf = scf.set(damp=0.5, diis_start_cycle=7)

    scf.kernel()
    if scf.converged:
        return scf, state

    gaps = state.get_gap()
    if any(gap is not None and gap < SMALL_GAP for gap in gaps):
        while scf.level_shift != increment_level_shift(scf.level_shift):
            scf.set(
                **init_config,
                level_shift=increment_level_shift(scf.level_shift),
            )
            scf.kernel()
            gaps = state.get_gap()
            sufficient_gap = all(gap is not None and gap >= SMALL_GAP for gap in gaps)
            scf.converged = scf.converged and sufficient_gap
            if scf.converged:
                return scf, state
            if sufficient_gap:
                break

    scf.set(**init_config, level_shift=0.0)
    scf = scf.newton()
    scf.kernel()
    if scf.converged:
        return scf, state

    return scf, state
