"""
OpenOrbitalOptimizer — Python bindings for the OpenOrbitalOptimizer SCF library.

Provides a robust SCF solver using ADIIS/EDIIS/DIIS convergence acceleration
that can be used as a drop-in replacement for PySCF's built-in SCF solver.
"""

from openorbitaloptimizer._core import SCFSolver

__all__ = ["SCFSolver"]

try:
    from openorbitaloptimizer.pyscf import run_ooo_scf  # noqa: F401

    __all__.append("run_ooo_scf")
except ImportError:
    # PySCF not installed — skip the driver
    pass
