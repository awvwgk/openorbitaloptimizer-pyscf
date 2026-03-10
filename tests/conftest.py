"""
Conftest for OpenOrbitalOptimizer tests.
"""

import os

# Prevent OpenMP / MKL / OpenBLAS from spawning excessive threads,
# which oversubscribes the CPUs when pytest runs many tests.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
