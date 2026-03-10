OpenOrbitalOptimizer for PySCF
==============================

This repository provides an implementation of the [OpenOrbitalOptimizer](https://github.com/susilehtola/openorbitaloptimizer) for the [PySCF](https://pyscf.org/) quantum chemistry package.


Installation
------------

The recommended way to install the OpenOrbitalOptimizer for PySCF is via [Conda](https://docs.conda.io/en/latest/).
You can create a new Conda environment with the required dependencies using the provided `environment.yml` file:

```bash
mamba env create -f environment.yml
mamba activate ooo-python
```

After activating the environment, you can install the OpenOrbitalOptimizer for PySCF using pip:

```bash
pip install openorbitaloptimizer-pyscf
```

Usage
-----

You can use the OpenOrbitalOptimizer for PySCF in your Python scripts as follows:

```python
from pyscf import gto, scf
from openorbitaloptimizer.pyscf import open_orbital_optimizer

mol = gto.M(
    atom='H 0 0 0; H 0 0 1',
    basis='sto-3g',
)
mf = scf.RHF(mol)
mf = open_orbital_optimizer(mf)
energy = mf.kernel()

print(f"Optimized energy: {energy}")
```
