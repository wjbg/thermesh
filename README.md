<h1 align="center">
<img src="fig/thermesh.svg" width="440">
</h1><br>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

Finite element code for transient one-dimensional heat conduction in Python.

## Introduction

This repository presents a one-dimensional transient heat conduction
problems using both linear and quadratic elements. The code has been
implemented in a Python package and is available for download. The
derivation of the equations is documented in this [pdf
file](thermesh-derivation.pdf).

Three validation cases are presented at the end of this document to
show that the code is implemented correctly. Further, the validation
cases can be used as examples and inspiration.

## Minimal example

Consider a domain of length L with a uniform initial temperature. The
temperature at one end is raised to a fixed value, while the other end
is kept at the initial temperature. The code below shows how to
calculate the evolution of temperature distribution in the domain with
time. A more elaborate implementation for this case, together with a
comparison with the analytical solution, can be found in
[step_change.py](src/step_change.py).

```python
import numpy as np
import thermesh as tm

# Domain information
L = 0.01
k, rho, cp = 0.72, 1560, 1450
cpeek = tm.isothermal_model(k, rho, cp)  # constitutive model

# Mesh generation using linear elements
nn = 11  # number of nodes
z = np.linspace(0, L, nn)  # node locations
mesh = tm.Mesh(z, tm.LinearElement)

# Boundary conditions
bc = [{"T": 0.0},  # T on the left
      {"T": 1.0}]  # T on the right

# Domain generation and initialization
domain = tm.Domain(mesh, cpeek, bc)
domain.set_T(np.zeros(nn))

# Solve
solver = {"dt": 0.1, "t_end": 100.0, "theta": 0.5}  # settings
t, T = tm.solve_ht(domain, solver)
```

## Install

You can simply clone the repository to your folder of choice using git:

```
git clone https://github.com/wjbg/thermesh.git
```

All functions are reasonably well-documented and the annotated examples should be sufficient to get you started.

## License

Free as defined in the [MIT](https://choosealicense.com/licenses/mit/)
license.
