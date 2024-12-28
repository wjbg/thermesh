# Thermesh
#
# Example
# Two connected domains

import thermesh as tm
import numpy as np
import matplotlib.pyplot as plt


#
# Constitutive models
# -----------------------------------------------------------------
#
# Thermesh provides two types of constitutive models, namely an
# isothermal and a piece-wise linear mode. Both models take
# temperature as an input and provide a dictionary with the
# conductivity, the density and the specific heat as an output.

# Constitutive model (for C/PAEK)
kz = np.array([[20.0, 0.8],  # heat conductivity [W/m.K] as function of T
               [400.0, 0.8]])
rho = np.array([[20.0, 1540.0],  # density [kg/m3] as function of T
               [400.0, 1540.0]])
cp = np.array([[20.0, 815.4],  # specific heat [J/kg.K] as function of T
               [250.0, 1521.5],
               [400.0, 1678.0]])
TC1225 = tm.piecewise_linear_model(kz, rho, cp)  # T-dependent properties

# Constitutive model for PEEK
kz, rho, cp = 0.27, 1310.0, 1100.0
PEEK = tm.isothermal_model(kz, rho, cp)  # Constant properties


#
# Boundary conditions and solver details
# -----------------------------------------------------------------

# Initial and boundary conditions
T_0 = 20.0  # initial temperature
T_inf = T_0  # Far-field temperature outside
T_in = 100.0  # Temperature inside

# Some solver details, see docs on Github for more information
theta = 0.5


#
# Domains
# -----------------------------------------------------------------

# Domain and mesh for composite
t = 2E-3  # thickness in m
nn = 25  # number of nodes
z_comp = np.linspace(0, t, nn)  # node locations
                                # the bottom surface is located at z = 0.0

# Domain and mesh for PEEK
t = 2E-3  # thickness in m
nn = 25  # number of nodes
z_peek = np.linspace(-t, 0, nn)  # node locations
                                 # the top surface is now located at z = 0.0


#
# Bonded assembly
# -----------------------------------------------------------------

# We need to combine the two domains. Let's start with the node
# locations; we ignore the first node of the second domain as that is
# located on z = 0.0, which is already equal to the last node of the
# first domain.
z = np.hstack((z_peek, z_comp[1:]))
mesh = tm.Mesh(z, tm.LinearElement)  # or tm.QuadraticElement

# We now have a mesh which consists of two materials, which means that
# we need to assign which elements belong to which material. All
# elements above z = 0.0 belong to the composite, which we will give
# index 1, while subdomain number for the PEEK remains 0.
for i, e in enumerate(mesh.elem):
    if e.nodes.mean() > 0.0:
        mesh.subdomain[i] = 1

# The boundary conditions are the same as for TSC.
bc = [{"T": T_in},  # constant T on PEEK side
      {"T": T_0}]   # constant T on C/PAEK side

# Now we need to supply two material models, with the PEEK at index 0
# and the C/PAEK at index 1
bonded = tm.Domain(mesh, [PEEK, TC1225], bc)

# Next we assign the temperature distribution. For now we will average
# the temperature at the contact (more exact estimations take into
# account the density, specific heat and element size...)
bonded.set_T(T_0*np.ones(len(mesh.nodes)))

# Solve
t_end = 25
n_steps = 50
solver = {"dt": t_end/n_steps, "t_end": t_end, "theta": theta}
t_bonded, T_bonded = tm.solve_ht(bonded, solver)

# Plot
# Set switch to True in case you'd like to plot data
if True:
    plt.plot(T_bonded[int(n_steps/20), :], z*1e3,
             label=f"t = {t_end/20} s.")
    plt.plot(T_bonded[int(n_steps/10), :], z*1e3,
             label=f"t = {t_end/10} s.")
    plt.plot(T_bonded[int(n_steps/4), :], z*1e3,
             label=f"t = {t_end/4} s.")
    plt.plot(T_bonded[int(n_steps/2), :], z*1e3,
             label=f"t = {t_end/2} s.")
    plt.plot(T_bonded[-1, :], z*1e3,
             label=f"t = {t_end} s.")
    plt.ylabel("z, mm")
    plt.xlabel("temperature, C")
    plt.title("temperature distribution")
    plt.legend()
    plt.show()
