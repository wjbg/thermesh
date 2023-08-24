# Thermesh
#
# Validation
# Constant heat flux at boundary of semi-infinite solid

import thermesh as tm
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# Domain information
L = 0.01  # Make sure the domain is large enough to be a semi-infinite solid
k, rho, cp = 0.72, 1560.0, 1450.0
T_0 = 0.0
q = 1E5  # Heat flux applied to boundary

#
# Analytical solution
# -------------------

def analytical_solution(x, t, rho, cp, k, q):
    """Returns analytical solution

    Arguments
    ---------
    x : nd.array(dtype=float, dim=1)
        Spatial coordinates.
    t : float
        Time.
    rho : float
        Density.
    cp : float
        Specific heat.
    k : float
        Thermal conductivity
    q : float
        Heat flux.

    Returns
    -------
    dT : nd.array(dtype=float, dim=1, len=len(x))
        Temperature change at x.

    """
    alpha = k/rho/cp
    dT = (q/k) * (np.sqrt(4*alpha*t/np.pi)*np.exp(-(x**2)/(4*alpha*t)) -
                  z*erfc(x/(np.sqrt(4*alpha*t))))
    return dT


# Plot analytical solution at the following times
t_inc = np.array([2, 10, 25])

# Domain coordinates and thermal diffusivity
z = np.linspace(0, L, 50)

with plt.style.context("report_style.mplstyle"):
    fig = plt.figure(figsize=(2.4, 2.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Analytical solution")
    ax.set_xlabel("z/L")
    ax.set_ylabel(r"$\Delta$T ($^{\circ}$C)")
    ax.set_xlim([0, 1])
    for t in t_inc:
        plt.plot(z/L, analytical_solution(z, t, rho, cp, k, q),
                 "k", linewidth=0.5)
    plt.text(0.05, 10, "2")
    plt.text(0.15, 45, "10")
    plt.text(0.45, 90, "25")
    plt.tight_layout()
    # plt.savefig("../fig/heatflux_analytical_sol.png", dpi=600)
    plt.show(block=False)

#
# Numerical solution
# ------------------

# Let's make a mesh using linear elements (LinearElement). The
# alternative is to use second-order elements (QuadraticElement).
nn = 11  # number of nodes
z = np.linspace(0, L, nn)
mesh = tm.Mesh(z, tm.LinearElement)  # Or `QuadraticElement` to
                                     # use quadratic shape functions

# The boundary conditions are provided in a two-item list of
# dictionaries. The first dictionary (or zeroth item in the list)
# applies to the start or left side of the domain, while the second
# item applies to the end or right side of the domain. The
# dictionaries can have the following keys:
#
#   "T" OR ( ("h" and "T_inf") AND/OR "q" ),
#
# with "T" an applied temperature, "h" and "T_inf" the convective heat
# transfer coefficient and far field temperature, respectively, while
# "q" represents a direct flux on the surface.
bc = [{"q": q},  # q on the left
      {"T": T_0}]  # T on the right

# Material model (CPEEK is a function that takes T as input)
cpeek = tm.isothermal_model(k, rho, cp)

# Define and solve problem
domain = tm.Domain(mesh, cpeek, bc)
domain.set_T(T_0*np.ones(nn))

# Solver details
theta = 0.5
dt = 0.5
solver = {"dt": dt, "t_end": 25.0, "theta": theta}
t, T = tm.solve_ht(domain, solver)

# Plot numerical solution
with plt.style.context("report_style.mplstyle"):
    fig = plt.figure(figsize=(2.4, 2.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("FE solution (dz/L=" + str((z[1]-z[0])/L) +
                 ", dt=" + str(dt) +
                 r" s., $\Theta$=" + str(theta) + ")")
    ax.set_xlabel("z/L")
    ax.set_ylabel(r"$\Delta$T ($^{\circ}$C)")
    ax.set_xlim([0, 1])
    t_inc = np.array([2, 10, 25])/dt
    for i in t_inc.astype(int):
        plt.plot(z/L, T[i, :], "k", linewidth=0.5)
    plt.text(0.05, 10, "2")
    plt.text(0.15, 45, "10")
    plt.text(0.45, 90, "25")
    plt.tight_layout()
    # plt.savefig("../fig/heatflux_FE_t0.5_dt0.1s.png", dpi=600)
    plt.show(block=False)
