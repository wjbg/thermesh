# Thermesh
#
# Validation
# Step change at one boundary

import thermesh as tm
import numpy as np
import matplotlib.pyplot as plt

# Domain information
L = 0.01
k, rho, cp = 0.72, 1560.0, 1450.0
T_0, T_bc = 0.0, [0.0, 1.0]  # initial and bc


#
# Analytical solution
# -------------------

def analytical_solution(x, t, alpha, T_end, T_0=0.0, N=20):
    """Returns analytical solution for a step change at one end.

    Arguments
    ---------
    x : nd.array(dtype=float, dim=1)
        Spatial coordinates.
    t : float
        Time.
    alpha : float
        Thermal diffusivity.
    T_end : float
        Instantaneous temperature at end.
    T_0 : float (defaults to 0.0)
        Initial temperature of domain
    N : int (defaults to 20)
        Number of summation terms to account for in solution.

    Returns
    -------
    T : nd.array(dtype=float, dim=1, len=len(x))
        Temperature at x.

    """
    summ = T_0*np.ones(len(x))
    L = x[-1]
    for n in range(1, N):
        summ += (T_end*np.cos(np.pi*n)/n * np.sin(n*np.pi*x/L) *
                 np.exp(-alpha*n**2*np.pi**2*t/L**2))
    return T_end*x/L + 2/np.pi * summ


# Plot analytical solution at the following times
t_inc = np.array([1, 5, 20, 100])

# Domain coordinates and thermal diffusivity
z = np.linspace(0, L, 50)
alpha = k/(rho*cp)

with plt.style.context("report_style.mplstyle"):
    fig = plt.figure(figsize=(2.4, 2.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Analytical solution (N=20)")
    ax.set_xlabel("z/L")
    ax.set_ylabel(r"T/T$_{\rm end}$")
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    for t in t_inc:
        plt.plot(z/L, analytical_solution(z, t, alpha, T_bc[-1], T_0),
                 "k", linewidth=0.5)
    plt.text(0.835, 0.03, "1")
    plt.text(0.7, 0.095, "5")
    plt.text(0.52, 0.2, "20")
    plt.text(0.30, 0.4, "100")
    plt.tight_layout()
    # plt.savefig("../img/step_analytical_sol.png", dpi=600)
    # plt.show(block=False)


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
bc = [{"T": T_bc[0]},  # T on the left
      {"T": T_bc[-1]}]  # T on the right

# Material model (CPEEK is a function that takes T as input)
cpeek = tm.isothermal_model(k, rho, cp)

# Define and solve problem
domain = tm.Domain(mesh, [cpeek], bc)
domain.set_T(T_0*np.ones(nn))

# Solver details
theta = 0.5
dt = 0.1
solver = {"dt": dt, "t_end": 100.0, "theta": theta}
t, T = tm.solve_ht(domain, solver)

# Plot numerical solution
with plt.style.context("report_style.mplstyle"):
    fig = plt.figure(figsize=(2.4, 2.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("FE solution (dz/L=" + str((z[1]-z[0])/L) +
                 ", dt=" + str(dt) +
                 r" s., $\Theta$=" + str(theta) + ")")
    ax.set_xlabel("z/L")
    ax.set_ylabel(r"T/T$_{\rm end}$")
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    t_inc = np.array([1, 5, 20, 100])/dt
    for i in t_inc.astype(int):
        plt.plot(z/L, T[i, :], "k", linewidth=0.5)
    plt.text(0.835, 0.05, "1")
    plt.text(0.69, 0.095, "5")
    plt.text(0.52, 0.2, "20")
    plt.text(0.30, 0.4, "100")
    plt.tight_layout()
    # plt.savefig("../img/step_FE_t0.5_dt0.1s.png", dpi=600)
    # plt.show(block=False)
