#!/usr/bin/env python
#
# Grad-Shafranov solver example
# Fixed boundary (square domain) with no X-points
#

# Plasma equilibrium (Grad-Shafranov) solver
import freegs4e

# Boundary conditions
import freegs4e.boundary as boundary

profiles = freegs4e.jtor.ConstrainPaxisIp(
    1e3, 1e5, 1.0  # Plasma pressure on axis [Pascals]  # Plasma current [Amps]
)  # fvac = R*Bt

eq = freegs4e.Equilibrium(
    Rmin=0.1,
    Rmax=2.0,
    Zmin=-1.0,
    Zmax=1.0,
    nx=65,
    ny=65,
    boundary=boundary.fixedBoundary,
)

# Nonlinear solver for Grad-Shafranov equation
freegs4e.solve(
    eq, profiles  # The equilibrium to adjust
)  # The toroidal current profile function

print("Done!")

# Some diagnostics
print("Poloidal beta: {}".format(eq.poloidalBeta()))
print("Pressure on axis: {} Pa".format(eq.pressure(0.0)))

# Plot equilibrium
from freegs4e.plotting import plotEquilibrium

plotEquilibrium(eq)
