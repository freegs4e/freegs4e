from freegs4e import geqdsk, machine
from freegs4e.plotting import plotEquilibrium

# Reading MAST equilibrium, up-down symmetric coils
tokamak = machine.MAST()

# with open("g014220.00200") as f:
with open("mast.geqdsk") as f:
    eq = geqdsk.read(f, tokamak, show=True)

plotEquilibrium(eq)
