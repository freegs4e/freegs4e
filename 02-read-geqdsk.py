from freegs4e import geqdsk, machine
from freegs4e.plotting import plotEquilibrium

# tokamak = machine.MAST_sym()
tokamak = machine.TestTokamak()

# with open("g014220.00200") as f:
with open("lsn.geqdsk") as f:
    eq = geqdsk.read(f, tokamak, show=True)

plotEquilibrium(eq)
