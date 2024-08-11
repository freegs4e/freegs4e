FreeGS4E: Free boundary Grad-Shafranov solver for time evolution
===========================================

FreeGS4E is a package based on [FreeGS](https://github.com/freegs-plasma/freegs), which calculates plasma equilibria for tokamak fusion experiments by solving the free boundary Grad-Shafranov equation. FreeGS4E is forked from FreeGS 0.6.1 and includes some performance optimisations that may also limit the use cases.

The primary use case for FreeGS4E is to provide a fast equilibrium solver for the FreeGSNKE code. This has resulted in some changes to the FreeGS codebase that mean FreeGS4E is no longer a drop-in replacement for FreeGS. FreeGS4E is also not intended to be a standalone equilibrium solver, and some features have been removed to improve performance. Users looking for a static equilibrium solver should use FreeGS.

Installing
----------

1. Create a new virtual environment.
2. Download this repository and install locally using `pip install -e .`

Documentation
-------------

The FreeGS manual is in the `docs` subdirectory.

Examples
--------

The Jupyter notebooks contain examples with additional notes

* MAST-example.ipynb 

There are also some Python scripts to run short tests
and examples

    $ python 01-freeboundary.py

This solves a free boundary problem, specifying the desired location of two X-points.
Writes the equilibrium to a G-EQDSK file "lsn.geqdsk"

    $ python 02-read-geqdsk.py

Reads in the file "lsn.geqdsk", inferring the coil currents from the plasma boundary
and profiles in the G-EQDSK file.

    $ python 03-mast.py

Calculates a double-null (CDND) equilibrium for MAST from scratch. Writes solution to
G-EQDSK file "mast.geqdsk"

    $ python 04-read-mast-geqdsk.py

Reads the file "mast.geqdsk", inferring the coil currents.

    $ python 05-fixed-boundary.py 

This example solves a fixed boundary problem, in which the square edges of the domain
are fixed. The plasma pressure on axis and plasma current are fixed.

    $ python 06-xpoints.py

This demonstrates the coil current control code, finding X-points, and marking core region
These routines are used inside the free boundary solver

License
-------

    Copyright 2024 Nicola C. Amorisco, George K. Holt, Adriano Agnello, and other contributors.

    FreeGS4E is licensed under the GNU Lesser General Public License version 3. The license text is included in the file LICENSE.

    The license text for FreeGS is reproduced below:

    Copyright 2016-2021 Ben Dudson, University of York, and other contributors.
    Email: benjamin.dudson@york.ac.uk

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

References
----------

* YoungMu Jeon, [Development of a free boundary Tokamak Equlibrium Solver](http://link.springer.com/article/10.3938/jkps.67.843)  [arXiv:1503.03135](https://arxiv.org/abs/1503.03135)
* S.Jardin "Computational Methods in Plasma Physics" CRC Press


Versions
--------

0.1.0 