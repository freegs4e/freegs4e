"""
Constraint class

This class uses constraints on xpoints, isoflux pairs, and psi values
during an inverse solve to identify the coil currents required to 
generated the equilibrium (with respect to the constraints). 

Additional (optional) constraints on coil current limits and the 
maximum total current through all the coils can also be used. 

Modified substantially from the original FreeGS code.

Copyright 2024 Nicola C. Amorisco, Adriano Agnello, George K. Holt, Ben Dudson.

FreeGS4E is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FreeGS4E is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FreeGS4E.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from numpy import array, dot, eye, transpose
from numpy.linalg import inv
from scipy import optimize

from . import critical


class constrain(object):
    """
    This class is used to adjust coil currents, according to some constraints,
    during an inverse solve.

    To use this class, create an instance by specifying at least one of the
    following type of constraints:

    xpoints - A list of X-point locations [(R,Z), ...]

    isoflux - A list of isoflux tuple pairs [(R1,Z1,R2,Z2), ...]

    psivals - A list of psi values/locations [(R,Z,psi), ...]

    Optional constraints can be:

    current_lims - A list of tuples [(l1,u1),(l2,u2)...(lN,uN)] each
    describing a lower and upper bound on the possible current in the
    coil (order must match coil order in eq object).

    max_total_current - The maximum total current through the coilset
    (can only be used if current_lims is set).

    The class can be initialised using:

    >>> constraints = constrain(xpoints = [(1.0, 1.1), (1.0,-1.0)]),
    >>> controlsystem(eq)

    where constraints will now attempt to create x-points at
    (R,Z) = (1.0, 1.1) and (1.0, -1.1) during the inverse solve.

    Constraints on the current_lims and max_total_current can be
    attributed to the original FreeGS package.

    """

    def __init__(
        self,
        xpoints=None,
        gamma=1e-12,
        isoflux=None,
        psivals=None,
        current_lims=None,
        max_total_current=None,
    ):
        """
        Create an instance, specifying the constraints to apply.
        """

        self.xpoints = xpoints if xpoints is not None else []
        self.gamma = gamma
        self.isoflux = isoflux if isoflux is not None else []
        self.psivals = psivals if psivals is not None else []
        self.current_lims = current_lims
        self.max_total_current = max_total_current

    def __call__(self, eq):
        """
        Apply constraints to the equilbirium object (during inverse solve).
        """

        tokamak = eq.getMachine()

        constraint_matrix = []
        constraint_rhs = []
        for xpt in self.xpoints:
            # Each x-point introduces two constraints
            # 1) Br = 0

            Br = eq.Br(xpt[0], xpt[1])

            # Add currents to cancel out this field
            constraint_rhs.append(-Br)
            constraint_matrix.append(tokamak.controlBr(xpt[0], xpt[1]))

            # 2) Bz = 0

            Bz = eq.Bz(xpt[0], xpt[1])

            # Add currents to cancel out this field
            constraint_rhs.append(-Bz)
            constraint_matrix.append(tokamak.controlBz(xpt[0], xpt[1]))

        # Constrain points to have the same flux
        for r1, z1, r2, z2 in self.isoflux:
            # Get Psi at (r1,z1) and (r2,z2)
            p1 = eq.psiRZ(r1, z1)
            p2 = eq.psiRZ(r2, z2)
            constraint_rhs.append(p2 - p1)

            # Coil responses
            c1 = tokamak.controlPsi(r1, z1)
            c2 = tokamak.controlPsi(r2, z2)
            # Control for the difference between p1 and p2
            c = [c1val - c2val for c1val, c2val in zip(c1, c2)]
            constraint_matrix.append(c)

        # Constrain the value of psi
        for r, z, psi in self.psivals:
            p1 = eq.psiRZ(r, z)
            constraint_rhs.append(psi - p1)

            # Coil responses
            c = tokamak.controlPsi(r, z)
            constraint_matrix.append(c)

        if not constraint_rhs:
            raise ValueError("No constraints given")

        # Constraint matrix
        A = array(constraint_matrix)
        b = np.reshape(array(constraint_rhs), (-1,))

        # Solve by Tikhonov regularisation
        # minimise || Ax - b ||^2 + ||gamma x ||^2
        #
        # x = (A^T A + gamma^2 I)^{-1}A^T b

        # number of controls (length of x)
        ncontrols = A.shape[1]

        # calculate the change in coil current
        ATA = A.T @ A + self.gamma**2 * np.eye(ncontrols)
        ATb = A.T @ b
        self.current_change = np.linalg.solve(ATA, ATb)

        # now we can check whether the optional constraints need to be included.

        # to do this we use the solution found above as an initial guess in another
        # constrained optimsiation problem.

        # the following  code is attributed to enhancements made to the orginal
        # FreeGS code.

        if self.current_lims is not None:

            # offset the current bounds using "current" currents in coils
            current_change_bounds = []
            current_values = tokamak.controlCurrents()
            for i in range(ncontrols):
                lower_lim = self.current_lims[i][0] - current_values[i]
                upper_lim = self.current_lims[i][1] - current_values[i]
                current_change_bounds.append((lower_lim, upper_lim))

            current_change_bnds = array(current_change_bounds)

            # reform the constraint matrices to include Tikhonov regularisation
            A2 = np.concatenate([A, self.gamma * eye(ncontrols)])
            b2 = np.concatenate([b, np.zeros(ncontrols)])

            # the objective function to minimize || A2x - b2 ||^2
            def objective(x):
                return (np.linalg.norm((A2 @ x) - b2)) ** 2

            # Additional constraints on the optimisation
            cons = []

            def max_total_currents(x):
                sum0 = 0.0
                for delta, i in zip(x, tokamak.controlCurrents()):
                    sum0 += abs(delta + i)
                return -(sum0 - self.max_total_current)

            if self.max_total_current is not None:
                con1 = {"type": "ineq", "fun": max_total_currents}
                cons.append(con1)

            # Use the analytical current change as the initial guess
            if self.current_change.shape[0] > 0:
                x0 = self.current_change
                sol = optimize.minimize(
                    objective,
                    x0,
                    method="SLSQP",
                    bounds=current_change_bnds,
                    constraints=cons,
                )
                if (
                    sol.success
                ):  # check for convergence (else use prior currents)
                    self.current_change = sol.x

        # store info for user
        tokamak.controlAdjust(self.current_change)

        # ensure that the last constraint used is set in the Equilibrium
        eq._constraints = self

    def plot(self, axis=None, show=True):
        """
        Plots constraints used for coil current control

        axis     - Specify the axis on which to plot
        show     - Call matplotlib.pyplot.show() before returning

        """
        from .plotting import plotConstraints

        return plotConstraints(self, axis=axis, show=show)


class ConstrainPsi2D(object):
    """
    Adjusts coil currents to minimise the square differences
    between psi[R,Z] and a target psi.

    Ignores constant offset differences between psi array
    """

    def __init__(self, target_psi, weights=None):
        """
        target_psi : 2D (R,Z) array
            Must be the same size as the equilibrium psi

        weights : float or 2D array of same size as target_psi
            Relative importance of each (R,Z) point in the fitting
            By default every point is equally weighted
            Set points to zero to ignore them in fitting.

        """
        if weights is None:
            weights = np.full(target_psi.shape, 1.0)

        # Remove the average so constant offsets are ignored
        self.target_psi = target_psi - np.average(target_psi, weights=weights)

        self.weights = weights

    def __call__(self, eq):
        """
        Apply constraints to Equilibrium eq
        """

        tokamak = eq.getMachine()
        start_currents = tokamak.controlCurrents()

        end_currents, _ = optimize.leastsq(
            self.psi_difference, start_currents, args=(eq,)
        )

        tokamak.setControlCurrents(end_currents)

        # Ensure that the last constraint used is set in the Equilibrium
        eq._constraints = self

    def psi_difference(self, currents, eq):
        """
        Difference between psi from equilibrium with the given currents
        and the target psi
        """
        eq.getMachine().setControlCurrents(currents)
        psi = eq.psi()
        psi_av = np.average(psi, weights=self.weights)
        return (
            (psi - psi_av - self.target_psi) * self.weights
        ).ravel()  # flatten array


class ConstrainPsiNorm2D(object):
    """
    Adjusts coil currents to minimise the square differences
    between normalised psi[R,Z] and a target normalised psi.
    """

    def __init__(self, target_psinorm, weights=1.0):
        """
        target_psinorm : 2D (R,Z) array
            Must be the same size as the equilibrium psi

        weights : float or 2D array of same size as target_psinorm
            Relative importance of each (R,Z) point in the fitting
            By default every point is equally weighted
            Set points to zero to ignore them in fitting.

        """
        self.target_psinorm = target_psinorm
        self.weights = weights

    def __call__(self, eq):
        """
        Apply constraints to Equilibrium eq
        """

        tokamak = eq.getMachine()
        start_currents = tokamak.controlCurrents()

        end_currents, _ = optimize.leastsq(
            self.psinorm_difference, start_currents, args=(eq,)
        )

        tokamak.setControlCurrents(end_currents)

        # Ensure that the last constraint used is set in the Equilibrium
        eq._constraints = self

    def psinorm_difference(self, currents, eq):
        """
        Difference between normalised psi from equilibrium with the given currents
        and the target psinorm
        """
        eq.getMachine().setControlCurrents(currents)
        psi = eq.psi()

        opt, xpt = critical.find_critical(eq.R, eq.Z, psi)
        if not opt:
            print("No O-points found!")
            print(opt, xpt)
            eq.plot()
            raise ValueError("No O-points found!")
        psi_axis = opt[0][2]

        if not xpt:
            print("No X-points found!")
            eq.plot()
            raise ValueError("No X-points found")
        psi_bndry = xpt[0][2]

        # Calculate normalised psi.
        # 0 = magnetic axis
        # 1 = plasma boundary
        psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)

        return (
            (psi_norm - self.target_psinorm) * self.weights
        ).ravel()  # flatten array
