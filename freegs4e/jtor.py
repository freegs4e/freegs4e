"""
 Classes representing plasma profiles.

 These must have the following methods:

   Jtor(R, Z, psi, psi_bndry=None)
      -> Return a numpy array of toroidal current density [J/m^2]
   pprime(psinorm)
      -> return p' at given normalised psi
   ffprime(psinorm)
      -> return ff' at given normalised psi
   pressure(psinorm)
      -> return p at given normalised psi
   fpol(psinorm)
      -> return f at given normalised psi
   fvac()
      -> f = R*Bt in vacuum


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
from numpy import clip, pi, reshape, sqrt, zeros
from scipy.integrate import quad, romb  # Romberg integration
from scipy.special import beta as spbeta
from scipy.special import betainc as spbinc

from . import critical
from .gradshafranov import mu0


class Profile(object):
    """
    Base class from which profiles classes can inherit

    This provides two methods:
       pressure(psinorm) and fpol(psinorm)

    which assume that the following methods are available:
       pprime(psinorm), ffprime(psinorm), fvac()

    """

    def pressure(self, psinorm, out=None):
        """
        Return p as a function of normalised psi by
        integrating pprime
        """

        if not hasattr(psinorm, "shape"):
            # Assume  a single value
            val, _ = quad(self.pprime, psinorm, 1.0)
            # Convert from integral in normalised psi to integral in psi
            return val * (self.psi_axis - self.psi_bndry)

        # Assume a NumPy array

        if out is None:
            out = zeros(psinorm.shape)

        pvals = reshape(psinorm, -1)
        ovals = reshape(out, -1)

        if len(pvals) != len(ovals):
            raise ValueError("Input and output arrays of different lengths")

        for i in range(len(pvals)):
            val, _ = quad(self.pprime, pvals[i], 1.0)
            # Convert from integral in normalised psi to integral in psi
            val *= self.psi_axis - self.psi_bndry
            ovals[i] = val

        return reshape(ovals, psinorm.shape)

    def fpol(self, psinorm, out=None):
        """
        Return f as a function of normalised psi

        """

        if not hasattr(psinorm, "__len__"):
            # Assume a single value

            val, _ = quad(self.ffprime, psinorm, 1.0)
            # Convert from integral in normalised psi to integral in psi
            val *= self.psi_axis - self.psi_bndry

            # ffprime = 0.5*d/dpsi(f^2)
            # Apply boundary condition at psinorm=1 val = fvac**2
            return sqrt(2.0 * val + self.fvac() ** 2)

        # Assume it's a NumPy array, or can be converted to one
        psinorm = np.array(psinorm)

        if out is None:
            out = zeros(psinorm.shape)

        pvals = reshape(psinorm, -1)
        ovals = reshape(out, -1)

        if len(pvals) != len(ovals):
            raise ValueError("Input and output arrays of different lengths")
        for i in range(len(pvals)):
            val, _ = quad(self.ffprime, pvals[i], 1.0)
            # Convert from integral in normalised psi to integral in psi
            val *= self.psi_axis - self.psi_bndry

            # ffprime = 0.5*d/dpsi(f^2)
            # Apply boundary condition at psinorm=1 val = fvac**2
            ovals[i] = sqrt(2.0 * val + self.fvac() ** 2)

        return reshape(ovals, psinorm.shape)

    def Jtor_part1(self, R, Z, psi, psi_bndry=None, mask_outside_limiter=None):
        """
        Similar code as original Jtor method, limited to identifying critical points.

        Parameters
        ----------
        R : np.ndarray
            R coordinates of the grid points
        Z : np.ndarray
            Z coordinates of the grid points
        psi : np.ndarray
            Poloidal field flux / 2*pi at each grid points (as returned by FreeGS.Equilibrium.psi())
        psi_bndry : float, optional
            Value of the poloidal field flux at the boundary of the plasma (last closed flux surface), by default None
        mask_outside_limiter : np.ndarray
            Mask of points outside the limiter, if any, optional

        Returns
        -------
        critical points

        Raises
        ------
        ValueError
            Raises ValueError if critical points incompatible with sign of plasma current
        """

        # Analyse the equilibrium, finding O- and X-points
        opt, xpt = critical.find_critical(
            R, Z, psi, self.mask_inside_limiter, self.Ip
        )

        if psi_bndry is not None:
            diverted_core_mask = critical.inside_mask(
                R, Z, psi, opt, xpt, mask_outside_limiter, psi_bndry
            )
        elif len(xpt) > 0:
            psi_bndry = xpt[0][2]
            self.psi_axis = opt[0][2]
            # # check correct sorting between psi_axis and psi_bndry
            if (self.psi_axis - psi_bndry) * self.Ip < 0:
                raise ValueError(
                    "Incorrect critical points! Likely due to not suitable psi_plasma"
                )
            diverted_core_mask = critical.inside_mask(
                R, Z, psi, opt, xpt, mask_outside_limiter, psi_bndry
            )
        else:
            # No X-points
            psi_bndry = psi[0, 0]
            diverted_core_mask = None
        return opt, xpt, diverted_core_mask, psi_bndry


class ConstrainBetapIp(Profile):
    """
    Constrain poloidal Beta and plasma current

    This is the constraint used in
    YoungMu Jeon arXiv:1503.03135

    """

    def __init__(self, betap, Ip, fvac, alpha_m=1.0, alpha_n=2.0, Raxis=1.0):
        """
        betap - Poloidal beta
        Ip    - Plasma current [Amps]
        fvac  - Vacuum f = R*Bt

        Raxis - R used in p' and ff' components
        """

        # Check inputs
        if alpha_m < 0:
            raise ValueError("alpha_m must be positive")
        if alpha_n < 0:
            raise ValueError("alpha_n must be positive")

        # Set parameters for later use
        self.betap = betap
        self.Ip = Ip
        self._fvac = fvac
        self.alpha_m = alpha_m
        self.alpha_n = alpha_n
        self.Raxis = Raxis

        # parameter to indicate that this is coming from FreeGS4E
        self.fast = True

    # def Jtor(self, R, Z, psi, psi_bndry=None):
    #     """Calculate toroidal plasma current

    #     Jtor = L * (Beta0*R/Raxis + (1-Beta0)*Raxis/R)*jtorshape

    #     where jtorshape is a shape function
    #     L and Beta0 are parameters which are set by constraints
    #     This function has been adapted from FreeGS to be more computationally efficient by calculating the integrals analytically and to save the xpt, opt and Jtor so they don't have to be recomputed

    #     Parameters
    #     ----------
    #     R : np.ndarray
    #         R coordinates of the grid points
    #     Z : np.ndarray
    #         Z coordinates of the grid points
    #     psi : np.ndarray
    #         Poloidal field flux / 2*pi at each grid points (as returned by FreeGS.Equilibrium.psi())
    #     psi_bndry : float, optional
    #         Value of the poloidal field flux at the boundary of the plasma (last closed flux surface), by default None

    #     Returns
    #     -------
    #     Jtor : np.ndarray
    #         Toroidal current density

    #     Raises
    #     ------
    #     ValueError
    #         Raises ValueError if it cannot find an O-point
    #     """

    #     # Analyse the equilibrium, finding O- and X-points
    #     opt, xpt = critical.find_critical(R, Z, psi)
    #     if not opt:
    #         raise ValueError("No O-points found!")
    #     psi_axis = opt[0][2]

    #     if psi_bndry is not None:
    #         mask = critical.inside_mask(R, Z, psi, opt, xpt, mask_outside_limiter, psi_bndry=psi_bndry)
    #     elif xpt:
    #         psi_bndry = xpt[0][2]
    #         mask = critical.core_mask(R, Z, psi, opt, xpt)
    #     else:
    #         # No X-points
    #         psi_bndry = psi[0, 0]
    #         mask = None

    #     dR = R[1, 0] - R[0, 0]
    #     dZ = Z[0, 1] - Z[0, 0]

    #     # Calculate normalised psi.
    #     # 0 = magnetic axis
    #     # 1 = plasma boundary
    #     psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)

    #     # Current profile shape
    #     jtorshape = (1.0 - np.clip(psi_norm, 0.0, 1.0) ** self.alpha_m) ** self.alpha_n

    #     if mask is not None:
    #         # If there is a masking function (X-points, limiters)
    #         jtorshape *= mask

    #     # Now apply constraints to define constants

    #     # Need integral of jtorshape to calculate pressure
    #     # Note factor to convert from normalised psi integral
    #     # def pshape(psinorm):
    #     #     shapeintegral, _ = quad(
    #     #         lambda x: (1.0 - x ** self.alpha_m) ** self.alpha_n, psinorm, 1.0
    #     #     )
    #     #     shapeintegral *= psi_bndry - psi_axis
    #     #     return shapeintegral

    #     # Pressure is
    #     #
    #     # p(psinorm) = - (L*Beta0/Raxis) * pshape(psinorm)
    #     #

    #     shapeintegral0 = spbeta(1./self.alpha_m , 1.0+self.alpha_n)/self.alpha_m

    #     nx, ny = psi_norm.shape
    #     pfunc = zeros((nx, ny))
    #     # for i in range(1, nx - 1):
    #     #     for j in range(1, ny - 1):
    #     #         if (psi_norm[i, j] >= 0.0) and (psi_norm[i, j] < 1.0):
    #     #             pfunc[i, j] = pshape(psi_norm[i, j])

    #     pfunc=shapeintegral0*(psi_bndry - psi_axis)*(1.0-spbinc(1./self.alpha_m , 1.0+self.alpha_n, np.clip(psi_norm,0.0001,0.9999)**(1/self.alpha_m)))

    #     if mask is not None:
    #         pfunc *= mask

    #     # Integrate over plasma
    #     # betap = (8pi/mu0) * int(p)dRdZ / Ip^2
    #     #       = - (8pi/mu0) * (L*Beta0/Raxis) * intp / Ip^2

    #     intp = np.sum(pfunc)*dR*dZ  # romb(romb(pfunc)) * dR * dZ

    # LBeta0 = -self.betap * (mu0 / (8.0 * pi)) * self.Raxis * self.Ip ** 2 / intp

    # # Integrate current components
    # IR = np.sum(jtorshape * R /self.Raxis)*dR*dZ # romb(romb(jtorshape * R / self.Raxis)) * dR * dZ
    # I_R = np.sum(jtorshape * self.Raxis / R)*dR*dZ # romb(romb(jtorshape * self.Raxis / R)) * dR * dZ

    # # Toroidal plasma current Ip is
    # #
    # # Ip = L * (Beta0 * IR + (1-Beta0)*I_R)
    # #    = L*Beta0*(IR - I_R) + L*I_R
    # #

    # L = self.Ip / I_R - LBeta0 * (IR / I_R - 1)
    # Beta0 = LBeta0 / L

    # # print("Constraints: L = %e, Beta0 = %e" % (L, Beta0))

    # # Toroidal current
    # Jtor = L * (Beta0 * R / self.Raxis + (1 - Beta0) * self.Raxis / R) * jtorshape
    # self.jtor = Jtor

    # self.L = L
    # self.Beta0 = Beta0
    # self.psi_bndry = psi_bndry
    # self.psi_axis = psi_axis

    # self.xpt = xpt
    # self.opt = opt

    # return Jtor

    # def Jtor_part1(self, R, Z, psi, psi_bndry=None):
    #     """
    #     Similar code as original Jtor method, limited to identifying critical points.

    #     Parameters
    #     ----------
    #     R : np.ndarray
    #         R coordinates of the grid points
    #     Z : np.ndarray
    #         Z coordinates of the grid points
    #     psi : np.ndarray
    #         Poloidal field flux / 2*pi at each grid points (as returned by FreeGS.Equilibrium.psi())
    #     psi_bndry : float, optional
    #         Value of the poloidal field flux at the boundary of the plasma (last closed flux surface), by default None

    #     Returns
    #     -------
    #     Jtor : np.ndarray
    #         Toroidal current density

    #     Raises
    #     ------
    #     ValueError
    #         Raises ValueError if it cannot find an O-point
    #     """

    #     # Analyse the equilibrium, finding O- and X-points
    #     opt, xpt = critical.find_critical(R, Z, psi)
    #     if not opt:
    #         raise ValueError("No O-points found!")

    #     return  opt, xpt

    def Jtor_part2(self, R, Z, psi, psi_axis, psi_bndry, mask):
        """
        Same code as original Jtor method, just split into two parts to enable
        identification of limiter plasma configurations.
        In part 2 psi_axis is replaced by self.psi_axis

        Calculate toroidal plasma current

        Jtor = L * (Beta0*R/Raxis + (1-Beta0)*Raxis/R)*jtorshape

        where jtorshape is a shape function
        L and Beta0 are parameters which are set by constraints
        This function has been adapted from FreeGS to be more computationally efficient by calculating the integrals analytically and to save the xpt, opt and Jtor so they don't have to be recomputed

        Parameters
        ----------
        R : np.ndarray
            R coordinates of the grid points
        Z : np.ndarray
            Z coordinates of the grid points
        psi : np.ndarray
            Poloidal field flux / 2*pi at each grid points (as returned by FreeGS.Equilibrium.psi())
        psi_bndry : float, optional
            Value of the poloidal field flux at the boundary of the plasma (last closed flux surface), by default None

        Returns
        -------
        Jtor : np.ndarray
            Toroidal current density

        Raises
        ------
        ValueError
            Raises ValueError if it cannot find an O-point
        """

        if psi_bndry is None:
            psi_bndry = psi[0, 0]
            self.psi_bndry = psi_bndry

        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]

        # Calculate normalised psi.
        # 0 = magnetic axis
        # 1 = plasma boundary
        psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)

        # Current profile shape
        jtorshape = (
            1.0 - np.clip(psi_norm, 0.0, 1.0) ** self.alpha_m
        ) ** self.alpha_n

        if mask is not None:
            # If there is a masking function (X-points, limiters)
            jtorshape *= mask

        # Now apply constraints to define constants

        # Need integral of jtorshape to calculate pressure
        # Note factor to convert from normalised psi integral
        # def pshape(psinorm):
        #     shapeintegral, _ = quad(
        #         lambda x: (1.0 - x ** self.alpha_m) ** self.alpha_n, psinorm, 1.0
        #     )
        #     shapeintegral *= psi_bndry - psi_axis
        #     return shapeintegral

        # Pressure is
        #
        # p(psinorm) = - (L*Beta0/Raxis) * pshape(psinorm)
        #

        shapeintegral0 = (
            spbeta(1.0 / self.alpha_m, 1.0 + self.alpha_n) / self.alpha_m
        )

        nx, ny = psi_norm.shape
        pfunc = zeros((nx, ny))
        # for i in range(1, nx - 1):
        #     for j in range(1, ny - 1):
        #         if (psi_norm[i, j] >= 0.0) and (psi_norm[i, j] < 1.0):
        #             pfunc[i, j] = pshape(psi_norm[i, j])

        pfunc = (
            shapeintegral0
            * (psi_bndry - psi_axis)
            * (
                1.0
                - spbinc(
                    1.0 / self.alpha_m,
                    1.0 + self.alpha_n,
                    np.clip(psi_norm, 0.0001, 0.9999) ** (1 / self.alpha_m),
                )
            )
        )

        if mask is not None:
            pfunc *= mask

        # Integrate over plasma
        # betap = (8pi/mu0) * int(p)dRdZ / Ip^2
        #       = - (8pi/mu0) * (L*Beta0/Raxis) * intp / Ip^2

        intp = np.sum(pfunc) * dR * dZ  # romb(romb(pfunc)) * dR * dZ

        LBeta0 = (
            -self.betap * (mu0 / (8.0 * pi)) * self.Raxis * self.Ip**2 / intp
        )

        # Integrate current components
        IR = (
            np.sum(jtorshape * R / self.Raxis) * dR * dZ
        )  # romb(romb(jtorshape * R / self.Raxis)) * dR * dZ
        I_R = (
            np.sum(jtorshape * self.Raxis / R) * dR * dZ
        )  # romb(romb(jtorshape * self.Raxis / R)) * dR * dZ

        # Toroidal plasma current Ip is
        #
        # Ip = L * (Beta0 * IR + (1-Beta0)*I_R)
        #    = L*Beta0*(IR - I_R) + L*I_R
        #

        L = self.Ip / I_R - LBeta0 * (IR / I_R - 1)
        Beta0 = LBeta0 / L

        # print("Constraints: L = %e, Beta0 = %e" % (L, Beta0))

        # Toroidal current
        Jtor = (
            L
            * (Beta0 * R / self.Raxis + (1 - Beta0) * self.Raxis / R)
            * jtorshape
        )
        self.jtor = Jtor

        self.L = L
        self.Beta0 = Beta0

        return Jtor

    # Profile functions
    def pprime(self, pn):
        """
        dp/dpsi as a function of normalised psi. 0 outside core
        Calculate pprimeshape inside the core only
        """
        shape = (1.0 - np.clip(pn, 0.0, 1.0) ** self.alpha_m) ** self.alpha_n
        return self.L * self.Beta0 / self.Raxis * shape

    def ffprime(self, pn):
        """
        f * df/dpsi as a function of normalised psi. 0 outside core.
        Calculate ffprimeshape inside the core only.
        """
        shape = (1.0 - np.clip(pn, 0.0, 1.0) ** self.alpha_m) ** self.alpha_n
        return mu0 * self.L * (1 - self.Beta0) * self.Raxis * shape

    def fvac(self):
        return self._fvac


class ConstrainPaxisIp(Profile):
    """
    Constrain pressure on axis and plasma current

    """

    def __init__(self, paxis, Ip, fvac, alpha_m=1.0, alpha_n=2.0, Raxis=1.0):
        """
        paxis - Pressure at magnetic axis [Pa]
        Ip    - Plasma current [Amps]
        fvac  - Vacuum f = R*Bt

        Raxis - R used in p' and ff' components
        """

        # Check inputs
        if alpha_m < 0:
            raise ValueError("alpha_m must be positive")
        if alpha_n < 0:
            raise ValueError("alpha_n must be positive")

        # Set parameters for later use
        self.paxis = paxis
        self.Ip = Ip
        self._fvac = fvac
        self.alpha_m = alpha_m
        self.alpha_n = alpha_n
        self.Raxis = Raxis

        # parameter to indicate that this is coming from FreeGS4E
        self.fast = True

    def Jtor_part2(self, R, Z, psi, psi_axis, psi_bndry, mask):
        """
        Same code as original Jtor method, just split into two parts to enable
        identification of limiter plasma configurations.
        In part 2 psi_axis is replaced by self.psi_axis

        Calculate toroidal plasma current

        Jtor = L * (Beta0*R/Raxis + (1-Beta0)*Raxis/R)*jtorshape

        where jtorshape is a shape function
        L and Beta0 are parameters which are set by constraints
        This function has been adapted from FreeGS to be more computationally efficient by calculating the integrals analytically and to save the xpt, opt and Jtor so they don't have to be recomputed

        Parameters
        ----------
        R : np.ndarray
            R coordinates of the grid points
        Z : np.ndarray
            Z coordinates of the grid points
        psi : np.ndarray
            Poloidal field flux / 2*pi at each grid points (as returned by FreeGS.Equilibrium.psi())
        psi_bndry : float, optional
            Value of the poloidal field flux at the boundary of the plasma (last closed flux surface), by default None

        Returns
        -------
        Jtor : np.ndarray
            Toroidal current density

        Raises
        ------
        ValueError
            Raises ValueError if it cannot find an O-point
        """
        if psi_bndry is None:
            psi_bndry = psi[0, 0]
            self.psi_bndry = psi_bndry

        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]

        # Calculate normalised psi.
        # 0 = magnetic axis
        # 1 = plasma boundary
        psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)

        # Current profile shape
        jtorshape = (
            1.0 - np.clip(psi_norm, 0.0, 1.0) ** self.alpha_m
        ) ** self.alpha_n

        if mask is not None:
            # If there is a masking function (X-points, limiters)
            jtorshape *= mask

        # Now apply constraints to define constants

        # Need integral of jtorshape to calculate paxis
        # Note factor to convert from normalised psi integral
        # shapeintegral, _ = quad(
        #     lambda x: (1.0 - x ** self.alpha_m) ** self.alpha_n, 0.0, 1.0
        # )
        shapeintegral = (
            spbeta(1.0 / self.alpha_m, 1.0 + self.alpha_n) / self.alpha_m
        )
        shapeintegral *= psi_bndry - psi_axis

        # Pressure on axis is
        #
        # paxis = - (L*Beta0/Raxis) * shapeintegral
        #

        # Integrate current components
        IR = (
            np.sum(jtorshape * R / self.Raxis) * dR * dZ
        )  # romb(romb(jtorshape * R / self.Raxis)) * dR * dZ
        I_R = (
            np.sum(jtorshape * self.Raxis / R) * dR * dZ
        )  # romb(romb(jtorshape * self.Raxis / R)) * dR * dZ

        # Toroidal plasma current Ip is
        #
        # Ip = L * (Beta0 * IR + (1-Beta0)*I_R)
        #    = L*Beta0*(IR - I_R) + L*I_R
        #

        LBeta0 = -self.paxis * self.Raxis / shapeintegral

        L = self.Ip / I_R - LBeta0 * (IR / I_R - 1)
        Beta0 = LBeta0 / L

        # print("Constraints: L = %e, Beta0 = %e" % (L, Beta0))

        # Toroidal current
        Jtor = (
            L
            * (Beta0 * R / self.Raxis + (1 - Beta0) * self.Raxis / R)
            * jtorshape
        )
        self.jtor = Jtor

        self.L = L
        self.Beta0 = Beta0

        return Jtor

    # Profile functions
    def pprime(self, pn):
        """
        dp/dpsi as a function of normalised psi. 0 outside core
        Calculate pprimeshape inside the core only
        """
        shape = (1.0 - np.clip(pn, 0.0, 1.0) ** self.alpha_m) ** self.alpha_n
        return self.L * self.Beta0 / self.Raxis * shape

    def ffprime(self, pn):
        """
        f * df/dpsi as a function of normalised psi. 0 outside core.
        Calculate ffprimeshape inside the core only.
        """
        shape = (1.0 - np.clip(pn, 0.0, 1.0) ** self.alpha_m) ** self.alpha_n
        return mu0 * self.L * (1 - self.Beta0) * self.Raxis * shape

    def fvac(self):
        return self._fvac


class Fiesta_Topeol(Profile):
    """
    As in Fiesta. Implements profile function as in Jeon arxiv:1503.03135 eq. 5
    and in Lao et al 1985 eq. 13.

    """

    def __init__(self, Beta0, Ip, fvac, alpha_m=1.0, alpha_n=2.0, Raxis=1.0):
        """
        paxis - Pressure at magnetic axis [Pa]
        Ip    - Plasma current [Amps]
        fvac  - Vacuum f = R*Bt

        Raxis - R used in p' and ff' components
        """

        # Check inputs
        if alpha_m < 0:
            raise ValueError("alpha_m must be positive")
        if alpha_n < 0:
            raise ValueError("alpha_n must be positive")

        # Set parameters for later use
        self.Beta0 = Beta0
        self.Ip = Ip
        self._fvac = fvac
        self.alpha_m = alpha_m
        self.alpha_n = alpha_n
        self.Raxis = Raxis

        # parameter to indicate that this is coming from FreeGS4E
        self.fast = True

    # def Jtor_part1(self, R, Z, psi, psi_bndry=None):
    #     """
    #     Same code as original Jtor method, just split into two parts to enable
    #     identification of limiter plasma configurations.

    #     Calculate toroidal plasma current

    #     Jtor = L * (Beta0*R/Raxis + (1-Beta0)*Raxis/R)*jtorshape

    #     where jtorshape is a shape function
    #     L and Beta0 are parameters which are set by constraints

    #     This function is adapted from FreeGS to save the xpts, opts, and Jtor so they don't have to be recomputed

    #     Parameters
    #     ----------
    #     R : np.ndarray
    #         R coordinates of the grid points
    #     Z : np.ndarray
    #         Z coordinates of the grid points
    #     psi : np.ndarray
    #         Poloidal field flux / 2*pi at each grid points (as returned by FreeGS.Equilibrium.psi())
    #     psi_bndry : float, optional
    #         Value of the poloidal field flux at the boundary of the plasma (last closed flux surface), by default None

    #     Returns
    #     -------
    #     Jtor : np.ndarray
    #         Toroidal current density

    #     Raises
    #     ------
    #     ValueError
    #         Raises ValueError if it cannot find an O-point
    #     """

    #     # Analyse the equilibrium, finding O- and X-points
    #     opt, xpt = critical.find_critical(R, Z, psi)
    #     if not opt:
    #         raise ValueError("No O-points found!")
    #     # psi_axis = opt[0][2]

    #     # if psi_bndry is not None:
    #     #     mask = critical.core_mask(R, Z, psi, opt, xpt, psi_bndry)
    #     # elif xpt:
    #     #     psi_bndry = xpt[0][2]
    #     #     mask = critical.core_mask(R, Z, psi, opt, xpt)
    #     # else:
    #     #     # No X-points
    #     #     psi_bndry = psi[0, 0]
    #     #     mask = None

    #     # # check correct sorting between psi_axis and psi_bndry
    #     # if (psi_axis-psi_bndry)*self.Ip < 0:
    #     #     raise ValueError("Incorrect critical points! Likely due to not suitable psi_plasma")

    #     # # added with respect to original Jtor
    #     # self.xpt = xpt
    #     # self.opt = opt
    #     # self.psi_bndry = psi_bndry
    #     # self.psi_axis = psi_axis

    #     return opt, xpt

    def Jtor_part2(self, R, Z, psi, psi_axis, psi_bndry, mask):
        """
        Same code as original Jtor method, just split into two parts to enable
        identification of limiter plasma configurations.
        In part 2 psi_axis is replaced by self.psi_axis

        Calculate toroidal plasma current

        Jtor = L * (Beta0*R/Raxis + (1-Beta0)*Raxis/R)*jtorshape

        where jtorshape is a shape function
        L and Beta0 are parameters which are set by constraints
        This function has been adapted from FreeGS to be more computationally efficient by calculating the integrals analytically and to save the xpt, opt and Jtor so they don't have to be recomputed

        Parameters
        ----------
        R : np.ndarray
            R coordinates of the grid points
        Z : np.ndarray
            Z coordinates of the grid points
        psi : np.ndarray
            Poloidal field flux / 2*pi at each grid points (as returned by FreeGS.Equilibrium.psi())
        psi_bndry : float, optional
            Value of the poloidal field flux at the boundary of the plasma (last closed flux surface), by default None

        Returns
        -------
        Jtor : np.ndarray
            Toroidal current density

        Raises
        ------
        ValueError
            Raises ValueError if it cannot find an O-point
        """
        if psi_bndry is None:
            psi_bndry = psi[0, 0]
            self.psi_bndry = psi_bndry

        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]

        # Calculate normalised psi.
        # 0 = magnetic axis
        # 1 = plasma boundary
        psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)

        # Current profile shape
        jtorshape = (
            1.0 - np.clip(psi_norm, 0.0, 1.0) ** self.alpha_m
        ) ** self.alpha_n

        if mask is not None:
            # If there is a masking function (X-points, limiters)
            jtorshape *= mask

        # Toroidal current
        Jtor = (
            self.Beta0 * R / self.Raxis + (1 - self.Beta0) * self.Raxis / R
        ) * jtorshape
        L = self.Ip / (np.sum(Jtor) * dR * dZ)
        self.jtor = L * Jtor

        self.L = L
        # self.Beta0 = Beta0

        return self.jtor

    # Profile functions
    def pprime(self, pn):
        """
        dp/dpsi as a function of normalised psi. 0 outside core
        Calculate pprimeshape inside the core only
        """
        shape = (1.0 - np.clip(pn, 0.0, 1.0) ** self.alpha_m) ** self.alpha_n
        return self.L * self.Beta0 / self.Raxis * shape

    def ffprime(self, pn):
        """
        f * df/dpsi as a function of normalised psi. 0 outside core.
        Calculate ffprimeshape inside the core only.
        """
        shape = (1.0 - np.clip(pn, 0.0, 1.0) ** self.alpha_m) ** self.alpha_n
        return mu0 * self.L * (1 - self.Beta0) * self.Raxis * shape

    def fvac(self):
        return self._fvac


class Lao85(Profile):
    """
    Implements Lao profile as from eqs. 2,4,5 in Lao et al 1985 Nucl.Fus.25

    J = \lambda * (R/R_axis P' + Raxis/R FF'/mu0)

    with P' = sum(alpha_i x^i) - sum(alpha_i) x^(n_P+1)
    FF' = sum(beta_i x^i) - sum(beta_i) x^(n_F+1)

    """

    def __init__(
        self,
        Ip,
        fvac,
        alpha,
        beta,
        alpha_logic=True,
        beta_logic=True,
        Raxis=1,
        Ip_logic=True,
    ):
        """
        Ip    - Plasma current [Amps]
        fvac  - Vacuum f = R*Bt
        alpha - polynomial coefficients for pprime, list or array
        beta - polynomial coefficients for ffprime, list or array
        alpha_logic - boole, as in Fiesta, if True include n_{P+1} degree term
        beta_logic - boole, as in Fiesta, if True include n_{F+1} degree term
        Ip_logic - boole, if True entire profile is re-normalised to satisfy Ip identically

        Raxis - R used in p' and ff' components
        """

        # Set parameters for later use
        self.alpha = np.array(alpha)
        self.alpha_logic = alpha_logic
        if alpha_logic:
            self.alpha = np.concatenate((self.alpha, [-np.sum(self.alpha)]))
        self.alpha_exp = np.arange(0, len(self.alpha))

        self.beta = np.array(beta)
        self.beta_logic = beta_logic
        if beta_logic:
            self.beta = np.concatenate((self.beta, [-np.sum(self.beta)]))
        self.beta_exp = np.arange(0, len(self.beta))

        self.Ip = Ip
        self.Ip_logic = Ip_logic
        if self.Ip_logic is False:
            self.L = 1

        self._fvac = fvac
        self.Raxis = Raxis

        # parameter to indicate that this is coming from FreeGS4E
        self.fast = True

    def Jtor_part2(self, R, Z, psi, psi_axis, psi_bndry, mask):
        """

        Parameters
        ----------
        R : np.ndarray
            R coordinates of the grid points
        Z : np.ndarray
            Z coordinates of the grid points
        psi : np.ndarray
            Poloidal field flux / 2*pi at each grid points (as returned by FreeGS.Equilibrium.psi())
        psi_bndry : float, optional
            Value of the poloidal field flux at the boundary of the plasma (last closed flux surface), by default None

        Returns
        -------
        Jtor : np.ndarray
            Toroidal current density

        Raises
        ------
        ValueError
            Raises ValueError if it cannot find an O-point
        """
        if psi_bndry is None:
            psi_bndry = psi[0, 0]
            self.psi_bndry = psi_bndry

        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]

        # Calculate normalised psi.
        # 0 = magnetic axis
        # 1 = plasma boundary
        psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)
        psi_norm = np.clip(psi_norm, 0.0, 1.0)

        # Current profile shape
        # Pprime
        pprime_term = (
            psi_norm[np.newaxis, :, :]
            ** self.alpha_exp[:, np.newaxis, np.newaxis]
        )
        pprime_term *= self.alpha[:, np.newaxis, np.newaxis]
        pprime_term = np.sum(pprime_term, axis=0)
        pprime_term *= R / self.Raxis
        # FFprime
        ffprime_term = (
            psi_norm[np.newaxis, :, :]
            ** self.beta_exp[:, np.newaxis, np.newaxis]
        )
        ffprime_term *= self.beta[:, np.newaxis, np.newaxis]
        ffprime_term = np.sum(ffprime_term, axis=0)
        ffprime_term *= self.Raxis / R
        ffprime_term /= mu0
        # Sum together
        Jtor = pprime_term + ffprime_term

        if mask is not None:
            # If there is a masking function (X-points, limiters)
            Jtor *= mask

        if self.Ip_logic:
            jtorIp = np.sum(Jtor)
            if jtorIp == 0:
                self.problem_psi = psi
                raise ValueError(
                    "Total plasma current is zero! Cannot renormalise."
                )
            L = self.Ip / (jtorIp * dR * dZ)
            Jtor = L * Jtor
        else:
            L = 1.0

        self.jtor = Jtor.copy()
        self.L = L

        return self.jtor

    # Profile functions
    def pprime(self, pn):
        """
        dp/dpsi as a function of normalised psi. 0 outside core
        Calculate pprimeshape inside the core only
        """
        pn_ = np.clip(np.array(pn))
        shape_pn = np.shape(pn_)

        shape = pn_[np.newaxis]**self.alpha_exp.reshape(list(np.shape(self.alpha_exp))+[1]*len(shape_pn))
        shape *= self.alpha.reshape(list(np.shape(self.alpha))+[1]*len(shape_pn))
        shape = np.sum(shape, axis=0)
        return self.L * shape / self.Raxis

    def ffprime(self, pn):
        """
        f * df/dpsi as a function of normalised psi. 0 outside core.
        Calculate ffprimeshape inside the core only.
        """
        pn_ = np.clip(np.array(pn))
        shape_pn = np.shape(pn_)

        shape = (
            pn_[np.newaxis]
            **self.beta_exp.reshape(list(np.shape(self.beta_exp))+[1]*len(shape_pn))
        )
        shape *= self.beta.reshape(list(np.shape(self.beta))+[1]*len(shape_pn))
        shape = np.sum(shape, axis=0)
        return self.L * shape * self.Raxis

    def pressure(self, pn):
        """Claculates the pressure specifically for the Lao profile.
        Avoids using numerical integration.

        Parameters
        ----------
        pn : np.array of normalised psi values
        """

        pn_ = np.clip(np.array(pn))[np.newaxis]
        shape_pn = np.shape(pn_)

        ones = np.ones_like(pn)
        integrated_coeffs = self.alpha / np.arange(1, len(self.alpha_exp) + 1)
        norm_pressure = np.sum(
            integrated_coeffs.reshape(list(np.shape(integrated_coeffs))+[1]*len(shape_pn))
            * (ones - pn ** (self.alpha_exp.reshape(list(np.shape(self.alpha_exp))+[1]*len(shape_pn)) + 1)),
            axis=0,
        )
        pressure = self.L * norm_pressure * (self.psi_axis - self.psi_bndry)
        return pressure

    def fvac(self):
        return self._fvac


class ProfilesPprimeFfprime:
    """
    Specified profile functions p'(psi), ff'(psi)

    Jtor = R*p' + ff'/(R*mu0)

    """

    def __init__(
        self, pprime_func, ffprime_func, fvac, p_func=None, f_func=None
    ):
        """
        pprime_func(psi_norm) - A function which returns dp/dpsi at given normalised flux
        ffprime_func(psi_norm) - A function which returns f*df/dpsi at given normalised flux (f = R*Bt)

        fvac - Vacuum f = R*Bt

        Optionally, the pres
        """
        self.pprime = pprime_func
        self.ffprime = ffprime_func
        self.p_func = p_func
        self.f_func = f_func
        self._fvac = fvac

    def Jtor(self, R, Z, psi, psi_bndry=None):
        """
        Calculate toroidal plasma current

        Jtor = R*p' + ff'/(R*mu0)
        """

        # Analyse the equilibrium, finding O- and X-points
        opt, xpt = critical.find_critical(R, Z, psi)
        if not opt:
            raise ValueError("No O-points found!")
        psi_axis = opt[0][2]

        if psi_bndry is not None:
            mask = critical.core_mask(R, Z, psi, opt, xpt, psi_bndry)
        elif xpt:
            psi_bndry = xpt[0][2]
            mask = critical.core_mask(R, Z, psi, opt, xpt)
        else:
            # No X-points
            psi_bndry = psi[0, 0]
            mask = None

        # Calculate normalised psi.
        # 0 = magnetic axis
        # 1 = plasma boundary
        psi_norm = clip((psi - psi_axis) / (psi_bndry - psi_axis), 0.0, 1.0)
        Jtor = R * self.pprime(psi_norm) + self.ffprime(psi_norm) / (R * mu0)

        if mask is not None:
            # If there is a masking function (X-points, limiters)
            Jtor *= mask

        return Jtor

    def pressure(self, psinorm, out=None):
        """
        Return pressure [Pa] at given value(s) of
        normalised psi.
        """
        if self.p_func is not None:
            # If a function exists then use it
            return self.p_func(psinorm)

        # If not, use base class to integrate
        return super(ProfilesPprimeFfprime, self).pressure(psinorm, out)

    def fpol(self, psinorm, out=None):
        """
        Return f=R*Bt at given value(s) of
        normalised psi.
        """
        if self.f_func is not None:
            # If a function exists then use it
            return self.f_func(psinorm)

        # If not, use base class to integrate
        return super(ProfilesPprimeFfprime, self).fpol(psinorm, out)

    def fvac(self):
        return self._fvac


class TensionSpline(Profile):
    """
    Implements tension spline profiles. Typically used for more modelling
    more complex shaped profiles (from magnetics + MSE plasma reconstructions).


    J = \lambda * (R/R_axis P' + Raxis/R FF'/mu0)

    with P' = sum_n f_n
    FF' = sum_n f_n

    where f_n is the tension spline.

    See https://catxmai.github.io/pdfs/Math212_ProjectReport.pdf for definition.

    """

    def __init__(
        self,
        Ip,
        fvac,
        pp_knots,
        pp_values,
        pp_values_2,
        pp_sigma,
        ffp_knots,
        ffp_values,
        ffp_values_2,
        ffp_sigma,
        Raxis=1,
        Ip_logic=True,
    ):
        """
        Ip    - Plasma current [Amps]
        fvac  - Vacuum f = R*Bt
        pp_knots - knot points of pprime, list or array
        pp_values - values of pprime at knot points, list or array
        pp_values_2 - values of 2nd derivative of pprime at knot points, list or array
        pp_sigma - pprime tension parameter, non-neative float
        ffp_knots - knot points of ffprime, list or array
        ffp_values - values of ffprime at knot points, list or array
        ffp_values_2 - values of 2nd derivative of ffprime at knot points, list or array
        ffp_sigma - ffprime tension parameter, non-neative float
        Raxis - R used in p' and ff' components]
        Ip_logic - boole, if True entire profile is re-normalised to satisfy Ip identically
        """

        # Set parameters for later use
        self.pp_knots = np.array(pp_knots)
        self.pp_values = np.array(pp_values)
        self.pp_values_2 = np.array(pp_values_2)
        self.pp_sigma = pp_sigma
        self.ffp_knots = np.array(ffp_knots)
        self.ffp_values = np.array(ffp_values)
        self.ffp_values_2 = np.array(ffp_values_2)
        self.ffp_sigma = ffp_sigma

        self.Ip = Ip
        self.Ip_logic = Ip_logic

        self._fvac = fvac
        self.Raxis = Raxis

        # parameter to indicate that this is coming from FreeGS4E
        self.fast = True

    def Jtor_part2(self, R, Z, psi, psi_axis, psi_bndry, mask):
        """

        Parameters
        ----------
        R : np.ndarray
            R coordinates of the grid points
        Z : np.ndarray
            Z coordinates of the grid points
        psi : np.ndarray
            Poloidal field flux / 2*pi at each grid points (as returned by FreeGS.Equilibrium.psi())
        psi_bndry : float, optional
            Value of the poloidal field flux at the boundary of the plasma (last closed flux surface), by default None

        Returns
        -------
        Jtor : np.ndarray
            Toroidal current density

        Raises
        ------
        ValueError
            Raises ValueError if it cannot find an O-point
        """
        if psi_bndry is None:
            psi_bndry = psi[0, 0]
            self.psi_bndry = psi_bndry

        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]

        # Calculate normalised psi.
        # 0 = magnetic axis
        # 1 = plasma boundary
        psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)
        psi_norm = np.clip(psi_norm, 0.0, 1.0)

        # Current profile shape

        # Pprime
        pprime_term = self.tension_spline(
            psi_norm,
            self.pp_knots,
            self.pp_values,
            self.pp_values_2,
            self.pp_sigma,
        )
        pprime_term *= R / self.Raxis

        # FFprime
        ffprime_term = self.tension_spline(
            psi_norm,
            self.ffp_knots,
            self.ffp_values,
            self.ffp_values_2,
            self.ffp_sigma,
        )
        ffprime_term *= self.Raxis / R
        ffprime_term /= mu0

        # Sum together
        Jtor = pprime_term + ffprime_term

        if mask is not None:
            # If there is a masking function (X-points, limiters)
            Jtor *= mask

        if self.Ip_logic:
            jtorIp = np.sum(Jtor)
            if jtorIp == 0:
                self.problem_psi = psi
                raise ValueError(
                    "Total plasma current is zero! Cannot renormalise."
                )
            L = self.Ip / (jtorIp * dR * dZ)
            Jtor = L * Jtor
        else:
            L = 1.0

        self.jtor = Jtor.copy()
        self.L = L

        return self.jtor

    # Profile functions
    def pprime(self, pn):
        """
        dp/dpsi as a function of normalised psi. 0 outside core
        Calculate pprimeshape inside the core only
        """

        shape = self.tension_spline(
            pn, self.pp_knots, self.pp_values, self.pp_values_2, self.pp_sigma
        )
        return self.L * shape / self.Raxis

    def ffprime(self, pn):
        """
        f * df/dpsi as a function of normalised psi. 0 outside core.
        Calculate ffprimeshape inside the core only.
        """

        shape = self.tension_spline(
            pn,
            self.ffp_knots,
            self.ffp_values,
            self.ffp_values_2,
            self.ffp_sigma,
        )
        return self.L * shape * self.Raxis

    def fvac(self):
        return self._fvac

    def tension_spline(self, x, xn, yn, zn, sigma):
        """
        Evaluate the tension spline at locations in x using knot points xn,
        values at knot points yn, and second derivative values at knot points zn.
        Tension parameter is a non-negative float sigma.

        """

        size = x.shape
        if len(size) > 1:
            x = x.flatten()

        # fixed parameters
        x_diffs = xn[1:] - xn[0:-1]
        sinh_diffs = np.sinh(sigma * x_diffs)

        # initial solution array (each column is f_n(x) for a different n)
        X = np.tile(x, (len(x_diffs), 1)).T

        # calculate the terms in the spline (vectorised for speed)
        t1 = (yn[0:-1] - zn[0:-1] / (sigma**2)) * ((xn[1:] - X) / x_diffs)
        t2 = (
            zn[0:-1] * np.sinh(sigma * (xn[1:] - X))
            + zn[1:] * np.sinh(sigma * (X - xn[0:-1]))
        ) / ((sigma**2) * sinh_diffs)
        t3 = (yn[1:] - zn[1:] / (sigma**2)) * ((X - xn[0:-1]) / x_diffs)

        # sum the values
        sol = t1 + t2 + t3

        # zero out values outisde range of each f_n(x) as they're not valid (recall definition of tension spline)
        for n in range(0, len(xn) - 1):
            ind = (xn[n] <= x) & (x <= xn[n + 1])
            sol[~ind, n] = 0

        # sum to find (alomst) final solution
        f = np.sum(sol, axis=1)

        # check if any of the interpolation and knot points are the same (if so we have double counted)
        for i in np.where(np.isin(x, xn))[0]:
            if i not in [0, len(x) - 1]:
                f[i] /= 2

        # rehape final output if required
        if len(size) > 1:
            return f.reshape(size)
        else:
            return f
