import numpy as np
import math
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from Atoms import Atoms
from Conversions import *

class Metrics:
    '''Class designed to calculate coefficient of determination (COD), root mean squared error (RMSE), 
        and mean absolute deviation (MAD) values'''

    def __init__(self, ytrue, ypred):
        self.ytrue = ytrue
        self.ypred = ypred
    def COD(self):
        return r2_score(self.ytrue, self.ypred)
    def RMSE(self):
        return math.sqrt(mean_squared_error(self.ytrue, self.ypred))
    def MAD(self):
        return sum([abs(yt - yp) for yp, yt in zip(self.ytrue, self.ypred)]) / len(self.ytrue)


class Interpolate():
    '''Class designed to interpolate the energy and dipole moment curve

        Functions:
            __standardPolynomialE - Use a standard polynomial to interpolate the energy curve
            __powerPolynomialD - Use a power series expansion to interpolate the dipole curve
            __powerPolynomialE - Use a power series expansion to interpolate the energy curve
            __transformDipoleMoment - Use to transform the dipole moment curve for isotopically 
                                        laballed and charged diatomic molecules

        Variables:
            self.data - Array of energy and dipole curve values
            self.masses - List of atomic masses
            self.numPoints - List of the number of interpolation points
            self.order_e - Order of the energy curve power series
            self.order_d - Order of the dipole curve power series
            self.pec_numPoints - Number of interpolation points for the energy curve power series
            self.dip_numPoints - Number of interpolation points for the dipole curve power series
            self.reducedMass - Reduced mass of the diatomic

    '''

    def __init__(self, *args, **kwargs): 
        self.data      = kwargs['temp_data']
        self.atoms     = kwargs['atoms']
        self.isotopes  = kwargs['isotopes']
        self.masses    = kwargs['masses']
        self.charge    = kwargs['charge']
        self.numPoints = kwargs['numpoints']
        self.order_e   = kwargs['order_e']
        self.order_d   = kwargs['order_d']

        self.pec_numPoints = self.numPoints[0]
        self.dip_numPoints = self.numPoints[1]

        self.reducedMass = ((self.masses[0] * self.masses[1]) / (self.masses[0] + self.masses[1])) * amu_kg

        self.__standardPolynomialE()
        self.__powerPolynomialE()

        if self.data.shape[0] == 3:
            self.__powerPolynomialD()


    def __standardPolynomialE(self):
        '''Function that uses a standard polynomial to fit the energy curve.
            
           Used to calculate the equilibrium bond distance and minimum energy.

           Variables:
                self.R          - Array of bond distances
                self.E          - Array of energy values
                self.MaxDeg     - Maximum degree to use for polynomial fit
                err_arr         - List of error values for polynomial fits
                self.polyerr    - Lowest mean absolute error value for polynomial fits 
                self.polyEDeg   - Polynomial degree with the lowest error for the energy curve fit
                self.polyEFit   - np.polyfit object which fits the energy curve
                self.polyEPol   - np.poly1d object which fits the energy curve
                self.PEC_r      - Array of interpolated bond distances (uses self.pec_numPoints)
                self.PEC_e      - Array of interpolated energy values (uses self.pec_numPoints)
                self.rEq        - Equilbrium bond distantce
                self.eEq        - Energy at equilibrium bond distance
                self.bEq        - Equilibrium rotational constant

        '''

        self.R = self.data[0] 
        self.E = self.data[1]

        self.MaxDeg = min(12, len(self.R) - 1)
        err_arr = np.zeros((self.MaxDeg))

        for deg in range(0, self.MaxDeg):
            fit_ = np.polyfit(self.R, self.E, deg)
            pol_ = np.poly1d(fit_)
            err_ = sum(abs(self.E - pol_(self.R))) / (len(self.R))
            err_arr[deg-1] = err_

        self.polyerr = np.amin(err_arr)

        self.polyEDeg = np.where(err_arr == np.amin(err_arr))[0][0]
        self.polyEFit = np.polyfit(self.R, self.E, self.polyEDeg)
        self.polyEPol = np.poly1d(self.polyEFit)

        self.PEC_r = np.linspace(min(self.R), max(self.R), self.pec_numPoints)
        self.PEC_e = self.polyEPol(self.PEC_r)

        self.rEq = self.PEC_r[np.where(self.PEC_e == np.amin(self.PEC_e))][0] # Equilibrium Energy
        self.eEq = self.PEC_e[np.where(self.PEC_e == np.amin(self.PEC_e))][0] # Equilibrium Bond Length
        self.bEq = h / ( 8 * np.pi**2 * c_cm * self.reducedMass * (self.rEq*ang_m)**2) # Equilbrium Rotation Constant

    def __powerPolynomialD(self):
        '''Function that uses a power series to fit the diople curve.

           Used to calculate the equilibrium dipole moment.

           Variables:
                self.D              - Array of dipole moment values
                self.R              - Array of bond distance values
                self.rEq            - Equilibrium bond distance
                ang_m               - Conversion factor between angstrom and meters
                D_au                - Conversion factor between Debye and atomic units
                self.polDfit        - np.polyfit object which fits the dipole curve
                self.polyDpol       - np.poly1d object which fits the dipole curve
                self.polyDerr_arr   - Array of error values
                met                 - Metrics class object
                self.D_cod          - Coefficient of determination value for dipole interpolation
                self.D_rmse         - Root mean squared error for dipole interpolation
                self.D_mad          - Mean absolute error value for dipole interpolation
                self.PEC_rd         - Array of interpolated distance values
                self.dip_numPoints  - Number of points for interpolation of dipole curvef
                self.PED_d          - Array of dipole moment values converted to atomic units
                self.PED_d_         - Array of interpolated dipole moment values
                self.dEq            - Equilibrium dipole momemnt value

        '''

        self.D = self.data[2]

        if self.charge != 0:
            if self.isotopes[0] != '0' or self.isotopes[1] != '0':
                self.__transformDipoleMoment()


        self.polyDfit = np.polyfit((self.R-self.rEq)*ang_m, self.D*D_au, self.order_d)
        self.polyDpol = np.poly1d(self.polyDfit)

        self.polyDerr_arr = self.D*D_au - self.polyDpol((self.R-self.rEq)*ang_m)
        self.polyDerr = sum(abs(self.polyDerr_arr)) / len(self.R)

        met = Metrics(self.D*D_au, self.polyDpol((self.R-self.rEq)*ang_m))
        self.D_cod  = met.COD()
        self.D_rmse = met.RMSE()
        self.D_mad  = met.MAD()

        self.PEC_rd = np.linspace(min(self.R), max(self.R), self.dip_numPoints)
        self.PEC_d  = self.D*D_au
        self.PEC_d_ = self.polyDpol((self.PEC_r-self.rEq)*ang_m)

        self.dEq = self.polyDpol(0)


    def __transformDipoleMoment(self, *args, **kwargs):
            '''Function used to transform the dipole moment functino for an isotopically
                   labelled and charged diatomic molecule

                Variables:
                    Atom1 - Identity of atom #1
                    Atom2 - Identity of atom #2
                    Iso1  - Isotope of atom #1
                    Iso2  - Isotope of atom #2 
                    D - Array of dipole moment values
                    R - Array of bond distance values

                Returns:
                    D - Transformed Dipole moment array

            '''

            atom1 = self.atoms[0]
            atom2 = self.atoms[1]

            mass1 = atom1[0]
            mass2 = atom2[0]

            mass1_sub = atom1[int(self.isotopes[0])]
            mass2_sub = atom2[int(self.isotopes[1])]

            if mass1_sub == mass1:
                ratio_old = mass2 / (mass1 + mass2)
                ratio_new = mass2_sub / (mass1_sub + mass2_sub)

            elif mass2_sub == mass2:
                ratio_old = mass1 / (mass1 + mass2)
                ratio_new = mass1_sub / (mass1_sub + mass2_sub)

            else:
                return

            if self.charge < 0:
                self.D -= ((self.R - self.rEq) * ang_bohr) * (ratio_old - ratio_new) / D_au
            else:
                self.D += ((self.R - self.rEq) * ang_bohr) * (ratio_old - ratio_new) / D_au
            

    def __powerPolynomialE(self):
        '''Function that uses a power series to fit the energy curve.

           Used to calculate the equilibrium vibration constant.

           Functions:
                PowerN - Modified power series beginning at the quadratic term

           Variables:
                self.R              - Array of distance values
                self.rEq            - Equilibrium bond distance
                self.R_             - Array of bond distance values - self.rEq
                self.E              - Array of energy values
                self.eEq            - Energy value at the equilibrium bond distance
                self.E_             - Array of energy values - self.eEq
                self.Coef           - Power series expansion coefficients
                self.order_e        - Order of power series expansion
                self.pec_numPoints  - Number of interpolation points for energy curve
                self.error          - Array of interpolation errors
                met                 - Metrics class
                self.inter_cod      - oefficient of determination value for energy interpolation 
                self.inter_rmse     - Root mean squared error for energy interpolation
                self.inter_mad      - Mean absolute error value for energy interpolation
                hart_J              - Conversion factor between hartree and joules
                ang_m               - Conversion factor between angstrom and meters
                self.k              - Force constant (Nm)
                self.nu             - Vibrational frequency (s^-1)
                self.omege          - Vibrational constant (cm^-1)

        '''


        def Power2(x, c2):
            return c2 * x**2
        def Power4(x, c2, c3, c4):
            return c2 * x**2 + c3 * x**3 + c4 * x**4
        def Power6(x, c2, c3, c4, c5, c6):
            return c2 * x**2 + c3 * x**3 + c4 * x**4 + c5 * x**5 + \
                    c6 * x**6
        def Power8(x, c2, c3, c4, c5, c6, c7, c8):
            return c2 * x**2 + c3 * x**3 + c4 * x**4 + c5 * x**5 + \
                    c6 * x**6 + c7 * x**7 + c8 * x**8
        def Power10(x, c2, c3, c4, c5, c6, c7, c8, c9, c10):
            return c2 * x**2 + c3 * x**3 + c4 * x**4 + c5 * x**5 + \
                    c6 * x**6 + c7 * x**7 + c8 * x**8 + c9 * x**9 + \
                    c10 * x**10
        def Power12(x, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12):
            return c2 * x**2 + c3 * x**3 + c4 * x**4 + c5 * x**5 + \
                    c6 * x**6 + c7 * x**7 + c8 * x**8 + c9 * x**9 + \
                    c10 * x**10 + c11 * x**11 + c12 * x**12
        def Power14(x, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14):
            return c2 * x**2 + c3 * x**3 + c4 * x**4 + c5 * x**5 + \
                    c6 * x**6 + c7 * x**7 + c8 * x**8 + c9 * x**9 + \
                    c10 * x**10 + c11 * x**11 + c12 * x**12 + c13 * x**13 + \
                    c14 * x**14
        def Power16(x, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16):
            return c2 * x**2 + c3 * x**3 + c4 * x**4 + c5 * x**5 + \
                    c6 * x**6 + c7 * x**7 + c8 * x**8 + c9 * x**9 + \
                    c10 * x**10 + c11 * x**11 + c12 * x**12 + c13 * x**13 + \
                    c14 * x**14 + c15 * x**15 + c16 * x**16

        self.R_ = self.R - self.rEq
        self.E_ = self.E - self.eEq

        self.Coef, pcov = curve_fit(eval("Power" + str(self.order_e)), self.R_, self.E_)

        self.PEC_r_ = np.linspace(np.amin(self.R_), np.amax(self.R_), self.pec_numPoints)
        self.PEC_e_ = eval("Power" + str(self.order_e))(self.PEC_r_, *self.Coef)

        self.error = self.E_ - eval("Power" + str(self.order_e))(self.R_, *self.Coef)

        met = Metrics(self.E_, eval("Power" + str(self.order_e))(self.R_, *self.Coef))
        self.inter_mad = met.MAD()
        self.inter_cod = met.COD()
        self.inter_rmse = met.RMSE()

        for j in range(len(self.Coef)):
            self.Coef[j] *= hart_J / ang_m**(j+2)   # Convert power series expansion coefficients to SI units

        self.k      = self.Coef[0] * 2
        self.nu     = np.sqrt(self.k/self.reducedMass)
        self.omega  = self.nu /  ( 2 * np.pi * c_cm)


