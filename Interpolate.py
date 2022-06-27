import numpy as np
import math
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from Atoms import Atoms
from Conversions import *

class Metrics:
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

    def __init__(self, data, masses, numPoints, order_e, order_d):
        self.data = data
        self.masses = masses
        self.pec_numPoints = numPoints[0]
        self.dip_numPoints = numPoints[0]
        self.order_e = order_e
        self.order_d = order_d

        self.reducedMass = ((self.masses[0] * self.masses[1]) / (self.masses[0] + self.masses[1])) * amu_kg

        self.__standardPolynomialE()
        self.__powerPolynomialE()

        if self.data.shape[0] == 3:
            self.__powerPolynomialD()

    def __standardPolynomialE(self):
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

        self.D = self.data[2]

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

    def __powerPolynomialE(self):

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

