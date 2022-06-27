import numpy as np
import math
import os
from scipy.integrate import quad 
from scipy.special import hermite
from scipy.optimize import curve_fit
from math import factorial as fac

from Conversions import *

class Hamil():

    def __init__(self, *args, **kwargs):

        if kwargs['ID'] == 'harm':
            self.harmonic = self.__Harmonic(*args, **kwargs)
        elif kwargs['ID'] == 'anharm':
            self.anharmonic = self.__AnHarmonic(*args, **kwargs)
        elif kwargs['ID'] == 'cent':
            self.centrifugal = self.__Centrifugal(*args, **kwargs)
        elif kwargs['ID'] == 'tdm':
            self.tdm = self.__DipoleMomentMatrix(*args, **kwargs)

    def __Norm(self, v_):

        #####################################################
        #   Function to calculate the normalization constant
        #       for a harmonic oscillator wave fucntion of 
        #       order v
        #####################################################
        
        return (1 / np.pi**0.25) * np.sqrt(self.beta / (2**v_ * math.factorial(v_)))

    def __Hermite(self, x):

        ###################################################
        # __Hermite Polynomials of Arbitrary Order 
        #   stored in an array where the index corresponds 
        #   to the exponent and the value corresponds to
        #   the coefficient
        # Examples:
        #   H0(x) =  1                      --> [1]
        #   H1(x) =  0 + 2x                 --> [ 0, 2]
        #   H2(x) = -2 +  0x + 4x^2         --> [-2, 0, 4]
        #   H3(x) =  0 - 12x + 0x^2 + 8x^3  --> [ 0,-12, 0, 8]
        ###################################################

        if x == 0:
            return np.array((1))
        elif x == 1:
            return np.array((0, 2))
        for j in range(1, x):
            if j == 1:
                Hn_1 = np.array((1))
                Hn   = np.array((0, 2))
            else:
                Hn_1 = Hn.copy()
                Hn   = H_np1.copy()
            H_np1 = np.zeros((j+2))
            H_np1[1:] = 2 * Hn
            H_np1 -= 2*j * np.append(Hn_1, np.zeros((2)))
        return H_np1

    def __Harmonic(self, *args, **kwargs):

        ##########################################################
        # Populate the harmonic Hamiltonian matrix up to size V
        #
        # Variables:
        #   V       - Hamiltonian matrix dimension
        #   h_bar   - reduced planck constant
        #   omega   - vibrational constant (s^-1)
        # Returns:
        #   H       - Harmonic Hamiltonian Matrix
        ##########################################################

        self.maxV = kwargs['maxv']
        self.nu = kwargs['nu']

        H = np.zeros((self.maxV, self.maxV))                  # Empty Hamiltonian matrix of size MaxV by MaxV
        for n in range(self.maxV):                                    # For loop over all v-values
            H[n,n] = (n * 2 + 1) * h_bar * self.nu * 0.5      # Assign value to Hamiltonian matrix 
        return H                                              # Return Matrix

    def __AnHarmonic(self, *args, **kwargs):

        #################################################################
        # Populate the Anharmonic Matrix up to size MaxV
        #
        # These integrals are calculated analytically 
        #   
        # The matrix is truncated to a size where all diagonal elements
        #   are ever increasing
        #
        # Variables:
        #   H           - Hamiltonian Matrix
        #   H_temp      - temporary Hamiltonian Matrix
        #   EXP         - Order of power series expansion coefficient
        #   C           - Power series expansion coefficient
        #   N_i         - __Normalization consant for bra 
        #   N_j         - __Normalization consant for ket
        #   H_i         - __Hermite polynomial array for bra
        #   H_j         - __Hermite polynomial array for ket
        #   Hi          - __Hermite polynomial coefficient for bra
        #   Hj          - __Hermite polynomial coefficient for ket
        #   inte        - running value of integral
        #   TotExp      - Total exponent given __Hermite polynomials of bra, ket, and power series expansion
        #   Hval        - Total value of bra and ket __Hermite polynomial coefficients
        #   Bval        - Value to account for powers of beta^n
        #   Fval        - Value of factorials given analtical solution to integral
        #
        # Returns:
        #   H - Hamiltonian Matrix
        #################################################################

        self.maxV  = kwargs['maxV']
        self.Ecoef = kwargs['coef']
        self.beta  = kwargs['beta']

        H = np.zeros((self.maxV, self.maxV))              # Empty Hamiltonian matrix of size MaxV by MaxV

        for n in range(1, len(self.Ecoef)):         # For loop over all power series expansion coefficients
            H_temp = np.zeros((self.maxV, self.maxV))     # Temporary Hamiltonian array
            EXP = n + 2                         # Order of power series expansion coefficient
            C = self.Ecoef[n] * ang_m**EXP          # Power series expansion coefficient

            for v in range(self.maxV):               # For loop over all v-values for bra
                N_i = self.__Norm(v)                   # Normalization consant for bra 
                H_i = self.__Hermite(v)                # Hermite polynomial array for bra

                for vv in range(v, self.maxV):       # For loop over all v-values for ket
                    N_j = self.__Norm(vv)              # Normalization consant for ket
                    H_j = self.__Hermite(vv)           # Hermite polynomial array for ket

                    inte = 0                    # Begin integration at a value of 0.0

                    for i in range(H_i.size):       # Loop over all __Hermite polynomial array values for bra
                        if H_i.size == 1:           # If __Hermite polynomial is equal to 1
                            Hi = H_i                # set value to array value (necessary for arrays of length 1)
                        else:           
                            Hi = H_i[i]             
                        if Hi != 0:                     # If hermite polynomial array value is not zero
                            for j in range(H_j.size):   # Loop over all __Hermite polynomial array values for ket
                                if H_j.size == 1:       # Same as above for ket    
                                    Hj = H_j
                                else:
                                    Hj = H_j[j]
                                if Hj != 0:
                                    TotExp = i + j + EXP        # Total exponent given __Hermite polynomials of bra, ket, and power series expansion

                                    if (TotExp % 2) == 0:                                           # Integral goes to 0 for odd values of TotExp
                                        Hval = float(Hi * Hj)                                       # Total value of bra and ket __Hermite polynomial coefficients
                                        Bval = (1. / self.beta**(EXP + 1))                               # Value to account for powers of beta^n
                                        Fval = fac(TotExp) / (2**TotExp * fac(int(TotExp/2)))       # Value of factorials given analtical solution to integral
                                        inte += Hval * Bval * Fval * np.sqrt(np.pi)                 # Add values to running integral value

                    H_temp[v, vv] = inte * N_i * N_j * C        # Assign integral value to entry in Hamiltonian Array
                    H_temp[vv, v] = inte * N_i * N_j * C        # Hamiltonian array is symmetric so match across the diagonal

            H += H_temp     # Add temporary Hamiltonian matrix to total anharmonic matrix

        return H

    def __Centrifugal(self, *args, **kwargs):

        #################################################################
        # Populate the centrifugal potential barrier Hamiltonian Matrix 
        #   up to size MaxV
        #
        # These integrals are calculated numerically using the trapezoid
        #   rule
        #################################################################

        Trap    = kwargs['Trap']

        self.maxJ = kwargs['maxJ']
        self.maxV = kwargs['maxV']
        self.rEq  = kwargs['rEq']
        self.beta = kwargs['beta']
        self.reduced_mass = kwargs['reduced_mass']

        try:
            old_mat = kwargs['cent'][1] / 2.
        except:
            old_mat = np.zeros((self.maxJ, self.maxV, self.maxV))


        if np.sum(old_mat) != 0 and old_mat.shape[1] == self.maxV:
            H = np.zeros((self.maxJ+1, self.maxV, self.maxV))
            for J in range(0, self.maxJ+1):
                J_Factor = J * (J+1)
                H[J] = old_mat * J_Factor
            return H

        else:
            H_temp = np.zeros((self.maxV, self.maxV))           # Emptry Hamiltonian array

            for v in range(self.maxV):                  # For loop over all v-values for bra
                N_i  = self.__Norm(v)                   # Normalization constant for bra
                H_ii = self.__Hermite(v)               # Hermite Polynomial for bra

                for vv in range(v, self.maxV):          # For loop over all v-values for ket
                    N_j  = self.__Norm(vv)              # Normalization constant for ket
                    H_jj = self.__Hermite(vv)          # Hermite Polynomial for ket

                    L = -0.9*self.rEq               # Left limit of integration 
                    R = -L                      # Right limit of integration

                    x = np.linspace(L, R, Trap)        # Array for integration values

                    hh = 0.                     # Initial value for polynomial portion of the integral

                    for i in range(H_ii.size):  # Loop over __Hermite polynomial for bra
                        if H_ii.size == 1:
                            Hi = H_ii
                        else:
                            Hi = H_ii[i]
                        if Hi != 0:
                            for j in range(H_jj.size):  # Loop over __Hermite polynomial for ket
                                if H_jj.size == 1:
                                    Hj = H_jj
                                else:
                                    Hj = H_jj[j]
                                if Hj != 0:
                                    hh += Hi * Hj * (self.beta*x)**(i + j)

                    y1 = np.exp(-(self.beta*x)**2)           # Array for exponential values for integration
                    y2 = (x + self.rEq)**2                  # Array for 1/R^2 values for integration
                    y3 = hh                             # Array for the polynomial portion of the integral

                    y = y1 / y2 * y3                    # Arrary for entire wave function

                    inte = 0.
                    for j in range(1, Trap):       # Use trapezoid rule to numerically evaluate the integral. Grid controlled by args.Trap
                        inte += (x[j] - x[j-1]) * 0.5 * (y[j-1] + y[j])

                    H_temp[v,vv] = inte*N_i*N_j         # Assign integral values to temporary Hamiltonian
                    H_temp[vv,v] = inte*N_i*N_j         # Assign integral values to temporary Hamiltonian


            H = np.zeros((self.maxJ+1, self.maxV, self.maxV))
            for J in range(0, self.maxJ+1):                # Introduce the J(J+1) prefactor for all J values to create Hamiltonian tensor
                J_Factor = (((0.5 * h_bar**2 * J * (J+1)) / (ang_m**2 * self.reduced_mass)))
                H[J] = H_temp * J_Factor
            return H

    def __DipoleMomentMatrix(self, *args, **kwargs):

        def TransformD(D, R, args):

            #TODO transform the dipole moment for charged isotopes

            #####################################################################
            # This function transforms the dipole moment array if a non-standard
            #   isotoped is provided and the diatomic molecule is charged
            #
            # Variables:
            #   D       - Array of dipole moment values
            #   R       - Array of bond distance values
            #   args    - Input arguments
            #
            # Returns
            #   D       - Transformed Dipole array
            #
            #####################################################################

            Atom1_s = Atoms(args.Atoms[0], 0)
            Atom2_s = Atoms(args.Atoms[1], 0)

            Atom1_n = Atoms(args.Atoms[0], args.Isotopes[0])
            Atom2_n = Atoms(args.Atoms[1], args.Isotopes[1])

            if Atom1_s == Atom1_n:
                Old_ratio = Atom2_s / (Atom1_s + Atom2_s)
                New_ratio = Atom2_n / (Atom1_n + Atom2_n)

                if args.Charge < 0:
                    D -= (R / bohr_m) * (Old_ratio - New_ratio)
                else:
                    D += (R / bohr_m) * (Old_ratio - New_ratio)

            elif Atom2_s == Atom2_n:
                Old_ratio = Atom1_s / (Atom1_s + Atom2_s)
                New_ratio = Atom1_n / (Atom1_n + Atom2_n)

                if args.Charge < 0:
                    D -= (R / bohr_m) * (Old_ratio - New_ratio)
                else:
                    D += (R / bohr_m) * (Old_ratio - New_ratio)

            else:
                exit()

            return D


        #################################################################
        # This function is used to construct the transition dipole moment
        #   matrix of size MaxV by MaxV
        #
        # These integrals are solved analytically
        #
        ##################################################################

        self.maxV  = kwargs['maxV']
        self.Dcoef = kwargs['coef']
        self.beta  = kwargs['beta']

        D = np.zeros((self.maxV, self.maxV))                 # Empty Hamiltonian matrix of size MaxV by MaxV    

        for v in range(self.maxV):                           # For loop over all v-values for bra
            N_i = self.__Norm(v)                             # Normalization consant for bra 
            H_i = self.__Hermite(v)                          # Hermite polynomial array for bra

            for vv in range(v, self.maxV):                   # For loop over all v-values for ket
                N_j = self.__Norm(vv)                        # Normalization consant for ket
                H_j = self.__Hermite(vv)                     # Hermite polynomial array for ket

                inte = 0.                                    # Begin integration at a value of 0.0
            
                for n in range(len(self.Dcoef)):             # For loop over all polynomial coefficients
                    mu = self.Dcoef[-1 - n] * ang_m**n       # Convert coefficient to appropriate units
                    
                    for i in range(H_i.size):                # Loop over all __Hermite polynomial array values for bra
                        if H_i.size == 1:           # If __Hermite polynomial is equal to 1
                            Hi = H_i                # set value to array value (necessary for arrays of length 1)
                        else:
                            Hi = H_i[i]
                        if Hi != 0:                          # If bra hermite polynomial array value is not zero
                            for j in range(H_j.size):        # Loop over all __Hermite polynomial array values for ket
                                if H_j.size == 1:           # If __Hermite polynomial is equal to 1
                                    Hj = H_j                # set value to array value (necessary for arrays of length 1)
                                else:
                                    Hj = H_j[j]

                                if Hj != 0:                  # If ket hermite polynomial array value is not zero
                                    TotExp = i + j + n       # Total exponent given __Hermite polynomials of bra, ket, and power series expansion
                                    if (TotExp % 2) == 0:                                       # Integral goes to 0 for odd values of TotExp
                                        Hval = float(Hi * Hj)                                   # Total value of bra and ket __Hermite polynomial coefficients
                                        Bval = 1./ (self.beta**(n + 1))                              # Value to account for powers of beta^n
                                        Fval = fac(TotExp) / (2**TotExp * fac(int(TotExp/2)))   # Value of factorials given analtical solution to integral
                                        inte += mu * Hval * Bval * Fval * np.sqrt(np.pi)        # Add values to running integral value
                
                D[v, vv] = inte * N_i * N_j             # Assign integral value to entry in Hamiltonian Array
                D[vv, v] = inte * N_i * N_j             # Hamiltonian array is symmetric so match across the diagonal
        
        return D

class Wavefunctions():

    def __init__(self, vals, vecs, trap, beta, L, R, maxV):
        self.vals = vals
        self.vecs = vecs
        self.trap = trap
        self.beta = beta
        self.L    = L
        self.R    = R
        self.maxV = maxV

        self.wfs = self.__GenerateWF()


    def __Hermite(self, x):
        if x == 0:
            return np.array((1))
        elif x == 1:
            return np.array((0, 2))
        for j in range(1, x):
            if j == 1:
                Hn_1 = np.array((1))
                Hn   = np.array((0, 2))
            else:
                Hn_1 = Hn.copy()
                Hn   = H_np1.copy()
            H_np1 = np.zeros((j+2))
            H_np1[1:] = 2 * Hn
            H_np1 -= 2*j * np.append(Hn_1, np.zeros((2)))
        return H_np1

    def __Norm(self, v_):
        return (1 / np.pi**0.25) * np.sqrt(self.beta / (2**v_ * math.factorial(v_)))

    def __EvalHerm(self, x, arr):
        ######################################################
        # Function used to evalute the __Hermite polynomials
        #
        # Variables:
        #   x   - Order of __Hermite polynomial
        #   arr - array of function values
        #
        # Returns:
        #   HermInte - Function values of __Hermite polynomial
        #
        ######################################################

        HermVal = self.__Hermite(x)
        HermInte = 0.
        exp = 0.
        if HermVal.ndim == 0:
            HermInte = np.ones((arr.size))
        else:
            for i in HermVal:
                if i != 0:
                    HermInte += i * arr**exp
                exp += 1
        return HermInte

    def __GenerateWF(self):

        Inte_full = np.zeros((self.vals.size, self.trap))

        I_full        = np.linspace(self.L*self.beta, self.R*self.beta, self.trap)
        II_full       = I_full**2
        IIe_full      = np.exp(-II_full / 2.)
        HermInte_full = np.zeros((self.vals.size, self.trap))

        for v in range(self.vals.size):
            HermInte_full[v]   = self.__EvalHerm(v, I_full)

        for level in range(self.vecs.shape[0]):      
            inte_full = np.zeros((self.trap))       
            for state in range(self.vecs.shape[1]):  
                C           = self.vecs[level, state]     
                N           = self.__Norm(state) / np.sqrt(self.beta)     
                Herm_full   = HermInte_full[state]            

                inte_full   += C * Herm_full * N * IIe_full   

            full_sq   = inte_full**2

            Inte_full[level]   = full_sq / np.amax(full_sq)

        return I_full, Inte_full

