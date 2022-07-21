import numpy as np
import math
import os
from scipy.integrate import quad 
from scipy.special import hermite
from scipy.optimize import curve_fit
from math import factorial as fac

from Conversions import *

import HamilF

class Hamil():
    '''Class used to contain all functions used to construct the Hamiltonian Matrices

        Functions:
            __Norm                  - Calculate normalization constants
            __Hermite               - Calculate Hermite Polynomials
            __Harmonic              - Populate the Harmonic Hamiltonian matrix
            __AnHarmonic            - Populate the Anharmonic Hamiltonian matrix
            __Centrifugal           - Populate the Centrifugal potential Hamiltonian matrix
            __DipoleMomentMatrix    - Populate the transition dipole moment matrix

        Returns the desired matrix

    '''

    def __init__(self, *args, **kwargs):

        if kwargs['ID'] == 'harm':
            if kwargs['method'] == 'python':
                self.harmonic = self.__Harmonic(*args, **kwargs)

            elif kwargs['method'] == 'fortran':
                HamilF.harmonic(kwargs['maxV']-1, kwargs['nu'])
                self.harmonic = np.loadtxt("harmonic.tmp").reshape(kwargs['maxV'], kwargs['maxV'])
                os.system("rm harmonic.tmp")

        elif kwargs['ID'] == 'anharm':
            if kwargs['method'] == 'python':
                self.anharmonic = self.__AnHarmonic(*args, **kwargs)
            
            elif kwargs['method'] == 'fortran':
                HamilF.anharmonic(kwargs['maxV']-1, kwargs['coef'], kwargs['beta'])
                self.anharmonic = np.loadtxt("anharmonic.tmp").reshape(kwargs['maxV'], kwargs['maxV'])
                os.system("rm anharmonic.tmp")

            for j in range(kwargs['maxV']):
                for jj in range(kwargs['maxV']):
                    print (self.anharmonic[j,jj]*J_cm, end=' ')
                print ()
            exit()

        elif kwargs['ID'] == 'cent':
            self.centrifugal = self.__Centrifugal(*args, **kwargs)
        
        elif kwargs['ID'] == 'tdm':
            self.tdm = self.__DipoleMomentMatrix(*args, **kwargs)

    def __Norm(self, v_):
        ''' Function to calculate the normalization constant for a harmonic oscillator wave fucntion of order v

            Variables:
                v_          - Vibrational quantum number
                self.beta   - Beta value for the diatomic molecule

            Returns:
                Normalization value

        '''

        return (1 / np.pi**0.25) * np.sqrt(self.beta / (2**v_ * math.factorial(v_)))

    def __Hermite(self, x):
        ''' Function used to calculate a Hermite polynomial of arbitrary order. 

            Variables:
                x     - Order of the Hermite polynomial
                Hn_1  - Array of the next lowest Hermite polynomial
                Hn    - Array of the current Hermite polynomial
                H_np1 - Array of the next highest Hermite polynomial
            
            Returns:
                An array where the index and value correspond to the exponent and the 
                coefficient, respectively. 

            Examples:
                H0(x) =  1                      --> [1]
                H1(x) =  0 + 2x                 --> [ 0, 2]
                H2(x) = -2 +  0x + 4x^2         --> [-2, 0, 4]
                H3(x) =  0 - 12x + 0x^2 + 8x^3  --> [ 0,-12, 0, 8]

        '''

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
        '''Function used to populate the Harmonic portion of a Hamiltonian matrix

            Variables:
                self.maxV - Hamiltonian matrix dimension
                self.nu   - Harmonic frequency (s^-1)
                h_bar     - Reduced Planck constant

            Returns:
                H - The harmonic Hamiltonain Matrix
        '''
                
        self.maxV = kwargs['maxV']
        self.nu = kwargs['nu']

        H = np.zeros((self.maxV, self.maxV))                  
        for n in range(self.maxV):                           
            H[n,n] = (n * 2 + 1) * h_bar * self.nu * 0.5   
        return H                                           


    def __AnHarmonic(self, *args, **kwargs):
        '''Function used to populate the Anharmonic Hamiltonian matrix.
            These integrals are calculated analytically. 

            Variables:
                self.maxV   - Hamiltonian matrix dimension
                self.Ecoef  - Array of power series expansion coefficients
                self.beta   - Beta value for the diatomic molecule
                H           - Hamiltonian Matrix
                H_temp      - temporary Hamiltonian Matrix
                EXP         - Order of power series expansion coefficient
                C           - Power series expansion coefficient
                N_i         - __Normalization consant for bra
                N_j         - __Normalization consant for ket
                H_i         - __Hermite polynomial array for bra
                H_j         - __Hermite polynomial array for ket
                Hi          - __Hermite polynomial coefficient for bra
                Hj          - __Hermite polynomial coefficient for ket
                inte        - running value of integral
                TotExp      - Total exponent given __Hermite polynomials of bra, ket, and power series expansion
                Hval        - Total value of bra and ket __Hermite polynomial coefficients
                Bval        - Value to account for powers of beta^n
                Fval        - Value of factorials given analtical solution to integral

            Returns:
                H - The anharmonic Hamiltonian matrix
        '''
        
        self.maxV  = kwargs['maxV']
        self.Ecoef = kwargs['coef']
        self.beta  = kwargs['beta']

        H = np.zeros((self.maxV, self.maxV))              

        for n in range(1, len(self.Ecoef)):         
            H_temp = np.zeros((self.maxV, self.maxV))
            EXP = n + 2                         
            C = self.Ecoef[n] * ang_m**EXP          

            for v in range(self.maxV):               
                N_i = self.__Norm(v)                  
                H_i = self.__Hermite(v)     

                for vv in range(v, self.maxV):       
                    N_j = self.__Norm(vv)            
                    H_j = self.__Hermite(vv)         

                    inte = 0      

                    for i in range(H_i.size):       
                        if H_i.size == 1:           
                            Hi = H_i                
                        else:           
                            Hi = H_i[i]             
                        if Hi != 0:                     
                            for j in range(H_j.size):   
                                if H_j.size == 1:           
                                    Hj = H_j
                                else:
                                    Hj = H_j[j]
                                if Hj != 0:
                                    TotExp = i + j + EXP

                                    if (TotExp % 2) == 0:                                           # Integral goes to 0 for odd functions
                                        Hval = float(Hi * Hj)                                       # Total Hermite portion
                                        Bval = (1. / self.beta**(EXP + 1))                          # Total beta portion
                                        Fval = fac(TotExp) / (2**TotExp * fac(int(TotExp/2)))       # Total factorial portion
                                        inte += Hval * Bval * Fval * np.sqrt(np.pi)                 
                    
                    H_temp[v, vv] = inte * N_i * N_j * C        
                    H_temp[vv, v] = inte * N_i * N_j * C        # Hamiltonian array is symmetric so match across the diagonal

            H += H_temp

        return H


    def __Centrifugal(self, *args, **kwargs):
        '''Function used to populate the centrifugal potential Hamiltonian Matrix.
            These integrals are solved numerically using the trapezoid rule

            Variables:
                Trap                - Number of intervals used for trapezoid rule
                self.maxJ           - Maximum rotational quantum number
                slef.maxV           - Maximum vibrational quantum number
                self.rEq            - Equilibrium bond distance
                self.beta           - Beta value for the diatomic molecule
                self.reduced_mass   - Reduced mass for the diatomic molecule
                old_mat             - Old centrifugal matrix to be read in
                J_Factor            - J(J+1) factor 
                H                   - Hamiltonian Matrix
                H_temp              - temporary Hamiltonian Matrix
                N_i                 - Normalization consant for bra
                N_j                 - Normalization consant for ket
                H_i                 - Hermite polynomial array for bra
                H_j                 - Hermite polynomial array for ket
                Hi                  - Hermite polynomial coefficient for bra
                Hj                  - Hermite polynomial coefficient for ket
                L                   - Left limit of integration
                R                   - Right limit of integraion
                x                   - Arrary of distance values used for integration
                hh                  - # Initial value for polynomial portion of the integral
                y1                  - Array for exponential values for integration
                y2                  - Array for 1/R^2 values for integration
                y3                  - Array for the polynomial portion of the integral
                y                   - Array for entire wave function
                inte                - running value of integral

            Returns:
                H - Centrifugal potential Hamiltonian tensor
        
        '''

        Trap    = kwargs['Trap']

        self.maxJ = kwargs['maxJ']
        self.maxV = kwargs['maxV']
        self.rEq  = kwargs['rEq']
        self.beta = kwargs['beta']
        self.reduced_mass = kwargs['reduced_mass']

        try:
            old_mat = kwargs['cent'][1] / 2.                            # If matrix has already been constructed, read in

        except:
            old_mat = np.zeros((self.maxJ, self.maxV, self.maxV))


        if np.sum(old_mat) != 0 and old_mat.shape[1] == self.maxV:      # Construct the tensor using the already computed matrix
            H = np.zeros((self.maxJ+1, self.maxV, self.maxV))
            
            for J in range(0, self.maxJ+1):
                J_Factor = J * (J+1)
                H[J] = old_mat * J_Factor
            
            return H

        else:
            H_temp = np.zeros((self.maxV, self.maxV))

            for v in range(self.maxV):               
                N_i  = self.__Norm(v)                
                H_i = self.__Hermite(v)             

                for vv in range(v, self.maxV):       
                    N_j  = self.__Norm(vv)           
                    H_j = self.__Hermite(vv)          

                    L = -0.9*self.rEq           
                    R = -L                      

                    x = np.linspace(L, R, Trap) 

                    hh = 0.                     

                    for i in range(H_i.size):  
                        if H_i.size == 1:
                            Hi = H_i
                        else:
                            Hi = H_i[i]
                        if Hi != 0:
                            for j in range(H_j.size):  
                                if H_j.size == 1:
                                    Hj = H_j
                                else:
                                    Hj = H_j[j]
                                if Hj != 0:
                                    hh += Hi * Hj * (self.beta*x)**(i + j)

                    y1 = np.exp(-(self.beta*x)**2)
                    y2 = (x + self.rEq)**2                  
                    y3 = hh                             

                    y = (y1 / y2) * y3  

                    inte = 0.
                    for j in range(1, Trap):       # Use trapezoid rule to numerically evaluate the integral. Grid controlled by args.Trap
                        inte += (x[j] - x[j-1]) * 0.5 * (y[j-1] + y[j])

                    H_temp[v,vv] = inte*N_i*N_j
                    H_temp[vv,v] = inte*N_i*N_j


            H = np.zeros((self.maxJ+1, self.maxV, self.maxV))
            
            for J in range(0, self.maxJ+1):                # Introduce the J(J+1) prefactor for all J values to create Hamiltonian tensor
                J_Factor = (((0.5 * h_bar**2 * J * (J+1)) / (ang_m**2 * self.reduced_mass)))
                H[J] = H_temp * J_Factor
            
            return H

    def __DipoleMomentMatrix(self, *args, **kwargs):
        '''Function used to populate the transition dipole moment matrix.
            
            These integrals are calculated analytically in the same as the 
                anharmonic Hamiltonian matrix.

            Variables:
                self.maxV           - Maximum vibrational quantum number
                self.Dcoef          - Coefficients for the power series expansion
                self.beta           - Beta value for the diatomic molecule
                H                   - Transition dipole moment Hamiltonian Matrix
                H_temp              - temporary Hamiltonian Matrix
                EXP                 - Order of power series expansion coefficient
                C                   - Power series expansion coefficient
                N_i                 - Normalization consant for bra
                N_j                 - Normalization consant for ket
                H_i                 - Hermite polynomial array for bra
                H_j                 - Hermite polynomial array for ket
                Hi                  - Hermite polynomial coefficient for bra
                Hj                  - Hermite polynomial coefficient for ket
                inte                - running value of integral
                TotExp              - Total exponent given __Hermite polynomials of bra, ket, and power series expansion
                Hval                - Total value of bra and ket __Hermite polynomial coefficients
                Bval                - Value to account for powers of beta^n
                Fval                - Value of factorials given analtical solution to integral

            Returns:
                H - Transition Dipole Moment Matrix
        '''

        self.maxV  = kwargs['maxV']
        self.Dcoef = kwargs['coef']
        self.beta  = kwargs['beta']

        H = np.zeros((self.maxV, self.maxV))    

        for v in range(self.maxV):                           
            N_i = self.__Norm(v)                              
            H_i = self.__Hermite(v)      

            for vv in range(v, self.maxV):                   
                N_j = self.__Norm(vv)                        
                H_j = self.__Hermite(vv)                     

                inte = 0.        

                for n in range(len(self.Dcoef)):             
                    mu = self.Dcoef[-1 - n] * ang_m**n       

                    for i in range(H_i.size):                
                        if H_i.size == 1:           
                            Hi = H_i                
                        else:
                            Hi = H_i[i]
                        if Hi != 0:                      
                            for j in range(H_j.size):       
                                if H_j.size == 1:           
                                    Hj = H_j                
                                else:
                                    Hj = H_j[j]

                                if Hj != 0:                 
                                    TotExp = i + j + n      
                                    if (TotExp % 2) == 0:   
                                        Hval = float(Hi * Hj) 
                                        Bval = 1./ (self.beta**(n + 1))   
                                        Fval = fac(TotExp) / (2**TotExp * fac(int(TotExp/2))) 
                                        inte += mu * Hval * Bval * Fval * np.sqrt(np.pi) 

                H[v, vv] = inte * N_i * N_j             
                H[vv, v] = inte * N_i * N_j        


        return H



class Wavefunctions():
    '''Class used to contruct the vibrational wave functions on given J-surfaces

        Functions:
            __Hermite       - Calculate Hermite polynomials
            __Norm          - Calculate normalization constant
            __EvalHerm      - Evaluate the Hermite polynomial on a list of bond lengths
            __GenerateWF    - Calculate the wave functions

    '''

    def __init__(self, *args, **kwargs): 
        self.vals = kwargs['vals']
        self.vecs = kwargs['vecs']
        self.trap = kwargs['trap']
        self.beta = kwargs['beta']
        self.L    = kwargs['L']
        self.R    = kwargs['R']
        self.maxV = kwargs['maxV']

        self.wfs = self.__GenerateWF()


    def __Hermite(self, x):
        ''' Function used to calculate a Hermite polynomial of arbitrary order.

            Variables:
                x     - Order of the Hermite polynomial
                Hn_1  - Array of the next lowest Hermite polynomial
                Hn    - Array of the current Hermite polynomial
                H_np1 - Array of the next highest Hermite polynomial

            Returns:
                An array where the index and value correspond to the exponent and the
                coefficient, respectively.

            Examples:
                H0(x) =  1                      --> [1]
                H1(x) =  0 + 2x                 --> [ 0, 2]
                H2(x) = -2 +  0x + 4x^2         --> [-2, 0, 4]
                H3(x) =  0 - 12x + 0x^2 + 8x^3  --> [ 0,-12, 0, 8]

        '''

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
        ''' Function to calculate the normalization constant for a harmonic oscillator wave fucntion of order v

            Variables:
                v_          - Vibrational quantum number
                self.beta   - Beta value for the diatomic molecule

            Returns:
                Normalization value

        '''

        return (1 / np.pi**0.25) * np.sqrt(self.beta / (2**v_ * math.factorial(v_)))

    def __EvalHerm(self, x, arr):
        '''Function used to evalute the __Hermite polynomials

            Variables:
                x   - Order of __Hermite polynomial
                arr - array of function values

            Returns:
                HermInte - Function values of __Hermite polynomial

        '''

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
        '''Function used to contruct the vibrational wavefunctions.
        
            Variables:
                Inte        - Array of integral values
                I_r         - Array of distance values
                I_sq        - Square of I
                I_exp       - Exponent component of integral values
                HermInte    - Array of Hermite polynomial values
                level       - Vibrational level  (bra)
                inte        - running integral for a given level
                state       - Specific row of eigenvector (ket)
                C           - Eigenvector value for given level and state
                N           - Normalization constant for given state
                Herm        - Hermite polynomial for given state


        '''

        Inte = np.zeros((self.vals.size, self.trap))

        I_r        = np.linspace(self.L*self.beta, self.R*self.beta, self.trap)
        I_sq       = I_r**2
        I_exp      = np.exp(-I_sq / 2.)
        HermInte   = np.zeros((self.vals.size, self.trap))

        for v in range(self.vals.size):
            HermInte[v]   = self.__EvalHerm(v, I_r)

        for level in range(self.vecs.shape[0]):      
            inte = np.zeros((self.trap))       
            for state in range(self.vecs.shape[1]):  
                C        = self.vecs[level, state]     
                N        = self.__Norm(state) / np.sqrt(self.beta)     
                Herm     = HermInte[state]            

                inte   += C * Herm * N * I_exp 

            inte_sq   = inte**2

            Inte[level]   = inte_sq / np.amax(inte_sq)

        return I_r, Inte

