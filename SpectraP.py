import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from Conversions import *

class Spectra():
    '''Class designed for the prediction of spectroscopic constants

        Functions:
            Vibrational                 - Calculate vibrational constants on different J-surfaces
            Rotational                  - Calculate rotational constants on different v-surfaces
            Rovibrational               - Calculate rovibratoinal constants 
            Excitations                 - Calculate all rovibrational excitations
            Turning Points              - Calculate all classicial turning points on differnet J-surfaces
            Dunham                      - Calculate rovibrational constants using a Dunham polynomial
            SimulatedVibrationalPop     - Calculate the population of different vibrational states
            SimulatedVibrationalInt     - Calculate the intesity of different vibrational transitions
            SimulatedRotationalPop      - Calculate the population of differnt rotational states
            SimulatedRotationalInt      - Calculate the intesity of different rotational transitions
            SimulatedRovibrationalPop   - Calculate the population of differnt rovibrational states
            SimulatedRovibrationalInt   - Calculate the intesity of different rovibrational transitions


    '''

    def __init__(self):
        super(Spectra, self).__init__()

    def Vibrational(self, vals, maxV, maxJ):
        '''Function used to calculate vibrational constants on different J-surfaces

            Variables:
                vals        - A matrix of eigenvalues
                temp_vals   - A truncated matrix of eigenvalues
                maxJ        - The maximum rotational quantum number
                maxV        - The maximum vibrational quantum number
                VibSpec     - Matrix of the vibrational constants
                VibMat      - Matrix used to store the systems of linear equations

            Returns:
                Matrix of vibrational constants

        '''

        temp_vals = vals[:maxJ+1, :maxV+1]

        VibSpec = np.zeros((maxJ+1, maxV+1))
        VibMat  = np.zeros((maxV+1, maxV+1))

        for v in range(maxV+1):
            for vv in range(maxV+1):
                VibMat[v, vv] = (v + 0.5) ** (vv+1)

        VibMat_inv = np.linalg.inv(VibMat)

        for j in range(maxJ+1):
            VibSpec[j] = np.matmul(VibMat_inv, temp_vals[j,:])
        
        return VibSpec.T

    def Rotational(self, vals, maxV, maxJ):
        '''Function used to calculate rotational constants on different v-surfaces

            Variables:
                vals        - A matrix of eigenvalues
                temp_vals   - A truncated matrix of eigenvalues
                maxJ        - The maximum rotational quantum number
                maxV        - The maximum vibrational quantum number
                RotSpec     - Matrix of the rotational constants
                RotMat      - Matrix used to store the systems of linear equations

            Returns:
                Matrix of rotational constants

        '''

        temp_vals = vals[1:maxJ+2, :maxV+1] - vals[0, :maxV+1]

        RotSpec = np.zeros((maxV+1, maxJ+1))
        RotMat  = np.zeros((maxJ+1, maxJ+1))

        for j in range(1, maxJ+2):
            for jj in range(1, maxJ+2):
                try:
                    RotMat[j-1, jj-1] = (j*(j+1))**jj       # Populate RotMat according to rotational expansion terms
                except:                                     # May fail due to jj exponent
                    break

        RotMat_inv = np.linalg.inv(RotMat)                  # Invert RotMat matrix

        for v in range(maxV+1):
            RotSpec[v] = np.matmul(RotMat_inv, temp_vals[:,v])      # Solve system of linear equations
        
        return RotSpec.T

    def Rovibrational(self, vals, maxV, maxJ):
        '''Function used to calculate rovibrational constants

            Variables:
                vals        - A matrix of eigenvalues
                temp_vals   - A truncated matrix of eigenvalues
                maxJ        - The maximum rotational quantum number
                maxV        - The maximum vibrational quantum number
                num_vib     - Number of vibrational surfaces
                num_rot     - Number of rotational surfaces
                num_rov     - Number of rovibratinoal surfaces
                TotVJ       - Total number of constants to predict
                CoupMat     - Matrix used to store the systems of linear equations
                vjmat       - Array of quantum numbers for all constants
                vjmat_      - Array of quantum numbers for coupled constants
                vjmat__     - Array of quantum numbers for non-coupled constants
                CoupMat_inv - Inverse of CoupMat
                CoupSpec    - Array of rovibrational constants

            Returns:
                Matrix of rotational constants

        '''

        temp_vals = vals[:maxJ+1, :maxV]

        num_vib = maxV
        num_rot = max(maxJ, 0)
        num_rov = (maxV * (maxJ+1)) - num_vib - num_rot

        totVJ = num_vib + num_rot + num_rov
        CoupMat  = np.zeros((totVJ, totVJ))                 

        vjmat   = np.zeros((totVJ, 2), dtype=int)           
        vjmat_  = np.zeros((num_rov, 2), dtype=int)         
        vjmat__ = np.zeros((num_vib+num_rot, 2), dtype=int)

        vj   = 0
        vj_  = 0
        vj__ = 0
        for j in range(temp_vals.shape[0]):
            for v in range(temp_vals.shape[1]):
                vjmat[vj, 0] = v
                vjmat[vj, 1] = j

                if v > 0 and j > 0:
                    vjmat_[vj_, 0] = v
                    vjmat_[vj_, 1] = j
                    vj_ += 1

                else:
                    vjmat__[vj__, 0] = v
                    vjmat__[vj__, 1] = j
                    vj__ += 1

                vj += 1

        for vj in range(totVJ):
            v = vjmat[vj,0]
            j = vjmat[vj,1]
            vj_coup = 0
            for vj_ in range(totVJ):
                if vj_ < num_vib:
                    CoupMat[vj, vj_] = (v + 0.5)**(vj_+1)
                elif (vj_ >= num_vib) and (vj_ < (num_vib+num_rot)):
                    CoupMat[vj, vj_] = j**(vj_-num_vib+1) * (j+1)**(vj_-num_vib+1)
                else:
                    vib_ = (v + 0.5)**vjmat_[vj_coup,0]
                    rot_ = j**(vjmat_[vj_coup,1]) * (j+1)**(vjmat_[vj_coup,1])
                    CoupMat[vj, vj_] = vib_ * rot_
                    vj_coup += 1


        CoupMat_inv = np.linalg.inv(CoupMat)                
        CoupSpec    = np.matmul(CoupMat_inv, temp_vals.flatten()) 
    
        return CoupSpec, np.append(vjmat__, vjmat_, axis=0)


    def Excitations(self, vals, vecs, maxV, maxJ, tdm):
        '''Function used to evaluate all rovibrational excitations
            
            Variables:
                excitations_mat - Matrix to store all excitation data
                    Quantum numbers (4)
                    Energy levels (3)
                    Intensity of transtion (3)
                tdm             - Transition dipole moment matrix
                vj              - Final vibrational state
                vi              - Initial vibrational state
                jl              - Final rotational state
                jk              - Initial rotational state
                E_jl            - Final energy level
                E_ik            - Initial energy level
                E               - Transition energy
                t               - Transition dipole moment element
                vecs            - Matrix of eigenvectors
                S               - Honl-London factor
                F               - f-value for the transition
                A               - Einstein E-coefficient

            Returns:
                Matrix of excitation data


        '''

        excitations_mat = np.zeros((10)).T

        tdm = tdm / D_au

        for vj in range(0, maxV+1):
            for vi in range(0, maxV+1):
                for jl in range(0, maxJ+1):
                    for jk in range(max(0, jl-1), min(jl+2, maxJ+1)):

                        E_jl = vals[jl,vj]  
                        E_ik = vals[jk,vi]  

                        E = (E_ik - E_jl) * cm_J 

                        t = np.matmul(vecs[jl,vj,:], np.matmul(tdm, vecs[jk,vi,:])) * D_CM

                        if E < 0.:     

                            if jl == jk:    
                                S = 1
                            elif jl < jk:
                                S = jl + 1
                            else:
                                S = jl

                            f = ((8  * np.pi**2 * m_e) / (3 * h**2 * e**2)) * E * S * t**2 / (2*jl + 1)
                            A = ((64 * np.pi**4) / (3 * h**4 * c**3)) * E**3 * S * t**2 / ((2*jl + 1)  * (4*np.pi*eps_0)) 
                            #A = (16. * np.pi**3 * (E/cm_J)**3 * t**2) / (3. * h * eps_0) / 1E-6 / (2 * jl + 1)  #CDMS Equation

                            excitations_mat = np.append(excitations_mat, np.asarray([vj, jl, vi, jk, E_jl, E_ik, E/cm_J, t/D_CM, f, A]).T, axis=0)
                            

                        elif E > 0.:
                            f = 0
                            A = 0

                            excitations_mat = np.append(excitations_mat, np.asarray([vj, jl, vi, jk, E_jl, E_ik, E/cm_J, t/D_CM, f, A]).T, axis=0)

        return excitations_mat.reshape(int(excitations_mat.shape[0]/10), 10)[1:]


    def TurningPoints(self, R, E, vals, rEq):
        '''Function to calculate all turning points for a given J surface
    
            Variables:
                tps         - Matrix of turning point values (Left, Right)
                zpoint      - Minimum energy point on energy curve
                Rl          - Left limit of bond distances
                Rr          - Right limit of bond distances
                El          - Left limit of energy values
                Er          - Right limit of energy values
                vals        - List of eigenvalues
                new_arr     - Temporary array of energy values minus the given energy level value
                new_arr_s   - Sorted array of new_arr

        '''

        tps = np.zeros((2, len(vals)))

        zpoint = np.where(R < rEq)[0][-1]

        Rl = R[:zpoint].copy()
        Rr = R[zpoint+1:].copy()

        El = E[:zpoint].copy()
        Er = E[zpoint+1:].copy()

        for j in range(vals.size):
            if vals[j] < np.amax(El):
                new_arr = abs(El - vals[j])
                new_arr_s = np.argsort(new_arr)

                tps[0,j] = Rl[new_arr_s[0]]
            
            if vals[j] < np.amax(Er):
                new_arr = abs(Er - vals[j])
                new_arr_s = np.argsort(new_arr)

                tps[1, j] = Rr[new_arr_s[0]]

        return tps

    def Dunham(self, R, E, rEq, eEq, reduced_mass, wEq, bEq):
        '''Function used to calculate the Dunham Y-parameters using a Dunham polynomial fit

            Functions:
                Dunham_Poly - Fit the energy curve to a dunham-type polynomial

            Variables:
                R           - Array of bond distances
                rEq         - Equilibrium bond distances
                E           - Array of energy values
                eEq         - Minimum energy of energy curve
                hart_cm     - Conversion of Hatree to wavenumber
                a_0         - Zeroth order coefficient for Dunham-type polynomial
                a           - Array of polynomial coefficients
                aN          - Nth order coefficient for Dunahm-type polynomial
                Y           - Dunham Y-parameters

            Returns:
                Dunham Y-parameters
                Dunham fit coefficients

        '''
    

        def Dunham_Poly(x, a1, a2, a3, a4, a5, a6):
                return (h * c_cm * a_0 * J_cm * x**2) * (1 + a1*x + a2*x**2 + a3*x**3 + a4*x**4 + a5*x**5 + a6*x**6)


        R = (R - rEq) / rEq          
        E = (E - eEq) * hart_cm      

        a_0 = wEq**2 / (4 * bEq)     

        a, pcov = curve_fit(Dunham_Poly, R, E)

        a1 = a[0]
        a2 = a[1]
        a3 = a[2]
        a4 = a[3]
        a5 = a[4]
        a6 = a[5]

        Y = np.zeros((5,5))

        Y[0,0] = ((bEq / 8.) * (3 * a2 - 7 * a1**2 / 4.))
        Y[1,0] = (wEq * (1 + (bEq**2 / (4 * wEq**2)) \
              * (25 * a4 - 95 * a1 * a3 / 2. - 67 * a2**2 / 4. + 459 * a1**2 * a2 / 8. - 1155 * a1**4 / 64.)))
        Y[2,0] = ((bEq / 2.) * (3 * ( a2 - 5 * a1**2 / 4.) + (bEq**2 / (2 * wEq**2)) * (245 * a6 \
              - 1365 * a1 * a5 / 2. - 885 * a2 * a4 / 2. - 1085 * a3**2 / 4. + 8535 * a1**2 * a4 / 8. \
              + 1707 * a2**3 / 8. + 7335 * a1 * a2 * a3 / 4. - 23865 * a1**3 * a3 / 16. \
              - 62013 * a1**2 * a2**2 / 32. + 239985 * a1**4 * a2 / 128. - 209055 * a1**6 / 512.)))
        Y[3,0] = ((bEq**2 / (2 * wEq)) \
              * (10 * a4 - 35 * a1 * a3 - 17 * a2**2 / 2. + 225 * a1**2 * a2 / 4. - 705 * a1**4 / 32.))
        Y[4,0] = ((5 * bEq**3 / wEq**2) * (7 * a6 / 2. - 63 * a1 * a5 / 4. - 33 * a2 * a4 / 4. \
              - 63 * a3**2 / 8. + 543 * a1**2 * a4 / 16. + 75 * a2**3 / 16. + 483 * a1 * a2 * a3 / 8. \
              - 1953 * a1**3 * a3 / 32. - 4989 * a1**2 * a2**2 / 64. + 23265 * a1**4 * a2 / 256. \
              - 23151 * a1**6 / 1024. ))
        Y[0,1] = (bEq * (1 + (bEq**2 / (2 * wEq**2)) \
              * (15 + 14.*a1 - 9. * a2 + 15.*a3 - 23 * a1 * a2 + 21 * (a1**2 + a1**3)/2.)))
        Y[1,1] = ((bEq**2 / wEq) * ( 6 * (1 + a1) + (bEq**2 / wEq**2) \
              * (175 * 185 * a1 - 335 * a2 / 2. + 190 * a3 - 225 * a4 / 2. + 175 * a5 + 2295 * a1**2 / 8. \
              - 459 * a1 * a2 + 1425 * a1 * a3 / 4. - 795 * a1 * a4 / 2. + 1005 * a2**2 / 8. - 715 * a2 * a3 / 2. \
              + 1155 * a1**3 / 4. - 9639 * a1**2 * a2 / 16. + 5145 * a1**2 * a3 / 8. + 4677 * a1 * a2**2 / 8. \
              - 14259 * a1**3 * a2 / 16. + 31185 * (a1**4 + a1**5) / 128.)))
        Y[2,1] = ((6. * bEq**3 / wEq**2) * (5 + 10 * a1 - 3 * a2 + 5 * a3 - 13 * a1 * a2 + 15 * (a1**2 + a1**3) / 2.))
        Y[3,1] = ((20 * bEq**4 / wEq**3) * (7 + 21 * a1 - 17 * a2 / 2. + 14 * a3 - 9 * a4 / 2. + 7 * a5 \
              + 225 * a1**2 / 8. - 45 * a1 * a2 + 105 * a1 * a3 / 4. - 51 * a1 * a4 / 2. + 51 * a2**2 / 8. \
              - 45 * a2 * a3 / 2. + 141 * a1**3 / 4. - 945 * a1**2 * a2 / 16. + 435 * a1**2 * a3 / 8. \
              + 411 * a1 * a2**2 / 8. - 1509 * a1**3 * a2 / 16. + 3807 * (a1**4 + a1**5) / 128.))
        Y[0,2] = (-(4 * bEq**3 / wEq**2) \
              * (1 + (bEq**2 / (2 * wEq**2)) * (163 + 199 * a1 - 119 * a2 + 90 * a3 - 45 * a4 \
              - 207 * a1 * a2 + 205 * a1 * a3 / 2. - 333 * a1**2 * a2 / 2. + 693 * a1**2 / 4. \
              + 45 * a2**2 + 126 * (a1**3 + a1**4 / 2.))))
        Y[1,2] = (-(12 * bEq**4 / wEq**3) * (19./2. + 9. * a1 + 9. * a1**2 / 2. - 4 * a2))
        Y[2,2] = (-(24 * bEq**5 / wEq**4) * (65 + 125 * a1 - 61 * a2 + 30 * a3 - 15 * a4 + 495 * a1**2 / 4. \
              - 117 * a1 * a2 + 26 * a2**2 + 95 * a1 * a3 / 2. - 207 * a1**2 * a2 / 2. \
              + 90 * (a1**3 + a1**4 / 2.)))
        Y[0,3] = (16 * bEq**5 * (3 + a1) / wEq ** 4)

        Y[1,3] = ((12 * bEq**6 / wEq**5) \
              * (233 + 279 * a1 + 189 * a1**2 + 63 * a1**3 - 88 * a1 * a2 - 120 * a2 + 80 * a3 / 3.))
        Y[0,4] = (-(64 * bEq**7 / wEq**6) * (13 + 9. * a1 - a2 + 9 * a1**2/4.))

        return Y.flatten(), a


    def SimulatedVibrationalPop(self, *args, **kwargs):
        '''Function to calculate the population of vibrational states
            Uses a Boltzmann distribution
            
            Variables:
                temp        - Temperature in kelvin
                J           - Rotational surface
                v           - Maximum vibrational state
                method      - Method to calculate populations
                                vib - A single J-surface
                                rov - All J-surfaces
                vals        - Eigenvalues
            
            Returns:
                pop         - List of populations for vibrational states

        '''

        temp   = kwargs['temp']
        J      = kwargs['J']
        v      = kwargs['v']
        method = kwargs['method']
        vals   = kwargs['vals']

        if method == 'vib':
            vals = vals[J,:v]
            pop = np.zeros((2, vals.size))
            for vv in range(vals.size):
                en = vals[vv]
                pop[0,vv] = en
                pop[1,vv] = (2*J+1) * np.exp(-1 * (en / J_cm) / (kb * temp))

            pop[1] /= np.sum(pop[1])

        elif method == 'rov':
            pop = np.zeros((vals.shape[0], 2, v))
            for j in range(vals.shape[0]):
                for vv in range(v):
                    pop[j,0,vv] = vals[j,vv]
                    pop[j,1,vv] = (2*j+1) * np.exp(-1 * (vals[j,vv] / J_cm) / (kb * temp))

            pop[:,1] /= np.sum(pop[:,1])

        return pop

    def SimulatedVibrationalInt(self, *args, **kwargs):
        '''Function to calculate the intensity of all vibrational transitions.

           Uses the previous calculated population values and the Einstein-A
               Coefficients.

            Variables:
                J           - Rotational surface
                v           - Highest vibrational state
                vals        - Eigenvalues
                vecs        - Eigenvectors
                tdm         - Transition dipole moment matrix
                method      - Method to calculate intensities
                                vib - A single J-surface
                                rov - All J-surfaces
                pop         - List of populations on vibrational states
                vv          - Initial vibrational state
                vv_         - Final vibrational state
                E_init      - Energy of initial state
                E_final     - Energy of final state
                E           - Energy of transition
                t           - Transition dipole moment element
                f           - f-value of transition
                A           - Einstein-A coefficient of transition
                m_e         - electron mass
                cm_J        - Conversion factor between wavenumbers and Joules
                h           - Reduced Plancks constant
                D_CM        - Conversion factor between Debye and Coulomb-meters
                eps_0       - Vacuum permitivity value
                c           - speed of light


            Returns:
                int_mat     - List of intensities for vibrational excitations

        '''

        J    = kwargs['J']
        v    = kwargs['v']
        vecs = kwargs['vec']
        vals = kwargs['val']
        tdm  = kwargs['tdm']
        pop  = kwargs['pop']
        method = kwargs['method']

        if method == 'vib':
            cc = 0
            int_mat = np.zeros((4, v*v+v))
            for vv in range(v+1):
                for vv_ in range(v):
                    E_init  = vals[J,vv]
                    E_final = vals[J,vv_]
                    E = E_init - E_final

                    t = np.matmul(vecs[J,vv,:], np.matmul(tdm, vecs[J,vv_,:])) / D_au

                    if abs(E) > 0 and abs(t) > 0:
                        f = ((8  * np.pi**2 * m_e) / (3 * h**2 * e**2)) * E*cm_J * (t * D_CM)**2 / (2*J + 1)
                        A = ((64 * np.pi**4) / (3 * h**4 * c**3)) * (E*cm_J)**3 * (t * D_CM)**2 / ((2*J + 1) * (4*np.pi*eps_0))
            
                        int_mat[0,cc] = E
                        int_mat[1,cc] = A * pop[1,vv_]
                        int_mat[2,cc] = vv_
                        int_mat[3,cc] = vv

                        cc += 1

        elif method == 'rov':
            int_mat = np.zeros((vals.shape[0], 4, v**2))
            for j in range(vals.shape[0]):
                cc = 0
                for vv in range(v):
                    for vv_ in range(v):
                        E_init  = vals[j,vv]
                        E_final = vals[j,vv_]
                        E = E_init - E_final

                        t = np.matmul(vecs[j,vv,:], np.matmul(tdm, vecs[j,vv_,:])) / D_au

                        if abs(E) > 0 and abs(t) > 0:
                            f = ((8  * np.pi**2 * m_e) / (3 * h**2 * e**2)) * E*cm_J * (t * D_CM)**2 / (2*j + 1)
                            A = ((64 * np.pi**4) / (3 * h**4 * c**3)) * (E*cm_J)**3 * (t * D_CM)**2 / ((2*j + 1) * (4*np.pi*eps_0))

                            int_mat[j,0,cc] = E
                            int_mat[j,1,cc] = A * pop[j,1,vv_]
                            int_mat[j,2,cc] = vv_
                            int_mat[j,3,cc] = vv

                            cc += 1

        return int_mat


    def SimulatedRotationalPop(self, *args, **kwargs):
        '''Function to calculate the population of vibrational states
            Uses a Boltzmann distribution

            Variables:
                temp        - Temperature in kelvin
                J           - Maximum rotational state
                v           - Vibrational surface
                method      - Method to calculate populations
                                rot - A single v-surface
                                rov - All v-surfaces
                vals        - Eigenvalues

            Returns:
                pop         - List of populations for rotational states

        '''
        
        temp   = kwargs['temp']
        J      = kwargs['J']
        v      = kwargs['v']
        method = kwargs['method']
        vals   = kwargs['vals']

        if method == 'rot':
            if J == 0:
                return np.zeros((2,1))
            vals = vals[:J,v]
            vals -= vals[0]
            pop = np.zeros((2, vals.size))
            for jj in range(vals.size):
                en = vals[jj]
                pop[0,jj] = en
                pop[1,jj] = (2 * jj + 1) * np.exp(-1 * (en / J_cm) / (kb * temp))

            pop[1] /= np.sum(pop[1])

        elif method == 'rov':
            if J == 0:
                return np.zeros((1,2,1))
            pop = np.zeros((J, 2, vals.shape[1]))
            for j in range(J): 
                for vv in range(vals.shape[1]):
                    pop[j,0,vv] = vals[j,vv]
                    pop[j,1,vv] = (2 * j + 1) * np.exp(-1 * (vals[j,vv] / J_cm) / (kb * temp))

            pop[:,1] /= np.sum(pop[:,1])

        return pop

    def SimulatedRotationalInt(self, *args, **kwargs):
        '''Function to calculate the intensity of all rotational transitions.
            
           Uses the previous calculated population values and the Einstein-A
               Coefficients.

            Variables:
                J           - Maximum rotational state
                v           - Vibrational surface
                vals        - Eigenvalues
                vecs        - Eigenvectors
                tdm         - Transition dipole moment matrix
                method      - Method to calculate intensities
                                rot - A single v-surface
                                rov - All v-surfaces
                pop         - List of population for all rotatitional states
                jj          - Initial rotational state
                jj_         - Final rotational state
                E_init      - Energy of initial state
                E_final     - Energy of final state
                E           - Energy of transition 
                t           - Transition dipole moment element
                S           - Honl-London factor
                f           - f-value of transition
                A           - Einstein-A coefficient of transition
                m_e         - electron mass
                cm_J        - Conversion factor between wavenumbers and Joules
                h           - Reduced Plancks constant
                D_CM        - Conversion factor between Debye and Coulomb-meters
                eps_0       - Vacuum permitivity value
                c           - speed of light


            Returns:
                int_mat     - List of intensities for rotational excitations

        '''

        J    = kwargs['J']
        v    = kwargs['v']
        vecs = kwargs['vec']
        vals = kwargs['val']
        tdm  = kwargs['tdm']
        pop  = kwargs['pop']
        method = kwargs['method']

        if method == 'rot':
            cc = 0
            int_mat = np.zeros((4, 2*J))
            for jj in range(0, J+1):
                for jj_ in range(max(0, jj-1), min(jj+2, J)):
                    E_init  = vals[jj,v]
                    E_final = vals[jj_,v]
                    E = E_init - E_final

                    t = np.matmul(vecs[jj,v,:], np.matmul(tdm, vecs[jj_,v,:])) / D_au

                    if abs(E) > 0 and abs(t) > 0:

                        if jj == jj_:
                            S = 1
                        elif jj < jj_:
                            S = jj + 1
                        else:
                            S = jj

                        f = ((8  * np.pi**2 * m_e) / (3 * h**2 * e**2)) * E*cm_J * S * (t * D_CM)**2 / (2*jj + 1)
                        A = ((64 * np.pi**4) / (3 * h**4 * c**3)) * (E*cm_J)**3 * S * (t * D_CM)**2 / ((2*jj + 1) * (4*np.pi*eps_0))

                        int_mat[0,cc] = E
                        int_mat[1,cc] = A * pop[1,jj_]
                        int_mat[2,cc] = jj_
                        int_mat[3,cc] = jj

                        cc += 1

        elif method == 'rov':
            int_mat = np.zeros((vals.shape[0], 4, J**2))
            for v in range(vals.shape[0]):
                cc = 0
                for jj in range(J):
                    for jj_ in range(J):
                        E_init  = vals[jj,v]
                        E_final = vals[jj_,v]
                        E = E_init - E_final
                        
                        t = np.matmul(vecs[jj,v,:], np.matmul(tdm, vecs[jj_,v,:])) / D_au

                        if abs(E) > 0 and abs(t) > 0:

                            if jj == jj_:
                                S = 1
                            elif jj < jj_:
                                S = jj + 1
                            else:
                                S = jj

                            f = ((8  * np.pi**2 * m_e) / (3 * h**2 * e**2)) * E*cm_J * S * (t * D_CM)**2 / (2*jj + 1)
                            A = ((64 * np.pi**4) / (3 * h**4 * c**3)) * (E*cm_J)**3 * S * (t * D_CM)**2 / ((2*jj + 1) * (4*np.pi*eps_0))

                            int_mat[v,0,cc] = E
                            int_mat[v,1,cc] = A * pop[jj_,1,v]
                            int_mat[v,2,cc] = jj_
                            int_mat[v,3,cc] = jj

                            cc += 1
        
        return int_mat
        

    def SimulatedRovibrationalPop(self, *args, **kwargs):
        '''Function to calculate the population of vibrational states
            Uses a Boltzmann distribution

            Variables:
                temp        - Temperature in kelvin
                J           - Maximum rotational state
                v           - Maximum vibrational state
                vals        - Eigenvalues

            Returns:
                pop         - List of populations for rovibrational states

        '''

        temp = kwargs['temp']
        J    = kwargs['J']
        v    = kwargs['v']
        vals = kwargs['vals']

        if J == 0 and v == 0:
            return np.zeros((4,1))

        if v == 0:
            pop  = np.zeros((4,J))
            vals = vals[:J,0]
            for jj in range(vals.size):
                en = vals[jj]
                pop[0,jj] = en
                pop[1,jj] = (2 * jj + 1) * np.exp(-1 * (en / J_cm) / (kb * temp))
                pop[2,jj] = jj
                pop[3,jj] = v
                
            pop[1] /= np.sum(pop[1])
            return pop

        elif J == 0:
            pop  = np.zeros((4,v))
            vals = vals[0,:v]
            for vv in range(vals.size):
                en = vals[vv]
                pop[0,vv] = en
                pop[1,vv] = np.exp(-1 * (en / J_cm) / (kb * temp))
                pop[2,vv] = J
                pop[3,vv] = vv

            pop[1] /= np.sum(pop[1])
            return pop

        else:
            vals  = vals[:J+1,:v+1]
            pop = np.zeros((4,vals.size))
            c = 0
            for jj in range(J+1):
                for vv in range(v+1):
                    en = vals[jj,vv]
                    pop[0,c] = en
                    pop[1,c] = (2 * jj + 1) * np.exp(-1 * (en / J_cm) / (kb * temp))
                    pop[2,c] = jj
                    pop[3,c] = vv
                    c+=1

            pop[1] /= np.sum(pop[1])
            return pop



    def SimulatedRovibrationalInt(self, *args, **kwargs):
        '''Function to calculate the intensity of all rotational transitions.

           Uses the previous calculated population values and the Einstein-A
               Coefficients.

            Variables:
                vecs        - Eigenvectors
                tdm         - Transition dipole moment matrix
                pop         - List of populations for all rovibrational states
                jj          - Initial rotational state
                j           - Final rotational state
                vv          - Initial vibrational state
                v           - Final vibrational state
                E_init      - Energy of initial state
                E_final     - Energy of final state
                E           - Energy of transition
                pop_init    - Population of initial state
                t           - Transition dipole moment element
                S           - Honl-London factor
                f           - f-value of transition
                A           - Einstein-A coefficient of transition
                m_e         - electron mass
                cm_J        - Conversion factor between wavenumbers and Joules
                h           - Reduced Plancks constant
                D_CM        - Conversion factor between Debye and Coulomb-meters
                eps_0       - Vacuum permitivity value
                c           - speed of light


            Returns:
                int_mat     - Matrix of intensity values

        '''

        vec = kwargs['vec']
        pop = kwargs['pop']
        tdm = kwargs['tdm']

        int_mat = np.zeros((8, pop.shape[1]*pop.shape[1]))

        cc = 0

        for x in range(pop.shape[1]):
            j = int(pop[2,x])
            v = int(pop[3,x])
            E_final = pop[0,x]
            for xx in range(pop.shape[1]):
                jj = int(pop[2,xx])
                vv = int(pop[3,xx])
                E_init = pop[0,xx]
                pop_init = pop[1,xx]

                if abs(j-jj) < 2:

                    E = E_final - E_init

                    t = np.matmul(vec[j,v,:], np.matmul(tdm, vec[jj,vv,:])) / D_au

                    if abs(E) > 0 and abs(t) > 0:
                        if j == jj:
                            S = 1
                        elif j < jj:
                            S = j + 1
                        else:
                            S = j

                        f = ((8  * np.pi**2 * m_e) / (3 * h**2 * e**2)) * E*cm_J * S * (t * D_CM)**2 / (2*j + 1)
                        A = ((64 * np.pi**4) / (3 * h**4 * c**3)) * (E*cm_J)**3 * S * (t * D_CM)**2 / ((2*j + 1) * (4*np.pi*eps_0))

                        int_mat[0,cc] = vv
                        int_mat[1,cc] = jj
                        int_mat[2,cc] = v
                        int_mat[3,cc] = j
                        int_mat[4,cc] = E_init
                        int_mat[5,cc] = E_final
                        int_mat[6,cc] = E
                        int_mat[7,cc] = A * pop_init

                        cc += 1  

        return int_mat

