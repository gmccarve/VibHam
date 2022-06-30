import sys
import numpy as np
import pandas as pd
import random
import os
import traceback
import io
import csv
import time

from Conversions import *
from Atoms import Atoms
from Interpolate import Interpolate
from Hamil import Hamil, Wavefunctions
from Spectra import Spectra
from Input import Input

class RunVibHam():

    def __init__(self, args):
        self.args = args

    def BREAK(self):
        print ("\n\t*****************************************************\n")

    def diagonalize(self, M):
        '''Function used to diagonalize square matrices or tensors composed of square matrices.'''
        
        val = np.zeros((M.shape[:2]))
        vec = np.zeros((M.shape))
        for j in range(M.shape[0]):
            val_, vec_ = np.linalg.eig(M[j])
            idx = val_.argsort()
            val[j] = val_[idx]
            vec[j] = vec_[:,idx].T
        return val*J_cm, vec


    def LoadData(self):
        if not self.args.Data:
            print ("MUST GIVE FILE TO READ IN")
            exit()

        if not os.path.exists(self.args.Data):
            print ("FILE", self.args.Data, "NOT FOUND")
            exit()

        try:
            self.Data = np.loadtxt(self.args.Data).transpose()
        except:
            print ("FILE", self.args.Data, "NOT ABLE TO BE READ BY NUMPY")
            exit()

        print ()
        print ("\tData file ", self.args.Data, " successfully loaded")
        print ()

        if self.Data.shape[0] == 2:
            self.Dip_bool = False
            
            print ("\tNo Dipole Moment Information Provided")
            print ("\tSkipping all TDM calculations\n")
        
        elif self.Data.shape[0] == 3:
            self.Dip_bool = True

            print ("\tDipole Moment Information Provided")
            print ("\tPerforming all TDM calculations\n")

        self.BREAK()
    
    
    def ConvertData(self):
        print ("\tConverting bond distance to Angstrom")
        print ("\tConverting electronic energy to Hartrees")

        if self.args.R_unit == 'ang':
            self.Data[0] *= 1
        elif self.args.R_unit == 'bohr':
            self.Data[0] *= bohr_ang
        elif args.R_unit == 'm':
            self.Data[0] *= m_ang

        if self.args.E_unit == 'hartree':
            self.Data[1]  *= 1
        elif args.E_unit == 'kj/mol':
            self.Data[1]  /= kcal_kj
            slef.Data[1]   /= hart_kcal
        elif args.E_unit == 'kcal/mol':
            self.Data[1]   /= hart_kcal
        elif args.E_unit == 'ev':
            self.Data[1]   /= hart_eV
        elif args.E_unit == 'cm':
            self.Data[1]   /= hart_cm
        elif args.E_unit == 'j':
            self.Data[1]   *= J_hart
            
        if self.Dip_bool == True:
            
            print ("\tConverting dipole moments to Debye")
            if self.args.Dip_unit == 'debye':
                self.Data[2]  *= 1
            elif self.args.Dip_unit == 'au':
                self.Data[2]  *= au_D

    def AtomicInformation(self):

        if not self.args.Atoms and not self.args.Masses:
            print ("\tMUST PROVIDE EITHER ATOMS OR MASSES\n")
            print ("\tQuitting Program\n")
            exit()

        self.Atoms = Atoms()

        if self.args.Atoms:
            self.Atom1 = self.Atoms.AtomDict[self.args.Atoms[0]]
            self.Atom2 = self.Atoms.AtomDict[self.args.Atoms[1]]

            print ()
            print ("\tAtom 1 - ", self.args.Atoms[0])
            print ("\tAtom 2 - ", self.args.Atoms[1])

            if not self.args.Isotopes:
                self.Iso1 = 0
                self.Iso2 = 0

            else:
                try:                                
                    self.Iso1 = self.Atom1[self.args.Isotopes[0]]
                    self.Iso2 = self.Atom2[self.args.Isotopes[1]]
                except:
                    print ("\tIsotope Not Found\n")
                    print ("\tQuitting Program\n")
                    exit()

            self.Mass1 = self.Atom1[self.Iso1]
            self.Mass2 = self.Atom2[self.Iso2]

        elif args.Masses:
            self.Mass1 = self.args.Masses[0]
            self.Mass2 = self.args.Masses[1]

        print ()
        print ("\tMass 1 - ", self.Mass1, "AMU")
        print ("\tMass 2 - ", self.Mass2, "AMU")
        print ()

        self.BREAK()


    def Interpolate(self):

        inter = Interpolate(temp_data = self.Data,
                            atoms     = [self.Atom1, self.Atom2],
                            isotopes  = [self.Iso1, self.Iso2],
                            masses    = [self.Mass1, self.Mass2],
                            charge    = self.args.Charge,
                            numpoints = [self.args.InterPoints, self.args.InterPoints],
                            order_e   = self.args.Energy_Fit,
                            order_d   = self.args.Dipole_Fit
                            )

        print ("\tInterpolating the potential energy curve\n")

        self.rEq = inter.rEq            # Equilibrium Bond Length
        self.eEq = inter.eEq            # Minimum Energy
        self.bEq = inter.bEq            # Equilibrium Rotational Constant
        self.wEq = inter.omega          # Equilbrium Vibrational Constant
        
        self.PEC_r = inter.PEC_r_
        self.PEC_e = inter.PEC_e_

        self.energy_coef = inter.Coef         # Power Series Expansion Coefficients

        self.reduced_mass = inter.reducedMass
        self.beta         = inter.beta
        self.nu           = inter.nu

        self.energy_mad = inter.inter_mad
        self.energy_cod = inter.inter_cod
        self.energy_rmse = inter.inter_rmse

        print ()
        print ("\tMinimum energy")
        print ("\t\t{:<.9f}{:>4s}".format(self.eEq,  "Eh"))
        print ("\t\t{:<.9f}{:>7s}".format(self.eEq*hart_cm,  "cm^-1"))
        print ("\t\t{:<.9f}{:>10s}".format(self.eEq*hart_kcal,  "kcal/mol"))

        print ()
        print ("\tEquilibrium Bond Distance")
        print ("\t\t{:<.8f}{:>9s}".format(self.rEq, "Angstrom"))
        print ("\t\t{:<.8f}{:>5s}".format(self.rEq*ang_bohr, "Bohr"))
        
        print ()
        print ("\tEquilibrium vibrational constant")
        print ("\t\t{:<.9f}{:>7s}".format(self.wEq,  "cm^-1"))
        print ("\t\t{:<.3f}{:>5s}".format(self.wEq*cm_mhz,  "MHz"))

        print ()
        print ("\tEquilibrium rotational constant")
        print ("\t\t{:<.9f}{:>7s}".format(self.bEq,  "cm^-1"))
        print ("\t\t{:<.3f}{:>6s}".format(self.bEq*cm_mhz,  "MHz"))

        print ()
        print ("\tReduced Mass")
        print ("\t\t{:<.9f}{:>6s}".format(self.reduced_mass/amu_kg,  "AMU"))

        print ()
        print ("\tBeta Value")
        print ("\t\t{:<.9f}{:>12s}".format(self.beta,  "Angstrom^-1"))

        print ()
        print ("\tHarmonic Frequency")
        print ("\t\t{:<.2f}{:>6s}".format(self.nu,  "s^-1"))

        print ("\n")
        print ("\tMean Absolute Error")
        print ("\t\t{:<.9f}{:>6s}".format(self.energy_mad,  "Eh"))

        print ()
        print ("\tRoot Mean Squared Error ")
        print ("\t\t{:<.9f}{:>6s}".format(self.energy_rmse,  "Eh"))

        print ()
        print ("\tCoefficient of Determination")
        print ("\t\t{:<.9f}{:>6s}".format(self.energy_cod,  ""))

        print ("\n")
        print ("\tPower Series Expansion Coefficients (Hartree/Angstrom^n)\n")
        for j in range(self.args.Energy_Fit, 1, -1):
            print ("\t\t{:>3d}{:>5s}{:>15e}".format(j, " -  ", self.energy_coef[self.args.Energy_Fit-j]))
        print ()


        if self.Dip_bool == True:
            self.BREAK()

            print ("\tInterpolating the dipole curve\n")

            self.dEq         = inter.dEq        # Equilibrium dipole moment value
            self.dip_mad     = inter.D_mad      # Mean Absolute Error
            self.dip_cod     = inter.D_cod      # Coefficient of Determination
            self.dip_rmse    = inter.D_rmse     # Root Mean Squared Error
            self.dipole_coef = inter.polyDfit   # Coefficients for dipole moment function

    
            print ()
            print ("\tEquilibrium Dipole Moment")
            print ("\t\t{:<.9f}{:>7s}".format(self.dEq,  "Debye"))
            print ("\t\t{:<.9f}{:>4s}".format(self.dEq*D_au,  "au"))

            print ()
            print ("\tMean Absolute Error")
            print ("\t\t{:<.9f}{:>6s}".format(self.dip_mad,  "Debye"))

            print ()
            print ("\tRoot Mean Squared Error ")
            print ("\t\t{:<.9f}{:>6s}".format(self.dip_rmse,  "Debye"))

            print ()
            print ("\tCoefficient of Determination")
            print ("\t\t{:<.9f}{:>6s}".format(self.dip_cod,  ""))

        
            print ("\n")
            print ("\tPower Series Expansion Coefficients (Debye/Angstrom^n)\n")
            for j in range(0, self.args.Dipole_Fit+1):
                print ("\t\t{:>3d}{:>5s}{:>15e}".format(j, " -  ", self.dipole_coef[self.args.Dipole_Fit-j]/(D_au/ang_m**j)))
            print ()

        self.BREAK()

    def GenerateMatrices(self):

        self.maxV = self.args.v
        self.maxJ = self.args.J
        
        print ("\tMaximum Vibrational Quantum Number - ", self.maxV)
        print ("\tMaximum Rotational Quantum Number  - ", self.maxJ)
        
        print ()
        print ("\tGenerating Harmonic Hamiltonian Matrix")
        
        gen_hamil = Hamil(ID   = 'harm',
                          maxv = self.maxV+1,
                          nu   = self.nu
                          )

        self.harmonic = gen_hamil.harmonic
        np.save("Harmonic_Matrix", self.harmonic)
        print ("\t\tMatrix saved to 'Harmonic_Matrix.npy'")

        print ()
        print ("\tGenerating Anharmonic Hamiltonian Matrix")

        gen_hamil = Hamil(ID = 'anharm',
                              maxV = self.maxV+1,
                              coef = self.energy_coef,
                              beta = self.beta
                              )
        
        self.anharmonic = gen_hamil.anharmonic
        np.save("ANharmonic_Matrix", self.anharmonic)
        print ("\t\tMatrix saved to 'Anharmonic_Matrix.npy'")


        if self.maxJ > 0:
            print ()
            print ("\tGenerating Centrifugal Potential Hamiltonian Matrix")

            gen_hamil = Hamil(ID = 'cent',
                              maxJ = self.maxJ,
                              maxV = self.maxV+1,
                              rEq  = self.rEq,
                              beta = self.beta,
                              reduced_mass = self.reduced_mass,
                              Trap = self.args.Trap
                              )

            self.centrifugal = gen_hamil.centrifugal
            np.save("Centrifugal_Matrix", self.centrifugal)
            print ("\t\tMatrix saved to 'Centrifugal_Matrix.npy'")


            print ()
            print ("\tGenerating Total Hamiltonian Matrix")

            self.total = self.harmonic + self.anharmonic + self.centrifugal
            np.save("Total_Matrix", self.total)
            print ("\t\tMatrix saved to 'Total_Matrix.npy'")

        
        else:
            print ()
            print ("\tGenerating Total Hamiltonian Matrix")

            self.total = np.zeros((1, self.maxV+1, self.maxV+1))
            self.total[0] = self.harmonic + self.anharmonic
            np.save("Total_Matrix", self.total)
            print ("\t\tMatrix saved to 'Total_Matrix.npy'")


        if self.Dip_bool == True:
            print ()
            print ("\tGenerating Transition Dipole Moment Hamiltonian Matrix")

            gen_hamil = Hamil(ID = 'tdm',
                              maxV = self.maxV+1,
                              coef = self.dipole_coef,
                              beta = self.beta
                              )

            self.tdm = gen_hamil.tdm
            np.save("TDM_Matrix", self.tdm)
            print ("\t\tMatrix saved to 'TDM_Matrix.npy'")



        print ()
        print ("\tChecking Matrix Stability")

        self.total_val, self.total_vec = self.diagonalize(self.total)

        stab_v = self.maxV+1
        stab_bool = False

        if np.amin(self.total_val.flatten()) < 0:
            while stab_bool == False:
                self.total_val, self.total_vec = self.diagonalize(self.total[:, :stab_v, :stab_v])

                if np.amin(self.total_val.flatten()) < 0:
                    stab_v -= 1 
                else:
                    stab_bool = True


        print ("\t\tMatrix Stable up to v = ", stab_v-1)

        
        print ()
        print ("\tDetermining Convergence of States up to", self.args.EigVal, "cm^-1")

        for j in range(self.maxJ+1):
            idx = self.total_val[j].argsort()
            self.total_val[j] = self.total_val[j,idx]
            self.total_vec[j] = self.total_vec[j, :, idx]

        self.total_val_, self.total_vec_ = self.diagonalize(self.total[:, :stab_v-1, :stab_v-1])

        for j in range(self.maxJ+1):
            idx = self.total_val_[j].argsort()
            self.total_val_[j] = self.total_val_[j,idx]
            self.total_vec_[j] = self.total_vec_[j, :, idx]
    
        self.trunc_arr = np.zeros((self.maxJ+1))
        for j in range(self.maxJ+1):
            diff_arr = self.total_val_[j] - self.total_val[j,:-1]
            trunc_val = np.where(diff_arr < self.args.EigVal)[0].flatten()[-1]
            self.trunc_arr[j] = trunc_val

            print ("\t\tEigenvalues converged up to v = ", trunc_val, "on the J = ", j, "surface")

        self.BREAK()

    def TurningPoints(self):
        tps = Spectra()

        self.tps = np.zeros((self.maxJ+1, 2, self.maxV+1))

        for j in range(self.maxJ+1):

            self.tps[j] = tps.TurningPoints(self.PEC_r+self.rEq,
                                            self.PEC_e*hart_cm,
                                            self.total_val[j],
                                            self.rEq,
                                            )

    def PrintEigen(self):
        self

def Main():

    start = time.time()

    args = Input()

    VibHam = RunVibHam(args)

    VibHam.LoadData()
    VibHam.ConvertData()
    VibHam.AtomicInformation()
    VibHam.Interpolate()
    VibHam.GenerateMatrices()
    VibHam.TurningPoints()
    VibHam.PrintEigen()

if __name__ == "__main__":


    Main()

