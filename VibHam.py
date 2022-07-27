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
from HamilP import Hamil, Wavefunctions
from SpectraP import Spectra
from Input import Input
import GUI

import SpectraF

class RunVibHam():

    def __init__(self, args):
        self.args = args

        self.maxV = self.args.v
        self.maxJ = self.args.J

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
            vec[j] = vec_[:,idx]
        return val*J_cm, vec


    def LoadData(self):
        '''Function used to load and order the energy/dipole curve information'''
        if not self.args.Data and not self.args.LoadData:
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

        idx = self.Data[0].argsort()
        self.Data = self.Data[:, idx]

        self.BREAK()
    
    
    def ConvertData(self):
        '''Function used to convert the datafile to appropriate units'''
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
        '''Function used to assign masses based on either atom-type and isotope of manually given mass'''
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
        '''Function used to interpolate the energy/dipole curves'''
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
            print ("\t\t{:<.9f}{:>7s}".format(self.dEq*au_D,  "Debye"))
            print ("\t\t{:<.9f}{:>4s}".format(self.dEq,  "au"))

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

    def GenerateHarmonicMatrix(self):
        '''Function used to generate the harmonic Hamiltonian matrix'''
        print ()
        print ("\tGenerating Harmonic Hamiltonian Matrix")

        gen_hamil = Hamil(ID     = 'harm',
                          maxV   = self.maxV+1,
                          nu     = self.nu, 
                          method = self.args.Method
                          )

        self.harmonic = gen_hamil.harmonic
        
        np.save("Harmonic_Matrix", self.harmonic)
        print ("\t\tMatrix saved to 'Harmonic_Matrix.npy'")


    def GenerateAnharmonicMatrix(self):
        '''Function used to generate the anharmonic Hamiltonian matrix'''
        print ()
        print ("\tGenerating Anharmonic Hamiltonian Matrix")

        gen_hamil = Hamil(ID = 'anharm',
                              maxV = self.maxV+1,
                              coef = self.energy_coef,
                              beta = self.beta, 
                              method = self.args.Method
                              )

        self.anharmonic = gen_hamil.anharmonic
        
        np.save("Anharmonic_Matrix", self.anharmonic)
        print ("\t\tMatrix saved to 'Anharmonic_Matrix.npy'")


    def GenerateCentrifugalMatrix(self):
        '''Function used to generate the centrifugal potential Hamiltonian matrix'''
        if self.maxJ > 0:
            print ()
            print ("\tGenerating Centrifugal Potential Hamiltonian Matrix")

            gen_hamil = Hamil(ID           = 'cent',
                              maxJ         = self.maxJ,
                              maxV         = self.maxV+1,
                              rEq          = self.rEq,
                              beta         = self.beta,
                              reduced_mass = self.reduced_mass,
                              Trap         = self.args.Trap, 
                              method       = self.args.Method
                              )

            self.centrifugal = gen_hamil.centrifugal
            
            np.save("Centrifugal_Matrix", self.centrifugal)
            print ("\t\tMatrix saved to 'Centrifugal_Matrix.npy'")

        else:
            self.centrifugal = np.zeros((1, self.maxV+1, self.maxV+1))


    def GenerateTotalMatrix(self):
        '''Function used to generate the total Hamiltonian matrix'''
        print ()
        print ("\tGenerating Total Hamiltonian Matrix")

        self.total = self.harmonic + self.anharmonic + self.centrifugal

        self.total_val, self.total_vec = self.diagonalize(self.total)

        np.save("Total_Matrix", self.total)
        print ("\t\tMatrix saved to 'Total_Matrix.npy'")

    
    def GenerateTDMMatrix(self):
        '''Function used to generate the transition dipole moment Hamiltonian matrix'''
        if self.Dip_bool == True:
            print ()
            print ("\tGenerating Transition Dipole Moment Hamiltonian Matrix")

            gen_hamil = Hamil(ID     = 'tdm',
                              maxV   = self.maxV+1,
                              coef   = self.dipole_coef,
                              beta   = self.beta, 
                              method = self.args.Method
                              )

            self.tdm = gen_hamil.tdm

            np.save("TDM_Matrix", self.tdm)
            print ("\t\tMatrix saved to 'TDM_Matrix.npy'")
        
        else:
            self.tdm = np.zeros((self.maxV+1, self.maxV+1))


    def CheckMatrixStability(self):
        '''Function used to check the stability of the total matrix through the presence
            of negative eigenvalues'''
        print ("\tChecking Matrix Stability")

        self.total_val, self.total_vec = self.diagonalize(self.total)

        self.stab_v = self.maxV+1
        stab_bool = False

        if np.amin(self.total_val.flatten()) < 0:
            while stab_bool == False:
                self.total_val, self.total_vec = self.diagonalize(self.total[:, :self.stab_v, :self.stab_v])

                if np.amin(self.total_val.flatten()) < 0:
                    self.stab_v -= 1
                else:
                    stab_bool = True

        print ("\t\tMatrix Stable up to v = ", self.stab_v-1)

    
    def CheckTruncationError(self):
        '''Function used to check the truncation error of the finite Hamiltonian matrix'''
        print ()
        print ("\tDetermining Convergence of States to Within", self.args.EigVal, "cm^-1")

        for j in range(self.maxJ+1):
            idx = self.total_val[j].argsort()
            self.total_val[j] = self.total_val[j,idx]
            self.total_vec[j] = self.total_vec[j, :, idx]

        self.total_val_, self.total_vec_ = self.diagonalize(self.total[:, :self.stab_v-1, :self.stab_v-1])

        for j in range(self.maxJ+1):
            idx = self.total_val_[j].argsort()
            self.total_val_[j] = self.total_val_[j,idx]
            self.total_vec_[j] = self.total_vec_[j, :, idx]

        self.trunc_arr     = np.zeros((self.maxJ+1))
        self.trunc_err_arr = np.zeros((self.maxJ+1, self.total_val.shape[1]-1))

        for j in range(self.maxJ+1):
            diff_arr  = self.total_val_[j] - self.total_val[j,:-1]
            trunc_val = np.where(diff_arr < self.args.EigVal)[0].flatten()[-1]
            
            self.trunc_arr[j]     = trunc_val
            self.trunc_err_arr[j] = diff_arr

            print ("\t\tEigenvalues converged up to v =", trunc_val, "on the J =", j, "surface")


    def GenerateMatrices(self):
        '''Function used to generate the required matrices'''

        print ("\tMaximum Vibrational Quantum Number - ", self.maxV)
        print ("\tMaximum Rotational Quantum Number  - ", self.maxJ)

        self.GenerateHarmonicMatrix()
        self.GenerateAnharmonicMatrix()
        self.GenerateCentrifugalMatrix()
        self.GenerateTotalMatrix()
        self.GenerateTDMMatrix()
        self.CheckMatrixStability()
        self.CheckTruncationError()

        self.BREAK()


    def LoadMatrices(self):
        '''Function used to load in precomputed matrices'''

        self.script_path = os.getcwd()

        print ("\tAttempting to load 'Harmonic_Matrix.npy'")

        if not os.path.exists("Harmonic_Matrix.npy"):
            print ("\t\tNo 'Harmonic_Matrix.npy' file found")
            self.GenerateHarmonicMatrix()
        
        else:
            try:
                self.harmonic = np.load("Harmonic_Matrix.npy")
                print ("\t\tMatrix successfully loaded.")
            
            except:
                print ("\t\tMatrix unable to be loaded.")
                self.GenerateHarmonicMatrix()

            if self.args.v != self.harmonic.shape[0]:
                self.BREAK()
                print ("\tArgument 'v' must match the size of the loaded matrices")
                print ("\tQutting program")
                sys.exit()


        print ("\n")
        print ("\tAttempting to load 'Anharmonic_Matrix.npy'")

        if not os.path.exists("Anharmonic_Matrix.npy"):
            print ("\t\tNo 'Anharmonic_Matrix.npy' file found")
            self.GenerateAnharmonicMatrix()

        else:
            try:
                self.anharmonic = np.load("Anharmonic_Matrix.npy")
                print ("\t\tMatrix successfully loaded.")

            except:
                print ("\t\tMatrix unable to be loaded.")
                self.GenerateAnharmonicMatrix()


        if self.maxJ > 0:
            print ("\n")
            print ("\tAttempting to load 'Centrifugal_Matrix.npy'")

            if not os.path.exists("Centrifugal_Matrix.npy"):
                print ("\t\tNo 'Centrifugal_Matrix.npy' file found")
                self.GenerateCentrifugalMatrix()

            else:
                try:
                    self.centrifugal = np.load("Centrifugal_Matrix.npy")
                    print ("\t\tMatrix successfully loaded.")

                except:
                    print ("\t\tMatrix unable to be loaded.")
                    self.GenerateCentrifugalMatrix()

        if self.Dip_bool == True:
            print ("\n")
            print ("\tAttempting to load 'TDM_Matrix.npy'")

            if not os.path.exists("TDM_Matrix.npy"):
                print ("\t\tNo 'TDM_Matrix.npy' file found")
                self.GenerateTDMMatrix()

            else:
                try:
                    self.tdm = np.load("TDM_Matrix.npy")
                    print ("\t\tMatrix successfully loaded.\n")

                except:
                    print ("\t\tMatrix unable to be loaded.")
                    self.GenerateTDMMatrix()


        self.GenerateTotalMatrix()
        
        self.BREAK()

        self.CheckMatrixStability()
        self.CheckTruncationError()

        self.BREAK()


    def TurningPoints(self):
        '''Function used to calculate the turning points along the energy curve'''
        tps = Spectra()

        self.tps = np.zeros((self.maxJ+1, 2, self.stab_v))

        for j in range(self.maxJ+1):

            self.tps[j] = tps.TurningPoints(self.PEC_r+self.rEq,
                                            self.PEC_e*hart_cm,
                                            self.total_val[j],
                                            self.rEq,
                                            )

    def PrintEigen(self):
        '''Function used to print the eigenvalue information'''

        if self.args.Print < 3:
            print ("\tThese are the converged energy levels and their respecitve turning points:")
            print ()
        else:
            print ("\tThese are the energy levels and their respecitve turning points:")
            print ()

        dash = "\t- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"

        val_strings = ['State', 'Energy (cm^-1)', 'Error', 'ΔE', 'LEFT TP', 'RIGHT TP', 'CENTER']


        if self.args.Print < 3:
            self.max_print_v = self.trunc_arr
        else:
            self.max_print_v = np.ones((self.maxV+1)) * self.maxV


        for j in range(self.total_val.shape[0]-1, -1, -1):

            print ("\tOn the J = ", j, "surface")
            print ()
            
            print ('\t{:>4s}{:>21s}{:>14s}{:>14s}{:>13s}{:>13s}{:>13s}'.format(*val_strings))
            
            if self.args.Print == 2 or self.args.Print == 4:
                print (dash)

                vec_strings = ['Vector', 'Contribution', 'Weight']
                print ('\t{:>4s}{:>21s}{:>13s}\n\n'.format(*vec_strings))

            for v in range(int(self.max_print_v[j]), -1, -1):

                if v == 0:
                    diff = 0.0
                else:
                    diff = self.total_val[j,v] - self.total_val[j,v-1]

                if v == self.maxV:
                    trunc_err = np.inf
                else:
                    trunc_err = self.trunc_err_arr[j,v]

                if self.tps[j,0,v] == 0.0:
                    left_tp = '-'
                    center_tp = '-'
                else:
                    left_tp = str(round(self.tps[j,0,v], 7))

                if self.tps[j,1,v] == 0.0:
                    right_tp = '-'
                    center_tp = '-'
                else:
                    right_tp = str(round(self.tps[j,1,v], 7))

                if left_tp != '-' and right_tp != '-':
                    center_tp = str((float(left_tp) + float(left_tp)) / 2.)
                else:
                    center_tp = '-'
                
                print ('\t{:<13s}{:>13f}{:>14e}{:>14f}{:>13s}{:>13s}{:>13s}'.format(
                            str(v),
                            round(self.total_val[j,v], 7),
                            round(trunc_err, 5),
                            round(diff, 7),
                            left_tp,
                            right_tp,
                            center_tp,
                            )
                            )

                if self.args.Print == 2 or self.args.Print == 4:
                    print (dash)

                    for vv in range(int(self.max_print_v[j]), -1, -1):
                        print ('\t{:<13s}{:>13f}{:>13f}'.format(
                            str(vv),
                            round(self.total_vec[j, vv, v], 7),
                            round(self.total_vec[j, vv, v]**2, 7)))
                    print ()
                    print ()

            print ()

        self.BREAK()

    def DissociationEnergy(self):
        '''Function used to approximate the dissociation energies'''
        self.diss_e = self.Data[1,-1] - self.eEq
        self.diss_err = self.Data[1,-1] - self.Data[1, -2]

        self.diss_0 = self.diss_e - self.total_val[0,0]/hart_cm
    
        print ("\tDissociation Energy\n")
        print ()
        print ("\tDe")
        print ("\t\t{:>12f}   +/- {:>12f}{:>10s}".format(self.diss_e, self.diss_err, "Eh"))
        print ("\t\t{:>12f}   +/- {:>12f}{:>10s}".format(self.diss_e*hart_eV, self.diss_err*hart_eV, "eV"))
        print ("\t\t{:>12f}   +/- {:>12f}{:>10s}".format(self.diss_e*hart_kcal, self.diss_err*hart_kcal, "kcal/mol"))
        print ("\t\t{:>12f}   +/- {:>12f}{:>10s}".format(self.diss_e*hart_cm, self.diss_err*hart_cm, "cm^-1"))
        print ()
        print ("\tD0")
        print ("\t\t{:>12f}   +/- {:>12f}{:>10s}".format(self.diss_0, self.diss_err, "Eh"))
        print ("\t\t{:>12f}   +/- {:>12f}{:>10s}".format(self.diss_0*hart_eV, self.diss_err*hart_eV, "eV"))
        print ("\t\t{:>12f}   +/- {:>12f}{:>10s}".format(self.diss_0*hart_kcal, self.diss_err*hart_kcal, "kcal/mol"))
        print ("\t\t{:>12f}   +/- {:>12f}{:>10s}".format(self.diss_0*hart_cm, self.diss_err*hart_cm, "cm^-1"))
        print ()
        
        self.BREAK()

    def Excitations(self):
        '''Function used to calculate all rovibrational excitations'''
        excite = Spectra()
        if self.args.Method == 'python':
            self.excitations = excite.Excitations(self.total_val,
                                                  self.total_vec,
                                                  int(self.max_print_v[0]),
                                                  self.maxJ,
                                                  self.tdm[:self.stab_v, :self.stab_v]
                                                  )
        elif self.args.Method == 'fortran':
            SpectraF.excitations(self.total_val,
                                 self.total_vec,
                                 int(self.max_print_v[0])-1,
                                 self.maxV,
                                 self.maxJ,
                                 self.tdm[:self.stab_v, :self.stab_v]
                                 )

            self.excitations = np.loadtxt("exc.tmp")
            os.system("rm exc.tmp")


        strings = ['Vi', 'Ji', 'Vf', 'Jf', 'Ei', 'Ef', 'ΔE', 'TDM', 'f', 'A']
        print ("\t{:<5s}{:<5s}{:<5s}{:<5s}{:>15s}{:>15s}{:>15s}{:>15s}{:>15s}{:>15s}".format(*strings))
    
        for c, val in enumerate(self.excitations):

            print ("\t{:<5d}{:<5d}{:<5d}{:<5d}{:>15f}{:>15f}{:>15f}{:>15e}{:>15e}{:>15e}".format(
                                    int(val[0]), int(val[1]), int(val[2]), int(val[3]), 
                                    val[4], val[5],val[6],
                                    val[7], 
                                    val[8],
                                    val[9]
                                    ))
            try:
                if self.excitations[c,0] != self.excitations[c+1,0]:
                    print ("\n")
                if self.excitations[c,2] != self.excitations[c+1,2]:
                    print ()
            except:
                pass

        self.BREAK()

    def Constants(self):
        '''Function used to calculate all rovibrational spectroscopic constants'''
        spectra = Spectra()

        print ("\tPure Vibrational Constants on Different J-Surfaces\n")

        self.vib_spec_values = spectra.Vibrational(self.total_val, 
                                                   min(self.args.Constants, int(self.trunc_arr[0]))+1,
                                                   self.maxJ
                                                   )

        spc = np.append(np.arange(0, self.maxJ+1, 4), self.maxJ+1)

        for v_ in range(spc.size-1):
            for j_ in range(spc[v_], spc[v_+1]):
                print ("\t{:>14s}".format("J = " + str(j_)), end='')
            print ()
            for vv_ in range(min(self.args.Constants, int(self.trunc_arr[0]))+1):
                if vv_ == 0:
                    print ("\t{:>6s}".format("we"), end=' ')
                elif vv_ == 1:
                    print ("\t{:>6s}".format("ωexe"), end=' ')
                elif vv_ == 2:
                    print ("\t{:>6s}".format("ωeye"), end=' ')
                elif vv_ == 3:
                    print ("\t{:>6s}".format("ωeze"), end=' ')
                else:
                    print ("\t{:>6s}".format("ωe" + str(vv_) + "e"), end=' ')

                for j_ in range(spc[v_], spc[v_+1]):
                    print ("\t{:>13e}".format(self.vib_spec_values[vv_, j_]), end=' ')
                print ()
            print ()

        self.BREAK()

        if self.args.J != 0:

            print ("\tPure Rotational Constants on Different v-Surfaces\n")

            self.rot_spec_values = spectra.Rotational(self.total_val, 
                                                      int(self.trunc_arr[0]),
                                                      min(self.args.Constants, self.maxJ-1)
                                                       )

            spc = np.append(np.arange(0, int(self.max_print_v[0])+1, 4), int(self.max_print_v[0])+1)

            for j_ in range(spc.size-1):
                for v_ in range(spc[j_], spc[j_+1]):
                    print ("\t{:>14s}".format("v = " + str(v_)), end='')
                print ()
                for jj_ in range(min(self.maxJ, self.args.Constants+1)):
                    if jj_ == 0:
                        print ("\t{:>6s}".format("Be"), end=' ')
                    elif jj_ == 1:
                        print ("\t{:>6s}".format("De"), end=' ')
                    elif jj_ == 2:
                        print ("\t{:>6s}".format("Fe"), end=' ')
                    elif jj_ == 3:
                        print ("\t{:>6s}".format("He"), end=' ')
                    else:
                        print ("\t{:>6s}".format(str(jj_) + "e"), end=' ')

                    for v_ in range(spc[j_], spc[j_+1]):
                        print ("\t{:>13e}".format(self.rot_spec_values[jj_, v_]), end=' ')
                    print ()
                print ()
            
            self.BREAK()

        if self.args.J > 1:

            self.rov_spec_values, vjmat = spectra.Rovibrational(self.total_val,
                                                                min(self.args.Constants, int(self.trunc_arr[0]))+1,
                                                                min(self.args.Constants+1, self.maxJ)
                                                                )

            print ("\tVibrational Constants on the Full Surface\n")
            print ("\t{:>10s}{:>17s}{:>15s}".format("Constant", "cm^-1", "MHz"))

            for v_ in range(min(self.args.Constants, int(self.trunc_arr[0]))+1):
                if v_ == 0:
                    s = 'we'
                elif v_ == 1:
                    s = 'wexe'
                elif v_ == 2:
                    s = 'weye' 
                elif v_ == 3:
                    s = 'weze'
                else:
                    s = 'we' + str(v_) + 'e'
                print ("\t{:>6s}{:>21e}{:>15e}".format(s, self.rov_spec_values[v_], self.rov_spec_values[v_]*cm_mhz))
                

            print ("\n")
            print ("\tRotational Constants on the Full Surface\n")
            print ("\t{:>10s}{:>17s}{:>15s}".format("Constant", "cm^-1", "MHz"))

            for j_ in range(min(self.args.Constants+1, self.maxJ)):
                if j_ == 0:
                    s = 'Be'
                elif j_ == 1:
                    s = 'De'
                elif j_ == 2:
                    s = 'Fe'
                elif j_ == 3:
                    s = 'He'
                else:
                    s = str(j_) + 'e'
                print ("\t{:>6s}{:>21e}{:>15e}".format(s, self.rov_spec_values[j_+v_+1], self.rov_spec_values[j_+v_+1]*cm_mhz))

            print ("\n")
            print ("\tRovibrational Coupling Constants\n")
            print ("\t{:>10s}{:>12s}{:>17s}{:>15s}".format("v", "J", "cm^-1", "MHz"))

            num_rov = self.rov_spec_values.size - j_ - v_ - 2

            for p in range(num_rov-2):
                n = j_+v_+p+2
                pv = vjmat[n,0]
                pj = vjmat[n,1]

                print ("\t{:>10d}{:>12d}{:>17e}{:>15e}".format(pv, 
                                                               pj, 
                                                               self.rov_spec_values[n], 
                                                               self.rov_spec_values[n] * cm_mhz))

            self.BREAK()


    def Dunham(self):
        '''Function used to interpolate the energy curve using a Dunham-type polynomial'''
        dunham = Spectra()
        self.dunham_Y, self.dunham_coef = dunham.Dunham(self.Data[0],
                                                        self.Data[1],
                                                        self.rEq,
                                                        self.eEq,
                                                        self.reduced_mass,
                                                        self.wEq,
                                                        self.bEq
                                                        )

        self.Y_id_one = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0 ,1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        self.Y_id_two = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]
        self.Y_id_thr = ['-',    'Be',    'De',   'He', 'Fe',
                        'we',   'alpha', 'beta', '-',   '-',
                        'wexe', 'gamma',  '-',   '-',   '-',
                        'weye', '-',      '-',   '-',   '-',
                        'weze', '-',      '-',   '-',   '-']


        print ("\tDunham Polynomial Fit Coefficients\n")
        strings = ['Parameter', 'cm^-1', 'MHz']
        print ('\t{:>0s}{:>18s}{:>23s}'.format(*strings))
        
        for c, coef in enumerate(self.dunham_coef):
            print ("\t{:>9s}{:>18f}{:>23f}".format('a' + str(c), coef, coef*cm_mhz))

        
        print ("\n")
        print ("\tDunham Y-Parameters / Spectroscopic Equivalents\n")
        strings = ['Parameter', 'Spectroscopic Constant', 'cm^-1', 'MHz']
        print ('\t{:>0s}{:>25s}{:>15s}{:>20}'.format(*strings))

        for p, param in enumerate(self.dunham_Y):
            dun_id = str(self.Y_id_one[p]) + str(self.Y_id_two[p])
            print("\t{:>9s}{:>14s}{:>26e}{:>20e}".format('Y' + dun_id, self.Y_id_thr[p], param, param*cm_mhz))

        self.BREAK()

    
    def End(self, start):
        minutes, seconds = divmod(time.time() - start, 60)
        print ("\tEnd of Program - ", "%d minutes, %d seconds" %(minutes, seconds), "\n")



def Main():
    '''Main function of the VibHam program'''
    start = time.time()

    args = Input()

    if args.i or args.Interactive:
        gui = GUI.main()
        sys.exit()

    VibHam = RunVibHam(args)

    VibHam.LoadData()
    VibHam.ConvertData()
    VibHam.AtomicInformation()
    VibHam.Interpolate()

    if not args.LoadData:
        VibHam.GenerateMatrices()
    else:
        VibHam.LoadMatrices()

    VibHam.TurningPoints()
    VibHam.PrintEigen()
    VibHam.DissociationEnergy()
    VibHam.Excitations()
    VibHam.Constants()
    VibHam.Dunham()
    VibHam.End(start)

if __name__ == "__main__":

    Main()

