#################################################################################
#   This is the main file for the VibHam program. This file performs many tasks:
#       1 - Reads in data file
#       2 - Converts necessary arrays to SI units
#       3 - Calls PolyFit to perform a polynomial fit of the curve
#           Returns the equilibrium bond distance and minimum energy value
#       4 - Calculates the reduced mass of the diatomic molecule
#       5 - Calculates the rotational constant
#       6 - Calls Hamil to popoulate the harmonic, anharmonic, and centrifugal 
#           potential barrier Hamiltonian Matrices and visualize the wave 
#           functions 
#       7 - Calls TDM To calculate all (ro)vibrational excitations
#       8 - Calls TDM to caculate the spectroscopic constants 
#       9 - Calls Dunham to calculate to refit the curve to a Dunham
#            Polynomial
##################################################################################


def Main():

    start = time.time()     # Value of inital time

    args = Input()          # Input Parser

    Dip_Check = False       # Check to perform dipole moment calculations

    ########################################################################
    # Load the Datasets and Initialize arrays                               
    #                                                                       
    # Re    - Bond distance data for electronic energy                      
    #       - May be given in Angstrom, bohr, or meters                     
    #                                                                       
    # Rd    - Bond distance data for dipole moment                          
    #       - May be given in Angstrom, bohr, or meters                     
    #                                                                       
    # E     - Electronic Energy Data                                        
    #       - May be given in Hartree, kcal/mol, kj/mol, wavenumbers, eV    
    #                                                                       
    # D     - Dipole Moment Data                                            
    #       - May be given in Debye or au                                   
    ########################################################################

    if not args.Data:
        print ("MUST GIVE FILE TO READ IN")
        exit()

    if not os.path.exists(args.Data):
        print ("FILE", args.Data, "NOT FOUND")
        exit()

    try:
        Data = np.loadtxt(args.Data).transpose()
    except:
        print ("FILE", args.Data, "NOT ABLE TO BE READ BY NUMPY")
        exit()

    Data = Data[:,np.argsort(Data[0,:])]        # Sort datafile matrix by increasing bond distance values

    Re = Data[0].copy()     # Array of bond distance information to be used for vibrational Hamiltonian
    Rd = Data[0].copy()     # Array f bond distance information to be used for dipole moment
    E  = Data[1].copy()     # Array of energy values

    if Data.shape[0] == 2:
        D  = np.zeros((1))      # Empty array for dipole moment function
        Dip_Check = False       # Boolean check for no dipole information

    elif Data.shape[0] == 3:
        D  = Data[2].copy()     # Array for dipole moment function
        Dip_Check = True        # Boolean check for dipole information

    print ()
    print ("\tData file ", args.Data, " successfully loaded")
    print ()

    #############################################################
    # If no dipole data is give, then skip all TDM calculations 
    #############################################################

    if Dip_Check == False:
        print ("\tNo Dipole Moment Information Provided")
        print ("\tSkipping all TDM calculations\n")

    #####################################
    # Converts:                         
    #   Bond Distance to meters         
    #   Electronic Energy to Joules     
    #   Dipole Moment to atomic units   
    #####################################

    print ("\tConverting bond distance to meters")
    print ("\tConverting electronic energy to joules")

    if args.R_unit == 'ang':
        Re *= ang_m
    elif args.R_unit == 'bohr':
        Re *= bohr_m
    elif args.R_unit == 'm':
        Re

    if args.E_unit == 'hartree':
        E       *= hart_J
    elif args.E_unit == 'kj/mol':
        E       *= kj_J
    elif args.E_unit == 'kcal/mol':
        E       *= kcal_J
    elif args.E_unit == 'ev':
        E       *= eV_J
    elif args.E_unit == 'cm':
        E       *= cm_J
    elif args.E_unit == 'j':
        E

    if Dip_Check == True:
        print ("\tConverting dipole moments to atomic units")
        if args.R_unit == 'ang':
            Rd *= ang_m
        elif args.R_unit == 'bohr':
            Rd *= bohr_m
        elif args.R_unit == 'm':
            Rd
        
        if args.Dip_unit == 'debye':
            D *= D_au
        elif args.Dip_unit == 'au':
            D

    ############################################
    # Standard polynomial Fit for the EST Data 
    #   Variables:
    #       Re - array of bond distances
    #       E  - array of energy values
    #   Returns
    #       R_eq  - equilibrium bond distances
    #       R_min - equilibrium energy
    ############################################

    BREAK()
    R_eq, E_min = PF(Re, E, args)
    
    ##############################
    # Calculate the reduced mass 
    ##############################

    if not args.Atoms and not args.Masses:
        print ("\tMUST PROVIDE EITHER ATOMS OR MASSES\n")
        print ("\tQuitting Program\n")
        exit()

    if args.Atoms:
        if not args.Isotopes:
            Atom1 = Atoms(args.Atoms[0], 0)     # Assign mass to atom 1 given most stable isotope
            Atom2 = Atoms(args.Atoms[1], 0)     # Assign mass to atom 2 given most stable isotope
        else:
            try:                                # Try to assign given isotoped to Atom 1
                Atom1 = Atoms(args.Atoms[0], args.Isotopes[0])
            except:
                print ("\tIsotope ", args.Atoms[0], args.Isotopes[0], "Not Found\n")
                print ("\tQuitting Program\n")
                exit()

            try:                                # Try to assign given isotoped to Atom 2
                Atom2 = Atoms(args.Atoms[1], args.Isotopes[1])
            except:
                print ("\tIsotope ", args.Atoms[1], args.Isotopes[1], "Not Found\n")
                print ("\tQuitting Program\n")
                exit()

    elif args.Masses:               # Assign masses using user given values
        Atom1 = args.Masses[0]
        Atom2 = args.Masses[1]

    r_mass = ((Atom1 * Atom2) / (Atom1 + Atom2))    # Reduced mass in amu

    print ("\tReduced Mass")
    print ("\t\t{:>.9e} amu".format(r_mass))
    print ("\t\t{:>.9e} kg".format(r_mass*amu_kg))
    print ()

    ####################################
    # Calculate the Rotational Constant 
    ####################################

    Be = h / (8 * np.pi**2 * c_cm * (r_mass*amu_kg) *  R_eq**2)
    
    print ("\tRotational Constant")
    print ("\t\t{:>.9e} cm^-1".format(Be))
    print ("\t\t{:>.9e} MHz".format(Be*cm_mhz))
    print ()

    #####################################
    # Vibrational Hamiltonian Functions 
    #   Variables:
    #       Re     - array of bond distance values
    #       E      - array of energy values
    #       R_eq   - equilibrium bond distance
    #       E_min  - equilibrium bond energy
    #       r_mass - reduced mass
    #       args   - input arguments
    #   Returns:
    #       H        - Hamiltonian Tensor
    #       beta     - beta value for diatomic
    #       MaxLevel - Maximum converged energy level
    #       Vals     - Eigenvalues
    #       Vecs     - Eigenvectors
    #       nu       - vibrational constant in s^-1
    #####################################

    BREAK()
    H, beta, MaxLevel, Vals, Vecs, nu = Hamil(Re, E, R_eq, E_min, r_mass, args)

    ########################################################################
    # Calculate Transition Dipole Moment Matrix if dipole moment is provided 
    #   Variables:
    #       Rd-R_eq         - array of bond distances shifted by R_eq
    #       D               - array of dipole moment values
    #       R_eq            - equilibrium bond distance
    #       beta            - beta value for diatomic
    #       Vals.shape[1]   - Size of dipole moment matrix to construct
    #       args            - input arguments
    ########################################################################

    if Dip_Check == True:
        BREAK()
        tdm = TDM(Rd-R_eq, D, R_eq, beta, Vals.shape[1], args)
    else:
        tdm = np.ones((Vals.shape[0], Vals.shape[0]))

    ########################################################################
    # Calculate all excitations using the converged energy levels 
    #   Variables:
    #       Rd-R_eq   - array of bond distances shifted by R_eq
    #       D         - array of dipole moment values
    #       R_eq      - equilibrium bond distance
    #       beta      - beta value for diatomic
    #       H         - Hamiltonian Matrices
    #       Vals      - Eigenvalues
    #       Vecs      - Eigenvectors
    #       MaxLevel  - Maximum converged energy level
    #       Dip_Check - Boolean if Dipole moment information provided
    #       args      - input arguments
    ########################################################################
    
    BREAK()
    Excitations(Vals, Vecs, MaxLevel+1, Dip_Check, tdm, args)

    #########################################################
    # Calculate Spectroscopic Constants 
    #   Variables:
    #       R_eq/ang_m  - Equilibrium bond distance in angstrom
    #       Vals        - Eigenvalues
    #       args        - input arguments
    ##########################################################

    BREAK()
    Constants(Vals[:,:MaxLevel+1], args)

    ##########################################
    # Calculate Dunham Polynomial Fit 
    #   Variables:
    #       Re - array of bond distances
    #       R_eq - equilibrium bond distance
    #       E - array of energy values
    #       E_min - equilibrium bond energy
    #       nu - vibrational constant in s^-1
    #       r_mass - reduced mass in kg
    ###########################################

    BREAK()
    Dunham(Re, R_eq, E, E_min, nu, r_mass*amu_kg)

    BREAK()

    minutes, seconds = divmod(time.time() - start, 60)
    print ("\tEnd of Program - ", "%d minutes, %d seconds" %(minutes, seconds), "\n")

    exit()


if __name__ == "__main__":

    try:                            # Try to import all necessary python libraries
        import numpy as np
        import math
        import os
        import matplotlib.pyplot as plt
        from scipy.integrate import quad 
        from scipy.special import hermite
        from scipy.optimize import curve_fit
        from math import factorial as fac
        import argparse
        import warnings
        import time
    except:
        print ("One or more of the following python libraries is not currently installed:")
        print ("\tnumpy\n\tmath\n\tos\n\tmatplotlib\n\tscipy\n\targparse\n\twarning\n\ttime")
        exit()
    
    try:                           # Try to load all necessary python files
        from Conversions import *
        from Input       import Input
        from Atoms       import Atoms   
        from PolyFit     import PolyFit as PF
        from Hamil       import Hamil
        from Spec        import TDM, Excitations, Constants
        from Dunham      import Dunham
        
    except:
        print ("One or more of the following files is not currently available:")
        print ("\tConversions.py\n\tInput.py\n\tAtoms.py\n\tPolyFit\n\tHamil\n\tSpec\n\tDunham")
        exit()

    Main()

