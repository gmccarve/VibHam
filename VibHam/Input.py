import argparse
import numpy as np

#############################################################################
#   This file contains a function which is the input parser for Vibham
#
#   To read all of the options and information for each input variable, type:
#       "python3 VibHam.py -h"
#
#   To read adjust an input variable, type:
#       "python3 VibHam -$VAR -$OPTION"
#       
#   where $VAR is the input variable you wish to change to $OPTION
############################################################################

def Input():
    parser = argparse.ArgumentParser(allow_abbrev=False,
                                     description='Program to examine the (Ro)Vibrational nature of a diatomic'
                                     )

    parser.version='1.0'

    parser.add_argument('-Data',
                        action='store',
                        help='Data File to be read in',
                        type=str,
                        )

    parser.add_argument('-Atoms',
                        action='store',
                        help='Atomic Species (2)',
                        type=str.upper,
                        nargs=2,
                        )

    parser.add_argument('-Masses',
                        action='store',
                        help='Atomic Masses (2)',
                        type=float,
                        nargs=2,
                        )

    parser.add_argument('-Isotopes',
                        action='store',
                        help='Isotopes to use (2)',
                        type=int,
                        nargs=2,
                        )

    parser.add_argument('-R_unit',
                        action='store',
                        help='Units for Bond Distance',
                        type=str.lower,
                        choices=['ang', 'bohr', 'm'],
                        default='ang'
                        )

    parser.add_argument('-E_unit',
                        action='store',
                        help='Units for Electronic Energy',
                        type=str.lower,
                        choices=['hartree', 'kj/mol', 'kcal/mol', 'ev', 'j', 'cm'],
                        default='hartree'
                        )

    parser.add_argument('-Dip_unit',
                        action='store',
                        help='Units for Dipole Moment',
                        type=str.lower,
                        choices=['debye', 'au'],
                        default='debye'
                        )

    parser.add_argument('-Energy_Fit',
                        action='store',
                        help='Degree of Polynomial Fit for Electronic Energy According to V(R-R_eq). Even Values Only',
                        type=int,
                        default=2,
                        choices=[2, 4, 6, 8, 10, 12, 14, 16]
                        )

    parser.add_argument('-Dipole_Fit',
                        action='store',
                        help='Degree of Standard Polynomial Fit for Dipole Moment',
                        type=int,
                        default=0,
                        choices=np.arange(0, 13)
                        )

    parser.add_argument('-EigVal',
                        action='store',
                        help='Convergence value for eigenvalues of Hamiltonian Matrix (cm^-1)',
                        type=float,
                        default=0.1
                        )

    parser.add_argument('-J',
                        action='store',
                        help='Maximum Value for Rotational Quantum number',
                        type=int,
                        default=0
                        )

    parser.add_argument('-Trap',
                        action='store',
                        help='Number of intervals for trapezoid rule for numerical integration',
                        type=int,
                        default=2000
                        )

    parser.add_argument('-v',
                        action='store',
                        help='Maximum Value for Vibrational Quantum Number',
                        type=int,
                        default=20
                        )

    parser.add_argument('-Print',
                        action='store',
                        help='Print level (1 - Converged Values, 2 - Converged Values + Vectors, 3 - All Values, 4 - All Values + Vecotrs)',
                        type=int,
                        choices=[1,2,3,4],
                        default=1
                        )
    
    parser.add_argument('-Charge',
                        action='store',
                        help='Charge of Molecule',
                        type=int,
                        default=0
                        )

    parser.add_argument('-InterPoints',
                        action='store',
                        help='Accuracy of polynomial fit',
                        type=int,
                        default=10000
                        )

    parser.add_argument('-Constants',
                        action='store',
                        help='Number of Spectroscopic Constants to Calculate',
                        type=int,
                        default=3
                        )

    parser.add_argument('-LoadData',
                        action='store_true'
                        )

    parser.add_argument('-Interactive',
                        action='store_true'
                        )

    parser.add_argument('-i',
                        action='store_true',
                        )

    parser.add_argument('-c',
                        action='store_true',
                        )

    parser.add_argument('-Method',
                        action='store',
                        help='Method to calculate Hamiltonian matrices',
                        type=str.lower,
                        choices=['fortran', 'python'],
                        default='fortran'
                        )





    return parser.parse_args()

