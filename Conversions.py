################################################
# This file provides a central location for:
#       Physical Constants
#       Energy Unit Conversions
#       Mass Unit Conversions
#       Distance Unit Conversions
#
#
#################################################



######################
# Physical Constants #
######################

h       = 6.62607015E-34              # Planck Constant (J s)
h_bar   = 1.054571817E-34             # Reduced Planck Constant (J s)

kb      = 1.38064852E-23              # Boltzmann Constant (m^2 kg s^-2 K^-1)

c       = 2.99792458E8                # Speed of Light (m/s)
c_cm    = 2.99792458e10               # Speed of Light (cm/s)

eps_0   = 8.85418781762039E-12        # Permittivity of Free Space (C^2  N^-1  m^-2)

e       = 1.602176634E-19             # Electron Charge (C)
m_e     = 9.10938356e-31              # Electron Rest Mass (kg)

a_0     = 5.29177E-11                 # Bohr Radius (m)

N       = 6.0221409E23                # Avogadros Number 

###########################
# Energy Unit Conversions #
###########################

hart_J    = 4.35974E-18     # Hartree to Joules
hart_eV   = 27.2114         # Hartree to eV
hart_cm   = 219474.6        # Hartree to wavenumbers
hart_kcal = 627.509         # Hartree to Kcal/Mol

kj_J      = 1000/N          # kj/mol to Joules
kcal_J    = 4.184*1000/N    # kcal/mol to Joules
eV_J      = 1.60218E-19     # eV to Joules
cm_J      = 1.98630E-23     # Wavenumbers to Joules

J_hart    = 1. / hart_J     # Joules to Hartree
J_cm      = 5.03445E22      # Joules to wavenumbers
J_eV      = 6.2421509E18    # Joules to eV

kcal_kj   = 4.184           # kJ to kcal


cm_mhz = c/1000.            # Wavenumbers to MegaHertz

##########################
# Mass Units Conversions #
##########################

amu_kg  = 1.660538782E-27   # Atomic Mass Units to Kg

##############################
# Distance Units Conversions #
##############################

ang_m    = 1E-10             # Angstrom to meters
m_ang    = 1/ang_m           # Meters to angstrom
bohr_m   = 5.2917E-11        # Bohr to meters
m_bohr   = 1/bohr_m          # Meters to bohr
bohr_ang = 0.529177          # Bohr to angstrom
ang_bohr = 1/bohr_ang        # Angstrom to bohr


#############################
# Dipole Moment Conversions #
#############################

au_D = 2.541                # Atomic units to Debye
D_au = 1. / au_D            # Debye to atomic units
D_CM = 3.33564E-30

