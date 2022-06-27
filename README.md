# VibHam
VibHam is an open source program to probe the rovibrational nature of diatomic molecules. VibHam is developed by Gavin McCarver with advising by Robert J. Hinde at the University of Tennessee, Knoxville in the chemistry department. The program uses a power series expansion to move beyond the harmonic oscillator approximation combined with vibrational Hamiltonian matrices to approximate the anharmonic wave functions. The VibHam program can examine atoms up to element 118 (Og) using the isotopic masses available through [NIST](https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl).

## Installation

Clone VibHam soruce from github.
```bash
git clone https://github.com/gmccarve/VibHam.git
```

## Usage

Use a terminal to call the program
```bash
python3 /location/of/VibHam/VibHam.py
```

Print the available input options without running VibHam
```bash
python3 /location/of/VibHam/VibHam.py -h
```

Please see the input_parameters.txt file for more information regarding the choice of inputs. To include any given input, include " -$INPUT $OPTIONS" after you call the VibHam program and change $INPUT to be any of the available input choices and $OPTIONS to your chosen options.

As an example, to run VibHam using a potential energy curve for H2 found in data.txt with rotational corrections up to J=3 for a 25x25 matrix, type:

```bash
python3 /location/of/VibHam/VibHam -Data data.txt -v 25 -J 3
```

Please note that VibHam will run with predefined input parameters but the user must define the atoms or masses of the diatomic being examined

## Examples

There are two examples in the Examples folder. These include the potential energy curve of CO and the potential energy curve and dipole moment curve of HF.

To run these examples, change the permissions for the "run_example" bash script to be executable:
```bash
chmod +x run_example
```
Then execute the "run_example" bash script:
```bash
./run_example
```
This will work for both CO and HF

## Citation

VibHam is currently being prepared for submission.


