# VibHam
VibHam is an open source program to probe the rovibrational spectroscopy of diatomic molecules. VibHam is developed by Gavin McCarver with advising by Robert J. Hinde at the University of Tennessee, Knoxville in the chemistry department. The program uses a power series expansion to move beyond the harmonic oscillator approximation combined with vibrational Hamiltonian matrices to approximate the anharmonic wave functions. The VibHam program can examine atoms up to element 118 (Og) using the isotopic masses available through [NIST](https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl).

## Installation

Clone VibHam source from github.
```bash
git clone https://github.com/gmccarve/VibHam.git

These python packages are required for the either the command-line or GUI usage of Vibham:
- numpy
- pandas
- matplotlib
- scipy
- scikit-learn (sklearn)
- PyQt5

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

Open the GUI for VibHam
```bash
python3 /location/of/VibHam/VibHam.py -i
```
or
```bash
 python3 /location/of/VibHam/VibHam.py -Interactive
```
or
```bash
python /location/of/VibHam/GUI.py
```

Please see the input_parameters.txt file for more information regarding the choice of inputs. To include any given input, include " -$INPUT $OPTIONS" after you call the VibHam program and change $INPUT to be any of the available input choices and $OPTIONS to your chosen options.


Please note that VibHam will run with predefined input parameters but the user must define the atoms or masses of the diatomic being examined and the data file.

## Examples

There are two examples in the Examples folder. These include the potential energy curve of CO and the potential energy curve and dipole moment curve of HF.

To run either of these examples, change the permissions for the "run_example" bash script to be executable:
```bash
chmod +x run_example
```
Then execute the "run_example" bash script:
```bash
./run_example
```

## Citation

VibHam is currently being prepared for submission.



