
## About OUR-PACKAGE

OUR-PACKAGE is a package written in Python/C++, designed to calculate the dielectric properties of various materials combined with molecular dynamics. This package construct deep learning models using the Wannier centers calculated by DFT as training data to predict the dipole moments of the system with high accuracy and efficiency. 

For more information, check the [documentation]().

## Features

- **interfaced with DFT packages**, including CPMD and Quantum Espresso.
- **implements the chemical bond-based approach**, enabling high accuracy on finite and extended, small and large molecular systems.
- **implements openMP and GPU supports**, making it highly efficient for high-performance parallel and distributed computing.
- **scripted using Pytorch**, allowing for fast training with python and prediction with C++.


## Command lines

- `CPml.py`: Main command to train&test models. 
- `dieltools`: C++ interface for predicting bond dipoles.
- `CPextract.py`: To retrieving data from DFT codes and `dieltools`.
- `CPmake.py`: To make input files for DFT codes.


## Documentation

Please visit the following webpage for installation and usages.


## Installation

Python commands are easily installed via `pip` as 
```bash
git clone https://github.com/dirac6582/dieltools
cd dieltools
pip install .
```

For C++ interface, we support `CMAKE`. Please read the [online documentation]() for details.


## Usage

For simple instruction and sample input files, see [`examples`](`examples`) directory. Also, following commands output sample input files for each command.

```bash
CPtrain.py sample
CPmake.py sample
```

For detailed explanations, please explore [the website]().


## Code structure

The repository is organized as follows:

- `docs`: documentations.
- `examples`: examples.
- `examples/tutorial`: examples for tutorials explained in documentations.
- `src/cpp`: source code of C++ interface.
- `src/cmdline`: source code of python commandline.
- `src/cpmd`: source code for data processing.
- `src/ml`:   source code for deep neural network.
- `src/diel`: source code for calculating dielectric property.
- `script`: additional scripts for developers.
- `test`: additional files for developers.


## Future issues

- Interface with `VASP`, `Wannier90`.
- `LAMMPS` integration for C++ interface.


## License and credits

The project OUR-PACKAGE is licensed under [GNU LGPLv3.0](./LICENSE).
If you use this code in any future publications, please cite the following publications for general purpose:


In addition, please follow [the bib file](CITATIONS.bib) to cite the methods you used.

## Authors

- Tomohito Amano (The University of Tokyo)
- Tamio Yamazaki (JSR-UTokyo Collaboration Hub, CURIE, JSR Corporation)
