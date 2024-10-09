
## About MLWC


[![GitHub release](https://img.shields.io/github/release/toamano/mlwc.svg?maxAge=86400)](https://github.com/toamano/mlwc/releases)
[![offline packages](https://img.shields.io/github/downloads/toamano/mlwc/total?label=offline%20packages)](https://github.com/toamano/mlwc/releases)
[![conda-forge](https://img.shields.io/conda/dn/conda-forge/mlwc?color=red&label=conda-forge&logo=conda-forge)](https://anaconda.org/conda-forge/mlwc)
[![pip install](https://img.shields.io/pypi/dm/mlwc?label=pip%20install)](https://pypi.org/project/mlwc)
[![Documentation Status](https://toamano.github.io/MLWC/badge)](https://toamano.github.io/MLWC/)

MLWC is a package written in Python/C++, designed to calculate the dielectric properties of various materials combined with molecular dynamics. This package construct deep learning models using the Wannier centers calculated from DFT as training data to predict the dipole moments of the system with high accuracy and efficiency. 

For more information, please check the [documentation](https://toamano.github.io/MLWC/).

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

For C++ interface, we support `CMake`. Please read the [online documentation](https://toamano.github.io/MLWC/) for details.


## Usage

For simple instruction and sample input files, see [`examples`](`examples`) directory. Also, following commands output sample input files for each command.

```bash
CPtrain.py sample
CPmake.py sample
```

For detailed explanations, please explore [the website](https://toamano.github.io/MLWC/).


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


## References

For detailed explanation of theory and implementation, please see the following publication

- [https://arxiv.org/abs/2407.08390](https://arxiv.org/abs/2407.08390)


## Future issues

- Interface with `VASP`, `Wannier90`.
- `LAMMPS` integration for C++ interface.


## License and credits

The project MLWC is licensed under [GNU LGPLv3.0](./LICENSE). If you use this code in any future publication, please cite the following publication:

- https://doi.org/10.48550/arXiv.2407.08390


## Authors

- Tomohito Amano (The University of Tokyo)
- Tamio Yamazaki (JSR-UTokyo Collaboration Hub, CURIE, JSR Corporation)
