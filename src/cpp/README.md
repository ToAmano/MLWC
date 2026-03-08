# src/cpp/

This directory contains C++ source code for various functionalities, including handling atom structures, calculating descriptors, and interfacing with Python's NumPy library.

## Description

This directory provides a set of C++ classes and functions for performing various tasks related to atomistic simulations and machine learning. These tools are designed to be used as a backend for Python scripts, providing efficient implementations for computationally intensive tasks.

## Modules

- `atoms.cpp`: Defines the `Atoms` class for representing atomic structures, including atomic numbers, positions, and unit cell information.
- `descriptor.cpp`: Provides functions for calculating descriptors from atomic structures.
- `dipole.cpp`: Provides functions for calculating dipole moments.
- `module_torch.cpp`: Provides functions for interfacing with PyTorch.
- `parse.cpp`: Provides functions for parsing input files.
- `predict.cpp`: Provides functions for making predictions using machine learning models.

## Subdirectories

- `atoms_io/`: Contains scripts related to reading and writing atom structures.
- `chemicalbond/`: Contains scripts related to chemical bonds.
- `include/`: Contains header files.
- `ml/`: Contains scripts related to machine learning.
- `postprocess/`: Contains scripts related to post-processing.
