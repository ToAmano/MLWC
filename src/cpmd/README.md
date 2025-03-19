# src/cpmd/

This directory contains modules for handling CPMD (Car-Parrinello Molecular Dynamics) simulation data.

## Description

This directory provides a set of scripts for reading, processing, and analyzing data from CPMD simulations. These scripts are designed to work with various CPMD output files, such as trajectory files, wavefunction files, and other data files.

## Modules

- `read_traj_cpmd.py`: Provides classes and functions for reading trajectory data from CPMD output files.
- `read_wfc_cpmd.py`: Provides classes and functions for reading wavefunction data from CPMD output files.
- `converter_cpmd.py`: Provides scripts for converting CPMD output files to other formats.
- `descripter.py`: Provides scripts for calculating descriptors from CPMD simulation data.
- `gromacs_wrap.py`: Provides scripts for interfacing with Gromacs.
- `pbc/`: Contains scripts related to periodic boundary conditions.

## Subdirectories

- `analysis_trajectory/`: Contains scripts for analyzing CPMD trajectories.
- `asign_wcs/`: Contains scripts for assigning Wannier centers.
- `bondcenter/`: Contains scripts related to bond centers.
- `distance/`: Contains scripts related to distance calculations.
- `pbc/`: Contains scripts related to periodic boundary conditions.
