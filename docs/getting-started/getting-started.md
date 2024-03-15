# Getting-started tutorial1: Isolated methanol

実際にモデルの訓練を行う．


## Required DFT/MD data for calculations

To train ML models for dipole moment, we only need two files:

- atomic coordinates with Wannier centers
- molecular structure

The first file is assumed to be the `extended xyz` format via `ase` package. The second file should be `.mol` file to be processed using `rdkit`. We prepared a simple example using the isolated methanol system for this tutorial. Necessary files can be downloaded as

```bash
download files
```

If you see the first 14 lines of `methanol.xyz`, you can find C,H,O and X, where `X` means the Wannier centers (WC). The alignment of atoms should be the same as the `*.mol`file.

```bash


```

They are visualized using `nglview` package via jupyter notebook as follows. 

```bash


```

Next, we dig into the `*.mol` file, which contains molecular structures including atomic and bonding information.

```bash

```


## Model training

