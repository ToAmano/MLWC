=====================================================
Getting-started tutorial No. 2: Isolated methanol
=====================================================

In this tutorial, we give an comprehensive tutorial to prepare training data and train models, taking isolated methanol as an example.

 Requirements
========================================

To perform the tutorial, you also need to insall ``CPMD`` package for DFT/MD calculations. See https://github.com/CPMD-code.

 Tutorial data
========================================

The reference files of this tutorial are given in ``examples/tutorial/2_liquidmethanol/`` directory. 


 Prepare training data
========================================

The acquisition of the training data requires two steps: generation of the structure and calculation of the Wannier centers. In the case of liquid structures, the acquisition of the structure is done with classical MD and only the Wannier center calculation is done with DFT due to computational cost considerations, while in the case of gaseous structures, the structure acquisition and Wannier center calculation can be done simultaneously with ab initio MD due to the lower computational cost. We will test the sequence using the CPMD package.

 Convert smiles to xyz
----------------------------------------

To begin with, we build an initial structure for CPMD. Let us prepare a csv file containing smiles.

.. code-block:: bash

    $cat methanol.csv
    Smiles,Name,density
    CO,METHANOL,0.791

The following simple python code ``csv2xyz.py`` will generate ``methaol.xyz`` (and ``methanol.mol`` for later usage).

.. code-block:: bash

    $cat csv2xyz.py
    import shutil
    import os
    import pandas as pd
    import rdkit.Chem
    import rdkit.Chem.AllChem
    # read csv
    input_file:str = "methanol.csv" # read csv
    poly = pd.read_csv(input_file)
    print(" --------- ")
    print(poly)
    print(" --------- ")
    # read smiles
    smiles:str = poly["Smiles"].to_list()[0]
    molname:str = poly["Name"].to_list()[0]
    # build molecule
    mol=rdkit.Chem.MolFromSmiles(smiles)
    print(mol)
    molH = rdkit.Chem.AddHs(mol) # add hydrogen
    print(molH)
    print(rdkit.Chem.MolToMolBlock(molH, includeStereo=False))

    rdkit.Chem.AllChem.EmbedMolecule(molH, rdkit.Chem.AllChem.ETKDGv3())
    print(molH)
    print(rdkit.Chem.MolToMolBlock(molH, includeStereo=False))
    rdkit.Chem.MolToMolFile(molH,'methanol.mol')
    rdkit.Chem.MolToXYZFile(molH,'methanol.xyz')
    # xyz = rdkit.Chem.MolToXYZBlock(molH)


 Prepare input for CPMD
----------------------------------------

``CPmake.py`` will yield input files for ``CPMD`` from ``methanol.xyz`` as follows.

.. code-block:: bash

    $CPmake.py cpmd workflow --i methanol.xyz -n 40000 -t 10 
    *****************************************************************
                            CPmake.py
                        Version. 0.0.1
    *****************************************************************

    ---------
    input geometry file ::  methanol.xyz
    output georelax calculation        :: georelax.inp
    output bomdrelax calculation       :: bomdrelax.inp
    output bomd restart+wf calculation :: bomd-wan-restart.inp
    output bomd restart+wf accumulator calculation :: bomd-wan-restart2.inp
    # of steps for restart      ::  40000
    timestep [a.u.] for restart ::  10
    atomic arrangement type     ::  default


``-n`` and ``-t`` specify the number of steps and the time step (in a.u.) for MD, respectively.  Therefore, we will run 400,000 [a.u.] ~ 9.7 [ps] calculation.

Four input files are for 1: geometry optimization, 2: initial relaxation, and 3&4: production run. 

.. note::

   Generated inputs are just samples. You should tune parameters for serious calculations.


We slightly modify the inputs for later convenience. The line ``DIPOLE DYNAMICS WANNIER SAMPLE`` decides how often the structure will be calculated. Set it to ``100`` to reduce computational cost.

.. code-block:: bash

    DIPOLE DYNAMICS WANNIER SAMPLE
    100


Secondly, you should add the simulation cell to the inputs. 

.. code-block:: bash

    DIPOLE DYNAMICS WANNIER SAMPLE
    100


We create ``tmp/`` and ``pseudo/`` directories to stock outputs and pseudo potentials, respectively. You also have to prepare ``C_MT_GIA_BLYP``, ``O_MT_GIA_BLYP``, and ``H_MT_BLYP.psp`` from CPMD pseudo potential directories and store them in ``pseudo/`` directory.


 Run CPMD
----------------------------------------

We execute three runs: geometry optimization, initial relaxation, and production Wannier run. They will take a few hours depending on your machine. We strongly recommend you to use supercomputers. Please be patient.

.. code-block:: bash

    mpirun cpmd.x georelax.inp >> georelax.out
    mpirun cpmd.x bomd-relax.inp >> bomd-relax.out
    mpirun cpmd.x bomd-wan-restart.inp >> bomd-wan-restart.out

After the calculation, you will see ``IONS+CENTERS.xyz`` in the ``tmp/`` directory, which contains atomic and WC coordinates. 

 Postprocess CPMD data
----------------------------------------

``IONS+CENTERS.xyz`` does not include the lattice information, which we need to add manually. We can use ``CPextract.py`` to do this.


.. code-block:: bash

    $CPextract.py extract -i IONS+CENTERS.xyz -s bomd-wan-restart.out IONS+CENTERS_cell.xyz


``-s`` specifies the stdout file of the CPMD calculation. The output file ``IONS+CENTERS_cell.xyz`` is ``extended xyz`` format, and can be processed by ``ase`` package.


 Train models
----------------------------------------

The previously prepared ``IONS+CENTERS_cell.xyz`` and ``methanol.mol`` are used for training ML models. As methanol has ``CH``, ``CO``, ``OH`` bonds and ``O`` lone pair, we have to train four independent ML models. The input file for ``CPtrain.py`` is given in ``yaml`` format. 
The input file for the CH bond is as follows.

.. code-block:: yaml

    model:
    modelname: model_ch  # specify name
    nfeature:  288       # length of descriptor
    M:         20        # M  (embedding matrix size)
    Mb:        6         # Mb (embedding matrix size, smaller than M)

    learning_rate:
    type: fix

    loss:
    type: mse        # mean square error

    data:
    type: xyz
    file: 
        - "IONS+CENTERS+cell_sorted_merge.xyz"
    itp_file: methanol.mol
    bond_type: CH # CH, CO, OH, O

    traininig:
    device:     cpu # Torch device (cpu/mps/cuda)
    batch_size: 32  # batch size for training 
    validation_vatch_size: 32 # batch size for validation
    max_epochs: 50
    learnint_rate: 1e-2 # starting learning rate
    n_train:   9000    # the number of training data
    n_val:     1000    # the number of validation data
    modeldir:  model_ch # directory to save models
    restart:   False    # If restart training 


For gas systems, we can reduce the model size without losing accuracy. 

We can train the CH bond model 

.. code-block:: bash

    $CPtrain.py train -i input.yaml

After the training, RMSE should be about ``0.001[D]`` to ``0.01[D]`` for isolated systems.


Next, you can change ``modelname``, ``bond_type``, and ``modeldir`` to corresponding bonds, and re-run ``CPtrain.py`` to train other 4 models.



Test a model
----------------------

We can check the quality of the trained model as follows. 


Calculate molecular dipole moment
-----------------------------------

Finally, we will calculate the average molecular dipole moment of methanol. The experimental value is ``1.62[D]``.
For this purpose, we invoke C++ interface with the following input. The calculation of molecular dipole moments is done without specifying any flag. 

.. code-block:: yaml

    model:
    modelname: model_ch  # specify name
    nfeature:  288       # length of descriptor
    M:         20        # M  (embedding matrix size)
    Mb:        6         # Mb (embedding matrix size, smaller than M)

    learning_rate:
    type: fix

    loss:
    type: mse        # mean square error

    data:
    type: xyz
    file: 
        - "IONS+CENTERS+cell_sorted_merge.xyz"
    itp_file: methanol.mol
    bond_type: CH # CH, CO, OH, O

    traininig:
    device:     cpu # Torch device (cpu/mps/cuda)
    batch_size: 32  # batch size for training 
    validation_vatch_size: 32 # batch size for validation
    max_epochs: 50
    learnint_rate: 1e-2 # starting learning rate
    n_train:   9000    # the number of training data
    n_val:     1000    # the number of validation data
    modeldir:  model_ch # directory to save models
    restart:   False    # If restart training 

We perform the calculation 

.. code-block:: bash

    dieltools.x 

The corresponding output file is ``DIELCONST``, which contains the mean molecular dipole moment, and ``molecule_dipole.txt``, which involve all the molecular dipole moments along the MD trajectory.
We can see the mean absolute dipole moment as 

.. code-block:: bash

    $cat DIELCONST

and we confirmed that the simulated value well agrees with the experimental one. 

