=====================================================
Getting-started tutorial No. 2: Isolated methanol
=====================================================

In this tutorial, we give an comprehensive tutorial to prepare training data and train models, taking isolated methanol as an example.


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

 Postprocess data
----------------------------------------

``IONS+CENTERS.xyz`` does not include 



To train ML models for dipole moment, we only need two files:

* atomic coordinates with Wannier centers
* molecular structure

The first file is assumed to be the ``extended xyz`` format via ``ase`` package, which also contains the supercell information. The second file should be a standard ``mol`` file to be processed using ``rdkit``. We prepared a simple example using the isolated methanol system for this tutorial. Necessary files can be downloaded as

.. code-block:: bash

    download files

The order of atoms should satisfy three things

* The atomic order must be molecule-by-molecule.
* The atomic order in each molecule should be the same as the `*.mol`file. 
* The WCs should come last.

If you see the first 14 lines of `methanol.xyz`, you can find C, four H,O and eight X, where `X` means the Wannier centers (WC). 

They are visualized using `nglview` package via jupyter notebook as follows. 

.. code-block:: python

		import nglview as nv
		import ase.io

		aseatoms = ase.io.read("mol_wan.xyz",index=":")

		w = nv.show_asetraj(aseatmoms,gui=True)
		w.clear_representations()
		w.add_label(radius=0.2,color="black",label_type="atom")
		w.add_ball_and_stick("_He",color="green",radius=0.004,aspectRatio=50)
		w.add_ball_and_stick("_Ne",color="cyan",radius=0.004,aspectRatio=50)
		w.add_ball_and_stick("_Ar",color="green",radius=0.004,aspectRatio=50)
		#w.add_ball_and_stick("_Li",color="cyan",radius=0.1)
		#w.add_ball_and_stick("_Be",color="blue",radius=0.1)
		w.add_ball_and_stick("_H")
		w.add_ball_and_stick("_C")
		w.add_ball_and_stick("_O")
		w.add_ball_and_stick("_N")

		#w.clear_representations()
		#w.add_label(radius=1,color="black",label_type="atom")
		#view.add_representation("ball+stick")
		#w.add_representation("ball+stick",selection=[i for i in range(0,n_atoms)],opacity=1.0)
		#w.add_representation("ball+stick",selection=[i for i in range(n_atoms,total_atoms)],opacity=1,aspectRatio=2)
		w.add_unitcell()
		w.update_unitcell()
		w


Next, we dig into the `*.mol` file, which contains molecular structures including atomic and bonding information. 

.. code-block:: bash

    6  5  0  0  0  0  0  0  0  0999 V2000
        0.9400    0.0200   -0.0900 C   0  0  0  0  0  0  0  0  0  0  0  0
        0.4700    0.2700   -1.4000 O   0  0  0  0  0  0  0  0  0  0  0  0
        0.5800   -0.9500    0.2400 H   0  0  0  0  0  0  0  0  0  0  0  0
        0.5700    0.8000    0.5800 H   0  0  0  0  0  0  0  0  0  0  0  0
        2.0400    0.0200   -0.0900 H   0  0  0  0  0  0  0  0  0  0  0  0
        0.8100    1.1400   -1.6700 H   0  0  0  0  0  0  0  0  0  0  0  0
    1  5  1  0  0  0  0
    1  3  1  0  0  0  0
    1  4  1  0  0  0  0
    2  1  1  0  0  0  0
    6  2  1  0  0  0  0
    M  END

The second to seventh lines are called atom block, which contain atomic coordinates and species in a single molecule. We only use atomic species for training. The following data is called atom block, representing bonding information. For example, 

.. code-block:: bash

    1  5  1  0  0  0  0

mean the first and fifth atom (C and H) have a chemical bond. The `*.mol` format is a standard format for molecular structures, and you can beasily find information on it.


Model training
================

Prepare input script
----------------------

To train models, we implemented ``CPtrain.py`` command written in pytorch. The command require ``yaml`` format file to specify parameters. Here is the example:

.. code-block:: yaml

    model:
    modelname: test  # specify name
    nfeature:  288   # length of descriptor
    M:         20    # M  (embedding matrix size)
    Mb:        6     # Mb (embedding matrix size, smaller than M)

    learning_rate:
    type: fix

    loss:
    type: mse        # mean square error

    data:
    type: descriptor # or xyz
    file:
    - "descs_bulk/cc"

    traininig:
    device:     cpu # Torchのdevice
    batch_size: 32  # batch size for training 
    validation_vatch_size: 32 # batch size for validation
    max_epochs: 40
    learnint_rate: 1e-2 # starting learning rate
    n_train: 2100000    # the number of training data
    n_val:     10000    # the number of validation data
    modeldir:  model_test # directory to save models
    restart:   False    # If restart training 

Parameters written above are basically necessary values (not optional). The input file consists of four parts:


+----------------+------------------------+
|  part name     | explanation            |            
+================+========================+
| model          |  ML model parameters   | 
+----------------+------------------------+
| learning_rate  | learning rate          | 
+----------------+------------------------+
| loss           | loss function          |
+----------------+------------------------+
| data           | training data          | 
+----------------+------------------------+
| training       | training parameters    |
+----------------+------------------------+

As Basic explanations are given above, we only add some important notes.

* Model parameters (nfeature, M, Mb) are basically enough for simple gas/liquid molecules
* Currently, we only support fixed learning rate. 
* Currently, loss function is Mean Squared Error (MSE).
* Training data should be :code:`descriptor` or :code:`xyz`.
* If training data type is :code:`descriptor`, the descripter file name should be :code:`*_descs.npy`, and the true file name should be :code:`*_true.npy`.



Train a model
----------------------

After the training script is prepared, we can start the training by simply running

.. code-block:: bash

    CPtrain.py train -i input.yaml


Test a model
----------------------

We can check the quality of the trained model 


Calculate dipoles
----------------------