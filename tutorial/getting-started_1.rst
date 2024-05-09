=====================================================
Getting-started tutorial No. 1: Isolated methanol
=====================================================

In this tutorial, we start from descriptor files to train ML dipole models of isolated methanol. 


Required DFT/MD data for calculations
========================================

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