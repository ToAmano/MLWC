###################################################################
 Treatment of chemical bonds
###################################################################


Overview
========

The chemical bonds are the central quantities in constructing the chemical-bond-based machine learning model for dipole moments.
In this code, all the information of a molecule is stored in ``MolecularInfo`` class, where we use ``RDKit`` to extract comprehensive molecular information from ``.mol`` files.

Features
========

* **Basic Molecular Information**: Extract atom counts, atom types, and bond information
* **Bond Analysis**: Detailed analysis of different bond types including single, double, triple, and aromatic bonds
* **Atomic Indexing**: Group atoms by element type for easy access
* **Special Bond Detection**: Identify COC (Carbon-Oxygen-Carbon) and COH (Carbon-Oxygen-Hydrogen) bonds
* **Representative Atom Detection**: Find the atom closest to the center of mass
* **Comprehensive Logging**: Detailed logging of analysis results

MolecularInfo Object
====================

The ``MolecularInfo`` object is defined in ``mlwc.bond.atomtype`` and contains the following attributes:

.. code-block:: python

   class MolecularInfo:
       num_atoms_per_mol: int                 # Total number of atoms
       atom_list: List[str]                   # List of atom symbols
       bonds_list: List[List[int]]            # Bond connections as [start, end] pairs
       num_bonds: int                         # Total number of bonds
       bonds_type: List[int]                  # Bond types (1=single, 2=double, 3=triple, 10=aromatic)
       representative_atom_index: int         # Index of representative atom
       bonds: Dict[str, List]                 # Bonds grouped by type
       bond_index: Dict[str, List[int]]       # Bond indices grouped by type
       atomic_index: Dict[str, List[int]]     # Atom indices grouped by element
       coh_index: List[int]                   # COH bond indices
       coc_index: List[int]                   # COC bond indices




Basic Usage
===========

We use the ``create_molecular_info`` function to prepare ``MolecularInfo`` instance.
Let us prepare the following ``methanol.mol`` file.

.. code-block:: python

   input_GMX.gro created by acpype (v: 2022.7.21) on Sat Dec 10 16:39:17 2022
      OpenBabel08222309493D

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

``create_molecular_info`` function takes the file path as an input and generate ``MolecularInfo`` instance as follows:

.. code-block:: python

   from mlwc.bond.extractor_rdkit import create_molecular_info

   # Extract molecular information from a .mol file
   molecular_info = create_molecular_info("example.mol")

   # Access basic information
   print(f"Number of atoms: {molecular_info.num_atoms_per_mol}")
   print(f"Atom list: {molecular_info.atom_list}")
   print(f"Number of bonds: {molecular_info.num_bonds}")

When success, ``create_molecular_info`` outputs the following information:

.. code-block:: python

   2025-06-09 01:00:20 - MLWC.mlwc.bond.extractor_rdkit - INFO -  =====================
   2025-06-09 01:00:20 - MLWC.mlwc.bond.extractor_rdkit - INFO -   Atomic coordinates
   2025-06-09 01:00:20 - MLWC.mlwc.bond.extractor_rdkit - INFO -  C 0.94 0.02 -0.09
   2025-06-09 01:00:20 - MLWC.mlwc.bond.extractor_rdkit - INFO -  O 0.47 0.27 -1.4
   2025-06-09 01:00:20 - MLWC.mlwc.bond.extractor_rdkit - INFO - === ReadMolFile Analysis Results ===
   2025-06-09 01:00:20 - MLWC.mlwc.bond.extractor_rdkit - INFO - Number of atoms: 6
   2025-06-09 01:00:20 - MLWC.mlwc.bond.extractor_rdkit - INFO - Atom list: ['C', 'O', 'H', 'H', 'H', 'H']
   2025-06-09 01:00:20 - MLWC.mlwc.bond.extractor_rdkit - INFO - Number of bonds: 5
   2025-06-09 01:00:20 - MLWC.mlwc.bond.extractor_rdkit - INFO - COH indices: [[0, 1, {'CO': 0, 'OH': 0}]]
   2025-06-09 01:00:20 - MLWC.mlwc.bond.extractor_rdkit - INFO - COC indices: []
   2025-06-09 01:00:20 - MLWC.mlwc.bond.extractor_rdkit - INFO - ================ Bond Analysis ================
   2025-06-09 01:00:20 - MLWC.mlwc.bond.extractor_rdkit - INFO - CH_1_bond: [[0, 4], [0, 2], [0, 3]]
   2025-06-09 01:00:20 - MLWC.mlwc.bond.extractor_rdkit - INFO - CO_1_bond: [[1, 0]]
   2025-06-09 01:00:20 - MLWC.mlwc.bond.extractor_rdkit - INFO - OH_1_bond: [[5, 1]]
   2025-06-09 01:00:20 - MLWC.mlwc.bond.extractor_rdkit - INFO - ========== Atomic Indices ==========
   2025-06-09 01:00:20 - MLWC.mlwc.bond.extractor_rdkit - INFO - o_list atoms: [1]
   2025-06-09 01:00:20 - MLWC.mlwc.bond.extractor_rdkit - INFO - c_list atoms: [0]
   2025-06-09 01:00:20 - MLWC.mlwc.bond.extractor_rdkit - INFO - h_list atoms: [2, 3, 4, 5]

Atomic Information
------------------

The ``MolecularInfo`` object has ``atom_list`` and ``atomic_index`` variables.
``atom_list`` is the primary variable and the list of atomic symbols in a molecule.
Other two variables are derived from ``atom_list``.
``atomic_index`` is the dictionary, which shows which positions each atomic species occupies in ``atom_list``.
``num_atoms_per_mol`` is the number of atoms in a single molecule, i.e. ``len(atom_list)``.

.. code-block:: python

   print(f"atom_list: {molecular_info.atom_list}")
   print(f"atomic_index: {molecular_info.atomic_index}")
   print(f"Number of atoms in a molecule: {molecular_info.num_atoms_per_mol}")

   atom_list: ['C', 'O', 'H', 'H', 'H', 'H']
   atomic_index: {'o_list': [1], 'n_list': [], 'c_list': [0], 'h_list': [2, 3, 4, 5], 's_list': [], 'f_list': []}
   Number of atoms in a molecule: 6

Another unique quantity is ``representative_atom_index``, which shows the nearest atomic index to the mass center of a molecule.
In the case of methanol, it is oxygen atom of index ``1``.

.. code-block:: python

   print(f"representative_atom_index: {molecular_info.representative_atom_index}")

   representative_atom_index: 1


Bond Information
----------------

One can identify the chemical bond by its both ends of atoms and type (single, double, triple, and aromatic)
The former is represented by ``bonds_list``, whose element is tuple of atomic index.

``num_bonds=len(bonds_list)`` is the number of bonds.

The latter is contained in ``bonds_type``.
The bond types are mapped to the integers as follows:

* **1**: Single bond
* **2**: Double bond
* **3**: Triple bond
* **10**: Aromatic bond

In the following example of methanol, all five bonds are single.

.. code-block:: python

   from molecular_extractor import create_molecular_info

   # Load and analyze a molecule
   mol_info = create_molecular_info("methanol.mol")

   # Print bond information
   print("\n=== Bond Information ===")
   print(f"Total bonds: {mol_info.num_bonds}")
   print(f"Bond connections: {mol_info.bonds_list}")
   print(f"Bond types: {mol_info.bonds_type}")

   === Bond Information ===
   Total bonds: 5
   Bond connections: [[0, 4], [0, 2], [0, 3], [1, 0], [5, 1]]
   Bond types: [1, 1, 1, 1, 1]

``bonds_list`` is further processed in ``bonds`` and ``bond_index`` dictionaries.
For example, the data of single CO bond is stored in ``bonds["CO_1_bond"]``.

.. code-block:: python

   from molecular_extractor import create_molecular_info

   mol_info = create_molecular_info("methanol.mol")

   # Analyze different bond types
   print("=== Bond Analysis ===")
   for bond_type, bond_list in mol_info.bonds.items():
       if bond_list:
           print(f"{bond_type}: {len(bond_list)} bonds")
           print(f"  Connections: {bond_list}")

   # Access bond indices for specific analysis
   print("\n=== Bond Indices ===")
   for bond_type, indices in mol_info.bond_index.items():
       if indices:
           print(f"{bond_type} bond indices: {indices}")

   === Bond Analysis ===
   CH_1_bond: 3 bonds
   Connections: [[0, 4], [0, 2], [0, 3]]
   CO_1_bond: 1 bonds
   Connections: [[1, 0]]
   OH_1_bond: 1 bonds
   Connections: [[5, 1]]

   === Bond Indices ===
   CH_1_bond bond indices: [0, 1, 2]
   CO_1_bond bond indices: [3]
   OH_1_bond bond indices: [4]
