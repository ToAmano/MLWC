

import numpy as np
import ase
from cpmd.asign_wcs import raw_calc_mol_coord_and_bc_mic_onemolecule

def test_raw_calc_mol_coord_and_bc_mic_onemolecule():
    # テスト内容は以前の例と同じです
    symbols = ['H', 'O', 'H']
    positions = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0]
    ]
    cell = [10.0, 10.0, 10.0]
    aseatoms = ase.Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

    mol_inds = [0, 1, 2]
    bonds_list_j = [[0, 1], [1, 2]]

    class MockItpData:
        representative_atom_index = 0

    itp_data = MockItpData()

    mol_coords, bond_centers = raw_calc_mol_coord_and_bc_mic_onemolecule(
        mol_inds=mol_inds,
        bonds_list_j=bonds_list_j,
        aseatoms=aseatoms,
        itp_data=itp_data
    )

    expected_mol_coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0]
    ])
    expected_bond_centers = np.array([
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.5]
    ])

    assert np.allclose(mol_coords, expected_mol_coords), "Molecular coordinates do not match expected values."
    assert np.allclose(bond_centers, expected_bond_centers), "Bond centers do not match expected values."