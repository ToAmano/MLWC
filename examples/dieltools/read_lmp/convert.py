#
import ase
import ase.io

traj = ase.io.read("test.lammpstrj",index=":")
# traj = ase.io.read("test_initial_64mol_npt_298K_pro.lammpstrj",index="1:3")

print(len(traj))
# ase.io.write("traj_test.xyz",traj)

new_traj = []
atomic_numbers = traj[0].get_chemical_symbols()
cell = traj[0].get_cell()
for atoms in traj:
    new_traj.append(
        ase.Atoms(atomic_numbers,
                  positions=atoms.get_positions(),
                  cell=cell,
                  pbc=[1, 1, 1])
    )


ase.io.write("traj.xyz",new_traj)
