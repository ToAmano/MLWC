import ase
import ase.io

data = ase.io.read("traj/IONS+CENTERS+cell_sorted_merge.xyz", index=":")
print(len(data))
