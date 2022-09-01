from aiida.orm import load_group
from aiida.orm import StructureData
from ase.io import read
from aiida import load_profile

# load profile
load_profile()

# input filename
filename="GaAs_mp-2534_conventional_standard.cif"

# load structure (currently only cif is available)
ase_structure=read(filename, format="cif")
s = StructureData(ase=ase_structure)

# load pseudopotential
family = load_group('SSSP/1.1/PBEsol/precision')
family = load_group('PseudoDojo/0.4/LDA/SR/standard/upf')

# print recommended cutoff
# # cutoffs = family.get_recommended_cutoffs(elements=('Ga', 'As'))  # From a tuple or list of element symbols

# From a `StructureData` node
cutoffs = family.get_recommended_cutoffs(structure=s)

# print data
print(" ecutwfc,  ecutrho ")
print(" ------------------")
print(cutoffs, "  in Ry")
