# use regacy implementation to generate reference data

import ase 
import ase.io
import numpy as np
import ml.atomtype
import cpmd.class_atoms_wan
from ml.atomtype import read_mol


dir="pg/"
filename=dir+"/pg_wc.xyz"
bondfilename=dir+"/pg.mol"

# read bondfile
read_mol(bondfilename)

atoms = ase.io.read(filename,index=0)
itp_data = ml.atomtype.read_mol(bondfilename)

# pbc+bond center
atoms_wan = cpmd.class_atoms_wan.atoms_wan(atoms,itp_data.num_atoms_per_mol,itp_data)
ase.io.write("atoms.xyz", atoms_wan.atoms_nowan)
print(atoms_wan.atoms_nowan.get_positions().tolist())
print(" ============== ")

# calculate NUM_MOL
NUM_MOL:int = atoms_wan.NUM_MOL
# calculate bond centers
results = atoms_wan.ASIGN.aseatom_to_mol_coord_bc(atoms_wan.atoms_nowan, itp_data, itp_data.bonds_list)
list_mol_coords, list_bond_centers =results
np.savetxt("list_mol_coords.txt",np.array(list_mol_coords).reshape(-1,3))
np.savetxt("list_bond_centers.txt",np.array(list_bond_centers).reshape(-1,3))
print(np.array(list_mol_coords).reshape(-1,3).tolist())
print(" =============== ")
print(np.array(list_bond_centers).reshape(-1,3).tolist())

test = atoms_wan.atoms_nowan
test.set_positions(np.array(list_mol_coords).reshape(-1,3))
ase.io.write("pbc_mol.xyz",test)

# 記述子計算
descs_x = atoms_wan.DESC.calc_bond_descripter_at_frame(atoms_wan.atoms_nowan, 
                                                        list_bond_centers, 
                                                        itp_data.ch_bond_index, 
                                                        "allinone",
                                                        4,
                                                        6,
                                                        24)
np.savetxt("descs_ch.txt",descs_x)

# 記述子計算
descs_x = atoms_wan.DESC.calc_bond_descripter_at_frame(atoms_wan.atoms_nowan, 
                                                        list_bond_centers, 
                                                        itp_data.co_bond_index, 
                                                        "allinone",
                                                        4,
                                                        6,
                                                        24)
np.savetxt("descs_co.txt",descs_x)


# 記述子計算
descs_x = atoms_wan.DESC.calc_bond_descripter_at_frame(atoms_wan.atoms_nowan, 
                                                       list_bond_centers, 
                                                        itp_data.oh_bond_index, 
                                                        "allinone",
                                                        4,
                                                        6,
                                                        24)
np.savetxt("descs_oh.txt",descs_x)

# 記述子計算
descs_x = atoms_wan.DESC.calc_bond_descripter_at_frame(atoms_wan.atoms_nowan, 
                                                       list_bond_centers, 
                                                        itp_data.cc_bond_index, 
                                                        "allinone",
                                                        4,
                                                        6,
                                                        24)
np.savetxt("descs_cc.txt",descs_x)

# 記述子計算
descs_x = atoms_wan.DESC.calc_lonepair_descripter_at_frame_type2(atoms_wan.atoms_nowan, 
                                                        list_mol_coords, 
                                                        itp_data.o_list, 
                                                        "allinone",
                                                        4,
                                                        6,
                                                        24)
np.savetxt("descs_olp.txt",descs_x)


# ボンド双極子計算
atoms_wan._calc_wcs()
true_y  = atoms_wan.DESC.calc_bondmu_descripter_at_frame(atoms_wan.list_mu_bonds, 
                                                        itp_data.ch_bond_index) # .reshape(-1,3)
print(" ========= True_y (CH) ====== ")
print(true_y.tolist())
# np.savetxt("true_y.txt",true_y)
true_y  = atoms_wan.DESC.calc_bondmu_descripter_at_frame(atoms_wan.list_mu_bonds, 
                                                        itp_data.co_bond_index) # .reshape(-1,3)
print(" ========= True_y (CO) ====== ")
print(true_y.tolist())
# 
true_y  = atoms_wan.DESC.calc_bondmu_descripter_at_frame(atoms_wan.list_mu_bonds, 
                                                        itp_data.oh_bond_index) # .reshape(-1,3)
print(" ========= True_y (OH) ====== ")
print(true_y.tolist())
# 
true_y  = atoms_wan.list_mu_lpO.reshape(-1,3)
print(" ========= True_y (Olp) ====== ")
print(true_y.tolist())
#true_y  = atoms_wan.DESC.calc_bondmu_descripter_at_frame(atoms_wan.list_mu_bonds, 
#                                                        itp_data.oh_bond_index) # .reshape(-1,3)
#print(" ========= True_y (Olp) ====== ")
#print(true_y.tolist())
