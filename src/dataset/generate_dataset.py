# use regacy implementation to generate reference data

import ase 
import ase.io
import numpy as np
import ml.atomtype
import cpmd.class_atoms_wan

filename="methanol/methanol.xyz"
bondfilename="methanol/methanol.mol"

atoms = ase.io.read(filename,index=0)
itp_data = ml.atomtype.read_mol(bondfilename)

# pbc+bond center
atoms_wan = cpmd.class_atoms_wan.atoms_wan(atoms,itp_data.num_atoms_per_mol,itp_data)
print(atoms_wan.atoms_nowan.get_positions().tolist())
print(" ============== ")


# calculate NUM_MOL
NUM_MOL:int = atoms_wan.NUM_MOL
# calculate bond centers
results = atoms_wan.ASIGN.aseatom_to_mol_coord_bc(atoms_wan.atoms_nowan, itp_data, itp_data.bonds_list)
list_mol_coords, list_bond_centers =results
print(np.array(list_mol_coords).reshape(-1,3).tolist())
print(" =============== ")
print(np.array(list_bond_centers).reshape(-1,3).tolist())

# 記述子計算
descs_x = atoms_wan.DESC.calc_bond_descripter_at_frame(atoms_wan.atoms_nowan, 
                                                       list_bond_centers, 
                                                        itp_data.ch_bond_index, 
                                                        "allinone",
                                                        4,
                                                        6,
                                                        24)
np.savetxt("descs_x.txt",descs_x)

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
