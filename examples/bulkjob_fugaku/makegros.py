
import ase

import mdtraj
import os

import cpmd.converter_cpmd
# ファイルを読み込み
traj=mdtraj.load("gromacs_test/eq_pbc.trr", top="gromacs_test/eq.pdb")

# ディレクトリをほって，そこにeq.groという名前で保存する．
num_config = len(traj)
print(num_config)

# まずはbulkjob用のdirをほる
os.system("mkdir bulkjob".format(str(i)))

for i in range(num_config):
    # print("step :: ", i)
    os.system("mkdir bulkjob/struc_{}".format(str(i)))
    traj[i].save_gro("bulkjob/struc_{}/final_structure.gro".format(str(i)))

    # groを読み込み入力を作成．
    ase_atoms=ase.io.read("bulkjob/struc_{}/final_structure.gro".format(str(i)))
    test=cpmd.converter_cpmd.make_cpmdinput(ase_atoms)
    test.make_bomd_oneshot(type="sorted")

    # 作成したファイルを移動させる．
    os.system("mv bomd-oneshot.inp bulkjob/struc_{}/bomd-oneshot.inp".format(str(i)))
    os.system("mv sort_index.txt   bulkjob/struc_{}/sort_index.txt".format(str(i)))
    os.system("mkdir bulkjob/struc_{}/tmp".format(str(i)))

