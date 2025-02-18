import pandas as pd

import cpmd.gromacs_wrap

# ----- System setting 1 :: change here -------
# num of molecules in supercell
num_molecules = 32
# density(g/cm3) from experiment
density = 0.791

# ----- System setting 2 :: DO NOT CHANGE  -------
dt = 0.2                           #[fs] MDの刻み時間：このまま使うことを推奨。
eq_temp = 25+273.15                #緩和計算させるときの温度 [K]
eq_steps = 25000                   #緩和計算するstep数。この例だと5 psec.
eq_cutoff = 4.8


# 最初のセルの構築
cpmd.gromacs_wrap.build_mixturegro(num_molecules,density,gro_filename="input.acpype/input_GMX.gro")

# gromacsの実行
cpmd.gromacs_wrap.build_initial_cell_gromacs(dt,eq_cutoff,eq_temp,eq_steps,num_molecules,density,gro_filename="input.acpype/input_GMX.gro",itp_filename="input.acpype/input_GMX.itp")


#構造可視化(matplotlib版)
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms
import ase.io
# mol1 = ase.io.read('eq.gro')
# # %matplotlib inline
# plot_atoms(mol1, rotation=('0x,0y,0z'))
# plt.show()
# plt.savefig("eqgro.png")


# CPMD/QE用インプットを作るためにgmx trajconvを使ってfinal_structure.groを作成
# gromacsの出力をいくつか利用する．
cpmd.gromacs_wrap.make_gro_for_qeinput()
