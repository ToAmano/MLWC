def make_ase(symbols,positions,cell):
    """座標,symbol,格子からase.atomsを作成する
    symbols,positions,cellはmdtrajからデータを与えるのでnm単位
    """
    from ase import Atoms # MIC計算用
    mols = Atoms(symbols=symbols,
                 positions=positions*10,  # nm(gro) to ang (ase)
                 cell= cell*10,
                 pbc=[1, 1, 1]) #pbcは周期境界条件．これをxyz軸にかけることを表している．
    return mols


import mdtraj
traj=mdtraj.load("eq_pbc.trr", top="eq.pdb")
print("traj :: ", len(traj))

# セルを取得
UNITCELL_VECTORS=traj[-1].unitcell_vectors.reshape([3,3])
# どうもこれだと格子の情報が入らないのでそれを入れる？

# symbolsを取得
table, bonds =traj[-1].topology.to_dataframe()
symbols=table['element']

# xyzを取得
traj_coords = traj.xyz

import ase.io
import ase

# list[atoms]の作成
answer_atomslist=[]
for positions in traj_coords:
    aseatom=make_ase(symbols,positions,UNITCELL_VECTORS)
    answer_atomslist.append(aseatom)
# 保存
ase.io.write("gromacs_trajectory_cell.xyz",answer_atomslist)

print("==========")
print(" end ")
