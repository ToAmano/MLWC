
#
# simple code to visualze XYZ by jupyter notebook  
# 
# file="si_2/si_traj.xyz"

import ase.io
import numpy as np
import nglview as nv
test=ase.io.read(filename, index=":")

# cp.xのoutputの場合，xyzの2行目が格子定数になっている．
count_line=0
# readlineで1行だけ読み込み
with open(filename) as f:
    while True:
        count_line=count_line+1
        line = f.readline()
        if count_line==2:
            break
cell_vector = np.array([float(s) for s in line.split()]).reshape([3,3])
print(cell_vector) # 格子定数を取得

# 格子定数をセット
for i in range(len(test)):
    test[i].set_cell(cell_vector) #Ang

#  ========================-

if JUMP==False:
    # 現状以下のjumpは正方晶にのみ有効
    # 格子の対角成分を取り出す．
    cell_check=cell_vector[np.arange(3),np.arange(3)]
    # print(cell_check)
    
    # 
    for i in np.arange(1,len(test)):
        coord=test[i].get_positions()
        coord_before=test[i-1].get_positions()
        # jumpする場合
        if np.any(np.abs(coord_before-coord)>cell_check/2):
            # print("step :: ", i)
            # jumpする場合には格子定数を追加する．
            tmp1=np.where(coord-coord_before>cell_check/2 , coord, coord+cell_check)
            tmp2=np.where(coord-coord_before<-cell_check/2, tmp1, tmp1-cell_check)
            test[i].set_positions(tmp2)

# print("fin nojump")
# import numpy as np
# for i in np.arange(1,len(test)):
#     coord=test[i].get_positions()
#     coord_before=test[i-1].get_positions()
#     # jumpする場合
#     if np.any(np.abs(coord_before-coord)>2.8):
#         print("step :: ", i)
#  ========================-



# set pbc
for i in np.arange(0,len(test)):
    test[i].pbc = (True, True, True)

# save filename in extended xyz
ase.io.write(filename+"_refine.xyz",images=test,format="xyz")


    
from ase.io.trajectory import Trajectory
# save filename in ase.traj
ase.io.write(filename+"_refine.traj",test,format="traj")

# reload as traj
traj = Trajectory(filename+"_refine.traj")
view=nv.show_asetraj(test)
view.parameters =dict(
                        camera_type="orthographic",
                        backgraound_color="black",
                        clip_dist=0
)
view.clear_representations()
view.add_representation("ball+stick")
#view.add_representation("spacefill",selection=[i for i in range(n_atoms,n_total_atoms)],opacity=0.1)
view.add_unitcell()
view.update_unitcell()
view
