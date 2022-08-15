#
# simple code to visualze XYZ by jupyter notebook  
# 

file="si_2/si_traj.xyz"

import ase.io
test=ase.io.read(file, index=":")

for i in range(len(test)):
    test[i].set_cell([5.6308,5.6308,5.6308]) #Ang
#for i in range(len(test)):
#    test[i].set_pbc((True,True,True))
#
#for i in range(len(test)):
#    test[i].set_pbc((False,False,False))
#

import numpy as np
for i in np.arange(1,len(test)):
    coord=test[i].get_positions()
    coord_before=test[i-1].get_positions()
    # jumpする場合
    if np.any(np.abs(coord_before-coord)>2.8):
        print("step :: ", i)
        # jumpする場合には格子定数を追加する．
        tmp1=np.where(coord-coord_before>2.8 , coord, coord+5.6308)
        tmp2=np.where(coord-coord_before<-2.8, tmp1, tmp1-5.6308)
        test[i].set_positions(tmp2)

# print("fin nojump")
# import numpy as np
# for i in np.arange(1,len(test)):
#     coord=test[i].get_positions()
#     coord_before=test[i-1].get_positions()
#     # jumpする場合
#     if np.any(np.abs(coord_before-coord)>2.8):
#         print("step :: ", i)

ase.io.write("si_2/si_test.xyz",images=test) #,format="xyz")


for i in np.arange(0,len(test)):
    test[i].pbc = (True, True, True)

