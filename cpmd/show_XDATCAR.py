import ase
from ase.io import read, write
from ase.io.vasp import read_vasp_xdatcar

test=read_vasp_xdatcar("XDATCAR", index=0)
print(len(test))

# これはちゃんとリスト+Atomsの形式を認識している！！
from ase.visualize import view
view(test, viewer='ngl')


#import nglview as nv
#import ase.io
#w = nv.show_ase(test)
#w.add_label(radius=0.8,color="black",label_type="atom")

#w.clear_representations()
#w.add_label(radius=1,color="black",label_type="atom")
#view.add_representation("ball+stick")
#w.add_representation("ball+stick",selection=[i for i in range(0,n_atoms)],opacity=1.0)
#w.add_representation("ball+stick",selection=[i for i in range(n_atoms,total_atoms)],opacity=1,aspectRatio=2)
#w.add_unitcell()
#w.update_unitcell()
#w
