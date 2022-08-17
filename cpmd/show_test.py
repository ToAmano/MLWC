#!/usr/bin/env python3
#
# simple code to visualze XDATCAR by jupyter notebook  
# 

import argparse

description='''
       Simple script for plotting IFCs and Distances relations.
      Usage:
      $ python analyze.py file1.fcs

      For details of available options, please type
      $ python analyze.py -h
'''

parser = argparse.ArgumentParser(description=description)
#
# 3. parser.add_argumentで受け取る引数を追加していく
parser.add_argument("Filename"  , help='trajectory filename. acceptable formats are *.xyz and XDATCAR')    # 必須の引数を追加
args = parser.parse_args()    # 4. 引数を解析


from ase.io import write
from ase.io.vasp import read_vasp_xdatcar
from ase.visualize import view
from ase.io.trajectory  import Trajectory
import nglview as nv

# index=0:: read all steps
# output is list ::[Atom(step=0),Atom(step=1),,,Atom(step=final)]
test=read_vasp_xdatcar("XDATCAR", index=0)
# using ase.visualize.view (not nglview)
# view(test, viewer='ngl')

write("XDATCAR.traj",test,format="traj")
# ase.io.trajectory.TrajectoryWriter("si_2/test.traj", test)

traj = Trajectory("XDATCAR.traj")
view=nv.show_asetraj(traj)
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


# w = nv.show_ase(test)
# w.add_label(radius=0.8,color="black",label_type="atom")

# w.clear_representations()
# w.add_label(radius=1,color="black",label_type="atom")
# view.add_representation("ball+stick")
# w.add_representation("ball+stick",selection=[i for i in range(0,n_atoms)],opacity=1.0)
# w.add_representation("ball+stick",selection=[i for i in range(n_atoms,total_atoms)],opacity=1,aspectRatio=2)
# w.add_unitcell()
# w.update_unitcell()
# w
