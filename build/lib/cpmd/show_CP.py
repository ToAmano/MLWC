
#
# simple code to visualze XYZ by jupyter notebook  
# 
# file="si_2/si_traj.xyz"


import ase.io
import numpy as np
import nglview as nv
from ase.io.trajectory import Trajectory
import cpmd.read_traj


if FORMAT == "CP":
    traj =  cpmd.read_traj.ReadCP(filename)
    view =  nv.show_asetraj(traj.ATOMS_LIST)
elif FORMAT == "VASP":
    traj = cpmd.read_traj.ReadXDATCAR(filename)
    view=nv.show_asetraj(traj.ATOMS_LIST)
    
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
