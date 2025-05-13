#
# simple code to visualze XYZ by jupyter notebook
#
# file="si_2/si_traj.xyz"


import ase.io
import nglview as nv
import numpy as np
from ase.io.trajectory import Trajectory

import dataio.cpx.read_traj

if FORMAT == "CP":
    traj = io.cpx.read_traj.ReadCP(filename)
    view = nv.show_asetraj(traj.ATOMS_LIST)
elif FORMAT == "VASP":
    traj = io.cpx.read_traj.ReadXDATCAR(filename)
    view = nv.show_asetraj(traj.ATOMS_LIST)

view.parameters = dict(
    camera_type="orthographic", backgraound_color="black", clip_dist=0
)
view.clear_representations()
view.add_representation("ball+stick")
# view.add_representation("spacefill",selection=[i for i in range(n_atoms,n_total_atoms)],opacity=0.1)
view.add_unitcell()
view.update_unitcell()
view
