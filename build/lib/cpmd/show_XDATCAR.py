'''
DEPLICATE :: delete in the future
'''


def Write_Traj(filename):
    '''
    Read XDATCAR file and convert it to the *.traj file.
    '''
    from ase.io import write
    from ase.io.vasp import read_vasp_xdatcar
    from ase.visualize import view
    from ase.io.trajectory  import Trajectory
    import nglview as nv
    
    # index=0:: read all steps
    # output is list ::[Atom(step=0),Atom(step=1),,,Atom(step=final)]
    test=read_vasp_xdatcar(filename, index=0)
    # using ase.visualize.view (not nglview)
    # view(test, viewer='ngl')
    
    write(filename+".traj",test,format="traj")
    # ase.io.trajectory.TrajectoryWriter("si_2/test.traj", test)
    traj = Trajectory(filename+".traj")
    return traj

# convert to ase.traj and re-read
traj=Write_Traj(filename)
# re-read ase.traj
view=nv.show_asetraj(traj)
view.parameters =dict(
                        camera_type="orthographic",
                        backgraound_color="black",
                        clip_dist=0
)
view.clear_representations()
view.add_representation("ball+stick")
view.add_unitcell()
view.update_unitcell()
view
