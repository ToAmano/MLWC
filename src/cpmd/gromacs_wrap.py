


def make_mdp_em(cutoff:float):
    '''
    make mdp file for energy minimization

    cutoff :: in Angstrom
    '''

    # * hard code :: mdp file
    mdp_file = "em.mdp"
    cutoff_radius    = cutoff/10.0 #Ang to nm
    
    lines = [
    "; VARIOUS PREPROCESSING OPTIONS",
    ";title                    = Yo",
    ";cpp                      = /usr/bin/cpp",
    "include                  =", 
    "define                   =", 
    "    ",
    "; RUN CONTROL PARAMETERS",
    "integrator               = steep",
    "nsteps                   = 1000000",
    "emtol                    = 10",
    "emstep                   = 0.1",
    "nstlist                  = 1",
    "cutoff-scheme            = verlet",
    "vdw-type                 = cut-off",
    "rlist                    = {}".format(cutoff_radius),
    "rvdw                     = {}".format(cutoff_radius),
    "rcoulomb                 = {}".format(cutoff_radius),
    ]

    with open(mdp_file, mode='w') as f:
        f.write('\n'.join(lines))

    return 0


def make_mdp_nvt(temp,steps,dt,cutoff):
    '''
    make mdp file for NVT run
    '''
    
    temperature      = temp
    simulation_steps = steps 
    time_step        = dt/1000.0  # ps
    cutoff_radius    = cutoff/10.0
    
    mdp_file = "run.mdp"

    lines = [
    "; VARIOUS PREPROCESSING OPTIONS",
    ";title                    = Yo",
    ";cpp                      = /usr/bin/cpp",
    "include                  =", 
    "define                   =", 
    "    ",
    "; RUN CONTROL PARAMETERS",
    "constraints              = none",
    "integrator               = md",
    "nsteps                   = {}".format(simulation_steps),
    "dt                       = {}".format(time_step),
    "nstlist                  = 1",
    "rlist                    = {}".format(cutoff_radius),
    "rvdw                     = {}".format(cutoff_radius),
    "rcoulomb                 = {}".format(cutoff_radius),
    "coulombtype              = pme",
    "cutoff-scheme            = verlet",
    "vdw-type                 = cut-off",        
    "tc-grps                  = system",
    "tau-t                    = 0.1",
    "gen-vel                  = yes",
    "gen-temp                 = {}".format(temperature),
    "ref-t                    = {}".format(temperature),
    "Pcoupl                   = no",
    "Tcoupl                    = v-rescale " ,
    "nstenergy                = 5",
    "nstxout                  = 5", 
    "nstfout                  = 5",
    "DispCorr                 = EnerPres",
    ]

    with open(mdp_file, mode='w') as f:
        f.write('\n'.join(lines))
    return 0

def build_initial_cell_gromacs(dt,eq_cutoff,eq_temp,eq_steps,max_atoms:float,density:float,gro_filename:str="input1.gro",itp_filename:str="input1.itp"):
    '''
    gro_filename:: input用のgroファイル名
    itp_filename:: input用のitpファイル名
    '''
    import os
    # check whether input files exist.
    if not os.path.isfile(gro_filename):
        print(" ERROR :: "+str(gro_filename)+" does not exist !!")
        print(" ")
        return 1
    if not os.path.isfile(itp_filename):
        print(" ERROR :: "+str(itp_filename)+" does not exist !!")
        print(" ")
        return 1
    

    import subprocess
    from subprocess import PIPE

    import pandas as pd
    
    import time 
    init_time = time.time()
    
    dt = dt

    import MDAnalysis as mda
#    from nglview.datafiles import PDB, XTC # これ，使ってなくない？

    #混合溶液を作成
    import mdapackmol
    import numpy as np
    from ase import units
    import shutil

    # load individual molecule files
    mol1 = mda.Universe(gro_filename)
    #num_mols1 = 30
    total_mol = int(max_atoms/(mol1.atoms.n_atoms))
    num_mols1 = total_mol
    mw_mol1 = np.sum(mol1.atoms.masses)
    total_weight = num_mols1 * mw_mol1 
    
    # Determine side length of a box with the density of mixture 
    #L = 12.0 # Ang. unit 
    d = density / 1e24 # Density in g/Ang3 
    volume = (total_weight / units.mol) / d
    L = volume**(1.0/3.0)
    print(" --------------      ")
    print(" print parameters ...")
    print(" CELL PARAMETER :: ", L/10)
    print(" VOLUME         :: ", volume)

    # 複数分子を含む系を作成する．
    system = mdapackmol.packmol(
    [ mdapackmol.PackmolStructure(
    mol1, number=num_mols1,
    instructions=["inside box "+str(0)+"  "+str(0)+"  "+str(0)+ "  "+str(L)+"  "+str(L)+"  "+str(L)]),])

    # 作成した系（system）をmixture.groへ保存
    system.atoms.write('mixture.gro')

    import os 
    os.environ['GMX_MAXBACKUP'] = '-1'

    # for gromacs-5 or later (init.groを作成)
    print(" RUNNING :: gmx editconf ... ( making init.gro) ")
    os.system("gmx editconf -f mixture.gro  -box "+ str(L/10.0)+"  "+str(L/10.0)+"  "+str(L/10.0) + "  " +" -o init.gro")
    print(" ----------- ")
    print(" FINISH gmx editconf")
    print(" ")

    #make top file for GAFF

    top_file = "system.top"
    mol_name1 = "input"
 
    lines = [
        "; input_GMX.top created by acpype (v: 2020-07-25T09:06:13CEST) on Fri Jul 31 07:59:08 2020",
        ";by acpype (v: 2020-07-25T09:06:13CEST) on Fri Jul 31 07:59:08 2020",
        "   ",
        "[ defaults ]",
        "; nbfunc        comb-rule       gen-pairs       fudgeLJ fudgeQQ",
        "1               2               yes             0.5     0.8333",
        "    ",
        "; Include input.itp topology", 
        "#include \"{}\"".format(itp_filename), # * hard code :: input1.itpに固定されている．
        "    ",
        "[ system ]",
        "input",
        "     ",
        "[ molecules ]",
        "; Compound        nmols" ,
        mol_name1 + "          {} ".format(num_mols1), 
    ]
        
    with open(top_file, mode='w') as f:
        f.write('\n'.join(lines))

    # Energy minimization
    import os
    import subprocess
    from subprocess import PIPE
    print(" -----------")
    print(' Minimizing energy')
    print(" ")
    
    os.environ['GMX_MAXBACKUP'] = '-1'

    # make mdp em ?
    make_mdp_em(eq_cutoff)

    #grompp
    os.environ['OMP_NUM_THREADS'] = '1'    
    os.system("gmx grompp -f em.mdp -p system.top -c init.gro -o em.tpr -maxwarn 10 ")
    print(" ")
    print(" FINISH gmx grompp")
    print(" ")
    
    #mdrun
    os.environ['OMP_NUM_THREADS'] = '1' 
    os.system("gmx mdrun -s em.tpr -o em.trr -e em.edr -c em.gro -nb cpu")
    print(" ")
    print(" FINISH gmx mdrun")
    print(" ")

    #Relax the geometry
    print(" ")
    print(" Running dynamics :Equilibration")
    print(" ")
  
    temp = eq_temp
    dt   = dt 
    steps = eq_steps
    make_mdp_nvt(temp,steps,dt,eq_cutoff)

    #grompp
    os.environ['OMP_NUM_THREADS'] = '1'
    os.system("gmx grompp -f run.mdp -p system.top -c em.gro -o eq.tpr -maxwarn 10 ".format(str(temp)))
    print(" ")
    print(" FINISH gmx grompp")
    print(" ")
  
    #mdrun (eq.groを作成)
    os.environ['OMP_NUM_THREADS'] = '6' 
    os.system("gmx mdrun -s eq.tpr -o eq.trr -e eq.edr -c eq.gro -nb cpu")
    print(" ")
    print(" FINISH gmx mdrun ")
    print(" ")

    print(" ------------- ")
    print(" summary")
    print(" elapsed time= {} sec.".format(time.time()-init_time))
    print(" ")
    return 0
