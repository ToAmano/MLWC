'''
simple class to treat CPMD.x outputs
'''

# file="si_2/si_traj.xyz"


import sys
import numpy as np
import cpmd.read_core

try:
    import ase.io, ase.io.trajectory, ase.io.vasp
except ImportError:
    sys.exit ('Error: ase not installed')
try:
    import linecache
except ImportError:
    sys.exit ('Error: linecache not installed')

    

class CPMD_ReadPOS(cpmd.read_core.custom_traj):
    '''
    read *.pos and pwin file into list of ase.atoms.

    input
    ---------------
    filename :: string
       *.pos filename
    pwin     :: string
       pwin filename for cell parameters and chemical symbols
    '''
    def __init__(self, filename:str, cpmdout:str):
        # read atoms from cpmdout
        tmp_atom=raw_cpmd_read_to_ase(cpmdout)
        tmp_symbol=tmp_atom.get_chemical_symbols()
        tmp_cell=tmp_atom.get_cell()
        # get timestep
        self.__timestep = raw_cpmd_get_timestep(cpmdout)
        # read pos from filename
        pos_list, time_list=raw_cpmd_read_pos(filename, self.__timestep)
        # make atoms
        atoms_list=cpmd.read_core.raw_make_atomslist(pos_list, tmp_cell, tmp_symbol)
        # initialize custom_traj
        super().__init__(atoms_list=atoms_list, unitcell_vector=tmp_cell, filename=filename, time=time_list)
        # pwinも保存
        self.__cpmdout=cpmdout

    def save(self, prefix:str = ""):
        if prefix == "":
            ase.io.write(self.filename+"_refine.xyz", self.ATOMS_LIST, format="extxyz")
            #raw_save_aseatoms(self.ATOMS_LIST,  xyz_filename=self.filename+"_refine.xyz")
        else:
            ase.io.write(prefix+"_refine.xyz", self.ATOMS_LIST, format="extxyz")            
            #raw_save_aseatoms(self.ATOMS_LIST,  xyz_filename=prefix+"_refine.xyz") 
        return 0

    def set_force_from_file(self,for_filename:str):
        '''
        add forces from *.for file.
        ---------------
        input:
          for_name :: *.for file name.
        
        '''
        # read *.for file
        for_list, time_list=raw_cpmd_read_force(for_filename, self.__timestep)
        self.set_force(for_list) # method from cpmd.read_core.custom_traj
        return 0

    def export_dfset_cpmdout(self,interval_step:int=100,start_step:int=0):
        '''
        interval_stepごとにDFSETファイルに書き出す．
        '''
        initial_atom=raw_cpmd_read_to_ase(self.__cpmdout)
        cpmd.read_core.raw_export_dfset(initial_atom,self.ATOMS_LIST,self.force,interval_step,start_step)
        return 0


# * --------------------
# * 以下classで利用する関数
# * --------------------


def raw_cpmd_read_unitcell_vector(filename:str):
    '''
    only read unitcell vector from stdoutput ( in bohr unit).
    in cp.x case, 2nd line is cell parameters.
    -------------
    input
      - filename(string) :: xyz filename
    output
      - unitcell_vector(3*3 np array) :: unitcell vectors in row wise. unit is angstrom in cp.x case.
    '''
    
    f = open(filename)
    while True:
        line = f.readline()
        if line.startswith(" LATTICE VECTOR A1(BOHR)"):
            data=line.split()
            unitcell_x = [float(data[3]),float(data[4]),float(data[5])]            
        if line.startswith(" LATTICE VECTOR A2(BOHR)"):
            data=line.split()
            unitcell_y = [float(data[3]),float(data[4]),float(data[5])]            
        if line.startswith(" LATTICE VECTOR A3(BOHR)"):
            data=line.split()
            unitcell_z = [float(data[3]),float(data[4]),float(data[5])]    
        if not line:
            break        

    # unit convert from Bohr to Angstrom
    unitcell_vector = np.array([unitcell_x,unitcell_y,unitcell_z]) * ase.units.Bohr 
    return unitcell_vector



def raw_cpmd_read_to_ase(filename:str)-> ase.atoms:
    '''
    CPMDのstdoutputから初期構造を読み込む．
    '''
    
    flag:int = 0
    
    coordinate=[]
    symbols   =[]
    
    f = open(filename)
    while True:
        line = f.readline()
        if not line:
            break
        
        if line.startswith(" ****************************************************************"):
            flag = 0
        
        if flag == 1: # flag =1の間だけ原子座標を収集する．
            data = line.split()
            coordinate.append([float(data[2]),float(data[3]),float(data[4])])
            symbols.append(data[1])
            
        if line.startswith("   NR   TYPE        X(BOHR)        Y(BOHR)        Z(BOHR)     MBL"):
            flag = 1
    
    # convert bohr to angstrom
    coordinate = np.array(coordinate) * ase.units.Bohr
    
    # read unitcell vector
    unitcell_vector = raw_cpmd_read_unitcell_vector(filename)
    
    # makeing ase.atoms
    atom_output = ase.Atoms(
        symbols,
        positions=coordinate,
        cell=unitcell_vector,
        pbc=[1, 1, 1])
    return atom_output
    



def raw_cpmd_get_timestep(filename:str)->float:
    '''
    CPMDのstdoutputからtimestepを取得
    '''
    f = open(filename)
    while True:
        line = f.readline()
        if line.startswith(" TIME STEP FOR IONS:"):
            timestep=float(line.split()[4])
        if not line:
            break
    return timestep
        

def raw_cpmd_get_numatom(filename:str)->int:
    '''
    CPMDの作るTRAJECTORYファイルの最初のconfigurationを読み込んで原子がいくつあるかをcount_lineで数える.
    get_nbandsと似た関数
    '''
    count_line:int=0
    check_line:int=0
    f = open(filename)

    while True:
        data = f.readline()
        count_line+=1
        if count_line == 1: # 1行目の時のtimestepを取得
            timestep:int = data.split()[0]
        if data.split()[0] == timestep: 
            check_line+=1
        else:
            break

    numatoms:int = count_line-1
    if not __debug__:
        print(" -------------- ")
        print(" finish reading nbands :: numatoms = ", numatoms)
        print("")
    return numatoms


def raw_cpmd_read_pos(filename:str,timestep:float):
    '''
    CPMDのTRAJECTORYから，座標とFORCEを読み込む
    pos_list :: positions
    cell_parameter ::
    chemical_symbol
    
    timestep :: a.u.

    TODO :: forceがあるかないかの判別を!
    '''
    
    print(" ")
    print(" --------  WARNING from raw_CPMD_read_pos -------- ")
    print(" Please check you are correct inputs (TRAJECTORY or FTRAJECTORY) ")
    print(" This code does not check inputs format... ")
    print(" ")
    
    # numatom(原子数)を取得
    numatom=raw_cpmd_get_numatom(filename)

    
    f   = open(filename, 'r') # read TRAJECTORY/FTRAJECTORY

    # return lists
    pos_list = []  # atoms list
    time_list = [] # time steps in ps 
    
    with open(filename) as f:
        lines = f.read().splitlines()

    lines = [l.split() for l in lines] #
    for i,l in enumerate(lines) :
        if (i%(numatom) == 0) and (i==0) : #初めの行
            block = []
            time_list.append(float(l[0])) # time in ps
            block.append([float(l[1]), float(l[2]), float(l[3]) ])
        elif i%(numatom) == 0 : # numatom+1の時にpos_listとtimeにappend
            pos_list.append(block)
            block = []
            block.append([float(l[1]), float(l[2]), float(l[3]) ])           
            time_list.append(float(l[0])) # time in ps
        else : #numatom個の座標を読み込み
            block.append([float(l[1]), float(l[2]), float(l[3]) ])
    # append final step
    pos_list.append(block)
    
    # convert units
    # TODO:: 単位変換のところをちゃんと定数化
    time_list = np.array(time_list) *timestep* 2.4189 * 1e-5 
    pos_list = np.array(pos_list) * ase.units.Bohr # posはbohrなのでAngへ変換している．
    #
    return pos_list, time_list



def raw_cpmd_read_force(filename:str,timestep:float):
    '''
    CPMDのTRAJECTORYから，座標とFORCEを読み込む
    pos_list :: positions
    cell_parameter ::
    chemical_symbol

    timestep :: a.u.

    TODO :: forceがあるかないかの判別を!
    '''
    
    print(" ")
    print(" --------  WARNING from raw_cpmd_read_force -------- ")
    print(" Please use FTRAJECTORY (not TRAJECTORY) ")
    print(" ")
    
    # numatom(原子数)を取得
    numatom=raw_cpmd_get_numatom(filename)
    
    f   = open(filename, 'r') # read TRAJECTORY/FTRAJECTORY

    # return lists
    for_list = []  # atoms list
    time_list = [] # time steps in ps 
    
    with open(filename) as f:
        lines = f.read().splitlines()

    lines = [l.split() for l in lines] #
    for i,l in enumerate(lines) :
        if (i%(numatom) == 0) and (i==0) : #初めの行
            block = []
            time_list.append(float(l[0])) # time in ps
            block.append([float(l[7]), float(l[8]), float(l[9]) ])
        elif i%(numatom) == 0 : # numatom+1の時にpos_listとtimeにappend
            for_list.append(block)
            block = []
            block.append([float(l[7]), float(l[8]), float(l[9]) ])
            time_list.append(float(l[0])) # time in ps
        else : #numatom個の座標を読み込み
            block.append([float(l[7]), float(l[8]), float(l[9]) ])

    # append final step
    for_list.append(block)
    
    # convert units
    # TODO:: 単位変換のところをちゃんと定数化
    time_list = np.array(time_list)*timestep*2.4189 * 1e-5 
    for_list = np.array(for_list)*2 # forceの単位はa.u.=HARTREE ATOMIC UNITS=Eh/bohr=2Ry/bohr (bohr and Ryd/bohr)
    #
    return for_list, time_list
