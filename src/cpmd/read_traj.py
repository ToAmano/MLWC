'''
simple class to treat CP.x output xyz file
This code is used in show_CP.py
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

    
class ReadCP(cpmd.read_core.custom_traj):
    '''
    cppp.xの出力するxyzファイルからtrajectoryを生成する．
    
    input
      - filename :: original xyz file from cppp.x

    method
    -------------------
    save :: prefixを与えなければfilename_refine.xyzの名前でextxyz形式で保存する．prefixを与えればその形式で保存．
    '''
    
    def __init__(self,filename:str):
        super().__init__(atoms_list=raw_transform_xyz(xyz_filename=filename, unitcell_vector=raw_read_unitcell_vector(filename), JUMP=False), unitcell_vector=raw_read_unitcell_vector(filename), filename=filename)
        #self.filename=filename
        #self.UNITCELL_VECTOR=raw_read_unitcell_vector(filename)
        #self.ATOMS_LIST=raw_transform_xyz(xyz_filename=filename, unitcell_vector=raw_read_unitcell_vector(filename), JUMP=False)
        # self.TRAJ=ase.io.trajectory.Trajectory(filename+"_refine.traj")

    #def nglview_traj(self):
    #    return raw_nglview_traj(self.ATOMS_LIST)
    # override custom_traj.save()
    def save(self, prefix:str = ""):
        if prefix == "":
            ase.io.write(self.filename+"_refine.xyz", self.ATOMS_LIST, format="extxyz")
            #raw_save_aseatoms(self.ATOMS_LIST,  xyz_filename=self.filename+"_refine.xyz")
        else:
            ase.io.write(prefix+"_refine.xyz", self.ATOMS_LIST, format="extxyz")
            #raw_save_aseatoms(self.ATOMS_LIST,  xyz_filename=prefix+"_refine.xyz") 
        return 0
    
class ReadXDATCAR(cpmd.read_core.custom_traj):
    '''
    input
      - filename :: original xdatcar file from VASP
    '''
    def __init__(self,filename:str):
        super().__init__(atoms_list=raw_transform_xdatcar(xdatcar_filename=filename), unitcell_vector=raw_transform_xdatcar(xdatcar_filename=filename)[0].get_cell(), filename=filename)
#        self.filename=filename
#        self.ATOMS_LIST=raw_transform_xdatcar(xdatcar_filename=filename)
#        self.UNITCELL_VECTOR=raw_transform_xdatcar(xdatcar_filename=filename)[0].get_cell()
        #self.TRAJ=ase.io.trajectory.Trajectory(filename+"_refine.traj")

    #def nglview_traj(self):
    #    return raw_nglview_traj(self.ATOMS_LIST)

    def save(self, prefix:str = ""):
        if prefix == "":
            ase.io.write(self.filename+"_refine.xyz", self.ATOMS_LIST, format="extxyz")
            #raw_save_aseatoms(self.ATOMS_LIST,  xyz_filename=self.filename+"_refine.xyz")
        else:
            ase.io.write(prefix+"_refine.xyz", self.ATOMS_LIST, format="extxyz")            
            #raw_save_aseatoms(self.ATOMS_LIST,  xyz_filename=prefix+"_refine.xyz") 
        return 0        

class ReadPOS(cpmd.read_core.custom_traj):
    '''
    read *.pos and pwin file into list of ase.atoms.

    input
    ---------------
    filename :: string
       *.pos filename
    pwin     :: string
       pwin filename for cell parameters and chemical symbols
    '''
    def __init__(self, filename:str, pwin:str):
        # read from pw.in
        tmp_atom=ase.io.read(pwin, format="espresso-in")
        tmp_symbol=tmp_atom.get_chemical_symbols()
        tmp_cell=tmp_atom.get_cell()
        # read from filename
        pos_list, time_list=raw_read_pos(filename)
        # make atoms
        atoms_list=cpmd.read_core.raw_make_atomslist(pos_list, tmp_cell, tmp_symbol)
        # initialize custom_traj
        super().__init__(atoms_list=atoms_list, unitcell_vector=tmp_cell, filename=filename, time=time_list)
        # pwinも保存
        self.__pwin=pwin

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
        for_list, time_list=raw_read_force(for_filename)
        self.set_force(for_list) # method from cpmd.read_core.custom_traj
        return 0

    def export_dfset_pwin(self,interval_step:int=100,start_step:int=0):
        '''
        interval_stepごとにDFSETファイルに書き出す．
        '''
        initial_atom=ase.io.read(self.__pwin)
        cpmd.read_core.raw_export_dfset(initial_atom,self.ATOMS_LIST,self.force,interval_step,start_step)
        return 0

def raw_read_unitcell_vector(filename:str):
    '''
    only read unitcell vector from xyz.
    in cp.x case, 2nd line is cell parameters.
    -------------
    input
      - filename(string) :: xyz filename
    output
      - unitcell_vector(3*3 np array) :: unitcell vectors in row wise. unit is angstrom in cp.x case.
    '''
    line=linecache.getline(filename, 2).split()
    if not len(line) == 9:
        print("ERROR :: xyzファイルの2行目の成分が9つ以上あります．", len(line) )
        sys.exit()
    unitcell_vector = np.array([float(s) for s in line ]).reshape([3,3])
    return unitcell_vector

def raw_transform_xyz(xyz_filename:str, unitcell_vector, JUMP:bool=False):
    '''
    read from cppp.x outputfile(xyz),
    resave as ase.xyz.
    --------------
    input
      - xyz_filename(string)          :: original xyz file from cppp.x (unit is Angstrom in cppp.x case)
      - unitcell_vector(3*3 np.array) :: unitcell vectors (unit should be the same as xyz_filename)
      - JUMP                          :: if crystal, set false.
    output
      - (ase.xyz)
    '''
    # read cppp.x output
    atoms_list=ase.io.read(xyz_filename, index=":")
    
    # 格子定数をセット
    for i in range(len(atoms_list)):
        atoms_list[i].set_cell(unitcell_vector) #Ang
        
    #  ========================
    if JUMP==False:
        '''
        結晶系で原子のジャンプを許さない場合はJUMP=Falseとする．
        現状以下のjumpは正方晶にのみ有効
        '''
        # 格子の対角成分を取り出す．
        cell_check=unitcell_vector[np.arange(3),np.arange(3)]
        # print(cell_check)
        # 
        for i in np.arange(1,len(atoms_list)):
            coord=atoms_list[i].get_positions()
            coord_before=atoms_list[i-1].get_positions()
            # jumpする場合
            if np.any(np.abs(coord_before-coord)>cell_check/2):
                # print("step :: ", i)
                # jumpする場合には格子定数を追加する．
                tmp1=np.where(coord-coord_before>cell_check/2 , coord, coord+cell_check)
                tmp2=np.where(coord-coord_before<-cell_check/2, tmp1, tmp1-cell_check)
                atoms_list[i].set_positions(tmp2)                    
        # print("fin nojump")
        # import numpy as np
        # for i in np.arange(1,len(atoms_list)):
        #     coord=atoms_list[i].get_positions()
        #     coord_before=atoms_list[i-1].get_positions()
        #     # jumpする場合
        #     if np.any(np.abs(coord_before-coord)>2.8):
        #         print("step :: ", i)
        #  ========================-
    # set pbc
    for i in np.arange(0,len(atoms_list)):
        atoms_list[i].pbc = (True, True, True)

    # save filename in extended xyz
    #if ifsave == True:
    #    ase.io.write(xyz_filename+"_refine.xyz",images=atoms_list,format="extxyz")
    #    ase.io.write(xyz_filename+"_refine.traj",images=atoms_list,format="traj")
    #
    return atoms_list


def raw_save_aseatoms(atoms_list:list, xyz_filename:str):
    '''
    DEPLICATE :: save ase.atoms to extxyz format.  
    ---------------
    input
      - atoms_list           ::
      - xyz_filename(string) :: extended xyz filename.

    output
    ---------------
      0

    Notes
    ---------------
    '''
    # save filename in ase.traj
    ase.io.write(xyz_filename, atoms_list,format="extxyz")
    # reload as traj
    return 0


def raw_transform_xdatcar(xdatcar_filename:str):
    '''
    Read XDATCAR file and convert it to ase.atoms list.

    input
    -----------------
    xdatcar_filename ::  string
       vasp xdatcar to parse

    Notes
    -----------------
    about usage of read_vasp_xdatcar ::
      - index=0:: read all steps
      - output is list ::[Atom(step=0),Atom(step=1),,,Atom(step=final)]    
    '''
    atom_list=ase.io.vasp.read_vasp_xdatcar(xdatcar_filename, index=0)

    # traj,xyzフォーマットで出力
    #if ifsave == True:
    #    ase.io.write(xdatcar_filename+"_refine.traj",atom_list,format="traj")
    #    ase.io.write(xdatcar_filename+"_refine.xyz" ,atom_list,format="extxyz")
    return atom_list

def get_numatom(filename:str):
    '''
    cp.xの作る*.posファイルの最初のconfigurationを読み込んで原子がいくつあるかをcount_lineで数える．
    configurationの判定はcheck_lineによって行われ，行の要素が2つあればcheck_lineと判定する．
    get_nbandsと似た関数
    '''
    count_line=0
    check_line=0
    f = open(filename)

    while True:
        count_line+=1
        if len(f.readline().split()) ==2: 
            check_line+=1
        if check_line == 2: # 2回目の時にbreakする．
            break

    numatoms = count_line-2 #2つ分引かないといけない
    if not __debug__:
        print(" -------------- ")
        print(" finish reading nbands :: numatoms = ", numatoms)
        print("")
    return numatoms

def raw_read_pos(filename:str):
    '''
    *.posファイルを読みこんでn*3のリストと時間情報を返す.
    原子種類の情報と格子定数の情報は別途ase.io.readから読み込む

    input
    ----------------
      - filename        :: str
            *.pos filename
    Returns
    -------
      - pos_list     :: list of arrays

    Notes
    -----
    格子定数は与えなくても良い．その場合格子定数を保持しないatoms.aseとして出力される．
    '''

    # numatom(原子数)を取得
    numatom=get_numatom(filename)

    # return lists
    pos_list = []  # atoms list
    time_list = [] # time steps in ps 
    
    with open(filename) as f:
        lines = f.read().splitlines()

    lines = [l.split() for l in lines] #
    for i,l in enumerate(lines) :
        if (i%(numatom+1) == 0) and (i==0) : #初めの行
            block = []
            time_list.append(float(l[1])) # time in ps
        elif i%(numatom+1) == 0 : # numatom+1の時にpos_listとtimeにappend
            pos_list.append(block)
            block = []
            time_list.append(float(l[1])) # time in ps
        else : #numatom個の座標を読み込み
            block.append([float(p) for p in l ])
    # final step
    pos_list.append(block)
    pos_list = np.array(pos_list) * ase.units.Bohr # posはbohrなのでAngへ変換している．
    #
    return pos_list, np.array(time_list)


def raw_read_force(for_filename:str):
    '''
    *.forファイルを読みこんでn*3のリストと時間情報を返す.

    input
    ----------------
      - filename        :: str
            *.pos filename
    Returns
    -------
      - pos_list     :: list of arrays

    Notes
    -----
    '''

    # numatom(原子数)を取得
    numatom=get_numatom(for_filename)

    # return lists
    for_list = []  # atoms list
    time_list = [] # time steps in ps 
    
    with open(for_filename) as f:
        lines = f.read().splitlines()

    lines = [l.split() for l in lines] #
    for i,l in enumerate(lines) :
        if (i%(numatom+1) == 0) and (i==0) : #初めの行
            block = []
            time_list.append(float(l[1])) # time in ps
        elif i%(numatom+1) == 0 : # numatom+1の時にpos_listとtimeにappend
            for_list.append(block)
            block = []
            time_list.append(float(l[1])) # time in ps
        else : #numatom個の座標を読み込み
            block.append([float(p) for p in l ])
    # final step
    for_list.append(block)
    for_list = np.array(for_list)*2 # forceの単位はa.u.=HARTREE ATOMIC UNITS=Eh/bohr=2Ry/bohr (bohr and Ryd/bohr)
    return for_list, np.array(time_list)


    

# DEPLECATE :: old code (delete in the future)

# def ReadXDATCAR(filename):
#     '''
#     Read XDATCAR file and convert it to the *.traj file.
#     '''

#     # index=0:: read all steps
#     # output is list ::[Atom(step=0),Atom(step=1),,,Atom(step=final)]
#     test=ase.io.vasp.read_vasp_xdatcar(filename, index=0)

#     # traj,xyzフォーマットで出力
#     ase.io.write(filename+"_refine.traj",test,format="traj")
#     ase.io.write(filename+"_refine.xyz",test,format="extxyz")
#     # ase.io.trajectory.TrajectoryWriter("si_2/test.traj", test)
#     traj = ase.io.trajectory.Trajectory(filename+"_refine.traj")
#     return traj


        
# DEPLECATE :: old code (delete in the future)

# class ReadCP:

#     def __init__(self,filename):
#         self.__filename=filename
#         self.UNITCELL_VECTOR=None
#         self.ATOMS_LIST=None
#         self.TRAJ=None
        
#     def read_UNITCELL_VECTOR(self):
#         '''
#         only read unitcell vector from xyz
#         '''
#         # UNITCELL_VECTOR=xyz[0].get_cell()
        
#         # cp.xのoutputの場合，xyzの2行目が格子定数になっている．
#         count_line=0
#         #readlineで1行だけ読み込み
#         with open(self.__filename) as f:
#             while True:
#                 count_line=count_line+1
#                 line = f.readline()
#                 if count_line==2:
#                     break

#         self.UNITCELL_VECTOR = np.array([float(s) for s in line.split()]).reshape([3,3])
#         # print(cell_vector) # 格子定数を取得
#         return self.UNITCELL_VECTOR
    
        
#     def transform_xyz(self, JUMP=False,unitcell_vector=self.UNITCELL_VECTOR):
#         '''
#         read from cppp.x outputfile(xyz),
#         resave as ase.xyz and ase.traj files.
#         This code should be called AFTER Unitcell is done
#         '''
#         if unitcell_vector==None:
#             print("ERROR :: unitcell_vector is empty. ")
#             return 1

#         # read cppp.x output
#         atoms_list=ase.io.read(self.__filename, index=":")

#         # 格子定数をセット
#         for i in range(len(atoms_list)):
#             atoms_list[i].set_cell(self.UNITCELL_VECTOR) #Ang

#         #  ========================
#         if JUMP==False:
#             '''
#             結晶系で原子のジャンプを許さない場合はJUMP=Falseとする．
#             現状以下のjumpは正方晶にのみ有効
#             '''
#             # 格子の対角成分を取り出す．
#             cell_check=self.UNITCELL_VECTOR[np.arange(3),np.arange(3)]
#             # print(cell_check)
#             # 
#             for i in np.arange(1,len(atoms_list)):
#                 coord=atoms_list[i].get_positions()
#                 coord_before=atoms_list[i-1].get_positions()
#                 # jumpする場合
#                 if np.any(np.abs(coord_before-coord)>cell_check/2):
#                     # print("step :: ", i)
#                     # jumpする場合には格子定数を追加する．
#                     tmp1=np.where(coord-coord_before>cell_check/2 , coord, coord+cell_check)
#                     tmp2=np.where(coord-coord_before<-cell_check/2, tmp1, tmp1-cell_check)
#                     atoms_list[i].set_positions(tmp2)                    
#             # print("fin nojump")
#             # import numpy as np
#             # for i in np.arange(1,len(atoms_list)):
#             #     coord=atoms_list[i].get_positions()
#             #     coord_before=atoms_list[i-1].get_positions()
#             #     # jumpする場合
#             #     if np.any(np.abs(coord_before-coord)>2.8):
#             #         print("step :: ", i)
#             #  ========================-
#         # set pbc
#         for i in np.arange(0,len(atoms_list)):
#             atoms_list[i].pbc = (True, True, True)

#         #
#         self.ATOMS_LIST=atoms_list
#         # save filename in extended xyz
#         ase.io.write(self.__filename+"_refine.xyz",images=atoms_list,format="xyz")    
#         return self.ATOMS_LIST

    
#     def save_Traj(self):

#         '''
#         This code should be called AFTER transform_xyz is done
#         '''
        
#         # save filename in extended xyz
#         ase.io.write(self.__filename+"_refine.xyz",images=self.ATOMS_LIST,format="xyz")    
#         # save filename in ase.traj
#         ase.io.write(self.__filename+"_refine.traj",self.ATOMS_LIST, format="traj")
#         self.TRAJ = ase.io.trajectory.Trajectory(self.__filename+"_refine.traj") # reload as traj
#         return self.TRAJ

        
        
