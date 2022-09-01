    
from ase.io import read
import ase
import sys
import numpy as np
from types import NoneType

try:
    import nglview
except ImportError:
    sys.exit ('Error: nglview not installed')

class custom_traj():
    """
    ase.atomsのリストを保持するためのクラス．ReadCP，ReadXDATCARで継承するための親クラスとして利用．
    さらに，dipole計算を可能にするためbec,timestep,dipoleを格納できるようにする．

    input
    ---------------
      self.ATOMS_LIST       :: list of ase.atoms 
      self.UNITCELL_VECTOR  :: 3*3 numpy array 
      self.filename         :: filename from which coordinates are read
      self.nstep            :: number of configuration. should be equal to len(self.ATOMS_LIST)
      self.time             :: times in ps unit.
      self.bec              :: can add later in e unit.
      self.dipole           :: can add later in debye unit.
    
    output
    ---------------
    UNITCELL_VECTOR :: 3*3 numpy array
      unitcell vectors of the first configuration. 
    
    Notes
    ---------------
    一つ問題があって，filenameをどうするか，という点．一応デフォルトの値を""にしておく．ReadCPなどではmethodのオーバーライドを行えば良い．
    また，この実装では格子定数と原子数が一定であるという前提になっている．
    """
    
    def __init__(self, atoms_list:list, unitcell_vector=None, filename="", time=None):
        self.ATOMS_LIST      = atoms_list
        self.UNITCELL_VECTOR = unitcell_vector
        self.filename        = filename
        self.nstep  =len(atoms_list) # automatic detection
        self.time   =None          
        self.bec    =None 
        self.dipole =None

        if type(time) != NoneType:
            self._initialize_time(time)
        
    def _initialize_time(self, time):
        '''
        initialize timedata 
        '''
        time=np.array(time) # if time is list, change it to np.array
        if self.nstep != len(time):
            print("ERROR :: len(atoms_list) != len(time). len(time) = ", len(time), "nstep = ", self.nstep)
            sys.exit()

        if len(np.unique(time)) != len(time):
            print("WARNING :: time has some duplicate values.")
        self.time = time

    def set_unitcell_from_atoms(self):
        '''
        set unitcell from self.atoms

        NOTEs
        ----------------
        将来的には__init__の中に組み込んで自動で追加する仕組みにしたい？
        '''
        self.UNITCELL_VECTOR= self.ATOMS_LIST.get_cell()

    def set_unitcell(self,unitcell_vector):
        '''
        set unitcell for self.UNITCELL_VECTOR and self.ATOMS_LIST
        '''
        self.UNITCELL_VECTOR= unitcell_vector
        for i in range(self.nstep):
            self.ATOMS_LIST[i].set_cell(unitcell_vector)
        return 0
        
    def set_bec(self,bec):
        if np.shape(bec)[1] != 3 or np.shape(bec)[2] != 3: # 型チェック
        #if np.shape(bec)[0] != self.nstep or np.shape(bec)[1] != 3 or np.shape(bec)[2] != 3:
            print("ERROR :: shape of bec incorrect :: (nstep,3,3)")
            sys.exit()
        self.bec = bec
        return self.bec

    def set_dipole(self, dipole):
        if np.shape(dipole)[1] != 3 :
        #if np.shape(dipole)[0] != self.nstep or np.shape(dipole)[1] != 3 :
            print("ERROR :: shape of bec incorrect :: (nstep,3)")
            sys.exit()
        self.dipole = dipole
        return self.dipole

    def set_charges(self,charge_list):
        '''
        スカラー電荷のリストを与えるとそれをase.atomsのリストに自動で加えてくれる．
        加えた電荷はase.get.charges()で確認できる．

        input
        ---------------
        charge_list :: list of float (numatom, nstep)
          
        '''
        # 長さが等しいかのテスト
        if not self.nstep == len(charge_list):
            print("ERROR :: steps of 2 files differ")
            print("steps for atoms_list :: ", len(self.ATOMS_LIST))
            print("steps for charge_list :: ", len(charge_list))
        if not len(self.ATOMS_LIST[0].get_chemical_symbols()) == len(charge_list[0]):
            print("ERROR :: # of atoms differ")
            print("# of atoms for atoms_list :: ", len(self.ATOMS_LIST[0].get_chemical_symbols()))
            print("# of atoms for charge_list :: ", len(charge_list[0]))
            
        for i in range(len(self.ATOMS_LIST)):
            self.ATOMS_LIST[i].set_initial_charges(charge_list[i])
        return self.ATOMS_LIST[i]
 
    def nglview_traj(self):
        return raw_nglview_traj(self.ATOMS_LIST)
        
    def save(self, prefix:str=""):
        if prefix == "":
            print("ERROR :: No prefix !!")
            sys.exit()
        cpmd.read_traj.raw_save_aseatoms(self.ATOMS_LIST, xyz_filename=prefix+"_refine.xyz")
        return 0

    def calc_dipole(self, mode="bec"):
        '''
        1 : mode=bec
        calculate dipoles from self.bec and ATOMS_LIST.
        dipole in Debye units. 

        2 : mode=scalar
        calculate dipoles from ATOMS_LIST and scalar charge in it.
        dipole in Debye units. atoms_list must include charges.

        Notes
        ----------------
        aseでは長さがAngstrom,電荷は電子電荷で扱っており，一方でDebyeは
        3.33564×10−30 C·mで定義されているので，これを変換している．
        
        Debye   = 3.33564e-30
        charge  = 1.602176634e-19
        ang      = 1.0e-10
        coef    = ang*charge/Debye
        print(coef)
        
        基本的には
        1[Ang*e]=4.8032[Debye]となる．
        '''
        dipole_list=[]
        if   mode == "bec":
            if type(self.bec) is NoneType:
                print("ERROR :: self.bec = None.")
                sys.exit()
        
            for p in range(self.nstep):
                tmp_dipole=np.einsum("ijk, ij -> k",self.bec, self.ATOMS_LIST[p].get_positions())
                dipole_list.append(tmp_dipole)

        elif mode == "scalar":
            for i in range(self.nstep):
                tmp_dipole=np.einsum("i,ij -> j",self.ATOMS_LIST[i].get_initial_charges(),self.ATOMS_LIST[i].get_positions())
                dipole_list.append(tmp_dipole)
                #
        else:
            pint("ERRIR :: incorrect mode")
        self.dipole=np.array(dipole_list)/ase.units.Debye
        return self.dipole


# ----------------------------------------
# raw functions used in custom_traj
#
#

def raw_nglview_traj(traj):
    '''
    asetrajをnglviewに渡すラッパー関数. asetrajはase.atomsのリストにいくつかのメソッドやプロパティを追加したものであり，show_asetrajは単なるase.atomsのリストも受け付けてくれる．
    従って，nglviewのためにase.trajを利用する必要はなくなった．
    '''
    view=nglview.show_asetraj(traj)

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
    return view
