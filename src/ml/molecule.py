

class bondinfo():
    '''
    ボンドの情報をもつクラス．基本的にはmoleculeクラスの中で使うことを想定して作成してある．
    特にペアの情報はmoleculeクラスと同時に使わないと意味がない．
    ボンドのindexの他に，対応するWCsの情報，ボンドセンターの座標を持つようにする．
    ボンドセンターは計算することもできるのだが，保持しておいた方が何かと楽なので．
    
    input
    -----------------------
       pair :: ペアの番号（これはsupercellの全原子に対する番号？）
       wcs  :: wcsの座標（トラジェクトリだけの場合もあり得るので，wcs == noneでも動くように．）
       bc   :: ボンドセンターの座標（これは後から計算させても良い気もするが．．．）
    '''
    def __init__(self,pair:list,bc,wcs:list):
        self.pair=pair
        self.wcs=wcs
        self.bc=bc

    def get_bond_dipole(self):
        '''
        ボンドの双極子を求める．
        単結合か二重結合かによって式が違うので要注意．
        '''
        if len(self.wcs) == 1: # 単結合の場合
            return (-2.0)*(self.wcs[0]-self.bc)
        if len(self.wcs) == 2: # 二重結合の場合
            return (-4.0)*((self.wcs[0]+self.wcs[1])/2-self.bc)
        if len(self.wcs) >= 3: # 三重結合の場合?
            print("WARGNING :: 三重結合")
            return 0
        
        

class lonepair():
    '''
    ローンペアの情報をもつクラス
    ローンペアのindexの他に，対応するWCsの情報を持っている．
    '''
    def __init__(self,atom:int,wcs:list):
        self.atom=atom
        if len(wcs) != 2:
            print("error :: # of wcs is not two.")
        self.wcs=wcs
    #
    def get_bond_dipole(self):
        '''
        ローンペアの双極子を求める．
        基本的にはローンペアにはWCsが二つあるという前提で話を進める．
        '''
        if len(self.wcs) == 2: # 二重結合の場合
            return (-4.0)*((self.wcs[0]+self.wcs[1])-self.bc)
        else:
            print("ERROR :: WCsが二つではありません")
            return 0
    
        
class molecule():
    '''

    -------------
    symbols :: 原子の種類（リスト）
    positions :: 原子座標（リスト）
    bonds     :: 原子の順番の番号に立脚したボンドリスト．
    lonepairs :: 原子の順番に立脚したローンペアリスト．
    bondsinfo :: WCsの情報を含んだボンド情報？
    loneinfo  :: lonepairの情報を含んだボンド情報？

    note
    -------------
     ase.atoms(https://wiki.fysik.dtu.dk/ase/_modules/ase/atoms.html#Atoms.get_positions)
    を参考にコードを作成している．

    ボンドリストはできればCHボンド，OHボンドなどのボンド種別で区別できるようにしたい．
    
    '''
    def __init__(self,symbols,positions,bonds:list,lonepairs:list):
        # 実態であるself.arraysを定義
        # この中に辞書形式でさまざまなプロパティを入れる．
        self.array={}
        #
        self.array["symbols"]=symbols
        self.array["positions"]=positions
        self.array["bonds"]=bonds
        self.array["lonepairs"]=lonepairs

    def get_positions(self):
        return self.arrays['positions'].copy()
    def get_chemical_symbols(self):
        """Get list of chemical symbol strings.
        Equivalent to ``list(atoms.symbols)``."""
        return self.arrays["symbols"].copy()
    def get_chemical_bonds(self):
        """Get list of chemical bonds list."""
        return self.arrays["bonds"].copy()
    def get_lonepairs(self):
        """Get list of lonepairs."""
        return self.arrays["lonepairs"].copy()

    def get_molecule_diople(self):
        """
        bondsとlonepairの情報から分子の全dipoleモーメントを求める．
        """
        import numpy as np
        molecule_dipole=np.zeros(3)
        for bond in self.array["bonds"]:
            molecule_dipole+=bond.get_bond_dipole()
            

            
class mdconfig():
    '''
    moleculeクラスを複数ひとまとめにしたクラスをmdconfigクラスとして別途定義しておく．

    input
    -----------
      atomlist :: 原子の番号リスト？これがあると分子クラスがいくつかあった時にその分子間の順番を区別できるかも？
    
    '''
    def __init__(self,atomlist):
        self.atomlist = atomlist


