

import numpy as np

class atom_type():
    '''
    各種の力場で使われている原子種と，その説明を入れる．
    '''
    def __init__(self,atom,description):
        self.atom = atom
        self.description = description

        
class gaff_atom_type():
    '''
    GAFFで定義されているatom typeを定義する．定義については
    https://ambermd.org/antechamber/gaff.html#atomtype
    を参照すること．
    '''
    atomlist = {
        "hc":atom_type("H","H on aliphatic C"),
        "ha":atom_type("H","H on aromatic C"),
        "hn":atom_type("H","H on N"),
        "ho":atom_type("H","H on O"),
        "hs":atom_type("H","H on S"),
        "hp":atom_type("H","H on P"),
        "o":atom_type("O","sp2 O in C=O, COO-"),
        "oh":atom_type("O","sp3 O in hydroxyl group"),
        "os":atom_type("O","sp3 O in ether and ester"),
        "c":atom_type("C","sp2 C in C=O, C=S"),
        "c1":atom_type("C","sp1 C"), 
        "c2":atom_type("C","sp2 C, aliphatic"),
        "c3":atom_type("C","sp3 C"),
        "ca":atom_type("C","sp2 C, aromatic"),
        "n": atom_type("N","sp2 N in amide"),
        "n1":atom_type("N","sp1 N "),
        "n2":atom_type("N","sp2 N with 2 subst."),
        "n3":atom_type("N","sp3 N with 3 subst."), # readl double bond  ?
        "n4":atom_type("N","sp3 N with 4 subst."), 
        "na":atom_type("N","sp2 N with 3 subst. "), 
        "nh":atom_type("N","amine N connected to the aromatic rings."), 
        "no":atom_type("N","N in nitro group."), 
        "s2":atom_type("S", "sp2 S (p=S, C=S etc)"),
        "sh":atom_type("S","sp2 S (p=S, C=S etc)"),
        "ss":atom_type("S","sp2 S (p=S, C=S etc)"),
        "s4":atom_type("S","sp2 S (p=S, C=S etc)"),
        "s6":atom_type("S","sp2 S (p=S, C=S etc)"),
        # 以下 urlでspecial atom typeと言われているもの
        "h1":atom_type("H","H on aliphatic C with 1 EW group"),
        "h2":atom_type("H","H on aliphatic C with 2 EW group"),
        "h3":atom_type("H","H on aliphatic C with 3 EW group"),
        "h4":atom_type("H","H on aliphatic C with 4 EW group"),
        "h5":atom_type("H","H on aliphatic C with 5 EW group"),
        "n":atom_type("N","aromatic nitrogen"),
        "nb":atom_type("N","inner sp2 N in conj. ring systems"),
        "nc":atom_type("N","inner sp2 N in conj. chain systems"),
        "nd":atom_type("N","inner sp2 N in conj. chain systems"),
        "sx":atom_type("S","conj. S, 3 subst."),
        "sy":atom_type("S","conj. S, 4 subst."), 
        "cc":atom_type("C","inner sp2 C in conj. ring systems"),
        "cd":atom_type("C","inner sp2 C in conj. ring systems"), 
        "ce":atom_type("C","inner sp2 C in conj. chain systems"), 
        "cf":atom_type("C","inner sp2 C in conj. chain systems"),  
        "cp":atom_type("C","bridge aromatic C"),  
        "cq":atom_type("C","bridge aromatic C"),   
        "cu":atom_type("C","sp2 C in three-memberred rings"), 
        "cv":atom_type("C","sp2 C in four-memberred rings "),
        "cx":atom_type("C","sp3 C in three-memberred rings"),
        "cy":atom_type("C","sp3 C in four-memberred rings "),
        "pb":atom_type("P","aromatic phosphorus"),
        "pc":atom_type("P","inner sp2 P in conj. ring systems "),
        "pd":atom_type("P","inner sp2 P in conj. ring systems "),
        "pe":atom_type("P","inner sp2 P in conj. chain systems"),
        "pf":atom_type("P","inner sp2 P in conj. chain systems"),
        "px":atom_type("P","conj. P, 3 subst."), 
        "py":atom_type("P","conj. P, 4 subst."),
    }
    


class read_itp():
    '''
    トポロジーファイル：itpの読み込み
    input
    ---------------
     filename :: itpファイルの名前

    output
    ---------------
     self.bonds_list :: ボンドの一覧
     self.num_atoms_per_mol :: 分子に含まれる原子数
     self.atomic_type :: GAFFで定義されている原子のタイプ
     self.atom_list :: 原子の種類（H,C,など）
     self.ch_bond
     self.co_bond
     self.cc_bond
     self.ch_bond
     self.oo_bond
     self.ring_bond
     self.o :: 原子のindex(O及びNはlone pairがあるのでそれ用)

    usage
    ---------------
    example 1) toluene
    import ml.atomtype
    tol_data=ml.atomtype.read_itp("input1.itp")
    tol_data.ch_bond # ch_bond7個を含むリストが得られる．
    
    note
    ---------------
    現状GAFF力場にのみ対応している．
    現状C,H,Oのみ実装して，P，N，Sについてはまだ実装していないので注意！！
    
    '''
    def __init__(self,filename):
        with open(filename) as f:
            lines = f.read().splitlines()
        lines = [l.split() for l in lines]

        # * ボンドの情報を読み込む．
        for i,l in enumerate(lines) :
            if "bonds" in l :
                indx = i
        # 
        bonds_list = []
        bi = 0
        while len(lines[indx+2+bi]) > 5: # bondsを見つけてから，空行へ行くまで．カラムが6以上ならば読み込む．
            p = int(lines[indx+2+bi][0])-1
            q = int(lines[indx+2+bi][1])-1
            bonds_list.append([p,q])
            bi = bi+1
        self.bonds_list=bonds_list

        # * 原子数を読み込む
        for i,l in enumerate(lines):
            if "atoms" in l:
                indx = i
            counter=0
        while len(lines[indx+2+counter]) > 5: # bondsを見つけてから，空行へ行くまで．カラムが6以上ならば読み込む．
            counter = counter+1
        #１つの分子内の総原子数
        self.num_atoms_per_mol = counter

        # * 原子タイプを読み込む
        atomic_type = []
        for i,l in enumerate(lines) :
            if "atoms" in l :
                indx = i
        counter=0
        while len(lines[indx+2+counter]) > 5: # bondsを見つけてから，空行へ行くまで．カラムが6以上ならば読み込む．
            atomic_type.append(lines[indx+2+counter][1]) 
            counter = counter+1
        self.atomic_type = atomic_type

        # * 原子種を割り当てる．
        atom_list = []
        for i in atomic_type:
            atom_list.append(gaff_atom_type.atomlist[i].atom)
        self.atom_list = atom_list
            
        print(" -----  ml.read_itp  :: parse results... -------")
        print(" bonds_list :: ", self.bonds_list)
        print(" counter    :: ", self.num_atoms_per_mol)
        print(" atomic_type:: ", self.atomic_type)
        print(" atom_list  :: ", self.atom_list)
        print(" -----------------------------------------------")
        
        # bond情報の取得
        self._get_bonds()
        # O/N lonepair情報の取得
        self._get_atomic_index()
        
        # 分子を表現するための原子のindexを指定
        # TODO :: itpファイルからこれを計算する部分を実装したい．
        self.representative_atom_index = 0
        


    def _get_bonds(self):
        '''
        self.bonds_listの中からch_bondsだけを取り出す．
        TODO :: hard code :: GAFFのみに対応している．
        '''
        ch_bond=[]
        co_bond=[]
        oh_bond=[]
        oo_bond=[]
        cc_bond=[]
        ring_bond=[] # これがベンゼン環
        for bond in self.bonds_list:
            # 原子タイプに変換
            tmp_type=[self.atomic_type[bond[0]],self.atomic_type[bond[1]]]
            # 原子種に変換
            tmp=[gaff_atom_type.atomlist[self.atomic_type[bond[0]]].atom,gaff_atom_type.atomlist[self.atomic_type[bond[1]]].atom]
            if tmp == ["H","C"] or tmp == ["C","H"]:
                ch_bond.append(bond)
            if tmp == ["O","C"] or tmp == ["C","O"]:
                co_bond.append(bond)
            if tmp == ["O","H"] or tmp == ["H","O"]:
                oh_bond.append(bond)
            if tmp == ["O","O"]:
                oo_bond.append(bond)
            if tmp == ["C","C"]: # CC結合はベンゼンとそれ以外で分ける
                if tmp_type != ["ca","ca"]: # ベンゼン環以外
                    cc_bond.append(bond)
                if tmp_type == ["ca","ca"]: # ベンゼン
                    ring_bond.append(bond)

        # TODO :: ベンゼン環は複数のリングに分解する．
        # この時，ナフタレンのようなことを考えると，完全には繋がっていない部分で分割するのが良い．
        # divide_cc_ring(ring_bond)
                    
        self.ch_bond=ch_bond
        self.co_bond=co_bond
        self.oh_bond=oh_bond
        self.oo_bond=oo_bond
        self.cc_bond=cc_bond
        self.ring_bond=ring_bond
        
        if len(ch_bond)+len(co_bond)+len(oh_bond)+len(oo_bond)+len(cc_bond)+len(ring_bond) != len(self.bonds_list):
                print(" ")
                print(" WARNING :: There are unkown bonds in self.bonds_list... ")
                print(" ")
        
        print(" ================ ")
        print(" CH bonds...      ",self.ch_bond)
        print(" CO bonds...      ",self.co_bond)
        print(" OH bonds...      ",self.oh_bond)
        print(" OO bonds...      ",self.oo_bond)
        print(" CC bonds...      ",self.cc_bond)
        print(" CC ring bonds... ",self.ring_bond)
        print(" ")
        
        # さらに，ボンドペアのリストをボンドインデックスに変換する
        # 実際のボンド[a,b]から，ボンド番号（bonds.index）への変換を行う
        self.ring_bond_index=raw_convert_bondpair_to_bondindex(ring_bond,self.bonds_list)
        self.ch_bond_index=raw_convert_bondpair_to_bondindex(ch_bond,self.bonds_list)
        self.co_bond_index=raw_convert_bondpair_to_bondindex(co_bond,self.bonds_list)
        self.oh_bond_index=raw_convert_bondpair_to_bondindex(oh_bond,self.bonds_list)
        self.oo_bond_index=raw_convert_bondpair_to_bondindex(oo_bond,self.bonds_list)
        self.cc_bond_index=raw_convert_bondpair_to_bondindex(cc_bond,self.bonds_list)
        
        print("")
        print(" ================== ")
        print(" ring_bond_index ", self.ring_bond_index)
        print(" ch_bond_index   ", self.ch_bond_index)
        print(" oh_bond_index   ", self.oh_bond_index)
        print(" co_bond_index   ", self.co_bond_index)
        print(" cc_bond_index   ", self.cc_bond_index)
        return 0

    def divide_cc_ring(self):
        '''
        TODO :: ccリングを繋がっている部分とそれ以外に分割する．
        '''
        return 0

    def _get_atomic_index(self):
        '''
        self.atom_listからO原子やN原子などのlonepairがある原子を見つけて，そのindexを返す．

        chemicalsymbol :"O"や"N"などの原子種
        '''
        self.o_list = [i for i, x in enumerate(self.atom_list) if x == "O"]
        self.n_list = [i for i, x in enumerate(self.atom_list) if x == "N"]
        print(" ================ ")
        print(" O atoms (lonepair)...      ",self.o_list)
        print(" N atoms (lonepair)...      ",self.n_list)
        return 0
        
def raw_convert_bondpair_to_bondindex(bonds,bonds_list):
        bond_index   = []
        # 実際のボンド[a,b]から，ボンド番号（bonds.index）への変換を行う
        for b in bonds :
            if b in bonds_list : #ボンドがリストに存在する場合
                bond_index.append(bonds_list.index(b))
            elif b[::-1] in bonds : # これはボンドの向きが逆の場合（b[1],b[0]）
                bond_index.append(bonds_list.index(b[::-1])) 
            else :
                print("there is no bond{} in bonds list.".format(b))
        return bond_index

class read_mol():
    '''
    山崎さんからの提案でrdkitを使って試す
    https://future-chem.com/rdkit-mol/
    '''
    def __init__(self,filename):
        #bond-listの作成(RDkit版)
        from rdkit import rdBase, Chem
        from rdkit.Chem import AllChem, Draw
        from rdkit.Chem.Draw import rdMolDraw2D

        # commands="obabel -igro {0}.gro -omol > {0}.mol".format(name)
        # proc = subprocess.run(commands, shell=True, stdout=PIPE, stderr=PIPE,encoding='utf-8')
        # output = proc.stdout
        mol_rdkit = Chem.MolFromMolFile(filename,sanitize=False,removeHs=False)
        #念の為、分子のケクレ化を施す
        Chem.Kekulize(mol_rdkit)
        self.mol_rdkit=mol_rdkit # 外部から制御できるように！（主にデバッグ用）
        # 原子数の取得
        self.num_atoms_per_mol=mol_rdkit.GetNumAtoms() 
        # atom list（原子番号）
        self.atom_list=[]
        for atom in mol_rdkit.GetAtoms():
            self.atom_list.append(atom.GetSymbol())
        
        #bonds_listの作成
        self.bonds_list=[]
        self.double_bonds=[]
        
        for i,b in enumerate(mol_rdkit.GetBonds()):
            indx0 = b.GetBeginAtomIdx()
            indx1 = b.GetEndAtomIdx()
            bond_type = b.GetBondType()
        
            self.bonds_list.append([indx0,indx1])
            if str(bond_type) == "DOUBLE" :
                self.double_bonds.append(i)
        print(" -----  ml.read_mol :: parse results... -------")
        print(" bonds_list :: ", self.bonds_list)
        print(" counter    :: ", self.num_atoms_per_mol)
        # print(" atomic_type:: ", self.atomic_type)
        print(" atom_list  :: ", self.atom_list)
        print(" -----------------------------------------------")
        
        # bond情報の取得
        self._get_bonds()
        # O/N lonepair情報の取得
        self._get_atomic_index()
        
        # 分子を表現するための原子のindexを指定
        # TODO :: itpファイルからこれを計算する部分を実装したい．
        # TODO :: ここはrdkitを使えばなんとかなるはず．
        self.representative_atom_index = 4
        self.representative_atom_index = self._find_representative_atom_index()
        print(" -----  ml.read_mol :: parse results... -------")
        print(" representative_atom_index  :: {}".format(self.representative_atom_index))
        print(" -----------------------------------------------")
    
    def _get_bonds(self):
        ch_bond=[]
        co_bond=[]
        oh_bond=[]
        oo_bond=[]
        cc_bond=[]
        ring_bond=[] # これがベンゼン環
        for bond in self.bonds_list:
            # 原子番号に変換
            tmp=[self.atom_list[bond[0]],self.atom_list[bond[1]]]
            
            if tmp == ["H","C"] or tmp == ["C","H"]:
                ch_bond.append(bond)
            if tmp == ["O","C"] or tmp == ["C","O"]:
                co_bond.append(bond)
            if tmp == ["O","H"] or tmp == ["H","O"]:
                oh_bond.append(bond)
            if tmp == ["O","O"]:
                oo_bond.append(bond)
            if tmp == ["C","C"]: # TODO :: ここは本来は二重結合の検出が必要
                if self.mol_rdkit.GetAtoms()[bond[0]].GetIsAromatic() == True & self.mol_rdkit.GetAtoms()[bond[1]].GetIsAromatic() == True: 
                    ring_bond.append(bond) # ベンゼン環が複数ある場合には未対応
                else:
                    cc_bond.append(bond) # 芳香族以外のみ検出
        # TODO :: ベンゼン環は複数のリングに分解する．
        # この時，ナフタレンのようなことを考えると，完全には繋がっていない部分で分割するのが良い．
        # divide_cc_ring(ring_bond)
        self.ch_bond=ch_bond
        self.co_bond=co_bond
        self.oh_bond=oh_bond
        self.oo_bond=oo_bond
        self.cc_bond=cc_bond
        self.ring_bond=ring_bond
        
        if len(ch_bond)+len(co_bond)+len(oh_bond)+len(oo_bond)+len(cc_bond)+len(ring_bond) != len(self.bonds_list):
            print(" ")
            print(" WARNING :: There are unkown bonds in self.bonds_list... ")
            print(" ")
        
        print(" ================ ")
        print(" CH bonds...      ",self.ch_bond)
        print(" CO bonds...      ",self.co_bond)
        print(" OH bonds...      ",self.oh_bond)
        print(" OO bonds...      ",self.oo_bond)
        print(" CC bonds...      ",self.cc_bond)
        print(" CC ring bonds... ",self.ring_bond)
        print(" ")
        
        # さらに，ボンドペアのリストをボンドインデックスに変換する
        # 実際のボンド[a,b]から，ボンド番号（bonds.index）への変換を行う
        self.ring_bond_index=raw_convert_bondpair_to_bondindex(ring_bond,self.bonds_list)
        self.ch_bond_index=raw_convert_bondpair_to_bondindex(ch_bond,self.bonds_list)
        self.co_bond_index=raw_convert_bondpair_to_bondindex(co_bond,self.bonds_list)
        self.oh_bond_index=raw_convert_bondpair_to_bondindex(oh_bond,self.bonds_list)
        self.oo_bond_index=raw_convert_bondpair_to_bondindex(oo_bond,self.bonds_list)
        self.cc_bond_index=raw_convert_bondpair_to_bondindex(cc_bond,self.bonds_list)
        
        print("")
        print(" ================== ")
        print(" ring_bond_index ", self.ring_bond_index)
        print(" ch_bond_index   ", self.ch_bond_index)
        print(" oh_bond_index   ", self.oh_bond_index)
        print(" co_bond_index   ", self.co_bond_index)
        print(" cc_bond_index   ", self.cc_bond_index)
        return 0
    
    def _get_atomic_index(self):
        '''
        self.atom_listからO原子やN原子などのlonepairがある原子を見つけて，そのindexを返す．
        chemicalsymbol :"O"や"N"などの原子種
        '''
        self.o_list = [i for i, x in enumerate(self.atom_list) if x == "O"]
        self.n_list = [i for i, x in enumerate(self.atom_list) if x == "N"]
        self.c_list = [i for i, x in enumerate(self.atom_list) if x == "C"]
        self.h_list = [i for i, x in enumerate(self.atom_list) if x == "H"]
        print(" ================ ")
        print(" O atoms (lonepair)...      ",self.o_list)
        print(" N atoms (lonepair)...      ",self.n_list)
        print(" C atoms ...                ",self.c_list)
        print(" H atoms ...                ",self.h_list)
        
        return 0
    
    def raw_convert_bondpair_to_bondindex(bonds,bonds_list):
        '''
        実際のボンド[a,b]から，ボンド番号（bonds.index）への変換を行う
        '''
        bond_index   = []
        for b in bonds :
            if b in bonds_list :
                bond_index.append(bonds_list.index(b))
            elif b[::-1] in bonds :
                bond_index.append(bonds_list.index(b[::-1])) 
            else :
                print("there is no bond{} in bonds list.".format(b))
        return bond_index
    
    def _find_representative_atom_index(self):
        '''
        読み込んだ座標からH以外の骨格だけを取り出し，その単純重心に最も近い原子のindexを返す
        https://stackoverflow.com/questions/71915443/rdkit-coordinates-for-atoms-in-a-molecule
        '''
        positions_skelton = []
        index_tmp = []
        for i, atom in enumerate(self.mol_rdkit.GetAtoms()):
            positions = self.mol_rdkit.GetConformer().GetAtomPosition(i)
            # print(atom.GetSymbol(), positions.x, positions.y, positions.z)
            if atom.GetSymbol() != "H": # H以外の原子のみを取り出す
                print(atom.GetSymbol(), positions.x, positions.y, positions.z)
                positions_skelton.append(np.array([positions.x, positions.y, positions.z]))
                index_tmp.append(i)
        # 平均値を求める
        positions_skelton=np.array(positions_skelton)
        positions_mean = np.mean(positions_skelton, axis=0)
        # positions_meanに一番近い原子を探す
        distance = np.linalg.norm(positions_skelton - positions_mean,axis=1)
        # print(distance)
        # 最小のindexを与える原子のindexを返す
        # print(index_tmp[np.argmin(distance)])
        return index_tmp[np.argmin(distance)]

class Node: # 分子情報（itp）をグラフ情報に格納するためのクラス
    """

    ノードの情報（分子の情報）を管理

    Attributes:
        index (int): 自分のノード番号（aseatomsでの番号）
        nears (list): 隣接リスト（bond_listに相当）
        parent (int): 親のノード番号（）
    """

    def __init__(self, index):
        self.index = index
        self.nears = []
        self.parent = -1

    def __repr__(self):
        return f"(index:{self.index}, nears:{self.nears}, parent:{self.parent})"


def raw_make_graph_from_itp(itp_data):
    '''
    itp_dataからグラフを作成して返す
    itp_data.bonds_list：itp_dataに定義されたボンドリスト
    itp_data.num_atoms_per_mol：分子内の原子数
    参考：https://qiita.com/keisuke-ota/items/6c1b4846b82f548b5dec
    '''
    # Nodeインスタンスを作成しnodesに格納
    nodes = [Node(i) for i in range(itp_data.num_atoms_per_mol)]
    
    # 隣接リストを付与
    for bond in itp_data.bonds_list:
        nodes[bond[0]].nears.append(bond[1])
        nodes[bond[1]].nears.append(bond[0])
    
    return nodes