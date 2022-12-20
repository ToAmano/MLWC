


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

    usage
    ---------------
    example 1) toluene
    import ml.atomtype
    tol_data=ml.atomtype.read_itp("input1.itp")
    tol_data.ch_bond # ch_bond7個を含むリストが得られる．
    
    note
    ---------------
    現状GAFF力場にのみ対応している．
      
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
        for i,l in enumerate(lines) :
            if "atoms" in l :
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

        return 0

    def divide_cc_ring(self):
        '''
        TODO :: ccリングを繋がっている部分とそれ以外に分割する．
        '''
        return 0
