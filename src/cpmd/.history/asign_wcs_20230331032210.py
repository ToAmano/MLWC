    
from ase.io import read
import ase
import sys
import numpy as np
# from types import NoneType

try:
    import nglview
except ImportError:
    sys.exit ('Error: nglview not installed')



def make_ase_with_WCs(list_atomic_nums,UNITCELL_VECTORS,list_mol_coords,list_bond_centers,list_bond_wfcs,list_dbond_wfcs,list_lpO_wfcs,list_lpN_wfcs):
    '''
    元の分子座標に加えて，WCsとボンドセンターを加えたase.atomsを作成する．
    '''
    # list_mol_coords,list_bond_centers =results
    # list_bond_wfcs,list_dbond_wfcs,list_lpO_wfcs,list_lpN_wfcs = results_wfcs

    new_coord = []
    new_atomic_num = []

    # 原子をnew_coordへappendする
    for mol_r,mol_at in zip(list_mol_coords,list_atomic_nums) :
        for r,at in zip(mol_r,mol_at) :
            new_atomic_num.append(at)
            new_coord.append(r)

    # ボンド中心及びボンドwfをnew_coordへappendする
    for mol_wc,mol_bc in zip(list_bond_wfcs,list_bond_centers) :
        for bond_wc,bond_bc in zip(mol_wc,mol_bc) :
            new_coord.append(bond_bc)
            new_atomic_num.append(2)  #ボンド？（原子番号2：Heを割り当て）
            for wc in bond_wc :
                new_coord.append(wc)
                new_atomic_num.append(10) # ワニエセンター（原子番号10：Neを割り当て）

    # print("new_coord (include bond center) ::", len(new_coord))            
    for mol_wc in list_dbond_wfcs :
        for dbond_wc in mol_wc :
            for wc in dbond_wc :
                new_coord.append(wc)
                new_atomic_num.append(10) # ワニエセンター（原子番号10：Neを割り当て）           

    # Oのローンペア
    for mol_lp in list_lpO_wfcs :
        for lp_wc in mol_lp :    
            for wc in lp_wc :
                new_coord.append(wc)
                new_atomic_num.append(10)

    # Nのローンペア
    for mol_lp in list_lpN_wfcs :
        for lp_wc in mol_lp :
            for wc in lp_wc :
                new_coord.append(wc)
                new_atomic_num.append(10)

    # change to numpy
    new_coord = np.array(new_coord)

    #WFCsと原子を合体させたAtomsオブジェクトを作成する．
    from ase import Atoms
    aseatoms_with_WC = Atoms(new_atomic_num,
        positions=new_coord,
        cell= UNITCELL_VECTORS,
        pbc=[1, 1, 1])
    return aseatoms_with_WC


class asign_wcs:
    import ase
    '''
    関数をメソッドとしてこちらにうつしていく．
    その際，基本となる変数をinitで定義する
    '''
    def __init__(self, NUM_MOL:int, NUM_MOL_ATOMS:int, UNITCELL_VECTORS):
        self.NUM_MOL       = NUM_MOL
        self.NUM_MOL_ATOMS = NUM_MOL_ATOMS
        self.UNITCELL_VECTORS = UNITCELL_VECTORS

    def aseatom_to_mol_coord_bc(self, ase_atoms:ase.atoms, unit_cell_bonds:list): # ase_atomsのボンドセンターを計算する
        return raw_aseatom_to_mol_coord_bc(ase_atoms, unit_cell_bonds, self.NUM_MOL_ATOMS, self.NUM_MOL)

    def make_aseatoms_from_wc(self, atom_coord:np.array,wfc_list): # wcsからase.atomsを作る
        return raw_make_aseatoms_from_wc(atom_coord,wfc_list,self.UNITCELL_VECTORS)

    def find_all_lonepairs(self, wfc_list,atO_list,list_mol_coords,picked_wfcs,wcs_num:int):
        return raw_find_all_lonepairs(wfc_list,atO_list,list_mol_coords,picked_wfcs,wcs_num:int,self.UNITCELL_VECTORS):
    # TODO :: ここはボンドごとに，例えばfind_all_chbondsのようにしたい
    def find_all_bonds(self, wfc_list,list_bond_centers,picked_wfcs): 
        return raw_find_all_bonds(wfc_list,list_bond_centers,picked_wfcs,self.UNITCELL_VECTORS)

    def raw_find_all_pi(self, wfc_list,list_bond_centers,picked_wfcs,double_bonds):
        return raw_find_all_pi(wfc_list,list_bond_centers,picked_wfcs,double_bonds,self.UNITCELL_VECTORS)
    
    
def raw_aseatom_to_mol_coord_bc(ase_atoms, unit_cell_bonds, NUM_MOL_ATOMS:int, NUM_MOL:int) :
    '''
    CPMDの結果のanswer_atomslistからconfig番目のconfigを読み込む．
    読み込んで，
     - 1: ボンドセンターの計算
     - 2: micを考慮した原子座標の再計算
    を行う．基本的にはcalc_mol_coordのwrapper関数
    
    input
    ------------
    ase_atoms       :: ase.atoms
    mol_ats         ::
    unit_cell_bonds ::

    output
    ------------
    list_mol_coords :: 
    list_bond_centers
    '''

    list_mol_coords=[] #分子の各原子の座標の格納用
    list_bond_centers=[] #各分子の化学結合の中点の座標リストの格納用

    # * 分子を構成する原子のインデックスのリストを作成する。（mol_at0をNUM_MOL回繰り返す）
    mol_at0 = [ i for i in range(NUM_MOL_ATOMS) ]
    mol_ats = [ [ int(at+NUM_MOL_ATOMS*indx) for at in mol_at0 ] for indx in range(NUM_MOL)]
    # for indx in range(NUM_MOL) :
    #    mol_ats.append([ int(at+NUM_MOL_ATOMS*indx) for at in mol_at0 ])

    for j in range(NUM_MOL): # 全ての分子に対する繰り返し．
        mol_inds=mol_ats[j]   # j番目の分子に入っている全ての原子のindex
        bonds_list=unit_cell_bonds[j]  # j番目の分子に入っている全ての原子のindex      
        mol_coords,bond_centers = raw_calc_mol_coord_and_bc_mic_onemolecule(mol_inds,bonds_list,ase_atoms) 
        list_mol_coords.append(mol_coords)
        list_bond_centers.append(bond_centers)

    # print(" ")
    # print(" PARSE RESULT ")
    # print(" 分子数 :: ", len(list_mol_coords))
    # print(" 分子あたり原子数 :: ", len(list_mol_coords[0]))
    # print(" 分子数（ボンド） :: ", len(list_bond_centers))
    # print(" 分子あたりボンド数 :: ", len(list_bond_centers[0]))
    return  [list_mol_coords,list_bond_centers]

def raw_calc_mol_coord_and_bc_mic_onemolecule(mol_inds,bonds_list,aseatoms) :
    '''
        # * 系内のあるひとつの分子に着目し，ボンドセンターと（micを考慮した）分子座標を計算する．
        inputs
        ----------------
        mol_inds # list :: 分子を構成する原子のindexのリスト (先頭原子が分子の始点となる)
        bonds_list # list[list]] :: 分子中の各結合の原子indexのリスト。
        mol_inds[0] :: 分子の先頭の原子．これを基準とするコードになってる．

        output
        -----------------
        # mol_coords # numpy array 分子を構成する原子の座標（MIC考慮）
        # bond_centers # numpy array 各結合の中点座標のリスト
    '''
    # vectors = aseatoms.get_all_distances(mic=True,vector=True)  #GromacsでPBC=Molの場合は、mic=Falseとする
    vectors = aseatoms.get_distances(mol_inds[0], mol_inds, mic=True, vector=True) # 必要なベクトルだけを求めることもできる．
    coords  = aseatoms.get_positions()
    
    # 分子内の原子の座標をR0基準に再計算
    R0    = coords[mol_inds[0]] # 最初の原子の座標
    
    # mol_indsとbonds_listを0始まりのインデックスで書き直す．
    # TODO :: hard code :: とりあえずの処置として，全てのインデックスの番号をずらすやり方をとる．
    # TODO :: これだと将来的に原子番号が綺麗な順番になっていない場合に対応できない．
    mol_inds_from_zero=[i-mol_inds[0] for i in mol_inds]
    bonds_list_from_zero=[[i[0]-mol_inds[0],i[1]-mol_inds[0]] for i in bonds_list]
    
    # 全ての原子（分子に含まれる）の座標を取得する．
    mol_coords=[R0+vectors[k] for k in mol_inds_from_zero ]
    # for k in mol_inds_from_zero: # 古いコード
    #     mol_coords.append(R0+vectors[k])
    
    # 全てのボンドセンターの座標を取得する．
    bond_centers = []
    # bond_infos = []
    for l in bonds_list_from_zero :
        # 二つのdrがボンドの両端の原子への距離
        bc = R0+(vectors[l[0]]+vectors[l[1]])/2.0 # R0にボンドセンターへの座標をたす．
        bond_centers.append(bc) 
        # bond_infos.append(molecule.bondinfo(pair=l,bc=bc, wcs=[]))
    return np.array(mol_coords), np.array(bond_centers)



# * find_lonepair/find_bondwcs/find_piの補助関数として，
# * 原子一つとWCsたちのase.atomsを作るmake_aseatomsを定義した
def raw_make_aseatoms_from_wc(atom_coord:np.array,wfc_list,UNITCELL_VECTORS):
    import ase.atoms
    #原子座標(atom_coord)を先頭においたAtomsオブジェクトを作成する
    atom_wcs_coord=np.array(list([atom_coord,])+list(wfc_list))
    num_element=len(atom_wcs_coord)    

    #ワニエ中心のラベルはAuとする
    elements = {"Au":79}
    atom_id= ["Au",]*num_element
    atom_id = [elements[i] for i in atom_id ]

    atom_wan = ase.Atoms(atom_id,
               positions=atom_wcs_coord,        
               cell= UNITCELL_VECTORS,   
               pbc=[1, 1, 1]) 
    return atom_wan

def raw_find_lonepairs(atom_coord:np.array,wfc_list,wcs_num:int,UNITCELL_VECTORS):
    '''
    ローンペアまたはボンドセンターから最も近いwcsを探索する
    input
    -------------
    wfc_list   :: WCsの座標リスト
    atom_coord :: 原子の座標（np.array）
    wcs_num    :: 見つけるwcsの数．1（N lonepair）か2（O lonepair）
    picked_wcs :: これはoption
    
    output
    -------------
    wcs_indices :: 答えのワニエのindex
    mu_lp :: 計算された双極子
    '''
    if wcs_num != 1 and wcs_num != 2:
        print("ERROR :: wcs_num should be 1 or 2 !!")
        print("wfc_num = ", wcs_num)
        return -1

    #選択した原子座標を先頭においたAtomsオブジェクトを作成する
    atom_wan=raw_make_aseatoms_from_wc(atom_coord,wfc_list,UNITCELL_VECTORS)
    num_element=len(wfc_list)+1 # 1はatom_coordの分

    # 先頭原子とWCsの距離ベクトル
    # TODO ここのmicをかける計算をaseに頼らず実行できるか？
    wfc_vectors = atom_wan.get_distances(0,range(1,num_element),mic=True,vector=True) #ワニエ中心は一度CP.xを通しているから点ごとにPBCが適用されている。mic=Trueでよい。
    #wfc_distances=WC.get_distances(0,range(1,nbands+1),mic=True,vector=False)
    wfc_distances=np.linalg.norm(wfc_vectors,axis=1)
    if wcs_num == 1:
        wcs_indices = np.argsort(wfc_distances).reshape(-1)[:1]
        mu_lp = (-2.0)*coef*wfc_vectors[wcs_indices[0]]
    if wcs_num == 2:
        wcs_indices = np.argsort(wfc_distances).reshape(-1)[:2] # 最も近いWCsのインデックスを二つ取り出す．
        # 二つのWannierCenterによる双極子モーメントを計算する．
        mu_lp = (-4.0)*coef*(wfc_vectors[wcs_indices[0]]+wfc_vectors[wcs_indices[1]])/2.0
    # 最後にwcsの（micがかかった）座標を取得
    wcs_lp=[atom_wan.get_positions()[0]+wfc_vectors[i] for i in wcs_indices]
    return wcs_indices, mu_lp, wcs_lp


def raw_find_bondwcs(atom_coord:np.array,wfc_list,wcs_num:int,UNITCELL_VECTORS):
    '''
    1：ボンドセンターから最も近いwcsを探索する
    input
    -------------
    wfc_list :: WCsの座標リスト
    atom_coord :: 原子の座標（np.array）
    wcs_num  :: 見つけるwcsの数．1（N lonepair）か2（O lonepair）
    
    output
    -------------
    wcs_indices :: 答えのワニエのindex
    mu_lp :: 計算された双極子
    '''
    import ase
    if wcs_num != 1 and wcs_num != 2:
        print("ERROR :: wcs_num should be 1 or 2 !!")
        print("wfc_num = ", wcs_num)
        return -1

    #選択した原子座標を先頭においたAtomsオブジェクトを作成する
    atom_wan=raw_make_aseatoms_from_wc(atom_coord,wfc_list,UNITCELL_VECTORS)
    num_element=len(wfc_list)+1 # 1はatom_coordの分

    # 先頭原子とWCsの距離ベクトル
    # TODO ここのmicをかける計算をaseに頼らず実行できるか？
    wfc_vectors = atom_wan.get_distances(0,range(1,num_element),mic=True,vector=True) #ワニエ中心は一度CP.xを通しているから点ごとにPBCが適用されている。mic=Trueでよい。
    #wfc_distances=WC.get_distances(0,range(1,nbands+1),mic=True,vector=False)
    wfc_distances=np.linalg.norm(wfc_vectors,axis=1)
    if wcs_num == 1:
        wcs_indices = np.argsort(wfc_distances).reshape(-1)[:1]
        mu_lp = (-2.0)*coef*wfc_vectors[wcs_indices[0]]
    if wcs_num == 2:
        wcs_indices = np.argsort(wfc_distances).reshape(-1)[:2] # 最も近いWCsのインデックスを二つ取り出す．
        # 二つのWannierCenterによる双極子モーメントを計算する．
        mu_lp = (-4.0)*coef*(wfc_vectors[wcs_indices[0]]+wfc_vectors[wcs_indices[1]])/2.0
    # 最後にwcsの（micがかかった）座標を取得
    wcs_bond=[atom_wan.get_positions()[0]+wfc_vectors[i] for i in wcs_indices]
    # for i in wcs_indices:
    #    wcs_bond.append(atom_wan.get_positions()[0]+wfc_vectors[i])
    return wcs_indices, mu_lp, wcs_bond

def raw_find_pi(atom_coord:np.array,wfc_list,r_threshold:float,picked_wcs,UNITCELL_VECTORS):
    '''
    find_lonepairから一行違うだけ！
    
    input
    -------------
    wfc_list :: WCsの座標リスト
    atom_coord :: 原子の座標（np.array）
    wcs_num  :: 見つけるwcsの数．1（N lonepair）か2（O lonepair）
    
    output
    -------------
    wcs_indices :: 答えのワニエのindex
    mu_lp :: 計算された双極子
    '''
    #選択した原子座標を先頭においたAtomsオブジェクトを作成する
    atom_wan=raw_make_aseatoms_from_wc(atom_coord,wfc_list,UNITCELL_VECTORS)
    num_element=len(wfc_list)+1 # 1はatom_coordの分

    # 先頭原子とWCsの距離ベクトル
    # TODO ここのmicをかける計算をaseに頼らず実行できるか？
    wfc_vectors = atom_wan.get_distances(0,range(1,num_element),mic=True,vector=True) #ワニエ中心は一度CP.xを通しているから点ごとにPBCが適用されている。mic=Trueでよい。
    #wfc_distances=WC.get_distances(0,range(1,nbands+1),mic=True,vector=False)
    wfc_distances=np.linalg.norm(wfc_vectors,axis=1)
    wcs_indices = np.argwhere(wfc_distances<r_threshold).reshape(-1) #まずは条件に合うのを抽出
    wcs_indices = [i for i in wcs_indices if i not in picked_wcs][:1]
    #[wfc_list[config][i] for i in range(len(wfc_list[config])) if i not in picked_wfcs]
    if len(wcs_indices) == 0:
        # print("WARNING :: ボンドに割り当てるWCがありません !! " )
        # print("WARNING :: この場合はreturnがNull Nullになります !!" )
        return None, None, None
    else:
        mu_lp = (-2.0)*coef*wfc_vectors[wcs_indices[0]]
        # 最後にwcsの（micがかかった）座標を取得
        wcs_dbond=[atom_wan.get_positions()[0]+wfc_vectors[i] for i in wcs_indices]
        # for i in wcs_indices:
        #    wcs_dbond.append(atom_wan.get_positions()[0]+wfc_vectors[i])
        return wcs_indices, mu_lp, wcs_dbond


# 
# * calc_mu_bond_lonepair用の部品関数たち
# 
def raw_find_all_lonepairs(wfc_list,atO_list,list_mol_coords,picked_wfcs,wcs_num:int,UNITCELL_VECTORS):
    '''
    最後の変数num_wcsでOローンペアとNローンペアに対応
    '''
    list_mu_lp   =  []
    list_lp_wfcs =  []
    
    if wcs_num != 1 and wcs_num != 2:
        print("ERROR :: wcs_num should be 1 or 2 !!")
        print("wfc_num = ", wcs_num)
        return -1
    
    # Oのローンペア
    for atOs,mol_coords in zip(atO_list,list_mol_coords) : #分子の数に関するループ
        mu_lpO_mol = []
        wcs_mol = [] # wcsの座標
        for atO in atOs : #ある分子内の原子に関するループ
            center_atom_coord  = mol_coords[atO]
            #wfc_list_exclude_pickedwfcs = [wfc_list[config][i] for i in range(len(wfc_list[config])) if i not in picked_wfcs]
            wcs_indices, mu_lp, wcs_lp = raw_find_lonepairs(center_atom_coord, wfc_list, wcs_num,UNITCELL_VECTORS)
            picked_wfcs        = picked_wfcs + list(wcs_indices)
            mu_lpO_mol.append(mu_lp)
            wcs_mol.append(wcs_lp)
        list_mu_lp.append(mu_lpO_mol)
        list_lp_wfcs.append(wcs_mol) 
    return np.array(list_mu_lp), np.array(list_lp_wfcs), picked_wfcs

def raw_find_all_bonds(wfc_list,list_bond_centers,picked_wfcs,UNITCELL_VECTORS):
    '''
    シングルボンドの部分を計算
    '''
    list_mu_bonds = []
    list_bond_wfcs = []

    for bcs in list_bond_centers :  # 分子数に関するループ
        mu_bonds_mol = []
        wcs_mol = []
        for bond_center_coord in bcs : # ある分子内のボンドセンターに関するループ
            # wfc_list_exclude_pickedwfcs = [wfc_list[config][i] for i in range(len(wfc_list[config])) if i not in picked_wfcs]
            wcs_indices, mu_bond,wcs_bond = raw_find_lonepairs(bond_center_coord, wfc_list, 1,UNITCELL_VECTORS)
            picked_wfcs        = picked_wfcs + list(wcs_indices)
            wcs_mol.append(wcs_bond)
            mu_bonds_mol.append(mu_bond)
        list_mu_bonds.append(mu_bonds_mol)
        list_bond_wfcs.append(wcs_mol)
    return np.array(list_mu_bonds), np.array(list_bond_wfcs), picked_wfcs

def raw_find_all_pi(wfc_list,list_bond_centers,picked_wfcs,double_bonds,UNITCELL_VECTORS):
    # TODO :: hard code :: ここは改善の余地あり．
    list_mu_pai = [] 
    list_pi_wfcs = []

    for bcs in list_bond_centers : # 再度ボンドセンターを探索
        dbcs = bcs[double_bonds] # double_bondsに属するものを選択
        mu_bonds_mol = []
        wcs_mol = []
        for dbond_center_coord  in dbcs : # 二重結合的に探索する？
            r_threshold = 0.65 # angstrom
            # wfc_list_exclude_pickedwfcs = [wfc_list[config][i] for i in range(len(wfc_list[config])) if i not in picked_wfcs]
            wcs_indices, mu_bond, wcs_bond = raw_find_pi(dbond_center_coord,wfc_list,r_threshold,picked_wfcs,UNITCELL_VECTORS)
            if wcs_indices is not None and mu_bond is not None:
                picked_wfcs        = picked_wfcs + list(wcs_indices)
                mu_bonds_mol.append(mu_bond)
                wcs_mol.append(wcs_bond)
        list_mu_pai.append(mu_bonds_mol)
        list_pi_wfcs.append(wcs_mol)
    return np.array(list_mu_pai), np.array(list_pi_wfcs), picked_wfcs


#
# * 全てのwcsの割り当て
def raw_calc_mu_bond_lonepair(wfc_list,ase_atoms,UNITCELL_VECTORS) :
    '''
    # * wfc_list：あるconfigでのワニエの座標リスト
    # * この時WCsの各ボンドへの割り当ても行われる．
    # * parsed_resultsはparse_cpmd_result関数の出力を入れる．(list_mol_coords,list_bond_centers)
    # * output
    ボンドの双極子: list_mu_bonds
    π結合の双極子：list_mu_pai
    Oのローンペアの双極子：list_mu_lpO
    Nのローンペアの双極子：list_mu_lpN
    '''
    raw_aseatom_to_mol_coord_bc(ase_atoms,unit_cell_bonds) 
#ワニエ中心を各分子に帰属する
results_mu=calc_mu_bond_lonepair(wfc_list[fr],results)
    
    list_mol_coords,list_bond_centers = parsed_results    

    ### INPUTS ###
    # parsed_results : 関数parse_cpmd_resultを参照 

    #各結合上のワニエ中心の座標を取得する
    r_threshold =0.65 #[ang.]　結合中点位置からどの距離までをワニエ中心とみなすかのしきい値

    from ase import Atoms
    list_mu_bonds = []
    picked_wfcs = [] #すでにアサインされたwcsを入れる．
    
    list_bond_wfcs = []

    ##########O原子とN原子上のLonePairを探索する###########

    list_mu_lpO = []
    list_mu_lpN = []
    
    list_lpO_wfcs = []
    list_lpN_wfcs = []

    # O/N原子があるところのリスト
    atO_list = [np.argwhere(js==8).reshape(-1) for js in list_atomic_nums] #原子番号8
    atN_list = [np.argwhere(js==7).reshape(-1) for js in list_atomic_nums] #原子番号7
    # for js in list_atomic_nums: # 古いコード
    #     atO = np.argwhere(js==8).reshape(-1) #原子番号8
    #     atN = np.argwhere(js==7).reshape(-1) #原子番号7
    #     atO_list.append(atO)
    #     atN_list.append(atN)

    # Oローンペア
    list_mu_lpO, list_lpO_wfcs, picked_wcs_O = raw_find_all_lonepairs(wfc_list,atO_list,list_mol_coords,picked_wfcs,2,UNITCELL_VECTORS)
    picked_wfcs = picked_wfcs + list(picked_wcs_O)

    # Nローンペア
    list_mu_lpN, list_lpN_wfcs, picked_wcs_N = raw_find_all_lonepairs(wfc_list,atO_list,list_mol_coords,picked_wfcs,1,UNITCELL_VECTORS)
    picked_wfcs = picked_wfcs + list(picked_wcs_N)

    # ボンドセンター    
    list_mu_bonds, list_bond_wfcs, picked_wcs_bond = raw_find_all_bonds(wfc_list,list_bond_centers,picked_wfcs,UNITCELL_VECTORS)    
    picked_wfcs = picked_wfcs + list(picked_wcs_bond)

    ##########π電子を探索する###########    
    # TODO :: hard code :: ここは改善の余地あり．
    list_mu_pai, list_pi_wfcs, picked_wcs_pi = raw_find_all_pi(wfc_list,list_bond_centers,picked_wfcs,UNITCELL_VECTORS) 
    picked_wfcs = picked_wfcs + list(picked_wcs_pi)

    ###for Debug###
    #print(np.sort(np.array(picked_wfcs)))
    #print(len(picked_wfcs))

    return list_mu_bonds,list_mu_pai,list_mu_lpO,list_mu_lpN, list_bond_wfcs,list_pi_wfcs,list_lpO_wfcs,list_lpN_wfcs
