# ===============
# BECを利用する計算で使う奴ら
# ===============


##=====================
## mdtrajからaseへの変換
##=====================
#
# caution！！：Atoms公式はangstromで入れよと言っているが，positionsとcellの単位が同じであれば問題ない．
# ということで，とりあえずはmdtrajのnmで入れるようになっている．
#
#
# 計算量低減のため，NUM_MOLやtableは外部から与えるようにしている．
#
# 
###########  input  ############
# CM_array:: 原子座標のリスト．[NUM_ATOM,3]次元配列
# UNITCELL_VECTORS:: 格子の形．[3,3]次元配列．mdtrajから引っ張るのを推奨

###########  output  ############
# return:: CM_arrayにMICを考慮した各原子間の距離．[NUM_ATOM, NUM_ATOM, 3]次元配列．
# 例えば自分自身との距離は0なので[i,i,:]=(0,0,0)となる．
 
# !! caution !! 同時に，この方法はmdtrajからaseへの変換を与える関数にもなっているので色々使えると思う．

def convert_mdtraj_to_ase(snap, NUM_ATOM, UNITCELL_VECTORS, elements):
    # idの取得
    # table, bonds =snap.topology.to_dataframe()
    # print(table.head(50))
    # 
    # 
    from ase import Atoms # MIC計算用
    mols = Atoms(symbols=elements ,  # table['element']
        positions=snap.xyz.reshape([NUM_ATOM,3])*10,  # nm(gro) to ang (ase)
        cell= UNITCELL_VECTORS*10,   
        pbc=[1, 1, 1]) #pbcは周期境界条件．これをxyz軸にかけることを表している．
    return mols   # ase object


## 

# alamodeのborninfo形式のファイルを読み込んでbecを返す．
def read_borninfo(filename):
    '''
    input :: borninfo
    output:: BEC
    '''
    import numpy as np
    rawdata=np.loadtxt(filename)
    num_atom=int(rawdata.shape[0]/3-1) #原子数
    return rawdata[3:,].reshape([num_atom,3,3])




# これは永久双極子の計算
# あるconfiguration，molsでのdipoleを計算
# mols.get_positions(pbc=True)
#
# positions :: [NUM_ATOM,3]次元配列
# charges   :: 古典の場合[NUM_ATOM]配列
# charges   :: BECの場合，[NUM_ATOM,3,3]次元配列
def calc_dipole_by_BEC(positions, charges) :
    import numpy as np 
    mu=np.einsum("ai, aij -> j", positions, charges )
    mu *= (1.602176487*10.0/3.33564)   # e*Ang to D
    return mu

