
# これは永久双極子の計算
# あるconfiguration，molsでのdipoleを計算
def calc_dipole(mols, charges) :
    import numpy as np 
    import ase 

    # mu = np.einsum("aj,a-> j",mols.get_positions(pbc=True), charges)*1.602176487*10.0/3.33564

    mu_x = np.sum(mols.get_positions(pbc=True)[:,0]*charges)
    mu_y = np.sum(mols.get_positions(pbc=True)[:,1]*charges)
    mu_z = np.sum(mols.get_positions(pbc=True)[:,2]*charges)
    mu_x *= 1.602176487*10.0/3.33564
    mu_y *= 1.602176487*10.0/3.33564
    mu_z *= 1.602176487*10.0/3.33564
    #return mu.tolist() 
    return [mu_x,mu_y,mu_z]


# 計算量低減のため，NUM_MOLやtableは外部から与えるようにしている．
def convert_mdtraj_to_ase(snap, NUM_MOL, elements):
    # idの取得
    # table, bonds =snap.topology.to_dataframe()
    # print(table.head(50))
    # 
    # 
    from ase import Atoms # MIC計算用
    mols = Atoms(symbols=elements ,  # table['element']
        positions=snap.xyz.reshape([NUM_MOL,3])*10,  # nm(gro) to ang (ase)
        cell= snap.unitcell_vectors.reshape([3,3])*10,   
        pbc=[1, 1, 1]) #pbcは周期境界条件．これをxyz軸にかけることを表している．
    return mols   # ase object



# mol2=convert_mdtraj_to_ase(traj[0])
# print(calc_dipole(mol2,charges))
# print(classic_dipole[0])

# print("######################")
# print(traj[0].xyz.reshape(432,3)[100,2])
# print(mol2.get_positions(pbc=True)[100,2])


# 誘起双極子の計算
def calc_induced_dipole(mols,mol_indx,charges,atomic_pol):    
    import numpy as np
    import ase
    
    #mols,mol_indx,charges,atomic_pol = parm 
    #print(type(mols))
    # 
    ind_mu=np.zeros(3)
    ind_mu_x = 0
    ind_mu_y = 0
    ind_mu_z = 0
    # mols::原子の部分．
    for p in range(len(mols)):
        # sel::[i=indx,q=charge] ただしpを除く．
        #sel = [[i,q] for i,m,q in zip(range(len(mols)),mol_indx,charges) if (i!=p) & (m!=mol_indx[p]) ]
        sel = [[i,q] for i,m,q in zip(range(len(mols)),mol_indx,charges) if (i!=p) ]
        # print(sel)
        #
        # indx::atomic indexs
        indx = [j[0] for j in sel]
        # q:: charges 
        q    = np.array([j[1] for j in sel])
        # 
        # ある原子pからの原子間の距離を取得        
        # pを中心とするので、ベクトルの向きとしては-1をかける
        vec  = -mols.get_distances(p,indx,mic=True, vector=True)  
        # print(vec.shape)
        # 
        # normを取得
        r    = np.linalg.norm(vec,axis=1).reshape([len(indx),1])
        #
        # x/r,y/r,z/rを取得
        vec_r = vec/(r*r*r)

        #vec_r_x = vec[:,0]/(r**3)
        #vec_r_y = vec[:,1]/(r**3)
        #vec_r_z = vec[:,2]/(r**3)
        # print(np.shape(vec_r_z))
        # 電荷をかける(qが原子の数だけのベクトルになっているので，これがベクトル同士の掛け算になっている！)
        E_vec = np.einsum("aj,a->j", vec_r, q)
        #Ex = np.sum(vec_r_x * q )
        #Ey = np.sum(vec_r_y * q )
        #Ez = np.sum(vec_r_z * q )
        # print(np.shape(vec_r_z * q))

        # polarizabilityをかけてinduced dipoleに追加する．
        ind_mu += atomic_pol[p]*E_vec

        #ind_mu_x += atomic_pol[p]*Ex 
        #ind_mu_y += atomic_pol[p]*Ey 
        #ind_mu_z += atomic_pol[p]*Ez 
    # 係数の調整[from e*Ang to Debye]
    ind_mu *= 1.602176487*10.0/3.33564
    # print(ind_mu.shape)
    # ind_mu_x *= 1.602176487*10.0/3.33564
    # ind_mu_y *= 1.602176487*10.0/3.33564
    # ind_mu_z *= 1.602176487*10.0/3.33564
    #
    #return [ind_mu_x,ind_mu_y,ind_mu_z]
    return ind_mu.tolist()



def convert_mdtraj_to_ase(snap):
    # idの取得
    table, bonds =snap.topology.to_dataframe()
    #print(table.head(50))
    # 
    # 
    from ase import Atoms # MIC計算用
    mols = Atoms(symbols=table['element'],
        positions=snap.xyz.reshape([432,3])*10,  # nm(gro) to ang (ase)
        cell= snap.unitcell_vectors.reshape([3,3])*10,   
        pbc=[1, 1, 1]) #pbcは周期境界条件．これをxyz軸にかけることを表している．
    return mols   # ase object
    


# MICの働き方調査
from ase import Atoms
d = 2.9
L = 10.0
wire = Atoms(['Au','Au','Au'],
             positions=[[0, 0, 0],
                        [L/8, 0, 0],
                        [0, 0, L/4]
                        ],
             cell=[L, L, L],
             pbc=[1, 1, 1])
#print(wire.get_distances(0,[1,2],mic=True, vector=True))
#print(wire.get_distances(1,[0,2],mic=True, vector=True))
#print(wire.get_distances(2,[0,1],mic=True, vector=True))

#print("")
#print(wire.get_all_distances(mic=True, vector=True))
#print("")

NUM_ATOM=3
charges=[1.0,1.0,1.0]
atomic_pol=[-2.0,2.0,2.0]
mol_indx=[0,0,0]


print("##########################")
print("##########################")



# !! 改善点 !! NUM_MOLを自動でとってくる．これはtpologyから取ってこれると思う．
traj, NUM_ATOM, NUM_CONFIG,NUM_MOL, VOLUME, ATOM_PER_MOL, UNITCELL_VECTORS, TIMESTEP = load_traj("NAP_trial/run0", NUM_MOL=24, PLOT=False)


mols=decide_MIC(traj[0].xyz.reshape([432,3]), UNITCELL_VECTORS)
mol2=convert_mdtraj_to_ase(traj[0])
print("###############")
print(mols*10-mol2.get_all_distances(mic=True,vector=True))
print("###############")

# ===================
# 分極率+RESPを読み込み（山崎さんコード）
# ===================


import numpy as np

# 分極
pols =np.array([8.793,8.793,8.812,8.636,8.812,8.793,8.793,8.812,8.636,8.812,
        2.522,2.522,2.515,2.515,2.522,2.522,2.515,2.515])

# 電荷
chars=np.array([-0.1133,-0.1133,-0.22,0.16,-0.22,-0.1133,-0.1133,-0.22,0.16,-0.22,
       0.1213, 0.1213, 0.132,0.132,0.1213,0.1213,0.132,0.132])

# 電荷と分極を複製して数の分だけ用意する．
a0 = 5.29177210903e-1
charges = np.tile(chars, NUM_MOL)
atomic_pol = np.tile(pols,NUM_MOL)*(a0**3) #ang
#
#分子のindexを作成する
mol_indx = np.array([int(i/18) for i in range(432)])

# 
CUTOFF = 1000          #単位はnm or Ang（input (MIC後）の座標系の単位と同じ．）



# 電荷による誘起双極子 [D] 
amano=decide_mu_MIC_vec(mols, charges, atomic_pol, CUTOFF)/100


yamazaki=calc_induced_dipole(mol2,mol_indx,charges,atomic_pol)
print("############")
print("ind_dipole_by_me", amano)
print("ind_dipole_by_yamazaki", yamazaki)


print("############")
print("ind_dipole_by_me", np.sum(amano,axis=0))
print("ind_dipole_by_yamazaki", yamazaki)
