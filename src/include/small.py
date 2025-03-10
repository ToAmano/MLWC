

# 簡単なパーツ部分のコード


def if_file_exist(filename):
    import os
    import sys
    is_file = os.path.isfile(filename)
    if not is_file: # itpファイルの存在を確認
        print("ERROR not found the file :: {} !! ".format(filename))    
        sys.exit("1")
    return 0


def python_version_check():
    import sys
    import os
    # import matplotlib.pyplot as plt

    python_major_ver = sys.version_info.major
    python_minor_ver = sys.version_info.minor

    print(" your python version is ... ", python_major_ver, python_minor_ver)

    if sys.version_info.minor < 9: # versionによる分岐 https://www.lifewithpython.com/2015/06/python-check-python-version.html
        print("WARNING :: recommended python version is 3.9 or above. Your version is :: {}".format(sys.version_info.major))
    elif sys.version_info.minor < 7:
        print("ERROR !! python is too old. Please use 3.7 or above. Your version is :: {}".format(sys.version_info.major))
    return 0


def calc_descripter_frame_descmode1(atoms_fr, fr, savedir, itp_data, NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS ):
    '''
    記述子の保存：あり
    ワニエの割り当て：なし
    機会学習:なし
    '''
    import cpmd.descripter
    import cpmd.asign_wcs
    import numpy as np
    
    # * wannierの割り当て部分のメソッド化
    ASIGN=cpmd.asign_wcs.asign_wcs(NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS)
    DESC=cpmd.descripter.descripter(NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS)
    # * 原子座標とボンドセンターの計算
    # 原子座標,ボンドセンターを分子基準で再計算
    results = ASIGN.aseatom_to_mol_coord_bc(atoms_fr, itp_data.bonds_list)
    list_mol_coords, list_bond_centers =results
    # * ボンドデータをさらにch/coなど種別ごとに分割 & 記述子を計算
    # mu_bondsの中身はchとringで分割する
    #mu_paiは全数をringにアサイン
    #mu_lpOとlpNはゼロ
    # ring
    if len(itp_data.ring_bond_index) != 0:
        Descs_ring = []
        ring_cent_mol = cpmd.descripter.find_specific_ringcenter(list_bond_centers, itp_data.ring_bond_index, 8, NUM_MOL)
        i=0 
        for bond_center in ring_cent_mol:
            mol_id = i % NUM_MOL // 1
            Descs_ring.append(DESC.get_desc_bondcent(atoms_fr,bond_center,mol_id))
            i+=1 

    # ch,oh,co,cc,
    Descs_ch=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.bond_index['CH_1_bond'])
    Descs_oh=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.bond_index['OH_1_bond'])
    Descs_co=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.bond_index['CO_1_bond'])
    Descs_cc=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.bond_index['CC_1_bond'])   
    # oローンペア
    Descs_o = DESC.calc_lonepair_descripter_at_frame(atoms_fr,list_mol_coords, itp_data.o_list, 8)

    # データが作成できているかの確認（debug）
    # print( " DESCRIPTOR SHAPE ")
    # print(" ring (Descs/data) ::", Descs_ring.shape)
    # print(" ch-bond (Descs/data) ::", Descs_ch.shape)
    # print(" cc-bond (Descs/data) ::", Descs_cc.shape)
    # print(" co-bond (Descs/data) ::", Descs_co.shape)
    # print(" oh-bond (Descs/data) ::", Descs_oh.shape)
    # print(" o-lone (Descs/data) ::", Descs_o.shape)

    # ring, CHボンド，CCボンド，COボンド，OHボンド，Oローンペアのsave
    if len(itp_data.ring_bond_index) != 0: np.savetxt(savedir+'Descs_ring_'+str(fr)+'.csv', Descs_ring, delimiter=',')
    if len(itp_data.bond_index['CH_1_bond']) != 0: np.savetxt(savedir+'Descs_ch_'+str(fr)+'.csv', Descs_ch, delimiter=',')
    if len(itp_data.bond_index['CC_1_bond']) != 0: np.savetxt(savedir+'Descs_cc_'+str(fr)+'.csv', Descs_cc, delimiter=',')
    if len(itp_data.bond_index['CO_1_bond']) != 0: np.savetxt(savedir+'Descs_co_'+str(fr)+'.csv', Descs_co, delimiter=',')
    if len(itp_data.bond_index['OH_1_bond']) != 0: np.savetxt(savedir+'Descs_oh_'+str(fr)+'.csv', Descs_oh, delimiter=',')                
    # Oローンペア
    if len(itp_data.o_list) != 0: np.savetxt(savedir+'Descs_o_'+str(fr)+'.csv', Descs_o, delimiter=',')
    return 0


def calc_descripter_frame2(atoms_fr, wannier_fr, fr, savedir, itp_data, NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS, double_bonds):
    '''
    記述子の保存：あり
    ワニエの割り当て：あり
    機会学習:なし
    '''
    import numpy as np
    import cpmd.descripter
    import cpmd.asign_wcs
    # * wannierの割り当て部分のメソッド化
    ASIGN=cpmd.asign_wcs.asign_wcs(NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS)
    DESC=cpmd.descripter.descripter(NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS)

    # * 原子座標とボンドセンターの計算
    # 原子座標,ボンドセンターを分子基準で再計算
    # TODO :: list_mol_coordsを使うのではなく，原子座標からatomsを作り直した方が良い．
    # TODO :: そうしておけば後ろでatomsを使う時にmicのことを気にしなくて良い（？）ので楽かも．
    results = ASIGN.aseatom_to_mol_coord_bc(atoms_fr, itp_data.bonds_list)
    list_mol_coords, list_bond_centers =results
    
    # wcsをbondに割り当て，bondの双極子まで計算
    results_mu = ASIGN.calc_mu_bond_lonepair(wannier_fr,atoms_fr,itp_data.bonds_list,double_bonds)
    list_mu_bonds,list_mu_pai,list_mu_lpO,list_mu_lpN, list_bond_wfcs,list_pi_wfcs,list_lpO_wfcs,list_lpN_wfcs = results_mu
    # wannnierをアサインしたase.atomsを作成する
    mol_with_WC = cpmd.asign_wcs.make_ase_with_WCs(atoms_fr.get_atomic_numbers(),NUM_MOL, UNITCELL_VECTORS,list_mol_coords,list_bond_centers,list_bond_wfcs,list_pi_wfcs,list_lpO_wfcs,list_lpN_wfcs)
    # 系の全双極子を計算
    # print(" list_mu_bonds {0}, list_mu_pai {1}, list_mu_lpO {2}, list_mu_lpN {3}".format(np.shape(list_mu_bonds),np.shape(list_mu_pai),np.shape(list_mu_lpO),np.shape(list_mu_lpN)))
    # ase.io.write(savedir+"molWC_"+str(fr)+".xyz", mol_with_WC)
    Mtot = []
    for i in range(NUM_MOL):
        Mtot.append(np.sum(list_mu_bonds[i],axis=0)+np.sum(list_mu_pai[i],axis=0)+np.sum(list_mu_lpO[i],axis=0)+np.sum(list_mu_lpN[i],axis=0))
    Mtot = np.array(Mtot)
    #unit cellの双極子モーメントの計算
    total_dipole = np.sum(Mtot,axis=0)
    # total_dipole = np.sum(list_mu_bonds,axis=0)+np.sum(list_mu_pai,axis=0)+np.sum(list_mu_lpO,axis=0)+np.sum(list_mu_lpN,axis=0)
    # ワニエセンターのアサイン
    #ワニエ中心を各分子に帰属する
    # results_mu=ASIGN.calc_mu_bond(atoms_fr,results)
    #ワニエ中心の座標を計算する
    # results_wfcs = ASIGN.assign_wfc_to_mol(atoms_fr,results) 

    # * ボンドデータをさらにch/coなど種別ごとに分割 & 記述子を計算
    # mu_bondsの中身はchとringで分割する
    #mu_paiは全数をringにアサイン
    #mu_lpOとlpNはゼロ
    # ring
    if len(itp_data.ring_bond_index) != 0:
        Descs_ring = []
        ring_cent_mol = cpmd.descripter.find_specific_ringcenter(list_bond_centers, itp_data.ring_bond_index, 8, NUM_MOL)
        i=0 
        for bond_center in ring_cent_mol:
            mol_id = i % NUM_MOL // 1
            Descs_ring.append(DESC.get_desc_bondcent(atoms_fr,bond_center,mol_id))
            i+=1 

    # ch,oh,co,cc,
    Descs_ch=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.bond_index['CH_1_bond'])
    Descs_oh=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.bond_index['OH_1_bond'])
    Descs_co=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.bond_index['CO_1_bond'])
    Descs_cc=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.bond_index['CC_1_bond'])   
    # oローンペア
    Descs_o = DESC.calc_lonepair_descripter_at_frame(atoms_fr,list_mol_coords, itp_data.o_list, 8)

    # データが作成できているかの確認（debug）
    # print( " DESCRIPTOR SHAPE ")
    # print(" ring (Descs/data) ::", Descs_ring.shape)
    # print(" ch-bond (Descs/data) ::", Descs_ch.shape)
    # print(" cc-bond (Descs/data) ::", Descs_cc.shape)
    # print(" co-bond (Descs/data) ::", Descs_co.shape)
    # print(" oh-bond (Descs/data) ::", Descs_oh.shape)
    # print(" o-lone (Descs/data) ::", Descs_o.shape)

    # ring, CHボンド，CCボンド，COボンド，OHボンド，Oローンペアのsave
    if len(itp_data.ring_bond_index) != 0: np.savetxt(savedir+'Descs_ring_'+str(fr)+'.csv', Descs_ring, delimiter=',')
    if len(itp_data.bond_index['CH_1_bond']) != 0: np.savetxt(savedir+'Descs_ch_'+str(fr)+'.csv', Descs_ch, delimiter=',')
    if len(itp_data.bond_index['CC_1_bond']) != 0: np.savetxt(savedir+'Descs_cc_'+str(fr)+'.csv', Descs_cc, delimiter=',')
    if len(itp_data.bond_index['CO_1_bond']) != 0: np.savetxt(savedir+'Descs_co_'+str(fr)+'.csv', Descs_co, delimiter=',')
    if len(itp_data.bond_index['OH_1_bond']) != 0: np.savetxt(savedir+'Descs_oh_'+str(fr)+'.csv', Descs_oh, delimiter=',')                
    # Oローンペア
    if len(itp_data.o_list) != 0: np.savetxt(savedir+'Descs_o_'+str(fr)+'.csv', Descs_o, delimiter=',')

    # データが作成できているかの確認（debug）
    # print( " DESCRIPTOR SHAPE ")
    # print(" ring (Descs/data) ::", Descs_ring.shape)
    # print(" ch-bond (Descs/data) ::", Descs_ch.shape)
    # print(" cc-bond (Descs/data) ::", Descs_cc.shape)
    # print(" co-bond (Descs/data) ::", Descs_co.shape)
    # print(" oh-bond (Descs/data) ::", Descs_oh.shape)
    # print(" o-lone (Descs/data) ::", Descs_o.shape)

    # ring, CHボンド, CCボンド, COボンド, OHボンド, Oローンペアの記述子を保存
    if len(itp_data.ring_bond_index) != 0: np.savetxt(savedir+'Descs_ring_'+str(fr)+'.csv', Descs_ring, delimiter=',')
    if len(itp_data.bond_index['CH_1_bond']) != 0: np.savetxt(savedir+'Descs_ch_'+str(fr)+'.csv', Descs_ch, delimiter=',')
    if len(itp_data.bond_index['CC_1_bond']) != 0: np.savetxt(savedir+'Descs_cc_'+str(fr)+'.csv', Descs_cc, delimiter=',')
    if len(itp_data.bond_index['CO_1_bond']) != 0: np.savetxt(savedir+'Descs_co_'+str(fr)+'.csv', Descs_co, delimiter=',')
    if len(itp_data.bond_index['OH_1_bond']) != 0: np.savetxt(savedir+'Descs_oh_'+str(fr)+'.csv', Descs_oh, delimiter=',')
    if len(itp_data.o_list) != 0:
        np.savetxt(savedir+'Descs_o_'+str(fr)+'.csv', Descs_o, delimiter=',')
    return mol_with_WC, total_dipole
    # >>>> 関数ここまで <<<<<

# * 記述子をロードして予測させる
def predict_dipole_mode1(fr,desc_dir):
    import numpy as np

    #
    # * 機械学習用のデータを読み込む
    # *
    #
    global model_ch_2
    global model_co_2
    global model_oh_2
    global model_o_2

    # デバイスの設定    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    nfeatures = 288

    # ring
    # descs_X_ring = np.loadtxt('Descs_ring.csv',delimiter=',')
    # data_y_ring = np.loadtxt('data_y_ring.csv',delimiter=',')

    # CHボンド，COボンド，OHボンド，Oローンペア
    descs_X_ch = np.loadtxt(desc_dir+'Descs_ch_'+str(fr)+'.csv',delimiter=',')
    descs_X_co = np.loadtxt(desc_dir+'Descs_co_'+str(fr)+'.csv',delimiter=',')
    descs_X_oh = np.loadtxt(desc_dir+'Descs_oh_'+str(fr)+'.csv',delimiter=',')
    descs_X_o =  np.loadtxt(desc_dir+'Descs_o_'+str(fr)+'.csv',delimiter=',')

    # オリジナルの記述子を一旦tensorへ
    X_ch = torch.from_numpy(descs_X_ch.astype(np.float32)).clone()
    X_oh = torch.from_numpy(descs_X_oh.astype(np.float32)).clone()
    X_co = torch.from_numpy(descs_X_co.astype(np.float32)).clone()
    X_o  = torch.from_numpy(descs_X_o.astype(np.float32)).clone()

    # 予測
    y_pred_ch  = model_ch_2(X_ch.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
    y_pred_co  = model_co_2(X_co.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
    y_pred_oh  = model_oh_2(X_oh.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
    y_pred_o   = model_o_2(X_o.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()

    # 最後にreshape
    # !! ここは形としては(NUM_MOL*len(bond_index),3)となるが，予測だけする場合NUM_MOLの情報をgetできないので
    # 1! reshape(-1,3)としてしまう．
    
    # TODO : hard code (分子数)
    # NUM_MOL = 64
    y_pred_ch = y_pred_ch.reshape((-1,3))
    y_pred_co = y_pred_co.reshape((-1,3))
    y_pred_oh = y_pred_oh.reshape((-1,3))
    y_pred_o  = y_pred_o.reshape((-1,3))
    # print("DEBUG :: shape ch/co/oh/o :: {0} {1} {2} {3}".format(np.shape(y_pred_ch),np.shape(y_pred_co),np.shape(y_pred_oh),np.shape(y_pred_o)))
    # if fr == 0: # debug
        # print("y_pred_ch ::", y_pred_ch)
        # print("y_pred_co ::", y_pred_co)
        # print("y_pred_oh ::", y_pred_oh)
        # print("y_pred_o  ::", y_pred_o)
        #予測したモデルを使ったUnit Cellの双極子モーメントの計算
    sum_dipole=np.sum(y_pred_ch,axis=0)+np.sum(y_pred_oh,axis=0)+np.sum(y_pred_co,axis=0)+np.sum(y_pred_o,axis=0)
    return sum_dipole


def calc_descripter_frame_and_predict_dipole(atoms_fr, fr, itp_data, NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS, model_ch_2, model_co_2, model_oh_2, model_o_2):
    import numpy as np
    
    '''
    機械学習での予測：あり
    ワニエのアサイン：なし
    '''
    if atoms_fr == None:
        return np.array([100,100,100]) # Noneの場合は100,100,100を代入する．もちろんこれはfakeである．
    import cpmd.descripter
    import cpmd.asign_wcs
    # * wannierの割り当て部分のメソッド化
    ASIGN=cpmd.asign_wcs.asign_wcs(NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS)
    DESC=cpmd.descripter.descripter(NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS)
    
    # * 原子座標とボンドセンターの計算
    # 原子座標,ボンドセンターを分子基準で再計算
    results = ASIGN.aseatom_to_mol_coord_bc(atoms_fr, itp_data.bonds_list)
    list_mol_coords, list_bond_centers =results

    # * ボンドデータをさらにch/coなど種別ごとに分割 & 記述子を計算
    # mu_bondsの中身はchとringで分割する
    #mu_paiは全数をringにアサイン
    #mu_lpOとlpNはゼロ
    # ring
    if len(itp_data.ring_bond_index) != 0:
        Descs_ring = []
        ring_cent_mol = cpmd.descripter.find_specific_ringcenter(list_bond_centers, itp_data.ring_bond_index, 8, NUM_MOL)
        i=0 
        for bond_center in ring_cent_mol:
            mol_id = i % NUM_MOL // 1
            Descs_ring.append(DESC.get_desc_bondcent(atoms_fr,bond_center,mol_id))
            i+=1 

    # ch,oh,co,cc,
    Descs_ch=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.bond_index['CH_1_bond'])
    Descs_oh=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.bond_index['OH_1_bond'])
    Descs_co=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.bond_index['CO_1_bond'])
    Descs_cc=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.bond_index['CC_1_bond'])   
    # oローンペア
    Descs_o = DESC.calc_lonepair_descripter_at_frame(atoms_fr,list_mol_coords, itp_data.o_list, 8)

    # データが作成できているかの確認（debug）
    # print( " DESCRIPTOR SHAPE ")
    # print(" ring (Descs/data) ::", Descs_ring.shape)
    # print(" ch-bond (Descs/data) ::", Descs_ch.shape)
    # print(" cc-bond (Descs/data) ::", Descs_cc.shape)
    # print(" co-bond (Descs/data) ::", Descs_co.shape)
    # print(" oh-bond (Descs/data) ::", Descs_oh.shape)
    # print(" o-lone (Descs/data) ::", Descs_o.shape)

    # オリジナルの記述子を一旦tensorへ
    X_ch = torch.from_numpy(Descs_ch.astype(np.float32)).clone()
    X_oh = torch.from_numpy(Descs_oh.astype(np.float32)).clone()
    X_co = torch.from_numpy(Descs_co.astype(np.float32)).clone()
    X_o  = torch.from_numpy(Descs_o.astype(np.float32)).clone()

    # # 機械学習モデルの変数
    # global model_ch_2
    # global model_co_2
    # global model_oh_2
    # global model_o_2

    # デバイスの設定    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    nfeatures = 288

    # 予測
    y_pred_ch  = model_ch_2(X_ch.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
    y_pred_co  = model_co_2(X_co.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
    y_pred_oh  = model_oh_2(X_oh.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
    y_pred_o   = model_o_2(X_o.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()

    # 最後にreshape
    # !! ここは形としては(NUM_MOL*len(bond_index),3)となるが，予測だけする場合NUM_MOLの情報をgetできないので
    # !! reshape(-1,3)としてしまう．

    # TODO : hard code (分子数)
    # NUM_MOL = 64
    y_pred_ch = y_pred_ch.reshape((-1,3))
    y_pred_co = y_pred_co.reshape((-1,3))
    y_pred_oh = y_pred_oh.reshape((-1,3))
    y_pred_o  = y_pred_o.reshape((-1,3))
    # print("DEBUG :: shape ch/co/oh/o :: {0} {1} {2} {3}".format(np.shape(y_pred_ch),np.shape(y_pred_co),np.shape(y_pred_oh),np.shape(y_pred_o)))
    global rank
    # if fr == 0 : # デバッグ用
    #     print(" DEBUG y_pred shape len(y_pred_*)")
    #     print("y_pred_ch ::", len(y_pred_ch))
    #     print("y_pred_co ::", len(y_pred_co))
    #     print("y_pred_oh ::", len(y_pred_oh))
    #     print("y_pred_o  ::", len(y_pred_o))
    #予測したモデルを使ったUnit Cellの双極子モーメントの計算
    sum_dipole=np.sum(y_pred_ch,axis=0)+np.sum(y_pred_oh,axis=0)+np.sum(y_pred_co,axis=0)+np.sum(y_pred_o,axis=0)

    return sum_dipole
