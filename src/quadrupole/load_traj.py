#====================
# 構造データの読み込み: mdtrajデータを直接読み込み
#====================
# eq_pbc.trrと，eq.pdbの二つのファイルを持ってくる必要がある．
#
#
def load_traj(directory, NUM_MOL=int(108), PLOT=False):
    #
    #########  input #############
    #  directory:: eq.pdbとeq_pbc.trrが入っているディレクトリ
    #  NUM_MOL:: 分子数の指定(trajファイルからatom数は指定できるが分子数を取り出せるかわからないので．)
    #  #ベンゼン108/シクロヘキサン108だったのでデフォルトを108にしておく
    #  
    #
    # 構造データの読み込み eq.pdbとeq_pbc.trrファイル
    trr_filename=directory+"/eq_pbc.trr"
    pdb_filename=directory+"/eq.pdb"
    #

    import os
    is_file = os.path.isfile(trr_filename)
    if not is_file:
        print("ERROR :: can not find eq_pbc.trr")
        return 1

    is_file = os.path.isfile(pdb_filename)
    if not is_file:
        print("ERROR :: can not find eq.pdb")
        return 1
    #
    print("")
    print(" success:: load eq_pbc.trr/eq.pdb")
    print("")
    
    #
    #diff_config=np.load("8ben_STR/original_structure.npy")
    #diff_config=np.load("benzene_MD_output2/original_structure.npy")
    import mdtraj
    import numpy as np
    traj=mdtraj.load(trr_filename, top=pdb_filename)
    
    from  ase.io.espresso import  read_espresso_in
    
    # トポロジーの作成．一応ここからNUM_MOLを取り出せるが．．．
    # Residue Sequence record (generally 1-based, but depends on topology)
    table, bonds =traj.topology.to_dataframe()
    resseq=np.array(table["resSeq"])

    # 種々のデータをloadする．
    NUM_ATOM    =traj.n_atoms #原子数
    NUM_CONFIG  =traj.n_frames #フレーム数
    VOLUME      =traj.unitcell_volumes[0] # 0番目のconfigurationの体積[nm^3]
    ATOM_PER_MOL=int(NUM_ATOM/NUM_MOL)  # 1分子あたりの原子数
    TIMESTEP    =traj.timestep # in psec 
    UNITCELL_VECTORS = traj.unitcell_vectors[0]  # 最初のconfigのvectorを出力3*3行列

    #============
    print("##=================")
    print("NUM_ATOM             ：：", NUM_ATOM)
    print("NUM_CONFIG           ：：", NUM_CONFIG)
    print("NUM_MOL              ：：", NUM_MOL)
    print("VOLUME[nm^3]         ：：", VOLUME)
    print("セルパラメータ(cubic)[nm]     ：：", np.cbrt(VOLUME))
    print("ATOM_PER_MOL         ：：", ATOM_PER_MOL)
    print("TIMESTEP[ps]         ：：", TIMESTEP)
    print("unitcell vectors[nm] ：：", UNITCELL_VECTORS)


    # ============================
    # 読み込んだ構造データを図示する．
    # ============================
    #
    if PLOT:
        import nglview as nv
        view=nv.show_mdtraj(traj,gui=True)
        view.parameters=dict(
            #density=1.02 #pyrimidine
            camera_type="orthographic",
            background_color="black",
            clip_dist=0)
        view.clear_representations()
        view.add_representation("ball+stick")
        view.add_unitcell()
        view.update_unitcell()
        view

    #
    return traj, NUM_ATOM, NUM_CONFIG,NUM_MOL, VOLUME, ATOM_PER_MOL, UNITCELL_VECTORS, TIMESTEP, resseq

