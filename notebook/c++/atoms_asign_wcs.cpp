// #define _DEBUG
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include <cstdio>
#include <vector>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream> // https://www.cns.s.u-tokyo.ac.jp/~masuoka/post/inputfile_cpp/
#include <regex> // using cmatch = std::match_results<const char*>;
#include <map> // https://bi.biopapyrus.jp/cpp/syntax/map.html
#include <cmath> 
#include <algorithm>
#include <numeric> // std::iota
#include <tuple> // https://tyfkda.github.io/blog/2021/06/26/cpp-multi-value.html
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <Eigen/Core> // 行列演算など基本的な機能．
#include "numpy.hpp"
#include "npy.hpp"
// #include "numpy_quiita.hpp" // https://qiita.com/ka_na_ta_n/items/608c7df3128abbf39c89
// numpy_quiitaはsscanf_sが読み込めず，残念ながら現状使えない．
// #include "atoms_core.cpp" // !! これをよむとまずい？
#include "atoms_io.cpp"
#include "mol_core.cpp"


/*
*/

std::vector<Eigen::Vector3d> raw_calc_mol_coord_mic_onemolecule(std::vector<int> mol_inds, std::vector<std::vector<int>> bonds_list_j, Atoms aseatoms, read_mol itp_data) {
    /*
    1つの分子のpbc-mol計算を実施し，
        - 分子座標
    の計算を行う．
    mol_inds :: 分子部分を示すインデックス
    bonds_list_j :: 分子内のボンドのリスト
    */
   // TODO :: グラフ理論に基づいたボンドセンターの計算を行うためにraw_get_pbc_molを定義する．
   // 通常のraw_get_distances_micを使って，mol_inds[0]からmol_indsへの距離を計算する．
   // TODO :: mol_inds[0]になっているが本来はmol_inds[representative_atom_index]になるべき．
    std::vector<Eigen::Vector3d> vectors = raw_get_distances_mic(aseatoms, mol_inds[itp_data.representative_atom_index], mol_inds, true, true);
#ifdef DEBUG
    std::cout << "vectors..." << std::endl;
    for (int i = 0; i < vectors.size(); i++) {
        std::cout << std::setw(10) << vectors[i][0] << vectors[i][1] <<  vectors[i][2] << std::endl;
    }
#endif // !DEBUG
    // mol_inds[itp_data.representative_atom_index]の座標を取得する．
    Eigen::Vector3d R0 = aseatoms.get_positions()[mol_inds[itp_data.representative_atom_index]];

    // mol_indsを0から始まるように変換する．
    std::vector<int> mol_inds_from_zero;
    for (int i = 0; i < mol_inds.size(); i++) {
        mol_inds_from_zero.push_back(mol_inds[i] - mol_inds[0]);
    }
    // 同様にボンドリストが0から始まるように変換する．
    // TODO :: そもそもbonds_list_jを使う必要ないよね？ mol_indsから直接計算できるもんな．．．
    std::vector<std::vector<int>> bonds_list_from_zero;
    for (int i = 0; i < bonds_list_j.size(); i++) {
        // std::vector<int> bond;
        // bond.push_back(bonds_list_j[i][0] - mol_inds[0]);
        // bond.push_back(bonds_list_j[i][1] - mol_inds[0]);
        bonds_list_from_zero.push_back({bonds_list_j[i][0] - mol_inds[0],bonds_list_j[i][1] - mol_inds[0]});
    }
    // 分子の座標を再計算する．
    std::vector<Eigen::Vector3d> mol_coords;
    for (int k = 0; k < mol_inds_from_zero.size(); k++) {
        Eigen::Vector3d mol_coord = R0 + vectors[mol_inds_from_zero[k]];
        mol_coords.push_back(mol_coord);
    }
    return mol_coords;
}

std::vector<Eigen::Vector3d> raw_calc_bc_mic_onemolecule(std::vector<int> mol_inds, std::vector<std::vector<int>> bonds_list_j,     std::vector<Eigen::Vector3d> mol_coords) {
    /*
        すでに計算された分子座標を使って，ボンドセンターを計算する．
        - ボンドセンター座標
        TODO :: この関数もbonds_list_jを使う必要がない．ただのbonds_listから計算可能．
    */
    // ボンドリストが0から始まるように変換する．
    // TODO :: そもそもbonds_list_jを使う必要ないよね？ mol_indsから直接計算できるもんな．．．
    std::vector<std::vector<int>> bonds_list_from_zero;
    for (int i = 0; i < bonds_list_j.size(); i++) {
        std::vector<int> bond;
        bond.push_back(bonds_list_j[i][0] - mol_inds[0]);
        bond.push_back(bonds_list_j[i][1] - mol_inds[0]);
        bonds_list_from_zero.push_back(bond);
    }
    // ボンドセンターを計算する．
    std::vector<Eigen::Vector3d> bond_centers;
    for (int l = 0; l < bonds_list_from_zero.size(); l++) {
        // Eigen::Vector3d bc = R0 + (vectors.col(bonds_list_from_zero[l][0]) + vectors.col(bonds_list_from_zero[l][1])) / 2.0;
        Eigen::Vector3d bc = (mol_coords[bonds_list_from_zero[l][0]] + mol_coords[bonds_list_from_zero[l][1]]) / 2.0;
        if ((mol_coords[bonds_list_from_zero[l][0]] - mol_coords[bonds_list_from_zero[l][1]]).norm() > 2.0) { // bond length is too long
            std::cout << "WARNING :: bond length is too long !! :: " << bonds_list_from_zero[l][0] << " " << bonds_list_from_zero[l][1] << " " <<(mol_coords[bonds_list_from_zero[l][0]] - mol_coords[bonds_list_from_zero[l][1]]).norm() << std::endl;
        }
        bond_centers.push_back(bc);
    }
    return bond_centers;
}

std::tuple<std::vector<std::vector<Eigen::Vector3d> >, std::vector<std::vector<Eigen::Vector3d> > > raw_aseatom_to_mol_coord_and_bc(Atoms ase_atoms, std::vector<std::vector<int>> bonds_list, read_mol itp_data, int NUM_MOL_ATOMS, int NUM_MOL) {
    /*
    ase_atomsから，
     - 1: ボンドセンターの計算
     - 2: micを考慮した原子座標の再計算
    を行う．基本的にはcalc_mol_coordのwrapper関数
    
    input
    ------------
    ase_atoms       :: ase.atoms
    mol_ats         ::
    bonds_list      :: itpdataに入っているボンドリスト
    output
    ------------
    list_mol_coords :: 
    list_bond_centers
    
    NOTE
    ------------
    2023/4/16 :: inputとしていたunit_cell_bondsをより基本的な変数bond_listへ変更．
    bond_listは1分子内でのボンドの一覧であり，そこからunit_cell_bondsを関数の内部で生成する．
    */
    std::array<std::vector<std::vector<double>>, 2> result;
    std::vector<std::vector<Eigen::Vector3d> > list_mol_coords; 
    std::vector<std::vector<Eigen::Vector3d> > list_bond_centers; 
    
    // 1分子内のindexを取得する．
    std::vector<int> mol_at0(NUM_MOL_ATOMS);
    for (int i = 0; i < NUM_MOL_ATOMS; i++) {
        mol_at0[i] = i;
    }
    // 1config内のindexを取得する．
    std::vector<std::vector<int>> mol_ats(NUM_MOL, std::vector<int>(NUM_MOL_ATOMS));
    for (int indx = 0; indx < NUM_MOL; indx++) {
        for (int i = 0; i < NUM_MOL_ATOMS; i++) {
            mol_ats[indx][i] = i + NUM_MOL_ATOMS * indx;
        }
    }
    
    // bonds_listも1config内のものに対応させる．（NUM_MOL*bonds_list型のリスト）
    std::vector<std::vector<std::vector<int> > > unit_cell_bonds(NUM_MOL, std::vector<std::vector<int> >(bonds_list.size()));
    for (int indx = 0; indx < NUM_MOL; indx++) {
        for (int i = 0; i < bonds_list.size(); i++) {
            unit_cell_bonds[indx][i] = {bonds_list[i][0] + NUM_MOL_ATOMS * indx, bonds_list[i][1] + NUM_MOL_ATOMS * indx};
        }
    }
    
    // 
    for (int j = 0; j < NUM_MOL; j++) {    // NUM_MOL個の分子に対するLoop
        // std::vector<int> mol_inds = mol_ats[j];
        // std::vector<std::array<int, 2>> bonds_list_j = unit_cell_bonds[j];
        std::vector<Eigen::Vector3d> mol_coords = raw_calc_mol_coord_mic_onemolecule(mol_ats[j], unit_cell_bonds[j], ase_atoms, itp_data);
        std::vector<Eigen::Vector3d> bond_centers = raw_calc_bc_mic_onemolecule(mol_ats[j], unit_cell_bonds[j], mol_coords);
        list_mol_coords.push_back(mol_coords);
        list_bond_centers.push_back(bond_centers);
    }
    return {list_mol_coords,list_bond_centers};
}

Atoms make_ase_with_BCs(std::vector<int> ase_atomicnumber, int NUM_MOL, std::vector<std::vector<double> > UNITCELL_VECTORS, std::vector<std::vector<Eigen::Vector3d> > list_mol_coords, std::vector<std::vector<Eigen::Vector3d> > list_bond_centers){
    /*
    list_mol_coordsとlist_bond_centersから新しいase.atomsを作成する．
    現状だと単にase.atomsを作成するだけで，用途としてはase.io.writeで保存するくらいだ．

    注意点として，push_backする順番に気をつけたい．文仕事に，原子，ボンドセンター(He割り当て)の順で割り当てる．
    基本的に，普通の扱いでは分けている分子ごとの部分をフラットにするだけ．

    TODO :: ase_atomicnumberとNUM_MOLの扱いは難しい．一つの分子のみ（混合溶液でない）ならread_molから原子のリストを持ってきても良い．
    TODO :: しかし，もっとも一般的なのはase_atoms.get_atomic_numbers()を使って一つづつappendしていくこと．
    */
    //Atoms make_ase_with_BCs( ){
    // std::vector<std::vector<Eigen::Vector3d> > list_mol_coords;
    // std::vector<std::vector<Eigen::Vector3d> > list_bond_centers;

    std::vector<Eigen::Vector3d> new_coord;
    std::vector<int> new_atomic_num;
    // std::vector<std::vector<int>> list_atomic_nums = ase_atomicnumber.reshape(NUM_MOL, -1);
    for (int i = 0; i < list_mol_coords.size(); i++) {    //分子数に関するLoop
        for (int j = 0; j < list_mol_coords[i].size(); j++) {    //分子内の原子数に関するLoop
            new_coord.push_back(list_mol_coords[i][j]);
            new_atomic_num.push_back(ase_atomicnumber[i*list_mol_coords.size()+j]);
        }
        for (int j = 0; j < list_bond_centers[i].size(); j++) {    //分子内のボンドセンター数に関するLoop
            new_coord.push_back(list_bond_centers[i][j]);
            new_atomic_num.push_back(2);
        }
    }
    // Atomsを作成する．    
    Atoms new_atoms(new_atomic_num, new_coord, UNITCELL_VECTORS, {1,1,1});
    return new_atoms;
}

Atoms raw_make_atoms_with_bc(Eigen::Vector3d bond_center, Atoms atoms, std::vector<std::vector<double> > UNITCELL_VECTORS){
    /*
    ######INPUTS#######
    bond_center     # Eigen::Vector3d 記述子を求めたい結合中心の座標
    list_mol_coords # array  分子ごとの原子座標
    list_atomic_nums #array  分子ごとの原子座標
    */
    std::vector<Eigen::Vector3d> list_mol_coords=atoms.get_positions();
    std::vector<int> list_atomic_num=atoms.get_atomic_numbers();

    // 結合中点bond_centerを先頭においたAtomsオブジェクトを作成する
    list_mol_coords.insert(list_mol_coords.begin(), bond_center);
    list_atomic_num.insert(list_atomic_num.begin(), 79); // 結合中心のラベルはAu(79)とする
    Atoms WBC = Atoms(list_atomic_num,
                    list_mol_coords,        
                    UNITCELL_VECTORS,   
                    {1, 1, 1});
    return WBC;
};

std::vector<Eigen::Vector3d> find_specific_bondcenter(std::vector<std::vector<Eigen::Vector3d> > list_bond_centers, std::vector<int> bond_index){
    /*
    list_bond_centersからbond_index情報をもとに特定のボンド（CHなど）だけ取り出す．
    TODO :: ここは将来的にはボンドをあらかじめ分割するようにして無くしてしまいたい．
    * @param[in] bond_index : 分子内のボンドセンターのindex（read_mol.ch_bond_indexなど．）
    * @return cent_mol : 特定のボンドの原子座標（(-1,3)型）
    */
    std::vector<Eigen::Vector3d> cent_mol;
    // ボンドセンターの座標と双極子をappendする．
    for (int i = 0; i < list_bond_centers.size(); i++) { //UnitCellの分子ごとに分割
        // chボンド部分（chボンドの重心をappend）
        for (int j = 0; j < bond_index.size(); j++) {    //分子内のボンドセンター数に関するLoop
#ifdef DEBUG
            std::cout << " print list_bond_centers[i][bond_index[j]] :: " << list_bond_centers[i][bond_index[j]][0] << std::endl;
#endif //! DEBUG
            cent_mol.push_back(list_bond_centers[i][bond_index[j]]);
        }
    }
    return cent_mol;
    };

double fs(double Rij,double Rcs,double Rc){
    /*
    カットオフ関数．
    #####Inputs####
    # Rij : float 原子間距離 [ang. unit] 
    # Rcs : float inner cut off [ang. unit]
    # Rc  : float outer cut off [ang. unit] 
    ####Outputs####
    # sij value 
    ###############
    */
    double s;
    if (Rij < Rcs){
        s = 1/Rij;
    } else if(Rij < Rc){
        s = (1/Rij)*(0.5*cos(M_PI*(Rij-Rcs)/(Rc-Rcs))+0.5);
    } else{
        s = 0;
    }
    return s;
}

std::vector<double> calc_descripter(std::vector<Eigen::Vector3d> dist_wVec,std::vector<int> atoms_index,double Rcs,double Rc,int MaxAt){
    /*
    ある原子種に対する記述子を作成する．
    input
    -----------
    dist_wVec :: ある原子種からの全ての原子に対する相対ベクトル．
    * @param[in] atoms_index :: 自分で指定したindexのみを考える
    TODO :: 現状intraとinterを分けているのでこうなっているが，その区別がいらないならinputは"C"とかにするとわかりやすい．
    MaxAt :: 最大の原子数
    */
    // 相対ベクトルdist_wVecのうち，自分自身との距離0を除き，かつatoms_indexに含まれるものだけを取り出す．
    // TODO :: もしdrsの中に0のものがあったらそれを排除したい．これはこれはローンペア計算の時に重要
    std::vector<Eigen::Vector3d> drs;
    for (int l = 1; l < atoms_index.size(); l++){ // 0は自分自身なので除外して1スタートとする．
        drs.push_back(dist_wVec[atoms_index[l]]);
    }
    // drsの絶対値を求める．
    std::vector<double> drs_abs;
    for (int j = 0; j < drs.size(); j++){
        drs_abs.push_back(drs[j].norm());
    }
    // カットオフ関数fsをかけた1/drsを計算する．
    std::vector<double> drs_inv;
    for (int j = 0; j < drs.size(); j++){
        drs_inv.push_back(fs(drs_abs[j],Rcs,Rc));
    }
    // 記述子（f(1,x/r,y/r,z/r)）を計算する．4次元ベクトルの配列
    std::vector<std::vector<double> > dij;
    for (int j = 0; j < drs.size(); j++){
        dij.push_back({drs_inv[j], drs_inv[j]*drs[j][0]/drs_abs[j], drs_inv[j]*drs[j][1]/drs_abs[j], drs_inv[j]*drs[j][2]/drs_abs[j]});
    }
    // dijを要素drs_invの大きい順（つまりdrsが小さい順）にソートする．要素としては4次元ベクトルの第0成分でのソート
    // https://qiita.com/Arusu_Dev/items/c36cdbc41fc77531205c
    std::sort(dij.begin(), dij.end(), [](const std::vector<double> &alpha,const std::vector<double> &beta){return alpha[0] < beta[0];});
    // 原子数がMaxAtよりも少なかったら０埋めして固定長にする。1原子あたり4要素(1,x/r,y/r,z/r)なので4*MaxAtの要素だけ保守する．
    // !! 多分だけど，現状では要素数が十分多いので要素を4*MaxAtにカットするだけで大丈夫だと思う．
    std::vector<double> dij_desc;
    if (dij.size() < MaxAt){
        for (int i = 0; i< dij.size(); i++){
            dij_desc.push_back(dij[i][0]);
            dij_desc.push_back(dij[i][1]);
            dij_desc.push_back(dij[i][2]);
            dij_desc.push_back(dij[i][3]);
        }
        for (int i = 0; i < MaxAt - dij.size(); i++){ 
            dij_desc.push_back(0); // 0埋め
            dij_desc.push_back(0);
            dij_desc.push_back(0);
            dij_desc.push_back(0);
        }
    } else {
        for (int i = 0; i< MaxAt; i++){
            dij_desc.push_back(dij[i][0]);
            dij_desc.push_back(dij[i][1]);
            dij_desc.push_back(dij[i][2]);
            dij_desc.push_back(dij[i][3]);
        }
    }
    return dij_desc;
};

std::vector<double> raw_get_desc_bondcent(Atoms atoms, Eigen::Vector3d bond_center, int mol_id, std::vector<std::vector<double> > UNITCELL_VECTORS, int NUM_MOL_ATOMS){
    /*
    ボンドセンター用の記述子を作成
    TODO : 引数が煩雑すぎるので見直したい．mol_idが必要なのはあまり賢くない．
    ######Inputs########
    atoms : ASE atom object 構造の入力
    Rcs : float inner cut off [ang. unit]
    Rc  : float outer cut off [ang. unit] 
    MaxAt : int 記述子に記載する原子数（これにより固定長の記述子となる）
    #bond_center : vector 記述子を計算したい結合の中心
    mol_id : 対象のbond_centerがどの分子に属するかを示す． 
    ######Outputs#######
    Desc : 原子番号,[List O原子のSij x MaxAt : H原子のSij x MaxAt] x 原子数 の二次元リストとなる
    */
    double Rcs = 4.0; // [ang. unit] TODO : hard code
    double Rc  = 6.0; // [ang. unit] TODO : hard code
    double MaxAt = 12; // とりあえずは12個の原子で良いはず．
    // ボンドセンターを追加したatoms
    Atoms atoms_w_bc = raw_make_atoms_with_bc(bond_center,atoms, UNITCELL_VECTORS);
    // ボンドセンターが含まれている分子の原子インデックスを取得
    std::vector<int> atoms_in_molecule(NUM_MOL_ATOMS);
    for (int i = 0; i < NUM_MOL_ATOMS; i++){
        atoms_in_molecule[i] = i + mol_id*NUM_MOL_ATOMS + 1; //結合中心を先頭に入れたAtomsなので+1が必要．
    }
    // 各原子の記述子を作成するにあたり，原子のindexを計算する．
    std::vector<int> atomic_numbers = atoms_w_bc.get_atomic_numbers();
    std::vector<int> Catoms_intra;
    std::vector<int> Catoms_inter;
    std::vector<int> Hatoms_intra;
    std::vector<int> Hatoms_inter;
    std::vector<int> Oatoms_intra;
    std::vector<int> Oatoms_inter;


    // TODO :: 現状intraとinterで分けているが，将来的には分けなくて良くなる．
    // 現状ではatoms_in_moleculeにはいっているかどうかの判定が必要．c++でこの判定はstd::findで可能．
    for (int i = 0; i < atomic_numbers.size();i++){
        bool if_bc_in_molecule = std::find(atoms_in_molecule.begin(), atoms_in_molecule.end(), i) != atoms_in_molecule.end();
        if        (atomic_numbers[i] == 6 && if_bc_in_molecule){
            Catoms_intra.push_back(i);
        } else if (atomic_numbers[i] == 6 && !(if_bc_in_molecule)){
            Catoms_inter.push_back(i);
        } else if (atomic_numbers[i] == 1 && if_bc_in_molecule){
            Hatoms_intra.push_back(i);
        } else if (atomic_numbers[i] == 1 && !(if_bc_in_molecule)){
            Hatoms_inter.push_back(i);
        } else if (atomic_numbers[i] == 8 && if_bc_in_molecule){
            Oatoms_intra.push_back(i);
        } else if (atomic_numbers[i] == 8 && !(if_bc_in_molecule)){
            Oatoms_inter.push_back(i);
        }
    }
    // 全ての原子との距離を求める．この際0-0間距離も含まれる．
    std::vector<int> range_at_list(atomic_numbers.size()); // (0~atomic_numbersまでの整数を格納．numpy.arangeと同等．)
    std::iota(range_at_list.begin(),range_at_list.end(), 0); // https://codezine.jp/article/detail/8778
    auto dist_wVec = raw_get_distances_mic(atoms_w_bc,0, range_at_list, true, true) ;

    auto dij_C_intra=calc_descripter(dist_wVec, Catoms_intra, Rcs,Rc,MaxAt);    // for C atoms (intra) 
    auto dij_H_intra=calc_descripter(dist_wVec, Hatoms_intra, Rcs,Rc,MaxAt);    // for H atoms (intra)
    auto dij_O_intra=calc_descripter(dist_wVec, Oatoms_intra, Rcs,Rc,MaxAt);    // for O  atoms (intra)
    auto dij_C_inter=calc_descripter(dist_wVec, Catoms_inter, Rcs,Rc,MaxAt);    // for C atoms (inter)
    auto dij_H_inter=calc_descripter(dist_wVec, Hatoms_inter,Rcs,Rc,MaxAt);    // for H atoms (inter)
    auto dij_O_inter=calc_descripter(dist_wVec, Oatoms_inter,Rcs,Rc,MaxAt);    // for O atoms (inter)
    // 連結する dij_C_intra+dij_H_intra+dij_O_intra+dij_C_inter+dij_H_inter+dij_O_inter
    dij_C_intra.insert(dij_C_intra.end(), dij_H_intra.begin(), dij_H_intra.end()); // 連結
    dij_C_intra.insert(dij_C_intra.end(), dij_O_intra.begin(), dij_O_intra.end()); // 連結
    dij_C_intra.insert(dij_C_intra.end(), dij_C_inter.begin(), dij_C_inter.end()); // 連結
    dij_C_intra.insert(dij_C_intra.end(), dij_H_inter.begin(), dij_H_inter.end()); // 連結
    dij_C_intra.insert(dij_C_intra.end(), dij_O_inter.begin(), dij_O_inter.end()); // 連結
    return dij_C_intra;
};

std::vector<std::vector<double> > raw_calc_bond_descripter_at_frame(Atoms atoms_fr, std::vector<std::vector< Eigen::Vector3d> > list_bond_centers, std::vector<int> bond_index, int NUM_MOL, std::vector<std::vector<double> > UNITCELL_VECTORS, int NUM_MOL_ATOMS){
    /* 
    1つのframe中の全てのボンドの記述子を計算する
    */
    std::vector<std::vector<double> > Descs;
    if (bond_index.size() != 0){  // bond_indexが0でなければ計算を実行
        auto cent_mol   = find_specific_bondcenter(list_bond_centers, bond_index); // 特定ボンドのBCの座標だけ取得
        for (int i = 0; i < cent_mol.size(); i++){
            int mol_id = i % NUM_MOL; // len(bond_index) # 対応する分子ID（mol_id）を出すように書き直す．ボンドが1分子内に複数ある場合，その数で割らないといけない．（メタノールならCH結合が3つあるので3でわる）
            auto dij = raw_get_desc_bondcent(atoms_fr, cent_mol[i], mol_id, UNITCELL_VECTORS, NUM_MOL_ATOMS);
            Descs.push_back(dij);
        }
    }
    return Descs;
};


void write_2dvector_csv(){
    /*
    2dvectorをcsvファイルに書き込む
    */
}
