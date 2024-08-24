
// #define _DEBUG

// https://github.com/microsoft/vscode-cpptools/issues/7413
#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

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
#include <time.h>     // for clock()
#include <numeric> // std::iota
#include <tuple> // https://tyfkda.github.io/blog/2021/06/26/cpp-multi-value.html
// #include <boost/numeric/ublas/vector.hpp>
// #include <boost/numeric/ublas/matrix.hpp>
// #include <boost/numeric/ublas/io.hpp>
#include <Eigen/Core> // 行列演算など基本的な機能．
#include "numpy.hpp"
#include "npy.hpp"
// #include "numpy_quiita.hpp" // https://qiita.com/ka_na_ta_n/items/608c7df3128abbf39c89
// numpy_quiitaはsscanf_sが読み込めず，残念ながら現状使えない．
#include "atoms_io.hpp"
#include "include/printvec.hpp"
#include "descriptor.hpp"

#define DEBUG_PRINT_VARIABLE(var) std::cout << #var << std::endl;

/*
*/

Atoms make_ase_with_BCs(const std::vector<int> &ase_atomicnumber,const int NUM_MOL,const std::vector<std::vector<double> > UNITCELL_VECTORS, const std::vector<std::vector<Eigen::Vector3d> > &list_mol_coords, const std::vector<std::vector<Eigen::Vector3d> > &list_bond_centers){
    /*
    list_mol_coordsとlist_bond_centersから新しいase.atomsを作成する．
    現状だと単にase.atomsを作成するだけで，用途としてはase.io.writeで保存するくらいだ．

    注意点として，push_backする順番に気をつけたい．文仕事に，原子，ボンドセンター(He割り当て)の順で割り当てる．
    基本的に，普通の扱いでは分けている分子ごとの部分をフラットにするだけ．

    TODO :: ase_atomicnumberとNUM_MOLの扱いは難しい．一つの分子のみ（混合溶液でない）ならread_molから原子のリストを持ってきても良い．
    TODO :: しかし，もっとも一般的なのはase_atoms.get_atomic_numbers()を使って一つづつappendしていくこと．

    TODO :: push_backを使っているので遅いかも．ただしここはまあpush_backでも良いだろう．
    */
    //Atoms make_ase_with_BCs( ){
    // std::vector<std::vector<Eigen::Vector3d> > list_mol_coords;
    // std::vector<std::vector<Eigen::Vector3d> > list_bond_centers;

    std::vector<Eigen::Vector3d> new_coord;
    std::vector<int> new_atomic_num;
    // std::vector<std::vector<int>> list_atomic_nums = ase_atomicnumber.reshape(NUM_MOL, -1);
    for (int i = 0, n=list_mol_coords.size(); i < n; i++) {//Loop over molecules
        for (int j = 0, size=list_mol_coords[i].size(); j < size; j++) {    //分子内の原子数に関するLoop
            new_coord.push_back(list_mol_coords[i][j]);
            new_atomic_num.push_back(ase_atomicnumber[i*list_mol_coords[i].size()+j]);
        }
        for (int j = 0, size=list_bond_centers[i].size(); j < size; j++) {    //分子内のボンドセンター数に関するLoop
            new_coord.push_back(list_bond_centers[i][j]);
            new_atomic_num.push_back(2); // bcにはHeを割り当てる．
        }
    }
    // Atomsを作成する．    
    Atoms new_atoms(new_atomic_num, new_coord, UNITCELL_VECTORS, {1,1,1});
    return new_atoms;
}

Atoms raw_make_atoms_with_bc(Eigen::Vector3d bond_center, const Atoms &aseatoms, std::vector<std::vector<double> > UNITCELL_VECTORS){
    /*
    TODO :: DEPRECATED ?
    ######INPUTS#######
    bond_center     # Eigen::Vector3d 記述子を求めたい結合中心の座標
    list_mol_coords # array  分子ごとの原子座標
    list_atomic_nums #array  分子ごとの原子座標
    */
    std::vector<Eigen::Vector3d> list_mol_coords=aseatoms.get_positions();
    std::vector<int> list_atomic_num=aseatoms.get_atomic_numbers();

    // 結合中点bond_centerを先頭においたAtomsオブジェクトを作成する
    list_mol_coords.insert(list_mol_coords.begin(), bond_center);
    list_atomic_num.insert(list_atomic_num.begin(), 79); // 結合中心のラベルはAu(79)とする
    Atoms WBC = Atoms(list_atomic_num,
                    list_mol_coords,        
                    UNITCELL_VECTORS,   
                    {1, 1, 1});
    return WBC;
};

std::vector<Eigen::Vector3d> get_coord_of_specific_bondcenter(const std::vector<std::vector<Eigen::Vector3d> > &list_bond_centers, std::vector<int> bond_index){
    /*
    list_bond_centersからbond_index情報をもとに特定のボンド（CHなど）だけ取り出す．
    TODO :: ここは将来的にはボンドをあらかじめ分割するようにして無くしてしまいたい．
    * @param[in] bond_index : 分子内のボンドセンターのindex（read_mol.ch_bond_indexなど．）
    * @return cent_mol : 特定のボンドの原子座標（(-1,3)型）
    
    TODO :: cent_molはボンドセンターの座標のリストで，要素数は（すべて同じ分子ならば）分子数×ボンドセンター数で計算できる．
    */
    std::vector<Eigen::Vector3d> cent_mol;
    // ボンドセンターの座標と双極子をappendする．
    for (int i = 0, i_size=list_bond_centers.size() ; i < i_size; i++) { //UnitCellの分子ごとに分割
        // chボンド部分（chボンドの重心をappend）
        for (int j = 0, j_size=bond_index.size(); j < j_size; j++) {    //分子内のボンドセンター数に関するLoop
#ifdef DEBUG
            std::cout << " print list_bond_centers[i][bond_index[j]] :: " << list_bond_centers[i][bond_index[j]][0] << std::endl;
#endif //! DEBUG
            cent_mol.push_back(list_bond_centers[i][bond_index[j]]);
        }
    }
    return cent_mol;
    };

// 2023/10/20 :: lonepair用に新しくget_coord_of_specific_lonepairを実装．list_molから
std::vector<Eigen::Vector3d> get_coord_of_specific_lonepair(const std::vector<std::vector<Eigen::Vector3d> > &list_atom_positions, std::vector<int> atom_index){
    /**
    list_bond_centersからbond_index情報をもとに特定のボンド（CHなど）だけ取り出す．
    * @param[in] list_atom_positions : [NUM_MOL, NUM_ATOM_PER_MOL, 3]型の配列
    * @param[in] atom_index : 分子内の原子ののindex（read_molの0スタートの原子indexと一致）
    * @return cent_mol : 特定のボンドの原子座標（(-1,3)型）
    
    TODO :: cent_molはボンドセンターの座標のリストで，要素数は（すべて同じ分子ならば）分子数×ボンドセンター数で計算できる．
    */
    std::vector<Eigen::Vector3d> cent_mol;
    // ボンドセンターの座標と双極子をappendする．
    for (int i = 0, i_size=list_atom_positions.size() ; i < i_size; i++) { //UnitCellの分子ごとに分割
        // chボンド部分（chボンドの重心をappend）
        for (int j = 0, j_size=atom_index.size(); j < j_size; j++) {    //分子内のボンドセンター数に関するLoop
#ifdef DEBUG
            std::cout << " print list_bond_centers[i][atom_index[j]] :: " << list_bond_centers[i][atom_index[j]][0] << std::endl;
#endif //! DEBUG
            cent_mol.push_back(list_atom_positions[i][atom_index[j]]);
        }
    }
    return cent_mol;
    };



double fs(double Rij,double Rcs,double Rc){
    /**
     * @brief cutoff function
     * Rij : float 原子間距離 [ang. unit] 
     * Rij : float 原子間距離 [ang. unit] 
     * Rc  : float outer cut off [ang. unit] 
     * sij value 
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

std::vector<double> calc_descripter(const std::vector<Eigen::Vector3d> &dist_wVec, const std::vector<int> &atoms_index,double Rcs,double Rc,int MaxAt){
    /**
    * @brief calculate descriptors for specific atomic index.
    input
    -----------
    * @param[in] dist_wVec :: ある原子種からの全ての原子に対する相対ベクトル．
    * @param[in] atoms_index :: 自分で指定したindexのみを考える（Catoms_intraなどのindex．0~NUM_ATOMで指定）
    TODO :: 1:現状intraとinterを分けているのでこうなっているが，その区別がいらないならinputは"C"とかにするとわかりやすい．

    TODO :: 2:そもそも，dist_wVecをinputにするのはわかりにくいと思う．atoms_indexとaseatomsがあればget_distancesでdist_wVEc[atoms_index]を作成可能．
    TODO :: 計算量ももとのコードと変わらない（もとのコードでは全ての原子との距離を計算しているので．）
    MaxAt :: 最大の原子数
    */
    // 相対ベクトルdist_wVecのうち，自分自身との距離0を除き，かつatoms_indexに含まれるものだけを取り出す．
    std::vector<Eigen::Vector3d> drs;
    for (int l = 0, size=atoms_index.size(); l < size; l++){ // atoms_indexもBCを含んだatomsでのindexなので0スタートでOK．
        if (dist_wVec[atoms_index[l]].norm() >  0.001){ // もしdrsの中に0のものがあったらそれを除外する．ローンペア計算時には絶対に存在する．
            drs.push_back(dist_wVec[atoms_index[l]]);
        }
    }
#ifdef DEBUG
    assert(drs.size() == atoms_index.size());
    std::cout << " drs.size() should be atoms_index.size() " << drs.size() << std::endl;
#endif //! DEBUG
    // drsの絶対値を求める．
    std::vector<double> drs_abs(drs.size());
    for (int j = 0; j < drs.size(); j++){
        drs_abs[j] = drs[j].norm();
    }
#ifdef DEBUG
    print_vec(drs, "drs");
#endif //! DEBUG
    // f(drs)（~1/drs）を求める．
    std::vector<double> drs_inv(drs.size());
    for (int j = 0; j < drs.size(); j++){
        // drs_inv.push_back(fs(drs_abs[j],Rcs,Rc));
        drs_inv[j] = fs(drs_abs[j],Rcs,Rc);
    }
    // 記述子（f(1,x/r,y/r,z/r)）を計算する．4次元ベクトルの配列
    // https://atcoder.jp/contests/APG4b/tasks/APG4b_t（多次元ベクトルの宣言）
    std::vector<std::vector<double> > dij(drs.size(),std::vector<double>(4));
    for (int j = 0; j < drs.size(); j++){
        dij[j] = {drs_inv[j], drs_inv[j]*drs[j][0]/drs_abs[j], drs_inv[j]*drs[j][1]/drs_abs[j], drs_inv[j]*drs[j][2]/drs_abs[j]};
    }
#ifdef DEBUG
    std::cout << "dij size " << dij.size() << " " << dij[0].size() << std::endl; 
#endif //! DEBUG
    // dijを要素drs_invの大きい順（つまりdrsが小さい順）にソートする．要素としては4次元ベクトルの第0成分でのソート
    // https://qiita.com/Arusu_Dev/items/c36cdbc41fc77531205c（ここは小さい順ソートなので符号を逆にした．）
    std::sort(dij.begin(), dij.end(), [](const std::vector<double> &alpha,const std::vector<double> &beta){return alpha[0] > beta[0];});
    // 原子数がMaxAtよりも少なかったら０埋めして固定長にする。1原子あたり4要素(1,x/r,y/r,z/r)なので4*MaxAtの要素だけ保守する．å
    // !! 最初に0で初期化しておけば，0埋めは不要．
    std::vector<double> dij_desc(4*MaxAt,0);
    // std::vector<double> dij_desc;
    if (int(dij.size()) < MaxAt){
        for (int i = 0; i< int(dij.size()); i++){
            dij_desc[i*4] = dij[i][0];
            dij_desc[i*4+1] = dij[i][1];
            dij_desc[i*4+2] = dij[i][2];
            dij_desc[i*4+3] = dij[i][3];
        }
    } else {
        for (int i = 0; i< MaxAt; i++){
            dij_desc[i*4] = dij[i][0];
            dij_desc[i*4+1] = dij[i][1];
            dij_desc[i*4+2] = dij[i][2];
            dij_desc[i*4+3] = dij[i][3];
        }
    }
    return dij_desc;
};

std::vector<double> raw_get_desc_bondcent(const Atoms &atoms, Eigen::Vector3d bond_center, int mol_id, std::vector<std::vector<double> > UNITCELL_VECTORS, int NUM_MOL_ATOMS,float Rcs, float Rc, int MaxAt){
    /**
     * @brief calculate a descriptor for a given bond center
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
    // double Rcs = 4.0; // [ang. unit] TODO : hard code
    // double Rc  = 6.0; // [ang. unit] TODO : hard code
    // int MaxAt = 12; // とりあえずは12個の原子で良いはず．
    // ボンドセンターを追加したatoms 
    Atoms atoms_with_bc = raw_make_atoms_with_bc(bond_center,atoms, UNITCELL_VECTORS);
    // 「ボンドセンターが含まれている分子(mol_id)」の原子インデックスを取得
    std::vector<int> atoms_in_molecule(NUM_MOL_ATOMS);
    for (int i = 0; i < NUM_MOL_ATOMS; i++){
        atoms_in_molecule[i] = i + mol_id*NUM_MOL_ATOMS + 1; //結合中心を先頭に入れたAtomsなので+1が必要．
    }
#ifdef DEBUG
    std::cout << " atoms_in_molecule :: ";
    for (int i=0; i< atoms_in_molecule.size(); i++){
        std::cout << atoms_in_molecule[i] << " ";
    }
    std::cout << std::endl;
#endif //! DEBUG

    // 各原子の記述子を作成するにあたり，原子のindexを計算する．
    std::vector<int> atomic_numbers = atoms_with_bc.get_atomic_numbers();
    std::vector<int> Catoms_intra, Catoms_inter, Hatoms_intra, Hatoms_inter, Oatoms_intra, Oatoms_inter;

    // TODO :: 現状intraとinterで分けているが，将来的には分けなくて良くなる．
    // ! 注意：Catoms_intraなども現在のBCを先頭においた状態でのindexとなっている．
    // 現状ではatoms_in_moleculeにはいっているかどうかの判定が必要．c++でこの判定はstd::findで可能．
    for (int i = 0; i < atomic_numbers.size();i++){ // 
        bool if_bc_in_molecule = std::find(atoms_in_molecule.begin(), atoms_in_molecule.end(), i) != atoms_in_molecule.end();
        if (i == 0){ continue; }    // 先頭は対象としているBCなのでスルーする．
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
        } else {
            std::cout << " Error : atomic number is not 1,6,8 :: " << atomic_numbers[i] << std::endl;
        }
    }
#ifdef DEBUG
    std::cout << "Catoms_intra :: ";
    for (int i=0; i< Catoms_intra.size(); i++){
        std::cout << Catoms_intra[i] << " ";
    }
    std::cout << std::endl;
#endif //! DEBUG

    // 全ての原子との距離を求める．この際0-0間距離も含まれる．
    std::vector<int> range_at_list(atomic_numbers.size()); // (0~atomic_numbersまでの整数を格納．numpy.arangeと同等．)
    std::iota(range_at_list.begin(),range_at_list.end(), 0); // https://codezine.jp/article/detail/8778
    auto dist_wVec = raw_get_distances_mic(atoms_with_bc,0,range_at_list, true, true) ;

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

std::vector<double> raw_get_desc_bondcent_allinone(const Atoms &atoms, Eigen::Vector3d bond_center, std::vector<std::vector<double> > UNITCELL_VECTORS, int NUM_MOL_ATOMS,float Rcs, float Rc, int MaxAt){
    /**
     * @brief calculate a descriptor for a given bond center
     * 
    TODO : 引数が煩雑すぎるので見直したい．
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

    // atoms with bondcent
    Atoms atoms_with_bc = raw_make_atoms_with_bc(bond_center,atoms, UNITCELL_VECTORS);

    // 各原子の記述子を作成するにあたり，原子のindexを計算する．
    std::vector<int> atomic_numbers = atoms_with_bc.get_atomic_numbers();
    std::vector<int> Catoms_all, Hatoms_all, Oatoms_all;

    // ! 注意：Catoms_intraなども現在のBCを先頭においた状態でのindexとなっている．
    // TODO :: これは絶対に重い計算になっているだろう．まずはpush_backをやめさせる．
    for (int i = 0, size=atomic_numbers.size(); i < size ;i++){ // 
        if (i == 0){ continue; }    // 先頭は対象としているBCなのでスルーする．
        if        (atomic_numbers[i] == 6){
            Catoms_all.push_back(i);
        }  else if (atomic_numbers[i] == 1){
            Hatoms_all.push_back(i);
        } else if (atomic_numbers[i] == 8 ){
            Oatoms_all.push_back(i);
        } else {
            std::cout << " Error : atomic number is not 1,6,8 :: " << atomic_numbers[i] << std::endl;
        }
    }
#ifdef DEBUG
    print_vec(Catoms_intra, "Catoms_intra");
#endif //! DEBUG

    // 全ての原子との距離を求める．この際0-0間距離も含まれる．
    std::vector<int> range_at_list(atomic_numbers.size()); // (0~atomic_numbersまでの整数を格納．numpy.arangeと同等．)
    std::iota(range_at_list.begin(),range_at_list.end(), 0); // https://codezine.jp/article/detail/8778
    auto dist_wVec = raw_get_distances_mic(atoms_with_bc,0,range_at_list, true, true) ;

    // 記述子の計算
    auto dij_C_all=calc_descripter(dist_wVec, Catoms_all, Rcs,Rc,MaxAt);    // for C atoms (all) 
    auto dij_H_all=calc_descripter(dist_wVec, Hatoms_all, Rcs,Rc,MaxAt);    // for H atoms (all)
    auto dij_O_all=calc_descripter(dist_wVec, Oatoms_all, Rcs,Rc,MaxAt);    // for O  atoms (all)

    // concat calculated descriptors dij_C_all+dij_H_all+dij_O_all
    // TODO :: 多分連結は効率が良くない．
    dij_C_all.insert(dij_C_all.end(), dij_H_all.begin(), dij_H_all.end()); // 連結
    dij_C_all.insert(dij_C_all.end(), dij_O_all.begin(), dij_O_all.end()); // 連結

    return dij_C_all;
};



std::vector<std::vector<double> > raw_calc_bond_descripter_at_frame(const Atoms &atoms_fr, const std::vector<std::vector< Eigen::Vector3d> > &list_bond_centers, std::vector<int> bond_index, int NUM_MOL, std::vector<std::vector<double> > UNITCELL_VECTORS, int NUM_MOL_ATOMS, std::string desctype,float Rcs, float Rc, int MaxAt){
    /**
     * @fn
     * 1つのframe中の全てのボンドの記述子を計算する
     * @param[in] atoms_fr : 1つのframeのAtoms
     * @param[in] bond_index : 計算したいbondのindex．read_mol.ch_bond_indexなどとして持ってくればOK．
     * @param[in] NUM_MOL : 分子の数
     * @param[in] desctype : 記述子の形を指定する．現状oldかallinone
     * @param[out] output : [NUM_BC_in_config, 288]型の配列
     * 
    TODO :: descs.push_backをやめて代入形式にする．Descsの宣言時にサイズを決める必要があり，それにはraw_get_desc_bondcentの形を決め打ちする必要がある．
    TODO :: 現状だと288次元で固定されているがそれで良いのかどうか，一回考えてみる必要がある．
    */
    // Error if len(bond_index)=0
    if (bond_index.size() == 0) {
        return {{}};
    }
    // get specific bond center coordinates from bond_index
    std::vector<Eigen::Vector3d> list_bc_coords = get_coord_of_specific_bondcenter(list_bond_centers, bond_index); 
    std::vector<std::vector<double> > Descs(list_bc_coords.size()); // return value
    if (desctype == "allinone") {
        #pragma omp for
        for (int i = 0; i < int(list_bc_coords.size()); i++){
            Descs[i] = raw_get_desc_bondcent_allinone(atoms_fr, list_bc_coords[i], UNITCELL_VECTORS, NUM_MOL_ATOMS,Rcs,Rc,MaxAt);
        }
    } else if (desctype == "old"){
        for (int i = 0; i < int(list_bc_coords.size()); i++){
            int mol_id = i % NUM_MOL / bond_index.size(); // len(bond_index) # 対応する分子ID（mol_id）を出すように書き直す．ボンドが1分子内に複数ある場合，その数で割らないといけない．（メタノールならCH結合が3つあるので3でわる）
#ifdef DEBUG
            std::cout << "mol_id: " << mol_id << std::endl;
#endif //! DEBUG
            auto dij = raw_get_desc_bondcent(atoms_fr, list_bc_coords[i], mol_id, UNITCELL_VECTORS, NUM_MOL_ATOMS,Rcs,Rc,MaxAt);
            Descs.push_back(dij);
        }
    } else {
        std::cerr << "ERROR : desctype is not defined. " << std::endl;
    }
    return Descs;
};



std::vector<std::vector<int>> raw_find_atomic_index(const Atoms &aseatoms, int atomic_number, int NUM_MOL) {
    /*
    aseatomsのうち，指定のatomic_number（O原子なら8）の原子のindexをNUM_MOL分割して返す．
    ! o_listが実装された今，使わなくても大丈夫な関数ではある．
    * @param aseatoms :: Atoms
    */
    std::vector<int> list_atomic_nums = aseatoms.get_atomic_numbers();
    int NUM_MOL_ATOMS = list_atomic_nums.size() / NUM_MOL;
    std::vector<std::vector<int>> at_list;
    for (int i = 0; i < NUM_MOL; i++) {
        // list_atomic_numのうち，i番目の分子の原子のindexを取得
        std::vector<int> js(list_atomic_nums.begin() + i * NUM_MOL_ATOMS, list_atomic_nums.begin() + (i + 1) * NUM_MOL_ATOMS);
        std::vector<int> temp;
        for (int j = 0; j < js.size(); j++) {
            if (js[j] == atomic_number) {
                temp.push_back(j);
            }
        }
        at_list.push_back(temp);
    }
    return at_list;
}

std::vector<Eigen::Vector3d> find_specific_lonepair(const std::vector<std::vector<Eigen::Vector3d> > &list_mol_coords, const Atoms &aseatoms, int atomic_number, int NUM_MOL) {
    /**
    * atomic_numberで指定される原子番号を持つ原子の座標をcent_mol[分子index][原子index]の形で返す．
    * @param[in] aseatoms :: 入力とするのAtomsオブジェクト
    * @param[in] atomic_number :: 指定する原子番号
    * @param[in] list_mol_coords :: 原子の座標リスト
    */
    std::vector<std::vector<int> > at_list = raw_find_atomic_index(aseatoms, atomic_number, NUM_MOL);
    std::vector<Eigen::Vector3d> list_coord_lonepair;
    
    for (int i = 0, i_size=at_list.size(); i < i_size; i++) { // 分子ごとのループ
        // std::vector<Eigen::Vector3d> tmp; 
        for (int j = 0, j_size=at_list[i].size(); j < j_size; j++) { // 分子内原子のループ
            list_coord_lonepair.push_back(list_mol_coords[i][at_list[i][j]]);
        }
        // cent_mol.push_back(tmp);
    }
    return list_coord_lonepair;
};


std::vector<Eigen::Vector3d> find_specific_lonepair_select(const std::vector<std::vector<Eigen::Vector3d> > &list_mol_coords, std::vector<int> at_list, int NUM_MOL) {
    /**
    * atomic_numberで指定される原子番号を持つ原子の座標をcent_mol[分子index][原子index]の形で返す．
    * 注意！！ get_coord_of_specific_lonepair/get_coord_of_bondcenterと同じ機能を持つ関数．
    * 注意！！ find_specific_lonepairはraw_find_atomic_indexで分子ごとに原子indexを取得している．
    * @param[in] list_mol_coords :: 原子の座標リストを，[分子index][原子index][3次元座標]の形で格納したもの
    * @param[in] at_list :: 分子内の原子のindexのリスト（分子内の，ということが重要で，o_indexやcoc_indexなどを指定する．
    * @param[in] NUM_MOL :: 分子の数
    * @param[out] list_coord_lonepair :: 指定された原子番号の原子の座標のリスト
    * 
    */
    std::vector<Eigen::Vector3d> list_coord_lonepair;
    // int num_atoms_per_mol = list_mol_coords[0].size(); //分子あたりの原子数，
    
    for (int mol_id = 0; mol_id< NUM_MOL; mol_id++) { // 分子ごとのループ
        for (int i = 0, at_index_size=at_list.size(); i < at_index_size; i++) { // 分子内の原子に関するループ
            list_coord_lonepair.push_back(list_mol_coords[mol_id][at_list[i]]); 
        };
    };
    return list_coord_lonepair;
};


std::vector<double> raw_get_desc_lonepair(const Atoms &atoms, Eigen::Vector3d lonepair_coord, int mol_id, std::vector<std::vector<double> > UNITCELL_VECTORS, int NUM_MOL_ATOMS,
    float Rcs, float Rc, int MaxAt){
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
    // double Rcs = 4.0; // [ang. unit] TODO : hard code
    // double Rc  = 6.0; // [ang. unit] TODO : hard code
    // int MaxAt = 12; // とりあえずは12個の原子で良いはず．
    // ボンドセンターを追加したatoms （同一のlonepair座標が含まれていることに注意）
    Atoms atoms_with_bc = raw_make_atoms_with_bc(lonepair_coord,atoms, UNITCELL_VECTORS);
    // 「ボンドセンターが含まれている分子(mol_id)」の原子インデックスを取得
    std::vector<int> atoms_in_molecule(NUM_MOL_ATOMS);
    for (int i = 0; i < NUM_MOL_ATOMS; i++){
        atoms_in_molecule[i] = i + mol_id*NUM_MOL_ATOMS + 1; //結合中心を先頭に入れたAtomsなので+1が必要．
    }
#ifdef DEBUG
    print_vec(atoms_in_molecule, "atoms_in_molecule");
#endif //! DEBUG

    // 各原子の記述子を作成するにあたり，原子のindexを計算する．
    std::vector<int> atomic_numbers = atoms_with_bc.get_atomic_numbers();
    std::vector<int> Catoms_intra, Catoms_inter, Hatoms_intra, Hatoms_inter, Oatoms_intra, Oatoms_inter;

    // こちらはintraとinterで分けた関数になっている．
    // ! 注意：Catoms_intraなども現在のBCを先頭においた状態でのindexとなっている．
    // 現状ではatoms_in_moleculeにはいっているかどうかの判定が必要．c++でこの判定はstd::findで可能．
    for (int i = 0; i < int(atomic_numbers.size());i++){ // 
        bool if_bc_in_molecule = std::find(atoms_in_molecule.begin(), atoms_in_molecule.end(), i) != atoms_in_molecule.end();
        if (i == 0){ continue; }    // 先頭は対象としているBCなのでスルーする．
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
        } else {
            std::cout << "Error : atomic number is not 1,6,8 :: " << atomic_numbers[i] << std::endl;
        }
    }
#ifdef DEBUG
    std::cout << "Catoms_intra :: ";
    for (int i=0; i< Catoms_intra.size(); i++){
        std::cout << Catoms_intra[i] << " ";
    }
    std::cout << std::endl;
#endif //! DEBUG

    // 全ての原子との距離を求める．この際0-0間距離も含まれる．
    std::vector<int> range_at_list(atomic_numbers.size()); // (0~atomic_numbersまでの整数を格納．numpy.arangeと同等．)
    std::iota(range_at_list.begin(),range_at_list.end(), 0); // https://codezine.jp/article/detail/8778
    auto dist_wVec = raw_get_distances_mic(atoms_with_bc,0, range_at_list, true, true) ;
    // 記述子を計算
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

std::vector<double> raw_get_desc_lonepair_allinone(const Atoms &atoms, Eigen::Vector3d lonepair_coord, std::vector<std::vector<double> > UNITCELL_VECTORS, int NUM_MOL_ATOMS,float Rcs, float Rc, int MaxAt){
    /*
    ボンドセンター用の記述子を作成
    こっちはallinoneで分子内外を分けない
    ######Inputs########
    atoms : ASE atom object 構造の入力
    Rcs : float inner cut off [ang. unit]
    Rc  : float outer cut off [ang. unit] 
    MaxAt : int 記述子に記載する原子数（これにより固定長の記述子となる）
    #bond_center : vector 記述子を計算したい結合の中心 
    ######Outputs#######
    Desc : 原子番号,[List O原子のSij x MaxAt : H原子のSij x MaxAt] x 原子数 の二次元リストとなる
    */
    // double Rcs = 4.0; // [ang. unit] TODO : hard code
    // double Rc  = 6.0; // [ang. unit] TODO : hard code
    // int MaxAt = 24; // !! 12*2 = 24
    // ボンドセンターを追加したatoms （同一のlonepair座標が含まれていることに注意）
    Atoms atoms_with_bc = raw_make_atoms_with_bc(lonepair_coord,atoms, UNITCELL_VECTORS);

    // 各原子の記述子を作成するにあたり，原子のindexを取得する．
    std::vector<int> atomic_numbers = atoms_with_bc.get_atomic_numbers();
    std::vector<int> Catoms_all, Hatoms_all, Oatoms_all;

    // ! 注意：Catoms_intraなども現在のBCを先頭においた状態でのindexとなっている．
    // 現状ではatoms_in_moleculeにはいっているかどうかの判定が必要．c++でこの判定はstd::findで可能．
    for (int i = 0; i < int(atomic_numbers.size());i++){ // 
        if (i == 0){ continue; }    // 先頭は対象としているBCなのでスルーする．
        if        (atomic_numbers[i] == 6 ){
            Catoms_all.push_back(i);
        } else if (atomic_numbers[i] == 1 ){
            Hatoms_all.push_back(i);
        } else if (atomic_numbers[i] == 8 ){
            Oatoms_all.push_back(i);
        } else {
            std::cout << "Error : atomic number is not 1,6,8 :: " << atomic_numbers[i] << std::endl;
        }
    }
#ifdef DEBUG
    print_vec(Catoms_intra, "Catoms_intra");
#endif //! DEBUG

    // 全ての原子との距離を求める．この際0-0間距離も含まれる．
    std::vector<int> range_at_list(atomic_numbers.size()); // (0~atomic_numbersまでの整数を格納．numpy.arangeと同等．)
    std::iota(range_at_list.begin(),range_at_list.end(), 0); // https://codezine.jp/article/detail/8778
    auto dist_wVec = raw_get_distances_mic(atoms_with_bc,0, range_at_list, true, true) ;
    // 記述子を計算
    auto dij_C_all=calc_descripter(dist_wVec, Catoms_all, Rcs,Rc,MaxAt);    // for C atoms 
    auto dij_H_all=calc_descripter(dist_wVec, Hatoms_all, Rcs,Rc,MaxAt);    // for H atoms 
    auto dij_O_all=calc_descripter(dist_wVec, Oatoms_all, Rcs,Rc,MaxAt);    // for O  atoms

    // 連結する dij_C_all+dij_H_all+dij_O_all
    dij_C_all.insert(dij_C_all.end(), dij_H_all.begin(), dij_H_all.end()); // 連結
    dij_C_all.insert(dij_C_all.end(), dij_O_all.begin(), dij_O_all.end()); // 連結

    return dij_C_all;
};


std::vector<std::vector<double> > raw_calc_lonepair_descripter_at_frame(const Atoms &atoms_fr, const std::vector<std::vector<Eigen::Vector3d> > &list_mol_coords, std::vector<int> at_list, int NUM_MOL, int atomic_number, std::vector<std::vector<double>> UNITCELL_VECTORS, int NUM_MOL_ATOMS, std::string desctype,float Rcs, float Rc, int MaxAt) {
    /*
    
    * @param[in] desctype :: 記述子のタイプを指定する（old, allinone）
    * @param[in] at_list :: 計算したい原子のindex．read_mol.ch_at_indexなどとして持ってくればOK．
    * @param[in] atomic_number :: 計算したい原子の原子番号．
    */
    // std::vector<int> at_list2 = raw_find_atomic_index(atoms_fr, atomic_index, NUM_MOL);
    
    std::vector<Eigen::Vector3d> list_lonepair_coords = find_specific_lonepair(list_mol_coords, atoms_fr, atomic_number, NUM_MOL); 
    std::vector<std::vector<double> > Descs(list_lonepair_coords.size()); // return value
    // if len(at_list)=0, return 0
    if (at_list.size() == 0) {
        return {{}};
    }
    if (desctype == "allinone"){
        #pragma omp for
        for (int i = 0; i < int(list_lonepair_coords.size()); i++){
            Descs[i] = raw_get_desc_lonepair_allinone(atoms_fr, list_lonepair_coords[i], UNITCELL_VECTORS, NUM_MOL_ATOMS,Rcs,Rc,MaxAt);
        }
    } else if (desctype == "old"){
        int i = 0;
        for (auto lonepair_coord : list_lonepair_coords) {
            int mol_id = i % NUM_MOL / at_list.size();
            Descs.push_back(raw_get_desc_lonepair(atoms_fr, lonepair_coord, mol_id, UNITCELL_VECTORS, NUM_MOL_ATOMS));
            i++;
        }
    } else {
        std::cerr << "ERROR : desctype is not defined. " << std::endl;
    }
    return Descs;
}


std::vector<std::vector<double> > raw_calc_lonepair_descripter_select_at_frame(const Atoms &atoms_fr, const std::vector<std::vector<Eigen::Vector3d> > &list_mol_coords, std::vector<int> at_list, int NUM_MOL, std::vector<std::vector<double>> UNITCELL_VECTORS, int NUM_MOL_ATOMS, std::string desctype) {
    /**
    * @fn raw_calc_lonepair_descripter_at_frameは原子番号で指定される原子の座標を取得するが，こちらは原子のindexで指定される（at_list）原子の座標を取得する．
    * @param[in] desctype :: 記述子のタイプを指定する（old, allinone）
    * @param[in] at_list :: 計算したい原子のindex．read_mol.coc_indexなどを持ってくればOK．
    * @param[in] atomic_number :: 計算したい原子の原子番号．
    * TODO :: サイズがNUM_MOL*at_list.size()になっているかをtest関数でチェック
    */
    // std::vector<int> at_list2 = raw_find_atomic_index(atoms_fr, atomic_index, NUM_MOL);
    
    std::vector<std::vector<double> > Descs;
    std::vector<Eigen::Vector3d> list_lonepair_coords = find_specific_lonepair_select(list_mol_coords, at_list, NUM_MOL); //! at_listで与えられる原子の座標を計算する．
    // std::cout << " DEBUG list_lonepair_coords.size() : " << list_lonepair_coords.size() << std::endl;
    if (at_list.size() != 0) { // at_listが非ゼロなら記述子計算を実行
        if (desctype == "allinone"){
            for (auto lonepair_coord : list_lonepair_coords) { // lonepair原子の座標を取得（全分子について）
                Descs.push_back(raw_get_desc_lonepair_allinone(atoms_fr, lonepair_coord, UNITCELL_VECTORS, NUM_MOL_ATOMS));
            }
        } else if (desctype == "old"){
            int i = 0;
            for (auto lonepair_coord : list_lonepair_coords) {
                int mol_id = i % NUM_MOL / at_list.size();
                Descs.push_back(raw_get_desc_lonepair(atoms_fr, lonepair_coord, mol_id, UNITCELL_VECTORS, NUM_MOL_ATOMS));
                i++;
            }
        } else {
            std::cerr << "ERROR : desctype is not defined. " << std::endl;
        }
    }
    return Descs;
}
