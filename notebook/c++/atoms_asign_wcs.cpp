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
// #include "atoms_core.cpp" // !! これをよむとまずい？
#include "atoms_io.cpp"
// #include "mol_core.cpp"

#define DEBUG_PRINT_VARIABLE(var) std::cout << #var << std::endl;

/*
*/

std::vector<Eigen::Vector3d> raw_calc_mol_coord_mic_onemolecule(std::vector<int> mol_inds, std::vector<std::vector<int>> bonds_list_j, const Atoms &aseatoms, const read_mol &itp_data) {
    /*
    1つの分子のpbc-mol計算を実施し，
        - 分子座標
    の計算を行う．
    * @param[in] mol_inds :: 分子部分を示すインデックス
    * @param[in] bonds_list_j :: 分子内のボンドのリスト
    TODO :: raw_get_pbc_molを実装したのでデバックを！！
    */
   // 通常のraw_get_distances_micを使って，mol_inds[0]からmol_indsへの距離を計算する．
    // std::vector<Eigen::Vector3d> vectors = raw_get_distances_mic(aseatoms, mol_inds[itp_data.representative_atom_index], mol_inds, true, true);
    // raw_get_pbc_molを利用する場合
    std::vector<Eigen::Vector3d> vectors = raw_get_pbc_mol(aseatoms, mol_inds, bonds_list_j, itp_data);

#ifdef DEBUG
    std::cout << "vectors..." << std::endl;
    for (int i = 0; i < vectors.size(); i++) {
        std::cout << std::setw(10) << vectors[i][0] << vectors[i][1] <<  vectors[i][2] << std::endl;
    }
#endif // !DEBUG
    // mol_inds[itp_data.representative_atom_index]の座標を取得する．
    Eigen::Vector3d R0 = aseatoms.get_positions()[mol_inds[itp_data.representative_atom_index]];

    // // mol_indsを0から始まるように変換する．
    // std::vector<int> mol_inds_from_zero(mol_inds.size());
    // for (int i = 0; i < mol_inds.size(); i++) {
    //     // mol_inds_from_zero.push_back(mol_inds[i] - mol_inds[0]); 
    //     mol_inds_from_zero[i]=(mol_inds[i] - mol_inds[0]);
    // }

    // // 分子の座標を再計算する．
    // std::vector<Eigen::Vector3d> mol_coords(mol_inds_from_zero.size());
    // for (int k = 0; k < mol_inds_from_zero.size(); k++) {
    //     // Eigen::Vector3d mol_coord = R0 + vectors[mol_inds_from_zero[k]];
    //     // mol_coords.push_back(mol_coord);
    //     mol_coords[k] = R0 + vectors[mol_inds_from_zero[k]];
    // }

    // 分子の座標をR0基準に再計算する．
    // mol_indsを0から始まるように変換して計算する．
    std::vector<Eigen::Vector3d> mol_coords(mol_inds.size());
    for (int k = 0, size=mol_inds.size(); k < size; k++) {
        // Eigen::Vector3d mol_coord = R0 + vectors[mol_inds_from_zero[k]];
        // mol_coords.push_back(mol_coord);
        mol_coords[k] = R0 + vectors[mol_inds[k]-mol_inds[0]];
    }
    return mol_coords;
}

std::vector<Eigen::Vector3d> raw_calc_bc_mic_onemolecule(const std::vector<int> mol_inds, const std::vector<std::vector<int>> bonds_list_j, const std::vector<Eigen::Vector3d> &mol_coords) {
    /*
        すでに計算された分子座標を使って，ボンドセンターを計算する．
        この時，入力となる分子座標mol_coordは絶対に1分子中の座標でないといけない．
        - ボンドセンター座標
        TODO :: この関数もbonds_list_jを使う必要がない．ただのbonds_listから計算可能．
    */
    // ボンドリストが0から始まるように変換する．
    // TODO :: そもそもbonds_list_jを使う必要ないよね？ mol_indsから直接計算できるもんな．．．
    std::vector<std::vector<int>> bonds_list_from_zero(bonds_list_j.size());
    for (int i = 0; i < bonds_list_j.size(); i++) {
        std::vector<int> bond={bonds_list_j[i][0] - mol_inds[0], bonds_list_j[i][1] - mol_inds[0]};
        bonds_list_from_zero[i] = bond;        
    }
    // ボンドセンターを計算する．
    std::vector<Eigen::Vector3d> bond_centers(bonds_list_from_zero.size());
    for (int l = 0; l < bonds_list_from_zero.size(); l++) {
        // Eigen::Vector3d bc = R0 + (vectors.col(bonds_list_from_zero[l][0]) + vectors.col(bonds_list_from_zero[l][1])) / 2.0;
        Eigen::Vector3d bc = (mol_coords[bonds_list_from_zero[l][0]] + mol_coords[bonds_list_from_zero[l][1]]) / 2.0;
        if ((mol_coords[bonds_list_from_zero[l][0]] - mol_coords[bonds_list_from_zero[l][1]]).norm() > 2.0) { // bond length is too long
            std::cout << "WARNING :: bond length is too long !! :: " << bonds_list_from_zero[l][0] << " " << bonds_list_from_zero[l][1] << " " <<(mol_coords[bonds_list_from_zero[l][0]] - mol_coords[bonds_list_from_zero[l][1]]).norm() << std::endl;
        }
        // bond_centers.push_back(bc);
        bond_centers[l] = bc;
    }
    return bond_centers;
}

std::tuple<std::vector<std::vector<Eigen::Vector3d> >, std::vector<std::vector<Eigen::Vector3d> > > raw_aseatom_to_mol_coord_and_bc(const Atoms &ase_atoms, const std::vector<std::vector<int>> bonds_list, const read_mol &itp_data, const int NUM_MOL_ATOMS, const int NUM_MOL) {
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
    std::vector<std::vector<Eigen::Vector3d> > list_mol_coords(NUM_MOL); 
    std::vector<std::vector<Eigen::Vector3d> > list_bond_centers(NUM_MOL); 
    
    // 1分子内のindexを取得する．
    std::vector<int> mol_at0(NUM_MOL_ATOMS);
    for (int i = 0; i < NUM_MOL_ATOMS; i++) {
        mol_at0[i] = i;
    }
    // 1config内の原子indexを取得する．
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
        std::vector<Eigen::Vector3d> bond_centers = raw_calc_bc_mic_onemolecule(mol_ats[j], unit_cell_bonds[j], mol_coords); //! mol_coordsを利用している
        list_mol_coords[j] = mol_coords;
        list_bond_centers[j] = bond_centers;
    }
    return {list_mol_coords,list_bond_centers};
}
