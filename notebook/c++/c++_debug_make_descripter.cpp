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
// #include "atoms_core.cpp" // !! これを入れるとエラーが出る？
// #include "atoms_io.cpp"   // !! これを入れるとエラーが出る？
// #include "mol_core.cpp"
#include "atoms_asign_wcs.cpp"


int main() {

    //! 原子数の取得
    int NUM_ATOM = raw_cpmd_num_atom("pg_gromacs.xyz");
    //! 格子定数の取得
    std::vector<std::vector<double> > UNITCELL_VECTORS = raw_cpmd_get_unitcell_xyz("pg_gromacs.xyz");
    //! xyzファイルから座標リストを取得
    std::vector<Atoms> atoms_list = ase_io_read("pg_gromacs.xyz");

    //! ボンドリストの取得
    // TODO :: 現状では，ボンドリストはmol_core.cpp内で定義されている．（こういうブラックボックスをなんとかしたい）
    read_mol test_read_mol;

    //! test raw_aseatom_to_mol_coord_and_bc
    int NUM_MOL_ATOMS = test_read_mol.num_atoms_per_mol;
    int NUM_MOL = int(NUM_ATOM/NUM_MOL_ATOMS); // UnitCell中の総分子数
    std::cout << "NUM_MOL :: " << NUM_MOL << std::endl;
    std::cout << "NUM_MOL_ATOMS :: " << NUM_MOL_ATOMS << std::endl;
    // for (int i=0; i< atoms_list.size();i++){
    for (int i=0; i< 10;i++){
        auto test_mol_bc = raw_aseatom_to_mol_coord_and_bc(atoms_list[i], test_read_mol.bonds_list, test_read_mol, NUM_MOL_ATOMS, NUM_MOL);
        auto test_mol=std::get<0>(test_mol_bc);
        auto test_bc =std::get<1>(test_mol_bc);

        //! test make_ase_with_BCs
        Atoms new_atoms = make_ase_with_BCs(atoms_list[i].get_atomic_numbers(), NUM_MOL, raw_cpmd_get_unitcell_xyz("pg_gromacs.xyz"), test_mol, test_bc);
    
        //! test ase_io_write
        ase_io_write(new_atoms, "test_atoms"+std::to_string(i)+".xyz");

        // //! test raw_calc_bond_descripter_at_frame (chボンドのテスト)
        // auto descs_ch = raw_calc_bond_descripter_at_frame(atoms_list[i], test_bc, test_read_mol.ch_bond_index, NUM_MOL, UNITCELL_VECTORS,  NUM_MOL_ATOMS);
        // //! test for save as npy file.
        // // descs_chの形を1dへ変形してnpyで保存．
        // // TODO :: さすがにもっと効率の良い方法があるはず．
        // std::vector<double> descs_ch_1d;
        // for (int i = 0; i < descs_ch.size(); i++) {
        //     for (int j = 0; j < descs_ch[i].size(); j++) {
        //         descs_ch_1d.push_back(descs_ch[i][j]);
        //     }
        // }
        // //! npy.hppを利用して保存する．
        // const std::vector<long unsigned> shape_descs_ch{descs_ch.size(), descs_ch[0].size()}; // vectorを1*12の形に保存
        // npy::SaveArrayAsNumpy("descs_ch"+std::to_string(i)+".npy", false, shape_descs_ch.size(), shape_descs_ch.data(), descs_ch_1d);

        // //! test raw_calc_bond_descripter_at_frame (ccのボンドのテスト)
        // //!! 注意：：ccボンドの場合，最近説のC原子への距離が二つのC原子で同じなので，ここの並びが変わることがあり得る．
        // auto descs_cc = raw_calc_bond_descripter_at_frame(atoms_list[i], test_bc, test_read_mol.cc_bond_index, NUM_MOL, UNITCELL_VECTORS,  NUM_MOL_ATOMS);
        // // descs_chの形を1dへ変形してnpyで保存．
        // // TODO :: さすがにもっと効率の良い方法があるはず．
        // std::vector<double> descs_cc_1d;
        // for (int i = 0; i < descs_cc.size(); i++) {
        //     for (int j = 0; j < descs_cc[i].size(); j++) {
        //         descs_cc_1d.push_back(descs_cc[i][j]);
        //     }
        // }
        // //! npy.hppを利用して保存する．
        // const std::vector<long unsigned> shape_descs_cc{descs_cc.size(), descs_cc[0].size()}; // vectorを1*12の形に保存
        // npy::SaveArrayAsNumpy("descs_cc"+std::to_string(i)+".npy", false, shape_descs_cc.size(), shape_descs_cc.data(), descs_cc_1d);

        // //! test raw_calc_bond_descripter_at_frame (coのボンドのテスト)
        // auto descs_co = raw_calc_bond_descripter_at_frame(atoms_list[i], test_bc, test_read_mol.co_bond_index, NUM_MOL, UNITCELL_VECTORS,  NUM_MOL_ATOMS);
        // //! test for save as npy file.
        // // descs_chの形を1dへ変形してnpyで保存．
        // // TODO :: さすがにもっと効率の良い方法があるはず．
        // std::vector<double> descs_co_1d;
        // for (int i = 0; i < descs_co.size(); i++) {
        //     for (int j = 0; j < descs_co[i].size(); j++) {
        //         descs_co_1d.push_back(descs_co[i][j]);
        //     }
        // }
        // //! npy.hppを利用して保存する．
        // const std::vector<long unsigned> shape_descs_co{descs_co.size(), descs_co[0].size()}; // vectorを1*12の形に保存
        // npy::SaveArrayAsNumpy("descs_co"+std::to_string(i)+".npy", false, shape_descs_co.size(), shape_descs_co.data(), descs_co_1d);

        //! test raw_calc_bond_descripter_at_frame (ohのボンドのテスト)
        std::cout << " start descs_oh calculation ... " << std::endl;
        auto descs_oh = raw_calc_bond_descripter_at_frame(atoms_list[i], test_bc, test_read_mol.oh_bond_index, NUM_MOL, UNITCELL_VECTORS,  NUM_MOL_ATOMS);
        // descs_chの形を1dへ変形してnpyで保存．
        // TODO :: さすがにもっと効率の良い方法があるはず．
        std::vector<double> descs_oh_1d;
        for (int i = 0; i < descs_oh.size(); i++) {
            for (int j = 0; j < descs_oh[i].size(); j++) {
                descs_oh_1d.push_back(descs_oh[i][j]);
            }
        }
        const std::vector<long unsigned> shape_descs_oh{descs_oh.size(), descs_oh[0].size()}; // vectorを1*12の形に保存
        npy::SaveArrayAsNumpy("descs_oh"+std::to_string(i)+".npy", false, shape_descs_oh.size(), shape_descs_oh.data(), descs_oh_1d);

        // //! test raw_calc_lonepair_descripter_at_frame （ローンペアのテスト）
        // auto descs_o = raw_calc_lonepair_descripter_at_frame(atoms_list[i], test_mol, test_read_mol.o_list, NUM_MOL, 8, UNITCELL_VECTORS,  NUM_MOL_ATOMS);
        // //! test for save as npy file.
        // // descs_chの形を1dへ変形してnpyで保存．
        // // TODO :: さすがにもっと効率の良い方法があるはず．
        // std::vector<double> descs_o_1d;
        // for (int i = 0; i < descs_o.size(); i++) {
        //     for (int j = 0; j < descs_o[i].size(); j++) {
        //         descs_o_1d.push_back(descs_o[i][j]);
        //     }
        // }
        // //! npy.hppを利用して保存する．
        // const std::vector<long unsigned> shape_descs_o{descs_o.size(), descs_o[0].size()}; // vectorを1*12の形に保存
        // npy::SaveArrayAsNumpy("descs_o"+std::to_string(i)+".npy", false, shape_descs_o.size(), shape_descs_o.data(), descs_o_1d);

    }
}