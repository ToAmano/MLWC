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

    //! test for sign
    std::cout << " Test for sign !!" << std::endl;
    std::cout << sign(-16) << " should be -1. "<< std::endl;
    std::cout << sign(16)  << " should be 1.  "<< std::endl;


    //! test for constructing Atoms
    std::cout << " Test for constructing Atoms " << std::endl;
    std::vector<int> test_num{1,2,3}; //原子番号
    std::vector<Eigen::Vector3d > test_positions{{1,2,3},{4,5,6},{7,8,9}}; //座標
    std::vector<std::vector<double> > UNITCELL_VECTORS{{16.267601013183594,0,0},{0,16.267601013183594,0},{0,0,16.267601013183594}}; //格子定数
    std::vector<bool> pbc{1,1,1}; 
    Atoms atoms(test_num,test_positions, UNITCELL_VECTORS,pbc);
    std::cout << "this is atomic num :: " << atoms.get_atomic_numbers()[0] << std::endl;
    std::cout << std::endl;

    //! test raw_get_distances_mic
    std::cout << " Test raw_get_distances_mic " << std::endl;
    std::vector<Eigen::Vector3d> test_distances = raw_get_distances_mic(atoms,0,{1,2},true,true);
    for (int i = 0; i < test_distances.size(); i++) {
        for (int j = 0; j < test_distances[i].size(); j++) {
            std::cout << test_distances[i][j] << " ";
        }
        std::cout << std::endl;
    }

    //! test for get_atomic_num
    std::cout << "test for get_atomic_num " << std::endl;
    int atomic_num = raw_cpmd_num_atom("gromacs_30.xyz");
    std::cout << "atomic_num :: " << atomic_num << std::endl;

    //! test for raw_cpmd_get_unitcell_xyz
    std::cout << "test for raw_cpmd_get_unitcell_xyz " << std::endl;
    std::vector<std::vector<double> > unitcell_vec = raw_cpmd_get_unitcell_xyz("pg_gromacs.xyz");
    for (int i = 0; i < unitcell_vec.size(); i++) {
        for (int j = 0; j < unitcell_vec[i].size(); j++) {
            std::cout << unitcell_vec[i][j] << " ";
        }
        std::cout << std::endl;
    }

    //! test for ase_io_read
    // 注意 :: gromacs_30.xyz はメタノール，pg_gromacs.xyがPG
    // 各変数にraw関数を入れることで追加の変数定義なしで使える．　
    std::cout << std::endl;
    std::cout << "test for ase_io_read... " << std::endl;
    std::vector<Atoms> atoms_list = ase_io_read("pg_gromacs.xyz", raw_cpmd_num_atom("pg_gromacs.xyz"), raw_cpmd_get_unitcell_xyz("pg_gromacs.xyz"));
    std::cout << "atomic num :: " << atoms_list[0].get_atomic_numbers().size() << " should be 390. " << std::endl;
    // for (int j = 0; j < atoms_list[1].get_atomic_numbers().size(); j++) {
    //     std::cout << atoms_list[1].get_atomic_numbers()[j] << " " << atoms_list[1].get_positions()[j][0] << std::endl;
    // }

    //! test for read_mol (すでにbond_listが与えられた状態からうまくいくか．)
    std::cout << std::endl;
    std::cout << "test for read_mol " << std::endl;
    read_mol test_read_mol;
    std::cout << "print ch_bond_index " << std::endl; 
    for (int i = 0; i < test_read_mol.ch_bond_index.size(); i++) {
        std::cout << test_read_mol.ch_bond_index[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "print ch_bond " << std::endl; 
    for (int i = 0; i < test_read_mol.ch_bond.size(); i++) {
        std::cout << test_read_mol.ch_bond[i][0] << " " << test_read_mol.ch_bond[i][1] << std::endl;
    }

    //! test raw_aseatom_to_mol_coord_and_bc
    std::cout << std::endl;
    std::cout << " test raw_aseatom_to_mol_coord_and_bc " << std::endl;
    int NUM_MOL_ATOMS = test_read_mol.num_atoms_per_mol;
    int NUM_ATOM = atomic_num;
    int NUM_MOL = int(NUM_ATOM/NUM_MOL_ATOMS); // UnitCell中の総分子数
    auto test_mol_bc = raw_aseatom_to_mol_coord_and_bc(atoms_list[0], test_read_mol.bonds_list, test_read_mol, NUM_MOL_ATOMS, NUM_MOL);
    auto test_mol=std::get<0>(test_mol_bc);
    auto test_bc =std::get<1>(test_mol_bc);

    //! test make_ase_with_BCs
    Atoms new_atoms = make_ase_with_BCs(atoms_list[0].get_atomic_numbers(), NUM_MOL, raw_cpmd_get_unitcell_xyz("pg_gromacs.xyz"), test_mol, test_bc);
    
    //! test ase_io_write
    std::vector<Atoms> test = {new_atoms};
    ase_io_write(test, "test.xyz");

    //! test ase_io_write(作成したtest.xyzが読み込み可能かどうか)
    std::vector<Atoms> test_read = ase_io_read("test.xyz", raw_cpmd_num_atom("test.xyz"), raw_cpmd_get_unitcell_xyz("test.xyz"));

    //! test raw_calc_bond_descripter_at_frame (まずは適当なボンド記述子が作成できるかどうか)
    auto descs_ch = raw_calc_bond_descripter_at_frame(atoms_list[0], test_bc, test_read_mol.ch_bond_index, NUM_MOL, UNITCELL_VECTORS,  NUM_MOL_ATOMS);

    //! test for save as npy file.
    // descs_chの形を1dへ変形してnpyで保存．
    // TODO :: さすがにもっと効率の良い方法があるはず．
    std::vector<double> descs_ch_1d;
    for (int i = 0; i < descs_ch.size(); i++) {
        for (int j = 0; j < descs_ch[i].size(); j++) {
            descs_ch_1d.push_back(descs_ch[i][j]);
        }
    }
    //! npy.hppを利用して保存する．
    const std::vector<long unsigned> shape_descs_ch{descs_ch.size(), descs_ch[0].size()}; // vectorを1*12の形に保存
    npy::SaveArrayAsNumpy("descs_ch.npy", false, shape_descs_ch.size(), shape_descs_ch.data(), descs_ch_1d);
    
    //! 一連の流れを再現する．

}