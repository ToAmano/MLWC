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
#include <regex>   // using cmatch = std::match_results<const char*>;
#include <map>     // https://bi.biopapyrus.jp/cpp/syntax/map.html
#include <cmath>
#include <algorithm>
#include <numeric> // std::iota
#include <tuple>   // https://tyfkda.github.io/blog/2021/06/26/cpp-multi-value.html
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <Eigen/Core> // 行列演算など基本的な機能．
#include "numpy.hpp"
#include "npy.hpp"
// #include "numpy_quiita.hpp" // https://qiita.com/ka_na_ta_n/items/608c7df3128abbf39c89
// numpy_quiitaはsscanf_sが読み込めず，残念ながら現状使えない．
#include "atoms_core.cpp"
#include "atoms_io.cpp"
#include "mol_core.cpp"
#include "atoms_asign_wcs.cpp"

int main()
{
    std::vector<int> vec;

    vec.push_back(5);
    vec.push_back(10);
    vec.push_back(15);

    Vector vect(vec);
    vect.print();
    // 5 10 15

    //! test for sign
    std::cout << " test for sign !!" << std::endl;
    std::cout << sign(-16) << std::endl;
    std::cout << sign(16) << std::endl;

    std::vector<int> test_num{1, 2, 3};
    std::vector<Eigen::Vector3d> test_positions{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    std::vector<std::vector<double>> UNITCELL_VECTORS{{16.267601013183594, 0, 0}, {0, 16.267601013183594, 0}, {0, 0, 16.267601013183594}};
    std::vector<bool> pbc{1, 1, 1};

    //! test for constructing Atoms
    Atoms atoms(test_num, test_positions, UNITCELL_VECTORS, pbc);
    std::cout << "this is atomic num " << atoms.atomic_num[0] << endl;
    std::cout << atoms.get_atomic_numbers()[0] << endl;

    //! test raw_get_distances_mic
    std::vector<Eigen::Vector3d> answer = raw_get_distances_mic(atoms, 0, {1, 2}, true, true);
    for (int i = 0; i < answer.size(); i++)
    {
        for (int j = 0; j < answer[i].size(); j++)
        {
            std::cout << answer[i][j] << " ";
        }
        std::cout << endl;
    }

    //! test for get_atomic_num
    std::cout << "test for get_atomic_num " << std::endl;
    int atomic_num = raw_cpmd_num_atom("gromacs_30.xyz");
    std::cout << atomic_num << std::endl;

    //! test for raw_cpmd_get_unitcell_xyz
    std::vector<std::vector<double>> unitcell_vec = raw_cpmd_get_unitcell_xyz("pg_gromacs.xyz");
    for (int i = 0; i < unitcell_vec.size(); i++)
    {
        for (int j = 0; j < unitcell_vec[i].size(); j++)
        {
            std::cout << unitcell_vec[i][j] << " ";
        }
        std::cout << endl;
    }

    //! test for ase_io_read
    // 注意 :: gromacs_30.xyz はメタノール，pg_gromacs.xyがPG
    std::vector<Atoms> atoms_list = ase_io_read("pg_gromacs.xyz", atomic_num, unitcell_vec);
    std::cout << "this is atomic num " << atoms_list[0].get_atomic_numbers().size() << endl;
    // for (int j = 0; j < atoms_list[1].get_atomic_numbers().size(); j++) {
    //     std::cout << atoms_list[1].get_atomic_numbers()[j] << " " << atoms_list[1].get_positions()[j][0] << std::endl;
    // }

    //! test for read_mol
    read_mol test_read_mol;
    // print test_read_mol.bond_index['CH_1_bond']
    for (int i = 0; i < test_read_mol.bond_index['CH_1_bond'].size(); i++)
    {
        std::cout << test_read_mol.bond_index['CH_1_bond'][i] << " ";
    }
    // print test_read_mol.ch_bond
    for (int i = 0; i < test_read_mol.ch_bond.size(); i++)
    {
        std::cout << test_read_mol.ch_bond[i][0] << " " << test_read_mol.ch_bond[i][1] << std::endl;
    }

    //! test raw_aseatom_to_mol_coord_and_bc
    int NUM_MOL_ATOMS = test_read_mol.num_atoms_per_mol;
    int NUM_ATOM = atomic_num;
    int NUM_MOL = int(NUM_ATOM / NUM_MOL_ATOMS); // UnitCell中の総分子数
    auto test_mol_bc = raw_aseatom_to_mol_coord_and_bc(atoms_list[0], test_read_mol.bonds_list, test_read_mol, NUM_MOL_ATOMS, NUM_MOL);
    auto test_mol = std::get<0>(test_mol_bc);
    auto test_bc = std::get<1>(test_mol_bc);

    //! test make_ase_with_BCs
    Atoms new_atoms = make_ase_with_BCs(atoms_list[0].get_atomic_numbers(), NUM_MOL, raw_cpmd_get_unitcell_xyz("pg_gromacs.xyz"), test_mol, test_bc);

    //! test ase_io_write
    std::vector<Atoms> test = {new_atoms};
    ase_io_write(test, "test.xyz");

    //! test ase_io_write(作成したtest.xyzが読み込み可能かどうか)
    std::vector<Atoms> test_read = ase_io_read("test.xyz", raw_cpmd_num_atom("test.xyz"), raw_cpmd_get_unitcell_xyz("test.xyz"));

    //! test raw_calc_bond_descripter_at_frame (まずは適当なボンド記述子が作成できるかどうか)
    auto descs_ch = raw_calc_bond_descripter_at_frame(atoms_list[0], test_bc, test_read_mol.bond_index['CH_1_bond'], NUM_MOL, UNITCELL_VECTORS, NUM_MOL_ATOMS);

    //! test for save as npy file.
    // descs_chの形を1dへ変形してnpyで保存．
    // TODO :: さすがにもっと効率の良い方法があるはず．
    std::vector<double> descs_ch_1d;
    for (int i = 0; i < descs_ch.size(); i++)
    {
        for (int j = 0; j < descs_ch[i].size(); j++)
        {
            descs_ch_1d.push_back(descs_ch[i][j]);
        }
    }
    //! npy.hppを利用して保存する．
    const std::vector<long unsigned> shape_descs_ch{descs_ch.size(), descs_ch[0].size()}; // vectorを1*12の形に保存
    npy::SaveArrayAsNumpy("descs_ch.npy", false, shape_descs_ch.size(), shape_descs_ch.data(), descs_ch_1d);

    // vectorをnpyへ保存（1次元の例） http://mglab.blogspot.com/2014/03/numpyhpp-npy.html
    const bool fortran_order{false}; // fortranの場合は配列の順番が逆になる．
    const std::vector<double> data1{1, 2, 3, 4, 5, 6};
    aoba::SaveArrayAsNumpy("1dvector.npy", 6, &data1[0]);

    // vectorをnpyへ保存（2次元の例） http://mglab.blogspot.com/2014/03/numpyhpp-npy.html
    const std::vector<std::vector<double>> data2{{1, 11}, {2, 12}, {3, 13}, {4, 14}, {5, 15}, {6, 16}};
    aoba::SaveArrayAsNumpy("2dvector.npy", 6, 2, &data2[0][0]);

    const std::vector<long unsigned> shape{1, 6}; // vectorを2*3の形に保存
    std::cout << shape.size() << " " << shape.data() << std::endl;
    // const std::vector<double> data3{1, 2, 3, 4, 5, 6};
    npy::SaveArrayAsNumpy("1dvector_v2.npy", fortran_order, shape.size(), shape.data(), data1);

    // !! このようにもともと2次元のデータを保存することはできない．
    // const std::vector<long unsigned> shape2{1, 12}; // vectorを1*12の形に保存
    // npy::SaveArrayAsNumpy("2dvector_v2.npy", fortran_order, shape2.size(), shape2.data(), data2);

    // !! numpy_quiitaを使う場合．ただしこれは現状使えてない．
    // SaveNpy("1dvector_v3.npy", data1);

    // したがって，最初に2次元データを1次元化する．（この時の順番が大事）
    // !! 以下のように自然にデータを1d化すればちゃんと2次元データとして保存できることがわかった．
    std::vector<double> data2_1d;
    for (int i = 0; i < data2.size(); i++)
    {
        for (int j = 0; j < data2[i].size(); j++)
        {
            data2_1d.push_back(data2[i][j]);
        }
    }
    const std::vector<long unsigned> shape3{6, 2}; // vectorを1*12の形に保存
    npy::SaveArrayAsNumpy("2dvector_v2.npy", fortran_order, shape3.size(), shape3.data(), data2_1d);
}