/**
 * @file ファイル名.h
 * @brief 簡単な説明
 * @author 書いた人
 * @date 日付（開始日？）
 */

#ifndef INCLUDE_HPP_predict
#define INCLUDE_HPP_predict

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
// #include <boost/numeric/ublas/vector.hpp>
// #include <boost/numeric/ublas/matrix.hpp>
// #include <boost/numeric/ublas/io.hpp>
// #include <rdkit/GraphMol/GraphMol.h>
// #include <rdkit/GraphMol/FileParsers/MolSupplier.h>
#include <Eigen/Core> // 行列演算など基本的な機能．
#include "torch/script.h" // pytorch
#include "atoms_io.hpp"
#include "include/constant.hpp"

#include "numpy.hpp"
#include "npy.hpp"
#include "descriptor.hpp" 

/*
予測部分の関数
*/

int save_descriptor(const std::vector<std::vector<double> > &descs, const std::string name, const int i) ;

Eigen::Vector3d predict_dipole(const std::vector<double> &descs, torch::jit::script::Module model_dipole) ;

std::tuple< std::vector< Eigen::Vector3d >, std::vector< Eigen::Vector3d > > predict_dipole_at_frame(int i, const Atoms &atoms, const std::vector<std::vector< Eigen::Vector3d> > &test_bc, const std::vector<int> bond_index, int NUM_MOL, std::vector<std::vector<double> > UNITCELL_VECTORS, int NUM_MOL_ATOMS, std::string desctype, bool SAVE_DESCS, torch::jit::script::Module model_dipole, Eigen::Vector3d &TotalDipole, std::vector< Eigen::Vector3d > &MoleculeDipoleList) ;


class dipole_frame {
public:
    std::vector< Eigen::Vector3d > MoleculeDipoleList; //! 分子ごとのボンド双極子リスト（最終的に分子双極子を得るには加算が必要）
    std::vector<Eigen::Vector3d> dipole_list; // bond dipoleの格納
    std::vector<std::vector<Eigen::Vector3d> > wannier_list; // wannier coordinateの格納（[NUM_MOL,NUM_WAN]の形）
    bool calc_wannier; // wannier coordinateを計算したかどうかのフラグ
    int descs_size;    // descsriptorのサイズ
    int num_molecule;  // 分子数
    int num_bond;      // ボンドの数: descriptor/分子数
    // コンストラクタ
    dipole_frame(int descs_size, int num_molecule); //descriptorのサイズ, 分子数で初期化する

    // メンバ関数
    void predict_bond_dipole_at_frame(const Atoms &atoms, const std::vector<std::vector< Eigen::Vector3d> > &test_bc, const std::vector<int> bond_index, int NUM_MOL, std::vector<std::vector<double> > UNITCELL_VECTORS, int NUM_MOL_ATOMS, std::string desctype, torch::jit::script::Module model_dipole ); // 予測してMoleculeDipoleList, dipole_list,wannier_listに値を代入する．
    void predict_lonepair_dipole_at_frame(const Atoms &atoms, const std::vector<std::vector< Eigen::Vector3d> > &test_bc, const std::vector<int> atom_index, int NUM_MOL, std::vector<std::vector<double> > UNITCELL_VECTORS, int NUM_MOL_ATOMS, std::string desctype, torch::jit::script::Module model_dipole ); 
    void predict_lonepair_dipole_select_at_frame(const Atoms &atoms, const std::vector<std::vector< Eigen::Vector3d> > &test_mol, const std::vector<int> atom_index, int NUM_MOL, std::vector<std::vector<double> > UNITCELL_VECTORS, int NUM_MOL_ATOMS, std::string desctype, torch::jit::script::Module model_dipole );
    void calculate_wannier_list(std::vector<std::vector< Eigen::Vector3d> > &test_bc, const std::vector<int> bond_index); // predict_dipole_at_frameの後にワニエの座標を計算する
    void calculate_lonepair_wannier_list(std::vector<std::vector< Eigen::Vector3d> > &test_mol, const std::vector<int> atom_index); // ローンペアの場合にワニエの座標を計算する
    void calculate_moldipole_list(); // predict_dipole_at_frameの後に分子双極子を計算する
    void save_descriptor_frame(int i, const Atoms &atoms, const std::vector<std::vector< Eigen::Vector3d> > &test_bc, const std::vector<int> bond_index, int NUM_MOL, std::vector<std::vector<double> > UNITCELL_VECTORS, int NUM_MOL_ATOMS, std::string desctype, bool SAVE_DESCS, torch::jit::script::Module model_dipole); // descriptorを保存する（frameごとに保存するとかなり大変なのでやめた方が良い．
    // 以下はポストプロセスに使う関数で，できれば分けた方が良い
    void calculate_coh_bond_dipole_at_frame(std::map<int, std::pair<int, int> > coh_bond_info, const std::vector< Eigen::Vector3d > o_dipole_list, const std::vector< Eigen::Vector3d > bond1_dipole_list, const std::vector< Eigen::Vector3d > bond2_dipole_list);
};


#endif //! INCLUDE_HPP_predict