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
#include "descriptor.hpp" // TODO hpp化

/*
予測部分の関数
*/

int save_descriptor(const std::vector<std::vector<double> > &descs, const std::string name, const int i) ;

Eigen::Vector3d predict_dipole(const std::vector<double> &descs, torch::jit::script::Module model_dipole) ;

std::tuple< std::vector< Eigen::Vector3d >, std::vector< Eigen::Vector3d > > predict_dipole_at_frame(int i, const Atoms &atoms, const std::vector<std::vector< Eigen::Vector3d> > &test_bc, const std::vector<int> bond_index, int NUM_MOL, std::vector<std::vector<double> > UNITCELL_VECTORS, int NUM_MOL_ATOMS, std::string desctype, bool SAVE_DESCS, torch::jit::script::Module model_dipole, Eigen::Vector3d &TotalDipole, std::vector< Eigen::Vector3d > &MoleculeDipoleList) ;


#endif //! INCLUDE_HPP_predict