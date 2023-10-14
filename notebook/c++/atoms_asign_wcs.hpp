/**
 * @file atoms_asign_wcs.cpp
 * @brief assign WCs to each bond/lonepair 
 * @author Tomohito Amano
 * @date 2023/10/15
 */

#pragma once

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
#include "atoms_io.hpp"

#define DEBUG_PRINT_VARIABLE(var) std::cout << #var << std::endl;

/*
*/

std::vector<Eigen::Vector3d> raw_calc_mol_coord_mic_onemolecule(std::vector<int> mol_inds, std::vector<std::vector<int>> bonds_list_j, const Atoms &aseatoms, const read_mol &itp_data);

std::vector<Eigen::Vector3d> raw_calc_bc_mic_onemolecule(const std::vector<int> mol_inds, const std::vector<std::vector<int>> bonds_list_j, const std::vector<Eigen::Vector3d> &mol_coords);

std::tuple<std::vector<std::vector<Eigen::Vector3d> >, std::vector<std::vector<Eigen::Vector3d> > > raw_aseatom_to_mol_coord_and_bc(const Atoms &ase_atoms, const std::vector<std::vector<int>> bonds_list, const read_mol &itp_data, const int NUM_MOL_ATOMS, const int NUM_MOL);

std::vector<Eigen::Vector3d> find_specific_bondcenter(const std::vector<std::vector<Eigen::Vector3d> > &list_bond_centers, const std::vector<int> bond_index);
