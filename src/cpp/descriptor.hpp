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
#include "include/printvec.hpp"

#define DEBUG_PRINT_VARIABLE(var) std::cout << #var << std::endl;

/*
*/

Atoms make_ase_with_BCs(const std::vector<int> &ase_atomicnumber,const int NUM_MOL,const std::vector<std::vector<double> > UNITCELL_VECTORS, const std::vector<std::vector<Eigen::Vector3d> > &list_mol_coords, const std::vector<std::vector<Eigen::Vector3d> > &list_bond_centers);

Atoms raw_make_atoms_with_bc(Eigen::Vector3d bond_center, const Atoms &aseatoms, std::vector<std::vector<double> > UNITCELL_VECTORS);

std::vector<Eigen::Vector3d> get_coord_of_specific_bondcenter(const std::vector<std::vector<Eigen::Vector3d> > &list_bond_centers, std::vector<int> bond_index);

std::vector<Eigen::Vector3d> get_coord_of_specific_lonepair(const std::vector<std::vector<Eigen::Vector3d> > &list_atom_positions, std::vector<int> atom_index);

double fs(double Rij,double Rcs,double Rc);

std::vector<double> calc_descripter(const std::vector<Eigen::Vector3d> &dist_wVec, const std::vector<int> &atoms_index,double Rcs,double Rc,int MaxAt);

std::vector<double> raw_get_desc_bondcent(const Atoms &atoms, Eigen::Vector3d bond_center, int mol_id, std::vector<std::vector<double> > UNITCELL_VECTORS, int NUM_MOL_ATOMS, float Rcs=4.0, float Rc=6.0, int MaxAt=12);

std::vector<double> raw_get_desc_bondcent_allinone(const Atoms &atoms, Eigen::Vector3d bond_center, std::vector<std::vector<double> > UNITCELL_VECTORS, int NUM_MOL_ATOMS, float Rcs=4.0, float Rc=6.0, int MaxAt=24);

std::vector<std::vector<double> > raw_calc_bond_descripter_at_frame(const Atoms &atoms_fr, const std::vector<std::vector< Eigen::Vector3d> > &list_bond_centers, std::vector<int> bond_index, int NUM_MOL, std::vector<std::vector<double> > UNITCELL_VECTORS, int NUM_MOL_ATOMS, std::string desctype,float Rcs=4.0, float Rc=6.0, int MaxAt=24);

std::vector<std::vector<int>> raw_find_atomic_index(const Atoms &aseatoms, int atomic_number, int NUM_MOL) ;

std::vector<Eigen::Vector3d> find_specific_lonepair(const std::vector<std::vector<Eigen::Vector3d> > &list_mol_coords, const Atoms &aseatoms, int atomic_number, int NUM_MOL);

std::vector<Eigen::Vector3d> find_specific_lonepair_select(const std::vector<std::vector<Eigen::Vector3d> > &list_mol_coords, std::vector<int> at_list, int NUM_MOL);

std::vector<double> raw_get_desc_lonepair(const Atoms &atoms, Eigen::Vector3d lonepair_coord, int mol_id, std::vector<std::vector<double> > UNITCELL_VECTORS, int NUM_MOL_ATOMS,float Rcs=4.0, float Rc=6.0, int MaxAt=24);

std::vector<double> raw_get_desc_lonepair_allinone(const Atoms &atoms, Eigen::Vector3d lonepair_coord, std::vector<std::vector<double> > UNITCELL_VECTORS, int NUM_MOL_ATOMS,float Rcs=4.0, float Rc=6.0, int MaxAt=24);

std::vector<std::vector<double> > raw_calc_lonepair_descripter_at_frame(const Atoms &atoms_fr, const std::vector<std::vector<Eigen::Vector3d> > &list_mol_coords, std::vector<int> at_list, int NUM_MOL, int atomic_number, std::vector<std::vector<double>> UNITCELL_VECTORS, int NUM_MOL_ATOMS, std::string desctype,float Rcs=4.0, float Rc=6.0, int MaxAt=24);

std::vector<std::vector<double> > raw_calc_lonepair_descripter_select_at_frame(const Atoms &atoms_fr, const std::vector<std::vector<Eigen::Vector3d> > &list_mol_coords, std::vector<int> at_list, int NUM_MOL, std::vector<std::vector<double>> UNITCELL_VECTORS, int NUM_MOL_ATOMS, std::string desctype) ;