#pragma once

// #define _DEBUG
#include <stdio.h>
#include <iomanip>
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
#include <Eigen/Core> // 行列演算など基本的な機能．
#include "numpy.hpp"
#include "npy.hpp"
// #include "numpy_quiita.hpp" // https://qiita.com/ka_na_ta_n/items/608c7df3128abbf39c89
// numpy_quiitaはsscanf_sが読み込めず，残念ながら現状使えない．
#include "atoms_core.hpp"


/*
Definition for ase_io_read & ase_io_write
*/


int raw_cpmd_num_atom(const std::string filename);

int get_num_atom_without_wannier(const std::string filename);

std::vector<std::vector<double> > raw_cpmd_get_unitcell_xyz(const std::string filename ) ;

// read lammps structure
std::vector<Atoms> ase_io_read_lammps(const std::string filename);

std::vector<Atoms> ase_io_read(const std::string filename, const int NUM_ATOM, const std::vector<std::vector<double> > unitcell_vec);

std::vector<Atoms> ase_io_read(std::string filename); // wrapper

std::vector<Atoms> ase_io_read(const std::string filename, const int NUM_ATOM, const std::vector<std::vector<double> > unitcell_vec, bool IF_REMOVE_WANNIER);

std::vector<Atoms> ase_io_read(const std::string filename,  bool IF_REMOVE_WANNIER);

int ase_io_write(const std::vector<Atoms> &atoms_list, const std::string filename );

int ase_io_write(const Atoms &aseatoms, std::string filename );

std::vector<Atoms> ase_io_convert_1mol(const std::vector<Atoms> aseatoms, const int NUM_ATOM_PER_MOL);