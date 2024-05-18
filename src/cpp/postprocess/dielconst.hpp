#pragma once

// #define _DEBUG
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <iomanip>
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
#include <deque> // deque
// #include <boost/numeric/ublas/vector.hpp>
// #include <boost/numeric/ublas/matrix.hpp>
// #include <boost/numeric/ublas/io.hpp>
#include <Eigen/Core> // 行列演算など基本的な機能．

// https://qiita.com/meshidenn/items/53b7c6f35c6259320241
#include <numeric> // accumulate関数に必要


double calc_moldipole_mean(const std::vector<std::vector<Eigen::Vector3d> >& result_molecule_dipole_list);

double calc_moldipole_stderr(const std::vector<std::vector<Eigen::Vector3d> >& result_molecule_dipole_list, double mean_absolute_mol_dipole);

double calc_M2(const std::vector<Eigen::Vector3d>& result_dipole_list);

double calc_M(const std::vector<Eigen::Vector3d>& result_dipole_list);

double calc_dielconst(double temperature,std::vector<std::vector<double> > UNITCELL_VECTORS, double mean_M2,double mean_M);

void postprocess_dielconst(const std::vector<Eigen::Vector3d>& result_dipole_list, const std::vector<std::vector<Eigen::Vector3d> >& result_molecule_dipole_list, double temperature, std::vector<std::vector<double> > UNITCELL_VECTORS, std::string savedir);
