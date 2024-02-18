/**
 * @file convert_gas.cpp
 * @brief gas計算で得られたデータを全て元の形式に変換する．おそらくちょっと遅い関数になってしまうので，将来的により早い方法を考えた方が良い．
 * @author Tomohito Amano
 * @date 2024/1/4
 */

// https://github.com/microsoft/vscode-cpptools/issues/7413
#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

// #define _DEBUG
#include <stdio.h>
#include <filesystem>
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
#include <Eigen/Core> // 行列演算など基本的な機能．
#include <Eigen/Dense> // vector3dにはこれが必要？

std::vector<std::vector<Eigen::Vector3d> > convert_bond_dipole(const std::vector<std::vector<Eigen::Vector3d> >& gas_dipole_list, const int NUM_CONFIG, const int NUM_MOL);

std::vector<Eigen::Vector3d> convert_total_dipole(const std::vector<Eigen::Vector3d>& gas_dipole_list, const int NUM_FRAME, const int NUM_MOL);
