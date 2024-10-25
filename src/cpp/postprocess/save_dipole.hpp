#pragma once

// https://github.com/microsoft/vscode-cpptools/issues/7413
#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

// #define _DEBUG
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstdio>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream> // https://www.cns.s.u-tokyo.ac.jp/~masuoka/post/inputfile_cpp/
#include <regex> // using cmatch = std::match_results<const char*>;
#include <numeric> // std::iota
#include <Eigen/Core> // 行列演算など基本的な機能．


std::string get_preface_line(std::vector<std::vector<double> > unitcell, double temperature, double timestep);

int save_totaldipole(const std::vector<Eigen::Vector3d>& result_dipole_list, std::vector<std::vector<double> > unitcell, double temperature, double timestep, std::string savedir);
int save_bonddipole(const std::vector<std::vector<Eigen::Vector3d> >& result_bond_dipole_list,std::string savedir,std::string filename);
int save_moleculedipole(const std::vector<std::vector<Eigen::Vector3d> >& result_mol_dipole_list,const std::string savedir);
int postprocess_save_bonddipole(
    const std::vector<std::vector<Eigen::Vector3d> >& result_ch_dipole_list,
    const std::vector<std::vector<Eigen::Vector3d> >& result_co_dipole_list,
    const std::vector<std::vector<Eigen::Vector3d> >& result_oh_dipole_list,
    const std::vector<std::vector<Eigen::Vector3d> >& result_cc_dipole_list,
    const std::vector<std::vector<Eigen::Vector3d> >& result_o_dipole_list,
    const std::vector<std::vector<Eigen::Vector3d> >& result_coc_dipole_list,
    const std::vector<std::vector<Eigen::Vector3d> >& result_coh_dipole_list,
    std::vector<std::vector<double> > unitcell, double temperature, double timestep,
    std::string savedir);
int postprocess_save_moleculedipole(
    const std::vector<std::vector<Eigen::Vector3d> >& result_molecule_dipole_list,
    std::vector<std::vector<double> > unitcell, double temperature, double timestep,
    std::string savedir);