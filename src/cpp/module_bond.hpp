/**
 * @file module_bond.hpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-02-05
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#pragma once

#include <Eigen/Core> // 行列演算など基本的な機能．
#include <Eigen/Dense> // vector3dにはこれが必要？
#include <iomanip>
#include <iostream>
#include <vector>
#include "include/stopwatch.hpp"
#include "chemicalbond/mol_core.hpp"


namespace module_bond{
class load_bond {
    /**
     * @brief module load bond for main.cpp
     * 
     */
public:
    int NUM_MOL_ATOMS;
    read_mol bondinfo;

    // コンストラクタ
    // , std::unique_ptr<diagnostics::Stopwatch> timer
    load_bond(std::string bondfilename, std::unique_ptr<diagnostics::Stopwatch> &timer); //descriptorのサイズ, 分子数で初期化する
private:
    std::string _bondfilename; // bond filename (absolute path)
    // メンバ関数
    int _get_NUM_MOL_ATOMS();
    int _get_bondinfo();
    // void predict_bond_dipole_at_frame(const Atoms &atoms, const std::vector<std::vector< Eigen::Vector3d> > &test_bc, const std::vector<int> bond_index, int NUM_MOL, std::vector<std::vector<double> > UNITCELL_VECTORS, int NUM_MOL_ATOMS, std::string desctype, torch::jit::script::Module model_dipole ); // 予測してMoleculeDipoleList, dipole_list,wannier_listに値を代入する．
};
}