/**
 * @file module_xyz.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-02-03
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
#include "atoms_core.hpp"

namespace module_xyz{
class load_xyz {
    /**
     * @brief module load xyz for main.cpp
     * 
     */
public:
    int ALL_NUM_ATOM;
    int NUM_ATOM;
    std::vector<std::vector<double> > UNITCELL_VECTORS;
    std::vector<Atoms> atoms_list; 
    int NUM_CONFIG; // totalのconfiguration数
    

    // コンストラクタ
    // , std::unique_ptr<diagnostics::Stopwatch> timer
    load_xyz(std::string xyzfilename, std::unique_ptr<diagnostics::Stopwatch> &timer); //descriptorのサイズ, 分子数で初期化する
private:
    std::string _xyzfilename; // xyz filename (absolute path)
    // メンバ関数
    int _get_ALL_NUM_ATOM();
    int _get_NUM_ATOM();
    int _get_UNITCELL_VECTOR();
    int _get_atoms_list();
    // void predict_bond_dipole_at_frame(const Atoms &atoms, const std::vector<std::vector< Eigen::Vector3d> > &test_bc, const std::vector<int> bond_index, int NUM_MOL, std::vector<std::vector<double> > UNITCELL_VECTORS, int NUM_MOL_ATOMS, std::string desctype, torch::jit::script::Module model_dipole ); // 予測してMoleculeDipoleList, dipole_list,wannier_listに値を代入する．
};
}