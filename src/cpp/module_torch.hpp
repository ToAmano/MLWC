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
#include <vector>
#include "torch/script.h" // pytorch

namespace module_torch{
class load_models {
    /**
     * @brief module load xyz for main.cpp
     * 
     */
private:
    std::string _model_dir; // xyz filename (absolute path)
public:
    // torch::jit::script::Module 型で module 変数の定義
    torch::jit::script::Module module_ch, module_cc, module_co, module_oh, module_o,module_coc,module_coh;
    // If true, calculate the associated bond
    bool IF_CALC_CH = false;
    bool IF_CALC_CC = false;
    bool IF_CALC_CO = false;
    bool IF_CALC_OH = false;
    bool IF_CALC_O = false;
    bool IF_CALC_COC = false;
    bool IF_CALC_COH = false;

    // constructor
    // , std::unique_ptr<diagnostics::Stopwatch> timer
    load_models(std::string model_dir, std::unique_ptr<diagnostics::Stopwatch> &timer); //descriptorのサイズ, 分子数で初期化する
private:
    // メンバ関数
    torch::jit::script::Module _load_model(std::string filename,bool& IF_CALC);
    int _get_models();
};
}