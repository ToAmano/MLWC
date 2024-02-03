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


#pragma once

#include "parse.cpp"


namespace module_input{
class load_input {
    /**
     * @brief module load xyz for main.cpp
     * 
     */
public:
    var_general var_gen;
    var_predict var_pre;
    var_descripter var_des;
    // コンストラクタ
    // , std::unique_ptr<diagnostics::Stopwatch> timer
    load_input(std::string inputfilename, std::unique_ptr<diagnostics::Stopwatch> &timer); //descriptorのサイズ, 分子数で初期化する
private:
    std::string _inputfilename; // input filename (absolute path)
    // メンバ関数
    int _get_input_text();
    int _check_savedir();
};
}