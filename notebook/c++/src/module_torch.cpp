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

#include <Eigen/Core> // 行列演算など基本的な機能．
#include <Eigen/Dense> // vector3dにはこれが必要？
#include <vector>
// #include "include/"
#include "include/manupilate_files.hpp"
#include "include/stopwatch.hpp"
#include "include/error.h"
#include "module_torch.hpp"
#include "torch/script.h" // pytorch

namespace module_torch{
load_models::load_models(std::string model_dir, std::unique_ptr<diagnostics::Stopwatch> &timer){
    // get model directory
    this->_model_dir = model_dir;
    std::cout << "" << std::endl;
    std::cout << " ************************** SYSTEM INFO :: reading ML models *************************** " << std::endl;
    //
    timer->reset(); // reset timer
    timer->start(); // restart timer
    load_models::_get_models(); // load ALL_NUM_ATOM (include wannier)
    timer->stop(); // stop timer
    std::cout << "     ELAPSED TIME (sec)      = " << timer->getElapsedSeconds() << std::endl;
    std::cout << std::endl;
};

int load_models::_get_models(){
    // 変換した学習済みモデルの読み込み
    // 実行パス（not 実行ファイルパス）からの絶対パスに変換 https://nompor.com/2019/02/16/post-5089/
    if (manupilate_files::IsFileExist(std::filesystem::absolute(this->_model_dir+"/model_ch.pt"))) {
        this->IF_CALC_CH = true;
        this->module_ch = torch::jit::load(std::filesystem::absolute(this->_model_dir+"/model_ch.pt"));
        // module_ch = torch::jit::load(this->.model_dir+"/Users/amano/works/research/dieltools/notebook/c++/202306014_model_rotate/model_ch.pt");
    }
    if (manupilate_files::IsFileExist(std::filesystem::absolute(this->_model_dir+"/model_cc.pt"))) {
        this->IF_CALC_CC = true;
        this->module_cc = torch::jit::load(std::filesystem::absolute(this->_model_dir+"/model_cc.pt"));
    }
    if (manupilate_files::IsFileExist(std::filesystem::absolute(this->_model_dir+"/model_co.pt"))) {
        this->IF_CALC_CO = true;
        this->module_co = torch::jit::load(std::filesystem::absolute(this->_model_dir+"/model_co.pt"));
    }
    if (manupilate_files::IsFileExist(std::filesystem::absolute(this->_model_dir+"/model_oh.pt"))) {
        this->IF_CALC_OH = true;
        this->module_oh = torch::jit::load(std::filesystem::absolute(this->_model_dir+"/model_oh.pt"));
    }
    if (manupilate_files::IsFileExist(std::filesystem::absolute(this->_model_dir+"/model_o.pt"))) {
        this->IF_CALC_O = true;
        this->module_o = torch::jit::load(std::filesystem::absolute(this->_model_dir+"/model_o.pt"));
    }

    if (manupilate_files::IsFileExist(std::filesystem::absolute(this->_model_dir+"/model_coc.pt"))) {
        this->IF_CALC_COC = true;
        this->module_coc = torch::jit::load(std::filesystem::absolute(this->_model_dir+"/model_coc.pt"));
    }
    if (manupilate_files::IsFileExist(std::filesystem::absolute(this->_model_dir+"/model_coh.pt"))) {
        this->IF_CALC_COH = true;
        this->module_coh = torch::jit::load(std::filesystem::absolute(this->_model_dir+"/model_coh.pt"));
    };
    std::cout << std::setw(30) << " IF_CALC_CH :: "  << this->IF_CALC_CH << std::endl;
    std::cout << std::setw(30) << " IF_CALC_CC :: "  << this->IF_CALC_CC << std::endl;
    std::cout << std::setw(30) << " IF_CALC_CO :: "  << this->IF_CALC_CO << std::endl;
    std::cout << std::setw(30) << " IF_CALC_OH :: "  << this->IF_CALC_OH << std::endl;
    std::cout << std::setw(30) << " IF_CALC_O :: "   << this->IF_CALC_O << std::endl;
    std::cout << std::setw(30) << " IF_CALC_COC :: " << this->IF_CALC_COC << std::endl;
    std::cout << std::setw(30) << " IF_CALC_COH :: " << this->IF_CALC_COH << std::endl;
    std::cout << std::setw(30) << " finish reading ML model file" << std::endl;

    // 全てのIF_CALC_*がfalseのままだったら計算を中止する
    int CHECK_CALC = IF_CALC_CH + IF_CALC_CC + IF_CALC_CO + IF_CALC_OH + IF_CALC_O + IF_CALC_COC + IF_CALC_COH ;
    if ( CHECK_CALC == 0 ){
        error::exit("module_torch", "ALL IF_CALC is false. Please check modeldir is correct.");
    };
    return 0;
};

} // END namespace


