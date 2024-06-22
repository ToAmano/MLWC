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
    std::cout << std::setw(30) << "  model directory :: " << this->_model_dir << std::endl;
    //
    timer->reset(); // reset timer
    timer->start(); // restart timer
    load_models::_get_models(); // load ALL_NUM_ATOM (include wannier)
    timer->stop(); // stop timer
    std::cout << "     ELAPSED TIME (sec)      = " << timer->getElapsedSeconds() << std::endl;
    std::cout << std::endl;
};


torch::jit::script::Module load_models::_load_model(std::string filename, bool& IF_CALC ){
    torch::jit::script::Module model;
    if (manupilate_files::IsFileExist(std::filesystem::absolute(filename))) {
        IF_CALC = true;
        model = torch::jit::load(std::filesystem::absolute(filename));
        std::cout << " Loaded " << filename << ": parameters are ..." << std::endl;
    	try
    	{
            std::cout << "  nfeatures = " << model.attr("nfeatures").toInt() << std::endl;
            std::cout << "  Rcs       = " << model.attr("Rcs").toDouble() << std::endl;
            std::cout << "  Rc        = " << model.attr("Rc").toDouble() << std::endl;
    	}
    	// catch (std::exception& ex)
        catch (...) // If member variables are not in the original file, add default values to them here.
    	{
    		std::cout << "Exception occurred!" << std::endl;
            model.register_attribute("nfeatures", c10::IntType::get(), torch::jit::IValue(288));
            model.register_attribute("Rcs", c10::FloatType::get(), torch::jit::IValue(4.0));
            model.register_attribute("Rc", c10::FloatType::get(), torch::jit::IValue(6.0));
            std::cout << "  nfeatures = " << model.attr("nfeatures").toInt() << std::endl;
            std::cout << "  Rcs       = " << model.attr("Rcs").toDouble() << std::endl;
            std::cout << "  Rc        = " << model.attr("Rc").toDouble() << std::endl;
    	}
    }
    return model;
};

int load_models::_get_models(){
    // 変換した学習済みモデルの読み込み
    // 実行パス（not 実行ファイルパス）からの絶対パスに変換 https://nompor.com/2019/02/16/post-5089/
    this->module_ch  = this->_load_model(this->_model_dir+"/model_ch.pt", this->IF_CALC_CH);
    this->module_cc  = this->_load_model(this->_model_dir+"/model_cc.pt", this->IF_CALC_CC);
    this->module_co  = this->_load_model(this->_model_dir+"/model_co.pt", this->IF_CALC_CO);
    this->module_oh  = this->_load_model(this->_model_dir+"/model_oh.pt", this->IF_CALC_OH);
    this->module_o   = this->_load_model(this->_model_dir+"/model_o.pt",  this->IF_CALC_O);
    this->module_coc = this->_load_model(this->_model_dir+"/model_coc.pt",  this->IF_CALC_COC);
    this->module_coh = this->_load_model(this->_model_dir+"/model_coh.pt",  this->IF_CALC_COH);

    // 2024/6/22 元の実装は以下
    // if (manupilate_files::IsFileExist(std::filesystem::absolute(this->_model_dir+"/model_cc.pt"))) {
    //     this->IF_CALC_CC = true;
    //     this->module_cc = torch::jit::load(std::filesystem::absolute(this->_model_dir+"/model_cc.pt"));
    //     std::cout << " Loaded CC bond model: parameters are ..." << std::endl;
    //     try
    //     {
    //         std::cout << "  nfeatures = " << this->module_cc.attr("nfeatures").toInt() << std::endl;
    //         std::cout << "  Rcs       = " << this->module_cc.attr("Rcs").toDouble() << std::endl;
    //         std::cout << "  Rc        = " << this->module_cc.attr("Rc").toDouble() << std::endl;
    //     }
    //     // catch(const std::exception& e)
    //     catch (...)
    //     {
    // 		std::cout << "Exception occurred!" << std::endl;
    //     }
    // }
    std::cout << std::setw(30) << " IF_CALC_CH :: "  << this->IF_CALC_CH << std::endl;
    std::cout << std::setw(30) << " IF_CALC_CC :: "  << this->IF_CALC_CC << std::endl;
    std::cout << std::setw(30) << " IF_CALC_CO :: "  << this->IF_CALC_CO << std::endl;
    std::cout << std::setw(30) << " IF_CALC_OH :: "  << this->IF_CALC_OH << std::endl;
    std::cout << std::setw(30) << " IF_CALC_O :: "   << this->IF_CALC_O << std::endl;
    std::cout << std::setw(30) << " IF_CALC_COC :: " << this->IF_CALC_COC << std::endl;
    std::cout << std::setw(30) << " IF_CALC_COH :: " << this->IF_CALC_COH << std::endl;
    std::cout << std::setw(30) << " finish reading ML model file" << std::endl;

    // Stop calculation if all the IF_CALC_* are False.
    int CHECK_CALC = this->IF_CALC_CH + this->IF_CALC_CC + this->IF_CALC_CO + this->IF_CALC_OH + this->IF_CALC_O + this->IF_CALC_COC + this->IF_CALC_COH ;
    if ( CHECK_CALC == 0 ){
        error::exit("module_torch", "ALL IF_CALC is false. Please check modeldir is correct.");
    };
    return 0;
};

} // END namespace


