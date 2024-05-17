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

#include <iomanip>
#include <iostream>
#include "include/error.h"
#include "include/stopwatch.hpp"
#include "include/manupilate_files.hpp"
#include "module_input.hpp"
#include "parse.hpp"
#include "yaml-cpp/yaml.h" 

namespace module_input{
load_input::load_input(std::string xyzfilename, std::unique_ptr<diagnostics::Stopwatch> &timer){

    // get filename
    this->_inputfilename = std::filesystem::absolute(xyzfilename);
    std::cout << std::endl;
    std::cout << " ************************** SYSTEM INFO :: reading input *************************** " << std::endl;
    std::cout << std::setw(30) << "    Reading input file  :: " << this->_inputfilename  << std::endl;
    if (!manupilate_files::IsFileExist(this->_inputfilename)) {
        error::exit("load_input", "Error: inputfile file does not exist.");
    }
    //
    timer->reset(); // reset timer
    timer->start(); // restart timer
    if (this->_inputfilename.ends_with("yaml")){
        load_input::_get_input_yaml(); // load yaml
    } else{
        load_input::_get_input_text(); // load text
    }
    load_input::_print_variable();
    load_input::_check_savedir();
    timer->stop(); // stop timer
    std::cout << "     ELAPSED TIME (sec)      = " << timer->getElapsedSeconds() << std::endl;
    std::cout << std::endl;
};

int load_input::_get_input_text(){
    auto [inp_general, inp_desc, inp_pred] = parse::locate_tag(this->_inputfilename);
    std::cout << "FINISH reading inp file !! " << std::endl;
    auto tmp_var_gen = parse::var_general(inp_general);
    auto tmp_var_des = parse::var_descripter(inp_desc);
    auto tmp_var_pre = parse::var_predict(inp_pred);
    this->var_gen = tmp_var_gen;
    this->var_des = tmp_var_des;
    this->var_pre = tmp_var_pre;

    std::cout << "FINISH parse inp file !! " << std::endl;
    //
    if (var_des.IF_COC){
        std::cout << " =============== " << std::endl;
        std::cout << "  IF_COC is true " << std::endl;
        std::cout << " =============== " << std::endl;
    }
    return 0;
};

int load_input::_get_input_yaml(){

    YAML::Node node   = YAML::LoadFile(this->_inputfilename);
    YAML::Node config_gen = node["general"];
    YAML::Node config_pre = node["predict"];
    YAML::Node config_des = node["descriptor"];
    if (config_gen.size() == 0){error::exit("_get_input_yaml", "ERROR NO &general fuund");};
    if (config_des.size() == 0){error::exit("_get_input_yaml", "ERROR NO &descriptor fuund");};
    if (config_pre.size() == 0){error::exit("_get_input_yaml", "ERROR NO &predict fuund");};
    std::cout << "FINISH reading inp file !! " << std::endl;
    auto tmp_var_gen = parse::var_general(config_gen);
    auto tmp_var_des = parse::var_descripter(config_des);
    auto tmp_var_pre = parse::var_predict(config_pre);
    this->var_gen = tmp_var_gen;
    this->var_des = tmp_var_des;
    this->var_pre = tmp_var_pre;

    std::cout << "FINISH parse inp file !! " << std::endl;
    //
    if (var_des.IF_COC){
        std::cout << " =============== " << std::endl;
        std::cout << "  IF_COC is true " << std::endl;
        std::cout << " =============== " << std::endl;
    }
    return 0;
};

int load_input::_print_variable(){
    this->var_gen.print_variable();
    this->var_des.print_variable();
    this->var_pre.print_variable();
    return 0;
}

int load_input::_check_savedir(){
    //! 保存するディレクトリの存在を確認
    if (!manupilate_files::IsDirExist(std::filesystem::absolute(this->var_gen.savedir))){
        error::exit("load_input" , " ERROR :: savedir does not exist !! "); // this->var_gen.savedir;
    }
    return 0;
};
} // END namespace