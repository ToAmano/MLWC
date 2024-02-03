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
#include "parse.cpp"


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
    load_input::_get_input_text(); // load ALL_NUM_ATOM (include wannier)
    load_input::_check_savedir();
    timer->stop(); // stop timer
    std::cout << "     ELAPSED TIME (sec)      = " << timer->getElapsedSeconds() << std::endl;
    std::cout << std::endl;
};

int load_input::_get_input_text(){
    auto [inp_general, inp_desc, inp_pred] = locate_tag(inp_filename);
    std::cout << "FINISH reading inp file !! " << std::endl;
    this->var_gen = var_general(inp_general);
    this->var_des = var_descripter(inp_desc);
    this->var_pre = var_predict(inp_pred);
    std::cout << "FINISH parse inp file !! " << std::endl;
    //
    if (this->var_des.IF_COC){
        std::cout << " =============== " << std::endl;
        std::cout << "  IF_COC is true " << std::endl;
        std::cout << " =============== " << std::endl;
    }
    return 0;
};

int load_input::_check_savedir(){
    //! 保存するディレクトリの存在を確認
    if (!manupilate_files::IsDirExist(std::filesystem::absolute(this->var_gen.savedir))){
        error::exit("load_input" , " ERROR :: savedir does not exist !! ::  "+this->var_gen.savedir);
    }
    return 0;
};