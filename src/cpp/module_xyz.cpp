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
#include <iomanip>
#include <iostream>
#include "include/manupilate_files.hpp"
#include "include/stopwatch.hpp"
#include "include/error.h"
#include "atoms_io.hpp"
#include "atoms_core.hpp"
#include "module_xyz.hpp"

namespace module_xyz{
load_xyz::load_xyz(std::string xyzfilename, std::unique_ptr<diagnostics::Stopwatch> &timer){

    // get filename
    this->_xyzfilename = std::filesystem::absolute(xyzfilename);
    //! 原子数の取得(もしXがあれば除く)
    std::cout << std::endl;
    std::cout << " ************************** SYSTEM INFO :: reading XYZ *************************** " << std::endl;
    std::cout << std::setw(30) << "    Reading the xyz file  :: " << this->_xyzfilename  << std::endl;
    if (!manupilate_files::IsFileExist(this->_xyzfilename)) {
        error::exit("load_xyz", "Error: xyzfile file does not exist.");
    }
    //
    timer->reset(); // reset timer
    timer->start(); // restart timer
    load_xyz::_get_ALL_NUM_ATOM(); // load ALL_NUM_ATOM (include wannier)
    load_xyz::_get_NUM_ATOM(); // load NUM_ATOM (exclude wannier)
    load_xyz::_get_UNITCELL_VECTOR(); // load UNITCELL_VECTOR
    load_xyz::_get_atoms_list(); // load atoms
    timer->stop(); // stop timer
    std::cout << "     ELAPSED TIME (sec)      = " << timer->getElapsedSeconds() << std::endl;
    std::cout << std::endl;
};

int load_xyz::_get_ALL_NUM_ATOM(){
    this->ALL_NUM_ATOM = raw_cpmd_num_atom(this->_xyzfilename); //! wannierを含む原子数
    if (! (manupilate_files::get_num_lines(this->_xyzfilename) % (ALL_NUM_ATOM+2) ==0 )){ //! 行数がちゃんと割り切れるかの確認
        error::exit("load_xyz", "ERROR(load_xyz::_get_ALL_NUM_ATOM) :: ALL_NUM_ATOM does not match the line of input xyz file \n PLEASE check you do not have new line in the final line"); //TODO :: 最後に改行があるとおかしいことになる．
    };
    return 0;
};

int load_xyz::_get_NUM_ATOM(){
    this->NUM_ATOM = get_num_atom_without_wannier(this->_xyzfilename); //! WANを除いた原子数
    std::cout << std::setw(30) << "   NUM_ATOM :: " << NUM_ATOM << std::endl;
    return 0;
};

int load_xyz::_get_UNITCELL_VECTOR(){
    this->UNITCELL_VECTORS = raw_cpmd_get_unitcell_xyz(std::filesystem::absolute(this->_xyzfilename));
    std::cout << std::setw(30) << "  UNITCELL_VECTORS (Ang) :: " << UNITCELL_VECTORS[0][0] << std::endl;
    return 0;
}

int load_xyz::_get_atoms_list(){
    /**
     * @brief get atoms from xyz
     * 
     */
    bool IF_REMOVE_WANNIER = true;
    this->atoms_list = ase_io_read(this->_xyzfilename, IF_REMOVE_WANNIER);
    this->NUM_CONFIG = atoms_list.size(); // totalのconfiguration数
    std::cout << " finish reading xyz file  "  <<  std::endl;
    std::cout << std::setw(30) << "   NUM_CONFIG :: " << NUM_CONFIG << std::endl;
    std::cout << " ------------------------------------" << std::endl;
    std::cout << "" << std::endl;
    return 0;
};

} // END namespace
