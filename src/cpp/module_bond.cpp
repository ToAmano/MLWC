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

#include <Eigen/Core> 
#include <Eigen/Dense> 
#include <vector>
#include <iomanip>
#include <iostream>
#include "include/manupilate_files.hpp"
#include "include/stopwatch.hpp"
#include "include/error.h"
#include "chemicalbond/mol_core.hpp"
#include "chemicalbond/mol_core_rdkit.hpp"
#include "module_bond.hpp"

namespace module_bond{
load_bond::load_bond(std::string bondfilename, std::unique_ptr<diagnostics::Stopwatch> &timer){

    // get filename
    this->_bondfilename = std::filesystem::absolute(bondfilename);
    std::cout << std::endl;
    std::cout << " ************************** SYSTEM INFO :: reading bondinfo *************************** " << std::endl;
    std::cout << std::setw(30) << "    Reading the xyz file  :: " << this->_bondfilename  << std::endl;
    if (!manupilate_files::IsFileExist(this->_bondfilename)) {
        error::exit("load_bond", "Error: bondfile file does not exist.");
    }
    //
    timer->reset(); // reset timer
    timer->start(); // restart timer
    load_bond::_get_bondinfo(); // load read_mol (include wannier)
    load_bond::_get_NUM_MOL_ATOMS(); // load NUM_MOL_ATOMS (exclude wannier)
    timer->stop(); // stop timer
    std::cout << "     ELAPSED TIME (sec)      = " << timer->getElapsedSeconds() << std::endl;
    std::cout << std::endl;
};

int load_bond::_get_NUM_MOL_ATOMS(){
    // TODO :: check if bondinfo is correctly setup
    this->NUM_MOL_ATOMS = this->bondinfo.num_atoms_per_mol; // # of atoms in one molecule
    std::cout << std::setw(10) << "  NUM_MOL_ATOMS :: " << this->NUM_MOL_ATOMS << std::endl;
    return 0;
};

int load_bond::_get_bondinfo(){
    this->bondinfo = read_mol(this->_bondfilename);
    std::cout << " finish reading bond file" << std::endl;
    return 0;
};

} // END namespace
