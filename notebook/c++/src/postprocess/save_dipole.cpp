// https://github.com/microsoft/vscode-cpptools/issues/7413
#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

/**
 * @file save_dipole.cpp
 * @author Tomohito Amano (you@domain.com)
 * @brief wrapper of savevec.cpp
 * @version 0.1
 * @date 2024-01-28
 * 
 * @copyright Copyright (c) 2024
 * 
 */

// #define _DEBUG
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstdio>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream> // https://www.cns.s.u-tokyo.ac.jp/~masuoka/post/inputfile_cpp/
#include <algorithm>
#include <numeric> // std::iota
// #include <boost/numeric/ublas/vector.hpp>
// #include <boost/numeric/ublas/matrix.hpp>
// #include <boost/numeric/ublas/io.hpp>
#include <Eigen/Core> // 行列演算など基本的な機能．

// https://qiita.com/meshidenn/items/53b7c6f35c6259320241
#include "../include/savevec.hpp"


int save_totaldipole(const std::vector<Eigen::Vector3d>& result_dipole_list, std::vector<std::vector<double> > unitcell, double temperature, double timestep, std::string savedir){

    // 最後にtotal双極子をファイルに保存
    std::stringstream ss;
    ss << "# index dipole_x dipole_y dipole_z \n"
        << "#UNITCELL[Ang] " 
        << unitcell[0][0] << " " << unitcell[0][1] << " " << unitcell[0][2] << " " << unitcell[1][0] << " " << unitcell[1][1] << " " << unitcell[1][2] << " " << unitcell[2][0] << " " << unitcell[2][1] << " " << unitcell[2][2]
        << "\n" 
        << "#TEMPERATURE[K] "
        << temperature
        << "\n"
        << "#TIMESTEP[fs]"
        << timestep;
    std::string firstline_tmp = ss.str();
    save_vec(result_dipole_list, savedir+"total_dipole.txt", firstline_tmp);
    return 0;
}


int save_bonddipole(const std::vector<std::vector<Eigen::Vector3d> >& result_bond_dipole_list,std::string savedir,std::string filename){
    // save files1: bond dipoleをファイルに保存
    // TODO :: （3D配列なのでもっと良い方法を考えないといけない） 
    std::string firstline = "# frame_index bond_index dipole_x dipole_y dipole_z";
    save_vec_index(result_bond_dipole_list,savedir+"/"+filename, firstline);
    return 0;
}

int save_moleculedipole(const std::vector<std::vector<Eigen::Vector3d> >& result_mol_dipole_list,std::string savedir){
    // 分子双極子の保存：本来3次元配列だが，frame,mol_id,d_x,d_y,d_zの形で保存することで二次元配列として保存する．
    std::string firstline = "# frame_index mol_index dipole_x dipole_y dipole_z";
    save_vec_index(result_mol_dipole_list, savedir+"/molecule_dipole.txt", firstline);
    return 0;
}

int postprocess_save_bonddipole(
    const std::vector<std::vector<Eigen::Vector3d> >& result_ch_dipole_list,
    const std::vector<std::vector<Eigen::Vector3d> >& result_co_dipole_list,
    const std::vector<std::vector<Eigen::Vector3d> >& result_oh_dipole_list,
    const std::vector<std::vector<Eigen::Vector3d> >& result_cc_dipole_list,
    const std::vector<std::vector<Eigen::Vector3d> >& result_o_dipole_list,
    const std::vector<std::vector<Eigen::Vector3d> >& result_coc_dipole_list,
    const std::vector<std::vector<Eigen::Vector3d> >& result_coh_dipole_list,
    const std::vector<std::vector<Eigen::Vector3d> >& result_molecule_dipole_list,
    std::string savedir){

    std::string firstline = "# frame_index bond_index dipole_x dipole_y dipole_z";

    // TODO :: （3D配列なのでもっと良い方法を考えないといけない） 
    save_vec_index(result_ch_dipole_list,savedir+"/ch_dipole.txt", "# chbond dipole \n" + firstline);
    save_vec_index(result_co_dipole_list,savedir+"/co_dipole.txt", "# cobond dipole \n" + firstline);
    save_vec_index(result_oh_dipole_list,savedir+"/oh_dipole.txt", "# ohbond dipole \n" + firstline);
    save_vec_index(result_cc_dipole_list,savedir+"/cc_dipole.txt", "# ccbond dipole \n" + firstline);
    save_vec_index(result_o_dipole_list ,savedir+ "/o_dipole.txt", "# obond dipole \n" + firstline);
    save_vec_index(result_coc_dipole_list,savedir+ "/coc_dipole.txt", "# cocbond dipole \n" + firstline);
    save_vec_index(result_coh_dipole_list,savedir+ "/coh_dipole.txt", "# cohbond dipole \n" + firstline);

    // 分子双極子の保存：本来3次元配列だが，frame,mol_id,d_x,d_y,d_zの形で保存することで二次元配列として保存する．
    save_vec_index(result_molecule_dipole_list, savedir+"/molecule_dipole.txt", "# molecular dipole \n" + firstline);

    return 0;

}