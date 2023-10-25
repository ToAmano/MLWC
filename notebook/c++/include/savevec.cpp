/*
2023/10/09
vectorをsaveする関数たち．
*/ 

#include<stdio.h>
#include<iomanip>
#include<vector>
#include<map>
#include<string>
#include<iostream>
#include<fstream>
#include <Eigen/Core> // 行列演算など基本的な機能．
#include"savevec.hpp"


void save_vec(const std::vector<std::vector<Eigen::Vector3d> >  &vector3d, const std::string outputfile_name, const std::string firstline){
    /**
     * @fn
     * 3d vecを単純に保存する．
    vector3d :: 保存する3d vector (ただし，最後はEigen::Vector3d)
    outputfile_name :: 出力ファイル名
    */
    std::cout << " save 3d vector into " << outputfile_name << std::endl; 
    std::ofstream fout(outputfile_name); 
    fout << firstline << std::endl;
    for (int i = 0, n=vector3d.size() ; i < n; i++){
        fout << std::setw(5) << i ; // 最初にiをprintすることで，3D arrayを通常の2d arrayとして保存可能 !!
        for (int j = 0; j< vector3d[i].size(); j++){
            fout << std::right << std::setw(16) << vector3d[i][j][0] << std::setw(16) << vector3d[i][j][1] << std::setw(16) << vector3d[i][j][2] << " " ;
        }
        fout << std::endl;
    }
    fout.close();
}


void save_vec(const std::vector<Eigen::Vector3d>  &vector2d, const std::string outputfile_name, const std::string firstline){
    /**
     * @fn
     * total_dipoleの保存用．2d vecの先頭にindeをつける．
     * @brief 要約説明
     * @param[in] vector2d        :: 保存する2d vector (ただし，最後はEigen::Vector3d)
     * @param[in] outputfile_name :: 出力ファイル名
    */
    std::cout << " save 2d vector into " << outputfile_name << std::endl; 
    std::ofstream fout(outputfile_name); 
    fout << firstline << std::endl;
    for (int i = 0, n = vector2d.size(); i < n; i++){
        fout << std::setw(5) << i << std::right << std::setw(16) << vector2d[i][0] << std::setw(16) << vector2d[i][1] << std::setw(16) << vector2d[i][2] << std::endl;
    }
    fout.close();
}


void save_vec_index(const std::vector<std::vector<Eigen::Vector3d> >  &vector3d, const std::string outputfile_name, const std::string firstline){
    /**
     * @fn
     * total_dipoleの保存用．2d vecの先頭にindeをつける．
     * @brief 要約説明
     * @param[in] vector2d        :: 保存する2d vector (ただし，最後はEigen::Vector3d)
     * @param[in] outputfile_name :: 出力ファイル名
    */
    std::cout << " save 3d vector into " << outputfile_name << std::endl; 
    std::ofstream fout(outputfile_name); 
    fout << firstline << std::endl;
    for (int i = 0, n = vector3d.size(); i < n; i++){ // frameについてのLoop
        for (int j=0, m = vector3d[i].size(); j<m; j++){ // 分子数についてのLoop
            fout << std::setw(5) << i << std::setw(5) << j << std::right << std::setw(16) << vector3d[i][j][0] << std::setw(16) << vector3d[i][j][1] << std::setw(16) << vector3d[i][j][2] << std::endl;
        }
    }
    fout.close();
}

    // std::ofstream fout_moleculedipole(var_des.savedir+"molecule_dipole.txt"); 
    // fout_moleculedipole << "# index dipole_x dipole_y dipole_z" << std::endl;
    // for (int i = 0; i < result_molecule_dipole_list.size(); i++){ // frameについてのLoop
    //     // 双極子の出力ファイル
    //     for (int j=0; j < NUM_MOL; j++){ // 分子数についてのLoop
    //         fout_moleculedipole << std::setw(5) << i << std::setw(5) << j << std::right << std::setw(16) << result_molecule_dipole_list[i][j][0] << std::setw(16) << result_molecule_dipole_list[i][j][1] << std::setw(16) << result_molecule_dipole_list[i][j][2] << std::endl;
    //     };
    // };
    // fout_moleculedipole.close();