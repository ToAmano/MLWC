/*
2023/7/28
vectorをprintする関数たち．
*/ 

#ifndef INCLUDE_HPP_savevec
#define INCLUDE_HPP_savevec

#include<stdio.h>
#include<vector>
#include<map>
#include<string>
#include<iostream>
#include<iostream>
#include<fstream>
#include <Eigen/Core> // 行列演算など基本的な機能．

/*
全ての基本的なvectorの保存関数をsave_vecという名前にする．overrideでinputの微妙な違いに対応する．
*/

// 3d vector（のうち，最後がEigen::vector3d)のsave関数
void save_vec(const std::vector<std::vector< Eigen::Vector3d > > &vector3d, const std::string outputfile_name,const std::string firstline="# index dipole_x dipole_y dipole_z");

void save_vec(const std::vector<Eigen::Vector3d>  &vector2d, const std::string outputfile_name, const std::string firstline="# index dipole_x dipole_y dipole_z");

// 3d vector
//  二つのindexを保存する．
void save_vec_index(const std::vector<std::vector<Eigen::Vector3d> > &vector3d, const std::string outputfile_name, const std::string firstline="# index dipole_x dipole_y dipole_z");

#endif //! INCLUDE_HPP_savevec