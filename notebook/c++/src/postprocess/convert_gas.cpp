/**
 * @file convert_gas.cpp
 * @brief gas計算で得られたデータを全て元の形式に変換する．おそらくちょっと遅い関数になってしまうので，将来的により早い方法を考えた方が良い．
 * @author Tomohito Amano
 * @date 2024/1/4
 */

// https://github.com/microsoft/vscode-cpptools/issues/7413
#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

// #define _DEBUG
#include <stdio.h>
#include <filesystem>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <string>
#include <cstdio>
#include <vector>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream> // https://www.cns.s.u-tokyo.ac.jp/~masuoka/post/inputfile_cpp/
#include <regex> // using cmatch = std::match_results<const char*>;
#include <map> // https://bi.biopapyrus.jp/cpp/syntax/map.html
#include <cmath> 
#include <algorithm>
#include <numeric> // std::iota
#include <tuple> // https://tyfkda.github.io/blog/2021/06/26/cpp-multi-value.html
#include <Eigen/Core> // 行列演算など基本的な機能．
#include <Eigen/Dense> // vector3dにはこれが必要？
#include "convert_gas.hpp"

std::vector<std::vector<Eigen::Vector3d> > convert_bond_dipole(const std::vector<std::vector<Eigen::Vector3d> >& gas_dipole_list, const int NUM_CONFIG, const int NUM_MOL){
    /**
     * @fn result_ch_dipole_listの類のボンド依存の量（[num_frame,num_bond,3d vector]）を変換する．
     * @fn よく考えると，これらの関数は以下のようなreshapeをすれば良い．
     * @fn gas model :: [num_frame*num_mol, num_bond, 3d vector]
     * @fn liquid    :: [num_frame, num_bond*num_mol, 3d vector]
     * @fn 
    */
    std::vector<std::vector<Eigen::Vector3d> > result_dipole_list(NUM_CONFIG); // 結果
    std::vector<Eigen::Vector3d> tmp; // 1frameでのvector
    for (int i=0; i<NUM_CONFIG; i++ ){ // 元のフレームに関するループ
        for (int j=0; j<NUM_MOL; j++){ // 分子数に関するループ
            // gas_dipoleの該当部分(frame,分子指定)をtmpにappendする
            tmp.insert(tmp.end(), std::begin(gas_dipole_list[i*NUM_MOL+j]), std::end(gas_dipole_list[i*NUM_MOL+j]));
        }
        result_dipole_list[i] = tmp;
        tmp.clear();
    }
    return result_dipole_list;
}


std::vector<Eigen::Vector3d> convert_total_dipole(const std::vector<Eigen::Vector3d>& gas_dipole_list, const int NUM_FRAME, const int NUM_MOL){
    /**
     * @brief total dipoleをgasからliquidに変換する
     * 
     * @return std::vector<Eigen::Vector3d> 
     */
    
    std::vector<Eigen::Vector3d> result_dipole_list(NUM_FRAME); // 結果
    Eigen::Vector3d tmp; // 1frameでのtotal dipole
    for (int i=0; i<NUM_FRAME; i++ ){ // 元のフレームに関するループ   
        tmp = Eigen::Vector3d::Zero(3); //初期化
        for (int j=0; j<NUM_MOL; j++){ // 分子数に関するループ
            tmp += gas_dipole_list[i*NUM_MOL+j]; // 全双極子を計算
        }
        result_dipole_list[i] = tmp;
    }
    return result_dipole_list;
} 
