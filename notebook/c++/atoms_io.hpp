// #define _DEBUG
#include <stdio.h>
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
// #include <boost/numeric/ublas/vector.hpp>
// #include <boost/numeric/ublas/matrix.hpp>
// #include <boost/numeric/ublas/io.hpp>
#include <Eigen/Core> // 行列演算など基本的な機能．
#include "numpy.hpp"
#include "npy.hpp"
// #include "numpy_quiita.hpp" // https://qiita.com/ka_na_ta_n/items/608c7df3128abbf39c89
// numpy_quiitaはsscanf_sが読み込めず，残念ながら現状使えない．
#include "atoms_core.cpp"



/*
ase_io_readとase_io_writeを定義するファイル．
*/

int raw_cpmd_num_atom(const std::string filename){
    /*
    xyzファイルから原子数を取得する．（ワニエセンターが入っている場合その原子数も入ってしまうので注意．）
    */
};


std::vector<std::vector<double> > raw_cpmd_get_unitcell_xyz(const std::string filename = "IONS+CENTERS.xyz") {
    /*
    xyzファイルから単位格子ベクトルを取得する．

     Lattice="16.267601013183594 0.0 0.0 0.0 16.267601013183594 0.0 0.0 0.0 16.267601013183594" Properties=species:S:1:pos:R:3 pbc="T T T"
    という文字列から，Lattice=" "の部分を抽出しないといけない．
    */
};


std::vector<Atoms> ase_io_read(const std::string filename, const int NUM_ATOM, const std::vector<std::vector<double> > unitcell_vec){
    /*
    TODO :: positionsとatomic_numのpush_backは除去できる．（いずれもNUM_ATOM個）
    MDトラジェクトリを含むxyzファイルから
        - 格子定数
        - 原子番号
        - 原子座標
    を取得して，Atomsのリストにして返す．
    読み込み簡単化&高速化のため，予めNUM_ATOMを取得しておく．
    */
}

std::vector<Atoms> ase_io_read(std::string filename){
    /*
    大元のase_io_read関数のオーバーロード版．ファイル名を入力するだけで格子定数などを全て取得する．
    */
}

std::vector<Atoms> ase_io_read(const std::string filename, const int NUM_ATOM, const std::vector<std::vector<double> > unitcell_vec, bool IF_REMOVE_WANNIER){
    /*
    TODO :: positionsとatomic_numのpush_backは除去できる．（いずれもNUM_ATOM個）
    MDトラジェクトリを含むxyzファイルから
        - 格子定数
        - 原子番号
        - 原子座標
    を取得して，Atomsのリストにして返す．
    読み込み簡単化&高速化のため，予めNUM_ATOMを取得しておく．
    */
}

int ase_io_write(const std::vector<Atoms> &atoms_list, const std::string filename ){
    /*
    */
};


int ase_io_write(const Atoms &aseatoms, std::string filename ){
    /*
    ase_io_writeの別バージョン（オーバーロード）
    入力がaseatomsひとつだけだった場合にどうなるかのチェック．
    */
};