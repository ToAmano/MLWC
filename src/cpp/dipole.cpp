// https://github.com/microsoft/vscode-cpptools/issues/7413
#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif
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
// #include <rdkit/GraphMol/GraphMol.h>
// #include <rdkit/GraphMol/FileParsers/MolSupplier.h>
#include <Eigen/Core> // 行列演算など基本的な機能．
#include <Eigen/Dense> // vector3dにはこれが必要？
#include "numpy.hpp"
#include "npy.hpp"
#include "include/printvec.hpp"
#include "mol_core.hpp"
// #include "numpy_quiita.hpp" // https://qiita.com/ka_na_ta_n/items/608c7df3128abbf39c89
// numpy_quiitaはsscanf_sが読み込めず，残念ながら現状使えない．



/**
 * ポスト処理で双極子を色々処理する部分のコード
 * 
*/


double calc_dielconst(std::vector<Eigen::Vector3d> result_dipole_list){
    /**
     * result_dipole_list :: [frame, 3dvector]の形の配列
     * 
    */
    // 変数の定義
    Eigen::Vector3d mm_average = Eigen::Vector3d::Zero();
    Eigen::Vector3d m_average  = Eigen::Vector3d::Zero();

    // <M^2>，<M>^2を計算する
    // TODO :: 発散に注意
    for (int i=0, N=result_dipole_list.size();i<N;i++){
        mm_average = mm_average + result_dipole_list[i]*result_dipole_list[i];
        m_average  = m_average  + result_dipole_list[i];
    };
    mm_average = mm_average/result_dipole_list.size();
    m_average = (m_average/result_dipole_list.size())*(m_average/result_dipole_list.size());

    // 比例係数を計算する

}