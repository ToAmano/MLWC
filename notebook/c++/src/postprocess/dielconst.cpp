/**
 * @file dielconst.cpp
 * @brief ポストプロセスとして，誘電定数や分子双極子の平均値を計算する
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
#include <fstream>
#include <iostream>
#include <iomanip>
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
#include <deque> // deque
// #include <boost/numeric/ublas/vector.hpp>
// #include <boost/numeric/ublas/matrix.hpp>
// #include <boost/numeric/ublas/io.hpp>
#include <Eigen/Core> // 行列演算など基本的な機能．
#include "dielconst.hpp"
// https://qiita.com/meshidenn/items/53b7c6f35c6259320241
#include <numeric> // accumulate関数に必要


double calc_moldipole_mean(std::vector<std::vector<Eigen::Vector3d> > result_molecule_dipole_list){
    /**
     * @fn 分子双極子の絶対値の平均値を計算する．最初に和をとるのだが，あまりに構造の数が多い場合に値の発散を防ぐため，
     * @fn 最初に1frameでの平均値を計算しておいて，次に各frameでの平均値を計算する．すなわち，以下のような変形をやる．
     * @fn sum_{i=1}^{N=frame*num_mol} M/N = sum_{i=1}^{frame} (sum_{j=1}^{num_mol} M/num_mol ) / frame
     * @param[in] result_molecule_dipole_list main.cppで計算する分子双極子データが入った配列
     * @return double absolute_mol_dipole 分子双極子の絶対値の全平均
    */
    int num_frame = result_molecule_dipole_list.size();
    int num_mol   = result_molecule_dipole_list[0].size();
    // Eigen::Vector3d tmp_absolute_mol_dipole = Eigen::Vector3d::Zero(); // 1frameでの分子双極子の絶対値の格納場所
    double absolute_mol_dipole           = 0 ; // 1frameでの分子双極子の絶対値の格納場所
    double tmp_absolute_mol_dipole_frame = 0 ; // frameごとの分子双極子の絶対値の格納場所
    // std::vector< std::vector<Eigen::Vector3d>>  vectors(, Eigen::Vector3d::Zero())
    for (int i=0;i<num_frame;i++){
        tmp_absolute_mol_dipole_frame = 0; // 初期化
        for (int j=0;j<num_mol;j++){
            // frame iでの分子双極子の絶対値を求めて足す
            tmp_absolute_mol_dipole_frame += result_molecule_dipole_list[i][j].norm();
            // std::cout << "sum1 = " << std::accumulate(result_molecule_dipole_list.begin(), result_molecule_dipole_list.end(), Eigen::MatrixXd::Zero(3,4).eval()) << std::endl;
        }
        // 1frameでの平均値を求める
        tmp_absolute_mol_dipole_frame = tmp_absolute_mol_dipole_frame/num_mol;
        // frameごとの平均値
        absolute_mol_dipole          += tmp_absolute_mol_dipole_frame;
    };
    absolute_mol_dipole = absolute_mol_dipole/num_frame;
    return absolute_mol_dipole;
    // std::cout << "sum1 = " << std::accumulate(.begin(), tensor.end(), Eigen::MatrixXd::Zero(3,4).eval()) << std::endl;
    // std::cout << "sum1 = " << std::accumulate(tensor.begin(), tensor.end(), Eigen::MatrixXd::Zero(3,4).eval()) << std::endl;
};

double calc_moldipole_stderr(std::vector<std::vector<Eigen::Vector3d> > result_molecule_dipole_list, double mean_absolute_mol_dipole){
    /**
     * @fn 分子双極子の絶対値の標準偏差を計算する．式としては
     * @fn s = sqrt(sum_{i}^{N}|m_i-m_mean|^2/N)
     * @fn で与えられる標準偏差である．（不変分散ではない）
     * @param[in] result_molecule_dipole_list main.cppで計算する分子双極子データが入った配列
     * @param[in] mean_absolute_mol_dipole calc_moldipole_meanで計算した分子双極子の絶対値
     * @return double absolute_mol_dipole 分子双極子の絶対値の全平均
    */
    int num_frame = result_molecule_dipole_list.size();
    int num_mol   = result_molecule_dipole_list[0].size();
    double stderr_absolute_mol_dipole    = 0 ; // frameごとの分子双極子の絶対値の格納場所
    // std::vector< std::vector<Eigen::Vector3d>>  vectors(, Eigen::Vector3d::Zero())
    for (int i=0;i<num_frame;i++){
        for (int j=0;j<num_mol;j++){
            // [i][j]の要素を計算
            stderr_absolute_mol_dipole += pow(result_molecule_dipole_list[i][j].norm()-mean_absolute_mol_dipole,2);
        }
    };
    stderr_absolute_mol_dipole = std::sqrt(stderr_absolute_mol_dipole/num_frame/num_mol);
    return stderr_absolute_mol_dipole;
};


double calc_M2(std::vector<Eigen::Vector3d> result_dipole_list){
    /**
     * @fn Total dipoleの二乗平均を計算する．式としては
     * @fn <M^2> = <M_x^2>+<M_y^2>+<M_z^2>
     * @fn で与えられる．誘電定数の計算に必要となる．
     * @fn 工夫として，発散を防ぐために，x,y,z成分を別々の変数に格納して和を取るようにしている．
     * @param[in] result_dipole_list main.cppで計算するtotal双極子データが入った配列．形は[NUM_CONFIG,3d vector]となる．
     * @return double <M^2>
     * TODO :: std::accumulatorを使ってスマートに実装したい．
    */
    int num_frame = result_dipole_list.size();
    double mean_square_x = 0;
    double mean_square_y = 0;
    double mean_square_z = 0;
    for (int i=0; i < num_frame; i++){
        mean_square_x += result_dipole_list[i][0]*result_dipole_list[i][0]; //M_x^2
        mean_square_y += result_dipole_list[i][1]*result_dipole_list[i][1]; //M_y^2
        mean_square_z += result_dipole_list[i][2]*result_dipole_list[i][2]; //M_z^2
    }
    mean_square_x = mean_square_x/num_frame;
    mean_square_y = mean_square_y/num_frame;
    mean_square_z = mean_square_z/num_frame;
    return mean_square_x+mean_square_y+mean_square_z;
}

double calc_M(std::vector<Eigen::Vector3d> result_dipole_list){
    /**
     * @fn Total dipoleの平均の二乗を計算する．式としては
     * @fn <M>^2 = <M_x>^2+<M_y>^2+<M_z>^2
     * @fn で与えられる．誘電定数の計算に必要となる．
     * @fn 工夫として，発散を防ぐために，x,y,z成分を別々の変数に格納して和を取るようにしている．
     * @param[in] result_dipole_list main.cppで計算するtotal双極子データが入った配列．形は[NUM_CONFIG,3d vector]となる．
     * @return double <M>^2
     * TODO :: std::accumulatorを使ってスマートに実装したい．
    */
    int num_frame = result_dipole_list.size();
    double mean_square_x = 0;
    double mean_square_y = 0;
    double mean_square_z = 0;
    for (int i=0; i < num_frame; i++){
        mean_square_x += result_dipole_list[i][0]; //M_x
        mean_square_y += result_dipole_list[i][1]; //M_y
        mean_square_z += result_dipole_list[i][2]; //M_z
    }
    mean_square_x = pow(mean_square_x/num_frame,2); 
    mean_square_y = pow(mean_square_y/num_frame,2);
    mean_square_z = pow(mean_square_z/num_frame,2);
    return mean_square_x+mean_square_y+mean_square_z;
}

double calc_dielconst(double temperature,std::vector<std::vector<double> > UNITCELL_VECTORS, double mean_M2,double mean_M){
    /**
     * @fn 誘電定数を計算する．
     * @fn eps^0 = eps^inf+(1/3eps_0*k_B*T*V)*(<M^2>-<M>^2)
     * @fn で与えられる．
     * @fn 完全な収束のためには，<M>^2=0となっていることが必要条件．
     * @fn ただし，それだけでは<M^2>が収束していないことも多いので，可視化してみる必要がある．
     * @fn 経験的には通常の分子性液体の場合，50nsのトラジェクトリでdt=1psごとにサンプリングし，合計5万構造程度とれば十分収束する．
     * @fn 極力多様な構造をサンプリングするため，トラジェクトリを長く，dtを大きくするのが処方箋．
     * @fn 先行研究では，系の大きさへの依存性はそこまでではないという報告がある．
     * @fn 単位変換については以下の処方箋に従う．
     * @fn 双極子：デバイ単位をC*mへ変換
     * @fn 格子定数：Angstrom単位をmへ変換
     * @fn 真空誘電率はF/m．F=C^2/Jより，C^2/Jmとなる．
     * @fn k_B：J/Kだから，温度をKで与えればk_B*T=Jとなる．
     * @fn 以上から誘電定数は無次元量となる．
     * @fn M^2/eps_0k_BTV = C^2m^2/((C^2/Jm)*J*m^3)= dimensionless
     * @fn また，eps^infについては，現状eps^inf=1（真空）で計算するようになっている．
     * @param[in] result_dipole_list main.cppで計算するtotal双極子データが入った配列．形は[NUM_CONFIG,3d vector]となる．
     * @return double dielconst
    */
    double eps0 = 8.8541878128e-12;
    double debye = 3.33564e-30;
    double nm3 = 1.0e-27;
    double nm = 1.0e-9;
    double A3 = 1.0e-30;
    double kb = 1.38064852e-23;
    // TODO :: 現在直方晶のみ対応
    double unitcell_volume = UNITCELL_VECTORS[0][0]*UNITCELL_VECTORS[1][1]*UNITCELL_VECTORS[2][2];
    std::cout << "WARNING :: Currently, eps^inf=1. And UNITCELL should be orthorhombic." << std::endl;
    double dielconst = 1.0+ ((mean_M2-mean_M)*(debye*debye))/(3.0*unitcell_volume*kb*temperature*eps0);
    return dielconst;
};


void postprocess_dielconst(std::vector<Eigen::Vector3d> result_dipole_list,std::vector<std::vector<Eigen::Vector3d> > result_molecule_dipole_list, double temperature, std::vector<std::vector<double> > UNITCELL_VECTORS){
    /**
     * @fn ポストプロセスとして，誘電定数まわりの計算をこなす．
     * @fn 1: 誘電定数の計算
     * @fn 2: <M^2>, <M>^2の出力
     * @fn 3: 分子双極子の計算
     * @param[in] result_dipole_list main.cppで計算するtotal双極子データが入った配列．形は[NUM_CONFIG,3d vector]となる．
     * @return double <M>^2
    */
    // 分子双極子周りの値
    double mean_absolute_mol_dipole = calc_moldipole_mean(result_molecule_dipole_list);
    double stderr_absolute_mol_dipole = calc_moldipole_stderr(result_molecule_dipole_list, mean_absolute_mol_dipole);
    // 誘電定数まわりの値
    double mean_M2 = calc_M2(result_dipole_list);
    double mean_M  = calc_M(result_dipole_list);
    double dielconst = calc_dielconst(temperature,UNITCELL_VECTORS, mean_M2,mean_M);

    // total dipoleは別ファイルへ出力するようにする．
    std::ofstream fout_dielconst("DIELCONST"); 
    fout_dielconst << "calculated mean dipole & dielectric constants" << std::endl;
    fout_dielconst << "WARNING eps^inf is fixed to 1.0, and we only support orthorhombic lattice." << std::endl;
    fout_dielconst << std::setw(30) << "mean_absolute_mol_dipole "   << std::right << std::setw(16) << mean_absolute_mol_dipole << std::endl;
    fout_dielconst << std::setw(30) << "stderr_absolute_mol_dipole " << std::right << std::setw(16) << stderr_absolute_mol_dipole << std::endl;
    fout_dielconst << std::setw(30) << "mean_M2(<M^2>) "             << std::right << std::setw(16) << mean_absolute_mol_dipole << std::endl;
    fout_dielconst << std::setw(30) << "mean_M(<M>^2)  "             << std::right << std::setw(16) << mean_absolute_mol_dipole << std::endl;
    fout_dielconst << std::setw(30) << "eps^0          "             << std::right << std::setw(16) << dielconst << std::endl;
    
};
