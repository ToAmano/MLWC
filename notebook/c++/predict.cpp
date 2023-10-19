/**
 * @file ファイル名.h
 * @brief 簡単な説明
 * @author 書いた人
 * @date 日付（開始日？）
 */

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

#include "atoms_io.hpp"
#include "descriptor.hpp" // TODO hpp化

#include "torch/script.h" // pytorch
#include <Eigen/Core> // 行列演算など基本的な機能．
#include "numpy.hpp"
#include "npy.hpp"
#include "predict.hpp"
#include "include/error.h"
#include "include/savevec.hpp"
#include "include/printvec.hpp"
#include "include/constant.hpp"


/*
予測部分の関数
*/

int save_descriptor(const std::vector<std::vector<double> > &descs, const std::string name, const int i){
    /**
    * descriptor_at_frameをnpy形式で保存する．    
    * 保存するファイル名はdescs_<name><i>.npyとなる
    */
    // descs_chの形を1dへ変形してnpyで保存．
    // TODO :: さすがにもっと効率の良い方法があるはず．
    std::vector<double> descs_1d;
    for (int i = 0, n=descs.size(); i < n; i++) {
        for (int j = 0; j < descs[i].size(); j++) { //これが288個のはず
            descs_1d.push_back(descs[i][j]); 
        }
    }
    //! npy.hppを利用して保存する．
    const std::vector<long unsigned> shape_descs{descs.size(), descs[0].size()}; // vectorを1*12の形に保存
    npy::SaveArrayAsNumpy("descs_"+name+std::to_string(i)+".npy", false, shape_descs.size(), shape_descs.data(), descs_1d);
    return 0;
};

Eigen::Vector3d predict_dipole(const std::vector<double> &descs, torch::jit::script::Module model_dipole){
    /**
     * 予測部分を関数化したもの．記述子，modelを用意すれば予測を行う．
     * TODO :: 記述子の長さは自動的に取得される用になっていて，長さのマッチングは行われない．
    */
    int descs_length=descs.size(); //! 記述子の長さ取得

    torch::Tensor input_for_model = torch::ones({1, descs_length}).to("cpu"); //! input用のtensor
    // torch::Tensor input = torch::tensor(torch::ArrayRef<double>({descs_ch[i]})).to("cpu");
    // https://stackoverflow.com/questions/63531428/convert-c-vectorvectorfloat-to-torchtensor
    // 入力となる記述子にvectorから値をcopy 
    // TODO :: 多分もっと綺麗な方法があるはず．．． ただ1次元ではなく(1,288)という形をしているが故にちょっと問題になっている．
    // torch::Tensor input = torch::from_blob(descs_tmp.data(), {1,288}).to("cpu");
    for (int k = 0; k<descs_length;k++){
        input_for_model[0][k] = descs[k];
    };
    // std::cout << input << std::endl ;
    // 推論と同時に出力結果を変数に格納
    // auto elements = module.forward({input}).toTuple() -> elements();
    torch::Tensor elements = model_dipole.forward({input_for_model}).toTensor() ;
    // 出力結果
    // std::cout << j << " " << elements[0][0].item() << " " << elements[0][1].item() << " " << elements[0][2].item() << std::endl;
    auto tmpDipole = Eigen::Vector3d {elements[0][0].item().toDouble(), elements[0][1].item().toDouble(), elements[0][2].item().toDouble()};
    return tmpDipole;
}

std::tuple< std::vector< Eigen::Vector3d >, std::vector< Eigen::Vector3d > > predict_dipole_at_frame(int i, const Atoms &atoms, const std::vector<std::vector< Eigen::Vector3d> > &test_bc, const std::vector<int> bond_index, int NUM_MOL, std::vector<std::vector<double> > UNITCELL_VECTORS, int NUM_MOL_ATOMS, std::string desctype, bool SAVE_DESCS, torch::jit::script::Module model_dipole, Eigen::Vector3d &TotalDipole, std::vector< Eigen::Vector3d > &MoleculeDipoleList ){
    /**
     * @fn
     * total_dipole, molecule_dipole, wannier_coordinateは
     * bond_listは固有のpropertyなのでそれを戻り値にする
     *  1: total dipole
     *  2: bond dipole自身
     *  3: molecule dipole
     *  4: wannier coordinate
     * @param[in] i : フレーム番号
     * @param[in] atoms : 計算するべきフレームのatoms
     * @param[in] test_bc : ボンドセンターの座標リスト[NUM_MOL, NUM_BOND_IN_MOL,3]型のリスト
     * @param[in] bond_index : ボンドインデックス
     * @param[out] TotalDipole : TotalDipoleにはそのまま加算してしまう．
    */
    
    // constant const2;
    auto descs_ch = raw_calc_bond_descripter_at_frame(atoms, test_bc, bond_index, NUM_MOL, UNITCELL_VECTORS,  NUM_MOL_ATOMS, desctype);
    auto list_bc_coords = get_coord_of_specific_bondcenter(test_bc, bond_index); // 特定ボンド(bond_indexで指定する）のBCの座標だけ取得

    // ! 出力用のリスト二つ
    std::vector<Eigen::Vector3d> tmp_dipole_list(descs_ch.size()); // bond dipoleの格納
    std::vector<Eigen::Vector3d> tmp_wannier_list(descs_ch.size()); // wannier coordinateの格納

    if ( SAVE_DESCS == true){ save_descriptor(descs_ch, "ch", i); }; //! 記述子の保存
    // ! descs_chの予測
    for (int j = 0, n=descs_ch.size(); j < n; j++) {        // loop over descs_ch
#ifdef DEBUG
        std::cout << "descs_ch size" << descs_ch[j].size() << std::endl;
        for (int k = 0; k<288;k++){
            std::cout << descs_ch[j][k] << " ";
        };
        std::cout << std::endl;
#endif //! DEBUG
        auto tmpDipole = predict_dipole(descs_ch[j], model_dipole); //! ボンドdipoleの計算
        // 以下後処理で，複数のpropertyを計算する．いずれも入力で渡しておいた方が良い．

        TotalDipole += tmpDipole; //! total dipole 
        // 双極子リスト (chボンドのリスト．これで全てのchボンドの値を出力できる．) 
        // TODO :: これに加えて，frameごとのchボンドの値も出力するといいかも．
        tmp_dipole_list[j] = tmpDipole;
        // auto output = elements[0].toTensor();
        //! 分子ごとに分けるには，test_read_mol.ch_bond_indexで割って現在の分子のindexを得れば良い．ADD THIS LINE
        int molecule_counter = j/bond_index.size(); // 0スタートでnum_molまで．
        int bondcenter_counter = j%bond_index.size(); // 0スタートでo_list.sizeまで．
        MoleculeDipoleList[molecule_counter]  += tmpDipole; 
        // ワニエの座標を計算(BC+dipole*coef)
        // Eigen::Vector3d tmp_wan_coord = list_bc_coords[molecule_counter][bondcenter_counter]+tmpDipole/(Ang*Charge/Debye)/(-2.0);
        Eigen::Vector3d tmp_wan_coord = list_bc_coords[j]+tmpDipole/(Ang*Charge/Debye)/(-2.0);

        tmp_wannier_list[j] = tmp_wan_coord ;
        std::cout << "tmp_wan_coord :: " << tmp_wan_coord[0] << tmp_wan_coord[1] << tmp_wan_coord[2] << std::endl;
    }
    return {tmp_dipole_list, tmp_wannier_list} ;
};