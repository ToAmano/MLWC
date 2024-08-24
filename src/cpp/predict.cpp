/**
 * @file ファイル名.h
 * @brief 簡単な説明
 * @author 書いた人
 * @date 日付（開始日？）
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
#include "descriptor.hpp" 

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
        for (int j = 0, m=descs[i].size(); j < m; j++) { //これが288個のはず
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
     * TODO :: 記述子の長さは自動的に取得される用になっていて，長さのマッチングは行われない．一応エラー処理をしたほうが良いだろう．
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

    // if dipole is too large (10D), print warning
    if (tmpDipole.norm()>10.0){
        std::cout << "WARNING :: tmpDipole is too large :: " << tmpDipole.norm() << std::endl;
    };
    return tmpDipole;
}


std::tuple< std::vector< Eigen::Vector3d >, std::vector< Eigen::Vector3d > > predict_dipole_at_frame(int i, const Atoms &atoms, const std::vector<std::vector< Eigen::Vector3d> > &test_bc, const std::vector<int> bond_index, int NUM_MOL, std::vector<std::vector<double> > UNITCELL_VECTORS, int NUM_MOL_ATOMS, std::string desctype, bool SAVE_DESCS, torch::jit::script::Module model_dipole, Eigen::Vector3d &TotalDipole, std::vector< Eigen::Vector3d > &MoleculeDipoleList ){
    /**
     * @fn DEPRECATED !!
     * total_dipole, molecule_dipoleは他のボンドの計算も行うので，入力として受け取って加算する？
     * bond_list,wannier_coordinateは固有のpropertyなのでそれを戻り値にする
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
        // std::cout << "tmp_wan_coord :: " << tmp_wan_coord[0] << tmp_wan_coord[1] << tmp_wan_coord[2] << std::endl;
    }
    return {tmp_dipole_list, tmp_wannier_list} ;
};

// Functions for dipole_frame class

/**
 * @brief Construct a new dipole frame::dipole frame object
 * @param descs_size 
 * @param num_molecule 
 */

dipole_frame::dipole_frame(int descs_size, int num_molecule){ // constructor
    if (descs_size%num_molecule != 0 ){ // descs size must be devided by num_molecule
        std::cout << " ERROR :: descs_size%num_molecule != 0 " << descs_size << " "<< num_molecule<< std::endl;
    }
    this->calc_wannier = false;
    this->descs_size = descs_size;
    this->num_molecule = num_molecule;
    this->num_bond     = this->descs_size/this->num_molecule; // 記述子のサイズ/分子数=分子あたりの記述子の数
    this->MoleculeDipoleList.resize(num_molecule,Eigen::Vector3d::Zero() ); // 0で初期化, 超大事
    this->dipole_list.resize(descs_size, Eigen::Vector3d::Zero()); // サイズの初期化．bond dipoleの格納
    this->wannier_list.resize(num_molecule, std::vector<Eigen::Vector3d> (this->num_bond)); // wannier coordinateの格納（NUM_MOL, NUM_BONDの形）
    // std::vector< Eigen::Vector3d > MoleculeDipoleList(num_molecule); 
    // std::vector<Eigen::Vector3d> dipole_list(descs_size); // bond dipoleの格納
    // std::vector<Eigen::Vector3d> wannier_list(descs_size); // wannier coordinateの格納
}

void dipole_frame::predict_bond_dipole_at_frame(const Atoms &atoms, const std::vector<std::vector< Eigen::Vector3d> > &test_bc, const std::vector<int> bond_index, int NUM_MOL, std::vector<std::vector<double> > UNITCELL_VECTORS, int NUM_MOL_ATOMS, std::string desctype, torch::jit::script::Module model_dipole ){
    /**
     * @fn
     * total_dipole, molecule_dipoleは他のボンドの計算も行うので，入力として受け取って加算する？
     * bond_list,wannier_coordinateは固有のpropertyなのでそれを戻り値にする
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
    
    // get Rcs, Rc, MaxAt from model_dipole
    float Rcs   = model_dipole.attr("Rcs").toDouble();
    float Rc    = model_dipole.attr("Rc").toDouble();
    int   MaxAt = int(model_dipole.attr("nfeatures").toInt()/4/3);

    // Calculate descriptor
    auto descs_ch = raw_calc_bond_descripter_at_frame(atoms, test_bc, bond_index, NUM_MOL, UNITCELL_VECTORS, NUM_MOL_ATOMS, desctype, Rcs, Rc, MaxAt);

    if (!(int(descs_ch.size()) == this->descs_size)){
        std::cout << "predict_bond_dipole_at_frame :: predict_dipole_at_frame :: The size of descs is wrong. " << descs_ch.size() << " " << this->descs_size << std::endl;
        std::cout << "predict_bond_dipole_at_frame :: predict_dipole_at_frame :: The size of descs is wrong. " << descs_ch[0][0] << std::endl;
        return;
    }

    // ! predict descs_ch
    #pragma omp for
    for (int j = 0; j < int(descs_ch.size()); j++) {        // loop over descs_ch
#ifdef DEBUG
        std::cout << "descs_ch size" << descs_ch[j].size() << std::endl;
        for (int k = 0; k<288;k++){
            std::cout << descs_ch[j][k] << " ";
        };
        std::cout << std::endl;
#endif //! DEBUG
        // 以下後処理で，複数のpropertyを計算する．いずれも入力で渡しておいた方が良い．
        // 双極子リスト (chボンドのリスト．これで全てのchボンドの値を出力できる．) 
        // TODO :: これに加えて，frameごとのchボンドの値も出力するといいかも．
        this->dipole_list[j] = predict_dipole(descs_ch[j], model_dipole); //! ボンドdipoleの計算
    };
    this->calc_wannier = true; // 計算終了フラグを真にする
};

void dipole_frame::predict_lonepair_dipole_at_frame(const Atoms &atoms, const std::vector<std::vector< Eigen::Vector3d> > &test_mol, const std::vector<int> atom_index, int NUM_MOL, std::vector<std::vector<double> > UNITCELL_VECTORS, int NUM_MOL_ATOMS, std::string desctype, torch::jit::script::Module model_dipole ){
    /**
     * @brief 原子番号8番に対して記述子を作成し，model_dipoleによって予測値を求める．
     * @param[in] atoms : 計算するべきフレームのatoms
     * @param[in] test_mol : 分子の座標リスト
     * @param[in] atom_index : 原子番号8番の原子のindex
     * 
     */
    // get Rcs, Rc, MaxAt from model_dipole
    float Rcs   = model_dipole.attr("Rcs").toDouble();
    float Rc    = model_dipole.attr("Rc").toDouble();
    int   MaxAt = int(model_dipole.attr("nfeatures").toInt()/4/3);
    auto descs_o = raw_calc_lonepair_descripter_at_frame(
        atoms, test_mol, atom_index, NUM_MOL, 8, 
        UNITCELL_VECTORS,  NUM_MOL_ATOMS, desctype, Rcs, Rc, MaxAt); // parallel calculation
    // ! prefict dipole_o
    #pragma omp for
    for (int j = 0; j < int(descs_o.size()); j++) { // loop over descs_o
        this->dipole_list[j] = predict_dipole(descs_o[j], model_dipole);
    };
    this->calc_wannier = true; // 計算終了フラグを真にする
}

void dipole_frame::predict_lonepair_dipole_select_at_frame(const Atoms &atoms, const std::vector<std::vector< Eigen::Vector3d> > &test_mol, const std::vector<int> atom_index, int NUM_MOL, std::vector<std::vector<double> > UNITCELL_VECTORS, int NUM_MOL_ATOMS, std::string desctype, torch::jit::script::Module model_dipole ){
    /**
     * @brief atom_indexで指定した原子に対して記述子を作成し，model_dipoleによって予測値を求める．
     * 
     * 
     * history
     * ==================
     * 12/18 :: fix bug in descripter calculation
     */
    auto descs_o = raw_calc_lonepair_descripter_select_at_frame(atoms, test_mol, atom_index, NUM_MOL, UNITCELL_VECTORS,  NUM_MOL_ATOMS, desctype);
    // std::cout << "DEBUG :: descs_o.size() :: " << descs_o.size() << std::endl;
    // std::cout << "DEBUG :: descs_o[1].size() :: " << descs_o[1].size() << std::endl;
    // ! prediction using descs_o
    for (int j = 0, n = descs_o.size(); j < n; j++) { // loop over descs_o
        auto tmpDipole = predict_dipole(descs_o[j], model_dipole); //! calc bond dipole
        this->dipole_list[j] = tmpDipole; 
        // std::cout << "DEBUG :: tmp_COC/COH_Dipole :: " << tmpDipole[0] << " " << tmpDipole[1] << " " << tmpDipole[2] << std::endl;
    };
    this->calc_wannier = true; // flag for calculation is true
}

void dipole_frame::calculate_wannier_list(std::vector<std::vector< Eigen::Vector3d> > &test_bc, const std::vector<int> bond_index){
    if (!(this->calc_wannier)){
        std::cout << "ERROR :: calculate_wannier_list :: you have to first calculate wannier coordinate." << std::endl;
        return;
    }
    // 特定ボンド(bond_indexで指定する）のBCの座標だけ取得 (ワニエの座標計算用)
    auto list_bc_coords = get_coord_of_specific_bondcenter(test_bc, bond_index); 
    int bond_index_size = int(this->descs_size/this->num_molecule); // bond_index.size()
    // ! descs_chの予測
    for (int j = 0; j < this->descs_size; j++) {        // loop over descs_ch
        // ワニエの座標を計算(BC+dipole*coef)
        // Eigen::Vector3d tmp_wan_coord = list_bc_coords[molecule_counter][bondcenter_counter]+tmpDipole/(Ang*Charge/Debye)/(-2.0);
        // TODO :: 現状single bondのみに対応している．
        Eigen::Vector3d tmp_wan_coord = list_bc_coords[j]+this->dipole_list[j]/(Ang*Charge/Debye)/(-2.0);
        int molecule_counter = j/bond_index_size; // 0スタートでnum_molまで．
        int bondcenter_counter = j%bond_index_size; // 0スタートでo_list.sizeまで．
        this->wannier_list[molecule_counter][bondcenter_counter] = tmp_wan_coord ;
        // std::cout << "tmp_wan_coord :: " << tmp_wan_coord[0] << tmp_wan_coord[1] << tmp_wan_coord[2] << std::endl;
    };
};


void dipole_frame::calculate_lonepair_wannier_list(std::vector<std::vector< Eigen::Vector3d> > &test_mol, const std::vector<int> atom_index){
    // dipole_listを計算したあと，それを実際のWCsの座標に変換する．
    if (!(this->calc_wannier)){
        std::cout << "ERROR :: calculate_wannier_list :: you have to first calculate wannier coordinate." << std::endl;
        return;
    }
    // 特定ボンド(bond_indexで指定する）のBCの座標だけ取得 (ワニエの座標計算用)
    // TODO :: ここで特定原子の座標を取得する．
    // find_specific_lonepair(test_mol,const Atoms &aseatoms, 8, this->num_molecule);
    auto list_lonepair_coords = get_coord_of_specific_lonepair(test_mol, atom_index); //! O原子の座標の計算
    int bond_index_size = int(this->descs_size/this->num_molecule); // bond_index.size() //! 記述子の数/分子数=分子あたりの記述子の数
    // ! descs_chの予測
    for (int j = 0; j < this->descs_size; j++) {        // loop over descs_ch
        // ワニエの座標を計算(O+dipole*coef)
        // Eigen::Vector3d tmp_wan_coord = list_bc_coords[molecule_counter][bondcenter_counter]+tmpDipole/(Ang*Charge/Debye)/(-2.0);
        // TODO :: 現状Oのローンペア（4でわる）のみに対応している．
        Eigen::Vector3d tmp_wan_coord = list_lonepair_coords[j]+this->dipole_list[j]/(Ang*Charge/Debye)/(-4.0);
        int molecule_counter = j/bond_index_size; // 0スタートでnum_molまで．
        int bondcenter_counter = j%bond_index_size; // 0スタートでo_list.sizeまで．
        this->wannier_list[molecule_counter][bondcenter_counter] = tmp_wan_coord ;
        // std::cout << "tmp_wan_coord :: " << tmp_wan_coord[0] << tmp_wan_coord[1] << tmp_wan_coord[2] << std::endl;
    };
};

void dipole_frame::calculate_moldipole_list(){
    /**
     * @fn dipole_listを利用して分子dipoleを計算する．
    */
    // TODO :: やはり，dipole_listの形状を1次元ではなく2次元[分子id,ボンドid]にした方が全体が綺麗になる気がする．
    if (!(this->calc_wannier)){
        std::cout << "calculate_wannier_list :: wannier coordinateを計算していないため，計算できません．" << std::endl;
        return;
    }
    int bond_index_size = int(this->descs_size/this->num_molecule); // bond_index.size()
    // ! calculate molecular dipole
    for (int j = 0; j < this->descs_size; j++) {        // loop over descs_ch
        // auto output = elements[0].toTensor();
        //! 分子ごとに分けるには，test_read_mol.ch_bond_indexで割って現在の分子のindexを得れば良い．ADD THIS LINE
        //! 現在のdescs(j)がどの分子に属するかを判定する．
        int molecule_counter = j/bond_index_size; // 0スタートでnum_molまで．
        int bondcenter_counter = j%bond_index_size; // 0スタートでo_list.sizeまで．
        // int test = j/this->num_molecule;
        this->MoleculeDipoleList[molecule_counter]  += this->dipole_list[j]; 
    }
}



void dipole_frame::save_descriptor_frame(int i, const Atoms &atoms, const std::vector<std::vector< Eigen::Vector3d> > &test_bc, const std::vector<int> bond_index, int NUM_MOL, std::vector<std::vector<double> > UNITCELL_VECTORS, int NUM_MOL_ATOMS, std::string desctype, bool SAVE_DESCS, torch::jit::script::Module model_dipole){
    // TODO :: model_dipole変数は不要．
    // 記述子計算;
    auto descs_ch = raw_calc_bond_descripter_at_frame(atoms, test_bc, bond_index, NUM_MOL, UNITCELL_VECTORS,  NUM_MOL_ATOMS, desctype);

    if (!(int(descs_ch.size()) == this->descs_size)){
        std::cout << "predict_dipole_at_frame :: descs_chのサイズが一致しません．" << std::endl;
        return;
    }
    save_descriptor(descs_ch, "ch", i); //! 記述子の保存
}

void dipole_frame::calculate_coh_bond_dipole_at_frame(std::map<int, std::pair<int, int> > coh_bond_info, const std::vector< Eigen::Vector3d > o_dipole_list, const std::vector< Eigen::Vector3d > bond1_dipole_list, const std::vector< Eigen::Vector3d > bond2_dipole_list){
    /**
     * @brief CO，OH，Oのボンド双極子から，COHのボンド双極子を計算する．
     * @fn 情報としては，まずセンターのO原子のリストが必要．
    */
    // TODO :: coh_bond_infoが残りのdipole_listたちと一致しているかのチェック（間違えてcoh_bond_infoに対してcocのデータを代入していないか）が絶対にあった方が良い．
    // dipole_listの大きさとcoc_bond_info*num_moleculeの大きさが等しくないといけない．
    if (dipole_list.size() != coh_bond_info.size()*num_molecule){
        std::cout << "ERROR :: calculate_coh_bond_dipole_at_frame :: size is inconsistent" << std::endl;
    };
    // TODO :: 各dipole_listの大きさの取得．もしかすると本来dipole_listは1次元配列ではなく二次元配列の方が良いかも．．．その場合はかなり大規模にこのpredict.cppを書き直す必要がある．
    int o_dipole_size = o_dipole_list.size()/num_molecule;
    int bond1_dipole_size = bond1_dipole_list.size()/num_molecule;
    int bond2_dipole_size = bond2_dipole_list.size()/num_molecule;

    // ! descs_oの予測
    int counter=0;
    // int o_index,bond1_index,bond2_index;
    for (int mol_id = 0; mol_id< num_molecule; mol_id++) { // 分子ごとのループ
        for (const auto& [key, value] : coh_bond_info){  //分子内のCOC/COH結合に対するループå
            // std::cout << key << " => " << value << "\n";
            // oの要素 o_dipole_list[coc_bond_info[j]] 
            // 1つ目のbondの要素bond1_dipole_list[]
            // 2つ目のbondの要素bond2_dipole_list[]
            // o_index = raw_convert_bondindex(o_list, key); // bondindex[i]から，ch_bond_index[j]を満たすjを返す．（要は変換）
            // bond1_index = raw_convert_bondindex(o_list, std::get<0>(value)); 
            // bond2_index = raw_convert_bondindex(o_list, std::get<1>(value)); 
            auto tmpDipole = o_dipole_list[o_dipole_size*mol_id+key]+bond1_dipole_list[bond1_dipole_size*mol_id+std::get<0>(value)]+bond2_dipole_list[bond2_dipole_size*mol_id+std::get<1>(value)]; //! COC/COHボンド双極子の計算
            this->dipole_list[counter] = tmpDipole; // counter番目のボンド双極子を代入
            counter += 1;
        };
    };
    // TODO :: ここはtrueにすると危ない説もある． 
    this->calc_wannier = true; // 計算終了フラグを真にする
}