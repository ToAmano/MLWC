

// #define _DEBUG
#include <stdio.h>
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
#include <cctype> // https://b.0218.jp/20150625194056.html
#include <filesystem> // std::filesystem::exists (c++17)
#include <numeric> // std::iota
#include <tuple> // https://tyfkda.github.io/blog/2021/06/26/cpp-multi-value.html
#include <time.h>     // for clock() http://vivi.dyndns.org/tech/cpp/timeMeasurement.html
#include <chrono> // https://qiita.com/yukiB/items/01f8e276d906bf443356
#include <omp.h> // OpenMP https://qiita.com/nocturnality/items/cca512d1043f33a3da2c
// #include <boost/numeric/ublas/vector.hpp>
// #include <boost/numeric/ublas/matrix.hpp>
// #include <boost/numeric/ublas/io.hpp>
#include <Eigen/Core> // 行列演算など基本的な機能．
#include "numpy.hpp"
#include "npy.hpp"
#include "torch/script.h" // pytorch
// #include "numpy_quiita.hpp" // https://qiita.com/ka_na_ta_n/items/608c7df3128abbf39c89
// numpy_quiitaはsscanf_sが読み込めず，残念ながら現状使えない．
#include "atoms_asign_wcs.hpp"
#include "descriptor.hpp"
#include "parse.cpp"
#include "include/error.h"
#include "include/savevec.hpp"
#include "include/printvec.hpp"
#include "include/constant.hpp"
#include "predict.hpp"
#include "atoms_io.hpp"
#include "atoms_core.hpp"


// #include <GraphMol/GraphMol.h>
// #include <GraphMol/FileParsers/MolSupplier.h>
// #include <GraphMol/FileParsers/MolWriters.h>
// #include <GraphMol/FileParsers/FileParsers.h>


// https://e-penguiner.com/cpp-function-check-file-exist/#index_id2
bool IsFileExist(const std::string& name) {
    return std::filesystem::is_regular_file(name);
}

bool IsDirExist(const std::string& name){
    return std::filesystem::is_directory(name);
}

int main(int argc, char *argv[]) {
    std::cout << " +-----------------------------------------------------------------+" << std::endl;
    std::cout << " +                         Program dieltools                       +" << std::endl;
    std::cout << " +-----------------------------------------------------------------+" << std::endl;
    // 
    bool SAVE_DESCS = false; // trueならデスクリプターをnpyで保存．

    // constantクラスを利用する
    // constant const;

    clock_t start = clock();    // スタート時間
     std::chrono::system_clock::time_point  start_c, end_c; // 型は auto で可
     start_c = std::chrono::system_clock::now(); // 計測開始時間

    // std::string xyz_filename="/Users/amano/works/research/dieltools/notebook/c++/gromacs_trajectory_cell.xyz";
    // std::string xyz_filename="/Users/amano/works/research/dieltools/notebook/c++/gromacs_pg_1ns_dt50fs.xyz";
    // std::string xyz_filename="/Users/amano/works/research/dieltools/notebook/c++/gromacs_pg_1ns_dt50fs_300.xyz";

    // read argv and try to open input files.
    if (argc < 2) {
        exit("main", "Error: incorrect inputs. Usage:: dieltools inpfile");
    }

    std::cout << " ------------------------------------" << std::endl;
    std::cout << " 2: Reading Input Variables... ";
    std::string inp_filename=argv[1];
    if (!IsFileExist(inp_filename)) {
        exit("main", "Error: inp file does not exist.");
    }
    auto [inp_general, inp_desc, inp_pred] = locate_tag(inp_filename);
    std::cout << "FINISH reading inp file !! " << std::endl;
    auto var_gen = var_general(inp_general);
    auto var_des = var_descripter(inp_desc);
    auto var_pre = var_predict(inp_pred);
    std::cout << "FINISH parse inp file !! " << std::endl;
    //
    if (var_des.IF_COC){
        std::cout << "IF_COC is true" << std::endl;
    }

    int SAVE_TRUEY = var_pre.save_truey; 
    if (var_pre.save_truey){
        std::cout << "save_truey is true" << std::endl;
    }

    //! 保存するディレクトリの存在を確認
    if (!IsFileExist(var_gen.savedir)){
        std::cout << " ERROR :: savedir does not exist !! " << std::endl;
        return 1;
    }


    //! 原子数の取得(もしXがあれば除く)
    std::cout << " ------------------------------------" << std::endl;
    std::cout << " 3: Reading the xyz file  :: " << std::filesystem::absolute(var_des.xyzfilename) << std::endl;
    if (!IsFileExist(std::filesystem::absolute(var_des.xyzfilename))) {
        exit("main", "Error: xyzfile file does not exist.");
    }
    // int NUM_ATOM = raw_cpmd_num_atom(std::filesystem::absolute(var_des.xyzfilename)); //! IF_REMOVE_WANNIERなら後から更新
    int NUM_ATOM = get_num_atom_without_wannier(std::filesystem::absolute(var_des.xyzfilename)); //! WANを除いた原子数
    std::cout << std::setw(10) << "NUM_ATOM :: " << NUM_ATOM << std::endl;
    // std::cout << std::setw(10) << "NUM_ATOM_WITHOUT_WAN :: " << NUM_ATOM_WITHOUT_WAN << std::endl;
    //! 格子定数の取得
    std::vector<std::vector<double> > UNITCELL_VECTORS = raw_cpmd_get_unitcell_xyz(std::filesystem::absolute(var_des.xyzfilename));
    std::cout << std::setw(10) << "UNITCELL_VECTORS :: " << UNITCELL_VECTORS[0][0] << std::endl;
    //! xyzファイルから座標リストを取得
    bool IF_REMOVE_WANNIER = true;
    std::vector<Atoms> atoms_list = ase_io_read(std::filesystem::absolute(var_des.xyzfilename), IF_REMOVE_WANNIER);
    std::cout << " finish reading xyz file :: " << atoms_list.size() << std::endl;

    //! ボンドリストの取得
    // TODO :: 現状では，別に作成したボンドファイルを読み込んでいる．
    // TODO :: 本来はrdkitからボンドリストを取得するようにしたい．
    std::cout << "" << std::endl;
    std::cout << " ------------------------------------" << std::endl;
    std::cout << " 4: Reading the bond file  :: " << std::filesystem::absolute(var_gen.bondfilename) << std::endl;
    if (!IsFileExist(std::filesystem::absolute(var_gen.bondfilename))) {
        exit("main", "Error: bond file does not exist.");
    }
    read_mol test_read_mol(std::filesystem::absolute(var_gen.bondfilename));
    int NUM_MOL_ATOMS = test_read_mol.num_atoms_per_mol;
    std::cout << std::setw(10) << "NUM_MOL_ATOMS :: " << NUM_MOL_ATOMS << std::endl;
    std::cout << " finish reading bond file" << std::endl;

    std::cout << " calculate NUM_MOL..." << std::endl;
    int NUM_MOL = int(NUM_ATOM/NUM_MOL_ATOMS); // UnitCell中の総分子数
    std::cout << std::setw(10) << "NUM_MOL :: " << NUM_MOL << std::endl;
    std::cout << " OK !! " << std::endl;

    //! 以下はrdkitでできるかのテスト．そのうちやってみせる！
    //! test raw_aseatom_to_mol_coord_and_bc

    // RDKit::ROMol *mol1 = RDKit::SmilesToMol( "Cc1ccccc1" );
    // std::string mol_file = "../../../../smiles/pg.acpype/input_GMX.mol";
    // RDKit::ROMol *mol1 = RDKit::MolFileToMol(mol_file);
    // std::shared_ptr<RDKit::ROMol> mol2( RDKit::MolFileToMol(mol_file) );
    // std::cout << *mol2 << std::endl;

    //! torchの予測モデル読み込み
    std::cout << "" << std::endl;
    std::cout << " ------------------------------------" << std::endl;
    std::cout << " 5: START reading ML model file" << std::endl;
    // torch::jit::script::Module 型で module 変数の定義
    torch::jit::script::Module module_ch, module_cc, module_co, module_oh, module_o,module_coc,module_coh;
    // 各モデルを計算するかのフラグ
    bool IF_CALC_CH = false;
    bool IF_CALC_CC = false;
    bool IF_CALC_CO = false;
    bool IF_CALC_OH = false;
    bool IF_CALC_O = false;
    bool IF_CALC_COC = false;
    bool IF_CALC_COH = false;

    // 変換した学習済みモデルの読み込み
    // 実行パス（not 実行ファイルパス）からの絶対パスに変換 https://nompor.com/2019/02/16/post-5089/
    if (IsFileExist(std::filesystem::absolute(var_pre.model_dir+"/model_ch.pt"))) {
        IF_CALC_CH = true;
        module_ch = torch::jit::load(std::filesystem::absolute(var_pre.model_dir+"/model_ch.pt"));
        // module_ch = torch::jit::load(var_pre.model_dir+"/Users/amano/works/research/dieltools/notebook/c++/202306014_model_rotate/model_ch.pt");
    }
    if (IsFileExist(std::filesystem::absolute(var_pre.model_dir+"/model_cc.pt"))) {
        IF_CALC_CC = true;
        module_cc = torch::jit::load(std::filesystem::absolute(var_pre.model_dir+"/model_cc.pt"));
    }
    if (IsFileExist(std::filesystem::absolute(var_pre.model_dir+"/model_co.pt"))) {
        IF_CALC_CO = true;
        module_co = torch::jit::load(std::filesystem::absolute(var_pre.model_dir+"/model_co.pt"));
    }
    if (IsFileExist(std::filesystem::absolute(var_pre.model_dir+"/model_oh.pt"))) {
        IF_CALC_OH = true;
        module_oh = torch::jit::load(std::filesystem::absolute(var_pre.model_dir+"/model_oh.pt"));
    }
    if (IsFileExist(std::filesystem::absolute(var_pre.model_dir+"/model_o.pt"))) {
        IF_CALC_O = true;
        module_o = torch::jit::load(std::filesystem::absolute(var_pre.model_dir+"/model_o.pt"));
    }

    if (IsFileExist(std::filesystem::absolute(var_pre.model_dir+"/model_coc.pt"))) {
        IF_CALC_COC = true;
        module_coc = torch::jit::load(std::filesystem::absolute(var_pre.model_dir+"/model_coc.pt"));
    }
    if (IsFileExist(std::filesystem::absolute(var_pre.model_dir+"/model_coh.pt"))) {
        IF_CALC_COH = true;
        module_coh = torch::jit::load(std::filesystem::absolute(var_pre.model_dir+"/model_coh.pt"));
    }
    std::cout << " IF_CALC_CH :: " << IF_CALC_CH << std::endl;
    std::cout << " IF_CALC_CC :: " << IF_CALC_CC << std::endl;
    std::cout << " IF_CALC_CO :: " << IF_CALC_CO << std::endl;
    std::cout << " IF_CALC_OH :: " << IF_CALC_OH << std::endl;
    std::cout << " IF_CALC_O :: " << IF_CALC_O << std::endl;
    std::cout << " IF_CALC_COC :: " << IF_CALC_COC << std::endl;
    std::cout << " IF_CALC_COH :: " << IF_CALC_COH << std::endl;
    std::cout << " finish reading ML model file" << std::endl;



    std::cout << "" << std::endl;
    std::cout << " ------------------------------------" << std::endl;
    std::cout << " start calculate descriptor&prediction !!" << std::endl;
    std::cout << " " << std::endl;
    std::cout << "   OMP information :: " << std::endl;
    std::cout << "   NUM parallel  :: " << std::endl;
    std::cout << "   structure / parallel :: " << std::endl;

    // Beginning of parallel region
#ifdef _DEBUG
    // Beginning of parallel region
    std::cout << "OMP Parallerization test " << std::endl;
    #pragma omp parallel
    {
        printf("Hello World... from thread = %d\n", omp_get_thread_num());
    }
#endif //! _DEBUG
    
    // 予め出力するtotal dipole用のリストを確保しておく
    std::vector<Eigen::Vector3d> result_dipole_list(atoms_list.size());

    // 予め出力する分子の双極子用のリストを確保しておく
    std::vector<std::vector<Eigen::Vector3d> > result_molecule_dipole_list(atoms_list.size(), std::vector<Eigen::Vector3d>(NUM_MOL, Eigen::Vector3d::Zero()));

    // 予め出力するワニエセンターの座標用のリストを確保しておく．
    // !! 
    std::vector<Atoms> result_atoms_list(atoms_list.size());

    // 予めSAVE_TRUEYで保存するbond dipole用のリストを確保しておく
    std::vector<std::vector<Eigen::Vector3d> > result_ch_dipole_list(atoms_list.size());
    std::vector<std::vector<Eigen::Vector3d> > result_co_dipole_list(atoms_list.size());
    std::vector<std::vector<Eigen::Vector3d> > result_cc_dipole_list(atoms_list.size());
    std::vector<std::vector<Eigen::Vector3d> > result_oh_dipole_list(atoms_list.size());
    std::vector<std::vector<Eigen::Vector3d> > result_o_dipole_list(atoms_list.size());

    #pragma omp parallel for
    for (int i=0; i< atoms_list.size(); i++){
        // ! 予測値用の双極子
        Eigen::Vector3d TotalDipole = Eigen::Vector3d::Zero();
        Eigen::Vector3d tmpDipole   = Eigen::Vector3d::Zero();
        
        // ! 入力となるtensor用（形式は1,288の形！！）
        // TODO :: hard code :: 入力記述子の形はどうやってコントロールしようか？
        torch::Tensor input = torch::ones({1, 288}).to("cpu");
        // ! 分子ごとの双極子の予測値用のリスト ADD THIS LINE (0で初期化)
        std::vector<Eigen::Vector3d> MoleculeDipoleList(NUM_MOL, Eigen::Vector3d::Zero()); 
        // ! ボンドごとの予測値を保存するための双極子変数
        Eigen::Vector3d Dipole_tmp = Eigen::Vector3d::Zero();
        // ! true_yを保存するためのやつ．
        std::vector<Eigen::Vector3d> true_y_list_coc;
        std::vector<Eigen::Vector3d> true_y_list_coh;
        // ! ワニエの座標保存用
        Eigen::Vector3d tmp_wan_coord;

        // pbc-molをかけた原子座標(test_mol)と，それを利用したbcを取得
        auto test_mol_bc = raw_aseatom_to_mol_coord_and_bc(atoms_list[i], test_read_mol.bonds_list, test_read_mol, NUM_MOL_ATOMS, NUM_MOL);
        std::vector<std::vector<Eigen::Vector3d> > test_mol=std::get<0>(test_mol_bc);
        std::vector<std::vector<Eigen::Vector3d> > test_bc =std::get<1>(test_mol_bc);

        // 各ボンドはここでここで定義しておこう．（第一引数は系全体の各種ボンドの数を表す）
        dipole_frame ch_dipole_frame   = dipole_frame(NUM_MOL*test_read_mol.ch_bond_index.size(), NUM_MOL);
        dipole_frame cc_dipole_frame   = dipole_frame(NUM_MOL*test_read_mol.cc_bond_index.size(), NUM_MOL);
        dipole_frame co_dipole_frame   = dipole_frame(NUM_MOL*test_read_mol.co_bond_index.size(), NUM_MOL);
        dipole_frame oh_dipole_frame   = dipole_frame(NUM_MOL*test_read_mol.oh_bond_index.size(), NUM_MOL);
        dipole_frame o_dipole_frame    = dipole_frame(NUM_MOL*test_read_mol.o_list.size(), NUM_MOL);
        dipole_frame coh_dipole_frame  = dipole_frame(NUM_MOL*test_read_mol.coh_list.size(), NUM_MOL); // coh/coc用
        dipole_frame coc_dipole_frame  = dipole_frame(NUM_MOL*test_read_mol.coc_list.size(), NUM_MOL); // coh/coc用


        //! chボンド双極子の作成
        if (IF_CALC_CH){
            // ! 以上の1frameの双極子予測計算をクラス化した．
            ch_dipole_frame.predict_bond_dipole_at_frame(atoms_list[i], test_bc, test_read_mol.ch_bond_index, NUM_MOL, UNITCELL_VECTORS,  NUM_MOL_ATOMS, var_des.desctype, module_ch);
            ch_dipole_frame.calculate_wannier_list(test_bc, test_read_mol.ch_bond_index);
            ch_dipole_frame.calculate_moldipole_list();
            // ! ch_dipole_listへの代入
            result_ch_dipole_list[i] = ch_dipole_frame.dipole_list;
            // * total dipoleに各ボンド双極子を足す
            for (int p = 0; p<ch_dipole_frame.dipole_list.size(); p++){
                TotalDipole += ch_dipole_frame.dipole_list[p];
            }
            // * 分子ごとの双極子にボンドの寄与を足す．
            for (int p=0; p<NUM_MOL; p++){
                MoleculeDipoleList[p] += ch_dipole_frame.MoleculeDipoleList[p];
            };
        } //! end if IF_CALC_CH

        //! ccボンド双極子の作成
        //!! 注意：：ccボンドの場合，最近説のC原子への距離が二つのC原子で同じなので，ここの並びが変わることがあり得る．
        if (IF_CALC_CC){
            // ! 以上の1frameの双極子予測計算をクラス化した．
            cc_dipole_frame.predict_bond_dipole_at_frame(atoms_list[i], test_bc, test_read_mol.cc_bond_index, NUM_MOL, UNITCELL_VECTORS,  NUM_MOL_ATOMS, var_des.desctype, module_cc);
            cc_dipole_frame.calculate_wannier_list(test_bc, test_read_mol.cc_bond_index);
            cc_dipole_frame.calculate_moldipole_list();
            // ! cc_dipole_listへの代入
            result_cc_dipole_list[i] = cc_dipole_frame.dipole_list;
            // * total dipoleに各ボンド双極子を足す
            for (int p = 0; p<cc_dipole_frame.dipole_list.size(); p++){
                TotalDipole += cc_dipole_frame.dipole_list[p];
            };
            // * 分子ごとの双極子にボンドの寄与を足す．
            for (int p=0; p<NUM_MOL; p++){
                MoleculeDipoleList[p] += cc_dipole_frame.MoleculeDipoleList[p];
            };
        } //! END_IF IF_CALC_CC

        //! test raw_calc_bond_descripter_at_frame (coのボンドのテスト)
        if (IF_CALC_CO){
            // ! 以上の1frameの双極子予測計算をクラス化した．
            co_dipole_frame.predict_bond_dipole_at_frame(atoms_list[i], test_bc, test_read_mol.co_bond_index, NUM_MOL, UNITCELL_VECTORS,  NUM_MOL_ATOMS, var_des.desctype, module_co);
            co_dipole_frame.calculate_wannier_list(test_bc, test_read_mol.co_bond_index);
            co_dipole_frame.calculate_moldipole_list();
            // ! co_dipole_listへの代入
            result_co_dipole_list[i] = co_dipole_frame.dipole_list;
            // * total dipoleに各ボンド双極子を足す
            for (int p = 0; p<co_dipole_frame.dipole_list.size(); p++){
                TotalDipole += co_dipole_frame.dipole_list[p];
            };
            // * 分子ごとの双極子にボンドの寄与を足す．
            for (int p=0; p<NUM_MOL; p++){
                MoleculeDipoleList[p] += co_dipole_frame.MoleculeDipoleList[p];
            };
        }; //! END_IF IF_CALC_CO

        //! test raw_calc_bond_descripter_at_frame (ohのボンドのテスト)
        if (IF_CALC_OH){
            // ! 以上の1frameの双極子予測計算をクラス化した．
            oh_dipole_frame.predict_bond_dipole_at_frame(atoms_list[i], test_bc, test_read_mol.oh_bond_index, NUM_MOL, UNITCELL_VECTORS,  NUM_MOL_ATOMS, var_des.desctype, module_oh);
            oh_dipole_frame.calculate_wannier_list(test_bc, test_read_mol.oh_bond_index);
            oh_dipole_frame.calculate_moldipole_list();
            // ! oh_dipole_listへの代入
            result_oh_dipole_list[i] = oh_dipole_frame.dipole_list;
            // * total dipoleに各ボンド双極子を足す
            for (int p = 0; p<oh_dipole_frame.dipole_list.size(); p++){
                TotalDipole += oh_dipole_frame.dipole_list[p];
            };
            // * 分子ごとの双極子にボンドの寄与を足す．
            for (int p=0; p<NUM_MOL; p++){
                MoleculeDipoleList[p] += oh_dipole_frame.MoleculeDipoleList[p];
            };
        }; //! END_IF IF_CALC_OH

        //! test raw_calc_lonepair_descripter_at_frame （ローンペアのテスト）
        if (IF_CALC_O){
            // ! 以上の1frameの双極子予測計算をクラス化した．
            o_dipole_frame.predict_lonepair_dipole_at_frame(atoms_list[i], test_mol, test_read_mol.o_list, NUM_MOL, UNITCELL_VECTORS, NUM_MOL_ATOMS, var_des.desctype, module_o);
            o_dipole_frame.calculate_lonepair_wannier_list(test_mol, test_read_mol.o_list); //test_molを指定しないとちゃんと動かないので注意！！
            o_dipole_frame.calculate_moldipole_list();
            // ! o_dipole_listへの代入
            result_o_dipole_list[i] = o_dipole_frame.dipole_list;
            // * total dipoleに各ボンド双極子を足す
            for (int p = 0; p<o_dipole_frame.dipole_list.size(); p++){
                TotalDipole += o_dipole_frame.dipole_list[p];
            };
            // * 分子ごとの双極子にボンドの寄与を足す．
            for (int p=0; p<NUM_MOL; p++){
                MoleculeDipoleList[p] += o_dipole_frame.MoleculeDipoleList[p];
            };
            // !
            // for (int p=0;p<tmp_o_dipole_list.size();p++){
            //     std::cout << (tmp_o_dipole_list[p]-o_dipole_frame.dipole_list[p]).norm() << std::endl;
            //     if ((tmp_o_dipole_list[p]-o_dipole_frame.dipole_list[p]).norm()>0.0001){
            //         std::cout << "WARNING :: tmp_o_dipole " << std::endl;
            //     };
            // }
        } //! END_IF IF_CALC_O

        // ! >>>>>>>>>>>>>>>
        // ! 1フレームの計算の終了
        // ! >>>>>>>>>>>>>>>
        if (omp_get_thread_num() == 1){ // スレッド1番でのみ出力
            std::cout << "TotalDipole :: " << i << " " << TotalDipole[0] << " "  << TotalDipole[1] << " "  << TotalDipole[2] << " " << std::endl;
        }
        // frameごとのtotal dipoleに代入
        result_dipole_list[i]=TotalDipole;

        // 計算された分子ごとの双極子をリストへ格納
        for (int j=0; j<NUM_MOL; j++){
            result_molecule_dipole_list[i][j]=MoleculeDipoleList[j];
        }

        // 計算されたbond centerとwannier centersをase atomsへ格納する．
        // 分子ごとにpushbackするので，ここでまとめて実行する必要がある．
        std::vector < Eigen::Vector3d > atoms_with_bc; // これを使う
        std::vector < int >             new_atomic_num; // これを使う
        std::vector < int >  atomic_numbers = atoms_list[i].get_atomic_numbers();
        for (int a=0; a< NUM_MOL; a++){ // 
            for (int b=0; b<test_mol[a].size();b++){ //原子座標
                atoms_with_bc.push_back(test_mol[a][b]);
                new_atomic_num.push_back(atomic_numbers[a*NUM_MOL_ATOMS+b]); //原子に対応するatoms_listの原子種
                // atoms_with_bc_index.push_back(test_mol[a][b])
            }
            for (int b=0; b<test_bc[a].size();b++){ //ボンドセンター
                atoms_with_bc.push_back(test_bc[a][b]);
                new_atomic_num.push_back(2); // ボンドセンターには原子番号2を割り当て

            }
            if (IF_CALC_CH){
                for (int b=0; b<ch_dipole_frame.wannier_list[a].size();b++){ //ch wannier
                    atoms_with_bc.push_back(ch_dipole_frame.wannier_list[a][b]);
                    new_atomic_num.push_back(100);
                }
            }
            if (IF_CALC_CC){
                for (int b=0; b<cc_dipole_frame.wannier_list[a].size();b++){ //ch wannier
                    atoms_with_bc.push_back(cc_dipole_frame.wannier_list[a][b]);
                    new_atomic_num.push_back(100);
                }
            }
            if (IF_CALC_CO){
                for (int b=0; b<co_dipole_frame.wannier_list[a].size();b++){ //ch wannier
                    atoms_with_bc.push_back(co_dipole_frame.wannier_list[a][b]);
                    new_atomic_num.push_back(100);
                }
            }
            if (IF_CALC_OH){
                for (int b=0; b<oh_dipole_frame.wannier_list[a].size();b++){ //ch wannier
                    atoms_with_bc.push_back(oh_dipole_frame.wannier_list[a][b]);
                    new_atomic_num.push_back(100);
                }
            }
            if (IF_CALC_O){
                for (int b=0; b<o_dipole_frame.wannier_list[a].size();b++){ //ch wannier
                    atoms_with_bc.push_back(o_dipole_frame.wannier_list[a][b]);
                    new_atomic_num.push_back(10);
                }
            }
        }
        Atoms tmp_atoms = Atoms(
            new_atomic_num,
            atoms_with_bc,
            UNITCELL_VECTORS,
            {true,true,true});
        // Atoms testtest = Atoms(atoms_list[i].get_atomic_numbers(), atoms_list[i].get_positions(), UNITCELL_VECTORS, {1,1,1});
        result_atoms_list[i] = tmp_atoms;
        new_atomic_num.clear(); // vectorのクリア
	    atoms_with_bc.clear();
     }
    std::cout << " finish calculate descriptor&prediction !!" << std::endl;
    std::cout << " now saving data..." << std::endl;

    // ! >>>>>>>>>>>>>>>>
    // ! 計算終了，最後のファイル保存
    // ! >>>>>>>>>>>>>>>>

    // 最後にtotal双極子をファイルに保存
    save_vec(result_dipole_list, var_gen.savedir+"total_dipole.txt", "# index dipole_x dipole_y dipole_z");
    // std::ofstream fout(var_des.savedir+"total_dipole.txt"); 
    // fout << "# index dipole_x dipole_y dipole_z" << std::endl;
    // for (int i = 0; i < result_dipole_list.size(); i++){
    //     fout << std::setw(5) << i << std::right << std::setw(16) << result_dipole_list[i][0] << std::setw(16) << result_dipole_list[i][1] << std::setw(16) << result_dipole_list[i][2] << std::endl;
    // }
    // fout.close();

    // save files1: bond dipoleをファイルに保存
    // TODO :: （3D配列なのでもっと良い方法を考えないといけない） 
    save_vec_index(result_ch_dipole_list,var_gen.savedir+"ch_dipole.txt", "# frame_index ch_index dipole_x dipole_y dipole_z" );
    save_vec_index(result_co_dipole_list,var_gen.savedir+"co_dipole.txt", "# frame_index co_index dipole_x dipole_y dipole_z" );
    save_vec_index(result_oh_dipole_list,var_gen.savedir+"oh_dipole.txt", "# frame_index oh_index dipole_x dipole_y dipole_z" );
    save_vec_index(result_cc_dipole_list,var_gen.savedir+"cc_dipole.txt", "# frame_index cc_index dipole_x dipole_y dipole_z" );
    save_vec_index(result_o_dipole_list ,var_gen.savedir+ "o_dipole.txt", "# frame_index o_index  dipole_x dipole_y dipole_z" );
    // 分子双極子の保存：本来3次元配列だが，frame,mol_id,d_x,d_y,d_zの形で保存することで二次元配列として保存する．
    save_vec_index(result_molecule_dipole_list, var_gen.savedir+"molecule_dipole.txt", "# frame_index mol_index dipole_x dipole_y dipole_z");

    // 最終的な結果をxyzに保存する．
    ase_io_write(result_atoms_list, var_gen.savedir+"mol_wan.xyz");

    // 時間計測関係
    clock_t end = clock();     // 終了時間
    end_c = std::chrono::system_clock::now();  // 計測終了時間
    double elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_c-start_c).count();
    std::cout << "duration (clock) = " << (double)(end - start) / CLOCKS_PER_SEC << "sec.\n";
    std::cout << "duration (chrono) = " << elapsed << "sec.\n";
    std::cout << "finish !! " << std::endl;

    return 0;
}
