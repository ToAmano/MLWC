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
// #include "atoms_core.cpp" // !! これを入れるとエラーが出る？
// #include "atoms_io.cpp"   // !! これを入れるとエラーが出る？
// #include "mol_core.cpp"
#include "atoms_asign_wcs.cpp"
#include "descriptor.cpp"
#include "parse.cpp"
#include "include/error.h"

// #include <GraphMol/GraphMol.h>
// #include <GraphMol/FileParsers/MolSupplier.h>
// #include <GraphMol/FileParsers/MolWriters.h>
// #include <GraphMol/FileParsers/FileParsers.h>


// https://e-penguiner.com/cpp-function-check-file-exist/#index_id2
bool IsFileExist(const std::string& name) {
    return std::filesystem::is_regular_file(name);
}


int main(int argc, char *argv[]) {
    std::cout << " +-----------------------------------------------------------------+" << std::endl;
    std::cout << " +                         Program dieltools                       +" << std::endl;
    std::cout << " +-----------------------------------------------------------------+" << std::endl;
    // 
    bool SAVE_DESCS = false; // trueならデスクリプターをnpyで保存．

    // 双極子の出力ファイル
    std::ofstream fout("total_dipole.txt"); 

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

    //! 原子数の取得(もしXがあれば除く)
    std::cout << " ------------------------------------" << std::endl;
    std::cout << " 3: Reading the xyz file  :: " << std::filesystem::absolute(var_des.xyzfilename) << std::endl;
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
    // 変換した学習済みモデルの読み込み
    // 実行パス（not 実行ファイルパス）からの絶対パスに変換 https://nompor.com/2019/02/16/post-5089/
    if (IsFileExist(std::filesystem::absolute(var_pre.model_dir+"/model_ch.pt"))) {
        module_ch = torch::jit::load(std::filesystem::absolute(var_pre.model_dir+"/model_ch.pt"));
        // module_ch = torch::jit::load(var_pre.model_dir+"/Users/amano/works/research/dieltools/notebook/c++/202306014_model_rotate/model_ch.pt");
    }
    if (IsFileExist(std::filesystem::absolute(var_pre.model_dir+"/model_cc.pt"))) {
        module_cc = torch::jit::load(std::filesystem::absolute(var_pre.model_dir+"/model_cc.pt"));
    }
    if (IsFileExist(std::filesystem::absolute(var_pre.model_dir+"/model_co.pt"))) {
        module_co = torch::jit::load(std::filesystem::absolute(var_pre.model_dir+"/model_co.pt"));
    }
    if (IsFileExist(std::filesystem::absolute(var_pre.model_dir+"/model_oh.pt"))) {
        module_oh = torch::jit::load(std::filesystem::absolute(var_pre.model_dir+"/model_oh.pt"));
    }
    if (IsFileExist(std::filesystem::absolute(var_pre.model_dir+"/model_o.pt"))) {
        module_o = torch::jit::load(std::filesystem::absolute(var_pre.model_dir+"/model_o.pt"));
    }

    if (IsFileExist(std::filesystem::absolute(var_pre.model_dir+"/model_coc.pt"))) {
        module_coc = torch::jit::load(std::filesystem::absolute(var_pre.model_dir+"/model_coc.pt"));
    }
    if (IsFileExist(std::filesystem::absolute(var_pre.model_dir+"/model_coh.pt"))) {
        module_coh = torch::jit::load(std::filesystem::absolute(var_pre.model_dir+"/model_coh.pt"));
    }
    std::cout << " finish reading ML model file" << std::endl;


    // Beginning of parallel region
#ifdef _DEBUG
    // Beginning of parallel region
    std::cout << "OMP Parallerization test " << std::endl;
    #pragma omp parallel
    {
        printf("Hello World... from thread = %d\n", omp_get_thread_num());
    }
#endif //! _DEBUG


    std::cout << " start calculate descriptor&prediction !!" << std::endl;
    // 予め出力する双極子用のリストを確保しておく
    std::vector<Eigen::Vector3d> result_dipole_list(atoms_list.size());

    #pragma omp parallel for
    for (int i=0; i< atoms_list.size(); i++){
        // ! 予測値用の双極子
        Eigen::Vector3d TotalDipole = Eigen::Vector3d::Zero();
        // ! 入力となるtensor用（形式は1,288の形！！）
        // TODO :: hard code :: 入力記述子の形はどうやってコントロールしようか？
        torch::Tensor input = torch::ones({1, 288}).to("cpu");
        // ! ボンドごとの予測値を保存するための双極子変数
        Eigen::Vector3d Dipole_tmp = Eigen::Vector3d::Zero();
        // ! true_yを保存するためのやつ．
        std::vector<Eigen::Vector3d> true_y_list_coc;
        std::vector<Eigen::Vector3d> true_y_list_coh;


        // pbc-molをかけた原子座標(test_mol)と，それを利用したbcを取得
        auto test_mol_bc = raw_aseatom_to_mol_coord_and_bc(atoms_list[i], test_read_mol.bonds_list, test_read_mol, NUM_MOL_ATOMS, NUM_MOL);
        auto test_mol=std::get<0>(test_mol_bc);
        auto test_bc =std::get<1>(test_mol_bc);

        //! chボンド双極子の作成
        auto descs_ch = raw_calc_bond_descripter_at_frame(atoms_list[i], test_bc, test_read_mol.ch_bond_index, NUM_MOL, UNITCELL_VECTORS,  NUM_MOL_ATOMS, var_des.desctype);
        if ( SAVE_DESCS == true){
            //! test for save as npy file.
            // descs_chの形を1dへ変形してnpyで保存．
            // TODO :: さすがにもっと効率の良い方法があるはず．
            std::vector<double> descs_ch_1d;
            for (int i = 0; i < descs_ch.size(); i++) {
                for (int j = 0; j < descs_ch[i].size(); j++) { //これが288個のはず
                    descs_ch_1d.push_back(descs_ch[i][j]); 
                }
            }
            //! npy.hppを利用して保存する．
            const std::vector<long unsigned> shape_descs_ch{descs_ch.size(), descs_ch[0].size()}; // vectorを1*12の形に保存
            npy::SaveArrayAsNumpy("descs_ch"+std::to_string(i)+".npy", false, shape_descs_ch.size(), shape_descs_ch.data(), descs_ch_1d);
        } //! end if SAVE_DESCS
        // ! descs_chの予測
        // loop over descs_ch
        for (int j = 0, n=descs_ch.size(); j < n; j++) {
#ifdef DEBUG
            std::cout << "descs_ch size" << descs_ch[j].size() << std::endl;
            for (int k = 0; k<288;k++){
                std::cout << descs_ch[j][k] << " ";
            };
            std::cout << std::endl;
#endif //! DEBUG
            // torch::Tensor input = torch::tensor(torch::ArrayRef<double>({descs_ch[i]})).to("cpu");
            // https://stackoverflow.com/questions/63531428/convert-c-vectorvectorfloat-to-torchtensor
            // 入力となる記述子にvectorから値をcopy 
            // TODO :: 多分もっと綺麗な方法があるはず．．． ただ1次元ではなく(1,288)という形をしているが故にちょっと問題になっている．
            // torch::Tensor input = torch::from_blob(descs_tmp.data(), {1,288}).to("cpu");
            for (int k = 0; k<288;k++){
                input[0][k] = descs_ch[j][k];
            };
            // std::cout << input << std::endl ;
            // 推論と同時に出力結果を変数に格納
            // auto elements = module.forward({input}).toTuple() -> elements();
            torch::Tensor elements = module_ch.forward({input}).toTensor() ;
            TotalDipole += Eigen::Vector3d {elements[0][0].item().toDouble(), elements[0][1].item().toDouble(), elements[0][2].item().toDouble()};
        }
        //! END test raw_calc_bond_descripter_at_frame (chボンドのテスト)

        //! test raw_calc_bond_descripter_at_frame (ccのボンドのテスト)
        //! ccボンド双極子の作成
        auto descs_cc = raw_calc_bond_descripter_at_frame(atoms_list[i], test_bc, test_read_mol.cc_bond_index, NUM_MOL, UNITCELL_VECTORS,  NUM_MOL_ATOMS, var_des.desctype);
        //!! 注意：：ccボンドの場合，最近説のC原子への距離が二つのC原子で同じなので，ここの並びが変わることがあり得る．
        if ( SAVE_DESCS == true ){
            // descs_chの形を1dへ変形してnpyで保存．
            // TODO :: さすがにもっと効率の良い方法があるはず．
            std::vector<double> descs_cc_1d;
            for (int i = 0; i < descs_cc.size(); i++) {
                for (int j = 0; j < descs_cc[i].size(); j++) {
                    descs_cc_1d.push_back(descs_cc[i][j]);
                }
            }
            //! npy.hppを利用して保存する．
            const std::vector<long unsigned> shape_descs_cc{descs_cc.size(), descs_cc[0].size()}; // vectorを1*12の形に保存
            npy::SaveArrayAsNumpy("descs_cc"+std::to_string(i)+".npy", false, shape_descs_cc.size(), shape_descs_cc.data(), descs_cc_1d);
        }
        // ! descs_ccの予測
        for (int j = 0, n=descs_cc.size() ; j < n; j++) {  // loop over descs_cc
            // torch::Tensor input = torch::tensor(torch::ArrayRef<double>({descs_ch[i]})).to("cpu");
            // https://stackoverflow.com/questions/63531428/convert-c-vectorvectorfloat-to-torchtensor
            // 入力となる記述子にvectorから値をcopy 
            // TODO :: 多分もっと綺麗な方法があるはず．．． ただ1次元ではなく(1,288)という形をしているが故にちょっと問題になっている．
            // torch::Tensor input = torch::from_blob(descs_tmp.data(), {1,288}).to("cpu");
            for (int k = 0; k<288;k++){
                input[0][k] = descs_cc[j][k];
            };

            // std::cout << input << std::endl ;
            // 推論と同時に出力結果を変数に格納
            // auto elements = module.forward({input}).toTuple() -> elements();
            torch::Tensor elements = module_cc.forward({input}).toTensor() ;

            // 出力結果
            // std::cout << j << " " << elements[0][0].item() << " " << elements[0][1].item() << " " << elements[0][2].item() << std::endl;
            TotalDipole += Eigen::Vector3d {elements[0][0].item().toDouble(), elements[0][1].item().toDouble(), elements[0][2].item().toDouble()};
            // auto output = elements[0].toTensor();
        }
        //! END test raw_calc_bond_descripter_at_frame (ccのボンドのテスト)



        //! test raw_calc_bond_descripter_at_frame (coのボンドのテスト)
        // ! coc/cohの場合にはスキップ（とりあえず）
        if (!var_des.IF_COC){
            auto descs_co = raw_calc_bond_descripter_at_frame(atoms_list[i], test_bc, test_read_mol.co_bond_index, NUM_MOL, UNITCELL_VECTORS,  NUM_MOL_ATOMS, var_des.desctype);
            if ( SAVE_DESCS == true){
                //! test for save as npy file.
                // descs_chの形を1dへ変形してnpyで保存．
                // TODO :: さすがにもっと効率の良い方法があるはず．
                std::vector<double> descs_co_1d;
                for (int i = 0; i < descs_co.size(); i++) {
                    for (int j = 0; j < descs_co[i].size(); j++) {
                        descs_co_1d.push_back(descs_co[i][j]);
                    }
                }
                //! npy.hppを利用して保存する．
                const std::vector<long unsigned> shape_descs_co{descs_co.size(), descs_co[0].size()}; // vectorを1*12の形に保存
                npy::SaveArrayAsNumpy("descs_co"+std::to_string(i)+".npy", false, shape_descs_co.size(), shape_descs_co.data(), descs_co_1d);
            } //! end if SAVE_DESCS
            // ! descs_coの予測
            // loop over descs_ch
            for (int j = 0, n=descs_co.size() ; j < n; j++) {

                // torch::Tensor input = torch::tensor(torch::ArrayRef<double>({descs_ch[i]})).to("cpu");
                // https://stackoverflow.com/questions/63531428/convert-c-vectorvectorfloat-to-torchtensor
                // 入力となる記述子にvectorから値をcopy 
                // TODO :: 多分もっと綺麗な方法があるはず．．． ただ1次元ではなく(1,288)という形をしているが故にちょっと問題になっている．
                // torch::Tensor input = torch::from_blob(descs_tmp.data(), {1,288}).to("cpu");
                for (int k = 0; k<288;k++){
                    input[0][k] = descs_co[j][k];
                };

                // std::cout << input << std::endl ;
                // 推論と同時に出力結果を変数に格納
                // auto elements = module.forward({input}).toTuple() -> elements();
                torch::Tensor elements = module_co.forward({input}).toTensor() ;

                // 出力結果
                // std::cout << j << " " << elements[0][0].item() << " " << elements[0][1].item() << " " << elements[0][2].item() << std::endl;
                TotalDipole += Eigen::Vector3d {elements[0][0].item().toDouble(), elements[0][1].item().toDouble(), elements[0][2].item().toDouble()};
                // auto output = elements[0].toTensor();
            }
        }

        //! test raw_calc_bond_descripter_at_frame (ohのボンドのテスト)
        // ! とりあえずCOC/COHの時はスキップ
        if (!var_des.IF_COC){
            // std::cout << " start descs_oh calculation ... " << std::endl;
            auto descs_oh = raw_calc_bond_descripter_at_frame(atoms_list[i], test_bc, test_read_mol.oh_bond_index, NUM_MOL, UNITCELL_VECTORS,  NUM_MOL_ATOMS, var_des.desctype);
            if ( SAVE_DESCS == true){
                // descs_chの形を1dへ変形してnpyで保存．
                // TODO :: さすがにもっと効率の良い方法があるはず．
                std::vector<double> descs_oh_1d;
                for (int i = 0; i < descs_oh.size(); i++) {
                    for (int j = 0; j < descs_oh[i].size(); j++) {
                        descs_oh_1d.push_back(descs_oh[i][j]);
                    }
                }
                const std::vector<long unsigned> shape_descs_oh{descs_oh.size(), descs_oh[0].size()}; // vectorを1*12の形に保存
                npy::SaveArrayAsNumpy("descs_oh"+std::to_string(i)+".npy", false, shape_descs_oh.size(), shape_descs_oh.data(), descs_oh_1d);
            }
            // ! descs_ohの予測
            // loop over descs_oh
            for (int j = 0, n=descs_oh.size(); j < n; j++) {

                // torch::Tensor input = torch::tensor(torch::ArrayRef<double>({descs_ch[i]})).to("cpu");
                // https://stackoverflow.com/questions/63531428/convert-c-vectorvectorfloat-to-torchtensor
                // 入力となる記述子にvectorから値をcopy 
                // TODO :: 多分もっと綺麗な方法があるはず．．． ただ1次元ではなく(1,288)という形をしているが故にちょっと問題になっている．
                // torch::Tensor input = torch::from_blob(descs_tmp.data(), {1,288}).to("cpu");
                for (int k = 0; k<288;k++){
                    input[0][k] = descs_oh[j][k];
                };

                // std::cout << input << std::endl ;
                // 推論と同時に出力結果を変数に格納
                // auto elements = module.forward({input}).toTuple() -> elements();
                torch::Tensor elements = module_oh.forward({input}).toTensor() ;

                // 出力結果
                // std::cout << j << " " << elements[0][0].item() << " " << elements[0][1].item() << " " << elements[0][2].item() << std::endl;
                TotalDipole += Eigen::Vector3d {elements[0][0].item().toDouble(), elements[0][1].item().toDouble(), elements[0][2].item().toDouble()};
                // auto output = elements[0].toTensor();
            }
        }
        //! END test raw_calc_bond_descripter_at_frame (ohのボンドのテスト)


        //! test raw_calc_lonepair_descripter_at_frame （ローンペアのテスト）
        // ! とりあえずCOC/COHの時はスキップ
        if (!var_des.IF_COC){
            auto descs_o = raw_calc_lonepair_descripter_at_frame(atoms_list[i], test_mol, test_read_mol.o_list, NUM_MOL, 8, UNITCELL_VECTORS,  NUM_MOL_ATOMS, var_des.desctype);
            if ( SAVE_DESCS == true){
                //! test for save as npy file.
                // descs_chの形を1dへ変形してnpyで保存．
                // TODO :: さすがにもっと効率の良い方法があるはず．
                std::vector<double> descs_o_1d;
                for (int i = 0; i < descs_o.size(); i++) {
                    for (int j = 0; j < descs_o[i].size(); j++) {
                        descs_o_1d.push_back(descs_o[i][j]);
                    }
                }
                //! npy.hppを利用して保存する．
                const std::vector<long unsigned> shape_descs_o{descs_o.size(), descs_o[0].size()}; // vectorを1*12の形に保存
                npy::SaveArrayAsNumpy("descs_o"+std::to_string(i)+".npy", false, shape_descs_o.size(), shape_descs_o.data(), descs_o_1d);
            }
            // ! descs_oの予測
            // loop over descs_o
            for (int j = 0, n = descs_o.size(); j < n; j++) {
                // torch::Tensor input = torch::tensor(torch::ArrayRef<double>({descs_ch[i]})).to("cpu");
                // https://stackoverflow.com/questions/63531428/convert-c-vectorvectorfloat-to-torchtensor
                // 入力となる記述子にvectorから値をcopy 
                // TODO :: 多分もっと綺麗な方法があるはず．．． ただ1次元ではなく(1,288)という形をしているが故にちょっと問題になっている．
                // torch::Tensor input = torch::from_blob(descs_tmp.data(), {1,288}).to("cpu");
                for (int k = 0; k<288;k++){
                    input[0][k] = descs_o[j][k];
                };

                // std::cout << input << std::endl ;
                // 推論と同時に出力結果を変数に格納
                // auto elements = module.forward({input}).toTuple() -> elements();
                torch::Tensor elements = module_o.forward({input}).toTensor() ;

                // 出力結果
                // std::cout << j << " " << elements[0][0].item() << " " << elements[0][1].item() << " " << elements[0][2].item() << std::endl;
                TotalDipole += Eigen::Vector3d {elements[0][0].item().toDouble(), elements[0][1].item().toDouble(), elements[0][2].item().toDouble()};
                // auto output = elements[0].toTensor();
            }
        }
        // ! END ローンペアのテスト

        //! test raw_calc_lonepair_descripter_at_frame2 （ローンペアのテスト2）
        // auto o_list_in_frame = get_atomslist_from_itpdata(test_read_mol.o_list, NUM_MOL, NUM_MOL_ATOMS);
        // auto descs_o_2 = raw_calc_lonepair_descripter_at_frame2(atoms_list[i], test_mol, o_list_in_frame, NUM_MOL, UNITCELL_VECTORS,  NUM_MOL_ATOMS, var_des.desctype);
        // // descs_o_2のデバック
        // for (int p=0; p<descs_o.size();p++){
        //     for (int q=0; q<descs_o[p].size();q++){
        //         if (descs_o[p][q] != descs_o_2[p][q]){
        //             std::cout << "ERROR :: descs_o and descs_o_2 are different at " << p << " " << q << std::endl;
        //         } else{
        //             std::cout << "descs_o and descs_o_2 are same at " << p << " " << q << std::endl;
        //         }
        //     }
        // }
        // ! descs_coc/cohの予測
        if (var_des.IF_COC){
            //! coc/coh記述子の計算
            // coc
            auto coc_list_in_frame = get_atomslist_from_itpdata(test_read_mol.coc_list, NUM_MOL, NUM_MOL_ATOMS);
            auto descs_coc = raw_calc_lonepair_descripter_at_frame2(atoms_list[i], test_mol, coc_list_in_frame, NUM_MOL, UNITCELL_VECTORS,  NUM_MOL_ATOMS, var_des.desctype);
            // coh
            auto coh_list_in_frame = get_atomslist_from_itpdata(test_read_mol.coh_list, NUM_MOL, NUM_MOL_ATOMS);
            auto descs_coh = raw_calc_lonepair_descripter_at_frame2(atoms_list[i], test_mol, coh_list_in_frame, NUM_MOL, UNITCELL_VECTORS,  NUM_MOL_ATOMS, var_des.desctype);

            //! 予測部分
            // loop over descs_coc
            for (int j = 0, n = descs_coc.size(); j < n; j++) {

                // torch::Tensor input = torch::tensor(torch::ArrayRef<double>({descs_ch[i]})).to("cpu");
                // https://stackoverflow.com/questions/63531428/convert-c-vectorvectorfloat-to-torchtensor
                // 入力となる記述子にvectorから値をcopy 
                // TODO :: 多分もっと綺麗な方法があるはず．．． ただ1次元ではなく(1,288)という形をしているが故にちょっと問題になっている．
                // torch::Tensor input = torch::from_blob(descs_tmp.data(), {1,288}).to("cpu");
                for (int k = 0; k<288;k++){
                    input[0][k] = descs_coc[j][k];
                };

                // std::cout << input << std::endl ;
                // 推論と同時に出力結果を変数に格納
                // auto elements = module.forward({input}).toTuple() -> elements();
                torch::Tensor elements = module_coc.forward({input}).toTensor() ;

                // 出力結果
                // std::cout << j << " " << elements[0][0].item() << " " << elements[0][1].item() << " " << elements[0][2].item() << std::endl;
                Dipole_tmp = Eigen::Vector3d {elements[0][0].item().toDouble(), elements[0][1].item().toDouble(), elements[0][2].item().toDouble()};
                TotalDipole += Dipole_tmp;
                if (SAVE_TRUEY){
                    true_y_list_coc.push_back(Dipole_tmp);
                }
                // auto output = elements[0].toTensor();
            }

            // loop over descs_coh
            for (int j = 0, n = descs_coh.size(); j < n; j++) {

                // torch::Tensor input = torch::tensor(torch::ArrayRef<double>({descs_ch[i]})).to("cpu");
                // https://stackoverflow.com/questions/63531428/convert-c-vectorvectorfloat-to-torchtensor
                // 入力となる記述子にvectorから値をcopy 
                // TODO :: 多分もっと綺麗な方法があるはず．．． ただ1次元ではなく(1,288)という形をしているが故にちょっと問題になっている．
                // torch::Tensor input = torch::from_blob(descs_tmp.data(), {1,288}).to("cpu");
                for (int k = 0; k<288;k++){
                    input[0][k] = descs_coh[j][k];
                };

                // std::cout << input << std::endl ;
                // 推論と同時に出力結果を変数に格納
                // auto elements = module.forward({input}).toTuple() -> elements();
                torch::Tensor elements = module_coh.forward({input}).toTensor() ;

                // 出力結果
                // std::cout << j << " " << elements[0][0].item() << " " << elements[0][1].item() << " " << elements[0][2].item() << std::endl;
                Dipole_tmp = Eigen::Vector3d {elements[0][0].item().toDouble(), elements[0][1].item().toDouble(), elements[0][2].item().toDouble()};
                TotalDipole += Dipole_tmp;
                if (SAVE_TRUEY){
                    true_y_list_coh.push_back(Dipole_tmp);
                }
                // auto output = elements[0].toTensor();
            }
            if (SAVE_TRUEY){
                //! npy.hppを利用して保存する．
                // * https://github.com/llohse/libnpy/blob/master/tests/test-save.cpp
                // 双極子の出力ファイル
                std::ofstream true_y_coh_fout("true_y_coh"+std::to_string(i)+".txt"); 
                for (int indx_t_y=0; indx_t_y< true_y_list_coh.size();indx_t_y++){
                    true_y_coh_fout <<  std::right << std::setw(16) << true_y_list_coh[indx_t_y][0] << std::setw(16) << true_y_list_coh[indx_t_y][1] << std::setw(16) << true_y_list_coh[indx_t_y][2] << std::endl;
                }
                true_y_coh_fout.close();
                // 双極子の出力ファイル
                std::ofstream true_y_coc_fout("true_y_coc"+std::to_string(i)+".txt"); 
                for (int indx_t_y=0; indx_t_y< true_y_list_coc.size();indx_t_y++){
                    true_y_coc_fout <<  std::right << std::setw(16) << true_y_list_coc[indx_t_y][0] << std::setw(16) << true_y_list_coc[indx_t_y][1] << std::setw(16) << true_y_list_coc[indx_t_y][2] << std::endl;
                }
                true_y_coc_fout.close();
                // const std::vector<long unsigned> shape_true_y_list_coh{true_y_list_coh.size()}; // 1d vectorの保存
                // npy::SaveArrayAsNumpy("true_y_coh"+std::to_string(i)+".npy", false,shape_true_y_list_coh.size(), shape_true_y_list_coh.data(), true_y_list_coh);
                // const std::vector<long unsigned> shape_true_y_list_coc{true_y_list_coc.size()}; // 1d vectorの保存
                // npy::SaveArrayAsNumpy("true_y_coc"+std::to_string(i)+".npy", false,shape_true_y_list_coc.size(), shape_true_y_list_coc.data(), true_y_list_coc);
            }
        } //! end if var_des.IF_COC(coc/coh)

        //! test raw_calc_bond_descripter_at_frame (coのボンドのテスト)
        auto descs_co = raw_calc_bond_descripter_at_frame(atoms_list[i], test_bc, test_read_mol.co_bond_index, NUM_MOL, UNITCELL_VECTORS,  NUM_MOL_ATOMS, var_des.desctype);
        if ( SAVE_DESCS == true){
            //! test for save as npy file.
            // descs_chの形を1dへ変形してnpyで保存．
            // TODO :: さすがにもっと効率の良い方法があるはず．
            std::vector<double> descs_co_1d;
            for (int i = 0; i < descs_co.size(); i++) {
                for (int j = 0; j < descs_co[i].size(); j++) {
                    descs_co_1d.push_back(descs_co[i][j]);
                }
            }
            //! npy.hppを利用して保存する．
            const std::vector<long unsigned> shape_descs_co{descs_co.size(), descs_co[0].size()}; // vectorを1*12の形に保存
            npy::SaveArrayAsNumpy("descs_co"+std::to_string(i)+".npy", false, shape_descs_co.size(), shape_descs_co.data(), descs_co_1d);
        }
        // ! descs_coの予測
        for (int j = 0, n=descs_co.size() ; j < n; j++) { // loop over descs_co
            // torch::Tensor input = torch::tensor(torch::ArrayRef<double>({descs_ch[i]})).to("cpu");
            // https://stackoverflow.com/questions/63531428/convert-c-vectorvectorfloat-to-torchtensor
            // 入力となる記述子にvectorから値をcopy 
            // TODO :: 多分もっと綺麗な方法があるはず．．． ただ1次元ではなく(1,288)という形をしているが故にちょっと問題になっている．
            // torch::Tensor input = torch::from_blob(descs_tmp.data(), {1,288}).to("cpu");
            for (int k = 0; k<288;k++){
                input[0][k] = descs_co[j][k];
            };
            // std::cout << input << std::endl ;
            // 推論と同時に出力結果を変数に格納
            // auto elements = module.forward({input}).toTuple() -> elements();
            torch::Tensor elements = module_co.forward({input}).toTensor() ;
            // 出力結果
            // std::cout << j << " " << elements[0][0].item() << " " << elements[0][1].item() << " " << elements[0][2].item() << std::endl;
            TotalDipole += Eigen::Vector3d {elements[0][0].item().toDouble(), elements[0][1].item().toDouble(), elements[0][2].item().toDouble()};
            // auto output = elements[0].toTensor();
        }

        //! test raw_calc_bond_descripter_at_frame (ohのボンドのテスト)
        // std::cout << " start descs_oh calculation ... " << std::endl;
        auto descs_oh = raw_calc_bond_descripter_at_frame(atoms_list[i], test_bc, test_read_mol.oh_bond_index, NUM_MOL, UNITCELL_VECTORS,  NUM_MOL_ATOMS, var_des.desctype);
        if ( SAVE_DESCS == true){
            // descs_chの形を1dへ変形してnpyで保存．
            // TODO :: さすがにもっと効率の良い方法があるはず．
            std::vector<double> descs_oh_1d;
            for (int i = 0; i < descs_oh.size(); i++) {
                for (int j = 0; j < descs_oh[i].size(); j++) {
                    descs_oh_1d.push_back(descs_oh[i][j]);
                }
            }
            const std::vector<long unsigned> shape_descs_oh{descs_oh.size(), descs_oh[0].size()}; // vectorを1*12の形に保存
            npy::SaveArrayAsNumpy("descs_oh"+std::to_string(i)+".npy", false, shape_descs_oh.size(), shape_descs_oh.data(), descs_oh_1d);
        }
        // ! descs_ohの予測
        for (int j = 0, n=descs_oh.size(); j < n; j++) { // loop over descs_ch
            // torch::Tensor input = torch::tensor(torch::ArrayRef<double>({descs_ch[i]})).to("cpu");
            // https://stackoverflow.com/questions/63531428/convert-c-vectorvectorfloat-to-torchtensor
            // 入力となる記述子にvectorから値をcopy 
            // TODO :: 多分もっと綺麗な方法があるはず．．． ただ1次元ではなく(1,288)という形をしているが故にちょっと問題になっている．
            // torch::Tensor input = torch::from_blob(descs_tmp.data(), {1,288}).to("cpu");
            for (int k = 0; k<288;k++){
                input[0][k] = descs_oh[j][k];
            };

            // std::cout << input << std::endl ;
            // 推論と同時に出力結果を変数に格納
            // auto elements = module.forward({input}).toTuple() -> elements();
            torch::Tensor elements = module_oh.forward({input}).toTensor() ;

            // 出力結果
            // std::cout << j << " " << elements[0][0].item() << " " << elements[0][1].item() << " " << elements[0][2].item() << std::endl;
            TotalDipole += Eigen::Vector3d {elements[0][0].item().toDouble(), elements[0][1].item().toDouble(), elements[0][2].item().toDouble()};
            // auto output = elements[0].toTensor();
        }

        //! test raw_calc_lonepair_descripter_at_frame （ローンペアのテスト）
        auto descs_o = raw_calc_lonepair_descripter_at_frame(atoms_list[i], test_mol, test_read_mol.o_list, NUM_MOL, 8, UNITCELL_VECTORS,  NUM_MOL_ATOMS, var_des.desctype);
        if ( SAVE_DESCS == true){
            //! test for save as npy file.
            // descs_chの形を1dへ変形してnpyで保存．
            // TODO :: さすがにもっと効率の良い方法があるはず．
            std::vector<double> descs_o_1d;
            for (int i = 0; i < descs_o.size(); i++) {
                for (int j = 0; j < descs_o[i].size(); j++) {
                    descs_o_1d.push_back(descs_o[i][j]);
                }
            }
            //! npy.hppを利用して保存する．
            const std::vector<long unsigned> shape_descs_o{descs_o.size(), descs_o[0].size()}; // vectorを1*12の形に保存
            npy::SaveArrayAsNumpy("descs_o"+std::to_string(i)+".npy", false, shape_descs_o.size(), shape_descs_o.data(), descs_o_1d);
        }
        // ! descs_oの予測
        for (int j = 0, n = descs_o.size(); j < n; j++) { // loop over descs_o
            // torch::Tensor input = torch::tensor(torch::ArrayRef<double>({descs_ch[i]})).to("cpu");
            // https://stackoverflow.com/questions/63531428/convert-c-vectorvectorfloat-to-torchtensor
            // 入力となる記述子にvectorから値をcopy 
            // TODO :: 多分もっと綺麗な方法があるはず．．． ただ1次元ではなく(1,288)という形をしているが故にちょっと問題になっている．
            // torch::Tensor input = torch::from_blob(descs_tmp.data(), {1,288}).to("cpu");
            for (int k = 0; k<288;k++){
                input[0][k] = descs_o[j][k];
            };
            // std::cout << input << std::endl ;
            // 推論と同時に出力結果を変数に格納
            // auto elements = module.forward({input}).toTuple() -> elements();
            torch::Tensor elements = module_o.forward({input}).toTensor() ;
            // 出力結果
            // std::cout << j << " " << elements[0][0].item() << " " << elements[0][1].item() << " " << elements[0][2].item() << std::endl;
            TotalDipole += Eigen::Vector3d {elements[0][0].item().toDouble(), elements[0][1].item().toDouble(), elements[0][2].item().toDouble()};
            // auto output = elements[0].toTensor();
        }


        if (omp_get_thread_num() == 1){ // スレッド1番でのみ出力
            std::cout << "TotalDipole :: " << i << " " << TotalDipole[0] << " "  << TotalDipole[1] << " "  << TotalDipole[2] << " " << std::endl;
        }
        result_dipole_list[i]=TotalDipole;
     }


    // ! >>>>>>>>>>>>>>>>
    // ! 計算終了，最後のファイル保存
    // ! >>>>>>>>>>>>>>>>

    // 最後にtotal双極子をファイルに保存
    fout << "# index dipole_x dipole_y dipole_z" << std::endl;
    for (int i = 0; i < result_dipole_list.size(); i++){
        fout << std::setw(5) << i << std::right << std::setw(16) << result_dipole_list[i][0] << std::setw(16) << result_dipole_list[i][1] << std::setw(16) << result_dipole_list[i][2] << std::endl;
    }
    fout.close();

    clock_t end = clock();     // 終了時間
    end_c = std::chrono::system_clock::now();  // 計測終了時間
    double elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_c-start_c).count();
    std::cout << "duration (clock) = " << (double)(end - start) / CLOCKS_PER_SEC << "sec.\n";
    std::cout << "duration (chrono) = " << elapsed << "sec.\n";
    std::cout << "finish !! " << std::endl;
    fout.close();

    return 0;
}
