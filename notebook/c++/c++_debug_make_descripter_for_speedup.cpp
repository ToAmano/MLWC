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
#include "parse.cpp"

// #include <GraphMol/GraphMol.h>
// #include <GraphMol/FileParsers/MolSupplier.h>
// #include <GraphMol/FileParsers/MolWriters.h>
// #include <GraphMol/FileParsers/FileParsers.h>


// https://e-penguiner.com/cpp-function-check-file-exist/#index_id2
bool IsFileExist(const std::string& name) {
    return std::filesystem::is_regular_file(name);
}


int main(int argc, char *argv[]) {

    // 
    bool SAVE_DESCS = false; // trueならデスクリプターをnpyで保存．

    // 双極子の出力ファイル
    std::ofstream fout("total_dipole.txt"); 

    // 入力ファイルを読み込み（テスト）


    clock_t start = clock();    // スタート時間
     std::chrono::system_clock::time_point  start_c, end_c; // 型は auto で可
     start_c = std::chrono::system_clock::now(); // 計測開始時間

    // std::string xyz_filename="/Users/amano/works/research/dieltools/notebook/c++/gromacs_trajectory_cell.xyz";
    // std::string xyz_filename="/Users/amano/works/research/dieltools/notebook/c++/gromacs_pg_1ns_dt50fs.xyz";
    // std::string xyz_filename="/Users/amano/works/research/dieltools/notebook/c++/gromacs_pg_1ns_dt50fs_300.xyz";
    if (argc < 3) {
        std::cout << "Error: xyz file does not provided." << std::endl;
        return 0;
    }

    std::string bond_filename=argv[1]; // bondファイル名を引数で指定．
    if (!IsFileExist(bond_filename)) {
        std::cout << "Error: bond file does not exist." << std::endl;
        return 0;
    }

    //TODO :: 実装中
    std::string inp_filename=argv[2];
    if (!IsFileExist(inp_filename)) {
        std::cout << "Error: inp file does not exist." << std::endl;
        return 0;
    }
    
    std::cout << "START code !! ";
    auto [inp_general, inp_desc, inp_pred] = locate_tag(inp_filename);
    auto var_gen = var_general(inp_general);
    auto var_des = var_descripter(inp_desc);
    auto var_pre = var_predict(inp_pred);


    //! 原子数の取得
    int NUM_ATOM = raw_cpmd_num_atom(var_des.xyzfilename);
    //! 格子定数の取得
    std::vector<std::vector<double> > UNITCELL_VECTORS = raw_cpmd_get_unitcell_xyz(var_des.xyzfilename);
    //! xyzファイルから座標リストを取得
    std::vector<Atoms> atoms_list = ase_io_read(var_des.xyzfilename);

    //! ボンドリストの取得
    // TODO :: 現状では，ボンドリストはmol_core.cpp内で定義されている．（こういうブラックボックスをなんとかしたい）
    // TODO :: 最悪でもボンドファイルはinput
    read_mol test_read_mol(bond_filename);

    // RDKit::ROMol *mol1 = RDKit::SmilesToMol( "Cc1ccccc1" );
    // std::string mol_file = "../../../../smiles/pg.acpype/input_GMX.mol";
    // RDKit::ROMol *mol1 = RDKit::MolFileToMol(mol_file);
    // std::shared_ptr<RDKit::ROMol> mol2( RDKit::MolFileToMol(mol_file) );
    // std::cout << *mol2 << std::endl;


    // std::cout << "start descs_cc prediction ... " << std::endl;
    // torch::jit::script::Module 型で module 変数の定義
    torch::jit::script::Module module_ch;
    // 変換した学習済みモデルの読み込み
    // TODO :: 現状ではpathが実行ファイルからの相対パスになっている．
    // TODO :: これではめちゃくちゃ使いにくいので，コードを実行した
    // TODO :: ディレクトリからのpathにしたい．
    module_ch = torch::jit::load(var_pre.model_dir+"/model_ch.pt");
//    module_ch = torch::jit::load(var_pre.model_dir+"/Users/amano/works/research/dieltools/notebook/c++/202306014_model_rotate/model_ch.pt");

    // torch::jit::script::Module 型で module 変数の定義
    torch::jit::script::Module module_cc;
    // 変換した学習済みモデルの読み込み
    module_cc = torch::jit::load(var_pre.model_dir+"/model_cc.pt");

    // torch::jit::script::Module 型で module 変数の定義
    torch::jit::script::Module module_co;
    // 変換した学習済みモデルの読み込み
    module_co = torch::jit::load(var_pre.model_dir+"/model_co.pt");

    // torch::jit::script::Module 型で module 変数の定義
    torch::jit::script::Module module_oh;
    // 変換した学習済みモデルの読み込み
    module_oh = torch::jit::load(var_pre.model_dir+"/model_oh.pt");

    // torch::jit::script::Module 型で module 変数の定義
    torch::jit::script::Module module_o;
    // 変換した学習済みモデルの読み込み
    module_o = torch::jit::load(var_pre.model_dir+"/model_o.pt");

    //! test raw_aseatom_to_mol_coord_and_bc
    int NUM_MOL_ATOMS = test_read_mol.num_atoms_per_mol;
    int NUM_MOL = int(NUM_ATOM/NUM_MOL_ATOMS); // UnitCell中の総分子数
    std::cout << "NUM_MOL :: " << NUM_MOL << std::endl;
    std::cout << "NUM_MOL_ATOMS :: " << NUM_MOL_ATOMS << std::endl;

    // Beginning of parallel region
    #pragma omp parallel
    {
        printf("Hello World... from thread = %d\n",
               omp_get_thread_num());
    }
    // Ending of parallel region

    // 予め出力する双極子用のリストを確保しておく
    std::vector<Eigen::Vector3d> result_dipole_list(atoms_list.size());

    // for (int i=0; i< atoms_list.size();i++){
    #pragma omp parallel for
    for (int i=0; i< atoms_list.size(); i++){
        auto test_mol_bc = raw_aseatom_to_mol_coord_and_bc(atoms_list[i], test_read_mol.bonds_list, test_read_mol, NUM_MOL_ATOMS, NUM_MOL);
        auto test_mol=std::get<0>(test_mol_bc);
        auto test_bc =std::get<1>(test_mol_bc);

        // //! test make_ase_with_BCs
        // Atoms new_atoms = make_ase_with_BCs(atoms_list[i].get_atomic_numbers(), NUM_MOL, raw_cpmd_get_unitcell_xyz(xyz_filename), test_mol, test_bc);
    
        // //! test ase_io_write
        // ase_io_write(new_atoms, "test_atoms"+std::to_string(i)+".xyz");

        //! test raw_calc_bond_descripter_at_frame (chボンドのテスト)
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

        //! test raw_calc_bond_descripter_at_frame (ccのボンドのテスト)
        //!! 注意：：ccボンドの場合，最近説のC原子への距離が二つのC原子で同じなので，ここの並びが変わることがあり得る．
        auto descs_cc = raw_calc_bond_descripter_at_frame(atoms_list[i], test_bc, test_read_mol.cc_bond_index, NUM_MOL, UNITCELL_VECTORS,  NUM_MOL_ATOMS, var_des.desctype);
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

        // ! 予測値用の双極子
        Eigen::Vector3d TotalDipole = Eigen::Vector3d::Zero();

        // ! 入力となるtensor用（形式は1,288の形！！）
        torch::Tensor input = torch::ones({1, 288}).to("cpu");

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

            // 出力結果
            // std::cout << j << " " << elements[0][0].item() << " " << elements[0][1].item() << " " << elements[0][2].item() << std::endl;
            TotalDipole += Eigen::Vector3d {elements[0][0].item().toDouble(), elements[0][1].item().toDouble(), elements[0][2].item().toDouble()};
            // auto output = elements[0].toTensor();
        }

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

        // ! descs_ohの予測
        // loop over descs_ch
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

        // ! descs_oの予測
        // loop over descs_ch
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

        // ! descs_ccの予測
        // loop over descs_ch
        for (int j = 0, n=descs_cc.size() ; j < n; j++) {

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

        if (omp_get_thread_num() == 1){ // スレッド1番でのみ出力
            std::cout << "TotalDipole :: " << i << " " << TotalDipole[0] << " "  << TotalDipole[1] << " "  << TotalDipole[2] << " " << std::endl;
        }
        result_dipole_list[i]=TotalDipole;
     }
    // 最後にファイルに保存
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
}