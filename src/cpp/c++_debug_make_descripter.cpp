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
#include <time.h>     // for clock() http://vivi.dyndns.org/tech/cpp/timeMeasurement.html
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


int main() {

    // 
    bool SAVE_DESCS = false; // trueならデスクリプターをnpyで保存．

    // 双極子の出力
    std::ofstream fout("total_dipole.txt"); 

    clock_t start = clock();    // スタート時間
    // std::string xyz_filename="/Users/amano/works/research/dieltools/notebook/c++/gromacs_trajectory_cell.xyz";
    std::string xyz_filename="/Users/amano/works/research/dieltools/notebook/c++/gromacs_pg_1ns_dt50fs.xyz";
    std::cout << "START code !! ";
    //! 原子数の取得
    int NUM_ATOM = raw_cpmd_num_atom(xyz_filename);
    //! 格子定数の取得
    std::vector<std::vector<double> > UNITCELL_VECTORS = raw_cpmd_get_unitcell_xyz(xyz_filename);
    //! xyzファイルから座標リストを取得
    std::vector<Atoms> atoms_list = ase_io_read(xyz_filename);


    //! ボンドリストの取得
    // TODO :: 現状では，ボンドリストはmol_core.cpp内で定義されている．（こういうブラックボックスをなんとかしたい）
    read_mol test_read_mol;

    //! test raw_aseatom_to_mol_coord_and_bc
    int NUM_MOL_ATOMS = test_read_mol.num_atoms_per_mol;
    int NUM_MOL = int(NUM_ATOM/NUM_MOL_ATOMS); // UnitCell中の総分子数
    std::cout << "NUM_MOL :: " << NUM_MOL << std::endl;
    std::cout << "NUM_MOL_ATOMS :: " << NUM_MOL_ATOMS << std::endl;
    // for (int i=0; i< atoms_list.size();i++){
    for (int i=0; i< atoms_list.size(); i++){
        auto test_mol_bc = raw_aseatom_to_mol_coord_and_bc(atoms_list[i], test_read_mol.bonds_list, test_read_mol, NUM_MOL_ATOMS, NUM_MOL);
        auto test_mol=std::get<0>(test_mol_bc);
        auto test_bc =std::get<1>(test_mol_bc);

        //! test make_ase_with_BCs
        Atoms new_atoms = make_ase_with_BCs(atoms_list[i].get_atomic_numbers(), NUM_MOL, raw_cpmd_get_unitcell_xyz(xyz_filename), test_mol, test_bc);
    
        //! test ase_io_write
        ase_io_write(new_atoms, "test_atoms"+std::to_string(i)+".xyz");

        //! test raw_calc_bond_descripter_at_frame (chボンドのテスト)
        auto descs_ch = raw_calc_bond_descripter_at_frame(atoms_list[i], test_bc, test_read_mol.ch_bond_index, NUM_MOL, UNITCELL_VECTORS,  NUM_MOL_ATOMS);
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
        }
        //! test raw_calc_bond_descripter_at_frame (ccのボンドのテスト)
        //!! 注意：：ccボンドの場合，最近説のC原子への距離が二つのC原子で同じなので，ここの並びが変わることがあり得る．
        auto descs_cc = raw_calc_bond_descripter_at_frame(atoms_list[i], test_bc, test_read_mol.cc_bond_index, NUM_MOL, UNITCELL_VECTORS,  NUM_MOL_ATOMS);
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

        //! test raw_calc_bond_descripter_at_frame (coのボンドのテスト)
        auto descs_co = raw_calc_bond_descripter_at_frame(atoms_list[i], test_bc, test_read_mol.co_bond_index, NUM_MOL, UNITCELL_VECTORS,  NUM_MOL_ATOMS);
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
        auto descs_oh = raw_calc_bond_descripter_at_frame(atoms_list[i], test_bc, test_read_mol.oh_bond_index, NUM_MOL, UNITCELL_VECTORS,  NUM_MOL_ATOMS);
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
        auto descs_o = raw_calc_lonepair_descripter_at_frame(atoms_list[i], test_mol, test_read_mol.o_list, NUM_MOL, 8, UNITCELL_VECTORS,  NUM_MOL_ATOMS);
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

        // ! descs_chの予測
        // std::cout << "start descs_ch prediction ... " << std::endl;
        // torch::jit::script::Module 型で module 変数の定義
        torch::jit::script::Module module_ch;
        // 変換した学習済みモデルの読み込み
        module_ch = torch::jit::load("/Users/amano/works/research/dieltools/notebook/c++/202306014_model_rotate/model_ch.pt");
        // loop over descs_ch
        for (int j = 0; j < descs_ch.size(); j++) {
#ifdef DEBUG
            std::cout << "descs_ch size" << descs_ch[j].size() << std::endl;
            for (int k = 0; k<288;k++){
                std::cout << descs_ch[j][k] << " ";
            };
            std::cout << std::endl;
#endif //! DEBUG
            // モデルへのサンプル入力テンソル（形式は1,288の形！！）
            torch::Tensor input = torch::ones({1, 288}).to("cpu");
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
        // std::cout << "start descs_co prediction ... " << std::endl;
        // torch::jit::script::Module 型で module 変数の定義
        torch::jit::script::Module module_co;
        // 変換した学習済みモデルの読み込み
        module_co = torch::jit::load("/Users/amano/works/research/dieltools/notebook/c++/202306014_model_rotate/model_co.pt");
        // loop over descs_ch
        for (int j = 0; j < descs_co.size(); j++) {

            // モデルへのサンプル入力テンソル（形式は1,288の形！！）
            torch::Tensor input = torch::ones({1, 288}).to("cpu");
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
        // std::cout << "start descs_oh prediction ... " << std::endl;
        // torch::jit::script::Module 型で module 変数の定義
        torch::jit::script::Module module_oh;
        // 変換した学習済みモデルの読み込み
        module_oh = torch::jit::load("/Users/amano/works/research/dieltools/notebook/c++/202306014_model_rotate/model_oh.pt");
        // loop over descs_ch
        for (int j = 0; j < descs_oh.size(); j++) {

            // モデルへのサンプル入力テンソル（形式は1,288の形！！）
            torch::Tensor input = torch::ones({1, 288}).to("cpu");
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
        // std::cout << "start descs_o prediction ... " << std::endl;
        // torch::jit::script::Module 型で module 変数の定義
        torch::jit::script::Module module_o;
        // 変換した学習済みモデルの読み込み
        module_o = torch::jit::load("/Users/amano/works/research/dieltools/notebook/c++/202306014_model_rotate/model_o.pt");
        // loop over descs_ch
        for (int j = 0; j < descs_o.size(); j++) {

            // モデルへのサンプル入力テンソル（形式は1,288の形！！）
            torch::Tensor input = torch::ones({1, 288}).to("cpu");
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
        // std::cout << "start descs_cc prediction ... " << std::endl;
        // torch::jit::script::Module 型で module 変数の定義
        torch::jit::script::Module module_cc;
        // 変換した学習済みモデルの読み込み
        module_cc = torch::jit::load("/Users/amano/works/research/dieltools/notebook/c++/202306014_model_rotate/model_cc.pt");
        // loop over descs_ch
        for (int j = 0; j < descs_cc.size(); j++) {

            // モデルへのサンプル入力テンソル（形式は1,288の形！！）
            torch::Tensor input = torch::ones({1, 288}).to("cpu");
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


        std::cout << "TotalDipole :: " << TotalDipole[0] << " "  << TotalDipole[1] << " "  << TotalDipole[2] << " " << std::endl;
        fout << std::right << std::setw(16) << TotalDipole[0] << std::setw(16) << TotalDipole[1] << std::setw(16) << TotalDipole[2] << std::endl;
    }

    clock_t end = clock();     // 終了時間
    std::cout << "duration = " << (double)(end - start) / CLOCKS_PER_SEC << "sec.\n";
}