// https://github.com/microsoft/vscode-cpptools/issues/7413
#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

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
#include <Eigen/Dense> // vector3dにはこれが必要？
#include "numpy.hpp"
#include "npy.hpp"
#include "torch/script.h" // pytorch
// #include "numpy_quiita.hpp" // https://qiita.com/ka_na_ta_n/items/608c7df3128abbf39c89
// numpy_quiitaはsscanf_sが読み込めず，残念ながら現状使えない．
#include "yaml-cpp/yaml.h" //https://github.com/jbeder/yaml-cpp

#include "atoms_asign_wcs.hpp"
#include "descriptor.hpp"
#include "parse.hpp"
#include "include/error.h"
#include "include/savevec.hpp"
#include "include/printvec.hpp"
#include "include/constant.hpp"
#include "include/manupilate_files.hpp"
#include "include/stopwatch.hpp"
#include "include/timer.hpp"
#include "postprocess/dielconst.hpp"
#include "predict.hpp"
#include "atoms_io.hpp"
#include "atoms_core.hpp"
#include "postprocess/convert_gas.hpp"
#include "postprocess/save_dipole.hpp"

#include "module_input.hpp"
#include "module_xyz.hpp"
#include "module_bond.hpp"
#include "module_torch.hpp"

#include <GraphMol/GraphMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

// #include <GraphMol/GraphMol.h>
// #include <GraphMol/FileParsers/MolSupplier.h>
// #include <GraphMol/FileParsers/MolWriters.h>
// #include <GraphMol/FileParsers/FileParsers.h>



int main(int argc, char *argv[]) {
    std::cout << " +-----------------------------------------------------------------+" << std::endl;
    std::cout << " +                         Program dieltools                       +" << std::endl;
    std::cout << " +-----------------------------------------------------------------+" << std::endl;
    diel_timer::print_current_time("     PROGRAM DIELTOOLS STARTED AT = "); // print current time
    // test for rdkit
    std::shared_ptr<RDKit::ROMol> mol1(RDKit::SmilesToMol("Cc1ccccc1"));  //分子の構築
    std::cout << "Number of atoms " << mol1->getNumAtoms() << std::endl;  //分子から情報所得
    
    // read argv and try to open input files.
    if (argc < 2) {
        error::exit("main", "Error: incorrect inputs. Usage:: dieltools inpfile");
    }
    // 
    // constant class
    // constant const;

    clock_t start = clock();    // start time
    std::chrono::system_clock::time_point  start_c, end_c; 
    start_c = std::chrono::system_clock::now(); // start time

    // stop watch class (caution: make instance after start time)
    // https://takap-tech.com/entry/2019/05/13/235416
    auto sw1 = diagnostics::Stopwatch::startNew();
    auto sw_total = diagnostics::Stopwatch::startNew(); // for total time
    sw_total->start(); // start total time
    // // 結果を取得
    // // std::cout << "Elapsed(nano sec) = " << sw1->getElapsedNanoseconds() << std::endl;
    // // std::cout << "Elapsed(milli sec) = " << sw1->getElapsedMilliseconds() << std::endl;
    // std::cout << " Elapsed(sec) = " << sw1->getElapsedSeconds() << std::endl;
    // // sw1->reset(); // リセットして計測を再開
    // // sw1->start();

    // necessary variables
    Atomicnum atomicnum;

    // read input (argv[1]=inputfilename)
    module_input::load_input module_load_input(argv[1],sw1);
    auto var_gen = module_load_input.var_gen;
    auto var_des = module_load_input.var_des;
    auto var_pre = module_load_input.var_pre;

    // read bondinfo
    // load bond
    module_bond::load_bond module_load_bond(var_gen.bondfilename,sw1);
    read_mol test_read_mol = module_load_bond.bondinfo;
    int NUM_MOL_ATOMS  = module_load_bond.NUM_MOL_ATOMS;

    // read xyz
    // Before loading the whole file, we only read the first frame and check consistency with the bondfile
    std::cout << " checking xyz file consistency with bond files ... " << std::endl;
    Atoms test_frame = read_frame(var_des.xyzfilename, 0);
    for (int i=0; i< int(test_read_mol.atom_list.size());i++){
        if (test_frame.get_atomic_numbers()[i] != atomicnum.atomicnum.at(test_read_mol.atom_list[i])){
            std::cout << " ERROR :: ATOMIC ARRANGEMENT NOT CONSISTENT" << std::endl;
            return 1;
        };
    };
    std::cout << " PASS :: atomic arrangement" << std::endl;

    // check len(test_frame)/NUM_MOL_ATOMS == int
    if (test_frame.get_atomic_numbers().size()%NUM_MOL_ATOMS != 0){
        std::cout << " ERROR :: ATOMIC NUMBER NOT CONSISTENT" << std::endl;
        return 1;
    } else{
        std::cout << " PASS :: atomic numbers" << std::endl;
    };


    // For longer trajectories, we first load bondfile then xyz.
    module_xyz::load_xyz module_load_xyz(var_des.xyzfilename, sw1);


    //!! NUM_MOL requires both xyz&bondinfo
    std::cout << " calculate NUM_MOL..." << std::endl;
    if (module_load_xyz.NUM_ATOM % NUM_MOL_ATOMS != 0){ // NUM_ATOM should be multiple of NUM_MOL_ATOMS
        std::cout << " ERROR :: NUM_ATOM is not multiple of NUM_MOL_ATOMS" << std::endl;
        return 1;
    }
    int NUM_MOL = int(module_load_xyz.NUM_ATOM/NUM_MOL_ATOMS); // UnitCell中の総分子数
    int ORIGINAL_NUM_MOL = int(module_load_xyz.NUM_ATOM/NUM_MOL_ATOMS); // Unitcell中の総分子数（gasがある場合の元の値）
    int ORIGINAL_NUM_CONFIG = module_load_xyz.NUM_CONFIG; // 元のNUM_CONFIGを保存しておく．
    std::cout << std::setw(10) << "NUM_MOL :: " << NUM_MOL << std::endl;
    std::cout << " OK !! " << std::endl;


    //! For gas model, we make xyz for a single molecule.
    if (var_des.IF_GAS){
        std::cout << " <<<<<<<<<<<<<<<<<<<<<<<<<<<<< " << std::endl;
        std::cout << " Invoke gas model calculation " << std::endl;
        std::cout << " <<<<<<<<<<<<<<<<<<<<<<<<<<<<< " << std::endl;
        // TODO :: ここはポインタ渡しにしておけば変数の変更は不要．
        std::vector<Atoms> atoms_list2 = ase_io_convert_1mol(module_load_xyz.atoms_list, NUM_MOL_ATOMS);
        module_load_xyz.atoms_list.clear();
        module_load_xyz.atoms_list = atoms_list2; // 変数を代入し直す必要がある．
        atoms_list2.clear();
        NUM_MOL = 1; // 分子数の更新
        std::cout << "len(atoms_list) :: " << module_load_xyz.atoms_list.size() << std::endl;
        module_load_xyz.NUM_CONFIG = module_load_xyz.atoms_list.size(); //NUM_CONFIGも更新
    };

    //!! read torch ML models
    module_torch::load_models module_load_models(var_pre.model_dir, sw1);


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
    std::vector<Eigen::Vector3d> result_dipole_list(module_load_xyz.NUM_CONFIG);

    // 予め出力する分子の双極子用のリストを確保しておく
    std::vector<std::vector<Eigen::Vector3d> > result_molecule_dipole_list(module_load_xyz.NUM_CONFIG, std::vector<Eigen::Vector3d>(NUM_MOL, Eigen::Vector3d::Zero()));

    // 予め出力するワニエセンターの座標用のリストを確保しておく．
    // !! 
    std::vector<Atoms> result_atoms_list(module_load_xyz.NUM_CONFIG);

    // 予めSAVE_TRUEYで保存するbond dipole用のリストを確保しておく（frame, num_bonds, 3d vector)
    std::vector<std::vector<Eigen::Vector3d> > result_ch_dipole_list(module_load_xyz.NUM_CONFIG);
    std::vector<std::vector<Eigen::Vector3d> > result_co_dipole_list(module_load_xyz.NUM_CONFIG);
    std::vector<std::vector<Eigen::Vector3d> > result_cc_dipole_list(module_load_xyz.NUM_CONFIG);
    std::vector<std::vector<Eigen::Vector3d> > result_oh_dipole_list(module_load_xyz.NUM_CONFIG);
    std::vector<std::vector<Eigen::Vector3d> > result_o_dipole_list(module_load_xyz.NUM_CONFIG);
    std::vector<std::vector<Eigen::Vector3d> > result_coc_dipole_list(module_load_xyz.NUM_CONFIG);
    std::vector<std::vector<Eigen::Vector3d> > result_coh_dipole_list(module_load_xyz.NUM_CONFIG);

    // STDOUT for checking if calculations work well 
    std::ofstream fout_stdout("STDOUT"); 
    fout_stdout << "calculated dipole at selected frames" << std::endl;

    // https://codezine.jp/article/detail/4786
    #pragma omp parallel for
    for (int i=0;i<1;i++){
        std::cout << " ************************** OPEMMP *************************** " << std::endl;
        std::cout << "   OMP information (num threads) :: " << omp_get_num_threads() << std::endl;
        std::cout << "   OMP information (max threads) :: " << omp_get_max_threads() << std::endl;
        std::cout << "   structure / parallel          :: " << module_load_xyz.NUM_CONFIG/omp_get_num_threads() << std::endl;
    };


    std::cout << "" << std::endl;
    std::cout << " ------------------------------------" << std::endl;
    std::cout << " start calculate descriptor&prediction !!" << std::endl;
    std::cout << " " << std::endl;
    sw1->start(); // timer for prediction
    // #pragma omp parallel for 
    // #pragma omp parallel
    for (int i=0; i< module_load_xyz.NUM_CONFIG; i++){ // ここは他のfor文のような構文にはできない(ompの影響．)
        Eigen::Vector3d TotalDipole = Eigen::Vector3d::Zero(); // ! Total dipole of the frame
        std::vector<Eigen::Vector3d> MoleculeDipoleList(NUM_MOL, Eigen::Vector3d::Zero()); // ! Molecular dipole list of the frame
        // ! true_yを保存するためのやつ．（coc,cohのみなのは理由があるのか？）
        std::vector<Eigen::Vector3d> true_y_list_coc;
        std::vector<Eigen::Vector3d> true_y_list_coh;

        // pbc-molをかけた原子座標(test_mol)と，それを利用したbcを取得
        auto test_mol_bc = raw_aseatom_to_mol_coord_and_bc(
            module_load_xyz.atoms_list[i], 
            test_read_mol.bonds_list, 
            test_read_mol, 
            NUM_MOL_ATOMS, NUM_MOL); // OMP parallel region
        std::vector<std::vector<Eigen::Vector3d> > test_mol=std::get<0>(test_mol_bc);
        std::vector<std::vector<Eigen::Vector3d> > test_bc =std::get<1>(test_mol_bc);

        // Define bonds （第一引数は系全体の各種ボンドの数を表す）
        dipole_frame ch_dipole_frame   = dipole_frame(NUM_MOL*test_read_mol.ch_bond_index.size(), NUM_MOL);
        dipole_frame cc_dipole_frame   = dipole_frame(NUM_MOL*test_read_mol.cc_bond_index.size(), NUM_MOL);
        dipole_frame co_dipole_frame   = dipole_frame(NUM_MOL*test_read_mol.co_bond_index.size(), NUM_MOL);
        dipole_frame oh_dipole_frame   = dipole_frame(NUM_MOL*test_read_mol.oh_bond_index.size(), NUM_MOL);
        dipole_frame o_dipole_frame    = dipole_frame(NUM_MOL*test_read_mol.o_list.size(), NUM_MOL);
        dipole_frame coh_dipole_frame  = dipole_frame(NUM_MOL*test_read_mol.coh_list.size(), NUM_MOL); // for coh/coc
        dipole_frame coc_dipole_frame  = dipole_frame(NUM_MOL*test_read_mol.coc_list.size(), NUM_MOL); // for coh/coc

        // TODO :: make class for 1 frame calculation
        // Currently, we sequencially calculate bond dipole for each bond type.
        // This is not efficient in terms of parallel calculations, and we should make a class for 1 frame.
        //   0: calculate bond centers
        //   1: calc descriptor&predict
        //      1-1: loop for bond centers
        //      1-2: check bond type
        //      1-3: calculate descriptor (PARALLEL 2)
        //! calc CH bond dipole
        if (module_load_models.IF_CALC_CH){
            ch_dipole_frame.predict_bond_dipole_at_frame(module_load_xyz.atoms_list[i], test_bc, test_read_mol.ch_bond_index, NUM_MOL, module_load_xyz.UNITCELL_VECTORS,  NUM_MOL_ATOMS, var_des.desctype, module_load_models.module_ch);
            ch_dipole_frame.calculate_wannier_list(test_bc, test_read_mol.ch_bond_index);
            ch_dipole_frame.calculate_moldipole_list();
            // ! calc ch_dipole_list
            result_ch_dipole_list[i] = ch_dipole_frame.dipole_list;
            // * calc total dipole (add each bond dipole)
            for (int p = 0; p< (int) ch_dipole_frame.dipole_list.size(); p++){
                TotalDipole += ch_dipole_frame.dipole_list[p];
            }
            // * calc molecular dipole (add each bond dipole)
            for (int p=0; p<NUM_MOL; p++){
                MoleculeDipoleList[p] += ch_dipole_frame.MoleculeDipoleList[p];
            };
        } //! end if module_load_models.IF_CALC_CH

        //! calc CC bond dipole
        if (module_load_models.IF_CALC_CC){
            cc_dipole_frame.predict_bond_dipole_at_frame(module_load_xyz.atoms_list[i], test_bc, test_read_mol.cc_bond_index, NUM_MOL, module_load_xyz.UNITCELL_VECTORS,  NUM_MOL_ATOMS, var_des.desctype, module_load_models.module_cc);
            cc_dipole_frame.calculate_wannier_list(test_bc, test_read_mol.cc_bond_index);
            cc_dipole_frame.calculate_moldipole_list();
            // ! calc cc_dipole_list
            result_cc_dipole_list[i] = cc_dipole_frame.dipole_list;
            // * total dipole
            for (int p = 0, q=cc_dipole_frame.dipole_list.size(); p < q; p++){
                TotalDipole += cc_dipole_frame.dipole_list[p];
            };
            // * molecular dipole
            for (int p=0; p<NUM_MOL; p++){
                MoleculeDipoleList[p] += cc_dipole_frame.MoleculeDipoleList[p];
            };
        } //! END_IF module_load_models.IF_CALC_CC

        //! calc CO bond dipole
        if (module_load_models.IF_CALC_CO){
            co_dipole_frame.predict_bond_dipole_at_frame(module_load_xyz.atoms_list[i], test_bc, test_read_mol.co_bond_index, NUM_MOL, module_load_xyz.UNITCELL_VECTORS,  NUM_MOL_ATOMS, var_des.desctype, module_load_models.module_co);
            co_dipole_frame.calculate_wannier_list(test_bc, test_read_mol.co_bond_index);
            co_dipole_frame.calculate_moldipole_list();
            // ! calc co_dipole_list
            result_co_dipole_list[i] = co_dipole_frame.dipole_list;
            // * total dipole
            for (int p = 0, q=co_dipole_frame.dipole_list.size(); p < q; p++){
                TotalDipole += co_dipole_frame.dipole_list[p];
            };
            // * molecular dipole
            for (int p=0; p<NUM_MOL; p++){
                MoleculeDipoleList[p] += co_dipole_frame.MoleculeDipoleList[p];
            };
        }; //! END_IF module_load_models.IF_CALC_CO

        //! calc OH bond dipole
        if (module_load_models.IF_CALC_OH){
            oh_dipole_frame.predict_bond_dipole_at_frame(module_load_xyz.atoms_list[i], test_bc, test_read_mol.oh_bond_index, NUM_MOL, module_load_xyz.UNITCELL_VECTORS,  NUM_MOL_ATOMS, var_des.desctype, module_load_models.module_oh);
            oh_dipole_frame.calculate_wannier_list(test_bc, test_read_mol.oh_bond_index);
            oh_dipole_frame.calculate_moldipole_list();
            // ! oh_dipole_listへの代入
            result_oh_dipole_list[i] = oh_dipole_frame.dipole_list;
            // * total dipole
            for (int p = 0,q = oh_dipole_frame.dipole_list.size(); p < q; p++){
                TotalDipole += oh_dipole_frame.dipole_list[p];
            };
            // * molecular dipole
            for (int p=0; p<NUM_MOL; p++){
                MoleculeDipoleList[p] += oh_dipole_frame.MoleculeDipoleList[p];
            };
        }; //! END_IF module_load_models.IF_CALC_OH

        //! calc O bond dipole
        if (module_load_models.IF_CALC_O){
            o_dipole_frame.predict_lonepair_dipole_at_frame(module_load_xyz.atoms_list[i], test_mol, test_read_mol.o_list, NUM_MOL, module_load_xyz.UNITCELL_VECTORS, NUM_MOL_ATOMS, var_des.desctype, module_load_models.module_o);
            o_dipole_frame.calculate_lonepair_wannier_list(test_mol, test_read_mol.o_list); //test_molを指定しないとちゃんと動かないので注意！！
            o_dipole_frame.calculate_moldipole_list();
            // ! o_dipole_list
            result_o_dipole_list[i] = o_dipole_frame.dipole_list;
            // * total dipole
            for (int p = 0, q = o_dipole_frame.dipole_list.size(); p < q; p++){
                TotalDipole += o_dipole_frame.dipole_list[p];
            };
            // * molecular dipole
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
        } //! END_IF module_load_models.IF_CALC_O

        //! calc COC bond dipole
        if (module_load_models.IF_CALC_COC){
            coc_dipole_frame.predict_lonepair_dipole_select_at_frame(module_load_xyz.atoms_list[i], test_mol, test_read_mol.coc_list, NUM_MOL, module_load_xyz.UNITCELL_VECTORS, NUM_MOL_ATOMS, var_des.desctype, module_load_models.module_coc);
            coc_dipole_frame.calculate_lonepair_wannier_list(test_mol, test_read_mol.coc_list); //test_molを指定しないとちゃんと動かないので注意！！
            coc_dipole_frame.calculate_moldipole_list();
            // ! o_dipole_list
            result_coc_dipole_list[i] = coc_dipole_frame.dipole_list;
            // * total dipole
            for (int p = 0, q = coc_dipole_frame.dipole_list.size(); p < q; p++){
                TotalDipole += coc_dipole_frame.dipole_list[p];
            };
            // * molecular dipole
            for (int p=0; p<NUM_MOL; p++){
                MoleculeDipoleList[p] += coc_dipole_frame.MoleculeDipoleList[p];
            };
            // !
            // for (int p=0;p<tmp_o_dipole_list.size();p++){
            //     std::cout << (tmp_o_dipole_list[p]-o_dipole_frame.dipole_list[p]).norm() << std::endl;
            //     if ((tmp_o_dipole_list[p]-o_dipole_frame.dipole_list[p]).norm()>0.0001){
            //         std::cout << "WARNING :: tmp_o_dipole " << std::endl;
            //     };
            // }
        } //! END_IF module_load_models.IF_CALC_COC

        //! calc COH bond dipole
        if (module_load_models.IF_CALC_COH){
            coh_dipole_frame.predict_lonepair_dipole_select_at_frame(module_load_xyz.atoms_list[i], test_mol, test_read_mol.coh_list, NUM_MOL, module_load_xyz.UNITCELL_VECTORS, NUM_MOL_ATOMS, var_des.desctype, module_load_models.module_coh);
            coh_dipole_frame.calculate_lonepair_wannier_list(test_mol, test_read_mol.coh_list); //test_molを指定しないとちゃんと動かないので注意！！
            coh_dipole_frame.calculate_moldipole_list();
            // ! coh_dipole_list
            result_coh_dipole_list[i] = coh_dipole_frame.dipole_list;
            // * total dipole
            for (int p = 0, q = coh_dipole_frame.dipole_list.size(); p < q; p++){
                TotalDipole += coh_dipole_frame.dipole_list[p];
            };
            // * molecular dipole
            for (int p=0; p<NUM_MOL; p++){
                MoleculeDipoleList[p] += coh_dipole_frame.MoleculeDipoleList[p];
            };
            // !
            // for (int p=0;p<tmp_o_dipole_list.size();p++){
            //     std::cout << (tmp_o_dipole_list[p]-o_dipole_frame.dipole_list[p]).norm() << std::endl;
            //     if ((tmp_o_dipole_list[p]-o_dipole_frame.dipole_list[p]).norm()>0.0001){
            //         std::cout << "WARNING :: tmp_o_dipole " << std::endl;
            //     };
            // }
        } //! END_IF module_load_models.IF_CALC_COH


        // ! >>>>>>>>>>>>>>>
        // ! Finalize 1 frame calculation
        // ! >>>>>>>>>>>>>>>
        if (omp_get_thread_num() == 1){ // output results in thread=1 to STDOUT file
            fout_stdout << "TotalDipole :: " << i << " " << TotalDipole[0] << " "  << TotalDipole[1] << " "  << TotalDipole[2] << " " << std::endl;
        }
        // frameごとのtotal dipoleに代入
        result_dipole_list[i]=TotalDipole;

        // 計算された分子ごとの双極子をresult_molecule_dipole_listリストへ格納
        for (int j=0; j<NUM_MOL; j++){ // i:frame数，j:分子数
            result_molecule_dipole_list[i][j]=MoleculeDipoleList[j];
        }

        // TODO :: move to test
        // // !! DEBUG :: moleculedipoleとtotaldipoleが一致するか？
        // Eigen::Vector3d tmp_totaldipole = Eigen::Vector3d::Zero();
        // for (int j=0; j<NUM_MOL; j++){ // i:frame数，j:分子数
        //     tmp_totaldipole += MoleculeDipoleList[j];
        // }
        // std::cout << " check mol dipole vs total dipole " <<  tmp_totaldipole[0]-TotalDipole[0] << " " << tmp_totaldipole[1]-TotalDipole[1] << " " << tmp_totaldipole[2]-TotalDipole[2] << std::endl;
        // if ((tmp_totaldipole-TotalDipole).norm()>0.0001){
        //     std::cout << "WARNING :: tmp_totaldipole is not equal to TotalDipole " << std::endl;
        // };

        // TODO :: ここを関数にしたい．
        // TODO :: その際，できればtest_molとtest_bcをまとめて1フレームでの情報をもつclassを作成する．
        // TODO :: そのクラスの一つの関数として以下のmake_aseを実装しておく．（もちろんraw versionも欲しい）
        // 計算されたbond centerとwannier centersをase atomsへ格納する．
        // 分子ごとにpushbackするので，ここでまとめて実行する必要がある．
        // WCsは，CH/CC/CO/OH/Oの順番
        std::vector < Eigen::Vector3d > atoms_with_bc; // これを使う
        std::vector < int >             new_atomic_num; // これを使う
        std::vector < int >  atomic_numbers = module_load_xyz.atoms_list[i].get_atomic_numbers();
        for (int a=0; a< NUM_MOL; a++){ // 分子に関するループ
            for (int b=0; b< (int) test_mol[a].size();b++){ //原子座標
                atoms_with_bc.push_back(test_mol[a][b]);
                new_atomic_num.push_back(atomic_numbers[a*NUM_MOL_ATOMS+b]); //原子に対応するatoms_listの原子種
                // atoms_with_bc_index.push_back(test_mol[a][b])
            }
            for (int b=0; b< (int) test_bc[a].size();b++){ //ボンドセンター
                atoms_with_bc.push_back(test_bc[a][b]);
                new_atomic_num.push_back(2); // ボンドセンターには原子番号2を割り当て

            }
            if (module_load_models.IF_CALC_CH){
                for (int b=0; b< (int) ch_dipole_frame.wannier_list[a].size();b++){ //ch wannier
                    atoms_with_bc.push_back(ch_dipole_frame.wannier_list[a][b]);
                    new_atomic_num.push_back(0); // WC@CHには原子番号100を割り当て
                }
            }
            if (module_load_models.IF_CALC_CC){
                for (int b=0; b< (int) cc_dipole_frame.wannier_list[a].size();b++){ //ch wannier
                    atoms_with_bc.push_back(cc_dipole_frame.wannier_list[a][b]);
                    new_atomic_num.push_back(102);// WC@CCには原子番号102を割り当て
                }
            }
            if (module_load_models.IF_CALC_CO){
                for (int b=0; b< (int) co_dipole_frame.wannier_list[a].size();b++){ //ch wannier
                    atoms_with_bc.push_back(co_dipole_frame.wannier_list[a][b]);
                    new_atomic_num.push_back(101);// WC@COには原子番号101を割り当て
                }
            }
            if (module_load_models.IF_CALC_OH){
                for (int b=0; b< (int) oh_dipole_frame.wannier_list[a].size();b++){ //ch wannier
                    atoms_with_bc.push_back(oh_dipole_frame.wannier_list[a][b]);
                    new_atomic_num.push_back(103); // WC@OHには原子番号101を割り当て
                }
            }
            if (module_load_models.IF_CALC_O){
                for (int b=0; b< (int) o_dipole_frame.wannier_list[a].size();b++){ //ch wannier
                    atoms_with_bc.push_back(o_dipole_frame.wannier_list[a][b]);
                    new_atomic_num.push_back(10); // O(WC)には原子番号10を割り当て
                }
            }
        }
        Atoms tmp_atoms = Atoms(
            new_atomic_num,
            atoms_with_bc,
            module_load_xyz.UNITCELL_VECTORS,
            {true,true,true});
        // Atoms testtest = Atoms(atoms_list[i].get_atomic_numbers(), atoms_list[i].get_positions(), UNITCELL_VECTORS, {1,1,1});
        result_atoms_list[i] = tmp_atoms;
        new_atomic_num.clear(); // vectorのクリア
	    atoms_with_bc.clear();
    }
    std::cout << " finish calculate descriptor&prediction !!" << std::endl;
    sw1->stop(); // stop timer    
    std::cout << "     ELAPSED TIME :: predict (chrono)      = " << sw1->getElapsedSeconds() << std::endl;
    sw1->reset(); // reset timer 
    std::cout << " " << std::endl;


    //! gasモデル計算の場合，11分子ごとのxyzを作成する
    // 変数の問題があって，一旦別の変数に渡したあと，元の変数をクリアして再代入する．
    if (var_des.IF_GAS){
        std::cout << " ************************** CONVERT TO LIQUID (IF_GAS) *************************** " << std::endl;
        std::cout << " Back convert to Liquid ... " << std::endl;
        std::cout << " convert total_dipole ... " << std::endl;
        auto result_dipole_list_tmp     = convert_total_dipole(result_dipole_list,    ORIGINAL_NUM_CONFIG, ORIGINAL_NUM_MOL);
        std::cout << " convert bond dipole ... " << std::endl;
        auto result_ch_dipole_list_tmp  = convert_bond_dipole(result_ch_dipole_list,  ORIGINAL_NUM_CONFIG, ORIGINAL_NUM_MOL);
        auto result_co_dipole_list_tmp  = convert_bond_dipole(result_co_dipole_list,  ORIGINAL_NUM_CONFIG, ORIGINAL_NUM_MOL);
        auto result_oh_dipole_list_tmp  = convert_bond_dipole(result_oh_dipole_list,  ORIGINAL_NUM_CONFIG, ORIGINAL_NUM_MOL);
        auto result_cc_dipole_list_tmp  = convert_bond_dipole(result_cc_dipole_list,  ORIGINAL_NUM_CONFIG, ORIGINAL_NUM_MOL);
        auto result_o_dipole_list_tmp   = convert_bond_dipole(result_o_dipole_list,   ORIGINAL_NUM_CONFIG, ORIGINAL_NUM_MOL);
        auto result_coc_dipole_list_tmp = convert_bond_dipole(result_coc_dipole_list, ORIGINAL_NUM_CONFIG, ORIGINAL_NUM_MOL);
        auto result_coh_dipole_list_tmp = convert_bond_dipole(result_coh_dipole_list, ORIGINAL_NUM_CONFIG, ORIGINAL_NUM_MOL);
        std::cout << " convert molecule dipole ... " << std::endl;
        auto result_molecule_dipole_list_tmp =  convert_bond_dipole(result_molecule_dipole_list,ORIGINAL_NUM_CONFIG, ORIGINAL_NUM_MOL);
        result_dipole_list.clear();
        result_ch_dipole_list.clear();
        result_co_dipole_list.clear();
        result_oh_dipole_list.clear();
        result_cc_dipole_list.clear();
        result_o_dipole_list.clear();
        result_coc_dipole_list.clear();
        result_coh_dipole_list.clear();
        result_molecule_dipole_list.clear();
        result_dipole_list.resize(ORIGINAL_NUM_CONFIG);
        result_ch_dipole_list.resize(ORIGINAL_NUM_CONFIG); // ここらへんはちょっと怪しくないか？ [frame,num_mol,3]の形になっているはずなのに．
        result_co_dipole_list.resize(ORIGINAL_NUM_CONFIG);
        result_oh_dipole_list.resize(ORIGINAL_NUM_CONFIG);
        result_cc_dipole_list.resize(ORIGINAL_NUM_CONFIG);
        result_o_dipole_list.resize(ORIGINAL_NUM_CONFIG);
        result_coc_dipole_list.resize(ORIGINAL_NUM_CONFIG);
        result_coh_dipole_list.resize(ORIGINAL_NUM_CONFIG);
        result_molecule_dipole_list.resize(ORIGINAL_NUM_CONFIG);
        result_dipole_list     = result_dipole_list_tmp;
        result_ch_dipole_list  = result_ch_dipole_list_tmp;
        result_co_dipole_list  = result_co_dipole_list_tmp;
        result_oh_dipole_list  = result_oh_dipole_list_tmp;
        result_cc_dipole_list  = result_cc_dipole_list_tmp;
        result_o_dipole_list   = result_o_dipole_list_tmp;
        result_coc_dipole_list = result_coc_dipole_list_tmp;
        result_coh_dipole_list = result_coh_dipole_list_tmp;
        result_molecule_dipole_list = result_molecule_dipole_list_tmp;
        std::cout << " Finish conversion !! " << std::endl;
    }


    std::cout << " ************************** POST PROCESS *************************** " << std::endl;
    std::cout << " Calculate mean molecular dipole & dielectric constant..." << std::endl;
    postprocess_dielconst(result_dipole_list,result_molecule_dipole_list, var_gen.temperature, module_load_xyz.UNITCELL_VECTORS,var_gen.savedir);

    //!! IF_COC/COHがfalseの場合，co,o,ohボンド双極子の計算から新しくCOC/COH双極子を計算する
    // TODO :: 変数の受け渡しがイマイチ綺麗でないので，もう少し良い方法を考えたい．
    // 情報としては，co,oh,oのdipole_listがあれば良い．frameごとに計算するので，並列化して計算する．
    std::cout << "" << std::endl;
    std::cout << "" << std::endl;
    if (!(module_load_models.IF_CALC_COH)){
        std::cout << " INVOKE POST PROCESS COH calculation !!" << std::endl;
#pragma omp parallel for 
        for (int i=0; i< (int) module_load_xyz.atoms_list.size(); i++){//フレームに関する並列化
            // coh_dipole_frame.dipole_list
            dipole_frame coh_dipole_frame  = dipole_frame(NUM_MOL*test_read_mol.coh_list.size(), NUM_MOL); // coh/coc用
            // TODO :: 注意!! ここは，CO，OHという順番でないといけない．coh_bond_info2もその順番を守っている．
            // TODO :: このようなコーディングはバグの温床になるのでやめないといけない．
            coh_dipole_frame.calculate_coh_bond_dipole_at_frame(test_read_mol.coh_bond_info2, result_o_dipole_list[i], result_co_dipole_list[i], result_oh_dipole_list[i]);
            result_coh_dipole_list[i] = coh_dipole_frame.dipole_list;
        }
    };
    if (!(module_load_models.IF_CALC_COC)){
        std::cout << " INVOKE POST PROCESS COC calculation !!" << std::endl;
#pragma omp parallel for 
        for (int i=0; i< (int) module_load_xyz.atoms_list.size(); i++){ //フレームに関する並列化
            // coh_dipole_frame.dipole_list
            dipole_frame coc_dipole_frame  = dipole_frame(NUM_MOL*test_read_mol.coc_list.size(), NUM_MOL); // coh/coc用
            coc_dipole_frame.calculate_coh_bond_dipole_at_frame(test_read_mol.coc_bond_info2, result_o_dipole_list[i], result_co_dipole_list[i], result_co_dipole_list[i]);
            result_coc_dipole_list[i] = coc_dipole_frame.dipole_list;
        }
    };


    //    dipole_frame coc_dipole_frame  = dipole_frame(NUM_MOL*test_read_mol.coc_list.size(), NUM_MOL); // coh/coc用



    std::cout << " ************************** SAVE DATA *************************** " << std::endl;
    std::cout << "  finished all calculations, now saving data..." << std::endl;
    sw1->start(); // 

    // save total dipole
    save_totaldipole(result_dipole_list, module_load_xyz.UNITCELL_VECTORS, var_gen.temperature, var_gen.timestep, var_gen.savedir);

    // save molecular dipoles
    postprocess_save_moleculedipole(
    result_molecule_dipole_list,
    module_load_xyz.UNITCELL_VECTORS, var_gen.temperature, var_gen.timestep,
    var_gen.savedir);

    // save bond dipoles
    if (var_gen.save_bonddipole == true){
        postprocess_save_bonddipole(
        result_ch_dipole_list,
        result_co_dipole_list,
        result_oh_dipole_list,
        result_cc_dipole_list,
        result_o_dipole_list,
        result_coc_dipole_list,
        result_coh_dipole_list,
        module_load_xyz.UNITCELL_VECTORS, var_gen.temperature, var_gen.timestep,
        var_gen.savedir);
    }

    // 最終的な結果をxyzに保存する．
    ase_io_write(result_atoms_list, var_gen.savedir+"/mol_wan.xyz");

    // stdoutを閉じる
    fout_stdout.close();
    // save time
    std::cout << "     ELAPSED TIME :: save data (chrono)      = " << sw1->getElapsedSeconds() << std::endl;
    sw1->reset(); // reset timer
    std::cout << " " << std::endl;

    // 時間計測関係
    clock_t end = clock();     // 終了時間
    end_c = std::chrono::system_clock::now();  // 計測終了時間
    double elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_c-start_c).count();

    // std::time_t end_time = std::chrono::system_clock::to_time_t(end_c);
    std::cout << "  ********************************************************************************" << std::endl;
    std::cout << "     CPU TIME (clock)           = " << (double)(end - start) / CLOCKS_PER_SEC << "sec." << std::endl;
    std::cout << "     ELAPSED TIME (chrono)      = " << sw_total->getElapsedSeconds() << "sec." << std::endl;
    // std::cout << "     ELAPSED TIME (chrono)      = " << elapsed << "sec." << std::endl;
    diel_timer::print_current_time("     PROGRAM DIELTOOLS ENDED AT = "); // print current time
    std::cout << "finish !! " << std::endl;

    // delete instances
    // delete sw1;
    // delete sw_total;

    return 0;
}
