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
#include "npy.hpp"
#include "numpy.hpp"
#include "torch/script.h" // pytorch
// #include "numpy_quiita.hpp" // https://qiita.com/ka_na_ta_n/items/608c7df3128abbf39c89
// numpy_quiitaはsscanf_sが読み込めず，残念ながら現状使えない．
// #include "atoms_core.cpp" 
// #include "atoms_io.cpp"   
#include "atoms_asign_wcs.cpp"
// #include "mol_core.cpp"
// #include "parse.cpp"

// #include <GraphMol/GraphMol.h>
// #include <GraphMol/FileParsers/MolSupplier.h>
// #include <GraphMol/FileParsers/MolWriters.h>
// #include <GraphMol/FileParsers/FileParsers.h>


// https://e-penguiner.com/cpp-function-check-file-exist/#index_id2
bool IsFileExist(const std::string& name) {
    return std::filesystem::is_regular_file(name);
}


int main(int argc, char *argv[]) {

    if (argc < 3) {
        std::cout << "Error: xyz file does not provided." << std::endl;
        return 0;
    }

    std::string bond_filename=argv[1]; // bondファイル名を引数で指定．
    if (!IsFileExist(bond_filename)) {
        std::cout << "Error: bond file does not exist." << std::endl;
        return 0;
    }

    std::string xyz_filename=argv[2]; // bondファイル名を引数で指定．
    if (!IsFileExist(xyz_filename)) {
        std::cout << "Error: xyz file does not exist." << std::endl;
        return 0;
    }


    //! ボンドリストの取得
    // TODO :: 現状では，ボンドリストはmol_core.cpp内で定義されている．（こういうブラックボックスをなんとかしたい）
    // TODO :: 最悪でもボンドファイルはinput
    read_mol test_read_mol(bond_filename);

    //! 原子数の取得
    int NUM_ATOM = raw_cpmd_num_atom(xyz_filename);
    //! 格子定数の取得
    std::vector<std::vector<double> > UNITCELL_VECTORS = raw_cpmd_get_unitcell_xyz(xyz_filename);
    //! xyzファイルから座標リストを取得
    std::vector<Atoms> atoms_list = ase_io_read(xyz_filename);
    int NUM_MOL_ATOMS = test_read_mol.num_atoms_per_mol;
    int NUM_MOL = int(NUM_ATOM/NUM_MOL_ATOMS); // UnitCell中の総分子数

    // 1分子内のindexを取得する．
    std::vector<int> mol_at0(NUM_MOL_ATOMS);
    for (int i = 0; i < NUM_MOL_ATOMS; i++) {
        mol_at0[i] = i;
    }
    // 1config内の原子indexを取得する．
    std::vector<std::vector<int>> mol_ats(NUM_MOL, std::vector<int>(NUM_MOL_ATOMS));
    for (int indx = 0; indx < NUM_MOL; indx++) {
        for (int i = 0; i < NUM_MOL_ATOMS; i++) {
            mol_ats[indx][i] = i + NUM_MOL_ATOMS * indx;
        }
    }

    auto test_nodes = raw_make_graph_from_itp(test_read_mol);
    // bfsのテスト
    test_raw_bfs(atoms_list[0], mol_ats[0], test_read_mol);
    return 0;
}