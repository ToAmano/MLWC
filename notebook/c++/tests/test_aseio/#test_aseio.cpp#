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
    std::cout << " +                         Program dieltools test                      +" << std::endl;
    std::cout << " +-----------------------------------------------------------------+" << std::endl;

    // 
    bool SAVE_DESCS = false; // trueならデスクリプターをnpyで保存．

    // std::string xyz_filename="/Users/amano/works/research/dieltools/notebook/c++/gromacs_trajectory_cell.xyz";
    // std::string xyz_filename="/Users/amano/works/research/dieltools/notebook/c++/gromacs_pg_1ns_dt50fs.xyz";
    // std::string xyz_filename="/Users/amano/works/research/dieltools/notebook/c++/gromacs_pg_1ns_dt50fs_300.xyz";

    // read argv and try to open input files.
    std::string xyzfilename="IONS+CENTERS.xyz";
    
    
    //! 原子数の取得
    std::cout << " Reading the xyz file  " << std::endl;
    int NUM_ATOM = raw_cpmd_num_atom(std::filesystem::absolute(xyzfilename));
    std::cout << std::setw(10) << "NUM_ATOM :: " << NUM_ATOM << std::endl;
    //! 格子定数の取得
    std::vector<std::vector<double> > UNITCELL_VECTORS = raw_cpmd_get_unitcell_xyz(std::filesystem::absolute(xyzfilename));
    std::cout << std::setw(10) << "UNITCELL_VECTORS :: " << UNITCELL_VECTORS[0][0] << std::endl;
    //! xyzファイルから座標リストを取得
    bool IF_WANNIER_REMOVE = true;
    std::vector<Atoms> atoms_list = ase_io_read(std::filesystem::absolute(xyzfilename), IF_WANNIER_REMOVE);
    std::cout << " finish reading xyz file :: " << atoms_list.size() << std::endl;

    //! 計算した座標リストを読み込み
    for (int j = 0, n = atoms_list[0].atomic_num.size(); j < n; j++) {
        std::cout << " atoms_list[" << 0 << "].positions :: " << atoms_list[0].atomic_num[j] << std::endl;
    };

    return 0;
}
