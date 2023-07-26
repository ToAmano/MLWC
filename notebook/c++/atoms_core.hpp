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
#include <deque> // deque
// #include <boost/numeric/ublas/vector.hpp>
// #include <boost/numeric/ublas/matrix.hpp>
// #include <boost/numeric/ublas/io.hpp>
#include <Eigen/Core> // 行列演算など基本的な機能．
#include "numpy.hpp"
#include "npy.hpp"
#include "mol_core.cpp"
// #include "numpy_quiita.hpp" // https://qiita.com/ka_na_ta_n/items/608c7df3128abbf39c89
// numpy_quiitaはsscanf_sが読み込めず，残念ながら現状使えない．


/*
ase.Atomsに対応する基本的な関数のみを定義．これが全ての基本となる．

*/

/*
 2023/5/30
 ase atomsに対応するAtomsクラスを定義する

どうも自作クラスをvectorに入れる場合は特殊な操作が必要な模様．
https://nprogram.hatenablog.com/entry/2017/07/05/073922
*/


class Atomicnum{
    /*
    原子種と原子番号の対応を定義するクラス
    */
    public:
        std::map<std::string, int> atomicnum;
        Atomicnum(){};
};

class Atomicchar{
    /*
    原子種と原子番号の対応を定義するクラス2
    */
    public:
        std::map<int, std::string> atomicchar;
        Atomicchar(){}
};

class Atoms {
public: // public変数
  // variables
  std::vector<int> atomic_num;
  std::vector<Eigen::Vector3d> positions; // !! ここEigenを利用．
  std::vector<std::vector<double> > cell;
  std::vector<bool> pbc;

  // member functions
  std::vector<int> get_atomic_numbers() const; // atomic_numを返す
  std::vector<Eigen::Vector3d> get_positions() const; // positionsを返す
  std::vector<std::vector<double> > get_cell() const; // cellを返す
  
  // constructor
  int number;
  Atoms(std::vector<int> atomic_numbers,
        std::vector<Eigen::Vector3d> atomic_positions,
        std::vector<std::vector<double> > UNITCELL_VECTORS,
        std::vector<bool> pbc_cell) 
        {};
};


// 実数Aの符号を返す
double sign(double A){}

// aseのatomsのget_distancesとまったく同じ実装
std::vector<Eigen::Vector3d> raw_get_distances_mic(const Atoms &aseatom, int a, std::vector<int> indices, bool mic=true, bool vector=false){};

// aseのatomsのget_distancesとまったく同じ実装の引数が一つの場合
Eigen::Vector3d raw_get_distances_mic(const Atoms &aseatom, int a, int indice, bool mic=true, bool vector=false){};

// bfsを使って原子間距離を計算する．
std::vector<Eigen::Vector3d> raw_bfs(Atoms aseatom, std::vector<Node>& nodes, std::vector<Eigen::Vector3d> vectors, std::vector<int>& mol_inds, int representative = 0) {};

// raw_bfsとraw_get_distances_micが同じ結果を与えるかを確認する関数．
int test_raw_bfs(const Atoms &aseatoms, std::vector<int> mol_inds, const read_mol &itp_data){};

// 通常時にはraw_get_distancesを使い，特殊な場合にraw_bgsを使う．
std::vector<Eigen::Vector3d> raw_get_pbc_mol(const Atoms &aseatoms, std::vector<int> mol_inds, std::vector<std::vector<int>> bonds_list_j, const read_mol &itp_data) {};

// bfs自体のテスト
int raw_bfs_test(std::vector<Node>& nodes, int representative = 0){};
