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
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <Eigen/Core> // 行列演算など基本的な機能．
#include "numpy.hpp"
#include "npy.hpp"
// #include "numpy_quiita.hpp" // https://qiita.com/ka_na_ta_n/items/608c7df3128abbf39c89
// numpy_quiitaはsscanf_sが読み込めず，残念ながら現状使えない．


/*
ase.Atomsに対応する基本的な関数のみを定義．

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
        Atomicnum(){
            atomicnum["C"] = 6;
            atomicnum["H"] = 1;
            atomicnum["O"] = 8;
        };
};

class Atomicchar{
    /*
    原子種と原子番号の対応を定義するクラス2
    */
    public:
        std::map<int, std::string> atomicchar;
        Atomicchar(){
            atomicchar[6] = "C";
            atomicchar[1] = "H";
            atomicchar[8] = "O";
            atomicchar[2] = "He";
        };
};

class Atoms {
public: // public変数
  std::vector<int> atomic_num;
  std::vector<Eigen::Vector3d> positions; // !! ここEigenを利用．
  std::vector<std::vector<double> > cell;
  std::vector<bool> pbc;
  // std::vector<int> get_atomic_numbers(); // atomic_numを返す
  
  int number;
  Atoms(std::vector<int> atomic_numbers,
        std::vector<Eigen::Vector3d> atomic_positions,
        std::vector<std::vector<double> > UNITCELL_VECTORS,
        std::vector<bool> pbc_cell
        )
        {
            // https://www.freecodecamp.org/news/cpp-vector-how-to-initialize-a-vector-in-a-constructor/
            this->atomic_num = atomic_numbers;
            this->positions = atomic_positions;
            this->cell = UNITCELL_VECTORS;
            this->pbc = pbc_cell;
        };

  std::vector<int> get_atomic_numbers()// atomic_numを返す
  {
   return this->atomic_num;
  };
  std::vector<Eigen::Vector3d> get_positions()// positionsを返す
  {
   return this->positions;
  };
  std::vector<std::vector<double> > get_cell() // cellを返す
    {
      return this->cell;
    }
};