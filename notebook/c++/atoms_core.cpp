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
#include <Eigen/Core> // 行列演算など基本的な機能．
#include "numpy.hpp"
#include "npy.hpp"
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
        Atomicnum(){
            atomicnum["C"] = 6;
            atomicnum["H"] = 1;
            atomicnum["O"] = 8;
            atomicnum["He"] = 2;
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
  // variables
  std::vector<int> atomic_num;
  std::vector<Eigen::Vector3d> positions; // !! ここEigenを利用．
  std::vector<std::vector<double> > cell;
  std::vector<bool> pbc;

  // member functions
  std::vector<int> get_atomic_numbers(); // atomic_numを返す
  std::vector<Eigen::Vector3d> get_positions(); // positionsを返す
  std::vector<std::vector<double> > get_cell(); // cellを返す
  
  // constructor
  int number;
  Atoms(std::vector<int> atomic_numbers,
        std::vector<Eigen::Vector3d> atomic_positions,
        std::vector<std::vector<double> > UNITCELL_VECTORS,
        std::vector<bool> pbc_cell) 
        {
            // https://www.freecodecamp.org/news/cpp-vector-how-to-initialize-a-vector-in-a-constructor/
            this->atomic_num = atomic_numbers;
            this->positions = atomic_positions;
            this->cell = UNITCELL_VECTORS;
            this->pbc = pbc_cell;
        };
};

std::vector<int> Atoms::get_atomic_numbers()// atomic_numを返す
  {
   return this->atomic_num;
  };

std::vector<Eigen::Vector3d> Atoms::get_positions()// positionsを返す
  {
   return this->positions;
  };

std::vector<std::vector<double> > Atoms::get_cell() // cellを返す
    {
      return this->cell;
    }





double sign(double A){
    // 実数Aの符号を返す（pbc計算で利用する．）
    // https://cvtech.cc/sign/
    return (A>0)-(A<0);
}

std::vector<Eigen::Vector3d> raw_get_distances_mic(Atoms aseatom, int a, std::vector<int> indices, bool mic=true, bool vector=false){
    /*
    ase.atomのget_distances関数(micあり)のc++実装版
    a: 求める原子のaseatomでの順番（index）
    TODO :: vector=Falseの場合は関数のオーバーロード（https://learn.microsoft.com/ja-jp/cpp/cpp/function-overloading?view=msvc-170）で対応しよう．
    */
    std::vector<Eigen::Vector3d > coordinate = aseatom.get_positions();
    Eigen::Vector3d reference_position = coordinate[a]; // TODO :: 慣れてきたら削除
    std::vector<Eigen::Vector3d> distance_without_mic(indices.size()); 
    // まずはcoordinate[a]からの相対ベクトルを計算する．
    for (int i = 0; i < indices.size(); i++) {
        // distance_without_mic.push_back(coordinate[indices[i]]-reference_position); // 座標にEigen::vectorを利用していることでベクトル減産が可能．
        distance_without_mic[i]=coordinate[indices[i]]-reference_position; // 座標にEigen::vectorを利用していることでベクトル減産が可能．
        // std::cout << "indices[i] = " << indices[i] << std::endl;
    }
    if (mic == true){ // mic = Trueの時だけmic再計算をする． 
        std::vector<std::vector<double> > cell = aseatom.get_cell(); // TODO :: 慣れてきたらポインタで取得する．
        double cell_x = cell[0][0]; // TODO :: 一般の格子に対応させる．
#ifdef _DEBUG
        std::cout << "CELL SIZE/2 :: " << cell_x/2 << std::endl;
#endif // !DEBUG
        for (int i = 0; i < distance_without_mic.size(); i++) {
            for (int cartesian_j = 0; cartesian_j <3; cartesian_j++){
                if (std::abs(distance_without_mic[i][cartesian_j]) > cell_x/2) { // TODO :: 各座標成分に対して独立にmicを実行する．これでOKか確認を！！
                    // distances = np.where(np.abs(distances) > cell/2, distances-cell*np.sign(distances),distances)のC++版
                    distance_without_mic[i][cartesian_j] = distance_without_mic[i][cartesian_j]-cell_x*sign(distance_without_mic[i][cartesian_j]);
                }
            }
        } 
    }
    return distance_without_mic;
    };