#ifndef ATOMS_CORE_H
#define ATOMS_CORE_H

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
// #include "atoms_core.hpp" // <>ではなく ""で囲う

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
	    atomicnum["X"]  = 100;
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
	    atomicchar[100] = "X";
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
  std::vector<int> get_atomic_numbers() const; // atomic_numを返す
  std::vector<Eigen::Vector3d> get_positions() const; // positionsを返す
  std::vector<std::vector<double> > get_cell() const; // cellを返す
  
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

std::vector<int> Atoms::get_atomic_numbers() const // atomic_numを返す
  {
   return this->atomic_num;
  };

std::vector<Eigen::Vector3d> Atoms::get_positions() const // positionsを返す
  {
   return this->positions;
  };

std::vector<std::vector<double> > Atoms::get_cell() const// cellを返す
    {
      return this->cell;
    }



double sign(double A){
    // 実数Aの符号を返す（pbc計算で利用する．）
    // https://cvtech.cc/sign/
    return (A>0)-(A<0);
}

std::vector<Eigen::Vector3d> raw_get_distances_mic(const Atoms &aseatom, int a, std::vector<int> indices, bool mic=true, bool vector=false){
    /*
    ase.atomのget_distances関数(micあり)のc++実装版
    a: 求める原子のaseatomでの順番（index）
    TODO :: vector=Falseの場合は出力がscalarになる．関数のオーバーロード（https://learn.microsoft.com/ja-jp/cpp/cpp/function-overloading?view=msvc-170）で対応しよう．
    TODO :: aseatomをconst修飾子+参照渡しにしてみたので，ちゃんと動くかどうかデバックを！！
    */
    std::vector<Eigen::Vector3d > coordinate = aseatom.get_positions();
    Eigen::Vector3d reference_position = coordinate[a]; // TODO :: 慣れてきたら削除
    std::vector<Eigen::Vector3d> distance_without_mic(indices.size()); 
    // まずはcoordinate[a]からの相対ベクトルを計算する．
    for (int i = 0, size=indices.size(); i < size; i++) {
        distance_without_mic[i]=coordinate[indices[i]]-reference_position; // 座標にEigen::vectorを利用していることでベクトル減産が可能．
        // std::cout << "indices[i] = " << indices[i] << std::endl;
    }
    if (mic == true){ // mic = Trueの時だけmic再計算をする． 
        std::vector<std::vector<double> > cell = aseatom.get_cell(); // TODO :: 慣れてきたらポインタで取得する．
        double cell_x = cell[0][0]; // TODO :: 一般の格子に対応させる．
#ifdef _DEBUG
        std::cout << "CELL SIZE/2 :: " << cell_x/2 << std::endl;
#endif // !DEBUG
        for (int i = 0, size=distance_without_mic.size(); i < size; i++) {
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

Eigen::Vector3d raw_get_distances_mic(const Atoms &aseatom, int a, int indice, bool mic=true, bool vector=false) {
    /*
    raw_get_distances_micのうち，3つ目の引数がvectorではなくintの場合のオーバーロード．
    このとき，単にraw_get_distances_micを返してしまうと，Eigen::Vector3dではなくstd::vector<Eigen::Vector3d > が帰ってしまうので，raw_get_distances_micの0番目の要素を返す．
    */
    std::vector<int> indices = {indice};
    return raw_get_distances_mic(aseatom, a, indices, mic, vector)[0]; 
};



std::vector<Eigen::Vector3d> raw_bfs(const Atoms &aseatom, std::vector<Node>& nodes, std::vector<Eigen::Vector3d> vectors, std::vector<int>& mol_inds, int representative = 0) {
    /*
    ase.atomのget_distances関数(micあり)のc++実装版．
    ! ただし，raw_get_distances_micでは引数がget_distancesと同じだったが，こちらは異なっていて，実装に特化した少し特殊な実装になっている．
    ! いったん通常のraw_get_distances_micでvectorsを計算したあと，vectorsの中に大きい要素があれば再度こちらの関数を利用して計算をやり直す．
    ! さらに，毎度bfs探索を行うので少し効率が悪いアルゴリズムになっている．
    raw_get_distances_micとはことなり，node情報を利用してmicを計算する．
    そのために，inputとしてnode情報のほか，mol_indsを受け取る．これでconfig内で自分がどこの分子に属しているかを判別する．
    mol_indsはrevised_vectorの計算のためだけに利用されるのでこれもあまり賢い変数の渡し方ではない．
    また，inputとしてvectorsも受け取る．
    a: 求める原子のaseatomでの順番（index）
    * @param[in] aseatom Atoms object to be analyzed.
    */
    std::deque<Node> queue;
    queue.push_back(nodes[representative]);
    nodes[representative].parent = 0;
    while (!queue.empty()) {
        Node node = queue.front();
        queue.pop_front();
        std::vector<int> nears = node.nears;
        for (int near : nears) {
            if (nodes[near].parent == -1) { // 親が-1の場合は未探索なので探索する
                queue.push_back(nodes[near]);
                nodes[near].parent = node.index;
                // node.indexとその隣接node(nodes[near])の距離ベクトルを計算する．
                // ! ここはvectorsに足していく感じで実行するので，実装がミスっていると悲惨なことになりやすいので注意．
                Eigen::Vector3d revised_vector = raw_get_distances_mic(aseatom, node.index + mol_inds[0], nodes[near].index + mol_inds[0], true, true);
                vectors[nodes[near].index] = vectors[node.index] + revised_vector;
            }
        }
    }
    return vectors;
}


std::vector<Eigen::Vector3d> raw_bfs(const Atoms &aseatom_in_molecule, std::vector<Node>& nodes, int representative = 0) {
    /*
    ase.atomのget_distances関数(micあり)のc++実装版．
    ! こちらは別の実装で，一つ目の実装ではaseatoms(系全体)をinputにしていたのに対して，こちらではそもそも分子内の原子だけを入力として受け取る．
    ! その結果，mol_indsは必要なくなっている．
    TODO :: これ，そもそも入力にvector要らないような気がするんだけどどうよ？
    TODO :: こっちの実装では入れずに作ってみたので後でデバックしてみよう．
    raw_get_distances_micとはことなり，node情報を利用してmicを計算する．
    そのために，inputとしてnode情報のほか，mol_indsを受け取る．これでconfig内で自分がどこの分子に属しているかを判別する．
    また，inputとしてvectorsも受け取る．
    a: 求める原子のaseatomでの順番（index）
    * @param[in] aseatom Atoms object to be analyzed.
    */
    // nodesのサイズとaseatomsのサイズが一緒じゃないとエラー
    if (nodes.size() != aseatom_in_molecule.get_atomic_numbers().size()) {
        std::cout << "ERROR :: nodes.size() != aseatoms_in_molecule.get_atomic_numbers().size()" << std::endl;
        exit(1);
    };
    std::vector<Eigen::Vector3d> vectors(nodes.size(), Eigen::Vector3d::Zero()); //ベクトルを0で初期化
    std::deque<Node> queue;
    queue.push_back(nodes[representative]);
    nodes[representative].parent = 0;
    while (!queue.empty()) {
        Node node = queue.front();
        queue.pop_front();
        std::vector<int> nears = node.nears;
        for (int near : nears) {
            if (nodes[near].parent == -1) { // 親が-1の場合は未探索なので探索する
                queue.push_back(nodes[near]);
                nodes[near].parent = node.index;
                // node.indexとその隣接node(nodes[near])の距離ベクトルを計算する．
                // ! ここはvectorsに足していく感じで実行するので，実装がミスっていると悲惨なことになりやすいので注意．
                Eigen::Vector3d revised_vector = raw_get_distances_mic(aseatom_in_molecule, node.index, nodes[near].index, true, true);
                vectors[nodes[near].index] = vectors[node.index] + revised_vector;
            }
        }
    }
    return vectors;
}


int test_raw_bfs(const Atoms &aseatoms, std::vector<int> mol_inds, const read_mol &itp_data){
    /*
    raw_bfsとraw_get_distances_micのテスト．低分子の場合には両者が同じ結果を与えることを期待したい．
    */
    // raw_get_distanes_micでのvector
    std::vector<Eigen::Vector3d> vectors = raw_get_distances_mic(aseatoms, mol_inds[itp_data.representative_atom_index], mol_inds, true, true);
    std::cout << "len vectors :: " << vectors.size() << std::endl;
    
    auto nodes = raw_make_graph_from_itp(itp_data);
    std::vector<Eigen::Vector3d> vectors2 = raw_bfs(aseatoms, nodes, vectors, mol_inds, itp_data.representative_atom_index);

    // vectorsとvectors2の差を計算する．
    for (int i=0, N=vectors.size(); i<N; i++){
        std::cout << i << " vectors[i] - vectors2[i] :: " << (vectors[i] - vectors2[i]).norm() << std::endl;
    }
    return 0;
}


std::vector<Eigen::Vector3d> raw_get_pbc_mol(const Atoms &aseatoms, std::vector<int> mol_inds, std::vector<std::vector<int>> bonds_list_j, const read_mol &itp_data) {
    /*
    純粋にpbcを計算するraw_get_distances_mic，およびraw_bfsのラッパー関数．
    まず，通常はraw_get_distances_micで計算を行う．
    しかし，raw_get_distances_micで計算したベクトルの中に大きい要素があれば，それはpbcをまたいでいると考えられる．
    そこで，その場合にはraw_bfsで計算し直す．
    */
    std::vector<Eigen::Vector3d> vectors = raw_get_distances_mic(aseatoms, mol_inds[itp_data.representative_atom_index], mol_inds, true, true);
    std::vector<Eigen::Vector3d> vectors_old = vectors;

    // ボンドリストを0から作り直す
    // TODO :: そもそもここはitp_dataがあれば取得可能なのでいらないはず．   
    std::vector<std::vector<int>> bonds_list_from_zero(bonds_list_j.size());
    for (int i = 0, N=bonds_list_j.size(); i < N; i++) {
        std::vector<int> bond={bonds_list_j[i][0] - mol_inds[0], bonds_list_j[i][1] - mol_inds[0]};
        // bond.push_back(bonds_list_j[i][0] - mol_inds[0]);
        // bond.push_back(bonds_list_j[i][1] - mol_inds[0]);
        // bonds_list_from_zero.push_back(bond);
        bonds_list_from_zero[i] = bond;        
    }

    bool IF_CALC_BFS = false; // 計算をやり直すかどうかのフラグ
    for (auto bond : bonds_list_from_zero) {
        // double bond_distance = std::sqrt(std::pow(vectors[bond[0]][0] - vectors[bond[1]][0], 2) + std::pow(vectors[bond[0]][1] - vectors[bond[1]][1], 2) + std::pow(vectors[bond[0]][2] - vectors[bond[1]][2], 2));
        // eigen::Vector3dのnorm計算
        double bond_distance = (vectors[bond[0]]-vectors[bond[1]]).norm();
        if (bond_distance > 3.0) {
            IF_CALC_BFS = true;
        }
    }
    
    if (IF_CALC_BFS == true) {
#ifdef DEBUG
        std::cout << "WARNING(raw_get_pbc_mol) :: mol_index " << mol_inds[0] << " :: recalculation of vectors is required." << std::endl;
#endif //! DEBUG
        auto nodes = raw_make_graph_from_itp(itp_data);
        std::vector<Eigen::Vector3d> vectors2 = raw_bfs(aseatoms, nodes, vectors, mol_inds, itp_data.representative_atom_index);
        vectors = vectors2;
    }
    
    return vectors;
}


int raw_bfs_test(std::vector<Node>& nodes, int representative = 0){
    /*
    bfs(幅優先探索)がちゃんと実装できているかのテスト
    nodesを与えて，representative番目のノードから探索を行う．
    うまくいけば0を返す．
    */
    std::deque<Node> queue;
    queue.push_back(nodes[representative]); // 最初はここからスタート
    nodes[representative].parent = 0;
    while (!queue.empty()) {
        Node node = queue.front(); //先頭要素を取得する
        std::cout << "takes !!" << node.index << std::endl;
        queue.pop_front();//先頭要素を削除する
        std::vector<int> nears = node.nears;
        for (int near : nears) { // node.nearsに入っているところを探索
            if (nodes[near].parent == -1) { //親が-1の場合は未探索なので探索する
                queue.push_back(nodes[near]); // queueに追加
                nodes[near].parent = node.index; //親を設定
            }
        }
    }
    // 親ノードを格納
    std::vector<int> parents_list;
    for (int i=0, N=nodes.size(); i< N; i++){
        parents_list.push_back(nodes[i].parent);
    };
    // -1が含まれていたらノード1に辿り着けないノードが存在する
    bool if_found_minus1 = (std::find(parents_list.begin(), parents_list.end(), -1) == parents_list.end());
    if (!(if_found_minus1)){
        std::cout << "BFS fails !!" << std::endl;
        for (int i=0, N=nodes.size(); i<N;i++){
            std::cout << "node/parent :: " << nodes[i].index << "/" << nodes[i].parent << std::endl;
        }
        return 1;
    } else{
        std::cout << "BFS succeed !!" << std::endl;
        for (int i=0, N=nodes.size();i<N;i++){
            std::cout << "node/parent :: " << nodes[i].index << "/" << nodes[i].parent << std::endl;
        }
    };
    return 0;
}

#endif //! ATOMS_CORE_H
