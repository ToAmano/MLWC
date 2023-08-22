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
// #include <rdkit/GraphMol/GraphMol.h>
// #include <rdkit/GraphMol/FileParsers/MolSupplier.h>
#include <Eigen/Core> // 行列演算など基本的な機能．
#include "numpy.hpp"
#include "npy.hpp"
// #include "numpy_quiita.hpp" // https://qiita.com/ka_na_ta_n/items/608c7df3128abbf39c89
// numpy_quiitaはsscanf_sが読み込めず，残念ながら現状使えない．


/*
ボンド情報などに関する基本的な部分のみを定義．

*/

/*
 2023/5/30
 ase atomsに対応するAtomsクラスを定義する

どうも自作クラスをvectorに入れる場合は特殊な操作が必要な模様．
https://nprogram.hatenablog.com/entry/2017/07/05/073922
*/

class read_mol{
    /*
    pythonの同名クラスと違い，別途ファイルからボンド情報を読み込む．
    読み込んだボンドをクラス変数に格納する．
     - num_atoms_per_mol : 原子数
     - atom_list : 原子番号のリスト

    */
    public: // とりあえずPGの例を実装
        // クラス変数
        // 原子数の取得
        // int num_atoms_per_mol= 13;
        int num_atoms_per_mol;
        // atom list（原子番号）
        // std::vector<std::string> atom_list{"O","C","C","O","C","H","H","H","H","H","H","H","H"};
        std::vector<std::string> atom_list;

        // bonds_listの作成
        // std::vector<std::vector<int> > bonds_list{{0, 1}, {0, 5}, {1, 2}, {1, 6}, {1, 7}, {2, 3}, {2, 4}, {2, 8}, {3, 9}, {4, 10}, {4, 11}, {4, 12}};
        std::vector<std::vector<int> > bonds_list;

        // double_bondの作成
        std::vector<std::vector<int> > double_bonds; //(PGの場合は)空リスト
        // 各種ボンド
        std::vector<std::vector<int> > ch_bond, co_bond, oh_bond, oo_bond, cc_bond, ring_bond;
        // ローンペア
        std::vector<int> o_list, n_list;
        // ボンドリスト
        std::vector<int> ch_bond_index,oh_bond_index,co_bond_index,oo_bond_index,cc_bond_index,ring_bond_index;
        // print(" -----  ml.read_mol :: parse results... -------")
        // print(" bonds_list :: ", self.bonds_list)
        // print(" counter    :: ", self.num_atoms_per_mol)
        // # print(" atomic_type:: ", self.atomic_type)
        // print(" atom_list  :: ", self.atom_list)
        // print(" -----------------------------------------------")
        
        // 代表原子の取得（デフォルト値を0にしておく）
        int representative_atom_index = 0;
  read_mol(std::string bondfilename); // コンストラクタ
        void _read_bondfile(std::string bondfilename);

        void _get_bonds();

        void _get_atomic_index();

  std::vector<int> raw_convert_bondpair_to_bondindex(std::vector<std::vector<int> > bonds, std::vector<std::vector<int> > bonds_list) ;
  
};

class Node {
    /*
    itpファイルを読み込み，ノードの隣接情報をグラフとして取得する．
    * @param : index : ノードのインデックス(aseatomsでの0スタート番号)
    * @param : nears : ノードの隣接ノードのインデックス(aseatomsでの0スタート番号)
    * @param : parent : ノードの親ノードのインデックス，-1で初期化
    */
    public:
        int index;
        std::vector<int> nears;
        int parent;

    Node(int index);   // Custom コンストラクタ
        // https://nobunaga.hatenablog.jp/entry/2016/07/03/230337
        // https://nprogram.hatenablog.com/entry/2017/07/05/073922
        // https://monozukuri-c.com/langcpp-copyconstructor/
    Node(const Node & node); // Copy constructor

  std::string __repr__();

    private:
  std::string toString(const std::vector<int>& vec);
};


std::vector<Node> raw_make_graph_from_itp(const read_mol& itp_data);
