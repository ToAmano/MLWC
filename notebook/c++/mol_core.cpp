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
        int num_atoms_per_mol= 13;
        // atom list（原子番号）
        std::vector<std::string> atom_list{"O","C","C","O","C","H","H","H","H","H","H","H","H"};
        // bonds_listの作成
        std::vector<std::vector<int> > bonds_list{{0, 1}, {0, 5}, {1, 2}, {1, 6}, {1, 7}, {2, 3}, {2, 4}, {2, 8}, {3, 9}, {4, 10}, {4, 11}, {4, 12}};
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
        
        // 代表原子の取得
        int representative_atom_index = 0;
        read_mol(){ //コンストラクタ
        // bond情報の取得（関数化．ボンドリストとatom_listがあれば再現できる）
            _get_bonds();
            // O/N lonepair情報の取得
            _get_atomic_index();
        }
        void _get_bonds(){
            // std::vector<int> ch_bond;
            // std::vector<int> co_bond;
            // vector<int> oh_bond;
            // vector<int> oo_bond;
            // vector<int> cc_bond;
            // vector<int> ring_bond; 
            for (auto bond : bonds_list) { // これはc++17以降の書き方？
                std::vector<std::string> tmp = {atom_list[bond[0]], atom_list[bond[1]]};
                if (tmp == std::vector<std::string>{"H", "C"} || tmp == std::vector<std::string>{"C", "H"}) {
                    ch_bond.push_back(bond);
                }
                if (tmp == std::vector<std::string>{"O", "C"} || tmp == std::vector<std::string>{"C", "O"}) {
                    co_bond.push_back(bond);
                }
                if (tmp == std::vector<std::string>{"O", "H"} || tmp == std::vector<std::string>{"H", "O"}) {
                    oh_bond.push_back(bond);
                }
                if (tmp == std::vector<std::string>{"O", "O"}) {
                    oo_bond.push_back(bond);
                }
                if (tmp == std::vector<std::string>{"C", "C"}) { 
                    // TODO :: ring bondの対応を！！
                    cc_bond.push_back(bond); 
                }
            };
            // this->ch_bond = ch_bond;
            // this->co_bond = co_bond;
            // this->oh_bond = oh_bond;
            // this->oo_bond = oo_bond;
            // this->cc_bond = cc_bond;
            // this->ring_bond = ring_bond;
            if (ch_bond.size() + co_bond.size() + oh_bond.size() + oo_bond.size() + cc_bond.size() + ring_bond.size() != bonds_list.size()) {
                std::cout << " " << std::endl;
                std::cout << " WARNING :: There are unkown bonds in self.bonds_list..." << std::endl;
            }
            // 最後にbondの印刷
            std::cout << "================" << std::endl;
            std::cout << "CH bond... ";
            for (int i = 0; i < ch_bond.size(); i++) {
                std::cout << "["<< ch_bond[i][0] << " " << ch_bond[i][1] << "] ";
            }
            std::cout << std::endl;
            std::cout << "OH bond... ";
            for (int i = 0; i < oh_bond.size(); i++) {
                std::cout << "["<< oh_bond[i][0] << " " << oh_bond[i][1] << "] ";
            }
            std::cout << std::endl;
            std::cout << "CO bond... ";
            for (int i = 0; i < co_bond.size(); i++) {
                std::cout << "["<< co_bond[i][0] << " " << co_bond[i][1] << "] ";
            }
            std::cout << std::endl;
            std::cout << "CC bond... ";
            for (int i = 0; i < cc_bond.size(); i++) {
                std::cout << "["<< cc_bond[i][0] << " " << cc_bond[i][1] << "] ";
            }
            std::cout << std::endl;

            // 以下ボンドリストへの変換
            // ring_bond_index=raw_convert_bondpair_to_bondindex(ring_bond,bonds_list)
            ch_bond_index=raw_convert_bondpair_to_bondindex(ch_bond,bonds_list);
            co_bond_index=raw_convert_bondpair_to_bondindex(co_bond,bonds_list);
            oh_bond_index=raw_convert_bondpair_to_bondindex(oh_bond,bonds_list);
            cc_bond_index=raw_convert_bondpair_to_bondindex(cc_bond,bonds_list);
            std::cout << "================" << std::endl;
            std::cout << "ch bond index... ";
            for (int i = 0; i < ch_bond_index.size(); i++) {
                std::cout << ch_bond_index[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "oh bond index... ";
            for (int i = 0; i < oh_bond_index.size(); i++) {
                std::cout << oh_bond_index[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "co bond index... ";
            for (int i = 0; i < co_bond_index.size(); i++) {
                std::cout << co_bond_index[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "cc bond index... ";
            for (int i = 0; i < cc_bond_index.size(); i++) {
                std::cout << cc_bond_index[i] << " ";
            }
            std::cout << std::endl;

        }

        void _get_atomic_index(){
            // O/N lonepair
            for (int i = 0; i < atom_list.size(); i++) {
                if (atom_list[i] == "O") {
                    o_list.push_back(i);
                } else if (atom_list[i] == "N") {
                    n_list.push_back(i);
                }
            }   
            std::cout << "================" << std::endl;
            std::cout << "O atoms (lonepair)... ";
            for (int i = 0; i < o_list.size(); i++) {
                std::cout << o_list[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "N atoms (lonepair)... ";
            for (int i = 0; i < n_list.size(); i++) {
                std::cout << n_list[i] << " ";
            }
            std::cout << std::endl;
        }

        std::vector<int> raw_convert_bondpair_to_bondindex(std::vector<std::vector<int> > bonds, std::vector<std::vector<int> > bonds_list) {
            /*
            ボンド[a,b]から，ボンド番号（bonds.index）への変換を行う．ボンド番号はbonds_list中のインデックス．
            bondsにch_bondsなどの一覧を入力し，それを番号のリストに変換する．

            ある要素がvectorに含まれているかどうかの判定はstd::findで可能．
            要素のindexはstd::distanceで取得可能．
            */
            std::vector<int> bond_index;
            for (auto b : bonds) {
                std::vector<int> reverse_b = {b[1], b[0]}; // bは絶対に2要素vector
                if (std::find(bonds_list.begin(), bonds_list.end(), b) != bonds_list.end()) {
                    bond_index.push_back(std::distance(bonds_list.begin(), std::find(bonds_list.begin(), bonds_list.end(), b)));
                } else if (std::find(bonds_list.begin(), bonds_list.end(), reverse_b) != bonds_list.end()) {
                     bond_index.push_back(std::distance(bonds_list.begin(), std::find(bonds_list.begin(), bonds_list.end(), reverse_b)));
                } else {
                    std::cout << "there is no bond" << b[0] << " " << b[1] << " in bonds list." << std::endl;
                }
            }
            return bond_index;
        };
};