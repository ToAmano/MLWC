/**
 * @file mol_core.cpp
 * @brief read_mol class for molecule information
 * @author Tomohito Amano
 * @date 2023/10/15
 */

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
#include "include/printvec.hpp"
#include "mol_core.hpp"
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


read_mol::read_mol(std::string bondfilename){ //コンストラクタ
    // bond_listとatom_listを読み込む
    _read_bondfile(bondfilename);
    // bond情報の取得（関数化．ボンドリストとatom_listがあれば再現できる）
    _get_bonds();
    // bond_indexの取得
    _get_bond_index();
    // O/N lonepair情報の取得
    _get_lonepair_atomic_index();
    // TODO :: get_atomic_indexの実装を行う．
    // _get_atomic_index(); 
    // COC/COH情報の取得
    _get_coc_and_coh_bond();
}

void read_mol::_read_bondfile(std::string bondfilename){
    /*
    原子リスト（&）とボンドリスト（&）を読み込む．
    TODO :: ちゃんとc++版のrdkitから情報を読み込むようにしたい．．．
    */
    std::ifstream ifs(bondfilename);
    if (ifs.fail()) {
        std::cerr << "Cannot open bondfile\n" << std::endl;
        exit(0);
    }
    std::cout << "start reading bondfile\n" << std::endl;
    std::string str; // stringstream用
    int atomic_index;
    std::string atomic_type;
    int bond_index_0, bond_index_1;
    bool IF_READ_ATOM = false;
    bool IF_READ_BOND = false;
    bool IF_READ_REPRESENTATIVE = false;
    std::vector<std::string> atomic_type_list;
    std::vector<int> atomic_index_list;
    // std::vector<std::vector<int> > bonds_list;
    while (getline(ifs,str)) {
#ifdef DEBUG
        std::cout << str << std::endl;
#endif //! DEBUG
        std::stringstream ss(str);
        if (str == "&atomlist") { // 最初の行で
            std::cout << "1st line :: start reading atomlist ";
            IF_READ_ATOM = true;
            continue;
        }
        if (str == "&bondlist") {
            std::cout << "2nd line :: start reading bomdlist ";
            IF_READ_ATOM = false;
            IF_READ_BOND = true;
            continue;
        }
        if (str == "&representative") {
            std::cout << "3rd line :: start reading representative atom ";
            IF_READ_ATOM = false;
            IF_READ_BOND = false;
            IF_READ_REPRESENTATIVE = true;
            continue;
        }
        if (IF_READ_ATOM){
            ss >> atomic_index >> atomic_type ;
            atomic_index_list.push_back(atomic_index);
            atomic_type_list.push_back(atomic_type);
            atom_list.push_back(atomic_type); // これがクラス変数
        }
        if (IF_READ_BOND){
            ss >> bond_index_0 >> bond_index_1;
            bonds_list.push_back({bond_index_0, bond_index_1});
        }
        if (IF_READ_REPRESENTATIVE){
            ss >> representative_atom_index; // 代表原子の取得（クラス変数）
        }
    }
    
    num_atoms_per_mol = atomic_index_list.size(); //クラス変数（原子数）
    // 最後にatomの印刷
    std::cout << "================" << std::endl;
    std::cout << "num_atoms_per_mol... " << num_atoms_per_mol << std::endl;
    
    // 最後にatomの印刷
    std::cout << "================" << std::endl;
    print_vec(atom_list, "atom_list");

    // 最後にbondの印刷
    std::cout << "================" << std::endl;
    print_vec(bonds_list, "bonds_list");

    // 最後にrepresentative_atomの印刷
    std::cout << "================" << std::endl;
    std::cout << "representative atom... ";
    std::cout << representative_atom_index << std::endl;
    std::cout << std::endl;
}

void read_mol::_get_bonds(){
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
    print_vec(ch_bond, "ch_bond"); 
    print_vec(oh_bond, "oh_bond"); 
    print_vec(co_bond, "co_bond"); 
    print_vec(cc_bond, "cc_bond"); 
}

void read_mol::_get_bond_index(){
    // 以下ボンドリストへの変換
    // ring_bond_index=raw_convert_bondpair_to_bondindex(ring_bond,bonds_list)
    ch_bond_index=raw_convert_bondpair_to_bondindex(ch_bond,bonds_list);
    co_bond_index=raw_convert_bondpair_to_bondindex(co_bond,bonds_list);
    oh_bond_index=raw_convert_bondpair_to_bondindex(oh_bond,bonds_list);
    cc_bond_index=raw_convert_bondpair_to_bondindex(cc_bond,bonds_list);
    std::cout << "================" << std::endl;
    print_vec(ch_bond_index, "ch_bond_index");
    print_vec(oh_bond_index, "oh_bond_index");
    print_vec(co_bond_index, "co_bond_index");
    print_vec(cc_bond_index, "cc_bond_index");            
}

void read_mol::_get_lonepair_atomic_index(){
    // O/N lonepair
for (int i = 0, N=atom_list.size(); i < N; i++) {
        if (atom_list[i] == "O") {
            o_list.push_back(i);
        } else if (atom_list[i] == "N") {
            n_list.push_back(i);
        }
    }   
    std::cout << "================" << std::endl;
    print_vec(o_list, "o_list (lonepair)");
    print_vec(n_list, "n_list (lonepair)");
}

void read_mol::_get_coc_and_coh_bond() { // coc,cohに対応するo原子のindexを返す．o_listに対応．
    for (int o_num = 0, n=o_list.size(); o_num < n; o_num++) {
        // まずはO原子の隣接原子を取得
        // std::vector<std::pair<std::string, std::vector<int>>> neighbor_atoms;
        std::vector<std::string> neighbor_atoms;

        for (auto bond : bonds_list) {
            if (bond[0] == o_list[o_num]) {
                // neighbor_atoms.push_back({atom_list[bond[1]], bond});
                neighbor_atoms.push_back(atom_list[bond[1]]);
            } else if (bond[1] == o_list[o_num]) {
                // neighbor_atoms.push_back({atom_list[bond[0]], bond});
                neighbor_atoms.push_back(atom_list[bond[0]]);
            }
        }
        // std::vector<std::string> neighbor_atoms_tmp = {neighbor_atoms[0][0], neighbor_atoms[1][0]};
        if (neighbor_atoms[0] == "C" && neighbor_atoms[1] == "H") {
            coh_list.push_back(o_list[o_num]);
            // int index_co = std::distance(co_bond.begin(), std::find(co_bond.begin(), co_bond.end(), neighbor_atoms[0].second));
            // int index_oh = std::distance(oh_bond.begin(), std::find(oh_bond.begin(), oh_bond.end(), neighbor_atoms[1].second));
            // coh_index.push_back({o_num, {{"CO", index_co}, {"OH", index_oh}}});
        } else if (neighbor_atoms[0] == "H" && neighbor_atoms[1] == "C") {
            coh_list.push_back(o_list[o_num]);
            // int index_co = std::distance(co_bond.begin(), std::find(co_bond.begin(), co_bond.end(), neighbor_atoms[1].second));
            // int index_oh = std::distance(oh_bond.begin(), std::find(oh_bond.begin(), oh_bond.end(), neighbor_atoms[0].second));
            // coh_index.push_back({o_num, {{"CO", index_co}, {"OH", index_oh}}});
        } else if (neighbor_atoms[0] == "C" && neighbor_atoms[1] == "C") {
            coc_list.push_back(o_list[o_num]);
        //     int index_co1 = std::distance(co_bond.begin(), std::find(co_bond.begin(), co_bond.end(), neighbor_atoms[0].second));
        //     int index_co2 = std::distance(co_bond.begin(), std::find(co_bond.begin(), co_bond.end(), neighbor_atoms[1].second));
        //     coc_index.push_back({o_num, {{"CO1", index_co1}, {"CO2", index_co2}}});
        }
    }
    std::cout << "================" << std::endl;
    std::cout << "O atoms in COC bond... " << std::endl; 
    for (int i = 0, n=coc_list.size(); i < n; i++) {
        std::cout << coc_list[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "O atoms in COH bond... " << std::endl;
    for (int i = 0, n=coh_list.size(); i < n; i++) {
        std::cout << coh_list[i] << " ";
    }
}


std::vector<int> read_mol::raw_convert_bondpair_to_bondindex(std::vector<std::vector<int> > bonds, std::vector<std::vector<int> > bonds_list) {
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


// Node::Node(int index) // custom コンストラクタ
// {
//             this->index = index;
//             this->parent = -1;  
// }

// // https://nobunaga.hatenablog.jp/entry/2016/07/03/230337
// // https://nprogram.hatenablog.com/entry/2017/07/05/073922
// // https://monozukuri-c.com/langcpp-copyconstructor/
// Node::Node(const Node & node){ // Copy constructor
//   this->index = node.index;
//   this->nears = node.nears;
//   this->parent = node.parent;
// }


// std::string Node::__repr__() {
//   return "(index:" + std::to_string(index) + ", nears:" + toString(nears) + ", parent:" + std::to_string(parent) + ")";
//   // return "(index:" + std::to_string(index) + ", nears:" + std::to_string(nears) + ", parent:" + std::to_string(parent) + ")";
// }

// std::string Node::toString(const std::vector<int>& vec) {
//   std::string result = "[";
//   for (int i = 0; i < vec.size(); i++) {
//     result += std::to_string(vec[i]);
//     if (i != vec.size() - 1) {
//       result += ", ";
//     }
//   }
//   result += "]";
//   return result;
// }




// class Node {
//     /*
//     itpファイルを読み込み，ノードの隣接情報をグラフとして取得する．
//     * @param : index : ノードのインデックス(aseatomsでの0スタート番号)
//     * @param : nears : ノードの隣接ノードのインデックス(aseatomsでの0スタート番号)
//     * @param : parent : ノードの親ノードのインデックス，-1で初期化
//     */
//     public:
//         int index;
//         std::vector<int> nears;
//         int parent;

Node::Node(int index) { // Custom コンストラクタ
	  this->index = index;
	  this->parent = -1;
	}

// https://nobunaga.hatenablog.jp/entry/2016/07/03/230337
        // https://nprogram.hatenablog.com/entry/2017/07/05/073922
        // https://monozukuri-c.com/langcpp-copyconstructor/
Node::Node(const Node & node){ // Copy constructor
            this->index = node.index;
            this->nears = node.nears;
            this->parent = node.parent;
}

std::string Node::__repr__() {
    return "(index:" + std::to_string(index) + ", nears:" + toString(nears) + ", parent:" + std::to_string(parent) + ")";
    // return "(index:" + std::to_string(index) + ", nears:" + std::to_string(nears) + ", parent:" + std::to_string(parent) + ")";
}

std::string Node::toString(const std::vector<int>& vec) {
    std::string result = "[";
    for (int i = 0, N=vec.size(); i < N; i++) {
        result += std::to_string(vec[i]);
        if (i != vec.size() - 1) {
            result += ", ";
        }
    }
    result += "]";
    return result;
}



std::vector<Node> raw_make_graph_from_itp(const read_mol& itp_data) {
    std::vector<Node> nodes; // ノードのリスト
    for (int i = 0; i < itp_data.num_atoms_per_mol; i++) {
        Node node(i);
        nodes.push_back(node);
    }

    // 全てのボンドリストを見て隣接情報を更新する
    for (auto bond : itp_data.bonds_list) {
        nodes[bond[0]].nears.push_back(bond[1]);
        nodes[bond[1]].nears.push_back(bond[0]);
    }

    return nodes;
}
