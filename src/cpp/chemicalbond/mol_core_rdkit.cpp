/**
 * @file mol_core_rdkit.cpp
 * @brief read_mol class for molecule information using RDKit
 * @author Tomohito Amano
 * @date 2024/06/07
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
#include <Eigen/Core> // 行列演算など基本的な機能．
#include "../include/printvec.hpp"
#include "mol_core_rdkit.hpp"
#include <iostream>
#include <GraphMol/GraphMol.h>
#include <GraphMol/FileParsers/FileParsers.h>


/**
 2023/5/30
ボンド情報などに関する基本的な部分のみを定義．

どうも自作クラスをvectorに入れる場合は特殊な操作が必要な模様．
https://nprogram.hatenablog.com/entry/2017/07/05/073922
*/

// default constructor
read_mol_rdkit::read_mol_rdkit(){};

read_mol_rdkit::read_mol_rdkit(std::string bondfilename){ //コンストラクタ
    // bond_listとatom_listを読み込む
    _read_bondfile(bondfilename);
    _get_num_atoms_per_mol(); // get num_atoms_per_mol
    _print_bond(); // print bond info
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

void read_mol_rdkit::_read_bondfile(std::string bondfilename){
    /**
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
    std::vector<int> atomic_index_list;
    // std::vector<std::vector<int> > bonds_list;
    std::shared_ptr<RDKit::ROMol> mol2(RDKit::MolFileToMol(bondfilename, true,false,true));  //分子の構築

    // get atom_list
    std::string atomic_number;
    for(auto atom: mol2->atoms()) {
        atomic_index = atom->getAtomicNum();
        if (atomic_number == "1"){ // H
            this->atom_list.push_back("H"); // これがクラス変数
        } else if (atomic_number == "6"){ // C
            this->atom_list.push_back("C"); // これがクラス変数
        } else if (atomic_number == "8"){
            this->atom_list.push_back("O"); // これがクラス変数
        } else if (atomic_number == "7"){
            this->atom_list.push_back("N"); // これがクラス変数
        } else{
            std::cout << "ERROR(_read_bondfile)" << std::endl;
        }
    }

    // get bonds_list
    for( unsigned int i = 0 , is = mol2->getNumBonds(false) ; i < is ; ++i ) {
        std::cout << "bond index :: " << i << " ";
        const RDKit::Bond *bond = mol2->getBondWithIdx( i );
        std::cout << "bond atoms :: " << bond->getBeginAtomIdx() << " " << bond->getEndAtomIdx() << std::endl;
        // bond->getBeginAtomIdx is unsigned int
        this->bonds_list.push_back({static_cast<int>(bond->getBeginAtomIdx()), static_cast<int>(bond->getEndAtomIdx())}); // これがクラス変数
        std::cout << "bond type :: " << bond->getBondType() << std::endl;
    }

    // get representative atom
    RDKit::Conformer &conf = mol2->getConformer();
    Eigen::Vector3d average_position(0,0,0);
    // if true, remove H, if false, keep H
    for(int indx=0; indx<mol2->getNumAtoms(true);indx++){ // loop over atoms 
        auto tmp_atom_position = conf.getAtomPos(indx);
        std::cout << conf.getAtomPos(indx) << std::endl;
        average_position += Eigen::Vector3d(tmp_atom_position[0], tmp_atom_position[1], tmp_atom_position[2]);
    }
    // average_position
    average_position = average_position / mol2->getNumAtoms(true);
    std::cout << "average_position :: " << average_position << std::endl;
    // 
    Eigen::Vector3d tmp_vector;
    double smallest_distance = 10000.0; // 大きい値にしておく
    int smallest_index = 0;
    // get nearest atom to the average position
    for(int indx=0; indx<mol2->getNumAtoms(false);indx++){ // loop over atoms 
        if (atom_list[indx] == "H") {continue;} // Hは除外
        auto tmp_atom_position = conf.getAtomPos(indx);
        tmp_vector = Eigen::Vector3d(tmp_atom_position[0], tmp_atom_position[1], tmp_atom_position[2]);
        double distance = (tmp_vector - average_position).norm();
        if (distance < smallest_distance){
            smallest_distance = distance;
            smallest_index = indx;
        }
    }
    this->representative_atom_index = smallest_index;
    std::cout << "smallest_distance, smallest_index :: " << smallest_distance << "  " << smallest_index << std::endl;
    std::cout << std::endl;
    
};

void read_mol_rdkit::_print_bond() const{
    std::cout << "================" << std::endl;
    std::cout << " num_atoms_per_mol... :: " << this->num_atoms_per_mol << std::endl;
    print_vec(this->atom_list,  " atom_list");
    print_vec(this->bonds_list, " bonds_list");
    std::cout << " representative atom  :: " << this->representative_atom_index << std::endl;
    std::cout << std::endl;
};

void read_mol_rdkit::_get_num_atoms_per_mol(){
    this->num_atoms_per_mol = this->mol2->getNumAtoms(false); // # of atoms
}

void read_mol_rdkit::_get_bonds(){
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

void read_mol_rdkit::_get_bond_index(){
    /**
     * @brief ボンドリスト({1,2}みたいなの)からボンドインデックスを取得
     * @
     * 
     */
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

void read_mol_rdkit::_get_lonepair_atomic_index(){
    // O/N lonepair
    for (int i = 0, N=atom_list.size(); i < N; i++) {
        if (atom_list[i] == "O") {
            o_list.push_back(i);
        } else if (atom_list[i] == "N") {
            n_list.push_back(i);
        }
    };  
    std::cout << "================" << std::endl;
    print_vec(o_list, "o_list (lonepair)");
    print_vec(n_list, "n_list (lonepair)");
}

void read_mol_rdkit::_get_coc_and_coh_bond() { // coc,cohに対応するo原子のindexを返す．o_listに対応．
    /**
     * @fn
     * coc/cohボンドの情報を取得．予測計算においては，coc/coh構造を持つO原子のindexだけわかれば良い．
     * 一方，assign計算をやる際には，O原子の両端のボンドの情報も必要．
     * @brief coc/cohボンドの情報を取得．
     * @param (引数名) 引数の説明
     * @param (引数名) 引数の説明
     * @return 戻り値の説明
     * @sa 参照すべき関数を書けばリンクが貼れる
     * @detail アルゴリズムとしては，全てのO原子に対するループを回して，各O原子が所属するボンドの情報を取得する．両端の原子がCCならCOCに，CHならCOHに振り分ける．
     * 同時に，O原子に対する両端ボンドの情報も保持しておきたい．
    */
    int counter;
    for (int o_num = 0, n=o_list.size(); o_num < n; o_num++) { // o_listに入っているO原子に関するLoopで，両方の隣接原子を検索する．
        // まずはO原子の隣接原子の種類を取得
        // std::vector<std::pair<std::string, std::vector<int>>> neighbor_atoms;
        std::vector<std::string> neighbor_atoms;
        std::vector<int> tmp_bond_index;
        std::vector<int> tmp_bond_index2;


        counter = 0; //ボンドindexのカウンター（ループごとに初期化）

        for (auto bond : bonds_list) { //ボンドリストのループで，O原子があるかどうかをチェックし，O原子があればその隣接原子の原子種類を取得．
            if (bond[0] == o_list[o_num]) {
                // neighbor_atoms.push_back({atom_list[bond[1]], bond});
                neighbor_atoms.push_back(atom_list[bond[1]]);
                tmp_bond_index.push_back(counter); // ボンド番号（全体のボンドの中で何番目か）
                // bond[1]の原子種類によってボンドの種類を判別し，raw_convertを実施
                if (atom_list[bond[1]] == "C"){ // COボンド
                    tmp_bond_index2.push_back(raw_convert_bondindex(co_bond_index,counter));
                } else if (atom_list[bond[1]] == "H"){ // OHボンド
                    tmp_bond_index2.push_back(raw_convert_bondindex(oh_bond_index,counter));
                } else{
                    std::cout << "ERROR(_get_coc_and_coh_bond)" << std::endl;
                };
            } else if (bond[1] == o_list[o_num]) {
                // neighbor_atoms.push_back({atom_list[bond[0]], bond});
                neighbor_atoms.push_back(atom_list[bond[0]]);
                tmp_bond_index.push_back(counter); // ボンド番号（全体のボンドの中で何番目か）
                // bond[1]の原子種類によってボンドの種類を判別し，raw_convertを実施
                if (atom_list[bond[0]] == "C"){ // COボンド
                    tmp_bond_index2.push_back(raw_convert_bondindex(co_bond_index,counter));
                } else if (atom_list[bond[0]] == "H"){ // OHボンド
                    tmp_bond_index2.push_back(raw_convert_bondindex(oh_bond_index,counter));
                } else{
                    std::cout << "ERROR(_get_coc_and_coh_bond)" << std::endl;
                };
            }
            counter += 1;
        }

        // std::vector<std::string> neighbor_atoms_tmp = {neighbor_atoms[0][0], neighbor_atoms[1][0]};
        if (neighbor_atoms[0] == "C" && neighbor_atoms[1] == "H") { //1番目にCO，二番目にOHを引いた場合
            coh_list.push_back(o_list[o_num]);
            // 対応するbond情報をcoh_bond_info/coc_bond_infoに格納する
            // TODO :: ここは，o_num（O原子内での番号）を入れるか，o_list[o_num]（全体の原子の中での番号）を入れるか精査が必要
            coh_bond_info[o_list[o_num]] = {tmp_bond_index[0],tmp_bond_index[1]};
            coh_bond_info2[o_num]        = {tmp_bond_index2[0],tmp_bond_index2[1]}; 

            // int index_co = std::distance(co_bond.begin(), std::find(co_bond.begin(), co_bond.end(), neighbor_atoms[0].second));
            // int index_oh = std::distance(oh_bond.begin(), std::find(oh_bond.begin(), oh_bond.end(), neighbor_atoms[1].second));
            // coh_index.push_back({o_num, {{"CO", index_co}, {"OH", index_oh}}});
        } else if (neighbor_atoms[0] == "H" && neighbor_atoms[1] == "C") { //1番目にOH，二番目にCOを引いた場合
            coh_list.push_back(o_list[o_num]);
            // 対応するbond情報をcoh_bond_info/coc_bond_infoに格納する
            // TODO :: ここは，o_num（O原子内での番号）を入れるか，o_list[o_num]（全体の原子の中での番号）を入れるか精査が必要
            coh_bond_info[o_list[o_num]] = {tmp_bond_index[1],tmp_bond_index[0]};
            coh_bond_info2[o_num]        = {tmp_bond_index2[1],tmp_bond_index2[0]};
            // int index_co = std::distance(co_bond.begin(), std::find(co_bond.begin(), co_bond.end(), neighbor_atoms[1].second));
            // int index_oh = std::distance(oh_bond.begin(), std::find(oh_bond.begin(), oh_bond.end(), neighbor_atoms[0].second));
            // coh_index.push_back({o_num, {{"CO", index_co}, {"OH", index_oh}}});
        } else if (neighbor_atoms[0] == "C" && neighbor_atoms[1] == "C") {
            coc_list.push_back(o_list[o_num]);
            // 対応するbond情報をcoh_bond_info/coc_bond_infoに格納する
            // TODO :: ここは，o_num（O原子内での番号）を入れるか，o_list[o_num]（全体の原子の中での番号）を入れるか精査が必要
            coc_bond_info[o_list[o_num]] = {tmp_bond_index[0],tmp_bond_index[1]};
            coc_bond_info2[o_num]        = {raw_convert_bondindex(co_bond_index,tmp_bond_index[0]),raw_convert_bondindex(co_bond_index,tmp_bond_index[1])};

        //     int index_co1 = std::distance(co_bond.begin(), std::find(co_bond.begin(), co_bond.end(), neighbor_atoms[0].second));
        //     int index_co2 = std::distance(co_bond.begin(), std::find(co_bond.begin(), co_bond.end(), neighbor_atoms[1].second));
        //     coc_index.push_back({o_num, {{"CO1", index_co1}, {"CO2", index_co2}}});
        }
    }
    std::cout << "================" << std::endl;
    std::cout << "COC bond size :: " << coc_list.size() << std::endl;
    std::cout << "O atoms in COC bond (coc_list)... " << std::endl; 
    for (int i = 0, n=coc_list.size(); i < n; i++) {
        std::cout << coc_list[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "COH bond size :: " << coh_list.size() << std::endl;
    std::cout << "O atoms in COH bond (coh_list)... " << std::endl;
    for (int i = 0, n=coh_list.size(); i < n; i++) {
        std::cout << coh_list[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "COC bond info(coc_bond_info))... " << std::endl;
    for (const auto& [key, value] : coc_bond_info){
        std::cout << key << " => " << std::get<0>(value) << " " << std::get<1>(value) << "\n";
    }

    std::cout << std::endl;
    std::cout << "COH bond info(coh_bond_info))... " << std::endl;
    for (const auto& [key, value] : coh_bond_info){
        std::cout << key << " => " << std::get<0>(value) << " " << std::get<1>(value)  << "\n";
    }

    std::cout << std::endl;
    std::cout << "COC bond info2(coc_bond_info2))... " << std::endl;
    for (const auto& [key, value] : coc_bond_info2){
        std::cout << key << " => " << std::get<0>(value) << " " <<  std::get<1>(value)  << "\n";
    }

    std::cout << std::endl;
    std::cout << "COH bond info2(coh_bond_info2))... " << std::endl;
    for (const auto& [key, value] : coh_bond_info2){
        std::cout << key << " => " << std::get<0>(value) << " " << std::get<1>(value)  << "\n";
    }

}

int raw_convert_bondindex(std::vector<int> xx_bond_index,int bondindex){
/**
 * @fn ボンド番号を与えると，それが対応するch_bond_indexの何番目に対応するかを返す．すなわちボンド番号iを入力として
 * @fn bonds_list[i] = ch_bond_index[j]
 * @fn を満たすインデックスjを返す．
 * @fn vectorから要素を検索し，そのindexを返すにはstd::findが使える．
 * @fn https://www.cns.s.u-tokyo.ac.jp/~masuoka/post/search_vector_index/
 * @fn
 * @fn ch_bond_indexなどだけでなく，o_listなど原子のリストにも使える．
*/
    std::vector<int>::iterator itr; // 検索用イテレータ
     bool flg_if_find = false; // 見つかったらtrueにする．
    // 特定ボンドを検索
    itr = std::find(xx_bond_index.begin(), xx_bond_index.end(), bondindex);
    if (!(itr == xx_bond_index.end())){
        flg_if_find = true; //見つかった場合はtrueへ
        const int wanted_index = std::distance(xx_bond_index.begin(), itr);
        return wanted_index;
    };
    std::cout << "ERROR(raw_convert_bondindex) :: not found index !! :: " << bondindex << std::endl;
    return -1;
}



std::vector<int> read_mol_rdkit::raw_convert_bondpair_to_bondindex(std::vector<std::vector<int> > bonds, std::vector<std::vector<int> > bonds_list) {
    /**
    * @fn ボンド[a,b]から，ボンド番号（bonds.index）への変換を行う．ボンド番号はbonds_list中のインデックス．
    * @fn bondsにch_bondsなどの一覧を入力し，それを番号のリストに変換する．
    * @fn 
    * @fn ある要素がvectorに含まれているかどうかの判定はstd::findで可能．
    * @fn 要素のindexはstd::distanceで取得可能．
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

