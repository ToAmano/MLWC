#pragma once

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

#include <utility> // https://rinatz.github.io/cpp-book/ch03-04-pairs/

#include <iostream>
#include <GraphMol/GraphMol.h>
#include <GraphMol/FileParsers/FileParsers.h>

/**
 * @brief 2023/5/30 ボンド情報などに関する基本的な部分のみを定義．
 * @details どうも自作クラスをvectorに入れる場合は特殊な操作が必要な模様． @n
 * https://nprogram.hatenablog.com/entry/2017/07/05/073922 @n
 * pythonの同名クラスと違い，別途ファイルからボンド情報を読み込む． @n
 * 読み込んだボンドをクラス変数に格納する． @n
 * とりあえずPGの例をコメントで残してあるので参考に． @n
 * - num_atoms_per_mol : 原子数 
 * - atom_list : 原子番号のリスト 
*/
class read_mol_rdkit{
    public: 
        // クラス変数たち
        // ! 分子あたりの原子数
        int num_atoms_per_mol;  // int num_atoms_per_mol= 13;
        // atom list（原子番号）
        // std::vector<std::string> atom_list{"O","C","C","O","C","H","H","H","H","H","H","H","H"};
        std::vector<std::string> atom_list;

        // bonds_listの作成
        // std::vector<std::vector<int> > bonds_list{{0, 1}, {0, 5}, {1, 2}, {1, 6}, {1, 7}, {2, 3}, {2, 4}, {2, 8}, {3, 9}, {4, 10}, {4, 11}, {4, 12}};
        std::vector<std::vector<int> > bonds_list;

        // double_bondの作成
        std::vector<std::vector<int> > double_bonds; //(PGの場合は)空リスト
        // 各種ボンド
        std::vector<std::vector<int> > ch_bond, co_bond, oh_bond, oo_bond, cc_bond, ring_bond, coh_bond, coc_bond;
        // ローンペア（O原子のindex）
        std::vector<int> o_list, n_list, coc_list, coh_list;
        // ボンドリスト
        std::vector<int> ch_bond_index,oh_bond_index,co_bond_index,oo_bond_index,cc_bond_index,ring_bond_index, coh_bond_index, coc_bond_index;
        // COC/COHの両端ボンドの情報の保持
        // TODO :: 2024/1/17 まだ実験的．今のところpost_processでCOC/COH双極子を計算することしか考えてない．
        // mapクラスとpairクラスを使っている．
        // pairの方には，bonds_listの番号を格納するのが良いだろう．
        std::map<int, std::pair<int, int> > coh_bond_info, coc_bond_info;
        // pairの方には， 直接ch_bond_indexなどのindex番号を与える．
        std::map<int, std::pair<int, int> > coh_bond_info2, coc_bond_info2;
        // rdkitのmolオブジェクト
        std::shared_ptr<RDKit::ROMol> mol2;
        
        // 代表原子の取得（デフォルト値を0にしておく）
        int representative_atom_index = 0;
        read_mol_rdkit(); // default constructor
        read_mol_rdkit(std::string bondfilename); // constructor
        void _read_bondfile(std::string bondfilename);

        void _get_bonds();
        void _get_atomic_index();
        void _get_bond_index();
        void _get_coc_and_coh_bond();
        void _get_lonepair_atomic_index();
        std::vector<int> raw_convert_bondpair_to_bondindex(std::vector<std::vector<int> > bonds, std::vector<std::vector<int> > bonds_list) ;
    private:
        void _print_bond() const; 
        void _get_num_atoms_per_mol();
};

int raw_convert_bondindex(std::vector<int> xx_bond_index, int bondindex); // bondindex[i]から，ch_bond_index[j]を満たすjを返す．（要は変換）
  


