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


using namespace std;

class Vector {
	std::vector<int> myVec;

public:
	Vector(std::vector<int> newVector) {
	    myVec = newVector;
	}
	
	void print() {
		for (int i = 0; i < myVec.size(); i++)
			cout << myVec[i] << " ";
	}
	
};

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


double sign(double A){
    // 実数Aの符号を返す
    // https://cvtech.cc/sign/
    return (A>0)-(A<0);
}

std::vector<Eigen::Vector3d> raw_get_distances_mic(Atoms aseatom, int a, std::vector<int> indices, bool mic=true, bool vector=false){
    /*
    ase.atomのget_distances関数(micあり)のc++実装版
    a: 求める原子のaseatomでの順番（index）
    */
    std::vector<Eigen::Vector3d > coordinate = aseatom.get_positions();
    Eigen::Vector3d reference_position = coordinate[a]; // TODO :: 慣れてきたら削除
    std::vector<Eigen::Vector3d> distance_without_mic; 
    // まずはcoordinate[a]からの相対ベクトルを計算する．
    for (int i = 0; i < indices.size(); i++) {
        distance_without_mic.push_back(coordinate[indices[i]]-reference_position); // 座標にEigen::vectorを利用していることでベクトル減産が可能．
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

int raw_cpmd_num_atom(std::string filename){
    /*
    CPMDのmd.outから原子数を取得する．
    */
    ifstream ifs(filename); // ファイル読み込み
    if (ifs.fail()) {
       cerr << "Cannot open file\n";
       exit(0);
    }
    int NUM_ATOM;
    string str;
    int counter = 0;
    getline(ifs,str);
    stringstream ss(str);
    ss >> NUM_ATOM;
    return NUM_ATOM;
};

std::vector<std::vector<double> > raw_cpmd_get_unitcell_xyz(std::string filename = "IONS+CENTERS.xyz") {
    /*
     Lattice="16.267601013183594 0.0 0.0 0.0 16.267601013183594 0.0 0.0 0.0 16.267601013183594" Properties=species:S:1:pos:R:3 pbc="T T T"
    という文字列から，Lattice=" "の部分を抽出しないといけない．
    */
    std::ifstream f(filename);
    std::string firstline, secondline;
    std::getline(f, firstline);
    std::getline(f, secondline);
    std::regex lattice_regex("Lattice=\".*\" ");
    std::smatch match;
    std::regex_search(secondline, match, lattice_regex);
    // https://qiita.com/yohm/items/7c82693b83d4c055fa7b
    // std::cout << "print match :: " << match.str() << std::endl; // DEBUG（ここまではOK．）

    std::string line = match.str();
    line = line.substr(9, line.size() - 11); // 9番目（Lattice="の後）から，"の前までを取得
    // std::cout << "print line :: " << line << std::endl; // DEBUG（ここまではOK．）

    std::vector<std::string> unitcell_vec_str;
    std::istringstream iss(line);
    std::string word;
    while (iss >> word) {
        unitcell_vec_str.push_back(word);
        // std::cout << "word = " << word << std::endl; // DEBUG（ここまではOK．）
    }
    std::vector<std::vector<double> > unitcell_vec(3, vector<double> (3, 0)); // ここで3*3の形に指定しないとダメだった．
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            unitcell_vec[i][j] = std::stod(unitcell_vec_str[i * 3 + j]);
#ifdef DEBUG
            std::cout <<  unitcell_vec[i][j] << " ";
            std::cout << std::endl;
#endif // DEBUG
        }
    }
    return unitcell_vec;
}

// def raw_cpmd_get_unitcell_xyz(filename:str="IONS+CENTERS.xyz")->np.ndarray:
//     '''
//     xyzの2行目を取得して格子定数を返す．
//     TODO :: 実装がaseで作ったxyzにしか適用できないと思うので注意！！
    
//     output
//     ------------
//     unitcell_vec :: 
//     '''
//     import re
//     import numpy as np
//     f = open(filename, mode="r")
//     firstline = f.readline().rstrip() # 1行目は廃棄
//     secondline = f.readline().rstrip()
//     line = re.search('Lattice=\".*\" ',secondline).group()
//     line = line.strip("Lattice=")
//     unitcell_vec_str = line.strip(" \"").split()
//     unitcell_vec = np.array([float(i) for i in unitcell_vec_str]).reshape((3,3))
//     return unitcell_vec

std::vector<Atoms> ase_io_read(std::string filename, int NUM_ATOM, std::vector<std::vector<double> > unitcell_vec){
    /*
    MDトラジェクトリを含むxyzファイルから
        - 格子定数
        - 原子番号
        - 原子座標
    を取得して，Atomsのリストにして返す．
    読み込み簡単化&高速化のため，予めNUM_ATOMを取得しておく．
    */
    //! test for Atomicnum
    Atomicnum atomicnum;

    ifstream ifs(filename); // ファイル読み込み
	if (ifs.fail()) {
	   cerr << "Cannot open file\n";
	   exit(0);
	}
	string str;
    string atom_id; //! 原子番号
    std::vector<int> atomic_num; //! 原子番号のリスト 
	Eigen::Vector3d tmp_position; //! 原子座標
    std::vector<Eigen::Vector3d> positions; //! 原子座標のリスト
    std::vector<Atoms> atoms_list; //! Atomsのリスト
    int counter = 1;
	double x_temp, y_temp, z_temp;
	while (getline(ifs,str)) {
	    stringstream ss(str);
        if (counter % (NUM_ATOM+2) != 1 && counter % (NUM_ATOM+2) != 2){ // 最初の2行は飛ばす．
	        ss >> atom_id >> x_temp >> y_temp >> z_temp;
            tmp_position = Eigen::Vector3d(x_temp, y_temp, z_temp);
            positions.push_back(tmp_position);
            atomic_num.push_back(atomicnum.atomicnum[atom_id]); // 原子種から原子番号へ変換
        }
        if (counter % (NUM_ATOM+2) == 0){ //最後の原子を読み込んだら，Atomsを作成
            Atoms tmp_atoms = Atoms(atomic_num, positions, unitcell_vec, {true,true,true});
            atoms_list.push_back(tmp_atoms);
            atomic_num.clear(); // vectorのクリア
            positions.clear();
        }
        counter += 1;
	}	    		
        //     if counter >= 2:
        //     # print(counter-2, lines) # debug
        //     symbol, x, y, z = lines.split()[:4]
        //     symbol = symbol.lower().capitalize()
        //     symbols[counter-2] = symbol
        //     positions[counter-2] = [float(x), float(y), float(z)]
        // if counter == NUM_ATOM+1:
        //     # print(" break !! ", lines) # debug
        //     break 
    return atoms_list;
}

class read_mol{
    /*
    pythonの同名クラスと違い，別途ファイルからボンド情報を読み込む．
    読み込んだボンドをクラス変数に格納する．
     - num_atoms_per_mol : 原子数
     - atom_list : 原子番号のリスト

    */
    public: // とりあえずPGの例を実装
        // 原子数の取得
        int num_atoms_per_mol= 13;
        // atom list（原子番号）
        std::vector<string> atom_list{"O","C","C","O","C","H","H","H","H","H","H","H","H"};
        // bonds_listの作成
        std::vector<std::vector<int> > bonds_list{{0, 1}, {0, 5}, {1, 2}, {1, 6}, {1, 7}, {2, 3}, {2, 4}, {2, 8}, {3, 9}, {4, 10}, {4, 11}, {4, 12}};
        // double_bondの作成
        std::vector<std::vector<int> > double_bonds; //(PGの場合は)空リスト
        // 各種ボンド
        std::vector<std::vector<int> > ch_bond;
        std::vector<std::vector<int> > co_bond;
        std::vector<std::vector<int> > oh_bond;
        std::vector<std::vector<int> > oo_bond;
        std::vector<std::vector<int> > cc_bond;
        std::vector<std::vector<int> > ring_bond; 
        // ローンペア
        std::vector<int> o_list;
        std::vector<int> n_list;
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
                vector<string> tmp = {atom_list[bond[0]], atom_list[bond[1]]};
                if (tmp == std::vector<string>{"H", "C"} || tmp == vector<string>{"C", "H"}) {
                    ch_bond.push_back(bond);
                }
                if (tmp == std::vector<string>{"O", "C"} || tmp == vector<string>{"C", "O"}) {
                    co_bond.push_back(bond);
                }
                if (tmp == std::vector<string>{"O", "H"} || tmp == vector<string>{"H", "O"}) {
                    oh_bond.push_back(bond);
                }
                if (tmp == std::vector<string>{"O", "O"}) {
                    oo_bond.push_back(bond);
                }
                if (tmp == std::vector<string>{"C", "C"}) { 
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

std::vector<Eigen::Vector3d> raw_calc_mol_coord_mic_onemolecule(std::vector<int> mol_inds, std::vector<std::vector<int>> bonds_list_j, Atoms aseatoms, read_mol itp_data) {
    /*
    1つの分子のpbc-mol計算を実施し，
        - 分子座標
    の計算を行う．
    mol_inds :: 分子部分を示すインデックス
    bonds_list_j :: 分子内のボンドのリスト

    */
   // TODO :: グラフ理論に基づいたボンドセンターの計算を行うためにraw_get_pbc_molを定義する．
   // 通常のraw_get_distances_micを使って，mol_inds[0]からmol_indsへの距離を計算する．
   // TODO :: mol_inds[0]になっているが本来はmol_inds[representative_atom_index]になるべき．
    std::vector<Eigen::Vector3d> vectors = raw_get_distances_mic(aseatoms, mol_inds[itp_data.representative_atom_index], mol_inds, true, true);
#ifdef DEBUG
    std::cout << "vectors..." << std::endl;
    for (int i = 0; i < vectors.size(); i++) {
        std::cout << std::setw(10) << vectors[i][0] << vectors[i][1] <<  vectors[i][2] << std::endl;
    }
#endif // !DEBUG
    // mol_inds[itp_data.representative_atom_index]の座標を取得する．
    Eigen::Vector3d R0 = aseatoms.get_positions()[mol_inds[itp_data.representative_atom_index]];

    // mol_indsを0から始まるように変換する．
    std::vector<int> mol_inds_from_zero;
    for (int i = 0; i < mol_inds.size(); i++) {
        mol_inds_from_zero.push_back(mol_inds[i] - mol_inds[0]);
    }
    // 同様にボンドリストが0から始まるように変換する．
    // TODO :: そもそもbonds_list_jを使う必要ないよね？ mol_indsから直接計算できるもんな．．．
    std::vector<std::vector<int>> bonds_list_from_zero;
    for (int i = 0; i < bonds_list_j.size(); i++) {
        // std::vector<int> bond;
        // bond.push_back(bonds_list_j[i][0] - mol_inds[0]);
        // bond.push_back(bonds_list_j[i][1] - mol_inds[0]);
        bonds_list_from_zero.push_back({bonds_list_j[i][0] - mol_inds[0],bonds_list_j[i][1] - mol_inds[0]});
    }
    // 分子の座標を再計算する．
    std::vector<Eigen::Vector3d> mol_coords;
    for (int k = 0; k < mol_inds_from_zero.size(); k++) {
        Eigen::Vector3d mol_coord = R0 + vectors[mol_inds_from_zero[k]];
        mol_coords.push_back(mol_coord);
    }
    return mol_coords;
}

std::vector<Eigen::Vector3d> raw_calc_bc_mic_onemolecule(std::vector<int> mol_inds, std::vector<std::vector<int>> bonds_list_j,     std::vector<Eigen::Vector3d> mol_coords) {
    /*
        すでに計算された分子座標を使って，ボンドセンターを計算する．
        - ボンドセンター座標
        TODO :: この関数もbonds_list_jを使う必要がない．ただのbonds_listから計算可能．
    */
    // ボンドリストが0から始まるように変換する．
    // TODO :: そもそもbonds_list_jを使う必要ないよね？ mol_indsから直接計算できるもんな．．．
    std::vector<std::vector<int>> bonds_list_from_zero;
    for (int i = 0; i < bonds_list_j.size(); i++) {
        std::vector<int> bond;
        bond.push_back(bonds_list_j[i][0] - mol_inds[0]);
        bond.push_back(bonds_list_j[i][1] - mol_inds[0]);
        bonds_list_from_zero.push_back(bond);
    }
    // ボンドセンターを計算する．
    std::vector<Eigen::Vector3d> bond_centers;
    for (int l = 0; l < bonds_list_from_zero.size(); l++) {
        // Eigen::Vector3d bc = R0 + (vectors.col(bonds_list_from_zero[l][0]) + vectors.col(bonds_list_from_zero[l][1])) / 2.0;
        Eigen::Vector3d bc = (mol_coords[bonds_list_from_zero[l][0]] + mol_coords[bonds_list_from_zero[l][1]]) / 2.0;
        if ((mol_coords[bonds_list_from_zero[l][0]] - mol_coords[bonds_list_from_zero[l][1]]).norm() > 2.0) { // bond length is too long
            std::cout << "WARNING :: bond length is too long !! :: " << bonds_list_from_zero[l][0] << " " << bonds_list_from_zero[l][1] << " " <<(mol_coords[bonds_list_from_zero[l][0]] - mol_coords[bonds_list_from_zero[l][1]]).norm() << std::endl;
        }
        bond_centers.push_back(bc);
    }
    return bond_centers;
}

std::tuple<std::vector<std::vector<Eigen::Vector3d> >, std::vector<std::vector<Eigen::Vector3d> > > raw_aseatom_to_mol_coord_and_bc(Atoms ase_atoms, std::vector<std::vector<int>> bonds_list, read_mol itp_data, int NUM_MOL_ATOMS, int NUM_MOL) {
    /*
    ase_atomsから，
     - 1: ボンドセンターの計算
     - 2: micを考慮した原子座標の再計算
    を行う．基本的にはcalc_mol_coordのwrapper関数
    
    input
    ------------
    ase_atoms       :: ase.atoms
    mol_ats         ::
    bonds_list      :: itpdataに入っているボンドリスト

    output
    ------------
    list_mol_coords :: 
    list_bond_centers
    
    NOTE
    ------------
    2023/4/16 :: inputとしていたunit_cell_bondsをより基本的な変数bond_listへ変更．
    bond_listは1分子内でのボンドの一覧であり，そこからunit_cell_bondsを関数の内部で生成する．
    */

    std::array<std::vector<std::vector<double>>, 2> result;
    std::vector<std::vector<Eigen::Vector3d> > list_mol_coords; 
    std::vector<std::vector<Eigen::Vector3d> > list_bond_centers; 
    
    // 1分子内のindexを取得する．
    std::vector<int> mol_at0(NUM_MOL_ATOMS);
    for (int i = 0; i < NUM_MOL_ATOMS; i++) {
        mol_at0[i] = i;
    }

    // 1config内のindexを取得する．
    std::vector<std::vector<int>> mol_ats(NUM_MOL, std::vector<int>(NUM_MOL_ATOMS));
    for (int indx = 0; indx < NUM_MOL; indx++) {
        for (int i = 0; i < NUM_MOL_ATOMS; i++) {
            mol_ats[indx][i] = i + NUM_MOL_ATOMS * indx;
        }
    }
    
    // bonds_listも1config内のものに対応させる．（NUM_MOL*bonds_list型のリスト）
    std::vector<std::vector<std::vector<int> > > unit_cell_bonds(NUM_MOL, std::vector<std::vector<int> >(bonds_list.size()));
    for (int indx = 0; indx < NUM_MOL; indx++) {
        for (int i = 0; i < bonds_list.size(); i++) {
            unit_cell_bonds[indx][i] = {bonds_list[i][0] + NUM_MOL_ATOMS * indx, bonds_list[i][1] + NUM_MOL_ATOMS * indx};
        }
    }
    
    // 
    for (int j = 0; j < NUM_MOL; j++) {    // NUM_MOL個の分子に対するLoop
        // std::vector<int> mol_inds = mol_ats[j];
        // std::vector<std::array<int, 2>> bonds_list_j = unit_cell_bonds[j];
        std::vector<Eigen::Vector3d> mol_coords = raw_calc_mol_coord_mic_onemolecule(mol_ats[j], unit_cell_bonds[j], ase_atoms, itp_data);
        std::vector<Eigen::Vector3d> bond_centers = raw_calc_bc_mic_onemolecule(mol_ats[j], unit_cell_bonds[j], mol_coords);
        list_mol_coords.push_back(mol_coords);
        list_bond_centers.push_back(bond_centers);
    }
    return {list_mol_coords,list_bond_centers};
}

Atoms make_ase_with_BCs(std::vector<int> ase_atomicnumber, int NUM_MOL, std::vector<std::vector<double> > UNITCELL_VECTORS, std::vector<std::vector<Eigen::Vector3d> > list_mol_coords, std::vector<std::vector<Eigen::Vector3d> > list_bond_centers){
    /*
    list_mol_coordsとlist_bond_centersから新しいase.atomsを作成する．
    現状だと単にase.atomsを作成するだけで，用途としてはase.io.writeで保存するくらいだ．

    注意点として，push_backする順番に気をつけたい．文仕事に，原子，ボンドセンター(He割り当て)の順で割り当てる．
    基本的に，普通の扱いでは分けている分子ごとの部分をフラットにするだけ．

    TODO :: ase_atomicnumberとNUM_MOLの扱いは難しい．一つの分子のみ（混合溶液でない）ならread_molから原子のリストを持ってきても良い．
    TODO :: しかし，もっとも一般的なのはase_atoms.get_atomic_numbers()を使って一つづつappendしていくこと．
    */
    //Atoms make_ase_with_BCs( ){
    // std::vector<std::vector<Eigen::Vector3d> > list_mol_coords;
    // std::vector<std::vector<Eigen::Vector3d> > list_bond_centers;

    std::vector<Eigen::Vector3d> new_coord;
    std::vector<int> new_atomic_num;
    // std::vector<std::vector<int>> list_atomic_nums = ase_atomicnumber.reshape(NUM_MOL, -1);
    for (int i = 0; i < list_mol_coords.size(); i++) {    //分子数に関するLoop
        for (int j = 0; j < list_mol_coords[i].size(); j++) {    //分子内の原子数に関するLoop
            new_coord.push_back(list_mol_coords[i][j]);
            new_atomic_num.push_back(ase_atomicnumber[i*list_mol_coords.size()+j]);
        }
        for (int j = 0; j < list_bond_centers[i].size(); j++) {    //分子内のボンドセンター数に関するLoop
            new_coord.push_back(list_bond_centers[i][j]);
            new_atomic_num.push_back(2);
        }
    }
    // Atomsを作成する．    
    Atoms new_atoms(new_atomic_num, new_coord, UNITCELL_VECTORS, {1,1,1});
    return new_atoms;
}

int ase_io_write(std::vector<Atoms> atoms_list, string filename ){
    /*
    TODO :: configurationが一つの場合にどうするかはちょっと問題か．
    */
    ofstream fout(filename); 
    // まず2行目の変な文字列を取得
    string two_line="Properties=species:S:1:pos:R:3 pbc=\"T T T\"";
    // Lattice="15.389699935913086 0.0 0.0 0.0 15.389699935913086 0.0 0.0 0.0 15.389699935913086" Properties=species:S:1:pos:R:3 pbc="T T T"

    // 原子番号から
    Atomicchar atomicchar;
    // 
    // 2行目以降の部分をファイルへ出力。
    for (int i = 0; i < atoms_list.size(); i++) {
        std::vector<Eigen::Vector3d> coords = atoms_list[i].get_positions(); // TODO :: ポインタ化
        std::vector<int> atomic_num = atoms_list[i].get_atomic_numbers();   // TODO ::ポインタ化

        fout << atomic_num.size() << std::endl; //1行目の原子数
        fout << "Lattice=\""; // 2行目のLattice=
        for (int cart_1 = 0; cart_1 < 3; cart_1++){ // ２行目の単位格子ベクトル
            for (int cart_2 = 0; cart_2 < 3; cart_2++){
                fout <<  atoms_list[i].get_cell()[cart_1][cart_2];
                if (cart_1 == 2 && cart_2 == 2){ fout << "\"";};
                fout << " ";
            }
        }
        fout << two_line << std::endl; // 2行目の変な文字列
        for (int j = 0; j < atomic_num.size(); j++){
            fout << std::left << std::setw(2) << atomicchar.atomicchar[atomic_num[j]];
            fout << std::right << std::setw(16) << coords[j][0] << std::setw(16) << coords[j][1] << std::setw(16) << coords[j][2] << endl;
        }
    }
    return 0;
}


Atoms raw_make_atoms_with_bc(Eigen::Vector3d bond_center, Atoms atoms, std::vector<std::vector<double> > UNITCELL_VECTORS){
    /*
    ######INPUTS#######
    bond_center     # Eigen::Vector3d 記述子を求めたい結合中心の座標
    list_mol_coords # array  分子ごとの原子座標
    list_atomic_nums #array  分子ごとの原子座標
    */
    std::vector<Eigen::Vector3d> list_mol_coords=atoms.get_positions();
    std::vector<int> list_atomic_num=atoms.get_atomic_numbers();

    // 結合中点bond_centerを先頭においたAtomsオブジェクトを作成する
    list_mol_coords.insert(list_mol_coords.begin(), bond_center);
    list_atomic_num.insert(list_atomic_num.begin(), 79); // 結合中心のラベルはAu(79)とする
    Atoms WBC = Atoms(list_atomic_num,
                    list_mol_coords,        
                    UNITCELL_VECTORS,   
                    {1, 1, 1});
    return WBC;
};

std::vector<Eigen::Vector3d> find_specific_bondcenter(std::vector<std::vector<Eigen::Vector3d> > list_bond_centers, std::vector<int> bond_index){
    /*
    list_bond_centersからbond_index情報をもとに特定のボンド（CHなど）だけ取り出す．
    TODO :: ここは将来的にはボンドをあらかじめ分割するようにして無くしてしまいたい．
    * @param[in] bond_index : 分子内のボンドセンターのindex（read_mol.ch_bond_indexなど．）
    * @return cent_mol : 特定のボンドの原子座標（(-1,3)型）
    */
    std::vector<Eigen::Vector3d> cent_mol;
    // ボンドセンターの座標と双極子をappendする．
    for (int i = 0; i < list_bond_centers.size(); i++) { //UnitCellの分子ごとに分割
        // chボンド部分（chボンドの重心をappend）
        for (int j = 0; j < bond_index.size(); j++) {    //分子内のボンドセンター数に関するLoop
            std::cout << " print list_bond_centers[i][bond_index[j]] :: " << list_bond_centers[i][bond_index[j]][0] << std::endl;
            cent_mol.push_back(list_bond_centers[i][bond_index[j]]);
        }
    }
    return cent_mol;
    };

double fs(double Rij,double Rcs,double Rc){
    /*
    #####Inputs####
    # Rij : float 原子間距離 [ang. unit] 
    # Rcs : float inner cut off [ang. unit]
    # Rc  : float outer cut off [ang. unit] 
    ####Outputs####
    # sij value 
    ###############
    */
    double s;
    if (Rij < Rcs){
        s = 1/Rij;
    } else if(Rij < Rc){
        s = (1/Rij)*(0.5*cos(M_PI*(Rij-Rcs)/(Rc-Rcs))+0.5);
    } else{
        s = 0;
    }
    return s;
}

std::vector<double> calc_descripter(std::vector<Eigen::Vector3d> dist_wVec,std::vector<int> atoms_index,double Rcs,double Rc,int MaxAt){
    /*
    ある原子種に対する記述子を作成する．
    input
    -----------
    * @param[in] dist_wVec :: ある原子種からの全ての原子に対する相対ベクトル．
    * @param[in] atoms_index :: 自分で指定したindexのみを考える
    TODO :: 現状intraとinterを分けているのでこうなっているが，その区別がいらないならinputは"C"とかにするとわかりやすい．
    MaxAt :: 最大の原子数
    */
    // 相対ベクトルdist_wVecのうち，自分自身との距離0を除き，かつatoms_indexに含まれるものだけを取り出す．
    // TODO :: もしdrsの中に0のものがあったらそれを排除したい．これはこれはローンペア計算の時に重要
    std::vector<Eigen::Vector3d> drs;
    for (int l = 1; l < atoms_index.size(); l++){ // 0は自分自身なので除外して1スタートとする．
        drs.push_back(dist_wVec[atoms_index[l]]);
    }
    // drsの絶対値を求める．
    std::vector<double> drs_abs;
    for (int j = 0; j < drs.size(); j++){
        drs_abs.push_back(drs[j].norm());
    }
    // カットオフ関数fsをかけた1/drsを計算する．
    std::vector<double> drs_inv;
    for (int j = 0; j < drs.size(); j++){
        drs_inv.push_back(fs(drs_abs[j],Rcs,Rc));
    }
    // 記述子（f(1,x/r,y/r,z/r)）を計算する．4次元ベクトルの配列
    std::vector<std::vector<double> > dij;
    for (int j = 0; j < drs.size(); j++){
        dij.push_back({drs_inv[j], drs_inv[j]*drs[j][0]/drs_abs[j], drs_inv[j]*drs[j][1]/drs_abs[j], drs_inv[j]*drs[j][2]/drs_abs[j]});
    }
    // dijを要素drs_invの大きい順（つまりdrsが小さい順）にソートする．要素としては4次元ベクトルの第0成分でのソート
    // https://qiita.com/Arusu_Dev/items/c36cdbc41fc77531205c
    std::sort(dij.begin(), dij.end(), [](const vector<double> &alpha,const vector<double> &beta){return alpha[0] < beta[0];});
    // 原子数がMaxAtよりも少なかったら０埋めして固定長にする。1原子あたり4要素(1,x/r,y/r,z/r)なので4*MaxAtの要素だけ保守する．
    std::vector<double> dij_desc;
    if (dij.size() < MaxAt){
        for (int i = 0; i< dij.size(); i++){
            dij_desc.push_back(dij[i][0]);
            dij_desc.push_back(dij[i][1]);
            dij_desc.push_back(dij[i][2]);
            dij_desc.push_back(dij[i][3]);
        }
        for (int i = 0; i < MaxAt - dij.size(); i++){ 
            dij_desc.push_back(0); // 0埋め
            dij_desc.push_back(0);
            dij_desc.push_back(0);
            dij_desc.push_back(0);
        }
    } else {
        for (int i = 0; i< MaxAt; i++){
            dij_desc.push_back(dij[i][0]);
            dij_desc.push_back(dij[i][1]);
            dij_desc.push_back(dij[i][2]);
            dij_desc.push_back(dij[i][3]);
        }
    }
    return dij_desc;
};

std::vector<double> raw_get_desc_bondcent(Atoms atoms, Eigen::Vector3d bond_center, int mol_id, std::vector<std::vector<double> > UNITCELL_VECTORS, int NUM_MOL_ATOMS){
    /*
    ボンドセンター用の記述子を作成
    TODO : 引数が煩雑すぎるので見直したい．mol_idが必要なのはあまり賢くない．
    ######Inputs########
    atoms : ASE atom object 構造の入力
    Rcs : float inner cut off [ang. unit]
    Rc  : float outer cut off [ang. unit] 
    MaxAt : int 記述子に記載する原子数（これにより固定長の記述子となる）
    #bond_center : vector 記述子を計算したい結合の中心
    mol_id : 対象のbond_centerがどの分子に属するかを示す． 
    ######Outputs#######
    Desc : 原子番号,[List O原子のSij x MaxAt : H原子のSij x MaxAt] x 原子数 の二次元リストとなる
    */
    double Rcs = 4.0; // [ang. unit] TODO : hard code
    double Rc  = 6.0; // [ang. unit] TODO : hard code
    double MaxAt = 12; // とりあえずは12個の原子で良いはず．
    // ボンドセンターを追加したatoms
    Atoms atoms_w_bc = raw_make_atoms_with_bc(bond_center,atoms, UNITCELL_VECTORS);
    // ボンドセンターが含まれている分子の原子インデックスを取得
    std::vector<int> atoms_in_molecule(NUM_MOL_ATOMS);
    for (int i = 0; i < NUM_MOL_ATOMS; i++){
        atoms_in_molecule[i] = i + mol_id*NUM_MOL_ATOMS + 1; //結合中心を先頭に入れたAtomsなので+1が必要．
    }
    // 各原子の記述子を作成するにあたり，原子のindexを計算する．
    std::vector<int> atomic_numbers = atoms_w_bc.get_atomic_numbers();
    std::vector<int> Catoms_intra;
    std::vector<int> Catoms_inter;
    std::vector<int> Hatoms_intra;
    std::vector<int> Hatoms_inter;
    std::vector<int> Oatoms_intra;
    std::vector<int> Oatoms_inter;


    // こちらの関数ではintraとinterで分けているが，allinoneでは分けない．
    for (int i = 0; i < atomic_numbers.size();i++){
        bool if_bc_in_molecule = std::find(atoms_in_molecule.begin(), atoms_in_molecule.end(), i) != atoms_in_molecule.end();
        if        (atomic_numbers[i] == 6 && if_bc_in_molecule){
            Catoms_intra.push_back(i);
        } else if (atomic_numbers[i] == 6 && !(if_bc_in_molecule)){
            Catoms_inter.push_back(i);
        } else if (atomic_numbers[i] == 1 && if_bc_in_molecule){
            Hatoms_intra.push_back(i);
        } else if (atomic_numbers[i] == 1 && !(if_bc_in_molecule)){
            Hatoms_inter.push_back(i);
        } else if (atomic_numbers[i] == 8 && if_bc_in_molecule){
            Oatoms_intra.push_back(i);
        } else if (atomic_numbers[i] == 8 && !(if_bc_in_molecule)){
            Oatoms_inter.push_back(i);
        }
    }
    // 全ての原子との距離を求める．この際0-0間距離も含まれる．
    std::vector<int> range_at_list(atomic_numbers.size()); // (0~atomic_numbersまでの整数を格納．numpy.arangeと同等．)
    std::iota(range_at_list.begin(),range_at_list.end(), 0); // https://codezine.jp/article/detail/8778
    auto dist_wVec = raw_get_distances_mic(atoms_w_bc,0, range_at_list, true, true) ;

    // for C atoms (intra) 
    auto dij_C_intra=calc_descripter(dist_wVec, Catoms_intra, Rcs,Rc,MaxAt);
    // for H atoms (intra)
    auto dij_H_intra=calc_descripter(dist_wVec, Hatoms_intra, Rcs,Rc,MaxAt);
    // for O  atoms (intra)
    auto dij_O_intra=calc_descripter(dist_wVec, Oatoms_intra, Rcs,Rc,MaxAt);
    // for C atoms (inter)
    auto dij_C_inter=calc_descripter(dist_wVec, Catoms_inter, Rcs,Rc,MaxAt);
    // for H atoms (inter)
    auto dij_H_inter=calc_descripter(dist_wVec, Hatoms_inter,Rcs,Rc,MaxAt);
    // for O atoms (inter)
    auto dij_O_inter=calc_descripter(dist_wVec, Oatoms_inter,Rcs,Rc,MaxAt);
    // 連結する dij_C_intra+dij_H_intra+dij_O_intra+dij_C_inter+dij_H_inter+dij_O_inter
    dij_C_intra.insert(dij_C_intra.end(), dij_H_intra.begin(), dij_H_intra.end()); // 連結
    dij_C_intra.insert(dij_C_intra.end(), dij_O_intra.begin(), dij_O_intra.end()); // 連結
    dij_C_intra.insert(dij_C_intra.end(), dij_C_inter.begin(), dij_C_inter.end()); // 連結
    dij_C_intra.insert(dij_C_intra.end(), dij_H_inter.begin(), dij_H_inter.end()); // 連結
    dij_C_intra.insert(dij_C_intra.end(), dij_O_inter.begin(), dij_O_inter.end()); // 連結

    return dij_C_intra;
};

std::vector<double> raw_get_desc_bondcent_allinone(Atoms atoms, Eigen::Vector3d bond_center, int mol_id, std::vector<std::vector<double> > UNITCELL_VECTORS, int NUM_MOL_ATOMS){
    /*
    ボンドセンター用の記述子を作成
    TODO : 引数が煩雑すぎるので見直したい．mol_idが必要なのはあまり賢くない．
    ######Inputs########
    atoms : ASE atom object 構造の入力
    Rcs : float inner cut off [ang. unit]
    Rc  : float outer cut off [ang. unit] 
    MaxAt : int 記述子に記載する原子数（これにより固定長の記述子となる）
    #bond_center : vector 記述子を計算したい結合の中心
    mol_id : 対象のbond_centerがどの分子に属するかを示す． 
    ######Outputs#######
    Desc : 原子番号,[List O原子のSij x MaxAt : H原子のSij x MaxAt] x 原子数 の二次元リストとなる
    */
    double Rcs = 4.0; // [ang. unit] TODO : hard code
    double Rc  = 6.0; // [ang. unit] TODO : hard code
    double MaxAt = 24; // !! ここは分子内外を分ける場合の2倍の値としておくのが安心．12*2=24
    // ボンドセンターを追加したatoms
    Atoms atoms_w_bc = raw_make_atoms_with_bc(bond_center,atoms, UNITCELL_VECTORS);
    // ボンドセンターが含まれている分子の原子インデックスを取得
    std::vector<int> atoms_in_molecule(NUM_MOL_ATOMS);
    for (int i = 0; i < NUM_MOL_ATOMS; i++){
        atoms_in_molecule[i] = i + mol_id*NUM_MOL_ATOMS + 1; //結合中心を先頭に入れたAtomsなので+1が必要．
    }
    // 各原子の記述子を作成するにあたり，原子のindexを計算する．
    std::vector<int> atomic_numbers = atoms_w_bc.get_atomic_numbers();
    std::vector<int> Catoms_all;
    std::vector<int> Catoms_inter;
    std::vector<int> Hatoms_all;
    std::vector<int> Hatoms_inter;
    std::vector<int> Oatoms_all;
    std::vector<int> Oatoms_inter;

    // atoms_w_bcの中での各原子種類のindexを取得する
    for (int i = 0; i < atomic_numbers.size();i++){
        if        (atomic_numbers[i] == 6 ){ 
            Catoms_all.push_back(i);
        } else if (atomic_numbers[i] == 1 ){
            Hatoms_all.push_back(i);
        } else if (atomic_numbers[i] == 8 ){
            Oatoms_all.push_back(i);
        } 
    }
    // 全ての原子との距離を求める．この際0-0間距離も含まれる．
    std::vector<int> range_at_list(atomic_numbers.size()); // (0~atomic_numbersまでの整数を格納．numpy.arangeと同等．)
    std::iota(range_at_list.begin(),range_at_list.end(), 0); // https://codezine.jp/article/detail/8778
    auto dist_wVec = raw_get_distances_mic(atoms_w_bc,0, range_at_list, true, true) ;

    // for C atoms (all) 
    auto dij_C_all=calc_descripter(dist_wVec, Catoms_all, Rcs,Rc,MaxAt);
    // for H atoms (all)
    auto dij_H_all=calc_descripter(dist_wVec, Hatoms_all, Rcs,Rc,MaxAt);
    // for O  atoms (all)
    auto dij_O_all=calc_descripter(dist_wVec, Oatoms_all, Rcs,Rc,MaxAt);

    // 連結する dij_C_all+dij_H_all+dij_O_all
    dij_C_all.insert(dij_C_all.end(), dij_H_all.begin(), dij_H_all.end()); // 連結
    dij_C_all.insert(dij_C_all.end(), dij_O_all.begin(), dij_O_all.end()); // 連結

    return dij_C_all;
};


std::vector<std::vector<double> > raw_calc_bond_descripter_at_frame(Atoms atoms_fr, std::vector<std::vector< Eigen::Vector3d> > list_bond_centers, std::vector<int> bond_index, int NUM_MOL, std::vector<std::vector<double> > UNITCELL_VECTORS, int NUM_MOL_ATOMS, string desctype){
    /* 
    1つのframe中の全てのボンドの記述子を計算する
     ! note
    ---------------
        2023/7/21 : 最後の変数desctypeでどの形の記述子を使うかを指定する
    */
    std::vector<std::vector<double> > Descs;
    if (bond_index.size() != 0){  // bond_indexが0でなければ計算を実行
        auto cent_mol   = find_specific_bondcenter(list_bond_centers, bond_index); // 特定ボンドのBCの座標だけ取得
        if (desctype == "allinone") {
            for (int i = 0; i < cent_mol.size(); i++){
                int mol_id = i % NUM_MOL; // len(bond_index) # 対応する分子ID（mol_id）を出すように書き直す．ボンドが1分子内に複数ある場合，その数で割らないといけない．（メタノールならCH結合が3つあるので3でわる）
                auto dij = raw_get_desc_bondcent_allinone(atoms_fr, cent_mol[i], mol_id, UNITCELL_VECTORS, NUM_MOL_ATOMS);
                Descs.push_back(dij);
            }
        } else if (desctype == "old") {
            for (int i = 0; i < cent_mol.size(); i++){
                int mol_id = i % NUM_MOL; // len(bond_index) # 対応する分子ID（mol_id）を出すように書き直す．ボンドが1分子内に複数ある場合，その数で割らないといけない．（メタノールならCH結合が3つあるので3でわる）
                auto dij = raw_get_desc_bondcent(atoms_fr, cent_mol[i], mol_id, UNITCELL_VECTORS, NUM_MOL_ATOMS);
                Descs.push_back(dij);
            }
        }
    }
    return Descs;
};


void write_2dvector_csv(){
    /*
    2dvectorをcsvファイルに書き込む
    */
}


int main() {
    vector<int> vec;
    
	vec.push_back(5);
	vec.push_back(10);
	vec.push_back(15);
	
	Vector vect(vec);
	vect.print();
	// 5 10 15 

    //! test for sign
    std::cout << " test for sign !!" << std::endl;
    std::cout << sign(-16) << std::endl;
    std::cout << sign(16) << std::endl;


    std::vector<int> test_num{1,2,3};
    std::vector<Eigen::Vector3d > test_positions{{1,2,3},{4,5,6},{7,8,9}};
    std::vector<std::vector<double> > UNITCELL_VECTORS{{16.267601013183594,0,0},{0,16.267601013183594,0},{0,0,16.267601013183594}};
    std::vector<bool> pbc{1,1,1};

    //! test for constructing Atoms
    Atoms atoms(test_num,test_positions, UNITCELL_VECTORS,pbc);
    std::cout << "this is atomic num " << atoms.atomic_num[0] << endl;
    std::cout << atoms.get_atomic_numbers()[0] << endl;

    //! test raw_get_distances_mic
    std::vector<Eigen::Vector3d> answer = raw_get_distances_mic(atoms,0,{1,2},true,true);
    for (int i = 0; i < answer.size(); i++) {
        for (int j = 0; j < answer[i].size(); j++) {
            std::cout << answer[i][j] << " ";
        }
        std::cout << endl;
    }

    //! test for get_atomic_num
    std::cout << "test for get_atomic_num " << std::endl;
    int atomic_num = raw_cpmd_num_atom("gromacs_30.xyz");
    std::cout << atomic_num << std::endl;

    //! test for raw_cpmd_get_unitcell_xyz
    std::vector<std::vector<double> > unitcell_vec = raw_cpmd_get_unitcell_xyz("pg_gromacs.xyz");
    for (int i = 0; i < unitcell_vec.size(); i++) {
        for (int j = 0; j < unitcell_vec[i].size(); j++) {
            std::cout << unitcell_vec[i][j] << " ";
        }
        std::cout << endl;
    }

    //! test for ase_io_read
    // 注意 :: gromacs_30.xyz はメタノール，pg_gromacs.xyがPG
    std::vector<Atoms> atoms_list = ase_io_read("pg_gromacs.xyz", atomic_num, unitcell_vec);
    std::cout << "this is atomic num " << atoms_list[0].get_atomic_numbers().size() << endl;
    // for (int j = 0; j < atoms_list[1].get_atomic_numbers().size(); j++) {
    //     std::cout << atoms_list[1].get_atomic_numbers()[j] << " " << atoms_list[1].get_positions()[j][0] << std::endl;
    // }

    //! test for read_mol
    read_mol test_read_mol;
    // print test_read_mol.ch_bond_index
    for (int i = 0; i < test_read_mol.ch_bond_index.size(); i++) {
        std::cout << test_read_mol.ch_bond_index[i] << " ";
    }
    // print test_read_mol.ch_bond
    for (int i = 0; i < test_read_mol.ch_bond.size(); i++) {
        std::cout << test_read_mol.ch_bond[i][0] << " " << test_read_mol.ch_bond[i][1] << std::endl;
    }

    //! test raw_aseatom_to_mol_coord_and_bc
    int NUM_MOL_ATOMS = test_read_mol.num_atoms_per_mol;
    int NUM_ATOM = atomic_num;
    int NUM_MOL = int(NUM_ATOM/NUM_MOL_ATOMS); // UnitCell中の総分子数
    auto test_mol_bc = raw_aseatom_to_mol_coord_and_bc(atoms_list[0], test_read_mol.bonds_list, test_read_mol, NUM_MOL_ATOMS, NUM_MOL);
    auto test_mol=std::get<0>(test_mol_bc);
    auto test_bc =std::get<1>(test_mol_bc);

    //! test make_ase_with_BCs
    Atoms new_atoms = make_ase_with_BCs(atoms_list[0].get_atomic_numbers(), NUM_MOL, raw_cpmd_get_unitcell_xyz("pg_gromacs.xyz"), test_mol, test_bc);

    //! test ase_io_write
    std::vector<Atoms> test = {new_atoms};
    ase_io_write(test, "test.xyz");

    //! test ase_io_write(作成したtest.xyzが読み込み可能かどうか)
    std::vector<Atoms> test_read = ase_io_read("test.xyz", raw_cpmd_num_atom("test.xyz"), raw_cpmd_get_unitcell_xyz("test.xyz"));

    //! test raw_calc_bond_descripter_at_frame (まずは適当なボンド記述子が作成できるかどうか)
    auto descs_ch = raw_calc_bond_descripter_at_frame(atoms_list[0], test_bc, test_read_mol.ch_bond_index, NUM_MOL, UNITCELL_VECTORS,  NUM_MOL_ATOMS);

    //! test for save as npy file.
    // descs_chの形を1dへ変形してnpyで保存．
    // TODO :: さすがにもっと効率の良い方法があるはず．
    std::vector<double> descs_ch_1d;
    for (int i = 0; i < descs_ch.size(); i++) {
        for (int j = 0; j < descs_ch[i].size(); j++) {
            descs_ch_1d.push_back(descs_ch[i][j]);
        }
    }
    //! npy.hppを利用して保存する．
    const std::vector<long unsigned> shape_descs_ch{descs_ch.size(), descs_ch[0].size()}; // vectorを1*12の形に保存
    npy::SaveArrayAsNumpy("descs_ch.npy", false, shape_descs_ch.size(), shape_descs_ch.data(), descs_ch_1d);
    
    // vectorをnpyへ保存（1次元の例） http://mglab.blogspot.com/2014/03/numpyhpp-npy.html
    const bool fortran_order{false}; // fortranの場合は配列の順番が逆になる．
    const std::vector<double> data1{1, 2, 3, 4, 5, 6};
    aoba::SaveArrayAsNumpy("1dvector.npy", 6, &data1[0]);

    // vectorをnpyへ保存（2次元の例） http://mglab.blogspot.com/2014/03/numpyhpp-npy.html
    const std::vector<std::vector<double> > data2{{1,11}, {2,12}, {3,13}, {4,14}, {5,15}, {6,16}};
    aoba::SaveArrayAsNumpy("2dvector.npy", 6, 2, &data2[0][0]);

    const std::vector<long unsigned> shape{1, 6}; // vectorを2*3の形に保存
    std::cout <<  shape.size() << " " << shape.data() << std::endl;
    // const std::vector<double> data3{1, 2, 3, 4, 5, 6};
    npy::SaveArrayAsNumpy("1dvector_v2.npy", fortran_order, shape.size(), shape.data(), data1);

    // !! このようにもともと2次元のデータを保存することはできない．
    // const std::vector<long unsigned> shape2{1, 12}; // vectorを1*12の形に保存
    // npy::SaveArrayAsNumpy("2dvector_v2.npy", fortran_order, shape2.size(), shape2.data(), data2);

    // !! numpy_quiitaを使う場合．ただしこれは現状使えてない．
    // SaveNpy("1dvector_v3.npy", data1);

    // したがって，最初に2次元データを1次元化する．（この時の順番が大事）
    // !! 以下のように自然にデータを1d化すればちゃんと2次元データとして保存できることがわかった．
    std::vector<double> data2_1d;
    for (int i = 0; i < data2.size(); i++) {
        for (int j = 0; j < data2[i].size(); j++) {
            data2_1d.push_back(data2[i][j]);
        }
    }
    const std::vector<long unsigned> shape3{6, 2}; // vectorを1*12の形に保存
    npy::SaveArrayAsNumpy("2dvector_v2.npy", fortran_order, shape3.size(), shape3.data(), data2_1d);
}