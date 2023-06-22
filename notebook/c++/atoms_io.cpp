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
#include "atoms_core.cpp"



/*
ase_io_readとase_io_writeを定義するファイル．
*/

int raw_cpmd_num_atom(std::string filename){
    /*
    xyzファイルから原子数を取得する．（ワニエセンターが入っている場合その原子数も入ってしまうので注意．）
    */
    std::ifstream ifs(filename); // ファイル読み込み
    if (ifs.fail()) {
       std::cerr << "Cannot open file\n";
       exit(0);
    }
    int NUM_ATOM;
    std::string str;
    int counter = 0;
    getline(ifs,str);
    std::stringstream ss(str);
    ss >> NUM_ATOM;
    return NUM_ATOM;
};


std::vector<std::vector<double> > raw_cpmd_get_unitcell_xyz(std::string filename = "IONS+CENTERS.xyz") {
    /*
    xyzファイルから単位格子ベクトルを取得する．

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
    std::vector<std::vector<double> > unitcell_vec(3, std::vector<double> (3, 0)); // ここで3*3の形に指定しないとダメだった．
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

    std::ifstream ifs(filename); // ファイル読み込み
	if (ifs.fail()) {
	   std::cerr << "Cannot open file\n";
	   exit(0);
	}
	std::string str;
    std::string atom_id; //! 原子番号
    std::vector<int> atomic_num; //! 原子番号のリスト 
	Eigen::Vector3d tmp_position; //! 原子座標
    std::vector<Eigen::Vector3d> positions; //! 原子座標のリスト
    std::vector<Atoms> atoms_list; //! Atomsのリスト
    int counter = 1;
	double x_temp, y_temp, z_temp;
	while (getline(ifs,str)) {
	    std::stringstream ss(str);
        if (counter % (NUM_ATOM+2) != 1 && counter % (NUM_ATOM+2) != 2){ // 最初の2行は飛ばす．
	        ss >> atom_id >> x_temp >> y_temp >> z_temp;
            tmp_position = Eigen::Vector3d(x_temp, y_temp, z_temp);
            positions.push_back(tmp_position);
            atomic_num.push_back(atomicnum.atomicnum.at(atom_id)); // 原子種から原子番号へ変換 // https://qiita.com/_EnumHack/items/f462042ec99a31881a81
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

std::vector<Atoms> ase_io_read(std::string filename){
    /*
    大元のase_io_read関数のオーバーロード版．ファイル名を入力にするだけで
    */
    return ase_io_read(filename, raw_cpmd_num_atom(filename), raw_cpmd_get_unitcell_xyz(filename));
}

int ase_io_write(std::vector<Atoms> atoms_list, std::string filename ){
    /*
    TODO :: configurationが一つの場合にどうするかはちょっと問題か．
    */
    std::ofstream fout(filename); 
    // まず2行目の変な文字列を取得
    std::string two_line="Properties=species:S:1:pos:R:3 pbc=\"T T T\"";
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
            fout << std::right << std::setw(16) << coords[j][0] << std::setw(16) << coords[j][1] << std::setw(16) << coords[j][2] << std::endl;
        }
    }
    return 0;
};


int ase_io_write(Atoms aseatoms, std::string filename ){
    /*
    ase_io_writeの別バージョン（オーバーロード）
    入力がaseatomsひとつだけだった場合にどうなるかのチェック．
    */
    std::ofstream fout(filename); 
    // まず2行目の変な文字列を取得
    std::string two_line="Properties=species:S:1:pos:R:3 pbc=\"T T T\"";
    // Lattice="15.389699935913086 0.0 0.0 0.0 15.389699935913086 0.0 0.0 0.0 15.389699935913086" Properties=species:S:1:pos:R:3 pbc="T T T"
    // 原子番号から
    Atomicchar atomicchar;
    // 
    std::vector<Eigen::Vector3d> coords = aseatoms.get_positions(); // TODO :: ポインタ化
    std::vector<int> atomic_num = aseatoms.get_atomic_numbers();   // TODO ::ポインタ化

    fout << atomic_num.size() << std::endl; //1行目の原子数
    fout << "Lattice=\""; // 2行目のLattice=
    for (int cart_1 = 0; cart_1 < 3; cart_1++){ // ２行目の単位格子ベクトル
        for (int cart_2 = 0; cart_2 < 3; cart_2++){
            fout <<  aseatoms.get_cell()[cart_1][cart_2];
            if (cart_1 == 2 && cart_2 == 2){ fout << "\"";};
            fout << " ";
        }
    }
    fout << two_line << std::endl; // 2行目の変な文字列
    for (int j = 0; j < atomic_num.size(); j++){
        fout << std::left << std::setw(2) << atomicchar.atomicchar.at(atomic_num[j]); // https://qiita.com/_EnumHack/items/f462042ec99a31881a81
        fout << std::right << std::setw(16) << coords[j][0] << std::setw(16) << coords[j][1] << std::setw(16) << coords[j][2] << std::endl;
    }
    return 0;
};