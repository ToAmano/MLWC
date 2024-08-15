// https://github.com/microsoft/vscode-cpptools/issues/7413
#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

// #define _DEBUG
#include <stdio.h>
#include <filesystem>
#include <iomanip>
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
#include "atoms_core.hpp"
#include "atoms_io.hpp"


/*
ase_io_readとase_io_writeを定義するファイル．
*/

bool isNumber(const std::string& str) {
    std::istringstream iss(str);
    double num;
    iss >> std::noskipws >> num; // 'noskipws' ensures that the entire string is parsed
    return iss.eof() && !iss.fail();
}



int raw_cpmd_num_atom(const std::string filename){
    /**
     @fn xyzファイルから原子数を取得する．（ワニエセンターが入っている場合その原子数も入ってしまうので注意．）
     @fn 基本的には1行目の数字を取得しているだけ．
    */
    std::ifstream ifs(std::filesystem::absolute(filename)); // ファイル読み込み
    if (ifs.fail()) {
       std::cerr << "Cannot open xyz file\n";
       exit(0);
    }
    int NUM_ATOM;
    std::string str;
    getline(ifs,str);
    std::stringstream ss(str);
    ss >> NUM_ATOM;
    return NUM_ATOM;
};

int get_num_atom_without_wannier(const std::string filename){
    /*
    xyzファイルから原子数を取得する．
    ワニエセンターがある場合，それを取り除く．
    基本的には1行目の数字を取得しているだけ．
    */
    std::ifstream ifs(std::filesystem::absolute(filename)); // ファイル読み込み
    if (ifs.fail()) {
       std::cerr << " get_num_atom_without_wannier :: Cannot open xyz file\n";
       exit(0);
    }
    int NUM_ATOM_WITHOUT_WAN=0; // the number of atoms without WC

    if (filename.ends_with(".xyz")){
        std::cout << "xyz mode" << std::endl;
        // 先にxyzファイルから原子数を取得する．
        int NUM_ATOM = raw_cpmd_num_atom(std::filesystem::absolute(filename));
        std::string str;

        std::string atom_id; //! 原子番号
        int counter = 1; //! 行数カウンター
        int index_atom = 0; //! 読み込んでいる原子のインデックス
        double x_temp, y_temp, z_temp;
        while (getline(ifs,str)) { //!1ループで離脱する
            std::stringstream ss(str);
            index_atom = counter % (NUM_ATOM+2);
            if (index_atom == 1 || index_atom == 2){ // 最初の2行は飛ばす．
                counter += 1;
                continue;   
            }
            ss >> atom_id >> x_temp >> y_temp >> z_temp; // 読み込み
            if (atom_id != "X"){ // ワニエセンターの場合以外はNUM_ATOMカウンターをインクリメント
                NUM_ATOM_WITHOUT_WAN += 1;
            }
            if (index_atom == 0){ //最後の原子を読み込んだら，Atomsを作成
                break;
            }
            counter += 1;
        }	    		
    } else if (filename.ends_with(".lammpstrj")){
        //!! TODO :: not without wannier
        NUM_ATOM_WITHOUT_WAN = get_num_atom_without_wannier_lammps(filename);
        std::cout << "lammps mode" << std::endl;
    } else{
        std::cout << "ERROR filename " << std::endl;
    }
    return NUM_ATOM_WITHOUT_WAN;
};


int get_num_atom_without_wannier_lammps(const std::string filename) {
    /*
    xyzファイルから単位格子ベクトルを取得する．

     Lattice="16.267601013183594 0.0 0.0 0.0 16.267601013183594 0.0 0.0 0.0 16.267601013183594" Properties=species:S:1:pos:R:3 pbc="T T T"
    という文字列から，Lattice=" "の部分を抽出しないといけない．
    */
    std::ifstream ifs(filename);
    // https://qiita.com/yohm/items/7c82693b83d4c055fa7b
    // std::cout << "print match :: " << match.str() << std::endl; // DEBUG（ここまではOK．）
	std::string str;
    int counter = 1; //! 行数カウンター
    std::vector<std::string> unitcell_vec_str;
    std::vector<std::vector<double> > unitcell_vec(3, std::vector<double> (3, 0)); // ここで3*3の形に指定しないとダメだった．
	int num_atom;
	while (std::getline(ifs,str)) {
        if (str.find("ITEM: NUMBER OF ATOMS") != std::string::npos) { // get box
            std::getline(ifs, str); // for x
            num_atom = std::stoi(str);
            break;
        }
    }
    return num_atom;
}



std::vector<std::vector<double> > raw_cpmd_get_unitcell(const std::string filename) {
    if (filename.ends_with(".xyz")){
        return raw_cpmd_get_unitcell_xyz(filename);
    } else if (filename.ends_with(".lammpstrj"))
    {
        std::cout << "lammps mode" << std::endl;
        return raw_cpmd_get_unitcell_lammps(filename);
    }
    
}

Atoms read_xyz_frame(const std::string& filename, int index) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }
    //! test for Atomicnum
    Atomicnum atomicnum;
    int current_frame = 0;
    int num_atoms;
    std::string comment; // for comment line 
    std::string line;
    std::string atom_id; //! 原子番号
    std::vector<int> atomic_num; //! 原子番号のリスト 
    std::vector<Eigen::Vector3d> positions; //! 原子座標のリスト
    std::vector<Atoms> atoms_list; //! Atomsのリスト
    int counter = 1; //! 行数カウンター
    int index_atom = 0; //! 読み込んでいる原子のインデックス
	double x_temp, y_temp, z_temp;
	Eigen::Vector3d tmp_position; //! 原子座標

    while (current_frame <= index && std::getline(file, line)) {
        num_atoms = std::stoi(line);  // 原子数を取得
        std::getline(file, comment);  // コメント行を取得

        if (current_frame == index) {
            std::vector<std::string> atoms;
            positions.reserve(num_atoms);  // 必要なサイズを予約

            for (int i = 0; i < num_atoms; ++i) {
                std::getline(file, line);
            	std::stringstream ss(line);
                // position/atomic_numの読み込み
                ss >> atom_id >> x_temp >> y_temp >> z_temp;
                tmp_position = Eigen::Vector3d(x_temp, y_temp, z_temp);
                positions.push_back(tmp_position);
                atomic_num.push_back(atomicnum.atomicnum.at(atom_id)); // 原子種から原子番号へ変換 // https://qiita.com/_EnumHack/items/f462042ec99a31881a81
            }
            Atoms tmp_atoms = Atoms(atomic_num, positions, raw_cpmd_get_unitcell_xyz(filename), {true,true,true});
            return tmp_atoms;  // フレームを返す
        } else {
            // 指定されたフレームでない場合は読み飛ばす
            for (int i = 0; i < num_atoms; ++i) {
                std::getline(file, line);
            }
        }
        ++current_frame;
    }
    throw std::out_of_range("Frame index out of range");
}


Atoms read_lammps_frame(const std::string& filename, int index) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    int current_frame = -1;
    std::string line;
    int timestep = 0;
    int num_atoms = 0;
	std::string str;
	Eigen::Vector3d tmp_position; //! 原子座標
    std::unordered_map<std::string, int> column_indices; // 原子構造の読み込み
    std::string atom_id; //! 原子番号
    std::vector<int> atomic_num; //! 原子番号のリスト 
    std::vector<Eigen::Vector3d> positions; //! 原子座標のリスト
    std::vector<Atoms> atoms_list; //! Atomsのリスト
    int counter = 1; //! 行数カウンター
    int index_atom = 0; //! 読み込んでいる原子のインデックス
    std::vector<std::string> unitcell_vec_str;
    std::vector<std::vector<double> > unitcell_vec(3, std::vector<double> (3, 0)); // ここで3*3の形に指定しないとダメだった．
	double x_temp, y_temp, z_temp;
    bool flag_box;
    double box_xlo, box_xhi, box_ylo, box_yhi, box_zlo, box_zhi;

    while (std::getline(ifs, str)) {
        if (line.find("ITEM: TIMESTEP") != std::string::npos) {
            current_frame++;
            if (current_frame == index) {
                std::getline(ifs, line);
                timestep = std::stoi(line);
            }
        } else if (line.find("ITEM: NUMBER OF ATOMS") != std::string::npos) {
            if (current_frame == index) {
                std::getline(ifs, line);
                num_atoms = std::stoi(line);
            }
        } else if (line.find("ITEM: BOX BOUNDS") != std::string::npos) {
            if (current_frame == index) {
                std::getline(ifs, str); // for x
                std::istringstream iss_x(str);
                iss_x >> box_xlo >> box_xhi;

                std::getline(ifs, str); // for y
                std::istringstream iss_y(str);
                iss_y >> box_ylo >> box_yhi;

                std::getline(ifs, str); // for z
                std::istringstream iss_z(str);
                iss_z >> box_zlo >> box_zhi;
                // 代入
                unitcell_vec[0][0] = box_xhi-box_xlo;
                unitcell_vec[0][1] = 0;
                unitcell_vec[0][2] = 0;
                unitcell_vec[1][0] = 0;
                unitcell_vec[1][1] = box_yhi-box_ylo;
                unitcell_vec[1][2] = 0;
                unitcell_vec[2][0] = 0;
                unitcell_vec[2][1] = 0;
                unitcell_vec[2][2] = box_zhi-box_zlo;
            }
        } else if (line.find("ITEM: ATOMS") != std::string::npos) {
            if (current_frame == index) {
                std::istringstream header_iss(str.substr(11)); // skip "ITEM: ATOMS"
                std::string column;
                int index = 0;
                while (header_iss >> column) {
                    column_indices[column] = index++; // index from 0
                    // std::cout << column  << column_indices[column] << std::endl;
                }
                while (std::getline(ifs, str)) {
                    if (str.find("ITEM: ") != std::string::npos) { // if find next structure, stop
                        break;
                    }
                    std::istringstream iss(str);
                    // Atom atom;
                    std::vector<double> data(column_indices.size()); // for atomic species & coordinates
                    std::string test_string;
                    for (size_t i = 0; i < data.size(); ++i) {
                        iss >> test_string;
                        // std::cout << " strings  " << test_string << std::endl;
                        if (test_string == "C"){
                            data[i] = 6;  // Replace string with 6   
                        } else if (test_string == "O")
                        {
                            data[i] = 8;  // Replace string with 8
                        } else if (test_string == "H")
                        {
                            data[i] = 1;  // Replace string with 1
                        } else {
                            data[i] = std::stod(test_string);
                        } // TODO add error handling if test_string is not float
                    }

                    // if (column_indices.find("id") != column_indices.end()) {
                    //    atom.id = static_cast<int>(data[column_indices["id"]]);
                    // }
                    if (column_indices.find("element") != column_indices.end()) {
                        // std::cout << " element  " << data[column_indices["element"]] << std::endl;
                        atomic_num.push_back(data[column_indices["element"]]);
                    }
                    if (column_indices.find("xu") != column_indices.end()) {
                        x_temp = data[column_indices["xu"]];
                    }
                    if (column_indices.find("yu") != column_indices.end()) {
                        y_temp = data[column_indices["yu"]];
                    }
                    if (column_indices.find("zu") != column_indices.end()) {
                        z_temp = data[column_indices["zu"]];
                    }
                    tmp_position = Eigen::Vector3d(x_temp, y_temp, z_temp); // atomic position
                    // current_timestep.atoms.push_back(atom);
                    positions.push_back(tmp_position);
                }
                Atoms tmp_atoms = Atoms(atomic_num, positions, unitcell_vec, {true,true,true});
                return tmp_atoms;
            } else {
                // 次のフレームへ移動するため、ATOMセクションをスキップ
                for (int i = 0; i < num_atoms; ++i) {
                    std::getline(ifs, line);
                }
            }
        }
    }

    throw std::out_of_range("Frame index out of range");
}

Atoms read_frame(const std::string& filename, int index){
    /*
    大元のase_io_read関数のオーバーロード版．ファイル名を入力するだけで格子定数などを全て取得する．
    */
    if (filename.ends_with("xyz")){
        std::cout << "trajectory format is xyz" << std::endl;
        return read_xyz_frame(filename, index);
    } else if (filename.ends_with("lammpstrj")){
        std::cout << "trajectory format is lammpstrj" << std::endl;
        return read_lammps_frame(filename);
    } else{
        std::cout << "ERROR :: file should be end with xyz or lammpstrj" << std::endl;
    }
}


std::vector<std::vector<double> > raw_cpmd_get_unitcell_xyz(const std::string filename = "IONS+CENTERS.xyz") {
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

    // 以下で"16.267601013183594 0.0 0.0 0.0 16.267601013183594 0.0 0.0 0.0 16.267601013183594"を3*3行列へ格納
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


std::vector<std::vector<double> > raw_cpmd_get_unitcell_lammps(const std::string filename) {
    /*
    xyzファイルから単位格子ベクトルを取得する．

     Lattice="16.267601013183594 0.0 0.0 0.0 16.267601013183594 0.0 0.0 0.0 16.267601013183594" Properties=species:S:1:pos:R:3 pbc="T T T"
    という文字列から，Lattice=" "の部分を抽出しないといけない．
    */
    std::ifstream ifs(filename);
    // https://qiita.com/yohm/items/7c82693b83d4c055fa7b
    // std::cout << "print match :: " << match.str() << std::endl; // DEBUG（ここまではOK．）
	std::string str;
    int counter = 1; //! 行数カウンター
    std::vector<std::string> unitcell_vec_str;
    std::vector<std::vector<double> > unitcell_vec(3, std::vector<double> (3, 0)); // ここで3*3の形に指定しないとダメだった．
	double x_temp, y_temp, z_temp;
    bool flag_box;
    double box_xlo, box_xhi, box_ylo, box_yhi, box_zlo, box_zhi;
	while (std::getline(ifs,str)) {
        if (str.find("ITEM: BOX BOUNDS") != std::string::npos) { // get box
            std::getline(ifs, str); // for x
            std::istringstream iss_x(str);
            iss_x >> box_xlo >> box_xhi;

            std::getline(ifs, str); // for y
            std::istringstream iss_y(str);
            iss_y >> box_ylo >> box_yhi;

            std::getline(ifs, str); // for z
            std::istringstream iss_z(str);
            iss_z >> box_zlo >> box_zhi;
            // 代入
            unitcell_vec[0][0] = box_xhi-box_xlo;
            unitcell_vec[0][1] = 0;
            unitcell_vec[0][2] = 0;
            unitcell_vec[1][0] = 0;
            unitcell_vec[1][1] = box_yhi-box_ylo;
            unitcell_vec[1][2] = 0;
            unitcell_vec[2][0] = 0;
            unitcell_vec[2][1] = 0;
            unitcell_vec[2][2] = box_zhi-box_zlo;
            break;
        }
    }
    return unitcell_vec;
}


std::vector<Atoms> ase_io_read(const std::string& filename, const int NUM_ATOM, const std::vector<std::vector<double> > unitcell_vec){
    /*
    TODO :: positionsとatomic_numのpush_backは除去できる．（いずれもNUM_ATOM個）
    MDトラジェクトリを含むxyzファイルから
        - 格子定数
        - 原子番号
        - 原子座標
    を取得して，Atomsのリストにして返す．
    読み込み簡単化&高速化のため，予めNUM_ATOMを取得しておく．
    */
    //! test for Atomicnum
    Atomicnum atomicnum;

    std::ifstream ifs(std::filesystem::absolute(filename)); // ファイル読み込み
	if (ifs.fail()) {
	   std::cerr << "Cannot open xyz file\n";
	   exit(0);
	}
	std::string str;
	Eigen::Vector3d tmp_position; //! 原子座標

    std::string atom_id; //! 原子番号
    std::vector<int> atomic_num; //! 原子番号のリスト 
    std::vector<Eigen::Vector3d> positions; //! 原子座標のリスト
    std::vector<Atoms> atoms_list; //! Atomsのリスト
    int counter = 1; //! 行数カウンター
    int index_atom = 0; //! 読み込んでいる原子のインデックス
	double x_temp, y_temp, z_temp;
	while (getline(ifs,str)) {
	    std::stringstream ss(str);
	    index_atom = counter % (NUM_ATOM+2);
	    if (index_atom == 1 || index_atom == 2){ // 最初の2行は飛ばす．
	      counter += 1;
	      continue;   
	    }
	    // position/atomic_numの読み込み
	    ss >> atom_id >> x_temp >> y_temp >> z_temp;
	    tmp_position = Eigen::Vector3d(x_temp, y_temp, z_temp);
	    positions.push_back(tmp_position);
	    atomic_num.push_back(atomicnum.atomicnum.at(atom_id)); // 原子種から原子番号へ変換 // https://qiita.com/_EnumHack/items/f462042ec99a31881a81
        
	    if (index_atom == 0){ //最後の原子を読み込んだら，Atomsを作成
	      Atoms tmp_atoms = Atoms(atomic_num, positions, unitcell_vec, {true,true,true});
	      atoms_list.push_back(tmp_atoms);
	      atomic_num.clear(); // vectorのクリア
	      positions.clear();
	    }
	    counter += 1;
	}	    		
	return atoms_list;
}

std::vector<Atoms> ase_io_read_lammps(const std::string& filename){
    /*
    TODO :: positionsとatomic_numのpush_backは除去できる．（いずれもNUM_ATOM個）
    MDトラジェクトリを含むxyzファイルから
        - 格子定数
        - 原子番号
        - 原子座標
    を取得して，Atomsのリストにして返す．
    読み込み簡単化&高速化のため，予めNUM_ATOMを取得しておく．
    */
    //! test for Atomicnum
    Atomicnum atomicnum;

    std::ifstream ifs(std::filesystem::absolute(filename)); // ファイル読み込み
	if (ifs.fail()) {
	   std::cerr << "Cannot open xyz file\n";
	   exit(0);
	}
	std::string str;
	Eigen::Vector3d tmp_position; //! 原子座標
    std::unordered_map<std::string, int> column_indices; // 原子構造の読み込み
    std::string atom_id; //! 原子番号
    std::vector<int> atomic_num; //! 原子番号のリスト 
    std::vector<Eigen::Vector3d> positions; //! 原子座標のリスト
    std::vector<Atoms> atoms_list; //! Atomsのリスト
    int counter = 1; //! 行数カウンター
    int index_atom = 0; //! 読み込んでいる原子のインデックス
    std::vector<std::string> unitcell_vec_str;
    std::vector<std::vector<double> > unitcell_vec(3, std::vector<double> (3, 0)); // ここで3*3の形に指定しないとダメだった．
	double x_temp, y_temp, z_temp;
    bool flag_box;
    double box_xlo, box_xhi, box_ylo, box_yhi, box_zlo, box_zhi;
	while (std::getline(ifs,str)) {
        if (str.find("ITEM: TIMESTEP") != std::string::npos) {  // get timestep
            std::cout << str << std::endl;
        }
        if (str.find("ITEM: BOX BOUNDS") != std::string::npos) { // get box
            std::getline(ifs, str); // for x
            std::istringstream iss_x(str);
            iss_x >> box_xlo >> box_xhi;

            std::getline(ifs, str); // for y
            std::istringstream iss_y(str);
            iss_y >> box_ylo >> box_yhi;

            std::getline(ifs, str); // for z
            std::istringstream iss_z(str);
            iss_z >> box_zlo >> box_zhi;
            // 代入
            unitcell_vec[0][0] = box_xhi-box_xlo;
            unitcell_vec[0][1] = 0;
            unitcell_vec[0][2] = 0;
            unitcell_vec[1][0] = 0;
            unitcell_vec[1][1] = box_yhi-box_ylo;
            unitcell_vec[1][2] = 0;
            unitcell_vec[2][0] = 0;
            unitcell_vec[2][1] = 0;
            unitcell_vec[2][2] = box_zhi-box_zlo;
        }
        if (str.find("ITEM: ATOMS") != std::string::npos){ // get atom
            std::istringstream header_iss(str.substr(11)); // skip "ITEM: ATOMS"
            std::string column;
            int index = 0;
            while (header_iss >> column) {
                column_indices[column] = index++; // index from 0
                // std::cout << column  << column_indices[column] << std::endl;
            }
            while (std::getline(ifs, str)) {
                if (str.find("ITEM: ") != std::string::npos) { // if find next structure, stop
                    break;
                }
                std::istringstream iss(str);
                // Atom atom;
                std::vector<double> data(column_indices.size()); // for atomic species & coordinates
                std::string test_string;
                for (size_t i = 0; i < data.size(); ++i) {
                    iss >> test_string;
                    // std::cout << " strings  " << test_string << std::endl;
                    if (test_string == "C"){
                        data[i] = 6;  // Replace string with 6   
                    } else if (test_string == "O")
                    {
                        data[i] = 8;  // Replace string with 8
                    } else if (test_string == "H")
                    {
                        data[i] = 1;  // Replace string with 1
                    } else {
                        data[i] = std::stod(test_string);
                    } // TODO add error handling if test_string is not float
                }

                // if (column_indices.find("id") != column_indices.end()) {
                //    atom.id = static_cast<int>(data[column_indices["id"]]);
                // }
                if (column_indices.find("element") != column_indices.end()) {
                    // std::cout << " element  " << data[column_indices["element"]] << std::endl;
                    atomic_num.push_back(data[column_indices["element"]]);
                }
                if (column_indices.find("xu") != column_indices.end()) {
                    x_temp = data[column_indices["xu"]];
                }
                if (column_indices.find("yu") != column_indices.end()) {
                    y_temp = data[column_indices["yu"]];
                }
                if (column_indices.find("zu") != column_indices.end()) {
                    z_temp = data[column_indices["zu"]];
                }
                tmp_position = Eigen::Vector3d(x_temp, y_temp, z_temp); // atomic position
                // current_timestep.atoms.push_back(atom);
                positions.push_back(tmp_position);
            }
            Atoms tmp_atoms = Atoms(atomic_num, positions, unitcell_vec, {true,true,true});
            atoms_list.push_back(tmp_atoms);
            atomic_num.clear(); // vectorのクリア
    	    positions.clear();
	    }
    }	
	return atoms_list;
}


std::vector<Atoms> ase_io_read(const std::string& filename){
    /*
    大元のase_io_read関数のオーバーロード版．ファイル名を入力するだけで格子定数などを全て取得する．
    */
    if (filename.ends_with("xyz")){
        std::cout << "trajectory format is xyz" << std::endl;
        return ase_io_read(filename, raw_cpmd_num_atom(filename), raw_cpmd_get_unitcell_xyz(filename));
    } else if (filename.ends_with("lammpstrj")){
        std::cout << "trajectory format is lammpstrj" << std::endl;
        return ase_io_read_lammps(filename);
    } else{
        std::cout << "ERROR :: file should be end with xyz or lammpstrj" << std::endl;
    }
}

std::vector<Atoms> ase_io_read(const std::string& filename, const int NUM_ATOM, const std::vector<std::vector<double> > unitcell_vec, bool IF_REMOVE_WANNIER){
    /*
    大元のase_io_read関数のオーバーロード版2．
    ワニエセンターが含まれる場合の関数．IF_REMOVE_WANNIER=trueなら，原子がXの場合に削除する
    */
    if (!IF_REMOVE_WANNIER){ 
        return ase_io_read(filename,NUM_ATOM, unitcell_vec);
    }

    //! test for Atomicnum
    Atomicnum atomicnum;

    std::ifstream ifs(std::filesystem::absolute(filename)); // ファイル読み込み
	if (ifs.fail()) {
	   std::cerr << "Cannot open xyz file\n";
	   exit(0);
	}
	std::string str;
	Eigen::Vector3d tmp_position; //! 原子座標

    std::string atom_id; //! 原子番号
    std::vector<int> atomic_num; //! 原子番号のリスト 
    std::vector<Eigen::Vector3d> positions; //! 原子座標のリスト
    std::vector<Atoms> atoms_list; //! Atomsのリスト
    int counter = 1; //! 行数カウンター
    int index_atom = 0; //! 読み込んでいる原子のインデックス
	double x_temp, y_temp, z_temp;
	while (getline(ifs,str)) {
	    std::stringstream ss(str);
        index_atom = counter % (NUM_ATOM+2);
        if (index_atom == 1 || index_atom == 2){ // 最初の2行は飛ばす．
            counter += 1;
            continue;   
        }
        ss >> atom_id >> x_temp >> y_temp >> z_temp; // 読み込み
        if (atom_id != "X"){ // ワニエセンターの場合以外は読み込む
            tmp_position = Eigen::Vector3d(x_temp, y_temp, z_temp);
            positions.push_back(tmp_position);
            atomic_num.push_back(atomicnum.atomicnum.at(atom_id)); // 原子種から原子番号へ変換 // https://qiita.com/_EnumHack/items/f462042ec99a31881a81
        }
        if (index_atom == 0){ //最後の原子を読み込んだら，Atomsを作成
            Atoms tmp_atoms = Atoms(atomic_num, positions, unitcell_vec, {true,true,true});
            atoms_list.push_back(tmp_atoms);
            atomic_num.clear(); // vectorのクリア
            positions.clear();
        }
        counter += 1;
	}	    		
    return atoms_list;
}

std::vector<Atoms> ase_io_read(const std::string& filename,  bool IF_REMOVE_WANNIER){
    /*
    ase_io_readのワニエ版．
    */
    if (filename.ends_with(".xyz")){
        return ase_io_read(filename, raw_cpmd_num_atom(filename), raw_cpmd_get_unitcell_xyz(filename), IF_REMOVE_WANNIER);
    } else if (filename.ends_with(".lammpstrj")){
        // TODO :: implement IF_REMOVE_WANNIER for lammps
        return ase_io_read_lammps(filename);
    }
}



int ase_io_write(const std::vector<Atoms> &atoms_list, std::string filename ){
    /*
    ase.io.writeのc++版，全く同じ引数を取るので使いやすい．
    */
    std::ofstream fout(filename); 
    // まず2行目の変な文字列を取得
    std::string two_line="Properties=species:S:1:pos:R:3 pbc=\"T T T\"";
    // Lattice="15.389699935913086 0.0 0.0 0.0 15.389699935913086 0.0 0.0 0.0 15.389699935913086" Properties=species:S:1:pos:R:3 pbc="T T T"

    // 原子番号から
    Atomicchar atomicchar;
    // 
    // 2行目以降の部分をファイルへ出力。
    for (int i = 0, N=atoms_list.size(); i < N; i++) {
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
        for (int j = 0, N2=atomic_num.size(); j < N2; j++){
            fout << std::left << std::setw(2) << atomicchar.atomicchar[atomic_num[j]];
            fout << std::right << std::setw(16) << coords[j][0] << std::setw(16) << coords[j][1] << std::setw(16) << coords[j][2] << std::endl;
        }
    }
    return 0;
};


int ase_io_write(const Atoms &aseatoms, std::string filename ){
    /*
    ase_io_writeの別バージョン（オーバーロード）
    入力がaseatomsひとつだけだった場合にも動くようにする．
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
    for (int j = 0, N=atomic_num.size(); j < N; j++){
        fout << std::left << std::setw(2) << atomicchar.atomicchar.at(atomic_num[j]); // https://qiita.com/_EnumHack/items/f462042ec99a31881a81
        fout << std::right << std::setw(16) << coords[j][0] << std::setw(16) << coords[j][1] << std::setw(16) << coords[j][2] << std::endl;
    }
    return 0;
};

/**
 * 以下，gas計算に関連する関数
 * 
*/

std::vector<Atoms> ase_io_convert_1mol(const std::vector<Atoms> aseatoms, const int NUM_ATOM_PER_MOL){

    // int NUM_ATOM_PER_MOL = 6;

    // 分子数
    int NUM_MOL = aseatoms[0].get_atomic_numbers().size() / NUM_ATOM_PER_MOL;
    std::cout << NUM_MOL << std::endl;

    std::vector<Atoms> list_iso_atoms;
    std::vector<Eigen::Vector3d> tmp_positions;
    std::vector<int> tmp_symbols;
    std::vector<std::vector<double> > tmp_cell;
    for (int i = 0; i < aseatoms.size(); i++) { //Loop over i番目のconfiguration
        tmp_positions = aseatoms[i].get_positions();
        tmp_symbols   = aseatoms[i].get_atomic_numbers();
        tmp_cell      = aseatoms[i].get_cell();
        for (int j = 0; j < NUM_MOL; j++) { // Loop over molecules
            Atoms tmp_atoms(
                std::vector<int>(tmp_symbols.begin() + j * NUM_ATOM_PER_MOL, tmp_symbols.begin() + (j + 1) * NUM_ATOM_PER_MOL),
                std::vector<Eigen::Vector3d>(tmp_positions.begin() + j * NUM_ATOM_PER_MOL, tmp_positions.begin() + (j + 1) * NUM_ATOM_PER_MOL),
                tmp_cell,
                {1, 1, 1}
            );
            list_iso_atoms.push_back(tmp_atoms);
        }
    }
    std::cout << "FINSH" << std::endl;
    std::cout << "len(list_iso_atoms) :: " << list_iso_atoms.size() << std::endl;
    ase_io_write(list_iso_atoms,"1mol.xyz");
    return list_iso_atoms;
};
