/**
 * @file atoms_asign_wcs.cpp
 * @brief グラフによるatomsの実装を行う
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
#include <Eigen/Core> // 行列演算など基本的な機能．
#include "numpy.hpp"
#include "npy.hpp"
// #include "numpy_quiita.hpp" // https://qiita.com/ka_na_ta_n/items/608c7df3128abbf39c89
// numpy_quiitaはsscanf_sが読み込めず，残念ながら現状使えない．
// #include "atoms_core.hpp"
// #include "atoms_io.hpp"

class Node{
    /**
     *  @class Node
    @brief  グラフのノード（点）の定義．atomのindex,座標,原子種類, 分子index
    */
    public:
        int index; //! atomsにおけるindex
        int molindex; //! 分子のindex
        int atomicnum; //! 原子番号
        Eigen::Vector3d position; //! 原子座標(angstrom)
        Node(int index, int atomicnum, Eigen::Vector3d position); // コンストラクタ

        // member functions
        get_index_in_molecule();
        // std::vector<int> get_atomic_numbers() const; // atomic_numを返す
        // std::vector<Eigen::Vector3d> get_positions() const; // positionsを返す
        // std::vector<std::vector<double> > get_cell() const; // cellを返す
};

Node::Node(int index, int molindex, int atomicnum, Eigen::Vector3d position){
    this->index     = index;
    this->molindex  = molindex;
    this->atomicnum = atomicnum;
    this->position  = position;
}

Node::get_index_in_molecule(){
    //     indexとmolindexから，分子内におけるindexを取得するmethod
    return this->index%this->molindex;
}


class Edge{
    /**
     *  @class Edge
    @brief  グラフのエッジ（辺）の定義．両端のatomのindex(これはatomsの中での番号と対応)，ボンドの種類（1,2,3重,予測値), ボンド双極子, ボンドワニエ座標(これはmethodでも良い), BC（methodでもok）が必要
    ボンドセンターとボンドdipole，ボンドの種類があれば，bond wannier centerは計算可能
    ただし，真値についてはbond wannier centerベースの方が扱いやすいという点がある
    */
    public:
        pair<int, int> neighbour; // 両端のatomのindex
        ind bondtype; // ボンドの種類を指定
        Eigen::Vector3d dipole; // bond dipole
        Eigen::Vector3d bondcenter; // bond center

        Edge(pair<int, int> neighbour, ind bondtype, Eigen::Vector3d dipole, ); // コンストラクタ
};

class Graph{
    /**
     *  @class Graph
    @brief  atomsに含まれるデータをそのままマップするGraph
    */
    // TODO :: construct :: atomsからの変換
    // TODO :: method :: atomsへの変換
    //隣接リストを表すペアのvectorのvector（これは多分あった方が良い）
    std::vector<vector<Pair>> adjList;
    std::vector<Node> nodes;
    std::vector<Edge> edges;

    //グラフコンストラクタ
    Graph(){
        // 

    };
}



typedef pair<int, int> Pair;
 
//グラフオブジェクトを表すクラス
class Graph_ref
{
public:
    //隣接リストを表すペアのvectorのvector（これは多分あった方が良い）
    std::vector<vector<Pair>> adjList;
    std::vector<Node> nodes;
    std::vector<Edge> edges;

    //グラフコンストラクタ
    Graph(){
        // 

    };

    //グラフコンストラクタ
    Graph(vector<Edge> const &edges, int n) //n::ノードの数
    {
        //vectorのサイズを変更して、vector<Edge>型の`n`要素を保持します。
        adjList.resize(n);

        //有向グラフにエッジを追加します
        for (auto &edge: edges)
        {
            int src = edge.src;
            int dest = edge.dest;
            int weight = edge.weight;
 
            //最後に挿入
            adjList[src].push_back(make_pair(dest, weight));
 
            //無向グラフの次のコードのコメントを解除します
            // adjList[dest].push_back(make_pair(src, weight));
        }
    }
};
 
//グラフの隣接リスト表現を印刷する関数
void printGraph(Graph const &graph, int n)
{
    for (int i = 0; i < n; i++)
    {
        //指定された頂点のすべての隣接する頂点を出力する関数
        for (Pair v: graph.adjList[i]) {
            cout << "(" << i << ", " << v.first << ", " << v.second << ") ";
        }
        cout << endl;
    }
}


//STLを使用したグラフの実装
int main()
{
    //上の図のようなグラフエッジのvector。
    //以下の形式の初期化vectorは
    // C++ 11、C++ 14、C++ 17では正常に動作しますが、C++98では失敗します。
    vector<Edge> edges =
    {
        //(x、y、w)—>重み`w`を持つ`x`から`y`へのエッジ
        {0, 1, 6}, {1, 2, 7}, {2, 0, 5}, {2, 1, 4}, {3, 2, 10}, {5, 4, 1}, {4, 5, 3}
    };
 
    //グラフ内のノードの総数(0から5までのラベルが付いています)
    int n = 6;
 
    //グラフを作成します
    Graph graph(edges, n);
 
    //グラフの隣接リスト表現を出力します
    printGraph(graph, n);
 
    return 0;
}


int raw_cpmd_num_atom(const std::string filename){
    /*
    xyzファイルから原子数を取得する．（ワニエセンターが入っている場合その原子数も入ってしまうので注意．）
    基本的には1行目の数字を取得しているだけ．
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
    // 先にxyzファイルから原子数を取得する．
    int NUM_ATOM = raw_cpmd_num_atom(std::filesystem::absolute(filename));

    std::ifstream ifs(std::filesystem::absolute(filename)); // ファイル読み込み
    if (ifs.fail()) {
       std::cerr << " get_num_atom_without_wannier :: Cannot open xyz file\n";
       exit(0);
    }
    int NUM_ATOM_WITHOUT_WAN=0; //原子数
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
    return NUM_ATOM_WITHOUT_WAN;
};

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


std::vector<Atoms> ase_io_read(const std::string filename, const int NUM_ATOM, const std::vector<std::vector<double> > unitcell_vec){
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

std::vector<Atoms> ase_io_read(std::string filename){
    /*
    大元のase_io_read関数のオーバーロード版．ファイル名を入力するだけで格子定数などを全て取得する．
    */
    return ase_io_read(filename, raw_cpmd_num_atom(filename), raw_cpmd_get_unitcell_xyz(filename));
}

std::vector<Atoms> ase_io_read(const std::string filename, const int NUM_ATOM, const std::vector<std::vector<double> > unitcell_vec, bool IF_REMOVE_WANNIER){
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

std::vector<Atoms> ase_io_read(const std::string filename,  bool IF_REMOVE_WANNIER){
    /*
    ase_io_readのワニエ版．
    */
    if (filename.ends_with(".xyz")){
        return ase_io_read(filename, raw_cpmd_num_atom(filename), raw_cpmd_get_unitcell_xyz(filename), IF_REMOVE_WANNIER);
    } else if (filename.ends_with(".lammpstrj")){
        return ase_io_read(filename, raw_cpmd_num_atom(filename), raw_cpmd_get_unitcell_xyz(filename), IF_REMOVE_WANNIER);
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

