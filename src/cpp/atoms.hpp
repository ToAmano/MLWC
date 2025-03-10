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
#include <regex>   // using cmatch = std::match_results<const char*>;
#include <map>     // https://bi.biopapyrus.jp/cpp/syntax/map.html
#include <cmath>
#include <algorithm>
#include <numeric> // std::iota
#include <tuple>   // https://tyfkda.github.io/blog/2021/06/26/cpp-multi-value.html
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <Eigen/Core> // 行列演算など基本的な機能．
#include "numpy.hpp"
#include "npy.hpp"
// #include "numpy_quiita.hpp" // https://qiita.com/ka_na_ta_n/items/608c7df3128abbf39c89
// numpy_quiitaはsscanf_sが読み込めず，残念ながら現状使えない．

// using namespace std;

class Vector
{
    std::vector<int> myVec;

public:
    Vector(std::vector<int> newVector)
    {
        myVec = newVector;
    }

    void print()
    {
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

class Atomicnum
{
    /*
    原子種と原子番号の対応を定義するクラス
    */
public:
    std::map<std::string, int> atomicnum;
    Atomicnum() {};
};

class Atomicchar
{
    /*
    原子種と原子番号の対応を定義するクラス2
    */
public:
    std::map<int, std::string> atomicchar;
    Atomicchar() {};
};

class Atoms
{
public: // public変数
    std::vector<int> atomic_num;
    std::vector<Eigen::Vector3d> positions; // !! ここEigenを利用．
    std::vector<std::vector<double>> cell;
    std::vector<bool> pbc;
    // std::vector<int> get_atomic_numbers(); // atomic_numを返す

    int number;
    Atoms(std::vector<int> atomic_numbers,
          std::vector<Eigen::Vector3d> atomic_positions,
          std::vector<std::vector<double>> UNITCELL_VECTORS,
          std::vector<bool> pbc_cell) {};

    std::vector<int> get_atomic_numbers() // atomic_numを返す
    {};
    std::vector<Eigen::Vector3d> get_positions() // positionsを返す
    {};
    std::vector<std::vector<double>> get_cell() // cellを返す
    {
    }
};

double sign(double A)
{
    // 実数Aの符号を返す
    // https://cvtech.cc/sign/
    return (A > 0) - (A < 0);
}

std::vector<Eigen::Vector3d> raw_get_distances_mic(Atoms aseatom, int a, std::vector<int> indices, bool mic = true, bool vector = false) {
    /*
    ase.atomのget_distances関数(micあり)のc++実装版
    a: 求める原子のaseatomでの順番（index）
    */
};

int raw_cpmd_num_atom(std::string filename) {
    /*
    CPMDのmd.outから原子数を取得する．
    */
};

std::vector<std::vector<double>> raw_cpmd_get_unitcell_xyz(std::string filename = "IONS+CENTERS.xyz") {
    /*
     Lattice="16.267601013183594 0.0 0.0 0.0 16.267601013183594 0.0 0.0 0.0 16.267601013183594" Properties=species:S:1:pos:R:3 pbc="T T T"
    という文字列から，Lattice=" "の部分を抽出しないといけない．
    */
};

std::vector<Atoms> ase_io_read(std::string filename, int NUM_ATOM, std::vector<std::vector<double>> unitcell_vec) {
    /*
    MDトラジェクトリを含むxyzファイルから
        - 格子定数
        - 原子番号
        - 原子座標
    を取得して，Atomsのリストにして返す．
    読み込み簡単化&高速化のため，予めNUM_ATOMを取得しておく．
    */
};

class read_mol
{
    /*
    pythonの同名クラスと違い，別途ファイルからボンド情報を読み込む．
    読み込んだボンドをクラス変数に格納する．
     - num_atoms_per_mol : 原子数
     - atom_list : 原子番号のリスト

    */
public: // とりあえずPGの例を実装
    // 原子数の取得
    int num_atoms_per_mol = 13;
    // atom list（原子番号）
    std::vector<string> atom_list{"O", "C", "C", "O", "C", "H", "H", "H", "H", "H", "H", "H", "H"};
    // bonds_listの作成
    std::vector<std::vector<int>> bonds_list{{0, 1}, {0, 5}, {1, 2}, {1, 6}, {1, 7}, {2, 3}, {2, 4}, {2, 8}, {3, 9}, {4, 10}, {4, 11}, {4, 12}};
    // double_bondの作成
    std::vector<std::vector<int>> double_bonds; //(PGの場合は)空リスト
    // 各種ボンド
    std::vector<std::vector<int>> ch_bond, co_bond, oh_bond, oo_bond, cc_bond, ring_bond;
    // ローンペア
    std::vector<int> o_list, n_list;
    // ボンドリスト
    std::vector<int> ch_bond_index, oh_bond_index, co_bond_index, oo_bond_index, cc_bond_index, ring_bond_index;
    // print(" -----  ml.read_mol :: parse results... -------")
    // print(" bonds_list :: ", self.bonds_list)
    // print(" counter    :: ", self.num_atoms_per_mol)
    // # print(" atomic_type:: ", self.atomic_type)
    // print(" atom_list  :: ", self.atom_list)
    // print(" -----------------------------------------------")

    // 代表原子の取得
    int representative_atom_index = 0;
    read_mol() {} // コンストラクタ
    void _get_bonds() {};
    void _get_atomic_index() {};
    std::vector<int> raw_convert_bondpair_to_bondindex(std::vector<std::vector<int>> bonds, std::vector<std::vector<int>> bonds_list) {
        /*
        ボンド[a,b]から，ボンド番号（bonds.index）への変換を行う．ボンド番号はbonds_list中のインデックス．
        bondsにch_bondsなどの一覧を入力し，それを番号のリストに変換する．

        ある要素がvectorに含まれているかどうかの判定はstd::findで可能．
        要素のindexはstd::distanceで取得可能．
        */
    };
};

Atoms raw_make_atoms_with_bc(Eigen::Vector3d bond_center, Atoms atoms, std::vector<std::vector<double>> UNITCELL_VECTORS)
{
    /*
    ######INPUTS#######
    bond_center     # Eigen::Vector3d 記述子を求めたい結合中心の座標
    list_mol_coords # array  分子ごとの原子座標
    list_atomic_nums #array  分子ごとの原子座標
    */
    std::vector<Eigen::Vector3d> list_mol_coords = atoms.get_positions();
    std::vector<int> list_atomic_num = atoms.get_atomic_numbers();

    // 結合中点bond_centerを先頭においたAtomsオブジェクトを作成する
    list_mol_coords.insert(list_mol_coords.begin(), bond_center);
    list_atomic_num.insert(list_atomic_num.begin(), 79); // 結合中心のラベルはAu(79)とする
    Atoms WBC = Atoms(list_atomic_num,
                      list_mol_coords,
                      UNITCELL_VECTORS,
                      {1, 1, 1});
    return WBC;
};

std::vector<Eigen::Vector3d> find_specific_bondcenter(std::vector<std::vector<Eigen::Vector3d>> list_bond_centers, std::vector<int> bond_index)
{
    /*
    list_bond_centersからbond_index情報をもとに特定のボンド（CHなど）だけ取り出す．
    TODO :: ここは将来的にはボンドをあらかじめ分割するようにして無くしてしまいたい．
    * @param[in] bond_index : 分子内のボンドセンターのindex（read_mol.bond_index['CH_1_bond']など．）
    * @return cent_mol : 特定のボンドの原子座標（(-1,3)型）
    */
    std::vector<Eigen::Vector3d> cent_mol;
    // ボンドセンターの座標と双極子をappendする．
    for (int i = 0; i < list_bond_centers.size(); i++)
    { // UnitCellの分子ごとに分割
        // chボンド部分（chボンドの重心をappend）
        for (int j = 0; j < bond_index.size(); j++)
        { // 分子内のボンドセンター数に関するLoop
            std::cout << " print list_bond_centers[i][bond_index[j]] :: " << list_bond_centers[i][bond_index[j]][0] << std::endl;
            cent_mol.push_back(list_bond_centers[i][bond_index[j]]);
        }
    }
    return cent_mol;
};

double fs(double Rij, double Rcs, double Rc)
{
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
    if (Rij < Rcs)
    {
        s = 1 / Rij;
    }
    else if (Rij < Rc)
    {
        s = (1 / Rij) * (0.5 * cos(M_PI * (Rij - Rcs) / (Rc - Rcs)) + 0.5);
    }
    else
    {
        s = 0;
    }
    return s;
}

std::vector<double> calc_descripter(std::vector<Eigen::Vector3d> dist_wVec, std::vector<int> atoms_index, double Rcs, double Rc, int MaxAt)
{
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
    for (int l = 1; l < atoms_index.size(); l++)
    { // 0は自分自身なので除外して1スタートとする．
        drs.push_back(dist_wVec[atoms_index[l]]);
    }
    // drsの絶対値を求める．
    std::vector<double> drs_abs;
    for (int j = 0; j < drs.size(); j++)
    {
        drs_abs.push_back(drs[j].norm());
    }
    // カットオフ関数fsをかけた1/drsを計算する．
    std::vector<double> drs_inv;
    for (int j = 0; j < drs.size(); j++)
    {
        drs_inv.push_back(fs(drs_abs[j], Rcs, Rc));
    }
    // 記述子（f(1,x/r,y/r,z/r)）を計算する．4次元ベクトルの配列
    std::vector<std::vector<double>> dij;
    for (int j = 0; j < drs.size(); j++)
    {
        dij.push_back({drs_inv[j], drs_inv[j] * drs[j][0] / drs_abs[j], drs_inv[j] * drs[j][1] / drs_abs[j], drs_inv[j] * drs[j][2] / drs_abs[j]});
    }
    // dijを要素drs_invの大きい順（つまりdrsが小さい順）にソートする．要素としては4次元ベクトルの第0成分でのソート
    // https://qiita.com/Arusu_Dev/items/c36cdbc41fc77531205c
    std::sort(dij.begin(), dij.end(), [](const vector<double> &alpha, const vector<double> &beta)
              { return alpha[0] < beta[0]; });
    // 原子数がMaxAtよりも少なかったら０埋めして固定長にする。1原子あたり4要素(1,x/r,y/r,z/r)なので4*MaxAtの要素だけ保守する．
    std::vector<double> dij_desc;
    if (dij.size() < MaxAt)
    {
        for (int i = 0; i < dij.size(); i++)
        {
            dij_desc.push_back(dij[i][0]);
            dij_desc.push_back(dij[i][1]);
            dij_desc.push_back(dij[i][2]);
            dij_desc.push_back(dij[i][3]);
        }
        for (int i = 0; i < MaxAt - dij.size(); i++)
        {
            dij_desc.push_back(0); // 0埋め
            dij_desc.push_back(0);
            dij_desc.push_back(0);
            dij_desc.push_back(0);
        }
    }
    else
    {
        for (int i = 0; i < MaxAt; i++)
        {
            dij_desc.push_back(dij[i][0]);
            dij_desc.push_back(dij[i][1]);
            dij_desc.push_back(dij[i][2]);
            dij_desc.push_back(dij[i][3]);
        }
    }
    return dij_desc;
};

std::vector<double> raw_get_desc_bondcent(Atoms atoms, Eigen::Vector3d bond_center, int mol_id, std::vector<std::vector<double>> UNITCELL_VECTORS, int NUM_MOL_ATOMS)
{
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
    double Rcs = 4.0;  // [ang. unit] TODO : hard code
    double Rc = 6.0;   // [ang. unit] TODO : hard code
    double MaxAt = 12; // とりあえずは12個の原子で良いはず．
    // ボンドセンターを追加したatoms
    Atoms atoms_w_bc = raw_make_atoms_with_bc(bond_center, atoms, UNITCELL_VECTORS);
    // ボンドセンターが含まれている分子の原子インデックスを取得
    std::vector<int> atoms_in_molecule(NUM_MOL_ATOMS);
    for (int i = 0; i < NUM_MOL_ATOMS; i++)
    {
        atoms_in_molecule[i] = i + mol_id * NUM_MOL_ATOMS + 1; // 結合中心を先頭に入れたAtomsなので+1が必要．
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
    for (int i = 0; i < atomic_numbers.size(); i++)
    {
        bool if_bc_in_molecule = std::find(atoms_in_molecule.begin(), atoms_in_molecule.end(), i) != atoms_in_molecule.end();
        if (atomic_numbers[i] == 6 && if_bc_in_molecule)
        {
            Catoms_intra.push_back(i);
        }
        else if (atomic_numbers[i] == 6 && !(if_bc_in_molecule))
        {
            Catoms_inter.push_back(i);
        }
        else if (atomic_numbers[i] == 1 && if_bc_in_molecule)
        {
            Hatoms_intra.push_back(i);
        }
        else if (atomic_numbers[i] == 1 && !(if_bc_in_molecule))
        {
            Hatoms_inter.push_back(i);
        }
        else if (atomic_numbers[i] == 8 && if_bc_in_molecule)
        {
            Oatoms_intra.push_back(i);
        }
        else if (atomic_numbers[i] == 8 && !(if_bc_in_molecule))
        {
            Oatoms_inter.push_back(i);
        }
    }
    // 全ての原子との距離を求める．この際0-0間距離も含まれる．
    std::vector<int> range_at_list(atomic_numbers.size());    // (0~atomic_numbersまでの整数を格納．numpy.arangeと同等．)
    std::iota(range_at_list.begin(), range_at_list.end(), 0); // https://codezine.jp/article/detail/8778
    auto dist_wVec = raw_get_distances_mic(atoms_w_bc, 0, range_at_list, true, true);

    // for C atoms (intra)
    auto dij_C_intra = calc_descripter(dist_wVec, Catoms_intra, Rcs, Rc, MaxAt);
    // for H atoms (intra)
    auto dij_H_intra = calc_descripter(dist_wVec, Hatoms_intra, Rcs, Rc, MaxAt);
    // for O  atoms (intra)
    auto dij_O_intra = calc_descripter(dist_wVec, Oatoms_intra, Rcs, Rc, MaxAt);
    // for C atoms (inter)
    auto dij_C_inter = calc_descripter(dist_wVec, Catoms_inter, Rcs, Rc, MaxAt);
    // for H atoms (inter)
    auto dij_H_inter = calc_descripter(dist_wVec, Hatoms_inter, Rcs, Rc, MaxAt);
    // for O atoms (inter)
    auto dij_O_inter = calc_descripter(dist_wVec, Oatoms_inter, Rcs, Rc, MaxAt);
    // 連結する dij_C_intra+dij_H_intra+dij_O_intra+dij_C_inter+dij_H_inter+dij_O_inter
    dij_C_intra.insert(dij_C_intra.end(), dij_H_intra.begin(), dij_H_intra.end()); // 連結
    dij_C_intra.insert(dij_C_intra.end(), dij_O_intra.begin(), dij_O_intra.end()); // 連結
    dij_C_intra.insert(dij_C_intra.end(), dij_C_inter.begin(), dij_C_inter.end()); // 連結
    dij_C_intra.insert(dij_C_intra.end(), dij_H_inter.begin(), dij_H_inter.end()); // 連結
    dij_C_intra.insert(dij_C_intra.end(), dij_O_inter.begin(), dij_O_inter.end()); // 連結

    return dij_C_intra;
};

std::vector<double> raw_get_desc_bondcent_allinone(Atoms atoms, Eigen::Vector3d bond_center, int mol_id, std::vector<std::vector<double>> UNITCELL_VECTORS, int NUM_MOL_ATOMS)
{
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
    double Rcs = 4.0;  // [ang. unit] TODO : hard code
    double Rc = 6.0;   // [ang. unit] TODO : hard code
    double MaxAt = 24; // !! ここは分子内外を分ける場合の2倍の値としておくのが安心．12*2=24
    // ボンドセンターを追加したatoms
    Atoms atoms_w_bc = raw_make_atoms_with_bc(bond_center, atoms, UNITCELL_VECTORS);
    // ボンドセンターが含まれている分子の原子インデックスを取得
    std::vector<int> atoms_in_molecule(NUM_MOL_ATOMS);
    for (int i = 0; i < NUM_MOL_ATOMS; i++)
    {
        atoms_in_molecule[i] = i + mol_id * NUM_MOL_ATOMS + 1; // 結合中心を先頭に入れたAtomsなので+1が必要．
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
    for (int i = 0; i < atomic_numbers.size(); i++)
    {
        if (atomic_numbers[i] == 6)
        {
            Catoms_all.push_back(i);
        }
        else if (atomic_numbers[i] == 1)
        {
            Hatoms_all.push_back(i);
        }
        else if (atomic_numbers[i] == 8)
        {
            Oatoms_all.push_back(i);
        }
    }
    // 全ての原子との距離を求める．この際0-0間距離も含まれる．
    std::vector<int> range_at_list(atomic_numbers.size());    // (0~atomic_numbersまでの整数を格納．numpy.arangeと同等．)
    std::iota(range_at_list.begin(), range_at_list.end(), 0); // https://codezine.jp/article/detail/8778
    auto dist_wVec = raw_get_distances_mic(atoms_w_bc, 0, range_at_list, true, true);

    // for C atoms (all)
    auto dij_C_all = calc_descripter(dist_wVec, Catoms_all, Rcs, Rc, MaxAt);
    // for H atoms (all)
    auto dij_H_all = calc_descripter(dist_wVec, Hatoms_all, Rcs, Rc, MaxAt);
    // for O  atoms (all)
    auto dij_O_all = calc_descripter(dist_wVec, Oatoms_all, Rcs, Rc, MaxAt);

    // 連結する dij_C_all+dij_H_all+dij_O_all
    dij_C_all.insert(dij_C_all.end(), dij_H_all.begin(), dij_H_all.end()); // 連結
    dij_C_all.insert(dij_C_all.end(), dij_O_all.begin(), dij_O_all.end()); // 連結

    return dij_C_all;
};

std::vector<std::vector<double>> raw_calc_bond_descripter_at_frame(Atoms atoms_fr, std::vector<std::vector<Eigen::Vector3d>> list_bond_centers, std::vector<int> bond_index, int NUM_MOL, std::vector<std::vector<double>> UNITCELL_VECTORS, int NUM_MOL_ATOMS, string desctype)
{
    /*
    1つのframe中の全てのボンドの記述子を計算する
     ! note
    ---------------
        2023/7/21 : 最後の変数desctypeでどの形の記述子を使うかを指定する
    */
    std::vector<std::vector<double>> Descs;
    if (bond_index.size() != 0)
    {                                                                            // bond_indexが0でなければ計算を実行
        auto cent_mol = find_specific_bondcenter(list_bond_centers, bond_index); // 特定ボンドのBCの座標だけ取得
        if (desctype == "allinone")
        {
            for (int i = 0; i < cent_mol.size(); i++)
            {
                int mol_id = i % NUM_MOL; // len(bond_index) # 対応する分子ID（mol_id）を出すように書き直す．ボンドが1分子内に複数ある場合，その数で割らないといけない．（メタノールならCH結合が3つあるので3でわる）
                auto dij = raw_get_desc_bondcent_allinone(atoms_fr, cent_mol[i], mol_id, UNITCELL_VECTORS, NUM_MOL_ATOMS);
                Descs.push_back(dij);
            }
        }
        else if (desctype == "old")
        {
            for (int i = 0; i < cent_mol.size(); i++)
            {
                int mol_id = i % NUM_MOL; // len(bond_index) # 対応する分子ID（mol_id）を出すように書き直す．ボンドが1分子内に複数ある場合，その数で割らないといけない．（メタノールならCH結合が3つあるので3でわる）
                auto dij = raw_get_desc_bondcent(atoms_fr, cent_mol[i], mol_id, UNITCELL_VECTORS, NUM_MOL_ATOMS);
                Descs.push_back(dij);
            }
        }
    }
    return Descs;
};

void write_2dvector_csv()
{
    /*
    2dvectorをcsvファイルに書き込む
    */
}

int main()
{
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

    std::vector<int> test_num{1, 2, 3};
    std::vector<Eigen::Vector3d> test_positions{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    std::vector<std::vector<double>> UNITCELL_VECTORS{{16.267601013183594, 0, 0}, {0, 16.267601013183594, 0}, {0, 0, 16.267601013183594}};
    std::vector<bool> pbc{1, 1, 1};

    //! test for constructing Atoms
    Atoms atoms(test_num, test_positions, UNITCELL_VECTORS, pbc);
    std::cout << "this is atomic num " << atoms.atomic_num[0] << endl;
    std::cout << atoms.get_atomic_numbers()[0] << endl;

    //! test raw_get_distances_mic
    std::vector<Eigen::Vector3d> answer = raw_get_distances_mic(atoms, 0, {1, 2}, true, true);
    for (int i = 0; i < answer.size(); i++)
    {
        for (int j = 0; j < answer[i].size(); j++)
        {
            std::cout << answer[i][j] << " ";
        }
        std::cout << endl;
    }

    //! test for get_atomic_num
    std::cout << "test for get_atomic_num " << std::endl;
    int atomic_num = raw_cpmd_num_atom("gromacs_30.xyz");
    std::cout << atomic_num << std::endl;

    //! test for raw_cpmd_get_unitcell_xyz
    std::vector<std::vector<double>> unitcell_vec = raw_cpmd_get_unitcell_xyz("pg_gromacs.xyz");
    for (int i = 0; i < unitcell_vec.size(); i++)
    {
        for (int j = 0; j < unitcell_vec[i].size(); j++)
        {
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
    // print test_read_mol.bond_index['CH_1_bond']
    for (int i = 0; i < test_read_mol.bond_index['CH_1_bond'].size(); i++)
    {
        std::cout << test_read_mol.bond_index['CH_1_bond'][i] << " ";
    }
    // print test_read_mol.ch_bond
    for (int i = 0; i < test_read_mol.ch_bond.size(); i++)
    {
        std::cout << test_read_mol.ch_bond[i][0] << " " << test_read_mol.ch_bond[i][1] << std::endl;
    }

    //! test raw_aseatom_to_mol_coord_and_bc
    int NUM_MOL_ATOMS = test_read_mol.num_atoms_per_mol;
    int NUM_ATOM = atomic_num;
    int NUM_MOL = int(NUM_ATOM / NUM_MOL_ATOMS); // UnitCell中の総分子数
    auto test_mol_bc = raw_aseatom_to_mol_coord_and_bc(atoms_list[0], test_read_mol.bonds_list, test_read_mol, NUM_MOL_ATOMS, NUM_MOL);
    auto test_mol = std::get<0>(test_mol_bc);
    auto test_bc = std::get<1>(test_mol_bc);

    //! test make_ase_with_BCs
    Atoms new_atoms = make_ase_with_BCs(atoms_list[0].get_atomic_numbers(), NUM_MOL, raw_cpmd_get_unitcell_xyz("pg_gromacs.xyz"), test_mol, test_bc);

    //! test ase_io_write
    std::vector<Atoms> test = {new_atoms};
    ase_io_write(test, "test.xyz");

    //! test ase_io_write(作成したtest.xyzが読み込み可能かどうか)
    std::vector<Atoms> test_read = ase_io_read("test.xyz", raw_cpmd_num_atom("test.xyz"), raw_cpmd_get_unitcell_xyz("test.xyz"));

    //! test raw_calc_bond_descripter_at_frame (まずは適当なボンド記述子が作成できるかどうか)
    auto descs_ch = raw_calc_bond_descripter_at_frame(atoms_list[0], test_bc, test_read_mol.bond_index['CH_1_bond'], NUM_MOL, UNITCELL_VECTORS, NUM_MOL_ATOMS);

    //! test for save as npy file.
    // descs_chの形を1dへ変形してnpyで保存．
    // TODO :: さすがにもっと効率の良い方法があるはず．
    std::vector<double> descs_ch_1d;
    for (int i = 0; i < descs_ch.size(); i++)
    {
        for (int j = 0; j < descs_ch[i].size(); j++)
        {
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
    const std::vector<std::vector<double>> data2{{1, 11}, {2, 12}, {3, 13}, {4, 14}, {5, 15}, {6, 16}};
    aoba::SaveArrayAsNumpy("2dvector.npy", 6, 2, &data2[0][0]);

    const std::vector<long unsigned> shape{1, 6}; // vectorを2*3の形に保存
    std::cout << shape.size() << " " << shape.data() << std::endl;
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
    for (int i = 0; i < data2.size(); i++)
    {
        for (int j = 0; j < data2[i].size(); j++)
        {
            data2_1d.push_back(data2[i][j]);
        }
    }
    const std::vector<long unsigned> shape3{6, 2}; // vectorを1*12の形に保存
    npy::SaveArrayAsNumpy("2dvector_v2.npy", fortran_order, shape3.size(), shape3.data(), data2_1d);
}