/**
 * @file atoms_io.cppのテスト関数
 * @brief atoms_io.cppのテスト関数
 * @author Tomohito Amano
 * @date 2024/1/4
 */

#include <gtest/gtest.h>
#include "atoms_io.hpp"

// ファイルのi/oがあるような場合のテストをどのように記述するべきか？
// 一つの方法は，どうやらmock関数というのを使う方法のようだ．これはかなり一般的な手法である．
// 次の方法は，このtest_atoms_io.cppファイル内にxyzを作成する関数を作ってしまうこと！！ これが良いだろう．


int raw_cpmd_num_atom(const std::string filename);

int get_num_atom_without_wannier(const std::string filename);

std::vector<std::vector<double> > raw_cpmd_get_unitcell_xyz(const std::string filename ) ;

std::vector<Atoms> ase_io_read(const std::string filename, const int NUM_ATOM, const std::vector<std::vector<double> > unitcell_vec);

std::vector<Atoms> ase_io_read(std::string filename);

std::vector<Atoms> ase_io_read(const std::string filename, const int NUM_ATOM, const std::vector<std::vector<double> > unitcell_vec, bool IF_REMOVE_WANNIER);

std::vector<Atoms> ase_io_read(const std::string filename,  bool IF_REMOVE_WANNIER);

int ase_io_write(const std::vector<Atoms> &atoms_list, const std::string filename );

int ase_io_write(const Atoms &aseatoms, std::string filename );

std::vector<Atoms> ase_io_convert_1mol(const std::vector<Atoms> aseatoms, const int NUM_ATOM_PER_MOL);


TEST(IsEvenTest, Negative) {
    EXPECT_FALSE(IsEven(-1));
    EXPECT_TRUE(IsEven(-2));
}

TEST(IsEvenTest, Zero) {
    EXPECT_TRUE(IsEven(0));
}

TEST(IsEvenTest, Positive) {
    EXPECT_FALSE(IsEven(1));
    EXPECT_TRUE(IsEven(2));
}