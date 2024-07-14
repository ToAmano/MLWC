/*
2023/7/28
汎用性のあるコードの入れ場所
*/ 

#ifndef INCLUDE_HPP
#define INCLUDE_HPP

#include<stdio.h>
#include<vector>
#include<map>
#include<string>
#include<iostream>

// 2d vectorの単純なprint関数
void print_vec(const std::vector<std::vector<int> > &vector2d);


// 2d vectorのプリント関数たち
void print_vec(const std::vector<std::vector<int> > &vector2d, const std::string variable_name);


// 1d vectorのプリント関数たち
void print_vec(const std::vector<int> &vector1d, const std::string variable_name);

void print_vec(const std::vector<std::string> &vector1d, const std::string variable_name);


void write_2dvector_csv();


#endif //! INCLUDE_HPP
