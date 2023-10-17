/*
2023/7/28
vectorをprintする関数たち．
*/ 

#include<stdio.h>
#include<vector>
#include<map>
#include<string>
#include<iostream>
#include"printvec.hpp"

#define PRINT_VAR(var) std::cout<<#var<< " :: ";

void print_vec(const std::vector<std::vector<int> > &vector2d){
    /*
    2次元ベクトルのうち，特に第二成分が2つだけのベクトルのprint関数
    */
            PRINT_VAR(vector2d);
            for (int i = 0, size=vector2d.size(); i < size; i++) {
                std::cout << "["<< vector2d[i][0] << " " << vector2d[i][1] << "] ";
            }
            std::cout << std::endl;
}

void print_vec(const std::vector<std::vector<int> > &vector2d, const std::string variable_name){
    /*
    2次元ベクトルのうち，特に第二成分が2つだけのベクトルのprint関数
    PRINT_VARがうまく動かないのでこちらの利用を推奨
    */
    std::cout << variable_name << " :: ";
        for (int i = 0, size=vector2d.size(); i < size; i++) {
            std::cout << "["<< vector2d[i][0] << " " << vector2d[i][1] << "] ";
        }
        std::cout << std::endl;
}

void print_vec(const std::vector<int> &vector1d, const std::string variable_name){
    /*
    1次元ベクトルのprint関数(int版)
    */
    std::cout << variable_name << " :: ";
        for (int i = 0, size=vector1d.size(); i < size; i++) {
            std::cout << vector1d[i] << " " ;
        }
        std::cout << std::endl;
}

void print_vec(const std::vector<std::string> &vector1d, const std::string variable_name){
    /*
    1次元ベクトルのprint関数(string版)
    */
    std::cout << variable_name << " :: ";
        for (int i = 0, size=vector1d.size(); i < size; i++) {
            std::cout << vector1d[i] << " " ;
        }
        std::cout << std::endl;
}

void write_2dvector_csv(){
    /*
    2dvectorをcsvファイルに書き込む
    */
}
