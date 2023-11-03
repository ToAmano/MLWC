/*
2023/7/28
vectorをprintする関数たち．
*/ 

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
#include <tuple> 
#include<stdio.h>
#include<iomanip>
#include<vector>
#include<map>
#include<string>
#include<iostream>
#include"manupilate_files.hpp"

// #define PRINT_VAR(var) std::cout<<#var<< " :: ";

int get_num_lines(const std::string filename){
    /**
     * @fn ファイルの行数を返す関数だけど，空白行があった場合にどうなるのかがよくわからない．
     * TODO :: 空白行の処理をどうするべきかを考える．特に，文中に空白があるなら放置でよく，文末に空白があるなら無視して欲しい．
     * 
    */
    std::ifstream ifs(std::filesystem::absolute(filename)); // ファイル読み込み
    if (ifs.fail()) {
        std::cerr << " get_num_lines :: Cannot open xyz file\n";
        exit(0);
    }
    int counter = 0;
    std::string str;
	while (getline(ifs,str)) { 
        counter += 1;
	}	 
    return counter;
};
