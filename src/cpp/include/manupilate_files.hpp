/*
2023/10/30
ファイルi/oに関する関数たち
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

namespace manupilate_files{
int get_num_lines(const std::string filename);

bool IsFileExist(const std::string& name);

bool IsDirExist(const std::string& name);
}