#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cctype> // https://b.0218.jp/20150625194056.html
#include <algorithm> 
#include "parse.hpp"

// 任意のdilimiterで分割する関数
// https://maku77.github.io/cpp/string/split.html
std::vector<std::string> split(const std::string& src, const char* delim = ",") {
    /*
    string :: 入力文字列
    delim :: 区切り文字
    */
    std::vector<std::string> vec;
    std::string::size_type len = src.length();

    for (std::string::size_type i = 0, n; i < len; i = n + 1) {
        n = src.find_first_of(delim, i);
        if (n == std::string::npos) {
            n = len;
        }
        vec.push_back(src.substr(i, n - i));
    }

    return vec;
}

std::string remove_space(const std::string& str) {
    /*
    文字列中の空白を削除する．
    str.erase(std::remove_if(str.begin(), str.end(), std::isspace), str.end())
    でも動くらしい．https://b.0218.jp/20150625194056.html
    */
   // return str.erase(std::remove_if(str.begin(), str.end(), std::isspace), str.end());
   // strから空白を削除する．
//    std::string str(s);
//     return str.erase(std::remove(str.begin(), str.end(), ' '), str.end());
    std::string str2 = "";
    // std::cout << str << std::endl;
    for ( int i=0, N=str.size(); i<N; i++){
        // std::cout << str[i] << std::endl;
        if (str[i] != ' ' and str[i] != '\t'){
            str2 = str2 + str[i];
        }
    }
    std::cout << str2 << std::endl;
    return str2;
};

std::tuple<std::vector<std::vector<std::string > >, std::vector<std::vector<std::string > >, std::vector<std::vector<std::string > > > locate_tag(std::string inputfilename ) {
    /*
    ! python版では入力読み込みと分割を別関数にしていたが，
    ! c++だとstringstreamの都合上一つにまとめておいた方が楽．
    */
    std::vector<std::vector<std::string > > input_general;
    std::vector<std::vector<std::string > > input_descripter;
    std::vector<std::vector<std::string > > input_predict;
    
    /* 入力ファイルから情報を読み込み*/
    std::ifstream ifs(inputfilename); // ファイル読み込み
    if (ifs.fail()) {
        std::cerr << "Cannot open file\n";
        exit(0);
    }
    std::string str;
    int IfReadGeneral = 0; // generalの部分を読み込んだかどうかのフラグ
    int IfReadDesc = 0; // descの部分を読み込んだかどうかのフラグ
    int IfReadPred = 0; // 予測部分を読み込んだかのフラグ
    while (getline(ifs,str)) {
        std::stringstream ss(str);
        if (str == "&general") { // read general
            IfReadGeneral = 1;
            IfReadDesc = 0;
            IfReadPred = 0;
            continue;
        }
        if (str == "&descriptor") { // read desc
            IfReadGeneral = 0;
            IfReadDesc = 1;
            IfReadPred = 0;
            continue;
        }
        if (str == "&predict") {
            IfReadGeneral = 0;
            IfReadDesc = 0;
            IfReadPred = 1;
            continue;
        }
        if (IfReadGeneral){
            input_general.push_back(split(remove_space(str), "="));
        }
        if (IfReadDesc){
            input_descripter.push_back(split(remove_space(str), "="));
        }
        if (IfReadPred){
            input_predict.push_back(split(remove_space(str), "="));
        }
    }    
    // 読み込み後にちゃんと全部読み込めているか確認
    if (input_general.size() == 0){
        std::cerr << "WARNING: input_general is empty." << std::endl;
    }
    if (input_descripter.size() == 0){
        std::cerr << "WARNING: input_descripter is empty." << std::endl;
    }
    if (input_predict.size() == 0){
        std::cerr << "WARNING: input_predict is empty." << std::endl;
    }
    return std::make_tuple(input_general, input_descripter, input_predict);
}

// class var_general
var_general::var_general(){};

var_general::var_general(std::vector< std::vector<std::string> > input_general){
    for (int i=0, N=input_general.size(); i<N; i++){
                    if (input_general[i][0] == "itpfilename"){
                        itpfilename = input_general[i][1];
                    } else if (input_general[i][0] == "bondfilename"){
                        bondfilename = input_general[i][1];
                    } else if (input_general[i][0] == "savedir"){
                        savedir      = input_general[i][1];
                    } else if (input_general[i][0] == "temperature"){
                        temperature  = std::stod(input_general[i][1]);
                    } else if (input_general[i][0] == "timestep"){
                        timestep     = std::stod(input_general[i][1]);
                    } else {
                        std::cerr << " WARNING: invalid input_general :: " << input_general[i][0] << std::endl;
                        std::cerr << "We ignore this line." << std::endl;
                }   
    }
    std::cout << "Finish reading ver_general " << std::endl;
};


// class var_descripter
var_descripter::var_descripter(){};

var_descripter::var_descripter(std::vector< std::vector<std::string> > input_descripter){
    for (int i=0, N=input_descripter.size(); i<N; i++){
                std::cout << input_descripter[i][0] << " " << input_descripter[i][1] << std::endl;
                if (input_descripter[i][0] == "calc"){
                    calc = stoi(input_descripter[i][1]);
                } else if (input_descripter[i][0] == "directory" ){
                    directory = input_descripter[i][1];
                } else if (input_descripter[i][0] == "xyzfilename" ) {
                    xyzfilename = input_descripter[i][1];
                } else if (input_descripter[i][0] == "savedir"){
                    savedir = input_descripter[i][1];
                } else if (input_descripter[i][0] == "descmode"){
                    descmode = input_descripter[i][1];
                } else if ( input_descripter[i][0] == "desctype"){ // old or allinone
                    desctype = input_descripter[i][1];
                } else if (input_descripter[i][0] == "step") {
                    step = stoi(input_descripter[i][1]);
                } else if (input_descripter[i][0] == "haswannier"){
                    haswannier = stoi(input_descripter[i][1]);
                } else if (input_descripter[i][0] == "interval"){
                    interval = stoi(input_descripter[i][1]);
                } else if (input_descripter[i][0] == "IF_COC"){
                    IF_COC = stoi(input_descripter[i][1]);
                } else if (input_descripter[i][0] == "gas"){
                    IF_GAS = stoi(input_descripter[i][1]);
                } else {
                    std::cerr << "WARNING: invalid input_descripter : " << input_descripter[i][0] << std::endl;
                    std::cerr << "We ignore this line." << std::endl;
                }   
    }
    std::cout << "Finish reading ver_descriptor " << std::endl;
};


var_predict::var_predict(){};


var_predict::var_predict(std::vector< std::vector<std::string> > input_predict){
    // まずはデフォルト値を代入
    // bondspecies = 4;
    // save_truey = 0;
    // ついでファイルから値を代入
    for (int i=0, N=input_predict.size(); i<N; i++){
            std::cout << input_predict[i][0] << " " << input_predict[i][1] << std::endl;
            if (input_predict[i][0] == "calc"){
                this->calc = stoi(input_predict[i][1]);
            } else if (input_predict[i][0] == "model_dir" ) {
                this->model_dir = input_predict[i][1];
            } else if (input_predict[i][0] == "desc_dir"){
                this->desc_dir = input_predict[i][1];
            } else if (input_predict[i][0] == "modelmode"){
                this->modelmode = input_predict[i][1];
            } else if (input_predict[i][0] == "bondspecies") {
                this->bondspecies = stoi(input_predict[i][1]);
            } else if (input_predict[i][0] == "save_truey"){
                this->save_truey = stoi(input_predict[i][1]);
            } else {
                std::cerr << "WARNING: invalid input_predict : " << input_predict[i][0] << std::endl;
                std::cerr << "We ignore this line." << std::endl;
            };
    };
    std::cout << "Finish reading ver_predict " << std::endl;
    std::cout << " " << std::endl;
};

