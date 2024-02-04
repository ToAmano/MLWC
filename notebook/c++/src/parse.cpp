#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <string>
#include <cctype> // https://b.0218.jp/20150625194056.html
#include <algorithm> 
#include "parse.hpp"
#include "yaml-cpp/yaml.h" //https://github.com/jbeder/yaml-cpp
#include "include/error.h"

namespace parse{
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

// >>> START FOR YAML >>>>>
int get_len_yaml(YAML::Node node){
    std::cout << "yaml size" << node.size() << std::endl;
    return node.size();
};

std::string get_val_yaml(YAML::Node node, std::string key){
    if (node[key]) {
        // std::cout << node[key].as<std::string>() << std::endl;
        return node[key].as<std::string>();
    } else {
        std::cout << " ERROR :: No key !!" << std::endl;
        return "no";
    }
};

bool if_val_exist(YAML::Node node, std::string key){
    /**
     * @brief 与えられたnodeにkeyが存在するか?
     * 
     */
    if (node[key]) {
        return true;
    } else {
        return false;
    };
}

// https://qiita.com/i153/items/38f9688a9c80b2cb7da7
int parse_required_argment(YAML::Node node, std::string key, int &variable){
    bool IF_EXIST = if_val_exist(node, key);
    if (IF_EXIST == true){
        variable = stoi(get_val_yaml(node,key));
        return 0;
    } else{
        std::cout << "parse_requied_argment :: ERROR KEYWARD NOT EXIST" << std::endl;
        std::exit(1);
    }
};

int parse_required_argment(YAML::Node node, std::string key, std::string &variable){
    bool IF_EXIST = if_val_exist(node, key);
    if (IF_EXIST == true){
        variable = get_val_yaml(node,key);
        return 0;
    } else{
        std::cout << "parse_requied_argment :: ERROR KEYWARD NOT EXIST" << std::endl;
        std::exit(1);
    }
};




std::string parse_optional_argment(YAML::Node node, std::string key, std::string default_val){
    bool IF_EXIST = if_val_exist(node, key);
    if (IF_EXIST == true){
        return get_val_yaml(node,key);
    } else{
        return default_val;
    }
};


// 必須の場合，if_val_exist = falseならerrorを出して終了させる．

// >>> END FOR YAML >>>>>


// class var_general
var_general::var_general(){};

var_general::var_general(std::vector< std::vector<std::string> > input_general){
    for (int i=0, N=input_general.size(); i<N; i++){
                    if (input_general[i][0] == "itpfilename"){
                        this->itpfilename = input_general[i][1];
                    } else if (input_general[i][0] == "bondfilename"){
                        this->bondfilename = input_general[i][1];
                    } else if (input_general[i][0] == "savedir"){
                        this->savedir      = input_general[i][1];
                    } else if (input_general[i][0] == "temperature"){
                        this->temperature  = std::stod(input_general[i][1]);
                    } else if (input_general[i][0] == "timestep"){
                        this->timestep     = std::stod(input_general[i][1]);
                    } else {
                        std::cerr << " WARNING: invalid input_general :: " << input_general[i][0] << std::endl;
                        std::cerr << "We ignore this line." << std::endl;
                }   
    }
    std::cout << " Finish reading ver_general " << std::endl;
};

var_general::var_general(YAML::Node node){
    parse_required_argment(node, "itpfilename", this->itpfilename);
    parse_required_argment(node, "bondfilename", this->bondfilename);
    parse_required_argment(node, "savedir", this->savedir);
    this->temperature  = std::stod(parse_optional_argment(node, "temperature", "300"));
    this->timestep     = std::stod(parse_optional_argment(node, "timestep", "0.5"));
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

var_descripter::var_descripter(YAML::Node node){
    parse_required_argment(node, "calc", this->calc);
    parse_required_argment(node, "directory", this->directory);
    parse_required_argment(node, "savedir", this->savedir);
    parse_required_argment(node, "xyzfilename", this->xyzfilename);
//    this->descmode     = parse_required_argment(node, "descmode");
    this->desctype     = parse_optional_argment(node, "desctype", "allinone"); // old or allinone
//    this->step         = stoi(parse_required_argment(node, "step"));
//    this->haswannier   = stoi(parse_required_argment(node, "haswannier"));
//    this->interval     = stoi(parse_required_argment(node, "interval"));
    this->IF_COC       = stoi(parse_optional_argment(node, "IF_COC", "0")); // 0=False
    this->IF_GAS       = stoi(parse_optional_argment(node, "IF_GAS", "0")); // 0=False
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

var_predict::var_predict(YAML::Node node){
    parse_required_argment(node, "calc", this->calc);
    parse_required_argment(node, "model_dir",this->model_dir);
//    this->desc_dir      = parse_required_argment(node, "desc_dir");
//    this->modelmode     = parse_required_argment(node, "modelmode");
//    this->bondspecies   = stoi(parse_required_argment(node, "bondspecies"));
//    this->save_truey    = stoi(parse_required_argment(node, "save_truey"));
}

} // END namespace