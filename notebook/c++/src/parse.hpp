#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <string>
#include <cctype> // https://b.0218.jp/20150625194056.html
#include <algorithm> 
#include "yaml-cpp/yaml.h" //https://github.com/jbeder/yaml-cpp
#include "include/error.h"

namespace parse{
// 任意のdilimiterで分割する関数
// https://maku77.github.io/cpp/string/split.html
//!! default variable should be only in header file
std::vector<std::string> split(const std::string& src, const char* delim);

std::string remove_space(const std::string& str);

std::tuple<std::vector<std::vector<std::string > >, std::vector<std::vector<std::string > >, std::vector<std::vector<std::string > > > locate_tag(std::string inputfilename );

// >>>> FOR YAML >>>>>>
int get_len_yaml(YAML::Node node); //YAML::Node node   = YAML::LoadFile("config.yaml");

std::string get_val_yaml(YAML::Node node, std::string key); // keyに対応するvalを取得(string)

int parse_required_argment(YAML::Node node, std::string key,int &variable);
int parse_required_argment(YAML::Node node, std::string key,std::string &variable);


std::string parse_optional_argment(YAML::Node node, std::string key, std::string default_val);

bool if_val_exist(YAML::Node node, std::string key);
// >>>> END FOR YAML >>>>>>



class var_general{
    /*
    general用の変数を一括管理する
    bool値はここでintに変換しておく
    */
    public:
        std::string itpfilename; // itpファイルのファイル名
        std::string bondfilename; // necessary:: bondファイルの名前
        std::string savedir; // necessary:: 記述子の保存dir
        double temperature = 300; // Temperature [K] (default 300K)
        double timestep    = 0.5; // MD timestep [fs] ( default 0.5fs)
        var_general();
        var_general(std::vector< std::vector<std::string> > input_general);
        var_general(YAML::Node node); // if input is yaml
};


class var_descripter{
    /*
    descripter用の変数を一括管理する
    bool値はここでintに変換しておく
    */
   public:
    int calc; // 計算するかどうかのフラグ（1がTrue，0がFalse）
    std::string directory; // xyzファイルのディレクトリ
    std::string xyzfilename; // xyzファイルのファイル名
    std::string savedir; // 記述子の保存dir
    std::string descmode; // 記述子の計算モード（1:nonwan，2:wan）
    int step; // 計算するステップ数(optional)
    // 初期値指定する場合(optional変数)
    int haswannier = 0; // 1がTrue，0がFalse (デフォルトが0, nonwanで有効)
    int interval = 1; // trajectoryを何ステップごとに処理するか．デフォルトは毎ステップ．(optional)
    std::string desctype = "old"; // 記述子の種類 old or allinone
    int IF_COC = 0; // 1がTrue，0がFalse (デフォルトが0, COC記述子を有効化する．) //!! 未使用
    int IF_GAS = 0; // 1がTrue， gasモデル計算を有効にする場合
    var_descripter();
    var_descripter(std::vector< std::vector<std::string> > input_descripter);
    var_descripter(YAML::Node node);
};


class var_predict{
    /*
    predict用の変数を一括管理する
    */

   public:
    int calc; // 計算するかどうかのフラグ（1がTrue，0がFalse）
    std::string model_dir; // modelのディレクトリ(必須)
    std::string desc_dir; // 記述子のロードdir
    std::string modelmode; // normal or rotate (2023/4/16) // 2024/2/4 :: C++版には不要
    int bondspecies = 4 ; // デフォルトの4はメタノールに対応 
    int save_truey = 0 ; // 1がTrue，0がFalse（true_yを保存するかどうか．）
    var_predict();
    var_predict(std::vector< std::vector<std::string> > input_predict);
    var_predict(YAML::Node node);
};

} //END namespace