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
#include "torch/script.h" // pytorch

int main(){
        // torch::jit::script::Module 型で module 変数の定義
        torch::jit::script::Module module;
        // 変換した学習済みモデルの読み込み
        module = torch::jit::load("/Users/amano/works/research/dieltools/notebook/c++/202306014_model_rotate/model_ch.pt");
        // モデルへのサンプル入力テンソル（形式は1,288の形！！）
        torch::Tensor input = torch::ones({1, 288}).to("cpu");
        std::cout << input << std::endl;
        // 推論と同時に出力結果を変数に格納
        // auto elements = module.forward({input}).toTuple() -> elements();
        torch::Tensor elements = module.forward({input}).toTensor() ;

        // 出力結果
        std::cout << elements << std::endl;
        // auto output = elements[0].toTensor();

    }
