// #include <torch/torch.h>
#include "torch/script.h"
#include <iostream>

int main() {
    // torch::jit::script::Module 型で module 変数の定義
    // torch::jit::script::Module module;
    torch::Tensor tensor = torch::rand({2, 3});
    //std::cout << tensor << std::endl;
    // std::cout << "test" << std::endl;

}