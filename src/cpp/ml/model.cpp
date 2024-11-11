#include <torch/script.h>
#include <iostream>
#include <string>

bool has_parameter(torch::jit::script::Module& model, const std::string& param_name) {
    // モデル内のすべてのパラメータを確認し、指定された名前のパラメータが存在するかを返す
    for (const auto& p : model.named_parameters()) {
        if (p.name == param_name) {
            return true;
        }
    }
    return false;
}