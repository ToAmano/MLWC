#include <iostream>
#include "torch/script.h"

int main(void){
    torch::Tensor x = torch::full({3, 3}, 1.5, torch::TensorOptions().dtype(torch::kFloat));
    std::cout << x << std::endl;
    return 0;
}
