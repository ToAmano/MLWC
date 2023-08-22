#include <filesystem>
#include <iostream>

// https://e-penguiner.com/cpp-function-check-file-exist/#index_id2
bool IsFileExist(const std::string& name) {
    return std::filesystem::is_regular_file(name);
}