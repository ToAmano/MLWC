cmake_minimum_required(VERSION 3.1)
project(postprocess_lib
    VERSION 1.0.0
    DESCRIPTION "library for postprocess"
    # URL関係のライブラリをインストールしないと動かないことがあるので
    # コメントアウトしておきます(2019/12/02 修正)
    # HOMEPAGE_URL "https://example.com"
    LANGUAGES CXX)
add_library(postprocess_lib STATIC ./dielconst.cpp)
target_compile_features(postprocess_lib PRIVATE cxx_std_17)
# target_include_directories(include_lib INTERFACE ./include)
set_target_properties(postprocess_lib
    PROPERTIES
    VERSION ${PROJECT_VERSION})
