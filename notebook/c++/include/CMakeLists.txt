cmake_minimum_required(VERSION 3.1)
project(include_lib
    VERSION 1.0.0
    DESCRIPTION "small library for treating vectors"
    # URL関係のライブラリをインストールしないと動かないことがあるので
    # コメントアウトしておきます(2019/12/02 修正)
    # HOMEPAGE_URL "https://example.com"
    LANGUAGES CXX)
add_library(include_lib STATIC ./printvec.cpp ./savevec.cpp ./constant.cpp)
target_compile_features(include_lib PRIVATE cxx_std_17)
# target_include_directories(include_lib INTERFACE ./include)
set_target_properties(include_lib
    PROPERTIES
    VERSION ${PROJECT_VERSION})
