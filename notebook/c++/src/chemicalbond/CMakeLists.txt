cmake_minimum_required(VERSION 3.1)
project(chemicalbond_lib
    VERSION 1.0.0
    DESCRIPTION "Library for treating chemical bond information in molecules"
    # URL関係のライブラリをインストールしないと動かないことがあるので
    # コメントアウトしておきます(2019/12/02 修正)
    # HOMEPAGE_URL "https://example.com"
    LANGUAGES CXX)
add_library(chemicalbond_lib STATIC ./mol_core.cpp )
target_compile_features(chemicalbond_lib PUBLIC cxx_std_17)
# target_include_directories(include_lib INTERFACE ./include)
set_target_properties(chemicalbond_lib
    PROPERTIES
    VERSION ${PROJECT_VERSION})


# >>> include ../include >>>>>>
# target_include_directories(include_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(include_lib PUBLIC ${PROJECT_SOURCE_DIR}/include)

# ライブラリをリンクする。
target_link_libraries(${PROJECT_NAME} include_lib) # library in ../include