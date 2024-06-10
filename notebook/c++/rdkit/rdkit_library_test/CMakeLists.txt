#
# CMakeLists.txt for building c++
# Requirements :: libtorch,openMP
# 

# 参考文献
# https://qiita.com/hi_to_san/items/00a5e9a75a8876b39492

# CMakeでOpenMPアプリケーション開発 - Qiita
# https://qiita.com/iwatake2222/items/5800fda029019ce8a276
# if(OpenMP_FOUND)
#     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# endif()

cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
message(STATUS " start cmake !! ")
# https://cmake.org/cmake/help/latest/guide/tutorial/A%20Basic%20Starting%20Point.html
set(CMAKE_CXX_STANDARD 20) # c++17 or above
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_CXX_FLAGS_RELEASE "-O3") 
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -O0 -D_DEBUG")

# Project Name
project(dieltools CXX)

message(STATUS " subdirectories ...")
add_subdirectory(./chemicalbond) # add CMakeLists.txt in ./chemicalbond

# Executable
add_executable(dieltools main.cpp)
# my-own library and other distributed libraries
# include TORCH, Eigen3, stdc++fs( for filesystem)
target_link_libraries(dieltools PUBLIC
                     test
                     ) # library in ./include, ./postprocess, ./chemicalbond

target_include_directories(dieltools PUBLIC
                              ./chemicalbond/
                           )

target_link_directories(dieltools PUBLIC
                              ./chemicalbond/
                              )

                           
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)

# 最適化・警告等のオプション
target_compile_options(${PROJECT_NAME} PUBLIC -O2 -Wall -Wextra)
# C++の標準規格の指定
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
# set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

