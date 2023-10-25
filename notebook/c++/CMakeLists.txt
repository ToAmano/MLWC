#
# CMakeLists.txt for building c++
# Requirements :: libtorch,openMP
# 
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
message(STATUS " start cmake !! ")
set(CMAKE_CXX_STANDARD 17) # c++17 or above
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_CXX_FLAGS_RELEASE "-O3") 
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -O0 -D_DEBUG")

# Project Name
project(dieltools CXX)

# get machine type (intel mac or m1 mac)
execute_process(COMMAND uname -m
            OUTPUT_VARIABLE IF_M1MAC
            OUTPUT_STRIP_TRAILING_WHITESPACE)

# add path for pytorch 
# TODO :: hard code :: ここは自動でpytorchを探してきてくれないと困る！！
list(APPEND CMAKE_PREFIX_PATH "~/src/pytorch-install/")
list(APPEND CMAKE_PREFIX_PATH "${HOMEBREW_HOME}/include/")
list(APPEND CMAKE_PREFIX_PATH "/usr/local/opt/libomp/")

# これは現在のpathを追加しているのだが意味があるかは不明．
# 2023/10/14 多分いらないので削除した
# list(APPEND CMAKE_PREFIX_PATH "~/works/research/dieltools/notebook/c++/")
# list(APPEND CMAKE_PREFIX_PATH "./") # これで現在のpathを追加できるか？

# Find Package
message(STATUS " ==================================== ")
message(STATUS " searching required packages ... ")
message(STATUS " ==================================== ")

find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)  # sometimes find openMP fails...
if (OpenMP_FOUND)
   message(STATUS "Found OpenMP: TRUE")
   include_directories(${OPENMP_INCLUDE_DIR}) # これはいらないかも
   set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
   set (openmplibrary OpenMP::OpenMP_CXX)
endif()

find_package(Eigen3 REQUIRED)
if (Eigen3_FOUND)
   message(STATUS "Found Eigen3: TRUE")
   include_directories(${EIGEN3_INCLUDE_DIR})
else()
   if (EIGEN3_INCLUDE)
      include_directories(${EIGEN3_INCLUDE})
   else()
      message(FATAL_ERROR "Eigen3 was not found.
      Please specify the location of boost include directories
      via -DEIGEN3_INCLUDE option.")
   endif()
endif()
message(STATUS " ==================================== ")
message(STATUS " Finish searching required packages !! ")
message(STATUS " ==================================== ")

# set compilers (以下のようにするとデフォルトでApple clangがセットされる)
# /Library/Developer/CommandLineTools/usr/bin/c++
# set(CMAKE_C_COMPILER /Library/Developer/CommandLineTools/usr/bin/gcc)
# set(CMAKE_CXX_COMPILER /Library/Developer/CommandLineTools/usr/bin/g++)


# if (${IF_M1MAC} MATCHES "x86_64") # intel macの場合
#    message(STATUS "start intel mac mode")

#    # libompのインストール先パスをbrewコマンドを実行して取得する.
#    execute_process(COMMAND brew --prefix libomp
#    OUTPUT_VARIABLE OpenMP_HOME
#    OUTPUT_STRIP_TRAILING_WHITESPACE)  
#    message(STATUS "OpenMP Root : ${OpenMP_HOME}")
   
#    # OpenMPのインストールチェック
#    set(OpenMP_C_LIB_NAMES "omp")
#    set(OpenMP_CXX_LIB_NAMES "omp")
#    set(OpenMP_omp_LIBRARY "${OpenMP_HOME}/lib/")
#    # キャッシュ変数なのでFORCEオプションをつける
#    #!! 以下のフラグの設定はapple clangの時のもの．特にfopenmpはapple clangのみ．
#    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -Wno-unused-command-line-argument -I${OpenMP_HOME}/include -lomp -L${OpenMP_omp_LIBRARY}" CACHE STRING "" FORCE) 
#    # 同じくキャッシュ変数なのでFORCEオプションをつける
#    set(OpenMP_C_FLAGS "-fopenmp -Wno-unused-command-line-argument -I${OpenMP_HOME}/include -lomp -L${OpenMP_omp_LIBRARY}" CACHE STRING "" FORCE)
# endif()

# if (${IF_M1MAC} MATCHES "arm64") # m1 macの場合
#    message(STATUS "start m1 mac mode")

#    # libompのインストール先パスをbrewコマンドを実行して取得する.
#    execute_process(COMMAND brew --prefix libomp
#    OUTPUT_VARIABLE OpenMP_HOME
#    OUTPUT_STRIP_TRAILING_WHITESPACE)  
#    message(STATUS "OpenMP Root : ${OpenMP_HOME}")

#    # どうもvscodeだとbrewコマンドが見つからないので，下手打ちするしかない．．．
#    # TODO :: hard code
#    set(OpenMP_HOME "/opt/homebrew/opt/libomp")

#    # OpenMPのインストールチェック
#    set(OpenMP_C_LIB_NAMES "omp")
#    set(OpenMP_CXX_LIB_NAMES "omp")
#    set(OpenMP_omp_LIBRARY "${OpenMP_HOME}/lib/")
#    # キャッシュ変数なのでFORCEオプションをつける
#    #!! 以下のフラグの設定はapple clangの時のもの．特にfopenmpはapple clangのみ．
#    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -Wno-unused-command-line-argument -I${OpenMP_HOME}/include -lomp -L${OpenMP_omp_LIBRARY}" CACHE STRING "" FORCE) 
#    # 同じくキャッシュ変数なのでFORCEオプションをつける
#    set(OpenMP_C_FLAGS "-fopenmp -Wno-unused-command-line-argument -I${OpenMP_HOME}/include -lomp -L${OpenMP_omp_LIBRARY}" CACHE STRING "" FORCE)
# endif()


# openMPが見つかったかのチェック（debug）
message(STATUS "OpenMP_C_Found   : ${OpenMP_C_FOUND}")
message(STATUS "OpenMP_CXX_Found : ${OpenMP_CXX_FOUND}")

# openMPのチェック
if (OpenMP_FOUND)
   message(STATUS "YES!! OpenMP found.")
else (OpenMP_FOUND)
   message(STATUS "No!!   OpenMP did not find.")
endif (OpenMP_FOUND)

# openMPのフラグ追加
if(OpenMP_FOUND)
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


# Create Executable File
# add_executable(${PROJECT_NAME} c++_debug_make_descripter_for_speedup.cpp)
# target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})

# add each files/headers
# include_directories("${PROJECT_SOURCE_DIR}/include")

add_subdirectory(./include)

set(SOURCES ${PROJECT_SOURCE_DIR}/main.cpp   # main file here !!
            ${PROJECT_SOURCE_DIR}/predict.cpp
            ${PROJECT_SOURCE_DIR}/atoms_core.cpp
            ${PROJECT_SOURCE_DIR}/mol_core.cpp
            ${PROJECT_SOURCE_DIR}/atoms_io.cpp
            ${PROJECT_SOURCE_DIR}/descriptor.cpp
            ${PROJECT_SOURCE_DIR}/atoms_asign_wcs.cpp )

set(HEADERS ${PROJECT_SOURCE_DIR}/predict.hpp
            ${PROJECT_SOURCE_DIR}/atoms_core.hpp
            ${PROJECT_SOURCE_DIR}/mol_core.hpp
            ${PROJECT_SOURCE_DIR}/atoms_io.hpp
            ${PROJECT_SOURCE_DIR}/descriptor.hpp
            ${PROJECT_SOURCE_DIR}/atoms_asign_wcs.hpp )

message("MPI_C_LIBRARIES = ${MPI_C_LIBRARIES}")
message("MPI_CXX_LIBRARIES = ${MPI_CXX_LIBRARIES}")
# Executable
add_executable(dieltools ${SOURCES} ${HEADERS})
target_link_libraries(dieltools include_lib) # library in ./include
# target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} ${LAPACK_LIBRARIES} ${MPI_CXX_LIBRARIES} ${openmplibrary})

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

# Create Executable File
# add_executable(${PROJECT_NAME} main.cpp)
# include TORCH, Eigen3, stdc++fs( for filesystem)
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} Eigen3::Eigen ${openmplibrary} stdc++fs)
# 最適化・警告等のオプション
target_compile_options(${PROJECT_NAME} PUBLIC -O2 -Wall -Wextra)
# C++の標準規格の指定
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
# set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)


# https://qiita.com/iwatake2222/items/064953735bf244546b24
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
