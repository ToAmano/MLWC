#
# CMakeLists.txt for building c++
# Requirements :: libtorch,openMP
# 

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
message(STATUS "start cmake !! ")


# Project Name
project(dieltools CXX)

set(CMAKE_CXX_STANDARD 17)

# get machine type (intel mac or m1 mac)
execute_process(COMMAND uname -m
                 OUTPUT_VARIABLE IF_M1MAC
                 OUTPUT_STRIP_TRAILING_WHITESPACE)

# add path for pytorch
list(APPEND CMAKE_PREFIX_PATH "~/src/pytorch-install/")
list(APPEND CMAKE_PREFIX_PATH "${HOMEBREW_HOME}/include/")
#
list(APPEND CMAKE_PREFIX_PATH "~/works/research/dieltools/notebook/c++/")

# 自作ライブラリ
add_library(printvec STATIC ../../include/printvec.cpp)
# add_library(npy STATIC ../../npy.hpp)
#add_library(atoms_core STATIC atoms_core.cpp)
#add_library(atoms_io STATIC atoms_io.cpp)
#add_library(mol_core STATIC mol_core.cpp)


# Find Package
find_package(Torch REQUIRED)
# find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)

# set compilers (以下のようにするとデフォルトでApple clangがセットされる)
# /Library/Developer/CommandLineTools/usr/bin/c++
# set(CMAKE_C_COMPILER /Library/Developer/CommandLineTools/usr/bin/gcc)
# set(CMAKE_CXX_COMPILER /Library/Developer/CommandLineTools/usr/bin/g++)

# # testについて
# include(FetchContent)
# FetchContent_Declare(catch2
#   GIT_REPOSITORY https://github.com/catchorg/Catch2.git
#   GIT_TAG v3.1.0 )
# FetchContent_MakeAvailable(Catch2)
# enable_testing()



if (${IF_M1MAC} MATCHES "x86_64") # intel macの場合
   message(STATUS "start intel mac mode")

   # libompのインストール先パスをbrewコマンドを実行して取得する.
   execute_process(COMMAND brew --prefix libomp
   OUTPUT_VARIABLE OpenMP_HOME
   OUTPUT_STRIP_TRAILING_WHITESPACE)  
   message(STATUS "OpenMP Root : ${OpenMP_HOME}")
   
   # OpenMPのインストールチェック
   set(OpenMP_C_LIB_NAMES "omp")
   set(OpenMP_CXX_LIB_NAMES "omp")
   set(OpenMP_omp_LIBRARY "${OpenMP_HOME}/lib/")
   # キャッシュ変数なのでFORCEオプションをつける
   #!! 以下のフラグの設定はapple clangの時のもの．特にfopenmpはapple clangのみ．
   set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -Wno-unused-command-line-argument -I${OpenMP_HOME}/include -lomp -L${OpenMP_omp_LIBRARY}" CACHE STRING "" FORCE) 
   # 同じくキャッシュ変数なのでFORCEオプションをつける
   set(OpenMP_C_FLAGS "-fopenmp -Wno-unused-command-line-argument -I${OpenMP_HOME}/include -lomp -L${OpenMP_omp_LIBRARY}" CACHE STRING "" FORCE)
   find_package(OpenMP REQUIRED)

   # Create Executable File
   add_executable(${PROJECT_NAME} test_aseio.cpp)
   target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} )
   target_link_libraries(${PROJECT_NAME} printvec)
   
   # 最適化・警告等のオプション
   target_compile_options(${PROJECT_NAME} PUBLIC -O2 -Wall -Wextra)
   # C++の標準規格の指定
   target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
   # set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

   # Create Executable File
   # add_executable(${PROJECT_NAME} c++_debug_make_descripter_for_speedup.cpp)
   # target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
   set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

   message(STATUS "OpenMP_C_Found   : ${OpenMP_C_FOUND}")
   message(STATUS "OpenMP_CXX_Found : ${OpenMP_CXX_FOUND}")

   if (OpenMP_FOUND)
      message(STATUS "YES!! OpenMP found.")
   else (OpenMP_FOUND)
      message(STATUS "No!!   OpenMP did not find.")
   endif (OpenMP_FOUND)

   if(OpenMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
   endif()
endif()

if (${IF_M1MAC} MATCHES "arm64") # m1 macの場合
   message(STATUS "start m1 mac mode")

   # libompのインストール先パスをbrewコマンドを実行して取得する.
   execute_process(COMMAND brew --prefix libomp
   OUTPUT_VARIABLE OpenMP_HOME
   OUTPUT_STRIP_TRAILING_WHITESPACE)  
   message(STATUS "OpenMP Root : ${OpenMP_HOME}")

   # どうもvscodeだとbrewコマンドが見つからないので，下手打ちするしかない．．．
   # TODO :: hard code
   set(OpenMP_HOME "/opt/homebrew/opt/libomp")

   # OpenMPのインストールチェック
   set(OpenMP_C_LIB_NAMES "omp")
   set(OpenMP_CXX_LIB_NAMES "omp")
   set(OpenMP_omp_LIBRARY "${OpenMP_HOME}/lib/")
   # キャッシュ変数なのでFORCEオプションをつける
   #!! 以下のフラグの設定はapple clangの時のもの．特にfopenmpはapple clangのみ．
   set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -Wno-unused-command-line-argument -I${OpenMP_HOME}/include -lomp -L${OpenMP_omp_LIBRARY}" CACHE STRING "" FORCE) 
   # 同じくキャッシュ変数なのでFORCEオプションをつける
   set(OpenMP_C_FLAGS "-fopenmp -Wno-unused-command-line-argument -I${OpenMP_HOME}/include -lomp -L${OpenMP_omp_LIBRARY}" CACHE STRING "" FORCE)
   find_package(OpenMP REQUIRED)

   # Create Executable File
   add_executable(${PROJECT_NAME} test_aseio.cpp)
   target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} )
   target_link_libraries(${PROJECT_NAME} printvec )

# 最適化・警告等のオプション
   target_compile_options(${PROJECT_NAME} PUBLIC -O2 -Wall -Wextra)
   # C++の標準規格の指定
   target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
   # set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

   # Create Executable File
   # add_executable(${PROJECT_NAME} c++_debug_make_descripter_for_speedup.cpp)
   # target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
   set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

   message(STATUS "OpenMP_C_Found   : ${OpenMP_C_FOUND}")
   message(STATUS "OpenMP_CXX_Found : ${OpenMP_CXX_FOUND}")

   if (OpenMP_FOUND)
      message(STATUS "YES!! OpenMP found.")
   else (OpenMP_FOUND)
      message(STATUS "No!!   OpenMP did not find.")
   endif (OpenMP_FOUND)

   if(OpenMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
   endif()
endif()

# link
target_link_libraries(${PROJECT_NAME} printvec)
target_include_directories(${PROJECT_NAME} PRIVATE ../../)


# test
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)


# 以下テストに関すること
# add_executable(test_fibonacci tests/test1/Test1_c++_bfs.cpp)
# target_include_directories(test_fibonacci PRIVATE ./)
# target_compile_options(test_fibonacci PRIVATE -Wall -Wextra)
# target_link_libraries(test_fibonacci Catch2::Catch2WithMain)
# target_link_libraries(test_fibonacci printvec) # 自作libraryのリンク
# target_link_libraries(test_fibonacci Eigen3::Eigen)
# target_link_libraries(test_fibonacci ${TORCH_LIBRARIES} )
# add_test(test_fibonacci test_fibonacci)

# https://qiita.com/iwatake2222/items/064953735bf244546b24
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
