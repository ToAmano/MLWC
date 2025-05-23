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

cmake_minimum_required(VERSION 3.20)
message(STATUS " start cmake !! ")
# https://cmake.org/cmake/help/latest/guide/tutorial/A%20Basic%20Starting%20Point.html
set(CMAKE_CXX_STANDARD 20) # c++20 or above
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
# list(APPEND CMAKE_PREFIX_PATH "~/src/pytorch-install/")
# list(APPEND CMAKE_PREFIX_PATH "~/src/libtorch-macos-arm64-2.4.0/")
# list(APPEND CMAKE_PREFIX_PATH "/home/k0151/k015124//src/libtorch/") # ohtaka

# path to eigen3
# list(APPEND CMAKE_PREFIX_PATH "/home/k0151/k015124/src/eigen-3.4.0/") 
# list(APPEND CMAKE_MODULE_PATH "/home/k0151/k015124/src/eigen-3.4.0/cmake/") 


# Find Package
message(STATUS "==================================== ")
message(STATUS " searching required packages ... ")
message(STATUS "==================================== ")


# automatically find libtorch
# https://qiita.com/ukyoda/items/629fb67d670cf7e8808a
if (NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE "python")
endif()

# Get Pytorch library cmake_prefix_path
# execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path, end='')" OUTPUT_VARIABLE pytorch_path)
execute_process(COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path, end='')" OUTPUT_VARIABLE pytorch_path)
list(APPEND CMAKE_PREFIX_PATH "${pytorch_path}") # CMAKE_PREFIX_PATHに追加
# list(APPEND CMAKE_PREFIX_PATH "/Users/amano/.pyenv/versions/anaconda3-2023.03/envs/test_dieltools/lib/python3.11/site-packages/torch/share/cmake")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

find_package(RDKit REQUIRED)
if(RDKit_FOUND)
  message(STATUS "YES!! RDkit found.")
  message(STATUS " FOUND RDBASE             :: ${RDBASE}")
  message(STATUS " FOUND RDKit_INCLUDE_DIR  :: ${RDKit_INCLUDE_DIR}")
  message(STATUS " FOUND RDKit_LIBRARY_DIR  :: ${RDKit_LIBRARY_DIR}")
else()
  message(STATUS "NO!! RDKit not found.")
endif()

find_package(Torch REQUIRED)
if (Torch_FOUND)
   message(STATUS " Found Torch: TRUE")
   # This line cause bag in ubuntu
   # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
   message(STATUS "TORCH_INCLUDE_DIRS  : ${TORCH_INCLUDE_DIRS}")
   message(STATUS "TORCH_LIBRARIES     : ${TORCH_LIBRARIES}")
   message(STATUS "TORCH_LIBRARY_DIRS  : ${TORCH_LIBRARY_DIRS}")   
else ()
   message(FATAL_ERROR "Found Torch: FALSE")
endif()

find_package(OpenMP REQUIRED)  # sometimes find openMP fails...
if (OpenMP_FOUND)
   message(STATUS " Found OpenMP: TRUE")
   message(STATUS " OPENMP_INCLUDE_DIR : ${OPENMP_INCLUDE_DIR}")
   # include_directories(${OPENMP_INCLUDE_DIR}) # これはいらないかも
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
   set (openmplibrary OpenMP::OpenMP_CXX)
else ()
   message(FATAL_ERROR "Found OpenMP: FALSE")
endif()

find_package(Eigen3 REQUIRED)
if (Eigen3_FOUND)
   message(STATUS " Found Eigen3: TRUE")
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

message(STATUS "==================================== ")
message(STATUS " Finish searching required packages !! ")
message(STATUS "==================================== ")

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


# openMPが見つかったかのチェックを別の方法でやる（debug）
message(STATUS " OpenMP_C_Found   : ${OpenMP_C_FOUND}")
message(STATUS " OpenMP_CXX_Found : ${OpenMP_CXX_FOUND}")


# Find Package
message(STATUS "==================================== ")
message(STATUS " COMPIER VERSIONS ... ")
message(STATUS "==================================== ")
# https://stackoverflow.com/questions/51154151/how-to-check-what-compiler-cmake-is-using
message(STATUS " C++ Compiler Version: ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS " C   Compiler Version: ${CMAKE_C_COMPILER_VERSION}")
# https://discourse.cmake.org/t/find-library-does-not-find-stdc-fs/4764/4
message(STATUS " C++ Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "")


# Create Executable File
# add_executable(${PROJECT_NAME} c++_debug_make_descripter_for_speedup.cpp)
# target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})

# add each files/headers
# include_directories("${PROJECT_SOURCE_DIR}/include")

message(STATUS " subdirectories ...")
add_subdirectory(./include) # add CMakeLists.txt in ./include
add_subdirectory(./postprocess) # add CMakeLists.txt in ./postprocess
add_subdirectory(./chemicalbond) # add CMakeLists.txt in ./chemicalbond


set(SOURCES ${PROJECT_SOURCE_DIR}/main.cpp   # main file here !!
            ${PROJECT_SOURCE_DIR}/predict.cpp
            ${PROJECT_SOURCE_DIR}/parse.cpp
            ${PROJECT_SOURCE_DIR}/atoms_core.cpp
            ${PROJECT_SOURCE_DIR}/atoms_io.cpp
            ${PROJECT_SOURCE_DIR}/descriptor.cpp
            ${PROJECT_SOURCE_DIR}/atoms_asign_wcs.cpp
            ${PROJECT_SOURCE_DIR}/module_xyz.cpp
            ${PROJECT_SOURCE_DIR}/module_bond.cpp
            ${PROJECT_SOURCE_DIR}/module_torch.cpp
            ${PROJECT_SOURCE_DIR}/module_input.cpp
            )

set(HEADERS ${PROJECT_SOURCE_DIR}/predict.hpp
            ${PROJECT_SOURCE_DIR}/parse.hpp
            ${PROJECT_SOURCE_DIR}/atoms_core.hpp
            ${PROJECT_SOURCE_DIR}/atoms_io.hpp
            ${PROJECT_SOURCE_DIR}/descriptor.hpp
            ${PROJECT_SOURCE_DIR}/atoms_asign_wcs.hpp
            ${PROJECT_SOURCE_DIR}/module_xyz.hpp
            ${PROJECT_SOURCE_DIR}/module_bond.hpp
            ${PROJECT_SOURCE_DIR}/module_torch.hpp
            ${PROJECT_SOURCE_DIR}/module_input.hpp
            )


message(STATUS " MPI_C_LIBRARIES = ${MPI_C_LIBRARIES}")
message(STATUS " MPI_CXX_LIBRARIES = ${MPI_CXX_LIBRARIES}")

# https://github.com/jbeder/yaml-cpp
include(FetchContent)

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG 0.8.0
  # GIT_TAG <tag_name> # Can be a tag (yaml-cpp-x.x.x), a commit hash, or a branch name (master)
)
FetchContent_GetProperties(yaml-cpp)

if(NOT yaml-cpp_POPULATED)
  message(STATUS "Fetching yaml-cpp...")
  FetchContent_Populate(yaml-cpp)
  add_subdirectory(${yaml-cpp_SOURCE_DIR} ${yaml-cpp_BINARY_DIR})
endif()

# target_link_libraries(YOUR_LIBRARY PUBLIC yaml-cpp::yaml-cpp) # The library or executable that

# todo 
include(FetchContent)
FetchContent_Declare(catch2
GIT_REPOSITORY https://github.com/catchorg/Catch2.git
GIT_TAG v3.1.0
)
FetchContent_MakeAvailable(Catch2)
enable_testing()


# Executable
add_executable(dieltools ${SOURCES} ${HEADERS})
# my-own library and other distributed libraries
# include TORCH, Eigen3, stdc++fs( for filesystem)
target_link_libraries(dieltools PUBLIC
                     include_lib
                     postprocess_lib
                     chemicalbond_lib
		     RDKitSmilesParse
                     yaml-cpp::yaml-cpp
                     ${TORCH_LIBRARIES}
                     Eigen3::Eigen
                     ${openmplibrary}
                     # ${RDKit_LIBS}
                     # ${Boost_LIBRARIES}
                     # stdc++fs # ohtaka
                     ) # library in ./include, ./postprocess, ./chemicalbond
# target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} ${LAPACK_LIBRARIES} ${MPI_CXX_LIBRARIES} ${openmplibrary})
                     
# https://github.com/Kitware/CMake/blob/master/Help/guide/tutorial/Step3/CMakeLists.txt
# https://cmake.org/cmake/help/latest/guide/tutorial/Adding%20a%20Library.html
target_include_directories(dieltools PUBLIC
                              ./include/
                              ./postprocess/
                              ./chemicalbond/
               		      ${RDKit_INCLUDE_DIR}
                              ${TORCH_INCLUDE_DIRS}
                              ${OPENMP_INCLUDE_DIR}
                           )

target_link_directories(dieltools PUBLIC
                              ./include/
                              ./postprocess/
                              ./chemicalbond/
			      ${RDKit_LIBRARY_DIR}
			      # /Users/amano/src/libtorch-macos-arm64-2.4.0/lib/
                              )
                           
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)

# Compiler options/features
target_compile_options(${PROJECT_NAME} PUBLIC -O2 -Wall -Wextra)
# C++ standard
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
# set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

# https://qiita.com/iwatake2222/items/064953735bf244546b24
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# test 
add_executable(test_parse_yaml test_parse_yaml.cpp parse.cpp parse.hpp)
target_compile_options(test_parse_yaml PRIVATE -Wall -Wextra)
target_link_libraries(test_parse_yaml Catch2::Catch2WithMain yaml-cpp::yaml-cpp)
add_test(NAME hoge_test # テスト名は hoge_test
         COMMAND test_parse_yaml) # name command
