#
# CMakeLists.txt for building c++
# Requirements :: libtorch,openMP
# 
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17) # c++17 or above
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_CXX_FLAGS_RELEASE "-O3") 
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -O0 -D_DEBUG")


# Project Name
project(test CXX)

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

# testについて
# TODO :: testは別途含めるようにした方が良い．
include(FetchContent)
FetchContent_Declare(catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.1.0 )
FetchContent_MakeAvailable(Catch2)
enable_testing()


# get machine type (intel mac or m1 mac)
execute_process(COMMAND uname -m
                 OUTPUT_VARIABLE IF_M1MAC
                 OUTPUT_STRIP_TRAILING_WHITESPACE)

# add path for pytorch
list(APPEND CMAKE_PREFIX_PATH "~/src/pytorch-install/")

# add path for my library
list(APPEND CMAKE_PREFIX_PATH "~/works/research/dieltools/notebook/c++/")
set(My_LIBRARY "~/works/research/dieltools/notebook/c++")
# 以降のターゲット全てにincludeディレクトリを適用
include_directories(${MY_LIBERARY})
# Find Package
find_package(Torch REQUIRED)
# find_package(OpenMP REQUIRED)

# set compilers (以下のようにするとデフォルトでApple clangがセットされる)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

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
   set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -Wno-unused-command-line-argument -I${OpenMP_HOME}/include -I${MY_LIBRARY} -L${MY_LIBRARY} -lomp -L${OpenMP_omp_LIBRARY}" CACHE STRING "" FORCE) 
   # 同じくキャッシュ変数なのでFORCEオプションをつける
   set(OpenMP_C_FLAGS "-fopenmp -Wno-unused-command-line-argument -I${OpenMP_HOME}/include -I${MY_LIBRARY} -I${MY_LIBRARY} -L${MY_LIBRARY} -lomp -L${OpenMP_omp_LIBRARY}" CACHE STRING "" FORCE)
   find_package(OpenMP REQUIRED)

   # Create Executable File
   add_executable(${PROJECT_NAME} Test1_c++_bfs.cpp)
   target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} )
   # 最適化・警告等のオプション
   target_compile_options(${PROJECT_NAME} PUBLIC -O2 -Wall)
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


# 以下テストに関すること
add_executable(test_fibonacci Test1_c++_bfs.cpp)
target_include_directories(test_fibonacci PRIVATE ./)
target_compile_options(test_fibonacci PRIVATE -Wall -Wextra)
target_link_libraries(test_fibonacci Catch2::Catch2WithMain)
# target_link_libraries(test_fibonacci printvec) # 自作libraryのリンク
# target_link_libraries(test_fibonacci mol_core) # 自作libraryのリンク
# target_link_libraries(test_fibonacci descriptor) # 自作libraryのリンク
# target_link_libraries(test_fibonacci atoms_core) # 自作libraryのリンク

target_link_libraries(test_fibonacci Eigen3::Eigen)
target_link_libraries(test_fibonacci ${TORCH_LIBRARIES} )
add_test(test_fibonacci test_fibonacci)

