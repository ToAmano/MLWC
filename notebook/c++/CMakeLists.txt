cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Project Name
project(test CXX)

execute_process(COMMAND uname -m
                 OUTPUT_VARIABLE IF_M1MAC
                 OUTPUT_STRIP_TRAILING_WHITESPACE)

# add path for pytorch
list(APPEND CMAKE_PREFIX_PATH "~/src/pytorch-install")

# Find Package
find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)

if (${IF_M1MAC} MATCHES "x86_64") # intel macの場合
   # set compilers
   set(CMAKE_C_COMPILER gcc)
   set(CMAKE_CXX_COMPILER g++)

   # Find Package
   find_package(Torch REQUIRED)

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
   set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -Wno-unused-command-line-argument -I${OpenMP_HOME}/include -lomp -L${OpenMP_omp_LIBRARY}" CACHE STRING "" FORCE) 
   # 同じくキャッシュ変数なのでFORCEオプションをつける
   set(OpenMP_C_FLAGS "-fopenmp -Wno-unused-command-line-argument -I${OpenMP_HOME}/include -lomp -L${OpenMP_omp_LIBRARY}" CACHE STRING "" FORCE)

   find_package(OpenMP REQUIRED)

   # Create Executable File
   add_executable(${PROJECT_NAME} c++_debug_make_descripter_for_speedup.cpp)
   target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} )
   # 最適化・警告等のオプション
   target_compile_options(${PROJECT_NAME} PUBLIC -O2 -Wall)
   # C++の標準規格の指定
   target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
   # set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

   # Create Executable File
   add_executable(${PROJECT_NAME} c++_debug_make_descripter_for_speedup.cpp)
   target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
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

