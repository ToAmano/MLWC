cmake_minimum_required(VERSION 3.27)
project(postprocess_lib
    VERSION 1.0.0
    DESCRIPTION "library for postprocess"
    # URL関係のライブラリをインストールしないと動かないことがあるので
    # コメントアウトしておきます(2019/12/02 修正)
    # HOMEPAGE_URL "https://example.com"
    LANGUAGES CXX)

message(STATUS " ==================================== ")
message(STATUS "  Building postprocess_lib ...        ")
message(STATUS " ==================================== ")


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

add_library(postprocess_lib STATIC 
            ./dielconst.cpp 
            ./convert_gas.cpp 
            ./save_dipole.cpp)

target_compile_features(postprocess_lib PRIVATE cxx_std_17)


# ohtaka
# target_link_libraries(postprocess_lib PRIVATE stdc++fs)

# target_include_directories(include_lib INTERFACE ./include)
set_target_properties(postprocess_lib
    PROPERTIES
    VERSION ${PROJECT_VERSION})

# >>> include ../include >>>>>>
# target_include_directories(include_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(
   postprocess_lib PUBLIC 
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/postprocess
    )

# ライブラリをリンクする。
target_link_libraries(
   postprocess_lib PUBLIC
   include_lib # library in ../include
   Eigen3::Eigen
)