cmake_minimum_required(VERSION 3.27)
project(chemicalbond_lib
    VERSION 1.0.0
    DESCRIPTION "Library for treating chemical bond information in molecules"
    # URL関係のライブラリをインストールしないと動かないことがあるので
    # コメントアウトしておきます(2019/12/02 修正)
    # HOMEPAGE_URL "https://example.com"
    LANGUAGES CXX)

message(STATUS " ==================================== ")
message(STATUS "  Building chemicalbond_lib ...       ")
message(STATUS " ==================================== ")

add_library(chemicalbond_lib STATIC ./mol_core.cpp ) # ./mol_core_rdkit.cpp)
target_compile_features(chemicalbond_lib PUBLIC cxx_std_17)
# target_include_directories(include_lib INTERFACE ./include)
set_target_properties(chemicalbond_lib
    PROPERTIES
    VERSION ${PROJECT_VERSION})

# automatically find libtorch
# https://qiita.com/ukyoda/items/629fb67d670cf7e8808a
if (NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE "python3")
endif()

# if (POLICY CMP0074)
#   cmake_policy(SET CMP0074 NEW)
# endif()

# https://stackoverflow.com/questions/55934253/make-cmakes-find-packageboost-find-headers-inside-conda-environment
set(Boost_DIR $ENV{CONDA_PREFIX})  # CACHE STRING "description")
set(BOOST_ROOT $ENV{CONDA_PREFIX}) # CACHE STRING "description")
set(RDKit_DIR  $ENV{CONDA_PREFIX}) # ${RDKit_ROOT_DIR})
message(STATUS "FOUND RDKit_DIR :: ${RDKit_DIR}")
set(RDKit_LIBRARY_DIR ${RDKit_DIR}/lib/)
set(RDKit_INCLUDE_DIR ${RDKit_DIR}/include/rdkit/)
# set(BOOST_LIBRARYDIR  ${Boost_DIR}/lib/)
set(Boost_NO_SYSTEM_PATHS ON CACHE STRING "description")

find_package(RDKit REQUIRED)
if(RDKit_FOUND)
  message(STATUS "YES!! RDkit found.")
else()
  message(STATUS "NO!! RDKit not found.")
endif()
find_package(Boost REQUIRED COMPONENTS)
if(Boost_FOUND)
  message(STATUS "YES!! Boost found.")
else()
  message(STATUS "NO!! Boost not found.")
endif()

message(STATUS " FOUND RDBASE             :: ${RDBASE}")
message(STATUS " FOUND RDKit_INCLUDE_DIR  :: ${RDKit_INCLUDE_DIR}")
message(STATUS " FOUND RDKit_LIBRARY_DIR  :: ${RDKit_LIBRARY_DIR}")
message(STATUS "")
message(STATUS "")
message(STATUS " FOUND BOOST_LIBRARYDIR   :: ${BOOST_LIBRARYDIR}")
message(STATUS " FOUND BOOST_LIBRARIES    :: ${Boost_LIBRARIES}") # https://cmake.org/cmake/help/v3.0/module/FindBoost.html
message(STATUS " FOUND Boost_LIBRARY_DIRS :: ${Boost_LIBRARY_DIRS}") # https://cmake.org/cmake/help/v3.0/module/FindBoost.html
message(STATUS " FOUND Boost_INCLUDE_DIRS :: ${Boost_INCLUDE_DIRS}")

# C++17
# set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)


# >>> include ../include >>>>>>
# target_include_directories(include_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(
    chemicalbond_lib  PUBLIC 
    ${PROJECT_SOURCE_DIR}/include
    ${RDKit_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_directories(
    chemicalbond_lib  PUBLIC
    ${RDKit_LIBRARY_DIR}
    ${Boost_LIBRARIES})

# 
 set(RDKitChemReactions RDKit_LIBS RDKitFileParsers RDKitSmilesParse RDKitDepictor
 RDKitRDGeometryLib RDKitRDGeneral RDKitSubstructMatch RDKitSubgraphs
 RDKitMolDraw2D RDKitGraphMol RDKitDistGeometry RDKitDistGeomHelpers
 RDKitMolAlign RDKitOptimizer RDKitForceField RDKitForceFieldHelpers
 RDKitAlignment RDKitForceField  RDKitMolTransforms RDKitEigenSolvers )
# Link libraries
target_link_libraries(
    ${PROJECT_NAME} 
    include_lib # library in ../include
    ${RDKit_LIBS}
    ${Boost_LIBRARIES}
    Eigen3::Eigen
)