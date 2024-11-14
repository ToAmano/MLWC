cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

##################
#
#
# see ${CONDA_PREFIX}/share/RDKit/Docs/Book/C++Examples/CMakeLists.txt
#
#
# usage:: cmake -DCONDA_PREFIX=${CONDA_PREFIX} ..
#==================

# Project Name
project(test CXX)

# list(APPEND CMAKE_PREFIX_PATH "/Users/amano/.pyenv/versions/anaconda3-2023.03/envs/test_dieltools/lib/")
# list(APPEND CMAKE_PREFIX_PATH "/Users/amano/.pyenv/versions/anaconda3-2023.03/envs/test_dieltools/include/rdkit/")
# list(APPEND CMAKE_PREFIX_PATH "/Users/amano/.pyenv/versions/anaconda3-2023.03/envs/test_dieltools/include/boost/")
# list(APPEND CMAKE_PREFIX_PATH "/Users/amano/.pyenv/versions/anaconda3-2023.03/envs/test_dieltools/")

# https://stackoverflow.com/questions/55934253/make-cmakes-find-packageboost-find-headers-inside-conda-environment
set(BOOST_ROOT $ENV{CONDA_PREFIX})
set(RDKit_DIR  $ENV{CONDA_PREFIX}) # ${RDKit_ROOT_DIR})
message(STATUS "FOUND RDKit_DIR :: ${RDKit_DIR}")
set(RDKit_LIBRARY_DIR ${RDKit_DIR}/lib/)
set(RDKit_INCLUDE_DIR ${RDKit_DIR}/include/rdkit/)
set(BOOST_LIBRARYDIR  ${BOOST_ROOT}/lib/)

find_package(RDKit REQUIRED)
if(RDKit_FOUND)
  message(STATUS "YES!! RDkit found.")
else()
  message(STATUS "NO!! RDKit not found.")
endif()
find_package(Boost REQUIRED COMPONENTS)

message(STATUS " FOUND RDBASE             :: ${RDBASE}")
message(STATUS " FOUND RDKit_INCLUDE_DIR  :: ${RDKit_INCLUDE_DIR}")
message(STATUS " FOUND RDKit_LIBRARY_DIR  :: ${RDKit_LIBRARY_DIR}")
message(STATUS " FOUND BOOST_LIBRARYDIR   :: ${BOOST_LIBRARYDIR}")

# Create Executable File
add_executable(${PROJECT_NAME}  example3.cpp )
# target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
# target_link_libraries(${PROJECT_NAME} ${RDKit_LIBRARIES})

# C++17
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

# OpenMP
message(STATUS "OpenMP_C_Found   : ${OpenMP_C_FOUND}")
message(STATUS "OpenMP_CXX_Found : ${OpenMP_CXX_FOUND}")

if (OpenMP_FOUND)
message(STATUS "YES!! OpenMP found.")
else (OpenMP_FOUND)
message(STATUS "No!!   OpenMP did not find.")
endif (OpenMP_FOUND)

if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "-pg ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-pg ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(Eigen3 REQUIRED)

target_include_directories(test
    PUBLIC
     ${Boost_INCLUDE_DIR}
     ${RDKit_INCLUDE_DIR}
     ${EIGEN3_INCLUDE_DIR})

target_link_directories(test
    PUBLIC
    ${RDKit_LIBRARY_DIR}
    ${BOOST_LIBRARYDIR})

 set(RDKit_LIBS RDKitChemReactions RDKitFileParsers RDKitSmilesParse RDKitDepictor
                RDKitRDGeometryLib RDKitRDGeneral RDKitSubstructMatch RDKitSubgraphs
                RDKitMolDraw2D RDKitGraphMol RDKitDistGeometry RDKitDistGeomHelpers
                RDKitMolAlign RDKitOptimizer RDKitForceField RDKitForceFieldHelpers
                RDKitAlignment RDKitForceField  RDKitMolTransforms RDKitEigenSolvers )

target_link_libraries(test
    ${RDKit_LIBS}
    ${Boost_LIBRARIES}
    Eigen3::Eigen
)

