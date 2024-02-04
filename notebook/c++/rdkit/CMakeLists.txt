cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Project Name
project(test CXX)


list(APPEND CMAKE_PREFIX_PATH "/Users/amano/src/rdkit/rdkit/")
list(APPEND CMAKE_PREFIX_PATH "/Users/amano/src/rdkit/")

set(RDKit_DIR ${RDKit_ROOT_DIR})
# Find Package
# find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)
# find_package(RDKit REQUIRED)

message(STATUS "  Boost_LIBRARIES  =${RDBASE}")

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


# Create Executable File
add_executable(${PROJECT_NAME}  example3.cpp )
# target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
# target_link_libraries(${PROJECT_NAME} ${RDKit_LIBRARIES})

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)


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


message(STATUS "${RDKit_INCLUDE_DIRS}")
target_include_directories(test
    PRIVATE
    /Users/amano/src/rdkit/rdkit/)
#    ${RDKit_INCLUDE_DIRS})

target_link_directories(test
    PUBLIC
    /Users/amano/src/rdkit/lib/)
#    /Users/amano/anaconda3/envs/dieltools/lib/)
#     ${RDKIT_LIBRARIES})


target_link_libraries(test
#    ChemReactions
#    DataStructs
#    Depictor
    RDKitFileParsers
#    GenericGroups
    RDKitGraphMol
#    MolDraw2D
#    MolTransforms
#    RDGeneral
#    RDGeometryLib
#    RingDecomposerLib
#    SmilesParse
#    SubstructMatch
#    coordgen
)

if(RDKit_FOUND)
  message(STATUS "YES!! RDkit found.")
endif()
