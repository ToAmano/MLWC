# 通常のコンパイル方法

g++ example3.cpp -c -I/Users/amano/rdkit/ -I/Users/amano/rdkit/rdkit/ -std=c++17
g++ example3.o -o example3 -L/Users/amano/src/rdkit/lib/ -lRDKitGraphMol -lRDKitFileParsers


# 実行時には，LIBRARY_PATHにrdkitのpathを追加する
export DYLD_LIBRARY_PATH=/Users/amano/src/rdkit/lib/:${DYLD_LIBRARY_PATH}
