# 通常のコンパイル方法

g++ example3.cpp -c -I/Users/amano/rdkit/ -I/Users/amano/rdkit/rdkit/ -std=c++17
g++ example3.o -o example3 -L/Users/amano/src/rdkit/lib/ -lRDKitGraphMol -lRDKitFileParsers -llibRDKitRDGeometryLib

# 実行時には，LIBRARY_PATHにrdkitのpathを追加する
export DYLD_LIBRARY_PATH=/Users/amano/src/rdkit/lib/:${DYLD_LIBRARY_PATH}



# conda installのrdkitを利用する方法
g++ example3.cpp -c -I/Users/amano/anaconda3/envs/dieltools/include/rdkit/  -std=c++17
g++ example3.o -o example3 -L/Users/amano/src/rdkit/lib/ -lRDKitGraphMol -lRDKitFileParsers -libRDKitRDGeometryLib


# 2024/6/6
condaのrdkitや，自分で入れたrdkitを使うと，boost関連のエラーが出てしまってうまく行かなかった．
特にmacの場合，homebrewでboostを入れているとcondaでboostを入れてもhomebrewが優先されてしまっている印象を受ける．そこで，homebrewのrdkitを使うとinstallに成功した．

ただし，これはmacだからこれでいいんだけど，他の環境ではcondaで完結した方法があると嬉しいのだが．．．


- 以下のコマンドでどこに入っているか確認できる
- python -c "import rdkit; print(rdkit.__file__)"
