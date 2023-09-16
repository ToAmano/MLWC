#!/bin/bash

cd csvfiles/
for f in ./*;
do
    # ファイル一つ毎の処理
    echo "file: $f"
    CPmake.py smile ${f}
done



# 


