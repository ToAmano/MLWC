#!/bin/bash

# データ抽出
for i in {0..10000}; do
    # stdoutのファイル名を取得する
    echo ${i}
    cat bulkjob/struc_${i}/tmp/WC_SPREAD >> WC_SPREAD_merge.txt
done