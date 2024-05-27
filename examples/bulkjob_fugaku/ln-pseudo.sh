#!/bin/bash


current_dir=`dirname $0`

for i in {0..9197};
do
    dir=bulkjob/struc_$i
    cp -r cpmd_test/pseudo/ $dir/
done
