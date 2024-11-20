




## gromacsからcpmdへの変換部分のスクリプトについて

コードとして，CPextract.pyとCPmake.pyというものを用意してある．大雑把にはcpmd/qeのコードからなんらかの情報を引き出すのがCPextract，逆にcpmd/qeのインプットを作成するのがCPmake.pyである．コードの使い方としては，

```
CPextract.py cpmd [sub-sub-command]
CPmake.py cpmd [sub-sub-command]
```

のように，cpmd+サブサブコマンドの組み合わせで使える．cpmdの代わりにcpを使うこともできて，こちらはqeのcp.x用のコマンド．しかし機能はcpmdに比べて限定的だ．CPextract.pyはかなり雑多な機能を備えているが，CPmake.pyの方は明快で，
- gromacsの座標データ
- 計算の種類（geometry-relax, cpmd-relax, bomd, cpmdなど）
を指定するとcpmd用インプットを作成する．サブコマンドの一覧としては以下のものを用意している．

- CPextract.py
  - energy       ENERGIESファイルを読み込んで，横軸時間，縦軸エネルギーのグラフを生成する．MDがうまく行っているかの確認のため．
  - dipole       DIPOLEファイルを読み込んで，横軸時間，縦軸双極子のグラフを生成する．
  - dfset        FTRAJECTORYファイルを読み込んで，ALAMODE用のdfsetファイルを生成する．
  - xyz          IONS+CENTERS.xyzからwannierの情報を取り除いてIONS_only.xyzを作成する．
  - sort         CPMDでは原子種を増やしすぎるとまずいので，並び替える必要がある．IONS+CENTERS.xyz中の原子を並び替える．
  - addlattice   IONS+CENTERS.xyzなどのxyzファイルに，格子定数情報を（cpmdのstdoutから）付与する．
- CPmake.py
  - georelax            cpmd.x geoetry relaxation計算
  - bomdrelax           cpmd.x bomd relaxation計算
  - bomd                cpmd.x bomd restart計算
  - bomdrestart         cpmd.x bomd restart計算
  - oneshot             cpmd.x bomd+wf oneshot計算（bulkjob用）
  - cpmd                cpmd.x cpmd restart計算
  - cpmdwan             cpmd.x cpmd+wf restart計算
  - workflow            cpmd.x bomd workflow（georelax+bomdrelax+bomdrestart）
  - workflow_cp         cpmd.x cpmd(not bomd) workflow（georelax+cpmdrelax+cpmdrestart）

最近のgromacsからbulkjobの流れを例として取り上げる．まずはgromacsでeq_pbd.trr及びeq.pdbを作成する．このファイルに対して，

- 各frameの構造を.groに書き出す
- 書き出した.groに対するcpmdの入力を作成する

という2ステップを踏む．これは1万構造に対してやると1時間くらいかかるので気長に待つ．．．

```
# ディレクトリをほって，eq.groという名前で保存する．
import ase

import mdtraj
import os
# ファイルを読み込み
traj=mdtraj.load("gromacs_test/eq_pbc.trr", top="gromacs_test/eq.pdb")

# ディレクトリをほって，そこにeq.groという名前で保存する．
num_config = len(traj)

for i in range(num_config):
    os.system("mkdir struc_{}".format(str(i)))
    traj[i].save_gro("struc_{}/final_structure.gro".format(str(i)))
```

書き出したfinal_structure.groというファイルに対して，CPmake.pyを適用する．`--type sorted`は原子種の並び替えを行う．

```
CPmake.py cpmd oneshot --i gromacs/eq.pdb --type sorted
```


以上の2ステップを，コマンドラインを使わずにpythonだけで完了することもできる．CPmake.pyの内部で読み込んでいる関数をpythonコードから呼び出す．こちらの方がgroファイルの書き出しを伴わない分多少早い気がする．

```
import ase

import mdtraj
import os

import cpmd.converter_cpmd
# ファイルを読み込み
traj=mdtraj.load("gromacs_test/eq_pbc.trr", top="gromacs_test/eq.pdb")

# ディレクトリをほって，そこにeq.groという名前で保存する．
num_config = len(traj)
print(num_config)

# まずはbulkjob用のdirをほる
os.system("mkdir bulkjob")

for i in range(num_config):
    # print("step :: ", i)
    os.system("mkdir bulkjob/struc_{}".format(str(i)))
    traj[i].save_gro("bulkjob/struc_{}/final_structure.gro".format(str(i)))

    # groを読み込み入力を作成．
    ase_atoms=ase.io.read("bulkjob/struc_{}/final_structure.gro".format(str(i)))
    test=cpmd.converter_cpmd.make_cpmdinput(ase_atoms)
    test.make_bomd_oneshot(type="sorted")

    # 作成したファイルを移動させる．
    os.system("mv bomd-oneshot.inp bulkjob/struc_{}/bomd-oneshot.inp".format(str(i)))
    os.system("mv sort_index.txt   bulkjob/struc_{}/sort_index.txt".format(str(i)))
    os.system("mkdir bulkjob/struc_{}/tmp".format(str(i)))
```

以上の流れでbulkjob/struc_0などのディレクトリにCPMDのインプットファイル`bomd-oneshot.inp`と原子種入れ替えのindexを書き出した`sort-index.txt`ができる．注意点として，これとは別に擬ポテンシャルファイルの設定が必要で，pseudoディレクトリを以下のようにコピーしてくる．

```
#!/bin/bash


current_dir=`dirname $0`

for i in {0..10000};
do
    dir=bulkjob/struc_$i
    cd $dir
    ln -s ../../cpmd_test/pseudo/ ./
    cd ../../
done
```

CPMDの富岳上での実行は，例えば

```
mpiexec -stdout-proc bomd-oneshot.out /vol0006/mdt1/data/hp230124/src/CPMD-4.3/bin/cpmd_fugakumpi/bin/cpmd.x bomd-oneshot.inp
```

でできる．注意点として，CPMDの生成するRESTART.1ファイルが結構大きくて容量を圧迫するので，計算後削除した方が良い．

```
rm tmp/RESTART.1
```

計算後に用いるファイルは，格子定数を取得する`bomd-oneshot.out`，原子種を並び替えるための`sort_index.txt`，さらにCPMDのワニエの座標出力`IONS+CENTERS.xyz`の三つ．まずは原子種の並び替えだが，これはかなり重い計算になるのでプリポストノードに投げた方が良い．一例は以下．

```
#!/bin/bash
#SBATCH -p mem2      # キューの指定
#SBATCH -n 56         # CPU数の指定
#SBATCH --mem 1500G  # メモリ量の指定
#SBATCH -t 24:00:00           # ジョブ実行時間の制限を指定
# #PJM -L "node=2"
# #PJM -L "rscgrp=small"
# #PJM -L "elapse=72:00:00"
#PJM -g "hp220331"
#PJM --name "prepost_test"
# #PJM --mpi "max-proc-per-node=48"
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM --llio "localtmp-size=10Gi"
#PJM --mail-list tragic44cg@icloud.com
#PJM -m b
#PJM -m e
#PJM -s
export PLE_MPI_STD_EMPTYFILE=off
export OMP_NUM_THREADS=1

pyenv global anaconda3-2022.05

for i in {3669..9999};
do
    cd bulkjob/struc_${i}
    echo ${i}
    CPextract.py cpmd sort -s ../../sort_index.txt -i tmp/IONS+CENTERS.xyz -o IONS+CENTERS_sorted.xyz 
    cd ../../
done;
```
これでIONS+CENTERS_sorted.xyzというファイルが各ディレクトリに生成される．あとはこれをcatで一つのファイルにまとめる．

```
#!/bin/bash

mkdir outputdata

current_dir=`dirname $0`
echo $current_dir

cd outputdata

# 一旦lnで一つのディレクトリにまとめる（やらなくても大丈夫）
for i in {0..10000};
# for i in {0..1};
do
    dir=bulkjob/struc_$i
    ln -s ../$dir/IONS+CENTERS_sorted.xyz IONS+CENTERS_$i.xyz
done

# catで一つにまとめる
for i in {0..10000};
# for i in {0..1};
do
    dir=bulkjob/struc_$i
	cat $dir/IONS+CENTERS_$i.xyz >> IONS+CENTERS_merge.xyz
done
```

書いていてちょっと思いついたが，ひょっとすると先にcatで一つにまとめてからCPextract sortをやった方が計算が早いかも．．．
最後にCPextract addlatticeで格子情報を付与して後処理は終了する．

```
CPextract.py cpmd addlattice -i IONS+CENTERS_merge.xyz -o IONS+CENTERS_merge_cell.xyz -s bomd-oneshot.out
```

これも下手すると1時間くらいかかるのでプリポストノードへ投げた方が良い．全体的に後処理にもpythonを使っているせいか処理が遅いのは今後の課題．



## obabelのコマンドたち

```
# rdkitでの読み込み用にmolファイルを作成する
obabel -i gro input_GMX.gro -o mol > input_GMX.mol
```

