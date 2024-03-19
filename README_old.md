# dieltools

ここのディレクトリに簡単に使えるコードの類を追加していく予定．コマンドラインツールと，pythonのパッケージがある．

:::note warn
2023/5/8 We divided the directory into the new repository named dieltools for accerelating development in VScode
:::



## 必要なソフト

コマンドラインで必要なものたち．
```
openbabel

packmol
```

## installation

下準備として仮想環境を作成し，必要なパッケージをインストール．

```bash
# 仮想環境
$ conda create -n dieltools python=3.10
$ conda activate dieltools

# notebookのためのパッケージ
$ conda install jupyter

# 必要なパッケージをconda経由でインストール
$ conda install -c conda-forge ase=3.22
$ conda install -c conda-forge nglview=3.0

# その他使うかもしれないパッケージ（gromacs_wrap.py用）
$ conda install -c bioconda gromacs
$ conda install -c rdkit rdkit
$ conda install -c conda-forge mdanalysis
$ conda install -c conda-forge packmol # (これはコマンドライン)
```

注意：富岳でpyenv+pipを使う場合，packmolはpipでは入れられない．そこでpyenvでanacondaを入れてcondaを使おう．intel64版のanacondaも今の所問題なく動いている．

```
spack load packmol
```

`gromacs_wrap`を使う場合はopenbabelとacpypeが必要．m1mac以外ならcondaから入れられる．

```
# acpype via conda
conda install -c conda-forge acpype
```

同様に`mdapackmol`も必要．これはgithubからマニュアルインストールするしかない．
従って，将来的にはmdapackmolは使わないようにしたい．
```
https://github.com/MDAnalysis/MDAPackmol/blob/master/docs/installation.rst
cd ~/src/MDAPackmol/
python setup.py install
```


## installation2 

手でマニュアルで色々インストールする他に，環境をcondaで自動作成する方法もある．本当はpipでやるのが良いのかもしれないが，現状自分がcondaで環境構築を行っているのでとりあえずそれで．
[ここ](https://qiita.com/yubessy/items/2dd43551aa8308dc7eca)を参考にした．dieltools.yamlファイルに入れるべきパッケージ一覧がはいっているので，それを使う．

```
conda env create --file dieltools.yaml
```

追記1::どうもこの方法だとrdkitとpytorchが干渉してしまう気がする．調査を継続．
追記2::ohtakaで使う場合，joblibのversionを1.2.0で使う．1.1系だと計算が遅くなる問題がある．



### pytorchについて

platformによってインストールのコマンドが異なる．

```
# m1 mac
conda install pytorch::pytorch torchvision torchaudio -c pytorch

# linux
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```


### simple installation
基本的にはこのディレクトリに

```bash
export PATH="/path/to/tools/:$PATH"
export PYTHONPATH="/path/to/tools/:$PYTHONPATH"

```
とパスを通すことでコマンドとして使える用になる．


### pip installation(推奨)
toolsディレクトリにいってpip installを実行することでインストールできる．eオプションは変更があった時に自動で変更を反映してくれる．

```bash
cd /path/to/tools
pip install -e .
```


## pythonパッケージdieltool v 0.0.2

cp.xの後処理を行うpythonパッケージ．cpmdとquadrupoleに分かれていて現在機能を色々追加中．具体的な使い方については01_Si/exampleにあるnotebookを参照．現状で仕様が大体固まっているのはcp.xの結果をaseに読み込むところまででその先のdipole計算部分についてはどうするべきか考え中．


## CLIツール

### show_traj.sh v.0.0.2

このコマンドはcppp.xの作るxyzファイル，またはVASPの作るXDATCARファイルをnglviewで可視化するためのコード．コマンドを打つと自動的にコードが書かれた状態のjupyter notebookが起動する．モジュールとして`show_traj`以下のコードを利用する．使い方はトラジェクトリのファイル名を引数として利用．ファイル名の末尾が`XDATCAR`か`.xyz`かでファイル形式を自動判定しているので，それ以外の形式のファイル名は受け付けない．また，`.xyz`の2行目から格子ベクトルの9個の数字を読み取っているのでこれに反する形式のxyzファイルは読み込めない．

```bash
# XDATCARの場合
show_traj.sh XDATCAR

# .xyzの場合
show_traj.sh test.xyz

```

いくつか現状の注意点

1. 中間ファイルとして`show_XDATCAR_tmp.py`または`show_CP_tmp.py`を作成する．
1. xyzファイルの場合，新しく格子定数の情報を含んだ`filename_refine.xyz`と`filename_refine.traj`を作成する．これらはaseから読み込めるのでさらなる解析には便利と思う．
1. XDATCARファイルの場合，新しくaseから読み込めるXDATCAR.trajファイルを作成する．
1. xyzファイルでPBCをかけない場合，現状では正方晶のみ対応している．
1. モジュールとしてnumpy,ase,nglviewを利用する．


### CPextract.py v0.0.2

CP.xとCPMD.x計算の出力を解析するための単純なコード．

できること．

- evpファイルを読み込んでエネルギー[eV]と温度[T]をグラフにする
- 

```bash 
# 出力はtest.evp_T.pdfとtest.evp_E.pdf
CPextract.py cp energy
```

Todo
- 現在実行中のrunの現在のステップ数を出力できると嬉しい．
 
### make_itp.py v.0.0.1
SMILESを含むcsvファイルから，gromacs用のinputを作成する簡単なスクリプト．
```bash
make_itp.py input.csv
```
出力が`input.acpype`以下に作成される．


### nose_mass.py v.0.0.2

VASP用に推奨されるNose質量を計算するコード．詳細についてはShuichi Nosé, J. Chem. Phys., 81, 511(1984)参照．CP.xの場合は最初から振動数を与えるようになっているので，これが系の特徴的な振動数の程度になるようにすれば良い．例えば結晶ならフォノン振動数くらい．


> 時間で書きますが古典MDだと普通は数psのオーダ．
> どんなタイミングで温度制御をかけたいかですが、あまり熱浴からの干渉をさせたくなければ長めに逆ならば短めに．
> Gromacsだと1psを良く使いますが、慎重にやりたいときは5～10ps．


### make_alm.py v.0.0.2

pw.inまたはSPOCARファイルから，alamode alm用のインプットを作成する．入力ファイルはフォーマットを-fで指定することもできるが，ファイル名がPOSCARで終わる場合はVASP，scf.inまたはpw.inで終わる場合はQEと自動判定して処理を行うのでなくても良い．

```bash 
# 入力ファイルと出力ファイルを指定する
make_alm.py SPOSCAR alm.in
```


### gitでの開発方針

大きな変更をローカルで行い，これをohtaka/fugakuで実行してデバックを行うのが基本的な流れ．具体的な手順としては

- ローカルにfeature/#10_nameなどと名前のついたbranchを切る
- このブランチで作業を行う．
- ローカルでデバックまですませてからリモートへpush
- ohtaka/fugakuでこのブランチをpullしてデバック
- 問題なければdevelopへmergeする．

となる．

大きな変更を行っている最中に小さな変更がしたくなることもある．例えば全然関係ないところで簡単なprint文を表示したいとか．こういうときは，ローカルのfeatureブランチからcheckoutした新しいブランチを作成してそこに変更を行うのかと思っていたが，どうもこれだとgit graphに反映されなくなってしまう．これには理由があって，マージがfast-forwardで実行されるとコミットオブジェクトが作成されないからだ．そこで，mergeするときに--no-ffオプションを常につけるようにすれば良い．

```
$ git checkout develop
Switched to branch 'develop'
$ git merge --no-ff myfeature
Updating ea1b82a..05e9557
(Summary of changes)
$ git branch -d myfeature
Deleted branch myfeature (was 05e9557).
$ git push origin develop
```

毎度つけるのが面倒な場合はgit configに書き込んでしまう方法がある．

```
git config --global merge.ff false
```


## dieltools C++版のインストール

ohtakaの場合，CMakeLists.txtでfsc++の部分をコメントアウトして実行する必要がある．
この有無はコンパイラ依存かも？
また，CMakelists_ohtaka.txtを使わないとコンパイルできない．．．


モジュールはGCC+intel MKL+openmpiの組み合わせでビルドする．

```bash
module load oneapi_mkl
(test_dieltools) [k015124@ohtaka1 build]$ module load gcc
(test_dieltools) [k015124@ohtaka1 build]$ module load oneapi_mkl
Currently Loaded Modulefiles:
 1) oneapi_mkl/2023.0.0   2) gcc/10.1.0

 ```
