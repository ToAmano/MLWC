# Package name

This package implements the Allegro E(3)-equivariant machine-learning interatomic potential (https://arxiv.org/abs/2204.05249).

![Allegro logo](./logo.png)

In particular, `allegro` implements the Allegro model as an **extension package** to the [NequIP package](https://github.com/mir-group/nequip).


## Installation
**Please note that this package CANNOT be installed from PyPI as `pip install allegro`.**

`allegro` requires the `nequip` package and its dependencies; please see the [NequIP installation instructions](https://github.com/mir-group/nequip#installation) for details.

Once `nequip` is installed, you can install `allegro` from source by running:
```bash
git clone --depth 1 https://github.com/mir-group/allegro.git
cd allegro
pip install .
```

## Tutorial
The best way to learn how to use Allegro is through the [Colab Tutorial](https://colab.research.google.com/drive/1yq2UwnET4loJYg_Fptt9kpklVaZvoHnq). This will run entirely on Google's cloud virtual machine, you do not need to install or run anything locally.

## Usage
Allegro models are trained, evaluated, deployed, etc. identically to NequIP models using the `nequip-*` commands. See the [NequIP README](https://github.com/mir-group/nequip#usage) for details.

The key difference between using an Allegro and NequIP model is in the options used to define the model. We provide two Allegro config files analogous to those in `nequip`:
 - [`configs/minimal.yaml`](`configs/minimal.yaml`): A minimal example of training a toy model on force data.
 - [`configs/example.yaml`](`configs/example.yaml`): Training a more realistic model on forces and energies. **Start here for real models!**

The key option that tells `nequip` to build an Allegro model is the `model_builders` option, which we set to:
```yaml
model_builders:
 - allegro.model.Allegro
 # the typical model builders from `nequip` are still used to wrap the core Allegro energy model:
 - PerSpeciesRescale
 - ForceOutput
 - RescaleEnergyEtc
```

## LAMMPS Integration

We offer a LAMMPS plugin [`pair_allegro`](https://github.com/mir-group/pair_allegro) to use Allegro models in LAMMPS simulations, including support for Kokkos acceleration and MPI and parallel simulations. Please see the [`pair_allegro`](https://github.com/mir-group/pair_allegro) repository for more details.

## References and citing

The Allegro model and the theory behind it is described in our pre-print:

> *Learning Local Equivariant Representations for Large-Scale Atomistic Dynamics* <br/>
> Albert Musaelian, Simon Batzner, Anders Johansson, Lixin Sun, Cameron J. Owen, Mordechai Kornbluth, Boris Kozinsky <br/>
> https://arxiv.org/abs/2204.05249 <br/>
> https://doi.org/10.48550/arXiv.2204.05249

The implementation of Allegro is built on NequIP [1], our framework for E(3)-equivariant interatomic potentials, and e3nn, [2] a general framework for building E(3)-equivariant neural networks. If you use this repository in your work, please consider citing the NequIP code [1] and e3nn [3] as well:

 1. https://github.com/mir-group/nequip
 2. https://e3nn.org
 3. https://doi.org/10.5281/zenodo.3724963

## Contact, questions, and contributing

If you have questions, please don't hesitate to reach out to batzner[at]g[dot]harvard[dot]edu and albym[at]seas[dot]harvard[dot]edu.

If you find a bug or have a proposal for a feature, please post it in the [Issues](https://github.com/mir-group/allegro/issues).
If you have a question, topic, or issue that isn't obviously one of those, try our [GitHub Disucssions](https://github.com/mir-group/allegro/discussions).

**If your post is related to the general NequIP framework/package, please post in the issues/discussion on [that repository](https://github.com/mir-group/nequip).** Discussions on this repository should be specific to the `allegro` package and Allegro model.

If you want to contribute to the code, please read [`CONTRIBUTING.md`](https://github.com/mir-group/nequip/blob/main/CONTRIBUTING.md) from the `nequip` repository; this repository follows all the same processes.
=======
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

```bash
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

```bash
git config --global merge.ff false
```

## ohtakaでのコンパイル

```bash
(base) [k015124@ohtaka1 build]$ module purge
(base) [k015124@ohtaka1 build]$ module load oneapi_mkl
(base) [k015124@ohtaka1 build]$ module load oneapi_mpi
(base) [k015124@ohtaka1 build]$ module load gcc
(base) [k015124@ohtaka1 build]$ module list
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
=======
oneapi_compilerを利用すると動かないというバグがあるので，gccを利用すること！！

