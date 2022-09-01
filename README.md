# Tools

ここのディレクトリに簡単に使えるコードの類を追加していく予定．コマンドラインツールと，pythonのパッケージがある．


## installation

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


## pythonパッケージdieltool v 0.0.1

cp.xの後処理を行うpythonパッケージ．cpmdとquadrupoleに分かれていて現在機能を色々追加中．具体的な使い方については01_Si/exampleにあるnotebookを参照．現状で仕様が大体固まっているのはcp.xの結果をaseに読み込むところまででその先のdipole計算部分についてはどうするべきか考え中．


## CLIツール

### show_traj.sh v.0.0.1

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


### CPextract.py v0.0.1

CP.x計算の出力を解析するための単純なコード．現状.evpファイルを読み込んでエネルギー[eV]と温度[T]をグラフにする仕様になっている．

```bash 
# 出力はtest.evp_T.pdfとtest.evp_E.pdf
CPextract.py test.evp
```

Todo
- コマンド名をcpmdに統一して，例えば
  ```bash
  cpmd extract test.evp
  ```
  みたいな形にしたい．こうすれば機能が整理できる．
- 現在実行中のrunの現在のステップ数を出力できると嬉しい．
 


### nose_mass.py v.0.0.1

VASP用に推奨されるNose質量を計算するコード．詳細についてはShuichi Nosé, J. Chem. Phys., 81, 511(1984)参照．CP.xの場合は最初から振動数を与えるようになっているので，これが系の特徴的な振動数の程度になるようにすれば良い．例えば結晶ならフォノン振動数くらい．


> 時間で書きますが古典MDだと普通は数psのオーダ．
> どんなタイミングで温度制御をかけたいかですが、あまり熱浴からの干渉をさせたくなければ長めに逆ならば短めに．
> Gromacsだと1psを良く使いますが、慎重にやりたいときは5～10ps．


### make_alm.py v.0.0.1

pw.inまたはSPOCARファイルから，alamode alm用のインプットを作成する．入力ファイルはフォーマットを-fで指定することもできるが，ファイル名がPOSCARで終わる場合はVASP，scf.inまたはpw.inで終わる場合はQEと自動判定して処理を行うのでなくても良い．

```bash 
# 入力ファイルと出力ファイルを指定する
make_alm.py SPOSCAR alm.in
```
