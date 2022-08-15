# Tools

ここのディレクトリに簡単に使えるコードの類を追加していく予定．基本的にはこのディレクトリに

```bash
export PATH="/path/to/tools/:$PATH"
```

とパスを通すことでコマンドとして使える用になる．

## show_traj.sh

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
1. xyzファイルでPBCをかけない場合，現状では正方晶のみ対応している．
1. モジュールとしてnumpy,ase,nglviewを利用する．

