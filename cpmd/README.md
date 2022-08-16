# CPMD

## CPMDでの時間の扱いについて

CPMDでは時間の単位をハートリーでとっており，これは
$$
1 a.u.=2.4189 * 10^-17 s = 2.4189 * 10^-2 [fs]
$$
である． サンプルなんかをみていると大体タイムステップをdt=5[a.u.]=0.121 [fs] 程度にとっていることが多い． これは通常のAIMDの1,2[fs]に比べて小さいが，CPMDでは各ステップで計算を収束させない代償として短い時間ステップが必要である．

この時間ステップで1psの計算を行うと
$$
 1[ps]/0.121[fs]=1000/0.121~8300
$$
ステップ必要である．もし10psなら8.3万ステップ必要．

出力するトラジェクトリは10ステップごと，1.21[fs]ごとくらいあればTHzの計算では十分である． これだと1psの計算で830構造を得ることになる．

## Nose質量について


## visualize trajectories

可視化にあたっては，ファイルを`ase`で読み込むか`mdtraj`で読み込むかという選択肢がある．可視化するためのエンジンとしては`nglview`を利用できるが，`ase`自体もシンプルな可視化手法を備えている（これはあまりおすすめしない）．基本的には`nglview`で可視化する手法に一本化しておくとコードの整理が楽かと思う．ちなみにnglviewで可視化可能なファイルの一覧については[公式ページ](http://nglviewer.org/nglview/latest/#usage)を参照のこと．

`ase`と`mdtraj`の違いについては，`mdtraj`の場合にはトポロジーファイルも必要であるというのが大きな違い．VASPとCP.xの出力については現状`ase`のほうが扱いやすいかなという気がする．`ase`のファイルを可視化するための基本的な手続きは以下の通り．

```python
# aseで簡単な可視化の場合の場合
# Atomsオブジェクトのリストをトラジェクトリとして可視化できる．
test=ase.io.read("filename", index=":") #これでAtomsのリストとして読み込む.
for i in range(len(test)):
    test[i].set_cell([a,b,c]) #格子の情報を追加
　　 test[i].pbc=(True,True,True) #格子を表示するために必須
from ase.visualize import view
view(test,viewer='ngl')

# nglviewの場合
# aseのatomsではなくtrajectoryとして読み込む必要がある．
test=ase.io.read("filename", index=":")
ase.io.write("test.traj",test) #一旦traj形式で書き出し
traj = ase.io.Trajectory("test.traj") #再度読み込み
view=nv.show_asetraj(traj)
view.parameters =dict(
                        camera_type="orthographic",
                        backgraound_color="black",
                        clip_dist=0
)
view.clear_representations()
view.add_representation("ball+stick")
view.add_unitcell() # unitcellの表示
view.update_unitcell()
view
```
<!-- https://osscar-docs.readthedocs.io/en/latest/visualizer/nglview_basic.html?highlight=show_ase -->
<!-- https://osscar-docs.readthedocs.io/en/latest/visualizer/nglview_advanced.html?highlight=show_ase -->


`ase`と`mdtraj`の行き来はそんなに単純ではない．以下現状の問題点とか注意点について．

1. aseで格子定数の情報を組み込んだxyzに変換しても（これをextended xyzとよぶ）mdtrajでは認識してくれない．

    mdtrajではxyzを読み込んだ後，別途格子定数の情報を追加した新しいmdtraj.trajectoryを作る必要がある．

1. aseにはatomsとtrajectoryがある．trajectoryはatomsの集合．nglviewで扱うにはtrajectoryにする必要がある．しかしながら，`.traj`ファイルはxyzなどと違って人間には読めないファイルである．

1. VASP XDATCAR (V.5.4.4)

    XDATCARファイルはaseで普通に読み込むことができる．また，VMDというソフトウェアで可視化することができる．

1. CP.x xyz
    posファイルをcppp.xでxyzファイルに変換する．xyzファイルは複数形式があるようで，例えばaseで読み込んでも格子定数は読み込んでくれない． 将来的にはcppp.x自体をいじってもっと使いやすくしたほうが良いと思うが，とりあえずは全部pythonで処理するようにしてある．

    この出力では周期境界条件が課されており，原子が格子の外側にはみ出すことはない．従って結晶を扱う時には別途周期境界条件を外す必要がある．gromacsだとtrajconvを利用して周期境界条件周りをかなり自由に操作することができるので，将来的には一回trajconvで扱える形に変換できるとベストだと思う．

## calculations with trajectories

トラジェクトリを読み込んで計算に活用する場合．
