# 01_Si_Crystal

## 19_Si_CPMD

Siのprimitive cell(Si原子8個)でのCPMDでの計算．
最初に格子定数の緩和を行なっており，得られた格子定数を全ての計算で利用している．


## 
| filename        | calclation type | description         |
|--|--|
| si_traj.xyz        | cp |cppp.x output xyz          |
| si_traj_refine.*   | cp |ase output by show_traj.sh |
| si_wan_traj.xyz    | cp-wf   | cppp.x output xyz          |
| si_wan_traj_refine.*| cp-wf  | ase output by show_traj.sh |

