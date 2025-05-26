# example 

This example shows how to calculate the distance correlation function.


# system
- PG
- 

# download 

download trajectory data from 

https://github.com/ToAmano/MLWC_sample_traj/blob/main/traj/pg/liquid/mol_wan_500frames.xyz


```bash
CPextract.py cpmd distanceft -l 0 5 --strategy distance --numatom 39 -t 2.5 -F mol_wan_500frames.xyz -m PG.mol
```
