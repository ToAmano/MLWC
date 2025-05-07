

## system

- methanol 48 molecule 

# download

download data from 

https://github.com/ToAmano/MLWC_sample_traj/blob/main/traj/methanol/liquid/mol_wan_small.xyz

```bash
CPextract.py cpmd vdos -m H --numatom 17 -F mol_wan_test.xyz
CPextract.py cpmd vdos -m C --numatom 17 -F mol_wan_test.xyz
CPextract.py cpmd vdos -m O --numatom 17 -F mol_wan_test.xyz
CPextract.py cpmd vdos -m com --numatom 17 -F mol_wan_test.xyz
CPextract.py cpmd vdos --numatom 17 -F mol_wan_test.xyz
```
