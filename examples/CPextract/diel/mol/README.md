

## system

- methanol 256 molecules
- dt: 10fs
- length: 10000 steps (100ps)
- high frequency dielectric constant: 1.7689 from experiment

## Download

download data from:

https://github.com/ToAmano/MLWC_sample_traj/blob/main/traj/methanol/liquid/molecule_dipole_small.txt


```bash
CPextract.py diel mol -F molecule_dipole_small.txt -E 1.7689 -s 1 -w 1
```
