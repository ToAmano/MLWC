

## system

- methanol 256 molecules
- dt: 10fs
- length: 10000 steps (100ps)
- high frequency dielectric constant: 1.7689 from experiment

```bash
CPextract.py diel spectra -F total_dipole.txt -E 1.7689 -s 1 -w 1 --fft True
```
