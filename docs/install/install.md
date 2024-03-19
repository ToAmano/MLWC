# Installation

The package is composed of three part

- command line interface to process data (written in python)
- module to train ML models (written in python)
- module to infer dipole moment using ML models (written in C++)

You can install the first two via `pip`, while we need cmake to install the last one. 

## requirement for the C++ code





## Download

You can download the whole package via git

```bash
  git clone https://github.com/ttadano/alamode.git
  cd alamode
  git checkout develop
```


## Install python packages

One may create a vertual environment through conda 

```
conda create -n your_env python==3.10
conda activate your_env
conda install pip
pip install --upgrade pip
```

Goint to the root directory of the package, you can install the package by

```
cd dieltools
pip install .
```

## Install C++ packages

```
  % mkdir _build; cd _build
  % cmake -DUSE_MKL_FFT=yes -DSPGLIB_ROOT=${SPGLIB_ROOT} \ 
    -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DCMAKE_CXX_FLAGS="-O2 -xHOST" ..
```


