=====================================================
Installation
=====================================================

In this tutorial, we start from descriptor files to train ML dipole models of isolated methanol. 



Installation overview
========================================

The package is composed of three part

- command line interface to process DFT/AIMD data (written in python)
- module to train ML models (written in python)
- module to infer dipole moment using ML models (written in C++)

You can install the first two via python package installer ``pip``, while we need ``cmake`` to install the last one.


Download
========================================

You can download the whole package via git

.. code-block:: bash

    git clone git@github.com:dirac6582/MLWC.git 
    cd MLWC
    git checkout develop

Please be sure to use ``develop`` branch. We define the ``root_dir`` as the root directory as 

.. code-block:: bash

    root_dir=`pwd`

for later convenience.


Install python packages
========================================

One may create a virtual environment through ``conda`` or ``virtualenv``. Here, we show how to create a virtual environment named ``your_env`` using ``conda``. Although we use ``conda`` for the virtual environment, we use ``pip`` for the package installation. 

.. code-block:: bash

    conda create -n your_env python==3.10
    conda activate your_env
    conda install pip
    pip install --upgrade pip

Goint to the root directory of the package, you can install the package using ``pip``.

.. code-block:: bash

    cd $root_dir
    pip install .

If the installation succeeds, you can execute various commands without additional path settings.

.. code-block:: bash

    CPextract.py --help
    CPtrain.py --help

These lines will print the help information.



Install C++ packages
========================================

Requirements
----------------------------------------

To install C++ packages, the following packages/commands are required.

* Eigen (https://eigen.tuxfamily.org/index.php?title=Main_Page)
* libtorch (https://pytorch.org/cppdocs/installing.html)
* RDKit (https://github.com/rdkit/rdkit)
* Boost (https://github.com/boostorg)
* cmake >= 3.27 (https://cmake.org/download/)
* c++ compiler 
* openMP library

``Boost`` is required by ``RDKit``. Among them, ``libtorch``, ``RDKit``, and ``Boost`` should be automatically installed in the previous section with ``pip``. Alternatively, you can build them from the source.


Check libtorch 
----------------------------------------

If you successfully installed ``pytorch`` via ``pip`` under the virtual environment provided by ``conda``, it is installed to 

.. code-block:: bash

    ls /path/to/your/conda/virtual/environment/lib/python3.10/site-packages/torch/

The directory depends on your python version. The exact path can be checked by executing the following ``python`` command.

.. code-block:: bash

    python -c "from distutils.sysconfig import get_python_lib;print(get_python_lib())"

``Libtorch`` libraries, headers, and ``CMake`` settings are in 

.. code-block:: bash

    # pytorch root directory (depends on your system)
    pytorch_root=${CONDA_PREFIX}/lib/python3.10/site-packages/torch/

    # shared libraries
    ls ${pytorch_root}/lib

    # header files
    ls ${pytorch_root}/include

    # CMake settings
    ls ${pytorch_root}/share/cmake


Basically, ``${CONDA_PREFIX}`` points to the root directory of the virtual environment. 


Install Eigen
----------------------------------------

Eigen is a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms. It is a header-only library, so you only need to download and include the header files in your project. You can download ``Eigen`` from ``gitlab`` as follows. 

.. code-block:: bash

    cd /path/to/where/you/want/to/install/eigen
    git clone --depth 1 https://gitlab.com/libeigen/eigen -b 3.4.0 eigen-3.4.0

Or you can download the tarball from the official website (https://eigen.tuxfamily.org/index.php?title=Main_Page).

.. code-block:: bash

    cd /path/to/where/you/want/to/install/eigen
    curl -O https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    tar xzf eigen-3.4.0.tar.gz


Install MLWC C++ packages
----------------------------------------

After preparing all the required packages, we can build MLWC C++ packages through ``cmake``. Now go to the source code directory and make `build` directory.

.. code-block:: bash

    cd ${root_dir}/src/cpp
    mkdir build
    cd build

Then, we may execute ``cmake`` like

.. code-block:: bash

    cmake ../ -DCMAKE_PREFIX_PATH="path/to/eigen;path/to/libtorch" -DCMAKE_MODULE_PATH=path/to/eigen/cmake -DBOOST_ROOT=${CONDA_PREFIX} -DBoost_NO_BOOST_CMAKE=ON -DBoost_NO_SYSTEM_PATHS=ON

Please be sure to replace ``path/to/eigen`` and ``path/to/libtorch`` with the actual path to the ``Eigen`` and ``libtorch`` directories. We have to quote your path list with ``"`` if using multiple paths. If you use libtorch in ``conda`` environment, ``/path/to/libtorch`` is ``pytorch_root`` defined above.
We also need to specify the `CMAKE_MODULE_PATH` to the Eigen3 cmake directory to activate the Module mode in cmake, because we did not build Eigen3. 

If the CMake has been executed successfully, then run the following make commands to build the package:

.. code-block:: bash

    make 

If everything works fine, you will have the executable named ``MLWC`` in ``${root_dir}/src/src/cpp/build/``. when you run the executable without any argument, you will see the following message.

.. code-block:: bash

    $ ${root_dir}/notebook/c++/src/build/MLWC
     +-----------------------------------------------------------------+
     +                         Program MLWC                       +
     +-----------------------------------------------------------------+
         PROGRAM MLWC STARTED AT = Thu Jan  1 09:00:00 1970


     ERROR in main  MESSAGE: Error: incorrect inputs. Usage:: MLWC inpfile

 
