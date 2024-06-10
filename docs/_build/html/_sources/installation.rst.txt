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

    git clone git@github.com:dirac6582/dieltools.git 
    cd dieltools
    git checkout develop

Please be sure to use `develop` branch. we define the ``root_dir`` as the root directory as 

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
* cmake >= 3.0 (https://cmake.org/download/)
* c++ compiler
* openMP

Among them, ``libtorch`` should be automatically installed in the previous section with ``pip``. Although you can build it from the source alternatively, we will use ``libtorch`` installed via ``pip`` below.


Check libtorch 
----------------------------------------

If you successfully installed ``pytorch`` via ``pip`` under the virtual environment provided by ``conda``, it is instaled to something like

.. code-block:: bash

    ls /path/to/your/conda/virtual/environment/lib/python3.10/site-packages/torch/

The exact path can be checked by executing the following ``python`` command.

.. code-block:: bash

    from distutils.sysconfig import get_python_lib
    print(get_python_lib())

``Libtorch`` libraries, headers, and ``CMake`` settings are in 

.. code-block:: bash

    pytorch_root=/path/to/your/conda/virtual/environment/lib/python3.10/site-packages/torch/

    # shared libraries
    ls ${pytorch_root}/lib

    # header files
    ls ${pytorch_root}/include

    # CMake settings
    ls ${pytorch_root}/share/cmake


Install Eigen
----------------------------------------

Eigen is a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms. It is a header-only library, so you only need to download and include the header files in your project. You can download the latest version of Eigen from this link. `this link <https://sphinx-users.jp/index.html>`

.. code-block:: bash


Install dieltools C++ packages
----------------------------------------

After preparing all the required packages, we can build dieltools C++ packages through ``cmake``. Now go to the source code directory and make `build` directory.

.. code-block:: bash

    cd ${root_dir}/src/cpp
    mkdir build
    cd build

Then, we may execute ``cmake`` like

.. code-block:: bash

    cmake ../ -DCMAKE_PREFIX_PATH=path/to/eigen -DCMAKE_PREFIX_PATH=path/to/libtorch

Please be sure to replace ``path/to/eigen`` and ``path/to/libtorch`` with the actual path to the ``Eigen`` and ``libtorch`` directories. 

If the CMake has been executed successfully, then run the following make commands to build the package:

.. code-block:: bash

    make 
    make install

If everything works fine, you will have the executable named ``dieltools`` in ``${root_dir}/src/src/cpp/build/``. If you run the executable without any arguments, you will see the following message.

.. code-block:: bash

    $ ${root_dir}/notebook/c++/src/build/dieltools
     +-----------------------------------------------------------------+
     +                         Program dieltools                       +
     +-----------------------------------------------------------------+
         PROGRAM DIELTOOLS STARTED AT = Thu Jan  1 09:00:00 1970


     ERROR in main  MESSAGE: Error: incorrect inputs. Usage:: dieltools inpfile

 
