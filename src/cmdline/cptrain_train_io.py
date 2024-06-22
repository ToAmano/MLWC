# -*- coding: utf-8 -*-
"""Input variables for CPtrain.py train command

Overview
------------------------


This module defines all the input parameters for ``CPtrain.py train`` comamnd, 
which performs the ML bond dipole model optimization.

Format of input files
------------------------

The input file is given in the ``yaml`` format, consisting of three main sections
defined as ``model``, ``data``, and ``traininig``. A simple example is as follows.

literal blocks::

    model:
        modelname: model_ch  # specify name
    
    data:
        type: xyz
        itp_file: methanol.mol

    traininig:
        device:     cpu # Torch device (cpu/mps/cuda)

The ``yaml`` format includes three major component: ``hash``, ``array``, and ``nest``.
Most variables are given in ``hash`` in the input, while ``array`` and ``nest`` are sometimes used.
We will state which format should be used for each variable.


demonstrates documentation as specified by the `Google Python
Style Guide`_. Docstrings may extend over multiple lines. Sections are created
with a section header and a colon followed by a block of indented text.

Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python example_google.py

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.


Todo:
     * For module TODOs
     * You have to also use ``sphinx.ext.todo`` extension

.. _Google Python Style Guide:
    http://google.github.io/styleguide/pyguide.html
"""

class variables_model:
    """Input variables for model section.
    
    This class contains all variables related to model specifications 
    that the user can specify in the input `yaml` file. 
    
    This module demonstrates documentation as specified by the `Google Python
    Style Guide`_. Docstrings may extend over multiple lines. Sections are created
    with a section header and a colon followed by a block of indented text.
    
    Attributes:
        modelname (str): Module level variables may be documented in 
            either the ``Attributes`` section of the module docstring, or in an
            inline docstring immediately following the variable.
    
        nfeature (int): The length of a single atomic descriptor. 
            Since the descriptor consists of C, H, and O atoms and is 
            represented by a 4-dimensional vector per atom, 
            the nfeature must be a multiple of 12. 
            In the case of a liquid, nfeature = 288 is sufficient with 24 atoms each.
    
        M (int): The size of the feature matrix. 
        
        Mb (int): The size of the feature matrix. 
    
    Raises:
        ValueError: From the theory, Mb must be smaller than M.
    """
    
    def __init__(self,yml:dict) -> None:
        """Example of docstring on the __init__ method.

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            yml (dict): Description of `param1`.
            
        .. automethod:: _evaporate
        """
        # parse yaml files1: model
        self.modelname:str = yml["model"]["modelname"]
        self.nfeature:int  = int(yml["model"]["nfeature"])
        self.M:int         = int(yml["model"]["M"])
        self.Mb:int        = int(yml["model"]["Mb"])
        # Validate the values
        self._validate_values()
    def _validate_values(self):
        if self.Mb > self.M:
            raise ValueError("ERROR :: Mb must be smaller than M")



class variables_data:
    """Input variables for training/validation data section.
    
    This class contains all variables related to model specifications 
    that the user can specify in the input `yaml` file. 
    
    This module demonstrates documentation as specified by the `Google Python
    Style Guide`_. Docstrings may extend over multiple lines. Sections are created
    with a section header and a colon followed by a block of indented text.
    
    Attributes:
        type (str): The type of input data. 
            The value should be `xyz`. 
        
        
        file_list (int): The list of `xyz` files containing both atomic and WC coordinates.
        
        
        itp_file (int): The `mol` file of a molecular structure.
        
        
        bond_name (int): The bond name to train. 
            The value should be one of "CH", "OH","CO","CC","O".
    
    """
    def __init__(self,yml:dict) -> None:
        # parse yaml files1: model
        self.type            = yml["data"]["type"]
        self.file_list:list  = yml["data"]["file"]
        self.itp_file:str    = yml["data"]["itp_file"]
        self.bond_name:str   = yml["data"]["bond_name"]
        # Validate the values
        self._validate_values()
    
    def _validate_values(self):
        if self.bond_name not in ["CH", "OH","CO","CC","O"]:
            raise ValueError("ERROR :: bond_name should be CH,OH,CO,CC or O")
        if self.type not in ["xyz"]:
            raise ValueError("ERROR :: type should be xyz")
            


class variables_training:
    """Input variables for training/validation data section.
    
    This class contains all variables related to model specifications 
    that the user can specify in the input `yaml` file. 
    
    This module demonstrates documentation as specified by the `Google Python
    Style Guide`_. Docstrings may extend over multiple lines. Sections are created
    with a section header and a colon followed by a block of indented text.
    
    Attributes:
        device (str): Module level variables may be documented in 
            either the ``Attributes`` section of the module docstring, or in an
            inline docstring immediately following the variable.
    
        batch_size (int): The length of a single atomic descriptor. 
            Since the descriptor consists of C, H, and O atoms and is 
            represented by a 4-dimensional vector per atom, 
            the nfeature must be a multiple of 12. 
            In the case of a liquid, nfeature = 288 is sufficient with 24 atoms each.
    
        validation_batch_size (int): The size of the feature matrix. 
        
        max_epochs (int): The maximum number of epochs. 

        learning_rate (int): The starting learning rate. We recommend `0.01`.

        n_train (int): The number of training data (the number of frame). 
            Therefore, if the number of atoms in the structure of the training data 
            is large, the `n_train` can be reduced.

        n_val (int): The number of validation data (the number of frame).

        modeldir (str): The directory to which the model files will be saved.

        restart (bool): If true, restart training from previous parameters.
    """
    def __init__(self,yml:dict) -> None:        
        # parse yaml 2: training
        self.device                     = yml["training"]["device"]   # Torchのdevice
        self.batch_size:int             = int(yml["training"]["batch_size"])  # 訓練のバッチサイズ
        self.validation_batch_size:int  = int(yml["training"]["validation_batch_size"]) # validationのバッチサイズ
        self.max_epochs:int             = int(yml["training"]["max_epochs"])
        self.learning_rate:float        = float(yml["training"]["learning_rate"]) # starting learning rate
        self.n_train:int                = int(yml["training"]["n_train"]) # データ数（xyzのフレーム数ではないので注意．純粋なデータ数）
        self.n_val:int                  = int(yml["training"]["n_val"])
        self.modeldir:str               = yml["training"]["modeldir"]
        self.restart                    = yml["training"]["restart"]
        