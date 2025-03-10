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

import torch
import os

class variables_general:
    """Input variables for model section.
    
    This class contains all variables related to model specifications 
    that the user can specify in the input `yaml` file. 
    
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
        
        seed (int): The random seed for initializing model parameters.
        
        hidden_layers_enet (list[int]): The number of neurons used in the embedding network. Default is [50,50]
        
        hidden_layers_fnet (list[int]): The number of neurons used in the fitting network. Default is [50,50]
    
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
        self.bondfilename:str = yml["general"]["bondfilename"]
        self.savedir:str      = yml["general"]["savedir"]
        try:
            self.temperature:float = float(yml["general"]["temperature"])
        except:
            print(" temperature is not set. Use default value :: 300.")
            self.temperature:float = 300.0
        try:
            self.timestep:float = float(yml["general"]["timestep"])
        except:
            print(" timestep is not set. Use default value :: 0.484.")
            self.timestep:float = 0.484
        try:    
            self.save_bonddipole:int = int(yml["general"]["save_bonddipole"])
        except:
            print(" save_bonddipole is not set. Use default value :: 1(True).")
            self.save_bonddipole:int = 1        
        
        # Validate the values
        self._validate_values()
    def _validate_values(self):
        if self.timestep < 0:
            raise ValueError("ERROR :: timestep must be larger than 0 (in input)")
        if self.temperature < 0:
            raise ValueError("ERROR :: temperature must be larger than 0 (in input)")
        if self.save_bonddipole not in [0,1]:
            raise ValueError("ERROR :: save_bonddipole must be 0 or 1")
        



class variables_descriptor:
    """Input variables for training/validation data section.
    
    This class contains all variables related to model specifications 
    that the user can specify in the input `yaml` file. 
    
    Attributes:
        type (str): The type of input data. 
            The value should be `xyz`. 
        
        
        file_list (int): The list of `xyz` files containing both atomic and WC coordinates.
        
        
        itp_file (int): The `mol` file of a molecular structure.
        
        
        bond_name (int): The bond name to train. 
            The value should be one of "CH", "OH","CO","CC","O".
    
    """
    def __init__(self,yml:dict) -> None:
        
        #     parse_required_argment(node, "calc", this->calc);
        #     parse_required_argment(node, "directory", this->directory);
        #     parse_required_argment(node, "savedir", this->savedir);
        #     parse_required_argment(node, "xyzfilename", this->xyzfilename);
        #     this->desctype     = parse_optional_argment(node, "desctype", "allinone"); // old or allinone
        #     this->IF_COC       = stoi(parse_optional_argment(node, "IF_COC", "0")); // 0=False
        #     this->IF_GAS       = stoi(parse_optional_argment(node, "IF_GAS", "0")); // 0=False
        
        # parse yaml files1: model
        self.calc:str           = int(yml["descriptor"]["calc"])
        self.directory:str      = yml["descriptor"]["directory"]
        self.xyzfilename:str    = yml["descriptor"]["xyzfilename"]
        self.Rcs:float          = float(yml["descriptor"]["Rcs"])
        self.Rc:float           = float(yml["descriptor"]["Rc"])
        try:
            self.desctype:str       = yml["descriptor"]["desctype"]
        except:
            print(" desctype is not set. Use default value :: allinone.")
            self.desctype:str = "allinone"
        try:
            self.IF_COC:int         = int(yml["descriptor"]["IF_COC"])
        except:
            print(" IF_COC is not set. Use default value :: 0 (False).")
            self.IF_COC:int         = 0
        try:
            self.IF_GAS:int         = int(yml["descriptor"]["IF_GAS"])
        except:
            print(" IF_GAS is not set. Use default value :: 0 (False).")
            self.IF_GAS:int         = 0

        # Validate the values
        self._validate_values()
    
    def _validate_values(self):
        if self.calc not in [0,1]:
            raise ValueError("ERROR :: calc should be 0 or 1")
        if os.path.isdir(self.directory) == False:
            raise ValueError("ERROR :: directory does not exist.")
        if os.path.isfile(self.directory+"/"+self.xyzfilename) == False:
            raise ValueError("ERROR :: xyzfilename does not exist.")
        if self.desctype not in ["allinone","old"]:
            raise ValueError("ERROR :: desctype should be allinone or old")
        if self.IF_COC not in [0,1]:
            raise ValueError("ERROR :: IF_COC should be 0 or 1")
        if self.IF_GAS not in [0,1]:
            raise ValueError("ERROR :: IF_GAS should be 0 or 1")

class variables_predict:
    """Input variables for training/validation data section.
    
    This class contains all variables related to model specifications 
    that the user can specify in the input `yaml` file. 
    
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
        # parse_required_argment(node, "calc", this->calc);
        # parse_required_argment(node, "model_dir",this->model_dir);
        self.calc = int(yml["predict"]["calc"])
        self.model_dir = yml["predict"]["model_dir"]
        self.device = yml["predict"]["device"]

        # Validate the values
        self._validate_values()
    
    def _validate_values(self):
        if self.calc not in [0,1]:
            raise ValueError("ERROR :: calc should be 0 or 1")
        if os.path.isdir(self.model_dir) == False:
            raise ValueError("ERROR :: model_dir does not exist.")
        if self.device not in ["cpu","cuda","mps"]:
            raise ValueError("ERROR :: device should be cpu, cuda or mps")
            
        