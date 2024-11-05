# -*- coding: utf-8 -*-
from __future__ import annotations # fugaku上のpython3.8で型指定をする方法（https://future-architect.github.io/articles/20201223/）

import argparse
import sys
import numpy as np
import argparse
import sys
import os
import ase.io
import ase

import torch       # ライブラリ「PyTorch」のtorchパッケージをインポート
import torch.nn as nn  # 「ニューラルネットワーク」モジュールの別名定義
import torch.multiprocessing as mp

import argparse
from ase.io.trajectory import Trajectory
import ml.parse # my package
import ml.dataset.mldataset_xyz
import ml.model.mlmodel_basic

import sys
import numpy as np
import ml.model.mlmodel_basic
import cpmd.class_atoms_wan 
import cpmd.asign_wcs
import yaml
    
# 物理定数
from include.constants import constant
# Debye   = 3.33564e-30
# charge  = 1.602176634e-019
# ang      = 1.0e-10
coef    = constant.Ang*constant.Charge/constant.Debye

from cmdline.cptrain_pred import cptrain_pred_io
from ml.descriptor.descriptor_abstract import Descriptor
from ml.descriptor.descriptor_torch    import Descriptor_torch_bondcenter

from include.mlwc_logger import root_logger
logger = root_logger(__name__)

def _format_name_length(name, width):
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
    if len(name) <= width:
        return "{: >{}}".format(name, width)
    else:
        name = name[-(width - 3) :]
        name = "-- " + name
        return name


def load_model(model_filename:str,device:str)->torch.jit.ScriptModule:
    if device not in ["cpu","cuda","mps"]:
        raise ValueError("ERROR :: device should be cpu or cuda or mps")
    model = None
    if os.path.isfile(model_filename):
        model = torch.jit.load(model_filename)
        model = model.to(device) 
        logger.info(f"{model_filename} :: {model}")
        model.share_memory() #https://knto-h.hatenablog.com/entry/2018/05/22/130745
        # TODO :: hard code
        model.eval()
    else:
        logger.info(f"{model_filename} is not loaded")
    return model


def mlpred(yaml_filename:str)->None:

    # read input yaml file
    with open(yaml_filename) as file:
        yml = yaml.safe_load(file)
        print(yml)
    input_general    = cptrain_pred_io.variables_general(yml)
    input_descriptor = cptrain_pred_io.variables_descriptor(yml)
    input_predict    = cptrain_pred_io.variables_predict(yml)
    
    # save input file to output directory
    with open(input_general.savedir+'/input.yaml','w')as f:
        yaml.dump(yml, f, default_flow_style=False, allow_unicode=True)

    #from torchinfo import summary
    #summary(model=model_ring)

    # * load data (xyz or descriptor)
    logger.info(" -------------------------------------- ")
    # * itpデータの読み込み
    # note :: itpファイルは記述子からデータを読み込む場合は不要なのでコメントアウトしておく
    import ml.atomtype
    # 実際の読み込み
    import os
    if not os.path.isfile(input_general.bondfilename):
        logger.error(f"ERROR :: itp file {input_general.bondfilename} does not exist")
        raise FileNotFoundError(f"ERROR :: itp file {input_general.bondfilename} does not exist")
    if input_general.bondfilename.endswith(".itp"):
        itp_data=ml.atomtype.read_itp(input_general.bondfilename)
    elif input_general.bondfilename.endswith(".mol"):
        itp_data=ml.atomtype.read_mol(input_general.bondfilename)
    else:
        logger.error("ERROR :: itp_filename should end with .itp or .mol")
        raise ValueError("ERROR :: itp_filename should end with .itp or .mol")

    # TODO :: ここで変数を定義してるのはあまりよろしくない．
    NUM_MOL_ATOMS:int=itp_data.num_atoms_per_mol
    logger.info(f" The number of atoms in a single molecule :: {NUM_MOL_ATOMS}")
    # atomic_type=itp_data.atomic_type
    
    # * load trajectories
    logger.info(f" Loading xyz file :: {input_descriptor.xyzfilename}")
    # check atomic arrangement is consistent with itp/mol files
    tmp_atoms = ase.io.read(input_descriptor.xyzfilename,index="1")
    print(tmp_atoms.get_chemical_symbols()[:NUM_MOL_ATOMS])
    if tmp_atoms.get_chemical_symbols()[:NUM_MOL_ATOMS] != itp_data.atom_list:
        raise ValueError("configuration different for xyz and itp !!")
    
    atoms_traj:list[ase.Atoms] = ase.io.read(input_descriptor.xyzfilename,index=":")
    logger.info(" Finish loading xyz file...")
    logger.info(f" The number of trajectories are {len(atoms_traj)}")
    logger.info("") 
    # NUM_MOL:int = len(atoms_traj[0])/NUM_MOL_ATOMS
    NUM_MOL:int = 48
    logger.info(f" The number of molecules in a single frame :: {NUM_MOL}")
    # * load model
    model_ch = load_model(input_predict.model_dir+'/model_ch.pt',input_predict.device)
    model_co = load_model(input_predict.model_dir+'model_co.pt',input_predict.device)
    model_oh = load_model(input_predict.model_dir+'model_oh.pt',input_predict.device)
    model_cc = load_model(input_predict.model_dir+'model_cc.pt',input_predict.device)
    model_o  = load_model(input_predict.model_dir+'model_o.pt',input_predict.device)
    # below is for coh/coc bindings
    model_coc = load_model(input_predict.model_dir+'model_coc.pt',input_predict.device)
    model_coh = load_model(input_predict.model_dir+'model_coh.pt',input_predict.device)
    
    # * output necessary information to total_dipole.txt, molecule_dipole.txt, ch_dipole.txt etc.
    file_total_dipole = open(input_general.savedir+"/total_dipole.txt","w")
    file_mol_dipole   = open(input_general.savedir+"/molecule_dipole.txt","w")
    file_ch_dipole    = open(input_general.savedir+"/ch_dipole.txt","w")
    file_co_dipole    = open(input_general.savedir+"/co_dipole.txt","w")
    file_oh_dipole    = open(input_general.savedir+"/oh_dipole.txt","w")
    file_cc_dipole    = open(input_general.savedir+"/cc_dipole.txt","w")
    file_o_dipole     = open(input_general.savedir+"/o_dipole.txt","w")
    file_coc_dipole   = open(input_general.savedir+"/coc_dipole.txt","w")
    file_coh_dipole   = open(input_general.savedir+"/coh_dipole.txt","w")
    
    # total dipole
    file_total_dipole.write("# index dipole_x dipole_y dipole_z \n")
    file_mol_dipole.write(" frame_index molecule_dipole_x molecule_dipole_y molecule_dipole_z \n")
    for file in [file_total_dipole,file_mol_dipole]:
        file.write("#UNITCELL [Ang] ")
        for i in range(3):
            for j in range(3):
                file.write(str(tmp_atoms.get_cell()[i][j])+" ")
        file.write("\n")
        file.write("#TEMPERATURE [K] "+str(input_general.temperature)+"\n")
        file.write("#TIMESTEP [fs]  MOLECULE_DIPOLE [Debye] \n")
    # bond dipole
    if input_general.save_bonddipole:
        for file in [file_ch_dipole,file_co_dipole,file_oh_dipole,file_cc_dipole,file_o_dipole,file_coc_dipole,file_coh_dipole]:
            file.write("# frame_index bond_index dipole_x dipole_y dipole_z \n")
            file.write("#UNITCELL [Ang] ")
            for i in range(3):
                for j in range(3):
                    file.write(str(tmp_atoms.get_cell()[i][j])+" ")
            file.write("\n")
            file.write("#TEMPERATURE [K] "+str(input_general.temperature)+"\n")
            file.write("#TIMESTEP [fs]  MOLECULE_DIPOLE [Debye] \n")

    # method to calculate bond centers
    
    # ASSIGN=cpmd.asign_wcs.asign_wcs(NUM_MOL,NUM_MOL_ATOMS,atoms_traj[0].get_cell())
    DESC  = Descriptor(Descriptor_torch_bondcenter) # set strategy


    # * loop over trajectories
    for fr_index, fr_atoms in enumerate(atoms_traj):
        # split atoms and wan
        atoms_wan = cpmd.class_atoms_wan.atoms_wan(fr_atoms,NUM_MOL_ATOMS,itp_data)
        # calculate NUM_MOL
        NUM_MOL:int = atoms_wan.NUM_MOL
        # calculate bond centers
        results = atoms_wan.ASIGN.aseatom_to_mol_coord_bc(fr_atoms, itp_data, itp_data.bonds_list)
        list_mol_coords, list_bond_centers =results
        fr_atoms = atoms_wan.atoms_nowan # reset atoms
        # set list_mol_coors to fr_atoms
        fr_atoms.set_positions(np.array(list_mol_coords).reshape((-1,3)))
        # the total dipole of the frame
        sum_dipole=np.zeros(3)
        # initialize dipole
        # TODO :: This may slightly slow down the calculation
        y_pred_ch = np.zeros((NUM_MOL,3))
        y_pred_co = np.zeros((NUM_MOL,3))
        y_pred_oh = np.zeros((NUM_MOL,3))
        y_pred_cc = np.zeros((NUM_MOL,3))
        y_pred_o  = np.zeros((NUM_MOL,3))
        y_pred_coc= np.zeros((NUM_MOL,3))
        y_pred_coh= np.zeros((NUM_MOL,3))

        if len(itp_data.ch_bond_index) != 0 and model_ch  != None:
            # extract the coordinates of ch_bond
            bond_centers = np.array(list_bond_centers)[:,itp_data.ch_bond_index,:].reshape((-1,3))

            Descs_ch     = DESC.calc_descriptor(atoms=fr_atoms,
                                                bond_centers=bond_centers,
                                                list_atomic_number=[6,1,8], 
                                                list_maxat=[24,24,24], 
                                                Rcs=input_descriptor.Rcs, 
                                                Rc=input_descriptor.Rc, 
                                                device=input_predict.device)
            X_ch = torch.from_numpy(Descs_ch.astype(np.float32)).clone()
            y_pred_ch  = model_ch(X_ch.to(input_predict.device)).to("cpu").detach().numpy()   # 予測 (NUM_MOL*len(bond_index),3)
            y_pred_ch = y_pred_ch.reshape((-1,3)) # # !! ここは形としては(NUM_MOL*len(bond_index),3)となるが，予測だけする場合NUM_MOLの情報をgetできないのでreshape(-1,3)としてしまう．
            del Descs_ch                
            sum_dipole += np.sum(y_pred_ch,axis=0) #双極子に加算
            if input_general.save_bonddipole:
                np.savetxt(file_ch_dipole,np.hstack([np.ones((len(y_pred_ch),1))*fr_index,np.arange(len(y_pred_ch)).reshape(-1,1),y_pred_ch]),fmt="%d %d %f %f %f")
                
        # co, oh, cc, o
        if len(itp_data.co_bond_index) != 0 and model_co  != None:
            bond_centers = np.array(list_bond_centers)[:,itp_data.co_bond_index,:].reshape((-1,3))
            Descs_co     = DESC.calc_descriptor(atoms=fr_atoms,
                                                bond_centers=bond_centers,
                                                list_atomic_number=[6,1,8], 
                                                list_maxat=[24,24,24], 
                                                Rcs=input_descriptor.Rcs, 
                                                Rc=input_descriptor.Rc, 
                                                device=input_predict.device)
            X_co = torch.from_numpy(Descs_co.astype(np.float32)).clone() # オリジナルの記述子を一旦tensorへ
            y_pred_co  = model_co(X_co.to(input_predict.device)).to("cpu").detach().numpy()
            y_pred_co = y_pred_co.reshape((-1,3))
            del Descs_co
            sum_dipole += np.sum(y_pred_co,axis=0)  #双極子に加算
            if input_general.save_bonddipole:
                np.savetxt(file_co_dipole,np.hstack([np.ones((len(y_pred_co),1))*fr_index,np.arange(len(y_pred_co)).reshape(-1,1),y_pred_co]),fmt="%d %d %f %f %f")
                
        if len(itp_data.oh_bond_index) != 0 and model_oh  != None:
            bond_centers = np.array(list_bond_centers)[:,itp_data.oh_bond_index,:].reshape((-1,3))
            Descs_oh     = DESC.calc_descriptor(atoms=fr_atoms,
                                                bond_centers=bond_centers,
                                                list_atomic_number=[6,1,8], 
                                                list_maxat=[24,24,24], 
                                                Rcs=input_descriptor.Rcs, 
                                                Rc=input_descriptor.Rc, 
                                                device=input_predict.device)
            X_oh = torch.from_numpy(Descs_oh.astype(np.float32)).clone() # オリジナルの記述子を一旦tensorへ
            y_pred_oh  = model_oh(X_oh.to(input_predict.device)).to("cpu").detach().numpy()
            y_pred_oh = y_pred_oh.reshape((-1,3))
            del Descs_oh
            sum_dipole += np.sum(y_pred_oh,axis=0)
            if input_general.save_bonddipole:
                np.savetxt(file_oh_dipole,np.hstack([np.ones((len(y_pred_oh),1))*fr_index,np.arange(len(y_pred_oh)).reshape(-1,1),y_pred_oh]),fmt="%d %d %f %f %f")
                
        if len(itp_data.cc_bond_index) != 0 and model_cc  != None:
            bond_centers = np.array(list_bond_centers)[:,itp_data.cc_bond_index,:].reshape((-1,3))
            Descs_cc     = DESC.calc_descriptor(atoms=fr_atoms,
                                                bond_centers=bond_centers,
                                                list_atomic_number=[6,1,8], 
                                                list_maxat=[24,24,24], 
                                                Rcs=input_descriptor.Rcs, 
                                                Rc=input_descriptor.Rc, 
                                                device=input_predict.device)
            X_cc = torch.from_numpy(Descs_cc.astype(np.float32)).clone() # オリジナルの記述子を一旦tensorへ
            y_pred_cc  = model_cc(X_cc.to(input_predict.device)).to("cpu").detach().numpy()
            y_pred_cc = y_pred_cc.reshape((-1,3))
            del Descs_cc
            sum_dipole += np.sum(y_pred_cc,axis=0)
            if input_general.save_bonddipole:
                np.savetxt(file_cc_dipole,np.hstack([np.ones((len(y_pred_cc),1))*fr_index,np.arange(len(y_pred_cc)).reshape(-1,1),y_pred_cc]),fmt="%d %d %f %f %f")
                
        if len(itp_data.o_list) != 0 and model_o  != None:
            o_positions = fr_atoms.get_positions()[np.argwhere(fr_atoms.get_atomic_numbers()==8).reshape(-1)] # o原子の座標
            # o_positions = o_positions.reshape((-1,3)) # これを入れないと，[*,1,3]の形になってしまう
            Descs_o     = DESC.calc_descriptor(atoms=fr_atoms,
                                                bond_centers=o_positions,
                                                list_atomic_number=[6,1,8], 
                                                list_maxat=[24,24,24], 
                                                Rcs=input_descriptor.Rcs, 
                                                Rc=input_descriptor.Rc, 
                                                device=input_predict.device)
            X_o = torch.from_numpy(Descs_o.astype(np.float32)).clone() # オリジナルの記述子を一旦tensorへ
            y_pred_o  = model_o(X_o.to(input_predict.device)).to("cpu").detach().numpy()
            y_pred_o = y_pred_o.reshape((-1,3))
            del Descs_o
            sum_dipole += np.sum(y_pred_o,axis=0)
            if input_general.save_bonddipole:
                np.savetxt(file_o_dipole,np.hstack([np.ones((len(y_pred_o),1))*fr_index,np.arange(len(y_pred_o)).reshape(-1,1),y_pred_o]),fmt="%d %d %f %f %f")
            
        # !! >>>> ここからCOH/COC >>>
        if len(itp_data.coc_index) != 0 and model_coc  != None:
            # TODO :: このままだと通常のo_listを使ってしまっていてまずい．
            # TODO :: ちゃんとcohに対応したo_listを作るようにする．
            o_positions = fr_atoms.get_positions()[np.argwhere(fr_atoms.get_atomic_numbers()==8).reshape(-1)] # o原子の座標
            Descs_coc     = DESC.calc_descriptor(atoms=fr_atoms,
                                                bond_centers=o_positions,
                                                list_atomic_number=[6,1,8], 
                                                list_maxat=[24,24,24], 
                                                Rcs=input_descriptor.Rcs, 
                                                Rc=input_descriptor.Rc, 
                                                device=input_predict.device)
            X_coc      = torch.from_numpy(Descs_coc.astype(np.float32)).clone() # オリジナルの記述子を一旦tensorへ
            y_pred_coc = model_coc(X_coc.to(input_predict.device)).to("cpu").detach().numpy()
            y_pred_coc = y_pred_coc.reshape((-1,3))
            del Descs_coc
            sum_dipole += np.sum(y_pred_coc,axis=0)
            if input_general.save_bonddipole:
                np.savetxt(file_coc_dipole,np.hstack([np.ones((len(y_pred_coc),1))*fr_index,np.arange(len(y_pred_coc)).reshape(-1,1),y_pred_coc]),fmt="%d %d %f %f %f")
            
        if len(itp_data.coh_index) != 0 and model_coh  != None:
            # TODO :: このままだと通常のo_listを使ってしまっていてまずい．
            # TODO :: ちゃんとcohに対応したo_listを作るようにする．
            o_positions = fr_atoms.get_positions()[np.argwhere(fr_atoms.get_atomic_numbers()==8).reshape(-1)] # o原子の座標
            Descs_coh     = DESC.calc_descriptor(atoms=fr_atoms,
                                                bond_centers=o_positions,
                                                list_atomic_number=[6,1,8], 
                                                list_maxat=[24,24,24], 
                                                Rcs=input_descriptor.Rcs, 
                                                Rc=input_descriptor.Rc, 
                                                device=input_predict.device)
            X_coh      = torch.from_numpy(Descs_coh.astype(np.float32)).clone() # オリジナルの記述子を一旦tensorへ
            y_pred_coh = model_coh(X_coh.to(input_predict.device)).to("cpu").detach().numpy()
            y_pred_coh = y_pred_coh.reshape((-1,3))
            del Descs_coh
            sum_dipole += np.sum(y_pred_coh,axis=0)
            if input_general.save_bonddipole:
                np.savetxt(file_coh_dipole,np.hstack([np.ones((len(y_pred_coh),1))*fr_index,np.arange(len(y_pred_coh)).reshape(-1,1),y_pred_coh]),fmt="%d %d %f %f %f")
            
        # !! <<< ここまでCOH/COC <<<
        # >>>>>>>  final process in the loop >>>>>>>
        # write to files
        file_total_dipole.write(f"{fr_index} {sum_dipole[0]} {sum_dipole[1]} {sum_dipole[2]}\n")
        # calculate molecular dipole
        molecule_dipole =   y_pred_ch.reshape((NUM_MOL,-1,3)).sum(axis=1) + \
                            y_pred_co.reshape((NUM_MOL,-1,3)).sum(axis=1) + \
                            y_pred_oh.reshape((NUM_MOL,-1,3)).sum(axis=1) + \
                            y_pred_cc.reshape((NUM_MOL,-1,3)).sum(axis=1) + \
                            y_pred_o.reshape((NUM_MOL,-1,3)).sum(axis=1) + \
                            y_pred_coc.reshape((NUM_MOL,-1,3)).sum(axis=1) + \
                            y_pred_coh.reshape((NUM_MOL,-1,3)).sum(axis=1)
        np.savetxt(file_mol_dipole,np.hstack([np.ones((len(molecule_dipole),1))*fr_index,np.arange(len(molecule_dipole)).reshape(-1,1),molecule_dipole]),fmt="%d %d %f %f %f")

    # finish writing
    file_total_dipole.close()
    file_mol_dipole.close()
    file_cc_dipole.close()
    file_ch_dipole.close()
    file_co_dipole.close()
    file_oh_dipole.close()
    file_o_dipole.close()
    file_coc_dipole.close()
    file_coh_dipole.close()
    return 0


def command_cptrain_pred(args)-> int:
    """mltrain train 
        wrapper for mlpred
    Args:
        args (_type_): _description_
    """
    mlpred(args.input)
    return 0
