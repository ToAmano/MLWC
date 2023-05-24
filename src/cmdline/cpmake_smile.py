#!/usr/bin/env python3

import sys,os,os.path
import argparse
import pandas as pd
# from rdkit import Chem
# from rdkit.Chem import Draw
# from rdkit.Chem import Descriptors
# from rdkit.ML.Descriptors import MoleculeDescriptors
# from rdkit.Chem import PandasTools

def make_itp(csv_filename):


    input_file:str = csv_filename
    poly = pd.read_csv(input_file)
    
    print(" --------- ")
    print(poly)
    print(" --------- ")
    
    
    #化学構造のsmilesリスト，ラベルリストを準備する．
    smiles_list_polymer:list = poly["Smiles"].to_list()    
    label_list:list = poly["Name"].to_list()
    
    # molオブジェクトのリストを作成
    # mols_list_polymer = [Chem.MolFromSmiles(smile) for smile in smiles_list_polymer]
    
    # print(str(smiles_list_polymer[0]))    
    
    #GAFF2/AM1-BCCをアサインする
    '''
    input
    -----------
    SMILES
    
    smiles.mol2 :: mol2フォーマットのファイルを出力
    '''

    # * hard code
    # どうも最初のsmilesだけinput.acpypeを作成するようだ．
    smiles = smiles_list_polymer[0] 
    
    os.system('echo "{0}" > {1}'.format(str(smiles), "input.smi"))
    # convert smiles to mol2 ( tripo mol2 format file)
    os.system('obabel -ismi {0} -O {1} --gen3D --conformer --nconf 5000 --weighted'.format("input.smi","input.mol2"))
    
    # making input.xyz ?
    os.system('obabel -imol2 {0} -oxyz -O {1}'.format("input.mol2","input.xyz"))
    # from ase.io import read, write
    # inp1 = read('input.xyz')
    
    # convert input.mol2 to input1.gro & input1.itp ?
    import platform
    if platform.system() == 'Linux':
        os.system('acpype -i {0} -c bcc -n 0 -m 1 -a gaff2 -f -o gmx -k "qm_theory=\'AM1\', grms_tol=0.05, scfconv=1.d-10, ndiis_attempts=700, "'.format("input.mol2"))
    else: # on mac
        os.system('acpype_docker -i {0} -c bcc -n 0 -m 1 -a gaff2 -f -o gmx -k "qm_theory=\'AM1\', grms_tol=0.05, scfconv=1.d-10, ndiis_attempts=700, "'.format("input.mol2"))
        os.system('acpype -i {0} -c bcc -n 0 -m 1 -a gaff2 -f -o gmx -k "qm_theory=\'AM1\', grms_tol=0.05, scfconv=1.d-10, ndiis_attempts=700, "'.format("input.mol2"))


    print(" --------- ")
    print(" FINISH acpype ")
    print(" Please check input_GMX.itp and input_GMX.gro ")
    print(" ")
    
    return 0
    


def command_smile(args):
    print(" ")
    print(" --------- ")
    print(" input smile file :: ", args.input )
    print(" ")
    make_itp(args.input)
    return 0


