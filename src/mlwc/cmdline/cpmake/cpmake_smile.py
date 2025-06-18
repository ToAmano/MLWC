#!/usr/bin/env python3

import os
import platform
import shutil

import pandas as pd

from mlwc.include.mlwc_logger import setup_library_logger

logger = setup_library_logger("MLWC." + __name__)


def make_itp(csv_filename: str):

    logger.info(" -------------- ")
    logger.info("  !! csv must contain Smiles and Name column ")
    logger.info(" -------------- ")

    input_file: str = csv_filename  # read csv
    poly: pd.DataFrame = pd.read_csv(input_file, comment="#")

    logger.info(" --------- ")
    logger.info(poly)
    logger.info(" --------- ")

    # 化学構造のsmilesリスト，ラベルリストを準備する．
    smiles_list_polymer: list = poly["Smiles"].to_list()
    label_list: list = poly["Name"].to_list()

    # molオブジェクトのリストを作成
    # mols_list_polymer = [Chem.MolFromSmiles(smile) for smile in smiles_list_polymer]

    # logger.info(str(smiles_list_polymer[0]))

    # GAFF2/AM1-BCCをアサインする
    """
    input
    -----------
    SMILES

    smiles.mol2 :: mol2フォーマットのファイルを出力
    """

    # * hard code
    # どうも最初のsmilesだけinput.acpypeを作成するようだ．
    smiles = smiles_list_polymer[0]
    molname = label_list[0]
    savedirname = molname + ".acpype/"
    defaultsavedirname = "input.acpype/"
    logger.info(" ------------ ")
    logger.info(" start convesion :: files will be saved to {0}".format(savedirname))
    logger.info(" ------------ ")
    if os.path.isdir(savedirname) == True:
        logger.info(" ERROR :: dir {0} exists !!".format(savedirname))
        return 1
    os.mkdir(defaultsavedirname)

    # !! TODO :: 全てのos.systemが正常に動作しない場合にエラー処理を行う
    os.system(f'echo "{str(smiles)}" > input.smi')
    # convert smiles to mol2 ( tripo mol2 format file)
    os.system(
        f"obabel -ismi input.smi -O {input.mol2} --gen3D --conformer --nconf 5000 --weighted"
    )

    # making input.xyz ?
    os.system(f"obabel -imol2 {'input.mol2'} -oxyz -O {'input.xyz'}")

    # convert input.mol2 to input1.gro & input1.itp ?
    logger.info(platform.system())
    if platform.system() == "Linux":
        os.system(
            f"acpype -s 86400 -i {'input.mol2'} -c bcc -n 0 -m 1 -a gaff2 -f -o gmx -k \"qm_theory='AM1', grms_tol=0.05, scfconv=1.d-10, ndiis_attempts=700, \""
        )
    elif platform.system() == "Darwin":  # on intel mac
        os.system(
            f"acpype -s 86400 -i {'input.mol2'} -c bcc -n 0 -m 1 -a gaff2 -f -o gmx -k \"qm_theory='AM1', grms_tol=0.05, scfconv=1.d-10, ndiis_attempts=700, \""
        )
        os.system(
            f"acpype_docker -s 86400 -i {'input.mol2'} -c bcc -n 0 -m 1 -a gaff2 -f -o gmx -k \"qm_theory='AM1', grms_tol=0.05, scfconv=1.d-10, ndiis_attempts=700, \""
        )
    else:  # on m1 mac
        os.system(
            f"acpype_docker -s 86400 -i {'input.mol2'} -c bcc -n 0 -m 1 -a gaff2 -f -o gmx -k \"qm_theory='AM1', grms_tol=0.05, scfconv=1.d-10, ndiis_attempts=700, \""
        )
        os.system(
            f"acpype -s 86400 -i {'input.mol2'} -c bcc -n 0 -m 1 -a gaff2 -f -o gmx -k \"qm_theory='AM1', grms_tol=0.05, scfconv=1.d-10, ndiis_attempts=700, \""
        )

    # convert input1.gro to input.mol
    logger.info(" --------- ")
    logger.info(" convert input_GMX.gro to input_GMX.mol (obabel) ")
    logger.info(" ")
    os.system(
        "obabel -i gro {0} -o mol -O {1}".format(
            defaultsavedirname + "/input_GMX.gro", defaultsavedirname + "/input_GMX.mol"
        )
    )

    # move input.mol2, input.smi, input.xyz to input.acpype/
    logger.info(" --------- ")
    logger.info(" mv input.mol2,input.smi,input.xyz to {0}".format(defaultsavedirname))
    logger.info(" ")

    shutil.move("input.mol2", defaultsavedirname + "/input.mol2_2")
    shutil.move("input.smi", defaultsavedirname)
    shutil.move("input.xyz", defaultsavedirname)

    # move all files
    shutil.move(defaultsavedirname, savedirname)
    logger.info(" --------- ")
    logger.info(" FINISH acpype ")
    logger.info(" Please check input_GMX.itp and input_GMX.gro ")
    logger.info(" ")

    return 0


def command_smile(args):
    logger.info(" ")
    logger.info(" --------- ")
    logger.info(" input smile file :: ", args.input)
    logger.info(" ")
    make_itp(args.input)
    return 0
