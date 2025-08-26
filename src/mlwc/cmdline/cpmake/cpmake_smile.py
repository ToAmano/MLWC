#!/usr/bin/env python3
"""
To use this file, install obabel and acpype.
https://open-babel.readthedocs.io/en/latest/UseTheLibrary/Python.html
"""

import os
import platform
import shutil

import pandas as pd

from mlwc.include.mlwc_logger import setup_library_logger

logger = setup_library_logger("MLWC." + __name__)


def _convert_smiles_to_mol(smiles: str) -> None:
    os.system(f'echo "{smiles}" > input.smi')
    os.system(
        "obabel -ismi input.smi -O input.mol2 --gen3D --conformer --nconf 5000 --weighted"
    )


def _convert_mol_to_xyz(input_molfile: str) -> None:
    if not os.path.isfile(input_molfile):
        raise FileNotFoundError(f"{input_molfile} not found")
    os.system(f"obabel -imol2 {input_molfile} -oxyz -O 'input.xyz'")


def _convert_mol_to_gro(input_grofile: str, output_molfile: str) -> None:
    pass


def _convert_gro_to_mol(input_grofile: str, output_molfile: str) -> None:
    logger.info("")
    logger.info(" convert input_GMX.gro to input_GMX.mol (obabel) ")
    logger.info("")
    os.system(f"obabel -i gro {input_grofile} -o mol -O {output_molfile}")


def make_itp(csv_filename: str):

    logger.info(" -------------- ")
    logger.info("  !! csv must contain Smiles and Name column ")
    logger.info(" -------------- ")

    poly: pd.DataFrame = pd.read_csv(csv_filename, comment="#")

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
    logger.info(" start convesion :: files will be saved to %s", savedirname)
    logger.info(" ------------ ")
    if os.path.isdir(savedirname) is True:
        logger.info(" ERROR :: dir %s exists !!", savedirname)
        return 1
    os.mkdir(defaultsavedirname)

    # !! TODO :: 全てのos.systemが正常に動作しない場合にエラー処理を行う

    # generate input.mol2
    _convert_smiles_to_mol(str(smiles))

    # making input.xyz ?
    _convert_mol_to_xyz("input.mol2")

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
    _convert_gro_to_mol(
        defaultsavedirname + "/input_GMX.gro", defaultsavedirname + "/input_GMX.mol"
    )

    # move input.mol2, input.smi, input.xyz to input.acpype/
    logger.info(" --------- ")
    logger.info(" mv input.mol2,input.smi,input.xyz to %s", defaultsavedirname)
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
    logger.info(" input smile file :: %s", args.input)
    logger.info(" ")
    make_itp(args.input)
    return 0
