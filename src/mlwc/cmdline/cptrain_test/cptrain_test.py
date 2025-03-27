#!/usr/bin/env python
# coding: utf-8
"""
This script provides a command-line interface for testing machine learning models
trained to predict properties of molecular systems, particularly focusing on
Wannier center (WC) based models. It loads a pre-trained model, molecular
structure data from an XYZ file, and atom type information from an ITP or MOL file.
The script then evaluates the model's performance by comparing its predictions
against the true values from the provided data, calculating metrics such as
Root Mean Squared Error (RMSE) and R-squared. The results are saved to files
and visualized using plots.
"""
from __future__ import annotations
import numpy as np
import torch
import time
import ase.io
import ase
from mlwc.ml.dataset.mldataset_xyz import DataSet_xyz, DataSet_xyz_coc
import mlwc.bond.atomtype
import mlwc.ml.train.ml_train  # for figures
from sklearn.metrics import r2_score
from mlwc.include.constants import constant
from mlwc.include.mlwc_logger import setup_cmdline_logger
# Debye   = 3.33564e-30
# charge  = 1.602176634e-019
# ang      = 1.0e-10
coef = constant.Ang*constant.Charge/constant.Debye

logger = setup_cmdline_logger("MLWC."+__name__)


def command_mltrain_test(args) -> int:
    """
    Wrapper for the `mltest` function to test a machine learning model.

    This function takes command-line arguments, extracts the input file path,
    and calls the `mltest` function to perform the model testing.

    Parameters
    ----------
    args : argparse.Namespace
        An object containing the command-line arguments, including the path
        to the input file.

    Returns
    -------
    int
        Returns 0 upon successful execution.

    Examples
    --------
    >>> args = argparse.Namespace(input='path/to/model.pth')
    >>> command_mltrain_test(args)
    0
    """
    mltest(args.input)
    return 0


def mltest(model_filename: str, xyz_filename: str, itp_filename: str, bond_name: str) -> None:
    """
    Tests a machine learning model for predicting molecular properties.

    This function loads a pre-trained PyTorch model, molecular structure data
    from an XYZ file, and atom type information from an ITP or MOL file.
    It then evaluates the model's performance by comparing its predictions
    against the true values from the provided data, calculating metrics such as
    Root Mean Squared Error (RMSE) and R-squared. The results are saved to files
    and visualized using plots.

    Parameters
    ----------
    model_filename : str
        Path to the pre-trained PyTorch model file (``.pth`` or ``.pt``).
    xyz_filename : str
        Path to the XYZ file containing the molecular structure data.
    itp_filename : str
        Path to the ITP or MOL file containing the atom type information.
    bond_name : str
        Name of the bond to be calculated.

    Returns
    -------
    None

    Examples
    --------
    >>> mltest('model.pth', 'traj.xyz', 'mol.itp', 'OH')
    """
    logger.info(" ")
    logger.info(" --------- ")
    logger.info(" subcommand test :: validation for ML models")
    logger.info(" ")

    # * モデルのロード ( torch scriptで読み込み)
    # https://take-tech-engineer.com/pytorch-model-save-load/
    # check cpu/gpu/mps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_filename).to(device)

    #
    logger.info(" ==========  Model Parameter information  ============ ")
    try:
        logger.info(f" M         = {model.M}")
    except:
        logger.info("The model do not contain M")
    try:
        logger.info(f" Mb        = {model.Mb}")
    except:
        logger.info("The model do not contain Mb")
    try:
        logger.info(f" nfeatures = {model.nfeatures}")
        MaxAt: int = int(model.nfeatures/4/3)
        logger.info(f" MaxAt     = {MaxAt}")
    except:
        logger.info("The model do not contain nfeatures")
    try:
        logger.info(f" Rcs = {model.Rcs}")
        logger.info(f" Rc = {model.Rc}")
        logger.info(f" type = {model.bondtype}")
        bond_name: str = model.bondtype  # 上書き
        Rcs: float = model.Rcs
        Rc: float = model.Rc
    except:
        logger.info(" WARNING :: model is old (not include Rc, Rcs, type)")
        Rcs: float = 4.0  # default value
        Rc: float = 6.0  # default value
    logger.info(" ====================== ")

    # * read itp
    # FIXME :: itpファイルは記述子からデータを読み込む場合は不要なのでコメントアウトしておく
    if itp_filename.endswith(".itp"):
        itp_data = mlwc.bond.atomtype.read_itp(itp_filename)
    elif itp_filename.endswith(".mol"):
        itp_data = mlwc.bond.atomtype.read_mol(itp_filename)
    else:
        logger.info("ERROR :: itp_filename should end with .itp or .mol")
    NUM_MOL_ATOMS = itp_data.num_atoms_per_mol

    # * load trajectory
    logger.info(" Loading xyz file :: ", xyz_filename)
    atoms_list = ase.io.read(xyz_filename, index=":")
    logger.info(f" Finish loading xyz file. len(traj) = {len(atoms_list)}")

    # * construct atoms_wan instance from xyz
    # note :: datasetから分離している理由は，wannierの割り当てを並列計算でやりたいため．
    import cpmd.class_atoms_wan

    logger.info(" splitting atoms into atoms and WCs")
    atoms_wan_list = []
    for atoms in atoms_list:
        atoms_wan_list.append(cpmd.class_atoms_wan.atoms_wan(
            atoms, NUM_MOL_ATOMS, itp_data))

    #
    # * Assign Wannier centers to bonds
    # TODO :: joblibでの並列化を試したが失敗した．
    # TODO :: どうもjoblibだとインスタンス変数への代入はうまくいかないっぽい．
    logger.info(" Assigning Wannier Centers")
    for atoms_wan_fr in atoms_wan_list:
        def y(x): return x._calc_wcs()
        y(atoms_wan_fr)
    logger.info(" Finish Assigning Wannier Centers")

    # atoms_wan_fr._calc_wcs() for atoms_wan_fr in atoms_wan_list

    # * dataset&dataloader
    # make dataset
    # 第二変数で訓練したいボンドのインデックスを指定する．
    # 第三変数は記述子のタイプを表す
    if bond_name == "CH":
        calculate_bond = itp_data.bond_index['CH_1_bond']
    elif bond_name == "OH":
        calculate_bond = itp_data.bond_index['OH_1_bond']
    elif bond_name == "CO":
        calculate_bond = itp_data.bond_index['CO_1_bond']
    elif bond_name == "CC":
        calculate_bond = itp_data.bond_index['CC_1_bond']
    elif bond_name == "O":
        calculate_bond = itp_data.o_list
    elif bond_name == "COC":
        logger.info("INVOKE COC")
    elif bond_name == "COH":
        logger.info("INVOKE COH")
    else:
        raise ValueError(
            f"ERROR :: bond_name should be CH,OH,CO,CC or O {bond_name}")

    # set dataset
    if bond_name in ["CH", "OH", "CO", "CC"]:
        dataset = DataSet_xyz(
            atoms_wan_list, calculate_bond, "allinone", Rcs=Rcs, Rc=Rc, MaxAt=24, bondtype="bond")
    elif bond_name == "O":
        dataset = DataSet_xyz(
            atoms_wan_list, calculate_bond, "allinone", Rcs=Rcs, Rc=Rc, MaxAt=24, bondtype="lonepair")
    elif bond_name == "COC":
        dataset = DataSet_xyz_coc(
            atoms_wan_list, itp_data, "allinone", Rcs=Rcs, Rc=Rc, MaxAt=24, bondtype="coc")
    elif bond_name == "COH":
        dataset = DataSet_xyz_coc(
            atoms_wan_list, itp_data, "allinone", Rcs=Rcs, Rc=Rc, MaxAt=24, bondtype="coh")
    else:
        raise ValueError("ERROR :: bond_name should be CH,OH,CO,CC or O")
    # FIXME :: hard code :: batch_size=32
    # FIXME :: num_worker = 0 for mps
    dataloader_valid = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, drop_last=False, pin_memory=True, num_workers=0)

    # lists for results
    pred_list: list = []
    true_list: list = []

    # * Test models
    start_time = time.perf_counter()  # start time check
    model.eval()  # model to evaluation mode
    with torch.no_grad():  # https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
        for data in dataloader_valid:
            # self.logger.debug("start batch valid")
            if data[0].dim() == 3:  # 3次元の場合[NUM_BATCH,NUM_BOND,288]はデータを整形する
                # TODO :: torch.reshape(data[0], (-1, 288)) does not work !!
                for data_1 in zip(data[0], data[1]):
                    # self.logger.debug(f" DEBUG :: data_1[0].shape = {data_1[0].shape} : data_1[1].shape = {data_1[1].shape}")
                    # self.batch_step(data_1,validation=True)
                    x = data_1[0].to(device)  # modve descriptor to device
                    y = data_1[1]
                    y_pred = model(x)
                    pred_list.append(y_pred.to("cpu").detach().numpy())
                    true_list.append(y.detach().numpy())
            if data[0].dim() == 2:  # 2次元の場合はそのまま
                # self.batch_step(data,validation=True)
                x = data_1[0]
                y = data_1[1]
                y_pred = model(x)
                pred_list.append(y_pred.to("cpu").detach().numpy())
                true_list.append(y.detach().numpy())
    #
    pred_list = np.array(pred_list).reshape(-1, 3)
    true_list = np.array(true_list).reshape(-1, 3)
    end_time = time.perf_counter()  # timer stop
    # calculate RSME
    rmse = np.sqrt(np.mean((true_list-pred_list)**2))
    # save results
    logger.info(" ======")
    logger.info("  Finish testing.")
    logger.info("  Save results as pred_true_list.txt")
    logger.info(f" RSME_train = {rmse}")
    logger.info(f' r^2        = {r2_score(true_list,pred_list)}')
    logger.info(" ")
    logger.info(' ELAPSED TIME  {:.2f}'.format((end_time-start_time)))
    logger.info(np.shape(pred_list))
    logger.info(np.shape(true_list))
    np.savetxt("pred_list.txt", pred_list)
    np.savetxt("true_list.txt", true_list)
    # make figures
    ml.train.ml_train.make_figure(pred_list, true_list)
    ml.train.ml_train.plot_residure_density(pred_list, true_list)
    return 0


def command_cptrain_test(args) -> int:
    """
    Command-line interface for testing a machine learning model.

    This function serves as the entry point for testing a pre-trained machine
    learning model using the provided command-line arguments. It parses the
    arguments to extract the model file, XYZ file, MOL/ITP file, and bond type,
    and then calls the `mltest` function to perform the actual testing.

    Parameters
    ----------
    args : argparse.Namespace
        An object containing the command-line arguments.

    Returns
    -------
    int
        Returns 0 upon successful execution.

    Examples
    --------
    >>> args = argparse.Namespace(model='model.pth', xyz='traj.xyz', mol='mol.itp', bond='OH')
    >>> command_cptrain_test(args)
    0
    """
    mltest(args.model, args.xyz, args.mol, args.bond)
    return 0
