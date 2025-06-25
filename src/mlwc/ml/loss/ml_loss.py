"""
This module defines the LossStatistics class, which is used to accumulate
loss function values over all batches for each loss component during
training and validation. It also provides methods for calculating and
saving loss statistics for each epoch.

This file provides the implementation for managing and accumulating loss statistics
during the training and validation phases of a machine learning model.
It includes functionalities for tracking loss values for each batch and epoch,
as well as saving these statistics to files for later analysis.
The LossStatistics class is designed to be flexible and can be used with various loss functions and model architectures.

"""

import numpy as np
import pandas as pd


class LossStatistics:
    """
    Accumulates loss function values over all batches for each loss component.

    This class provides methods for accumulating loss values during training and validation,
    calculating statistics such as mean loss and RMSE, and saving these statistics to files.

    Attributes
    ----------
    modeldir : str
        The directory where the model and loss statistics are saved.

    Examples
    --------
    >>> loss_stats = LossStatistics(modeldir="my_model")
    >>> loss_stats.add_train_batch_loss(loss=0.1, iepoch=1)
    >>> loss_stats.add_valid_batch_loss(loss=0.05, iepoch=1)
    >>> # After each epoch:
    >>> # loss_stats.add_train_epoch_loss()
    >>> # loss_stats.add_valid_epoch_loss()
    >>> # loss_stats.reset_train_batch_loss()
    >>> # loss_stats.reset_valid_batch_loss()
    """

    def __init__(self, modeldir: str):
        """
        Initializes the LossStatistics object.

        Parameters
        ----------
        modeldir : str
            The directory where the model and loss statistics are saved.

        """
        # conter # of epochs
        self.modeldir = modeldir

        # https://bering.hatenadiary.com/entry/2023/05/15/064223
        # validation rmse/loss for each batch
        # pd.DataFrame(["epoch", "batch", "loss", "rmse"])
        self.df_batch_valid: list[dict] = []
        # training rmse/loss for each batch
        # pd.DataFrame(["epoch", "batch", "loss", "rmse"])
        self.df_batch_train: list[dict] = []

        # epoch loss (train and valid)
        # validation/training rmse/loss for each epoch
        # pd.DataFrame(["epoch", "train_loss", "valid_loss", "train_rmse", "valid_rmse"])
        self.df_epoch = []

        # !! :: 実装の発想としてはいくつかある．
        # !! :: 一つは愚直に全てのデータをlistに保持しておくパターン．最後にpandasにしてデータを保存する．
        # !! :: もう一つは，epochごとにデータを廃棄する方法．epochごとにデータを保存する．
        # !! :: 2024/3/24 :: 結局epochごとにデータを廃棄することにした．

    def add_train_batch_loss(self, loss: float, iepoch: int) -> None:
        """Adds the training batch loss to the statistics.

        This method appends the loss value of a training batch to the internal list `self.df_batch_train`.
        It also saves the loss value to a text file for record keeping.
        If a new epoch starts, it calculates and saves the epoch loss before resetting the batch loss.

        Parameters
        ----------
        loss : float
            The loss value for the training batch.
        iepoch : int
            The current epoch number.

        Returns
        -------
        None

        Examples
        --------
        >>> loss_stats = LossStatistics(modeldir="my_model")
        >>> loss_stats.add_train_batch_loss(loss=0.1, iepoch=1)
        """
        with open(f"{self.modeldir}/train_batch_loss.txt", "a") as f:
            print(
                f"{iepoch}  {len(self.df_batch_train)} {loss} {np.sqrt(loss)}", file=f
            )

        # if new epoch, print epoch result
        if len(self.df_batch_train) != 0:  # skip if no content in the list
            if iepoch > self.df_batch_train[-1]["epoch"]:
                self.add_train_epoch_loss()
                self.reset_train_batch_loss()

        self.df_batch_train.append(
            {
                "epoch": iepoch,
                "batch": len(self.df_batch_train),
                "loss": loss,
                "rmse": np.sqrt(loss),
            }
        )

    def add_valid_batch_loss(self, loss: float, iepoch: int) -> None:
        """Adds the validation batch loss to the statistics.

        This method appends the loss value of a validation batch to the internal list `self.df_batch_valid`.
        It also saves the loss value to a text file for record keeping.
        If a new epoch starts, it calculates and saves the epoch loss before resetting the batch loss.

        Parameters
        ----------
        loss : float
            The loss value for the validation batch.
        iepoch : int
            The current epoch number.

        Returns
        -------
        None

        Examples
        --------
        >>> loss_stats = LossStatistics(modeldir="my_model")
        >>> loss_stats.add_valid_batch_loss(loss=0.05, iepoch=1)
        """
        with open(f"{self.modeldir}/valid_batch_loss.txt", "a") as f:
            print(
                f"{iepoch}  {len(self.df_batch_valid)} {loss} {np.sqrt(loss)}", file=f
            )

        # if new epoch, print epoch result
        if len(self.df_batch_valid) != 0:  # skip if no content in the list
            if iepoch > self.df_batch_valid[-1]["epoch"]:
                self.add_valid_epoch_loss()
                self.reset_valid_batch_loss()

        self.df_batch_valid.append(
            {
                "epoch": iepoch,
                "batch": len(self.df_batch_valid),
                "loss": loss,
                "rmse": np.sqrt(loss),
            }
        )

    def add_valid_epoch_loss(self) -> None:
        """Adds the validation epoch loss to the statistics.

        Calculates the mean loss and RMSE for the validation epoch based on the accumulated batch losses in `self.df_batch_valid`.
        Saves the epoch loss to a text file.

        Returns
        -------
        None

        Examples
        --------
        >>> loss_stats = LossStatistics(modeldir="my_model")
        >>> loss_stats.df_batch_valid = [{"epoch": 1, "batch": 0, "loss": 0.05, "rmse": 0.2236}]
        >>> loss_stats.add_valid_epoch_loss()
        """
        # batch loss to epoch loss
        tmp_epoch_mean = pd.DataFrame(self.df_batch_valid).mean()
        # save data
        with open(f"{self.modeldir}/valid_epoch_loss.txt", "a") as f:
            print(
                f"{tmp_epoch_mean['epoch']} {tmp_epoch_mean['loss']} {tmp_epoch_mean['rmse']}",
                file=f,
            )

    def add_train_epoch_loss(self) -> None:
        """Adds the training epoch loss to the statistics.

        Calculates the mean loss and RMSE for the training epoch based on the accumulated batch losses in `self.df_batch_train`.
        Saves the epoch loss to a text file.

        Returns
        -------
        None

        Examples
        --------
        >>> loss_stats = LossStatistics(modeldir="my_model")
        >>> loss_stats.df_batch_train = [{"epoch": 1, "batch": 0, "loss": 0.1, "rmse": 0.3162}]
        >>> loss_stats.add_train_epoch_loss()
        """
        # batch loss to epoch loss
        # https://deepage.net/features/pandas-mean.html
        tmp_epoch_mean = pd.DataFrame(self.df_batch_train).mean()
        # save data
        with open(f"{self.modeldir}/train_epoch_loss.txt", "a") as f:
            print(
                f"{tmp_epoch_mean['epoch']} {tmp_epoch_mean['loss']} {tmp_epoch_mean['rmse']}",
                file=f,
            )

    def reset_valid_batch_loss(self) -> None:
        """Resets the validation batch loss list.

        This method clears the `self.df_batch_valid` list, effectively resetting the accumulated validation batch losses for the current epoch.

        Returns
        -------
        None

        Examples
        --------
        >>> loss_stats = LossStatistics(modeldir="my_model")
        >>> loss_stats.df_batch_valid = [{"epoch": 1, "batch": 0, "loss": 0.05, "rmse": 0.2236}]
        >>> loss_stats.reset_valid_batch_loss()
        >>> print(loss_stats.df_batch_valid)
        []
        """
        self.df_batch_valid = []

    def reset_train_batch_loss(self) -> None:
        """Resets the training batch loss list.

        This method clears the `self.df_batch_train` list, effectively resetting the accumulated training batch losses for the current epoch.

        Returns
        -------
        None

        Examples
        --------
        >>> loss_stats = LossStatistics(modeldir="my_model")
        >>> loss_stats.df_batch_train = [{"epoch": 1, "batch": 0, "loss": 0.1, "rmse": 0.3162}]
        >>> loss_stats.reset_train_batch_loss()
        >>> print(loss_stats.df_batch_train)
        []
        """
        self.df_batch_train = []

    def save_train_batch_loss():
        return 0

    def print_current_result():
        """Prints the current result.

        This method is a placeholder and currently does nothing.

        Returns
        -------
        int
            Returns 0.

        Examples
        --------
        >>> loss_stats = LossStatistics(modeldir="my_model")
        >>> loss_stats.print_current_result()
        0
        """
        return 0
