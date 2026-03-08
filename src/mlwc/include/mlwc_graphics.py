import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from mlwc.include.mlwc_logger import setup_library_logger

logger = setup_library_logger("MLWC." + __name__)


def make_pred_true_figure(
    pred_values: np.ndarray,
    true_values: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    title: str = "Prediction vs. True",
    xlabel: str = "ML predicted dipole [D]",
    ylabel: str = "DFT simulated dipole [D]",
) -> plt.Axes:
    """
    Creates a scatter plot of true values vs. predicted values.

    If an Axes object is provided, it draws on it. Otherwise, it creates a new Figure and Axes.

    Parameters
    ----------
    pred_values : np.ndarray
        An array of predicted values.
    true_values : np.ndarray
        An array of true values.
    ax : Optional[plt.Axes], optional
        The matplotlib Axes to draw on. If None, a new Figure/Axes is created.
    title : str, optional
        The title for the plot.
    xlabel : str, optional
        The label for the x-axis.
    ylabel : str, optional
        The label for the y-axis.

    Returns
    -------
    plt.Axes
        The Axes object with the plot.
    """
    logger.info("")
    logger.info(" Plot prediction and true values")
    logger.info("================================")
    logger.info(" ")
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(true_values, pred_values, alpha=0.3, s=10)
    ax.set_title(title, fontsize=24)
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.grid(True)
    ax.tick_params(labelsize=20)

    # Add y=x identity line for reference
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.75, zorder=0)
    ax.set_aspect("equal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    return ax


def _calculate_gaussian_kde(
    data_x: np.ndarray, data_y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates Gaussian Kernel Density Estimation for scatter plot coloring.

    Parameters
    ----------
    data_x : np.ndarray
        X-coordinates of the data points.
    data_y : np.ndarray
        Y-coordinates of the data points.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Sorted x, y, and z (density) values for plotting.
    """
    xy = np.vstack([data_x, data_y])
    z = gaussian_kde(xy)(xy)
    # Sort points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = data_x[idx], data_y[idx], z[idx]
    return x, y, z


def plot_residue_density(
    pred_list: np.ndarray,
    true_list: np.ndarray,
    *,
    fig: Optional[plt.Figure] = None,
    axes: Optional[np.ndarray] = None,
    titles: Optional[List[str]] = None,
    xlabel: str = "ML dipole [D]",
    ylabel: str = "DFT dipole [D]",
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plots density maps of prediction vs. true values for each component (x, y, z).

    If Figure and Axes objects are provided, it draws on them.
    Otherwise, it creates a new 1x3 subplot Figure.

    Parameters
    ----------
    pred_list : np.ndarray
        Nx3 array of predicted vectors.
    true_list : np.ndarray
        Nx3 array of true vectors.
    fig : Optional[plt.Figure], optional
        The matplotlib Figure to draw on.
    axes : Optional[np.ndarray], optional
        A numpy array of 3 matplotlib Axes to draw on.
    titles : Optional[List[str]], optional
        A list of 3 titles for the subplots.
    xlabel : str, optional
        The label for the x-axis for all subplots.
    ylabel : str, optional
        The label for the y-axis for all subplots.

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        The Figure and array of Axes objects with the plots.
    """
    logger.info("")
    logger.info(" calculate density map (takes a few minutes)")
    logger.info("============================================")
    logger.info(" ")
    if axes is None or fig is None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    if titles is None:
        titles = [f"Dipole_{comp}" for comp in ["x", "y", "z"]]

    for i, ax in enumerate(axes):
        x_kde, y_kde, z_kde = _calculate_gaussian_kde(pred_list[:, i], true_list[:, i])

        im = ax.scatter(x_kde, y_kde, c=z_kde, s=50, cmap="jet")
        fig.colorbar(im, ax=ax)

        ax.set_title(titles[i])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)

    fig.tight_layout()
    return fig, axes
