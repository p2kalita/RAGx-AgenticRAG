"""
### 4.2. Bivariate Analysis - Predictor Association Heatmap

This module visualizes the pairwise Cramér's V association matrix between predictor variables
using a styled seaborn heatmap.
"""

###############################################################
# Heatmap
###############################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_association_heatmap(
    association_matrix: pd.DataFrame,
    figsize: tuple = (9, 7),
    cmap: str = "YlGnBu",
    vmin: float = 0.0,
    vmax: float = 1.0,
    linewidths: float = 0.5,
    title: str = "Association Between Predictor Variables"
) -> None:
    """
    Render heatmap of the predictor association matrix.

    Parameters
    ----------
    association_matrix : pd.DataFrame
        Pairwise Cramér's V association matrix.
    figsize : tuple, default=(9, 7)
        Figure dimension.
    cmap : str, default='YlGnBu'
        Colormap.
    vmin : float, default=0.0
        Colormap minimum value.
    vmax : float, default=1.0
        Colormap maximum value.
    linewidths : float, default=0.5
        Width of lines dividing each cell.
    title : str, default="Association Between Predictor Variables"
        Title of the heatmap.
    """
    plt.figure(figsize=figsize)

    sns.heatmap(
        association_matrix,
        annot=True,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=linewidths
    )

    plt.title(title)

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    import os
    import importlib.util

    current_dir = os.path.dirname(os.path.abspath(__file__))
    loader_path = os.path.join(current_dir, "01_environment_and_data_loading.py")
    matrix_path = os.path.join(current_dir, "06_bivariate_cramers_v_matrix.py")

    if os.path.exists(loader_path) and os.path.exists(matrix_path):
        spec_load = importlib.util.spec_from_file_location("step_01_loader", loader_path)
        mod_load = importlib.util.module_from_spec(spec_load)
        spec_load.loader.exec_module(mod_load)
        df = mod_load.load_cleaned_dataset()

        spec_matrix = importlib.util.spec_from_file_location("step_06_matrix", matrix_path)
        mod_matrix = importlib.util.module_from_spec(spec_matrix)
        spec_matrix.loader.exec_module(mod_matrix)
        matrix = mod_matrix.compute_association_matrix(df)

        plot_association_heatmap(matrix)

