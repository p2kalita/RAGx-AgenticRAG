"""
### 4.2. Bivariate Analysis (Predictor Association & Multicollinearity)

###############################################################
# STEP 5
# ASSOCIATION AMONG PREDICTORS
# AND MULTICOLLINEARITY ANALYSIS
###############################################################
"""

from typing import List, Optional
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

###############################################################
# Predictor Variables
###############################################################

DEFAULT_PREDICTORS = [
    "Question Type",
    "Question Complexity",
    "Technology Domain",
    "Business Capability",
    "Question Intent"
]

###############################################################
# Function to calculate Cramer's V
###############################################################


def cramers_v(x, y):
    """
    Calculate Cramér's V statistic for categorical-categorical association.

    Parameters
    ----------
    x : array-like or pd.Series
        First categorical variable.
    y : array-like or pd.Series
        Second categorical variable.

    Returns
    -------
    float
        Cramér's V association metric between 0 and 1.
    """
    contingency = pd.crosstab(x, y)

    chi2 = chi2_contingency(contingency)[0]

    n = contingency.sum().sum()

    min_dim = min(contingency.shape)-1

    return np.sqrt(chi2/(n*min_dim))


###############################################################
# Create Association Matrix
###############################################################


def compute_association_matrix(
    df: pd.DataFrame,
    predictors: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute pairwise Cramér's V association matrix across specified predictor variables.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the predictor columns.
    predictors : Optional[List[str]], optional
        List of predictor variables. Defaults to DEFAULT_PREDICTORS.

    Returns
    -------
    pd.DataFrame
        Symmetric matrix of pairwise Cramér's V values.
    """
    pred_cols = predictors if predictors is not None else DEFAULT_PREDICTORS
    association_matrix = pd.DataFrame(
        index=pred_cols,
        columns=pred_cols,
        dtype=float
    )

    for row in pred_cols:
        for col in pred_cols:
            association_matrix.loc[row, col] = cramers_v(df[row], df[col])

    return association_matrix


def display_association_matrix(association_matrix: pd.DataFrame) -> None:
    """
    Display formatted Cramér's V association matrix.

    Parameters
    ----------
    association_matrix : pd.DataFrame
        Association matrix to display.
    """
    print("=" * 90)
    print("CRAMER'S V ASSOCIATION MATRIX")
    print("=" * 90)

    try:
        display(association_matrix.round(3))
    except NameError:
        print(association_matrix.round(3))


def run_cramers_v_analysis(
    df: pd.DataFrame,
    predictors: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Execute Cramér's V association matrix calculation and display results.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing predictors.
    predictors : Optional[List[str]], optional
        List of predictors.

    Returns
    -------
    pd.DataFrame
        Pairwise Cramér's V association matrix.
    """
    matrix = compute_association_matrix(df, predictors)
    display_association_matrix(matrix)
    return matrix


if __name__ == "__main__":
    import os
    import importlib.util

    current_dir = os.path.dirname(os.path.abspath(__file__))
    loader_path = os.path.join(current_dir, "01_environment_and_data_loading.py")
    if os.path.exists(loader_path):
        spec = importlib.util.spec_from_file_location("step_01_loader", loader_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        df = mod.load_cleaned_dataset()
        matrix = run_cramers_v_analysis(df)

