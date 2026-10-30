"""
### 4.2. Bivariate Analysis - Variance Inflation Factor (VIF) Analysis

This module performs Variance Inflation Factor (VIF) analysis on one-hot encoded predictor variables
to detect and evaluate multicollinearity.
"""

###############################################################
# STEP 5
# VIF ANALYSIS
###############################################################

from typing import List, Optional
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

DEFAULT_PREDICTORS = [
    "Question Type",
    "Question Complexity",
    "Technology Domain",
    "Business Capability",
    "Question Intent"
]


def prepare_encoded_predictors(
    df: pd.DataFrame,
    predictors: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    One-hot encode categorical predictors (drop_first=True) and cast boolean indicators to integers.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the predictor columns.
    predictors : Optional[List[str]], optional
        List of predictor variables. Defaults to DEFAULT_PREDICTORS.

    Returns
    -------
    pd.DataFrame
        One-hot encoded design matrix X.
    """
    pred_cols = predictors if predictors is not None else DEFAULT_PREDICTORS
    X = pd.get_dummies(df[pred_cols], drop_first=True)
    X = X.astype(int)
    return X


def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for each variable in design matrix X.

    Parameters
    ----------
    X : pd.DataFrame
        Encoded predictor design matrix.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'Variable' and 'VIF' sorted descending by VIF.
    """
    print("\n")
    print("=" * 90)
    print("VARIANCE INFLATION FACTOR (VIF)")
    print("=" * 90)

    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]
    vif = vif.sort_values(by="VIF", ascending=False)

    try:
        display(vif)
    except NameError:
        print(vif)

    return vif


def run_vif_analysis(
    df: pd.DataFrame,
    predictors: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Execute full VIF pipeline: encoding predictors and computing VIF table.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing predictors.
    predictors : Optional[List[str]], optional
        List of predictors.

    Returns
    -------
    pd.DataFrame
        VIF table sorted descending.
    """
    X = prepare_encoded_predictors(df, predictors)
    vif_table = calculate_vif(X)
    return vif_table


if __name__ == "__main__":
    import os
    import importlib.util

    current_dir = os.path.dirname(os.path.abspath(__file__))
    loader_path = os.path.join(current_dir, "01_environment_and_data_loading.py")

    if os.path.exists(loader_path):
        spec_load = importlib.util.spec_from_file_location("step_01_loader", loader_path)
        mod_load = importlib.util.module_from_spec(spec_load)
        spec_load.loader.exec_module(mod_load)
        df = mod_load.load_cleaned_dataset()
        vif = run_vif_analysis(df)

