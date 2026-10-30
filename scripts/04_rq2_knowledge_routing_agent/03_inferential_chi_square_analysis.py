"""
## 1. Inferential Statistics (RQ2 specific) - Chi-square & Cramér's V Analysis

This module performs bivariate Chi-square tests of independence and computes Cramér's V effect sizes
between enterprise presales features and the target repository classes.
"""

###############################################################
# CHI-SQUARE TEST
###############################################################

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

DEFAULT_TARGET = "Enterprise Knowledge Required"
DEFAULT_FEATURES = [
    "Question Type",
    "Question Intent",
    "Business Capability",
    "Question Complexity"
]


def chi_square_test(feature, target, df=None) -> Dict[str, Any]:
    """
    Compute Chi-square statistic, p-value, degrees of freedom, and Cramér's V.

    Parameters
    ----------
    feature : str
        Predictor column name.
    target : str
        Target column name.
    df : pd.DataFrame, optional
        Dataset containing the columns.

    Returns
    -------
    Dict[str, Any]
        Dictionary of statistical metrics.
    """
    table = pd.crosstab(
        df[feature],
        df[target]
    )

    chi2, p, dof, expected = chi2_contingency(table)

    n = table.sum().sum()

    k = min(table.shape) - 1

    cramers_v = np.sqrt(
        chi2 / (n * k)
    )

    print("=" * 70)
    print(feature)
    print("=" * 70)
    print("Chi-square :", round(chi2, 2))
    print("p-value :", round(p, 5))
    print("Degrees of Freedom :", dof)
    print("Cramer's V :", round(cramers_v, 3))

    return {
        "Feature": feature,
        "Chi-square": round(chi2, 2),
        "p-value": round(p, 5),
        "Degrees of Freedom": dof,
        "Cramer's V": round(cramers_v, 3)
    }


def run_chi_square_analysis(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    target: str = DEFAULT_TARGET
) -> List[Dict[str, Any]]:
    """
    Iterate over candidate features and run Chi-square tests against target repository.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze.
    features : Optional[List[str]], optional
        List of candidate features. Defaults to DEFAULT_FEATURES.
    target : str, default=DEFAULT_TARGET
        Target variable name.

    Returns
    -------
    List[Dict[str, Any]]
        List of test results.
    """
    feat_cols = features if features is not None else DEFAULT_FEATURES
    results = []
    for feature in feat_cols:
        res = chi_square_test(feature, target, df=df)
        results.append(res)
    return results


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
        run_chi_square_analysis(df)

