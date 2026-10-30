"""
# Lets analyze the numerical variables.

###############################################################
# NUMERICAL VARIABLE ANALYSIS
#
# Analyse the numerical variables within the dataset.
###############################################################
"""

from typing import List, Optional
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_NUMERIC_COLUMNS = [
    'Question Length',
    'Number of Technologies Mentioned',
    'Estimated RFP Relevance (1-5)'
]


def analyze_numerical_variable(df: pd.DataFrame, col: str, bins: int = 20) -> pd.Series:
    """
    Compute descriptive statistics and render histogram for a single numerical variable.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the column.
    col : str
        Numerical column name.
    bins : int, default=20
        Number of histogram bins.

    Returns
    -------
    pd.Series
        Descriptive summary statistics of the column.
    """
    print("\n")
    print("=" * 60)
    print(col)
    print("=" * 60)

    summary = df[col].describe()
    print(summary)

    plt.figure(figsize=(8, 4))

    plt.hist(df[col], bins=bins)

    plt.title(col)

    plt.xlabel(col)

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.show()

    return summary


def run_univariate_numerical_analysis(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> dict:
    """
    Perform univariate numerical analysis across all specified numerical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze.
    columns : Optional[List[str]], optional
        List of numerical column names. Defaults to DEFAULT_NUMERIC_COLUMNS.

    Returns
    -------
    dict
        Dictionary mapping column names to their descriptive summary series.
    """
    cols_to_analyze = columns if columns is not None else DEFAULT_NUMERIC_COLUMNS
    summaries = {}
    for col in cols_to_analyze:
        summaries[col] = analyze_numerical_variable(df, col)
    return summaries


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
        run_univariate_numerical_analysis(df)

