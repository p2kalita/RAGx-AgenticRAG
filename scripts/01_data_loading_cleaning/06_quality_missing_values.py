"""
## 2. Dataset Quality Assessment - Missing Values

This module evaluates missing values across all variables in the Curated Enterprise Presales Research Corpus.
"""

import pandas as pd


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and display missing value counts and percentages per column.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to evaluate.

    Returns
    -------
    pd.DataFrame
        DataFrame containing 'Missing Count' and 'Missing %' for each column.
    """
    print("\n")
    print("=" * 60)
    print("MISSING VALUES")
    print("=" * 60)

    missing = pd.DataFrame({
        'Missing Count': df.isnull().sum(),
        'Missing %': round(df.isnull().mean() * 100, 2)
    })

    print(missing)
    return missing


if __name__ == "__main__":
    import os
    import importlib.util

    current_dir = os.path.dirname(os.path.abspath(__file__))
    loader_path = os.path.join(current_dir, "02_dataset_loading.py")
    if os.path.exists(loader_path):
        spec = importlib.util.spec_from_file_location("step_02_dataset_loading", loader_path)
        module_02 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module_02)
        df = module_02.load_dataset()
        check_missing_values(df)
