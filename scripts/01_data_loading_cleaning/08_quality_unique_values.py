"""
## 2. Dataset Quality Assessment - Unique Values per Column

The dataset was manually curated and annotated prior to analysis. Consequently, only basic quality checks such as missing values, duplicate records, and consistency of categorical labels are performed. Any identified issues are documented before proceeding with the exploratory analysis.
"""

import pandas as pd


def check_unique_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and display the count of unique values per column.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to evaluate.

    Returns
    -------
    pd.DataFrame
        DataFrame containing 'Unique Values' count per column.
    """
    print("\n")
    print("=" * 60)
    print("UNIQUE VALUES PER COLUMN")
    print("=" * 60)

    unique_values = pd.DataFrame({
        'Unique Values': df.nunique()
    })

    print(unique_values)
    return unique_values


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
        check_unique_values(df)
