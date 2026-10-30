"""
## 2. Dataset Quality Assessment - Duplicate Records

This module evaluates the presence of duplicate rows in the Curated Enterprise Presales Research Corpus.
"""

import pandas as pd


def check_duplicate_records(df: pd.DataFrame) -> int:
    """
    Compute and display the number of duplicate rows in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to evaluate.

    Returns
    -------
    int
        Count of duplicate rows.
    """
    print("\n")
    print("=" * 60)
    print("DUPLICATE RECORDS")
    print("=" * 60)

    duplicates = df.duplicated().sum()

    print(f"Duplicate Rows : {duplicates}")
    return int(duplicates)


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
        check_duplicate_records(df)
