"""
## 1. Dataset loading and inspection - Initial Inspection

This module performs initial inspection routines on the loaded enterprise presales dataset,
evaluating shapes, schemas, sample rows, descriptive statistics, uniqueness, null values,
and duplicate rows.
"""

import pandas as pd


def inspect_dataset(df: pd.DataFrame) -> None:
    """
    Perform baseline inspection operations on the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The enterprise presales DataFrame to inspect.
    """
    print(df.shape)
    print(df.info())

    print(df.head())

    print(df.describe(include="all"))

    print(df.nunique())

    print(df.isnull().sum())

    print(df.duplicated().sum())


if __name__ == "__main__":
    import os
    import sys
    import importlib.util

    current_dir = os.path.dirname(os.path.abspath(__file__))
    loader_path = os.path.join(current_dir, "02_dataset_loading.py")
    if os.path.exists(loader_path):
        spec = importlib.util.spec_from_file_location("step_02_dataset_loading", loader_path)
        module_02 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module_02)
        df = module_02.load_dataset()
        inspect_dataset(df)
