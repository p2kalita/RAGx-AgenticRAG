"""
## 2. Dataset Quality Assessment

Before beginning exploratory analysis, the dataset is examined for completeness, duplicate records, data types, missing values, and the number of unique categories in each variable. This assessment helps identify potential data quality issues that may affect subsequent statistical analysis and machine learning models.
"""

###############################################################
# DATA QUALITY ASSESSMENT
#
# This section evaluates the overall quality of the dataset.
# The following checks are performed:
#
# 1. Dataset dimensions
# 2. Data types
# 3. Missing values
# 4. Duplicate records
# 5. Unique values per variable
###############################################################

import pandas as pd


def check_dataset_dimensions(df: pd.DataFrame) -> tuple:
    """
    Display and return the number of rows and columns in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to evaluate.

    Returns
    -------
    tuple
        (rows, columns)
    """
    print("=" * 60)
    print("DATASET DIMENSIONS")
    print("=" * 60)
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")
    return df.shape[0], df.shape[1]


def check_column_data_types(df: pd.DataFrame) -> pd.Series:
    """
    Display and return the data types of all columns in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to evaluate.

    Returns
    -------
    pd.Series
        Series containing column data types.
    """
    print("\n")
    print("=" * 60)
    print("COLUMN DATA TYPES")
    print("=" * 60)
    print(df.dtypes)
    return df.dtypes


def run_dimensions_and_types_check(df: pd.DataFrame) -> None:
    """
    Execute dimension and data type checks.
    """
    check_dataset_dimensions(df)
    check_column_data_types(df)


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
        run_dimensions_and_types_check(df)
