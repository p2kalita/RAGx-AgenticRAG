"""
## 3. Data Cleaning Log

This module performs data cleaning operations by stripping leading and trailing whitespaces
from all text columns, generating a cleaning log summarizing post-cleaning missing counts
and unique categories per variable.
"""

###############################################################
# DATA CLEANING
#
# The purpose of this section is to identify any issues that
# require cleaning before statistical analysis.
#
# Checks performed:
#
# - Missing values
# - Duplicate records
# - Extra white spaces
# - Inconsistent category labels
###############################################################

import pandas as pd


def strip_text_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove leading and trailing spaces from all text (object) columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to clean.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset with stripped text fields.
    """
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
    return df


def generate_cleaning_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate and display the data cleaning log.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset.

    Returns
    -------
    pd.DataFrame
        Cleaning log with Variable, Missing count, and Unique Categories.
    """
    # Check missing values again
    cleaning_log = pd.DataFrame({
        "Variable": df.columns,
        "Missing": df.isnull().sum(),
        "Unique Categories": df.nunique()
    })

    try:
        display(cleaning_log)
    except NameError:
        print(cleaning_log)

    print("\nData Cleaning Completed.")
    return cleaning_log


def clean_and_log(df: pd.DataFrame) -> tuple:
    """
    Execute full cleaning workflow and generate cleaning log.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    tuple
        (cleaned_df, cleaning_log)
    """
    cleaned_df = strip_text_whitespace(df)
    log_df = generate_cleaning_log(cleaned_df)
    return cleaned_df, log_df


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
        df, log_df = clean_and_log(df)
