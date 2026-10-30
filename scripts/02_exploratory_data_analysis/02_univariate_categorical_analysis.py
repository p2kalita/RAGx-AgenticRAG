"""
#PART-A : EDA

## 4. Exploratory Data Analysis

### 4.1 Univariate Analysis
"""

###############################################################
# EXPLORATORY DATA ANALYSIS (EDA)
#
# Objective:
# Understand the distribution of categorical variables before
# performing statistical analysis and machine learning.
#
# The following variables are analysed:
#   - Vendor
#   - Technology Domain
#   - Cloud Platform
#   - Business Capability
#   - Question Type
#   - Question Intent
#   - Question Complexity
#   - Public Knowledge Only?
#   - Enterprise Knowledge Required
###############################################################

from typing import List, Optional
import pandas as pd
import matplotlib.pyplot as plt

# List of important categorical variables
DEFAULT_EDA_COLUMNS = [
    'Vendor',
    'Technology Domain',
    'Cloud Platform',
    'Business Capability',
    'Question Type',
    'Question Intent',
    'Question Complexity',
    'Public Knowledge Only?',
    'Enterprise Knowledge Required'
]


def analyze_categorical_variable(df: pd.DataFrame, col: str) -> None:
    """
    Compute frequency table, percentage distribution, and render bar chart for a single categorical variable.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the column.
    col : str
        Column name to analyze.
    """
    print("\n" + "="*70)
    print(f"Frequency Distribution : {col}")
    print("="*70)

    # Display frequency table
    print(df[col].value_counts())

    # Display percentage distribution
    print("\nPercentage Distribution")
    print(round(df[col].value_counts(normalize=True)*100, 2))

    # Plot bar chart
    plt.figure(figsize=(10, 5))

    df[col].value_counts().plot(kind='bar')

    plt.title(f'Distribution of {col}')

    plt.ylabel('Number of Records')

    plt.xlabel(col)

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.show()


def run_univariate_categorical_analysis(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> None:
    """
    Loop through each categorical variable to perform univariate analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze.
    columns : Optional[List[str]], optional
        List of categorical columns to analyze. Defaults to DEFAULT_EDA_COLUMNS.
    """
    cols_to_analyze = columns if columns is not None else DEFAULT_EDA_COLUMNS
    for col in cols_to_analyze:
        analyze_categorical_variable(df, col)


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
        run_univariate_categorical_analysis(df)

