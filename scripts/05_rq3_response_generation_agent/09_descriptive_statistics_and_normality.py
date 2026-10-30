"""
Descriptive Statistics: Summarize the performance of each response generation strategy using mean, standard deviation, median and interquartile range (IQR).

Boxplot: Compare the distribution of Response Quality Index (RQI) across the four response generation systems.

Assumption Testing - Determine whether parametric statistical tests are appropriate:
1. Normality Test (Shapiro-Wilk)
"""

###############################################################
# RQ3 - STEP 24, 25, 26 : DESCRIPTIVE STATISTICS, BOXPLOT & NORMALITY
###############################################################

from typing import List, Optional
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro

DEFAULT_SCORE_COLUMNS = ['Baseline', 'GroundTruth', 'Predicted', 'AllRepositories']


def compute_descriptive_statistics(
    analysis_df: pd.DataFrame,
    score_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generate descriptive statistics summary with Mean, Std, Median, and IQR.
    """
    cols = score_columns if score_columns is not None else DEFAULT_SCORE_COLUMNS
    summary = analysis_df[cols].describe().T

    summary["Median"] = analysis_df[cols].median()
    summary["IQR"] = (
        analysis_df[cols].quantile(0.75) - analysis_df[cols].quantile(0.25)
    )

    try:
        display(summary)
    except NameError:
        print(summary)

    return summary


def plot_rqi_boxplot(
    analysis_df: pd.DataFrame,
    score_columns: Optional[List[str]] = None,
    figsize: tuple = (9, 6)
) -> None:
    """
    Render boxplot comparing the distribution of Response Quality Index (RQI) across systems.
    """
    cols = score_columns if score_columns is not None else DEFAULT_SCORE_COLUMNS
    plt.figure(figsize=figsize)
    analysis_df[cols].boxplot()
    plt.ylabel("Response Quality Index")
    plt.title("Comparison of Response Quality")
    plt.tight_layout()
    plt.show()


def test_normality_assumptions(
    analysis_df: pd.DataFrame,
    score_columns: Optional[List[str]] = None
) -> dict:
    """
    Perform Shapiro-Wilk test of normality across all response generation strategies.
    """
    cols = score_columns if score_columns is not None else DEFAULT_SCORE_COLUMNS
    results = {}

    for col in cols:
        stat, p = shapiro(analysis_df[col])
        print("=" * 60)
        print(col)
        print("Statistic:", round(stat, 3))
        print("p-value:", round(p, 4))
        if p > 0.05:
            conclusion = "Normally Distributed"
        else:
            conclusion = "Not Normally Distributed"
        print(conclusion)

        results[col] = {
            "Statistic": round(stat, 3),
            "p-value": round(p, 4),
            "Conclusion": conclusion
        }

    return results


def run_descriptive_and_normality_checks(analysis_df: pd.DataFrame) -> tuple:
    """
    Execute full descriptive statistics and normality assumption test suite.
    """
    summary = compute_descriptive_statistics(analysis_df)
    plot_rqi_boxplot(analysis_df)
    normality = test_normality_assumptions(analysis_df)
    return summary, normality


if __name__ == "__main__":
    print("Descriptive statistics and normality module ready.")

