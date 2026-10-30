"""
Repeated Measures ANOVA: Compare mean response quality across the four response generation strategies.

Friedman Test: Non-parametric comparison of the four response generation strategies.
"""

###############################################################
# RQ3 - STEP 27 & 28 : REPEATED MEASURES ANOVA & FRIEDMAN TEST
###############################################################

from typing import Tuple, Any
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from scipy.stats import friedmanchisquare


def run_repeated_measures_anova(analysis_df: pd.DataFrame) -> Any:
    """
    Perform Repeated Measures ANOVA across methods using Question as subject.

    Parameters
    ----------
    analysis_df : pd.DataFrame
        DataFrame with Question and the 4 method columns.

    Returns
    -------
    Any
        Fitted AnovaRM results.
    """
    print("=" * 80)
    print("REPEATED MEASURES ANOVA")
    print("=" * 80)

    anova_df = analysis_df.melt(
        id_vars="Question",
        var_name="Method",
        value_name="RQI"
    )

    anova = AnovaRM(
        anova_df,
        depvar="RQI",
        subject="Question",
        within=["Method"]
    )

    results = anova.fit()
    print(results)
    return results


def run_friedman_test(analysis_df: pd.DataFrame) -> Tuple[float, float]:
    """
    Perform non-parametric Friedman test across the four response generation strategies.

    Returns
    -------
    Tuple[float, float]
        (Chi-square statistic, p-value)
    """
    print("\n" + "=" * 80)
    print("FRIEDMAN TEST (NON-PARAMETRIC)")
    print("=" * 80)

    stat, p = friedmanchisquare(
        analysis_df["Baseline"],
        analysis_df["GroundTruth"],
        analysis_df["Predicted"],
        analysis_df["AllRepositories"]
    )

    print("Statistic:", stat)
    print("p-value:", p)
    return stat, p


def run_hypothesis_tests(analysis_df: pd.DataFrame) -> Tuple[Any, Tuple[float, float]]:
    """
    Execute both parametric (ANOVA) and non-parametric (Friedman) omni-bus hypothesis tests.
    """
    anova_res = run_repeated_measures_anova(analysis_df)
    friedman_res = run_friedman_test(analysis_df)
    return anova_res, friedman_res


if __name__ == "__main__":
    print("Repeated Measures ANOVA and Friedman test module ready.")

