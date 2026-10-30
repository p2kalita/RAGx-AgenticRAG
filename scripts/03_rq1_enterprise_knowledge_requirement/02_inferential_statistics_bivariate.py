"""
#PART-B: RQ1

## 1. Inferential Statistics (RQ1 specific)

Objective: This section explores the relationship between the characteristics of enterprise presales questions and whether those questions can be answered using publicly available knowledge alone versus requiring proprietary enterprise knowledge.
"""

###############################################################
# RQ EXPLORATORY ANALYSIS FUNCTION
#
# Performs:
# 1. Frequency Table
# 2. Percentage Table
# 3. Stacked Bar Chart
# 4. Chi-square Test
# 5. Cramer's V
# 6. Statistical Interpretation
###############################################################

from typing import List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import numpy as np

# Default target and predictors for RQ1
DEFAULT_TARGET = "Public Knowledge Only?"
DEFAULT_PREDICTORS = [
    "Question Type",
    "Question Complexity",
    "Technology Domain",
    "Business Capability",
    "Question Intent"
]


def analyze_predictor_vs_target(df, predictor, target, summary_results=None):
    """
    Performs bivariate inferential analysis of a candidate predictor against the target.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing predictor and target.
    predictor : str
        Predictor column name.
    target : str
        Target column name.
    summary_results : list, optional
        Accumulator list for storing test statistics and effect sizes.
    """
    print("\n" + "="*80)
    print(f"Analyzing: {predictor} vs {target}")
    print("="*80)

    try:
        # ----------------------------------------------------------
        # Cross Tabulation
        # ----------------------------------------------------------

        contingency = pd.crosstab(df[predictor], df[target])

        print("\nContingency Table\n")
        try:
            display(contingency)
        except NameError:
            print(contingency)

        # ----------------------------------------------------------
        # Row Percentages
        # ----------------------------------------------------------

        percentage = pd.crosstab(
            df[predictor],
            df[target],
            normalize='index'
        ) * 100

        print("\nPercentage Distribution\n")
        try:
            display(percentage.round(2))
        except NameError:
            print(percentage.round(2))

        # ----------------------------------------------------------
        # Stacked Bar Chart
        # ----------------------------------------------------------

        ax = percentage.plot(
            kind='bar',
            stacked=True,
            figsize=(10,6)
        )

        plt.title(f"{predictor} vs {target}")

        plt.ylabel("Percentage")

        plt.xlabel(predictor)

        plt.xticks(rotation=30)

        plt.legend(title=target)

        plt.tight_layout()

        plt.show()

        # ----------------------------------------------------------
        # Chi-square Test
        # ----------------------------------------------------------

        chi2, p, dof, expected = chi2_contingency(contingency)

        print("\n")
        print("="*60)
        print("Chi-square Test")
        print("="*60)

        print(f"Chi-square Statistic : {chi2:.3f}")

        print(f"Degrees of Freedom   : {dof}")

        print(f"P-value              : {p:.5f}")

        # ----------------------------------------------------------
        # Cramer's V
        # ----------------------------------------------------------

        n = contingency.sum().sum()

        # Ensure min_dim is not zero to prevent ZeroDivisionError
        min_dim = min(contingency.shape) - 1
        if min_dim == 0:
            cramers_v = np.nan # Or handle as appropriate, e.g., 0 or skip
            print("Cramer's V: Cannot be calculated (min_dim is 0).")
        else:
            cramers_v = np.sqrt(chi2/(n*min_dim))
            print(f"Cramer's V           : {cramers_v:.3f}")

        # ----------------------------------------------------------
        # Interpretation
        # ----------------------------------------------------------

        print("\n")
        print("="*60)
        print("Interpretation")
        print("="*60)

        if p < 0.05:
            print("✓ Reject the Null Hypothesis")
            print("There is a statistically significant association.")

        else:
            print("✓ Fail to Reject the Null Hypothesis")
            print("No statistically significant association detected.")

        if pd.isna(cramers_v) or cramers_v < 0.10:
            strength = "Negligible"

        elif cramers_v < 0.30:
            strength = "Weak"

        elif cramers_v < 0.50:
            strength = "Moderate"

        else:
            strength = "Strong"

        print(f"Association Strength : {strength}")

        print("\nSuggested for Logistic Regression:")

        if p < 0.05:
            print("YES")

        else:
            print("Consider excluding this predictor.")

        # ----------------------------------------------------------
        # Store Results
        # ----------------------------------------------------------

        if pd.isna(cramers_v) or cramers_v < 0.10:
            effect = "Negligible"
        elif cramers_v < 0.30:
            effect = "Weak"
        elif cramers_v < 0.50:
            effect = "Moderate"
        else:
            effect = "Strong"

        if summary_results is not None:
            summary_results.append({

                "Predictor": predictor,

                "Chi-square": round(chi2,3),

                "Degrees of Freedom": dof,

                "P-value": round(p,5),

                "Cramer's V": round(cramers_v,3) if not pd.isna(cramers_v) else cramers_v,

                "Effect Size": effect,

                "Statistically Significant":
                    "Yes" if p < 0.05 else "No",

                "Include in Logistic Regression":
                    "Yes" if p < 0.05 else "No"

            })
    except Exception as e:
        print(f"An error occurred while processing {predictor}: {e}")
        print("Skipping this predictor and continuing with the next ones if any.")


def run_rq1_predictor_analysis(
    df: pd.DataFrame,
    predictors: Optional[List[str]] = None,
    target: str = DEFAULT_TARGET
) -> list:
    """
    Run inferential analysis across all candidate predictors for RQ1.

    Parameters
    ----------
    df : pd.DataFrame
        Enterprise presales dataset.
    predictors : Optional[List[str]], optional
        List of candidate predictor column names.
    target : str, default=DEFAULT_TARGET
        Target variable name.

    Returns
    -------
    list
        List of dictionary records containing statistical results.
    """
    pred_cols = predictors if predictors is not None else DEFAULT_PREDICTORS
    summary_results = []
    for predictor in pred_cols:
        analyze_predictor_vs_target(df, predictor, target, summary_results)
    return summary_results


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
        summary_results = run_rq1_predictor_analysis(df)

