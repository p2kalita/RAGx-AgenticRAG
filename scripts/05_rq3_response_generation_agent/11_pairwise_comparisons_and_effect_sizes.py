"""
Pairwise Comparisons - Identify which response generation strategies differ significantly from one another.

Effect Size (Cohen's d) - Quantify the practical significance of observed differences between methods.
"""

###############################################################
# RQ3 - STEP 29 & 30 : PAIRED T-TESTS & COHEN'S D EFFECT SIZE
###############################################################

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

DEFAULT_STRATEGY_PAIRS = [
    ("Baseline", "GroundTruth"),
    ("Baseline", "Predicted"),
    ("Baseline", "AllRepositories"),
    ("GroundTruth", "Predicted"),
    ("GroundTruth", "AllRepositories"),
    ("Predicted", "AllRepositories")
]


def cohens_d(x, y):
    """
    Compute paired Cohen's d effect size: mean(x - y) / std(x - y).
    """
    diff = x - y
    return diff.mean() / diff.std()


def run_paired_t_tests(
    analysis_df: pd.DataFrame,
    pairs: Optional[List[Tuple[str, str]]] = None
) -> List[Dict[str, Any]]:
    """
    Execute paired t-tests across specified strategy pairs.
    """
    comparison_pairs = pairs if pairs is not None else DEFAULT_STRATEGY_PAIRS
    results = []

    print("=" * 80)
    print("PAIRED T-TESTS")
    print("=" * 80)

    for a, b in comparison_pairs:
        stat, p = ttest_rel(
            analysis_df[a],
            analysis_df[b]
        )
        print("=" * 60)
        print(a, "vs", b)
        print("t =", round(stat, 3))
        print("p =", round(p, 4))

        results.append({
            "Comparison": f"{a} vs {b}",
            "t_stat": round(stat, 3),
            "p_value": round(p, 4)
        })

    return results


def run_effect_size_analysis(
    analysis_df: pd.DataFrame,
    pairs: Optional[List[Tuple[str, str]]] = None
) -> List[Dict[str, Any]]:
    """
    Compute Cohen's d effect size across all strategy pairs.
    """
    comparison_pairs = pairs if pairs is not None else DEFAULT_STRATEGY_PAIRS
    results = []

    print("\n" + "=" * 80)
    print("COHEN'S D EFFECT SIZES")
    print("=" * 80)

    for a, b in comparison_pairs:
        d = cohens_d(
            analysis_df[a],
            analysis_df[b]
        )
        print(a, "vs", b, " Cohen's d =", round(d, 3))

        results.append({
            "Comparison": f"{a} vs {b}",
            "cohens_d": round(d, 3)
        })

    return results


def run_pairwise_and_effect_size_pipeline(
    analysis_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute both paired t-tests and Cohen's d effect size analyses.
    """
    t_tests = run_paired_t_tests(analysis_df)
    effect_sizes = run_effect_size_analysis(analysis_df)
    return pd.DataFrame(t_tests), pd.DataFrame(effect_sizes)


if __name__ == "__main__":
    print("Pairwise comparisons and effect sizes module ready.")

