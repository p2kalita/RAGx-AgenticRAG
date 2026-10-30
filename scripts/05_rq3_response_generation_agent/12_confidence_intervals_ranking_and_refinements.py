"""
Confidence Intervals - Estimate uncertainty around the mean Response Quality Index for each strategy.

Mean Comparison Plot: Visualize average RQI with 95% confidence intervals.

Ranking of Methods: Rank all response generation strategies based on mean Response Quality Index (RQI).

Pending refinement: For the evaluation set, I adont want to use the first 100 questions (head(100)), because those are heavily AWS-centric and may bias the experiment. Instead, I want to create a stratified sample of 100 questions that preserves the distribution of Question Type, Technology Domain, and Enterprise Knowledge Required. This will make my evaluation more representative and strengthen the validity of my conclusions in the dissertation.

Additional Refinement: Rather than using only the overall score from the LLM Judge, we can keep all five dimensions:

Accuracy
Completeness
Enterprise Relevance
Technical Correctness
Actionability

Then perform the same statistical analysis for each dimension, in addition to the overall RQI. This lets me  make much richer claims, such as:

Knowledge routing significantly improved enterprise relevance (p < 0.01) but had little effect on technical correctness.
Ground-truth routing produced the highest completeness scores.
Searching all repositories increased completeness but reduced actionability due to irrelevant context.

More refinments:
- Redesign the baseline to use only the product knowledge repository instead of GPT's internal knowledge.
- Improve the LLM Judge so it scores each criterion separately (accuracy, completeness, enterprise relevance, technical correctness, actionability, groundedness) and computes RQI deterministically.
- Add RQI improvement by Question Complexity and by Question Type to identify where enterprise knowledge delivers the most value.
Compare correctly routed vs. incorrectly routed questions using RQ2 predictions.
- Add a win-rate table (which method achieved the highest RQI per question) and a critical difference or violin plot for a clearer visual comparison.
"""

###############################################################
# RQ3 - STEP 31, 32, 33 : CONFIDENCE INTERVALS, RANKING & PLOT
###############################################################

from typing import List, Optional, Dict, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

DEFAULT_SCORE_COLUMNS = ['Baseline', 'GroundTruth', 'Predicted', 'AllRepositories']


def compute_confidence_intervals(
    analysis_df: pd.DataFrame,
    score_columns: Optional[List[str]] = None,
    confidence: float = 0.95
) -> Dict[str, Tuple[float, float]]:
    """
    Compute 95% Confidence Intervals for mean RQI for each strategy.
    """
    cols = score_columns if score_columns is not None else DEFAULT_SCORE_COLUMNS
    ci_dict = {}

    print("=" * 80)
    print("95% CONFIDENCE INTERVALS")
    print("=" * 80)

    for col in cols:
        mean = analysis_df[col].mean()
        sem = analysis_df[col].sem()
        ci = t.interval(
            confidence=confidence,
            df=len(analysis_df) - 1,
            loc=mean,
            scale=sem
        )
        print("=" * 60)
        print(col)
        print("95% CI:", ci)
        ci_dict[col] = ci

    return ci_dict


def plot_mean_comparison(
    analysis_df: pd.DataFrame,
    score_columns: Optional[List[str]] = None,
    figsize: tuple = (9, 6)
) -> Tuple[pd.Series, pd.Series]:
    """
    Render mean comparison bar chart with 95% error bars (sem * 1.96).
    """
    cols = score_columns if score_columns is not None else DEFAULT_SCORE_COLUMNS

    means = analysis_df[cols].mean()
    errors = analysis_df[cols].sem() * 1.96

    plt.figure(figsize=figsize)
    plt.bar(
        means.index,
        means.values,
        yerr=errors,
        capsize=8
    )
    plt.ylabel("Mean Response Quality")
    plt.title("Response Quality Comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()

    return means, errors


def rank_generation_methods(
    analysis_df: pd.DataFrame,
    score_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Rank response generation strategies based on Mean RQI descending.
    """
    cols = score_columns if score_columns is not None else DEFAULT_SCORE_COLUMNS
    means = analysis_df[cols].mean()

    ranking = pd.DataFrame({
        "Method": means.index,
        "Mean RQI": means.values
    })

    ranking = ranking.sort_values(
        "Mean RQI",
        ascending=False
    )

    try:
        display(ranking)
    except NameError:
        print(ranking)

    return ranking


def run_synthesis_and_ranking_pipeline(
    analysis_df: pd.DataFrame
) -> Tuple[Dict[str, Tuple[float, float]], pd.DataFrame]:
    """
    Execute full synthesis, plotting, and strategy ranking pipeline.
    """
    ci_results = compute_confidence_intervals(analysis_df)
    plot_mean_comparison(analysis_df)
    ranking_df = rank_generation_methods(analysis_df)
    return ci_results, ranking_df


if __name__ == "__main__":
    print("Confidence intervals, ranking, and synthesis module ready.")

