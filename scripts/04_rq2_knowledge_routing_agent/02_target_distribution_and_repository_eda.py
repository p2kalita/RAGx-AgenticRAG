"""
# PART-C: RQ2 – Knowledge Routing Agent

## 1. Inferential Statistics (RQ2 specific)

The dataset contains four repository classes representing the enterprise knowledge required to answer a question. Approximately 46% of questions require Architecture Blueprints and Methodologies in addition to public documentation, followed by Governance (21%) and Pricing (17%). Only 16% of questions can be answered using public documentation alone. This indicates that the majority of enterprise presales questions require internal organizational knowledge beyond publicly available information.

From the above, we note that "Architecture" questions predominantly require Architecture Blueprints and SME expertise, while Governance questions are largely associated with Governance Frameworks and Risk Registers. Commercial questions exhibit stronger associations with Pricing Templates. These initial observations suggest that repository selection is strongly influenced by business context.
"""

###############################################################
# RQ2 : TARGET VARIABLE DISTRIBUTION & QUESTION TYPE vs REPOSITORY
###############################################################

from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_target_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute count and percentage distribution of Enterprise Knowledge Required and display countplot.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing 'Enterprise Knowledge Required'.

    Returns
    -------
    pd.DataFrame
        Distribution DataFrame with Count and Percentage.
    """
    print("=" * 80)
    print("Enterprise Knowledge Repository Distribution")
    print("=" * 80)

    repo_counts = df["Enterprise Knowledge Required"].value_counts()
    repo_percent = round(
        df["Enterprise Knowledge Required"].value_counts(normalize=True) * 100,
        2
    )

    distribution = pd.DataFrame({
        "Count": repo_counts,
        "Percentage": repo_percent
    })

    try:
        display(distribution)
    except NameError:
        print(distribution)

    plt.figure(figsize=(10, 6))
    sns.countplot(
        data=df,
        y="Enterprise Knowledge Required",
        order=repo_counts.index
    )
    plt.title("Distribution of Enterprise Knowledge Repository")
    plt.xlabel("Number of Questions")
    plt.ylabel("Repository")
    plt.tight_layout()
    plt.show()

    return distribution


def analyze_question_type_vs_repository(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate cross-tabulation and heatmap of Question Type vs Enterprise Knowledge Required.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing 'Question Type' and 'Enterprise Knowledge Required'.

    Returns
    -------
    pd.DataFrame
        Cross-tabulation DataFrame.
    """
    cross_tab = pd.crosstab(
        df["Question Type"],
        df["Enterprise Knowledge Required"]
    )

    try:
        display(cross_tab)
    except NameError:
        print(cross_tab)

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        cross_tab,
        annot=True,
        fmt="d",
        cmap="Blues"
    )
    plt.title("Question Type vs Enterprise Repository")
    plt.tight_layout()
    plt.show()

    return cross_tab


def run_repository_eda(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute full EDA for repository distribution and cross-tabulation.
    """
    dist = analyze_target_distribution(df)
    crosstab = analyze_question_type_vs_repository(df)
    return dist, crosstab


if __name__ == "__main__":
    import os
    import importlib.util

    current_dir = os.path.dirname(os.path.abspath(__file__))
    loader_path = os.path.join(current_dir, "01_environment_and_data_loading.py")

    if os.path.exists(loader_path):
        spec_load = importlib.util.spec_from_file_location("step_01_loader", loader_path)
        mod_load = importlib.util.module_from_spec(spec_load)
        spec_load.loader.exec_module(mod_load)
        df = mod_load.load_cleaned_dataset()
        run_repository_eda(df)

