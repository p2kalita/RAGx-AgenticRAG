"""
The above initial quality assessment indicates that the dataset is highly complete. All core research variables contain 900 non-null observations, while Publication Date, Accepted Answer Available?, and Accepted Answer URL contain no values because these fields were intentionally excluded from the current corpus. No duplicate observations were identified, confirming that each record represents a unique enterprise presales question–answer pair

Initial Observations

A few notable characteristics of the corpus emerge from the summary statistics:

- The dataset covers 8 technology vendors and 8 source websites, providing diversity across enterprise platforms.
- Question Type and Question Intent each contain 7 categories, aligning with the proposed inferential analyses for RQ2 and RQ4.
- The target variable Public Knowledge Only? contains two classes (Yes and Partially), confirming that Binary Logistic Regression is the appropriate statistical model for RQ1.
- Enterprise Knowledge Required contains 4 categories, supporting the planned Chi-square analyses.
- The variable Estimated RFP Relevance (1–5) has a mean score of 4.87, suggesting that the curated corpus primarily consists of questions that are highly relevant to enterprise presales.
"""

from typing import Optional
import pandas as pd


def review_initial_observations(df: Optional[pd.DataFrame] = None) -> None:
    """
    Review initial observations and notable characteristics of the corpus based on summary statistics.

    Parameters
    ----------
    df : Optional[pd.DataFrame], optional
        The enterprise presales DataFrame to evaluate against the initial observations.
    """
    observations = [
        "The dataset covers 8 technology vendors and 8 source websites, providing diversity across enterprise platforms.",
        "Question Type and Question Intent each contain 7 categories, aligning with the proposed inferential analyses for RQ2 and RQ4.",
        "The target variable Public Knowledge Only? contains two classes (Yes and Partially), confirming that Binary Logistic Regression is the appropriate statistical model for RQ1.",
        "Enterprise Knowledge Required contains 4 categories, supporting the planned Chi-square analyses.",
        "The variable Estimated RFP Relevance (1–5) has a mean score of 4.87, suggesting that the curated corpus primarily consists of questions that are highly relevant to enterprise presales.",
    ]

    print("=" * 60)
    print("INITIAL OBSERVATIONS")
    print("=" * 60)
    for obs in observations:
        print(f"- {obs}")

    if df is not None:
        print("\nVerifying corpus characteristics against loaded dataset:")
        if "Technology Vendor" in df.columns:
            print(f"Technology Vendors count: {df['Technology Vendor'].nunique()}")
        if "Source Website" in df.columns:
            print(f"Source Websites count: {df['Source Website'].nunique()}")
        if "Question Type" in df.columns:
            print(f"Question Type categories: {df['Question Type'].nunique()}")
        if "Question Intent" in df.columns:
            print(f"Question Intent categories: {df['Question Intent'].nunique()}")
        if "Public Knowledge Only?" in df.columns:
            print(f"Public Knowledge Only? classes: {df['Public Knowledge Only?'].unique().tolist()}")
        if "Enterprise Knowledge Required" in df.columns:
            print(f"Enterprise Knowledge Required categories: {df['Enterprise Knowledge Required'].nunique()}")
        if "Estimated RFP Relevance (1–5)" in df.columns:
            print(f"Estimated RFP Relevance (1–5) mean: {df['Estimated RFP Relevance (1–5)'].mean():.2f}")


if __name__ == "__main__":
    review_initial_observations()
