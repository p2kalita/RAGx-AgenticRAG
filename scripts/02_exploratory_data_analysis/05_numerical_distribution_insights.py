"""
Three numerical variables were analysed to understand the structural characteristics of the curated corpus. Enterprise presales questions contain an average of 10.6 words, with question lengths ranging from 4 to 40 words, indicating considerable variation in the complexity and specificity of user queries. Most questions fall between 7 and 13 words, suggesting that the corpus primarily consists of concise but information-rich enterprise questions.

The average number of technologies referenced within a single question is 2.74, with some questions mentioning up to 12 technologies. This observation reinforces the multi-technology nature of enterprise presales, where responses frequently require reasoning across multiple products, services or architectural components rather than focusing on a single technology.

The Estimated RFP Relevance score has a mean value of 4.87 out of 5, with more than half of the observations receiving the maximum rating. This confirms that the curated corpus predominantly contains questions that are highly representative of enterprise presales activities, making it suitable for evaluating AI-assisted knowledge orchestration
"""

from typing import Optional
import pandas as pd


def review_numerical_distribution_insights(df: Optional[pd.DataFrame] = None) -> None:
    """
    Review structural characteristics of numerical distributions within the curated corpus.

    Parameters
    ----------
    df : Optional[pd.DataFrame], optional
        The enterprise presales DataFrame to evaluate against numerical benchmarks.
    """
    print("=" * 80)
    print("NUMERICAL DISTRIBUTION INSIGHTS & STRUCTURAL CHARACTERISTICS")
    print("=" * 80)
    print("- Question Length: Mean 10.6 words (range 4 to 40 words, IQR 7 to 13 words).")
    print("- Technologies Mentioned: Mean 2.74 technologies (max 12), multi-technology reasoning.")
    print("- RFP Relevance: Mean 4.87 / 5.0, highly representative of presales requirements.")

    if df is not None:
        print("\nEmpirical Metrics from Loaded Dataset:")
        if "Question Length" in df.columns:
            mean_len = df["Question Length"].mean()
            min_len = df["Question Length"].min()
            max_len = df["Question Length"].max()
            print(f"  Question Length: Mean = {mean_len:.2f}, Min = {min_len}, Max = {max_len}")
        if "Number of Technologies Mentioned" in df.columns:
            mean_tech = df["Number of Technologies Mentioned"].mean()
            max_tech = df["Number of Technologies Mentioned"].max()
            print(f"  Technologies Mentioned: Mean = {mean_tech:.2f}, Max = {max_tech}")
        if "Estimated RFP Relevance (1-5)" in df.columns:
            mean_rfp = df["Estimated RFP Relevance (1-5)"].mean()
            print(f"  Estimated RFP Relevance Mean: {mean_rfp:.2f}")


if __name__ == "__main__":
    review_numerical_distribution_insights()

