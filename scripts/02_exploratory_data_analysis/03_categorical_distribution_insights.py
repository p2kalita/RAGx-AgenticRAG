"""
Frequency analysis was performed to understand the composition of the Curated Enterprise Presales Research Corpus (CEPRC) and assess whether the data adequately represents the research problem. The analysis shows that the corpus is intentionally centred on enterprise cloud technologies, with AWS contributing 820 of the 900 observations (91.1%), reflecting the greater availability of high-quality public documentation and enterprise technical discussions for AWS compared to other vendors. Smaller representations from Databricks, IBM, Cisco, Google Cloud, Salesforce, Snowflake and Microsoft were retained to introduce diversity across technology platforms rather than to achieve equal vendor representation.

The distribution of technology domains indicates that the corpus is dominated by Security/Governance (41.7%) and Networking (23.0%), followed by Generative AI (11.8%) and Analytics/Data Platform (8.4%). These distributions closely mirror the business capability categories because the annotation framework intentionally maps each technology domain to a corresponding enterprise business capability.

Analysis of question characteristics shows that Architecture (35.2%), Commercial/Pricing (19.3%), and Best Practice (15.0%) questions collectively account for nearly 70% of the corpus. Similarly, most questions seek presales guidance (40.4%) or implementation approaches (24.4%), indicating that the dataset primarily represents practical decision-making scenarios encountered during enterprise solution design.

Question complexity is intentionally skewed towards High complexity (76.7%), with only 5.2% classified as Low complexity. This reflects the research objective of evaluating AI-assisted enterprise presales, where questions frequently require contextual reasoning and the integration of multiple enterprise knowledge assets rather than simple factual retrieval.

The target variable Public Knowledge Only? shows that only 16.2% of questions can be answered using publicly available documentation alone, while 83.8% require additional enterprise knowledge. Likewise, the Enterprise Knowledge Required variable demonstrates that almost half of the corpus (45.7%) requires a combination of public documentation, architecture blueprints, implementation methodologies and subject matter expertise. These findings provide preliminary evidence that enterprise AI systems require effective knowledge orchestration beyond conventional Retrieval-Augmented Generation (RAG) to support presales activities.
"""

from typing import Optional
import pandas as pd


def review_categorical_distribution_insights(df: Optional[pd.DataFrame] = None) -> None:
    """
    Review and verify the empirical frequency findings and structural proportions of the corpus.

    Parameters
    ----------
    df : Optional[pd.DataFrame], optional
        The enterprise presales DataFrame to evaluate against the corpus findings.
    """
    print("=" * 80)
    print("CATEGORICAL DISTRIBUTION INSIGHTS & CORPUS COMPOSITION")
    print("=" * 80)
    print("- Vendor Diversity: Centred on enterprise cloud (AWS: 91.1%, plus 7 diverse platforms).")
    print("- Technology Domains: Dominated by Security/Governance (41.7%) and Networking (23.0%).")
    print("- Question Characteristics: Architecture (35.2%), Commercial/Pricing (19.3%), Best Practice (15.0%).")
    print("- Question Complexity: High complexity (76.7%), reflecting contextual reasoning requirements.")
    print("- Knowledge Boundaries: 83.8% require enterprise knowledge beyond public documentation.")
    print("- Orchestration Need: 45.7% require multi-asset combination (public docs, blueprints, methodologies, SME).")

    if df is not None:
        print("\nEmpirical Proportions from Loaded Dataset:")
        if "Vendor" in df.columns:
            aws_pct = (df["Vendor"] == "AWS").mean() * 100
            print(f"  AWS Proportion: {aws_pct:.1f}%")
        if "Technology Domain" in df.columns:
            sec_pct = (df["Technology Domain"] == "Security/Governance").mean() * 100
            net_pct = (df["Technology Domain"] == "Networking").mean() * 100
            print(f"  Security/Governance Proportion: {sec_pct:.1f}%")
            print(f"  Networking Proportion: {net_pct:.1f}%")
        if "Question Complexity" in df.columns:
            high_pct = (df["Question Complexity"] == "High").mean() * 100
            print(f"  High Complexity Proportion: {high_pct:.1f}%")
        if "Public Knowledge Only?" in df.columns:
            enterprise_req_pct = (df["Public Knowledge Only?"] != "Yes").mean() * 100
            print(f"  Requires Enterprise Knowledge: {enterprise_req_pct:.1f}%")


if __name__ == "__main__":
    review_categorical_distribution_insights()

