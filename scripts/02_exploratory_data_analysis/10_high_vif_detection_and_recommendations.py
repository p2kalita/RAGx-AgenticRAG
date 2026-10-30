"""
### 4.2. Bivariate Analysis - High VIF Detection, Recommendations & Modelling Synthesis

The Cramér's V association matrix indicates that most predictor pairs exhibit weak to moderate associations, suggesting that they capture distinct aspects of enterprise presales questions. The only exception is Technology Domain and Business Capability, which show a perfect association (Cramér's V = 1.000), indicating that these variables contain identical information and are effectively duplicate representations of the same underlying concept. Moderate associations were observed between Question Complexity and both Technology Domain and Business Capability (Cramér's V = 0.631), while all remaining associations were below 0.40, indicating acceptable independence among predictors.

The VIF analysis further confirms this finding. All dummy variables corresponding to Technology Domain and Business Capability exhibit infinite VIF values, indicating perfect multicollinearity arising from the one-to-one mapping between these two variables. Additionally, three categories of Question Intent show VIF values between 5 and 10, suggesting moderate multicollinearity, although this is expected because one-hot encoded categories of the same variable are naturally correlated. No concerning multicollinearity was observed for Question Type or Question Complexity.

Based on these results, Business Capability and Technology Domain should not be included together in the Binary Logistic Regression model. Since both variables convey identical information, retaining only one avoids redundancy without sacrificing explanatory power. Given that this study investigates enterprise presales knowledge orchestration, Business Capability is the more appropriate variable because it better reflects business-oriented knowledge requirements and aligns more closely with the research objectives. Consequently, the final regression model should include Question Type, Question Complexity, Question Intent, and Business Capability, while excluding Technology Domain.
"""

###############################################################
# Flag High VIF and Overall Recommendation
###############################################################

from typing import List, Optional
import pandas as pd


def flag_high_vif_variables(
    vif: pd.DataFrame,
    threshold: float = 5.0
) -> pd.DataFrame:
    """
    Filter and display variables with VIF exceeding the specified threshold.

    Parameters
    ----------
    vif : pd.DataFrame
        DataFrame with 'Variable' and 'VIF' columns.
    threshold : float, default=5.0
        VIF threshold to flag severe multicollinearity.

    Returns
    -------
    pd.DataFrame
        Filtered subset of variables with VIF > threshold.
    """
    print("\n")
    print("=" * 90)
    print("VARIABLES WITH HIGH MULTICOLLINEARITY")
    print("=" * 90)

    high_vif = vif[vif["VIF"] > threshold]

    if len(high_vif) == 0:
        print("No multicollinearity detected.")
    else:
        try:
            display(high_vif)
        except NameError:
            print(high_vif)

    return high_vif


def generate_modelling_recommendation(
    redundant_pairs: List[list],
    high_vif: pd.DataFrame
) -> str:
    """
    Evaluate redundancy and multicollinearity flags to generate actionable modelling recommendation.

    Parameters
    ----------
    redundant_pairs : List[list]
        List of highly associated predictor pairs from Cramér's V.
    high_vif : pd.DataFrame
        DataFrame of predictors with high VIF.

    Returns
    -------
    str
        Summary recommendation message.
    """
    print("\n")
    print("=" * 90)
    print("MODELLING RECOMMENDATION")
    print("=" * 90)

    if len(redundant_pairs) == 0 and len(high_vif) == 0:
        recommendation = "All predictors can be retained for Logistic Regression."
    elif len(redundant_pairs) > 0:
        recommendation = "Consider removing one variable from each highly associated pair."
    elif len(high_vif) > 0:
        recommendation = "Consider removing variables with high VIF values."
    else:
        recommendation = "Review predictor associations."

    print(recommendation)
    return recommendation


def evaluate_multicollinearity_and_recommendations(
    vif: pd.DataFrame,
    redundant_pairs: Optional[List[list]] = None,
    vif_threshold: float = 5.0
) -> tuple:
    """
    Execute complete multicollinearity evaluation and output recommendation.

    Parameters
    ----------
    vif : pd.DataFrame
        VIF table.
    redundant_pairs : Optional[List[list]], optional
        Redundant pairs list.
    vif_threshold : float, default=5.0
        VIF cutoff.

    Returns
    -------
    tuple
        (high_vif, recommendation)
    """
    pairs = redundant_pairs if redundant_pairs is not None else []
    high_vif = flag_high_vif_variables(vif, threshold=vif_threshold)
    recommendation = generate_modelling_recommendation(pairs, high_vif)
    return high_vif, recommendation


if __name__ == "__main__":
    import os
    import importlib.util

    current_dir = os.path.dirname(os.path.abspath(__file__))
    loader_path = os.path.join(current_dir, "01_environment_and_data_loading.py")
    matrix_path = os.path.join(current_dir, "06_bivariate_cramers_v_matrix.py")
    pairs_path = os.path.join(current_dir, "08_redundant_predictor_pairs.py")
    vif_path = os.path.join(current_dir, "09_multicollinearity_vif_analysis.py")

    if (os.path.exists(loader_path) and os.path.exists(matrix_path) and
            os.path.exists(pairs_path) and os.path.exists(vif_path)):
        spec_load = importlib.util.spec_from_file_location("step_01_loader", loader_path)
        mod_load = importlib.util.module_from_spec(spec_load)
        spec_load.loader.exec_module(mod_load)
        df = mod_load.load_cleaned_dataset()

        spec_matrix = importlib.util.spec_from_file_location("step_06_matrix", matrix_path)
        mod_matrix = importlib.util.module_from_spec(spec_matrix)
        spec_matrix.loader.exec_module(mod_matrix)
        matrix = mod_matrix.compute_association_matrix(df)

        spec_pairs = importlib.util.spec_from_file_location("step_08_pairs", pairs_path)
        mod_pairs = importlib.util.module_from_spec(spec_pairs)
        spec_pairs.loader.exec_module(mod_pairs)
        redundant_pairs, _ = mod_pairs.identify_highly_associated_predictors(matrix)

        spec_vif = importlib.util.spec_from_file_location("step_09_vif", vif_path)
        mod_vif = importlib.util.module_from_spec(spec_vif)
        spec_vif.loader.exec_module(mod_vif)
        vif = mod_vif.run_vif_analysis(df)

        evaluate_multicollinearity_and_recommendations(vif, redundant_pairs)

