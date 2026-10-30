"""
### 4.2. Bivariate Analysis - Redundant Predictor Pairs Detection

This module filters and identifies highly associated predictor pairs based on a defined
Cramér's V threshold (default: 0.80) to detect redundancy prior to regression modelling.
"""

###############################################################
# Identify Highly Associated Predictors
###############################################################

from typing import List, Tuple, Optional
import pandas as pd


def identify_highly_associated_predictors(
    association_matrix: pd.DataFrame,
    threshold: float = 0.80
) -> Tuple[List[list], Optional[pd.DataFrame]]:
    """
    Identify predictor pairs exceeding the Cramér's V association threshold.

    Parameters
    ----------
    association_matrix : pd.DataFrame
        Pairwise Cramér's V association matrix.
    threshold : float, default=0.80
        Association threshold above which pairs are flagged as redundant.

    Returns
    -------
    Tuple[List[list], Optional[pd.DataFrame]]
        List of redundant pair lists and the resulting DataFrame (or None if empty).
    """
    print("\n")
    print("=" * 90)
    print("HIGHLY ASSOCIATED PREDICTOR PAIRS")
    print("=" * 90)

    predictors = list(association_matrix.columns)
    redundant_pairs = []

    for i in range(len(predictors)):
        for j in range(i + 1, len(predictors)):
            value = association_matrix.iloc[i, j]
            if value >= threshold:
                redundant_pairs.append(
                    [
                        predictors[i],
                        predictors[j],
                        round(value, 3)
                    ]
                )

    redundant_df = None
    if len(redundant_pairs) == 0:
        print("No highly associated predictor pairs detected.")
    else:
        redundant_df = pd.DataFrame(
            redundant_pairs,
            columns=[
                "Predictor 1",
                "Predictor 2",
                "Cramer's V"
            ]
        )
        try:
            display(redundant_df)
        except NameError:
            print(redundant_df)

    return redundant_pairs, redundant_df


if __name__ == "__main__":
    import os
    import importlib.util

    current_dir = os.path.dirname(os.path.abspath(__file__))
    loader_path = os.path.join(current_dir, "01_environment_and_data_loading.py")
    matrix_path = os.path.join(current_dir, "06_bivariate_cramers_v_matrix.py")

    if os.path.exists(loader_path) and os.path.exists(matrix_path):
        spec_load = importlib.util.spec_from_file_location("step_01_loader", loader_path)
        mod_load = importlib.util.module_from_spec(spec_load)
        spec_load.loader.exec_module(mod_load)
        df = mod_load.load_cleaned_dataset()

        spec_matrix = importlib.util.spec_from_file_location("step_06_matrix", matrix_path)
        mod_matrix = importlib.util.module_from_spec(spec_matrix)
        spec_matrix.loader.exec_module(mod_matrix)
        matrix = mod_matrix.compute_association_matrix(df)

        identify_highly_associated_predictors(matrix)

