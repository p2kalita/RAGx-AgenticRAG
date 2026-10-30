"""
## 2. Feature Engineering

From the above analysis, we are not using Technology Domain.
All predictors are categorical and Logistic Regression needs numbers.
"""

###############################################################
# FEATURE ENGINEERING
###############################################################

from typing import List, Tuple, Optional
import pandas as pd

# Final predictor variables selected for modeling
FINAL_PREDICTORS = [
    "Question Type",
    "Question Complexity",
    "Question Intent",
    "Business Capability"
]

DEFAULT_TARGET = "Public Knowledge Only?"


def select_predictors() -> List[str]:
    """
    Return the final selected predictor variables excluding collinear features.

    Returns
    -------
    List[str]
        List of predictor column names.
    """
    return list(FINAL_PREDICTORS)


def encode_features(
    df: pd.DataFrame,
    predictors: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Perform One-Hot Encoding on categorical predictors and cast to integer.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing predictors.
    predictors : Optional[List[str]], optional
        List of categorical predictor columns. Defaults to FINAL_PREDICTORS.

    Returns
    -------
    pd.DataFrame
        One-hot encoded design matrix X.
    """
    pred_cols = predictors if predictors is not None else FINAL_PREDICTORS
    X = pd.get_dummies(df[pred_cols], drop_first=True)
    X = X.astype(int)
    return X


def encode_target(
    df: pd.DataFrame,
    target: str = DEFAULT_TARGET
) -> pd.Series:
    """
    Encode the binary target variable ('Yes' -> 1, 'Partially' -> 0).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the target column.
    target : str, default=DEFAULT_TARGET
        Name of target column.

    Returns
    -------
    pd.Series
        Binary integer series y.
    """
    y = df[target].map({
        "Yes": 1,
        "Partially": 0
    })
    return y


def prepare_features_and_target(
    df: pd.DataFrame,
    predictors: Optional[List[str]] = None,
    target: str = DEFAULT_TARGET
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Execute full feature engineering and target encoding pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to prepare.
    predictors : Optional[List[str]], optional
        Predictor columns.
    target : str
        Target column.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X, y)
    """
    X = encode_features(df, predictors)
    y = encode_target(df, target)

    print(X.shape)
    print(y.value_counts())

    return X, y


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

        X, y = prepare_features_and_target(df)

