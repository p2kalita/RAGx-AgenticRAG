"""
## 3. Baseline ML models

Let's try different models except Logistic Regression (since the target is multi-class).
This module partitions the dataset into train and test splits (80/20, stratify=y, random_state=42)
and initializes baseline tree-based classifiers: Decision Tree, Random Forest, and XGBoost.
"""

###############################################################
# BUILD BASELINE MODELS (RQ2)
###############################################################

from typing import Tuple, Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split feature matrix X and multiclass target y using stratified sampling.

    Parameters
    ----------
    X : pd.DataFrame
        Design matrix.
    y : pd.Series
        Target vector.
    test_size : float, default=0.2
        Test set fraction.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def build_baseline_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Construct baseline multi-class tree models.

    Parameters
    ----------
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Dictionary of baseline models.
    """
    baseline_models = {
        "Decision Tree":
            DecisionTreeClassifier(
                random_state=random_state
            ),

        "Random Forest":
            RandomForestClassifier(
                random_state=random_state
            ),

        "XGBoost":
            XGBClassifier(
                random_state=random_state,
                eval_metric="mlogloss"
            )
    }
    return baseline_models


if __name__ == "__main__":
    import os
    import importlib.util

    current_dir = os.path.dirname(os.path.abspath(__file__))
    loader_path = os.path.join(current_dir, "01_environment_and_data_loading.py")
    fe_path = os.path.join(current_dir, "04_feature_engineering_and_encoding.py")

    if os.path.exists(loader_path) and os.path.exists(fe_path):
        spec_load = importlib.util.spec_from_file_location("step_01_loader", loader_path)
        mod_load = importlib.util.module_from_spec(spec_load)
        spec_load.loader.exec_module(mod_load)
        df = mod_load.load_cleaned_dataset()

        spec_fe = importlib.util.spec_from_file_location("step_04_fe", fe_path)
        mod_fe = importlib.util.module_from_spec(spec_fe)
        spec_fe.loader.exec_module(mod_fe)
        X, y, encoder, cols_dropped = mod_fe.prepare_features_and_target(df)

        X_train, X_test, y_train, y_test = split_dataset(X, y)
        baseline_models = build_baseline_models()
        print("Baseline models initialized successfully.")

