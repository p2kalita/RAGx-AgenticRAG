"""
## 3. Baseline ML models

This module handles stratified dataset partitioning, instantiating the baseline model dictionary
(Logistic Regression, Balanced Logistic, Decision Tree, Random Forest, XGBoost),
and fitting the statsmodels regularized logit model.
"""

###############################################################
# Split Dataset & BUILD BASELINE MODELS
###############################################################

from typing import Tuple, Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import statsmodels.api as sm


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split feature matrix X and target y using stratified sampling.

    Parameters
    ----------
    X : pd.DataFrame
        Design matrix.
    y : pd.Series
        Target vector.
    test_size : float, default=0.20
        Fraction of records reserved for test set.
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

    print("Training Records :", len(X_train))
    print("Testing Records :", len(X_test))

    return X_train, X_test, y_train, y_test


def build_baseline_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Instantiate baseline classifiers for benchmarking.

    Parameters
    ----------
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Dictionary of baseline estimators.
    """
    baseline_models = {
        "Logistic Regression":
            LogisticRegression(
                random_state=random_state,
                max_iter=1000
            ),

        "Balanced Logistic":
            LogisticRegression(
                class_weight="balanced",
                random_state=random_state,
                max_iter=1000
            ),

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
                eval_metric="logloss"
            )
    }
    return baseline_models


def fit_statsmodels_logit(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = 0.01
) -> Any:
    """
    Fit L1-regularized logistic regression via statsmodels and print summary table.

    Parameters
    ----------
    X : pd.DataFrame
        Predictor matrix.
    y : pd.Series
        Target vector.
    alpha : float, default=0.01
        Regularization penalty.

    Returns
    -------
    Any
        Statsmodels regularized logit result wrapper.
    """
    X_sm = sm.add_constant(X)
    logit = sm.Logit(y, X_sm)
    result = logit.fit_regularized(alpha=alpha)
    print(result.summary())
    return result


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
        X, y = mod_fe.prepare_features_and_target(df)

        X_train, X_test, y_train, y_test = split_dataset(X, y)
        baseline_models = build_baseline_models()
        sm_result = fit_statsmodels_logit(X, y)

