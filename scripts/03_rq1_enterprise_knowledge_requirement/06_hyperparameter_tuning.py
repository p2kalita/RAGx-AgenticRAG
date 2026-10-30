"""
## 4. Hyperparameter Tuning

This module performs RandomizedSearchCV hyperparameter optimization on Decision Tree,
Random Forest, and XGBoost classifiers targeting the ROC-AUC objective.
"""

###############################################################
# HYPERPARAMETER TUNING
###############################################################

from typing import Tuple, Any
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def tune_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> Tuple[Any, RandomizedSearchCV]:
    """
    Tune Decision Tree hyperparameters using RandomizedSearchCV.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    Tuple[Any, RandomizedSearchCV]
        (best_estimator, search_object)
    """
    tree_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    tree_search = RandomizedSearchCV(
        DecisionTreeClassifier(random_state=random_state),
        tree_grid,
        n_iter=15,
        cv=5,
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=-1
    )

    tree_search.fit(X_train, y_train)
    tree_tuned = tree_search.best_estimator_
    return tree_tuned, tree_search


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> Tuple[Any, RandomizedSearchCV]:
    """
    Tune Random Forest hyperparameters using RandomizedSearchCV.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    Tuple[Any, RandomizedSearchCV]
        (best_estimator, search_object)
    """
    rf_grid = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [5, 10, 15, None],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }

    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=random_state),
        rf_grid,
        n_iter=20,
        cv=5,
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=-1
    )

    rf_search.fit(X_train, y_train)
    rf_tuned = rf_search.best_estimator_
    return rf_tuned, rf_search


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> Tuple[Any, RandomizedSearchCV]:
    """
    Tune XGBoost hyperparameters using RandomizedSearchCV.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    Tuple[Any, RandomizedSearchCV]
        (best_estimator, search_object)
    """
    xgb_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1],
        "colsample_bytree": [0.8, 1]
    }

    xgb_search = RandomizedSearchCV(
        XGBClassifier(
            random_state=random_state,
            eval_metric="logloss"
        ),
        xgb_grid,
        n_iter=20,
        cv=5,
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=-1
    )

    xgb_search.fit(X_train, y_train)
    xgb_tuned = xgb_search.best_estimator_
    return xgb_tuned, xgb_search


def run_all_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> Tuple[Any, Any, Any]:
    """
    Run hyperparameter tuning for Decision Tree, Random Forest, and XGBoost.

    Returns
    -------
    Tuple[Any, Any, Any]
        (tree_tuned, rf_tuned, xgb_tuned)
    """
    tree_tuned, _ = tune_decision_tree(X_train, y_train, random_state)
    rf_tuned, _ = tune_random_forest(X_train, y_train, random_state)
    xgb_tuned, _ = tune_xgboost(X_train, y_train, random_state)
    return tree_tuned, rf_tuned, xgb_tuned


if __name__ == "__main__":
    import os
    import importlib.util

    current_dir = os.path.dirname(os.path.abspath(__file__))
    loader_path = os.path.join(current_dir, "01_environment_and_data_loading.py")
    fe_path = os.path.join(current_dir, "04_feature_engineering_and_encoding.py")
    split_path = os.path.join(current_dir, "05_dataset_split_and_baseline_models.py")

    if os.path.exists(loader_path) and os.path.exists(fe_path) and os.path.exists(split_path):
        spec_load = importlib.util.spec_from_file_location("step_01_loader", loader_path)
        mod_load = importlib.util.module_from_spec(spec_load)
        spec_load.loader.exec_module(mod_load)
        df = mod_load.load_cleaned_dataset()

        spec_fe = importlib.util.spec_from_file_location("step_04_fe", fe_path)
        mod_fe = importlib.util.module_from_spec(spec_fe)
        spec_fe.loader.exec_module(mod_fe)
        X, y = mod_fe.prepare_features_and_target(df)

        spec_split = importlib.util.spec_from_file_location("step_05_split", split_path)
        mod_split = importlib.util.module_from_spec(spec_split)
        spec_split.loader.exec_module(mod_split)
        X_train, X_test, y_train, y_test = mod_split.split_dataset(X, y)

        tree_tuned, rf_tuned, xgb_tuned = run_all_tuning(X_train, y_train)
        print("Tuning completed successfully.")

