"""
## 4. Hyperparameters

This module performs RandomizedSearchCV hyperparameter optimization targeting the f1_macro objective
for multi-class Decision Tree, Random Forest, and XGBoost classifiers.
It preserves the helper function clean_xgb_col_names to sanitize feature names for XGBoost compatibility.
"""

###############################################################
# HYPERPARAMETER TUNING (RQ2)
###############################################################

from typing import Tuple, Any
import re
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def clean_xgb_col_names(df):
    """
    Function to clean column names for XGBoost compatibility.
    Replaces non-alphanumeric characters with underscores and strips edges.
    """
    cols = df.columns
    new_cols = []
    for col in cols:
        # Replace non-alphanumeric characters (excluding underscore) with underscore
        new_col = re.sub('[^A-Za-z0-9_]+', '_', col)
        # Remove leading/trailing underscores if any
        new_col = new_col.strip('_')
        new_cols.append(new_col)
    df.columns = new_cols
    return df


def tune_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> Tuple[Any, RandomizedSearchCV]:
    """
    Tune Decision Tree hyperparameters optimizing f1_macro.
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
        cv=5,
        n_iter=15,
        scoring="f1_macro",
        random_state=random_state,
        n_jobs=-1
    )

    tree_search.fit(X_train, y_train)
    tree_tuned = tree_search.best_estimator_
    print(tree_search.best_params_)
    return tree_tuned, tree_search


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> Tuple[Any, RandomizedSearchCV]:
    """
    Tune Random Forest hyperparameters optimizing f1_macro.
    """
    rf_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }

    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=random_state),
        rf_grid,
        cv=5,
        n_iter=15,
        scoring="f1_macro",
        random_state=random_state,
        n_jobs=-1
    )

    rf_search.fit(X_train, y_train)
    rf_tuned = rf_search.best_estimator_
    print(rf_search.best_params_)
    return rf_tuned, rf_search


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> Tuple[Any, RandomizedSearchCV]:
    """
    Tune XGBoost hyperparameters optimizing f1_macro.
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
            eval_metric="mlogloss"
        ),
        xgb_grid,
        cv=5,
        n_iter=15,
        scoring="f1_macro",
        random_state=random_state,
        n_jobs=-1
    )

    xgb_search.fit(X_train, y_train)
    xgb_tuned = xgb_search.best_estimator_
    print(xgb_search.best_params_)
    return xgb_tuned, xgb_search


def run_all_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> Tuple[Any, Any, Any]:
    """
    Run full tuning pipeline on sanitized feature matrix.
    """
    X_train_clean = clean_xgb_col_names(X_train.copy())
    tree_tuned, _ = tune_decision_tree(X_train_clean, y_train, random_state)
    rf_tuned, _ = tune_random_forest(X_train_clean, y_train, random_state)
    xgb_tuned, _ = tune_xgboost(X_train_clean, y_train, random_state)
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
        X, y, encoder, cols_dropped = mod_fe.prepare_features_and_target(df)

        spec_split = importlib.util.spec_from_file_location("step_05_split", split_path)
        mod_split = importlib.util.module_from_spec(spec_split)
        spec_split.loader.exec_module(mod_split)
        X_train, X_test, y_train, y_test = mod_split.split_dataset(X, y)

        tree_tuned, rf_tuned, xgb_tuned = run_all_tuning(X_train, y_train)
        print("Hyperparameter tuning completed.")

