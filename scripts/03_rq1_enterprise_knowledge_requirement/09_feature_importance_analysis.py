"""
## 5. Preliminary Analysis - Feature Importance Extraction & Visualization

This module extracts feature importances (or absolute coefficients for logistic models)
across tree and linear classifiers, stores them in structured DataFrames,
and renders horizontal bar charts of the top 15 features per model.
"""

###############################################################
# FEATURE IMPORTANCE & FEATURE IMPORTANCE PLOTS
###############################################################

from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_feature_importances(
    trained_models: Dict[str, Any],
    feature_names: pd.Index
) -> Dict[str, pd.DataFrame]:
    """
    Extract feature importance metrics across all tree-based and linear models.

    Parameters
    ----------
    trained_models : Dict[str, Any]
        Dictionary of trained model instances.
    feature_names : pd.Index
        Column names corresponding to design matrix X.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping model names to sorted feature importance DataFrames.
    """
    importance_models = {
        "Balanced Logistic": trained_models["Balanced Logistic"],
        "Decision Tree": trained_models["Decision Tree"],
        "Decision Tree (Tuned)": trained_models["Decision Tree (Tuned)"],
        "Random Forest": trained_models["Random Forest"],
        "Random Forest (Tuned)": trained_models["Random Forest (Tuned)"],
        "XGBoost": trained_models["XGBoost"],
        "XGBoost (Tuned)": trained_models["XGBoost (Tuned)"]
    }

    feature_importance_dataframes = {}

    for name, model in importance_models.items():
        if hasattr(model, "coef_"):
            importance = np.abs(model.coef_[0])
        else:
            importance = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values(
            "Importance",
            ascending=False
        )

        feature_importance_dataframes[name] = importance_df

        print("\n", "=" * 70)
        print(name)
        try:
            display(importance_df.head(15))
        except NameError:
            print(importance_df.head(15))

    return feature_importance_dataframes


def plot_feature_importances(
    feature_importance_dataframes: Dict[str, pd.DataFrame],
    top_n: int = 15,
    figsize: tuple = (8, 6)
) -> None:
    """
    Render horizontal bar chart for top N features per model.
    """
    for model_name, df_plot in feature_importance_dataframes.items():
        df_sub = df_plot.sort_values(
            by="Importance",
            ascending=False
        ).head(top_n)

        plt.figure(figsize=figsize)
        plt.barh(
            df_sub["Feature"],
            df_sub["Importance"]
        )
        plt.gca().invert_yaxis()
        plt.title(model_name)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import os
    import importlib.util

    current_dir = os.path.dirname(os.path.abspath(__file__))
    loader_path = os.path.join(current_dir, "01_environment_and_data_loading.py")
    fe_path = os.path.join(current_dir, "04_feature_engineering_and_encoding.py")
    split_path = os.path.join(current_dir, "05_dataset_split_and_baseline_models.py")
    tuning_path = os.path.join(current_dir, "06_hyperparameter_tuning.py")
    train_path = os.path.join(current_dir, "07_model_training_and_evaluation.py")

    if (os.path.exists(loader_path) and os.path.exists(fe_path) and
            os.path.exists(split_path) and os.path.exists(tuning_path) and os.path.exists(train_path)):
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
        baseline_models = mod_split.build_baseline_models()

        spec_tuning = importlib.util.spec_from_file_location("step_06_tuning", tuning_path)
        mod_tuning = importlib.util.module_from_spec(spec_tuning)
        spec_tuning.loader.exec_module(mod_tuning)
        tree_tuned, rf_tuned, xgb_tuned = mod_tuning.run_all_tuning(X_train, y_train)

        spec_train = importlib.util.spec_from_file_location("step_07_train", train_path)
        mod_train = importlib.util.module_from_spec(spec_train)
        spec_train.loader.exec_module(mod_train)
        models = mod_train.assemble_all_models(baseline_models, tree_tuned, rf_tuned, xgb_tuned)
        trained_models, predictions, probabilities = mod_train.train_all_models(models, X_train, y_train, X_test)

        feature_importance_dfs = extract_feature_importances(trained_models, X.columns)
        plot_feature_importances(feature_importance_dfs)

