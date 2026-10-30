"""
## 4. Model Evaluation - Feature Importance Comparison

This module extracts and ranks Gini/entropy feature importances across the three tuned models
(Decision Tree, Random Forest, XGBoost), plots a 3-panel horizontal bar chart of the top 10 features,
and prints formatted top 10 tables for each model.
"""

###############################################################
# STEP : FEATURE IMPORTANCE COMPARISON
###############################################################

from typing import Dict, Any
import pandas as pd
import matplotlib.pyplot as plt


def extract_and_plot_feature_importances(
    trained_models: Dict[str, Any],
    feature_names: pd.Index,
    top_n: int = 10,
    figsize: tuple = (22, 8)
) -> Dict[str, pd.DataFrame]:
    """
    Extract feature importances for tuned models, plot 3-panel comparison, and display top features.

    Parameters
    ----------
    trained_models : Dict[str, Any]
        Dictionary of trained model instances.
    feature_names : pd.Index
        Feature column names.
    top_n : int, default=10
        Number of top features to plot and display.
    figsize : tuple, default=(22, 8)
        Plot figure size.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping model names to sorted feature importance DataFrames.
    """
    models_to_plot = {
        "Decision Tree (Tuned)": trained_models["Decision Tree (Tuned)"],
        "Random Forest (Tuned)": trained_models["Random Forest (Tuned)"],
        "XGBoost (Tuned)": trained_models["XGBoost (Tuned)"]
    }

    feature_importance_tables = {}
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, (name, model) in zip(axes, models_to_plot.items()):
        importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        })

        importance = importance.sort_values(
            by="Importance",
            ascending=False
        )

        feature_importance_tables[name] = importance
        top10 = importance.head(top_n)

        ax.barh(
            top10["Feature"],
            top10["Importance"]
        )
        ax.set_title(name)
        ax.invert_yaxis()

    plt.tight_layout()
    plt.show()

    # Display Top 10 Features
    for name in feature_importance_tables:
        print("\n", "=" * 80)
        print(name)
        print("=" * 80)
        try:
            display(feature_importance_tables[name].head(top_n))
        except NameError:
            print(feature_importance_tables[name].head(top_n))

    return feature_importance_tables


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
        X, y, encoder, cols_dropped = mod_fe.prepare_features_and_target(df)

        spec_split = importlib.util.spec_from_file_location("step_05_split", split_path)
        mod_split = importlib.util.module_from_spec(spec_split)
        spec_split.loader.exec_module(mod_split)
        X_train, X_test, y_train, y_test = mod_split.split_dataset(X, y)
        baseline_models = mod_split.build_baseline_models()

        spec_tuning = importlib.util.spec_from_file_location("step_06_tuning", tuning_path)
        mod_tuning = importlib.util.module_from_spec(spec_tuning)
        spec_tuning.loader.exec_module(mod_tuning)
        X_train_clean = mod_tuning.clean_xgb_col_names(X_train.copy())
        X_test_clean = mod_tuning.clean_xgb_col_names(X_test.copy())
        tree_tuned, rf_tuned, xgb_tuned = mod_tuning.run_all_tuning(X_train_clean, y_train)

        spec_train = importlib.util.spec_from_file_location("step_07_train", train_path)
        mod_train = importlib.util.module_from_spec(spec_train)
        spec_train.loader.exec_module(mod_train)
        models = mod_train.assemble_all_models(baseline_models, tree_tuned, rf_tuned, xgb_tuned)
        trained_models, predictions, probabilities = mod_train.train_all_models(models, X_train_clean, y_train, X_test_clean)

        extract_and_plot_feature_importances(trained_models, X.columns)

