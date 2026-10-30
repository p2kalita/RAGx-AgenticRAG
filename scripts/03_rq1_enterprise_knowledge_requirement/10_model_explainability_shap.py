"""
## 5. Preliminary Analysis - Explainability (Decision Tree & SHAP)

This module implements model explainability:
1. Decision Tree Visualization via plot_tree
2. Global SHAP Explainability for Random Forest (Tuned) - Bar and Beeswarm
3. Global SHAP Explainability for XGBoost (Tuned) - Bar and Beeswarm
4. Local SHAP Explanation for a representative test sample via waterfall plot
"""

###############################################################
# EXPLAINABILITY
#
# 1. Decision Tree Visualization
# 2. SHAP - Random Forest (Tuned)
# 3. SHAP - XGBoost (Tuned)
###############################################################

from typing import Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import shap


def visualize_decision_tree(
    tree_model: Any,
    feature_names: pd.Index,
    figsize: tuple = (24, 12),
    fontsize: int = 8
) -> None:
    """
    Render decision tree diagram with colored nodes and class annotations.
    """
    print("=" * 80)
    print("Decision Tree Visualization")
    print("=" * 80)

    plt.figure(figsize=figsize)
    plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=["Partially", "Enterprise Knowledge"],
        filled=True,
        rounded=True,
        fontsize=fontsize
    )
    plt.title("Decision Tree (Tuned)")
    plt.show()


def explain_random_forest_shap(
    rf_model: Any,
    X_test: pd.DataFrame
) -> Any:
    """
    Compute and display global SHAP explanations for tuned Random Forest model.
    """
    print("=" * 80)
    print("SHAP Explainability : Random Forest (Tuned)")
    print("=" * 80)

    rf_explainer = shap.TreeExplainer(rf_model)
    rf_shap = rf_explainer.shap_values(X_test)

    # Handle binary classification output
    if isinstance(rf_shap, list):
        rf_shap = rf_shap[1]

    # Global Feature Importance
    shap.summary_plot(
        rf_shap,
        X_test,
        plot_type="bar",
        show=True
    )

    # Beeswarm Plot
    shap.summary_plot(
        rf_shap,
        X_test,
        show=True
    )

    return rf_shap


def explain_xgboost_shap(
    xgb_model: Any,
    X_test: pd.DataFrame
) -> Tuple[Any, Any]:
    """
    Compute and display global and local SHAP explanations for tuned XGBoost model.
    """
    print("=" * 80)
    print("SHAP Explainability : XGBoost (Tuned)")
    print("=" * 80)

    xgb_explainer = shap.TreeExplainer(xgb_model)
    xgb_shap = xgb_explainer.shap_values(X_test)

    # Handle binary classification output
    if isinstance(xgb_shap, list):
        xgb_shap = xgb_shap[1]

    # Global Importance
    shap.summary_plot(
        xgb_shap,
        X_test,
        plot_type="bar",
        show=True
    )

    # Beeswarm
    shap.summary_plot(
        xgb_shap,
        X_test,
        show=True
    )

    ###############################################################
    # PART 4 : Local Explainability
    ###############################################################

    print("=" * 80)
    print("Local SHAP Explanation (First Test Sample)")
    print("=" * 80)

    base_val = xgb_explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)) and len(base_val) > 1:
        base_val = base_val[1]

    shap.plots.waterfall(
        shap.Explanation(
            values=xgb_shap[0],
            base_values=base_val,
            data=X_test.iloc[0],
            feature_names=X_test.columns
        )
    )

    return xgb_shap, xgb_explainer


def run_explainability_suite(
    trained_models: Dict[str, Any],
    X: pd.DataFrame,
    X_test: pd.DataFrame
) -> None:
    """
    Execute entire explainability pipeline.
    """
    visualize_decision_tree(trained_models["Decision Tree (Tuned)"], X.columns)
    explain_random_forest_shap(trained_models["Random Forest (Tuned)"], X_test)
    explain_xgboost_shap(trained_models["XGBoost (Tuned)"], X_test)


if __name__ == "__main__":
    import os
    import importlib.util
    import numpy as np

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

        run_explainability_suite(trained_models, X, X_test)

