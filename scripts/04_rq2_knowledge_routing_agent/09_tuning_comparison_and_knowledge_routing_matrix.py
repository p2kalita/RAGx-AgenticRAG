"""
## 4. Model Evaluation - Tuning Comparison & Knowledge Routing Matrix

This module constructs the default vs. tuned F1 improvement comparison table
and computes the empirical Knowledge Routing Matrix (Question Intent vs. Enterprise Knowledge Required, row normalized)
visualized via a seaborn heatmap.
"""

###############################################################
# DEFAULT vs TUNED COMPARISON & KNOWLEDGE ROUTING MATRIX
###############################################################

from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def build_tuning_comparison(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute F1 improvement comparing default and tuned models.
    """
    comparison = []
    pairs = [
        ("Decision Tree", "Decision Tree (Tuned)"),
        ("Random Forest", "Random Forest (Tuned)"),
        ("XGBoost", "XGBoost (Tuned)")
    ]

    for default, tuned in pairs:
        d = results_df[results_df.Model == default].iloc[0]
        t = results_df[results_df.Model == tuned].iloc[0]

        comparison.append({
            "Algorithm": default,
            "Default F1": d["F1"],
            "Tuned F1": t["F1"],
            "Improvement": t["F1"] - d["F1"]
        })

    comparison_df = pd.DataFrame(comparison)

    try:
        display(comparison_df.round(3))
    except NameError:
        print(comparison_df.round(3))

    return comparison_df


def generate_knowledge_routing_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate normalized Knowledge Routing Matrix and render heatmap.
    """
    routing = pd.crosstab(
        df["Question Intent"],
        df["Enterprise Knowledge Required"],
        normalize="index"
    )

    try:
        display(round(routing * 100, 1))
    except NameError:
        print(round(routing * 100, 1))

    plt.figure(figsize=(12, 7))
    sns.heatmap(
        routing,
        annot=True,
        cmap="YlGnBu",
        fmt=".2f"
    )
    plt.title("Knowledge Routing Matrix")
    plt.xlabel("Recommended Repository")
    plt.ylabel("Question Intent")
    plt.tight_layout()
    plt.show()

    return routing


def run_comparison_and_routing_matrix(
    results_df: pd.DataFrame,
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute tuning comparison and routing matrix analysis.
    """
    comp_df = build_tuning_comparison(results_df)
    routing_df = generate_knowledge_routing_matrix(df)
    return comp_df, routing_df


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
        results_df = mod_train.evaluate_models(trained_models, predictions, probabilities, y_test)

        run_comparison_and_routing_matrix(results_df, df)

