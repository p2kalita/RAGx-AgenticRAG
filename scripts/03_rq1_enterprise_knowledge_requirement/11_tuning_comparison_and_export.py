"""
## 5. Preliminary Analysis - Default vs Tuned Comparison & Artifact Persistence

To do:  Since the dataset is imbalanced, optimizing for accuracy pushes the models toward predicting the majority class. I would recommend change tuning objective to scoring="f1" or scoring="recall" (or f1_macro/f1_weighted depending on  chosen emphasis). This is likely to improve the tuned Random Forest and XGBoost models considerably while making the model selection more consistent with the business objective of correctly identifying questions that require enterprise knowledge.
"""

###############################################################
# DEFAULT vs TUNED MODEL COMPARISON & SAVE ARTIFACTS
###############################################################

from typing import Dict, Any, Optional
import os
import pandas as pd


def build_tuning_comparison_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct comparison DataFrame of baseline vs. tuned metrics and compute net delta improvements.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing evaluation results for all models.

    Returns
    -------
    pd.DataFrame
        Comparison DataFrame with Accuracy, F1, and ROC-AUC metrics and improvements.
    """
    comparison = pd.DataFrame({
        "Algorithm": [
            "Decision Tree",
            "Random Forest",
            "XGBoost"
        ],

        "Default Accuracy": [
            results_df.loc[results_df.Model == "Decision Tree", "Accuracy"].values[0],
            results_df.loc[results_df.Model == "Random Forest", "Accuracy"].values[0],
            results_df.loc[results_df.Model == "XGBoost", "Accuracy"].values[0]
        ],

        "Tuned Accuracy": [
            results_df.loc[results_df.Model == "Decision Tree (Tuned)", "Accuracy"].values[0],
            results_df.loc[results_df.Model == "Random Forest (Tuned)", "Accuracy"].values[0],
            results_df.loc[results_df.Model == "XGBoost (Tuned)", "Accuracy"].values[0]
        ],

        "Default F1": [
            results_df.loc[results_df.Model == "Decision Tree", "F1"].values[0],
            results_df.loc[results_df.Model == "Random Forest", "F1"].values[0],
            results_df.loc[results_df.Model == "XGBoost", "F1"].values[0]
        ],

        "Tuned F1": [
            results_df.loc[results_df.Model == "Decision Tree (Tuned)", "F1"].values[0],
            results_df.loc[results_df.Model == "Random Forest (Tuned)", "F1"].values[0],
            results_df.loc[results_df.Model == "XGBoost (Tuned)", "F1"].values[0]
        ],

        "Default ROC-AUC": [
            results_df.loc[results_df.Model == "Decision Tree", "ROC_AUC"].values[0],
            results_df.loc[results_df.Model == "Random Forest", "ROC_AUC"].values[0],
            results_df.loc[results_df.Model == "XGBoost", "ROC_AUC"].values[0]
        ],

        "Tuned ROC-AUC": [
            results_df.loc[results_df.Model == "Decision Tree (Tuned)", "ROC_AUC"].values[0],
            results_df.loc[results_df.Model == "Random Forest (Tuned)", "ROC_AUC"].values[0],
            results_df.loc[results_df.Model == "XGBoost (Tuned)", "ROC_AUC"].values[0]
        ]
    })

    comparison["Accuracy Improvement"] = (
        comparison["Tuned Accuracy"] - comparison["Default Accuracy"]
    )

    comparison["F1 Improvement"] = (
        comparison["Tuned F1"] - comparison["Default F1"]
    )

    comparison["ROC Improvement"] = (
        comparison["Tuned ROC-AUC"] - comparison["Default ROC-AUC"]
    )

    try:
        display(comparison.round(3))
    except NameError:
        print(comparison.round(3))

    return comparison


def build_top_features_comparison(
    trained_models: Dict[str, Any],
    feature_names: pd.Index,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Generate comparison table of top important features between Random Forest (Tuned) and XGBoost (Tuned).
    """
    print("\n" + "=" * 80)
    print("TOP 10 IMPORTANT FEATURES COMPARISON")
    print("=" * 80)

    comparison = pd.DataFrame({
        "Random Forest": trained_models["Random Forest (Tuned)"].feature_importances_,
        "XGBoost": trained_models["XGBoost (Tuned)"].feature_importances_
    }, index=feature_names)

    comparison["Average Importance"] = comparison.mean(axis=1)

    comparison = comparison.sort_values(
        "Average Importance",
        ascending=False
    )

    try:
        display(comparison.head(top_n).round(4))
    except NameError:
        print(comparison.head(top_n).round(4))

    return comparison


def save_research_artifacts(
    results_df: pd.DataFrame,
    comparison: pd.DataFrame,
    feature_importance_dataframes: Optional[Dict[str, pd.DataFrame]] = None,
    base_dir: str = "/content/drive/MyDrive/DataAnalytics_Capstone"
) -> None:
    """
    Export all research outputs: model comparisons, hyperparameter tuning comparison,
    and individual feature importance CSV files.
    """
    os.makedirs(base_dir, exist_ok=True)

    # Save to Excel
    excel_path = f"{base_dir}/Hyperparameter_Tuning_Comparison.xlsx"
    comparison.to_excel(excel_path, index=False)
    print("Hyperparameter tuning comparison saved to Excel.")

    # Save results_df
    results_csv = f"{base_dir}/Model_Comparison.csv"
    results_df.to_csv(results_csv, index=False)
    print("Model comparison saved.")

    # Iterate over the stored feature importance DataFrames
    if feature_importance_dataframes is not None:
        for model_name, importance_df in feature_importance_dataframes.items():
            filename = model_name.replace(" ", "_") + "_Importance.csv"
            importance_df.to_csv(filename, index=False)

    # Also saving hyperparameter comparisons to CSV
    tuning_csv = f"{base_dir}/Hyperparameter_Tuning_Comparison.csv"
    comparison.to_csv(tuning_csv, index=False)


if __name__ == "__main__":
    import os
    import importlib.util

    current_dir = os.path.dirname(os.path.abspath(__file__))
    loader_path = os.path.join(current_dir, "01_environment_and_data_loading.py")
    fe_path = os.path.join(current_dir, "04_feature_engineering_and_encoding.py")
    split_path = os.path.join(current_dir, "05_dataset_split_and_baseline_models.py")
    tuning_path = os.path.join(current_dir, "06_hyperparameter_tuning.py")
    train_path = os.path.join(current_dir, "07_model_training_and_evaluation.py")
    fi_path = os.path.join(current_dir, "09_feature_importance_analysis.py")

    if (os.path.exists(loader_path) and os.path.exists(fe_path) and
            os.path.exists(split_path) and os.path.exists(tuning_path) and
            os.path.exists(train_path) and os.path.exists(fi_path)):
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
        results_df = mod_train.evaluate_models(trained_models, predictions, probabilities, y_test)

        spec_fi = importlib.util.spec_from_file_location("step_09_fi", fi_path)
        mod_fi = importlib.util.module_from_spec(spec_fi)
        spec_fi.loader.exec_module(mod_fi)
        feature_importance_dfs = mod_fi.extract_feature_importances(trained_models, X.columns)

        comparison = build_tuning_comparison_table(results_df)
        top_features = build_top_features_comparison(trained_models, X.columns)
        save_research_artifacts(results_df, comparison, feature_importance_dfs)

