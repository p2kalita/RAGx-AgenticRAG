"""
## 4. Model Evaluation

This module combines baseline and tuned multi-class models, fits them on sanitized training features,
generates test predictions and predicted class probabilities, and evaluates performance across:
Accuracy, Balanced Accuracy, Precision (macro), Recall (macro), F1 (macro), and ROC_AUC (multiclass OvR macro).
It also generates the detailed classification report and confusion matrix display for the top-performing model.
"""

###############################################################
# COMBINE ALL MODELS, TRAIN ALL MODELS & MODEL EVALUATION
###############################################################

from typing import Dict, Any, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, classification_report,
    ConfusionMatrixDisplay
)


def assemble_all_models(
    baseline_models: Dict[str, Any],
    tree_tuned: Any,
    rf_tuned: Any,
    xgb_tuned: Any
) -> Dict[str, Any]:
    """
    Combine baseline and tuned models into a single dictionary.
    """
    models = {
        "Decision Tree": baseline_models["Decision Tree"],
        "Decision Tree (Tuned)": tree_tuned,
        "Random Forest": baseline_models["Random Forest"],
        "Random Forest (Tuned)": rf_tuned,
        "XGBoost": baseline_models["XGBoost"],
        "XGBoost (Tuned)": xgb_tuned
    }
    return models


def train_all_models(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Fit each model and compute predictions and probability distributions.
    """
    trained_models = {}
    predictions = {}
    probabilities = {}

    for name, model in models.items():
        print(f"Training {name}")
        model.fit(X_train, y_train)
        trained_models[name] = model
        predictions[name] = model.predict(X_test)
        probabilities[name] = model.predict_proba(X_test)

    return trained_models, predictions, probabilities


def evaluate_models(
    trained_models: Dict[str, Any],
    predictions: Dict[str, Any],
    probabilities: Dict[str, Any],
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Calculate comprehensive multiclass evaluation metrics and rank by F1 macro descending.
    """
    results = []

    for name in trained_models:
        pred = predictions[name]
        prob = probabilities[name]

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Balanced Accuracy": balanced_accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, average="macro"),
            "Recall": recall_score(y_test, pred, average="macro"),
            "F1": f1_score(y_test, pred, average="macro"),
            "ROC_AUC": roc_auc_score(
                y_test,
                prob,
                multi_class="ovr",
                average="macro"
            )
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="F1",
        ascending=False
    )

    try:
        display(results_df.round(3))
    except NameError:
        print(results_df.round(3))

    return results_df


def report_and_plot_best_model(
    results_df: pd.DataFrame,
    predictions: Dict[str, Any],
    y_test: pd.Series,
    class_names: Any
) -> str:
    """
    Print classification report and display confusion matrix for the top-performing model.
    """
    best_model_name = results_df.iloc[0]["Model"]
    print("\n" + "=" * 80)
    print(f"BEST MODEL : {best_model_name}")
    print("=" * 80)

    print(
        classification_report(
            y_test,
            predictions[best_model_name],
            target_names=class_names
        )
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions[best_model_name],
        display_labels=class_names,
        cmap="Blues",
        ax=ax
    )
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.show()

    return best_model_name


if __name__ == "__main__":
    import os
    import importlib.util

    current_dir = os.path.dirname(os.path.abspath(__file__))
    loader_path = os.path.join(current_dir, "01_environment_and_data_loading.py")
    fe_path = os.path.join(current_dir, "04_feature_engineering_and_encoding.py")
    split_path = os.path.join(current_dir, "05_dataset_split_and_baseline_models.py")
    tuning_path = os.path.join(current_dir, "06_hyperparameter_tuning.py")

    if (os.path.exists(loader_path) and os.path.exists(fe_path) and
            os.path.exists(split_path) and os.path.exists(tuning_path)):
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

        models = assemble_all_models(baseline_models, tree_tuned, rf_tuned, xgb_tuned)
        trained_models, predictions, probabilities = train_all_models(models, X_train_clean, y_train, X_test_clean)
        results_df = evaluate_models(trained_models, predictions, probabilities, y_test)
        best_model = report_and_plot_best_model(results_df, predictions, y_test, encoder.classes_)

