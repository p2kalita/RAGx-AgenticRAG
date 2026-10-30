"""
## 5. Preliminary Analysis - Stratified Cross-Validation & Diagnostics

This module performs 5-fold Stratified Cross-Validation across all models,
and generates diagnostic visualizations: Confusion Matrices and ROC Curves.
"""

###############################################################
# STRATIFIED CROSS VALIDATION, CONFUSION MATRICES & ROC CURVES
###############################################################

from typing import Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def run_stratified_cross_validation(
    trained_models: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Execute 5-fold Stratified Cross-Validation computing mean accuracy, precision, recall, f1, and roc_auc.

    Returns
    -------
    pd.DataFrame
        DataFrame summarizing cross-validation performance.
    """
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    metrics = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    cv_results = []

    for name, model in trained_models.items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=metrics,
            n_jobs=-1
        )

        cv_results.append({
            "Model": name,
            "Accuracy": scores["test_accuracy"].mean(),
            "Precision": scores["test_precision"].mean(),
            "Recall": scores["test_recall"].mean(),
            "F1": scores["test_f1"].mean(),
            "ROC_AUC": scores["test_roc_auc"].mean()
        })

    cv_df = pd.DataFrame(cv_results)

    try:
        display(cv_df.round(3))
    except NameError:
        print(cv_df.round(3))

    return cv_df


def plot_confusion_matrices(
    trained_models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figsize: tuple = (5, 5),
    cmap: str = "Blues"
) -> None:
    """
    Render Confusion Matrix display for every trained model.
    """
    for name, model in trained_models.items():
        fig, ax = plt.subplots(figsize=figsize)
        ConfusionMatrixDisplay.from_estimator(
            model,
            X_test,
            y_test,
            cmap=cmap,
            ax=ax
        )
        plt.title(name)
        plt.show()


def plot_roc_curves(
    trained_models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figsize: tuple = (8, 6)
) -> None:
    """
    Plot overlaid ROC Curves for all trained models.
    """
    plt.figure(figsize=figsize)
    for name, model in trained_models.items():
        RocCurveDisplay.from_estimator(
            model,
            X_test,
            y_test,
            name=name,
            ax=plt.gca()
        )
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

        cv_df = run_stratified_cross_validation(trained_models, X, y)
        plot_confusion_matrices(trained_models, X_test, y_test)
        plot_roc_curves(trained_models, X_test, y_test)

