"""
## Save Knowledge Routing Agent

The Knowledge Routing Agent successfully predicts the enterprise knowledge repository required for a given question. Statistical analysis identified significant relationships between question characteristics and repository selection, while machine learning models demonstrated strong predictive capability. Tree-based models consistently identified Question Intent, Business Capability and Question Type as the most influential routing variables. The resulting routing matrix forms the decision logic for the proposed Agentic RAG architecture.

One enhancement for later analysis:
My current RQ2 predicts the entire repository combination (e.g., Public Documentation | Architecture Blueprint | Methodology | SME) as a single four-class label. That works, but it doesn't fully reflect how a routing agent behaves.

A more agentic design would treat each repository as an independent decision:

Architecture Blueprint? → Yes/No
Methodology? → Yes/No
Pricing Template? → Yes/No
Governance Framework? → Yes/No
SME Consultation? → Yes/No

This becomes a multi-label classification problem rather than a multi-class one. It matches the architecture I described in  synopsis much more closely because the agent can route a question to multiple repositories simultaneously instead of choosing only one combined label.

For my interim report, I am keeping the current four-class implementation—it is simpler to explain and demonstrates the concept well. For  final dissertation, however, I would  consider evolving RQ2 into the multi-label formulation. It would align much better with the design of an actual enterprise Knowledge Routing Agent and strengthen the novelty of my research.

Save the routing model and the objects RQ3 needs to predict repositories for the evaluation questions.
"""

###############################################################
# ROUTING DEMONSTRATION, BUSINESS IMPACT & HANDOFF EXPORT
###############################################################

from typing import Dict, Any, List, Tuple
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run_routing_demonstration(
    best_model_name: str,
    trained_models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    encoder: Any
) -> pd.DataFrame:
    """
    Demonstrate routing predictions vs ground truth repositories on the test set.
    """
    best_model = trained_models[best_model_name]
    predictions = best_model.predict(X_test)
    predicted_repository = encoder.inverse_transform(predictions)
    actual_repository = encoder.inverse_transform(y_test)

    demo = X_test.copy()
    demo["Actual Repository"] = actual_repository
    demo["Predicted Repository"] = predicted_repository
    demo["Routing Correct"] = np.where(
        demo["Actual Repository"] == demo["Predicted Repository"],
        "Yes",
        "No"
    )

    try:
        display(
            demo[["Actual Repository", "Predicted Repository", "Routing Correct"]].head(20)
        )
    except NameError:
        print(
            demo[["Actual Repository", "Predicted Repository", "Routing Correct"]].head(20)
        )

    return demo


def analyze_business_impact(demo: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize routing impact into correct vs potential misrouting risk.
    """
    impact = demo.copy()
    impact["Business Impact"] = np.where(
        impact["Routing Correct"] == "Yes",
        "Correct repository retrieved",
        "Potential retrieval from incorrect repository"
    )

    try:
        display(
            impact[["Actual Repository", "Predicted Repository", "Routing Correct", "Business Impact"]].head(20)
        )
    except NameError:
        print(
            impact[["Actual Repository", "Predicted Repository", "Routing Correct", "Business Impact"]].head(20)
        )

    return impact


def compute_routing_accuracy_by_repository(impact: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate and plot routing accuracy broken down by repository class.
    """
    routing_accuracy = (
        impact
        .groupby("Actual Repository")["Routing Correct"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index()
    )

    routing_accuracy.columns = [
        "Repository",
        "Routing Accuracy (%)"
    ]

    try:
        display(routing_accuracy.round(2))
    except NameError:
        print(routing_accuracy.round(2))

    plt.figure(figsize=(8, 5))
    plt.bar(
        routing_accuracy["Repository"],
        routing_accuracy["Routing Accuracy (%)"]
    )
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.title("Knowledge Routing Accuracy by Repository")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

    return routing_accuracy


def generate_routing_decision_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the deterministic decision routing table (mode repository per Question Intent).
    """
    routing_table = (
        df.groupby("Question Intent")["Enterprise Knowledge Required"]
        .apply(lambda x: x.value_counts().idxmax())
        .reset_index()
    )

    routing_table.columns = [
        "Question Intent",
        "Recommended Repository"
    ]

    try:
        display(routing_table)
    except NameError:
        print(routing_table)

    return routing_table


def save_rq2_routing_bundle(
    df: pd.DataFrame,
    X: pd.DataFrame,
    X_train: pd.DataFrame,
    encoder: Any,
    trained_models: Dict[str, Any],
    best_model_name: str,
    columns_to_drop: List[str],
    handoff_dir: str = "/content/drive/MyDrive/DataAnalytics_Capstone/handoff"
) -> str:
    """
    Persist the complete routing artifacts bundle required by RQ3 for response generation.
    """
    os.makedirs(handoff_dir, exist_ok=True)

    rq2_bundle = {
        "df": df,
        "X": X,
        "X_train": X_train,
        "encoder": encoder,
        "trained_models": trained_models,
        "best_model_name": best_model_name,
        "columns_to_drop": columns_to_drop
    }

    bundle_path = f"{handoff_dir}/rq2_routing_bundle.pkl"
    with open(bundle_path, "wb") as f:
        pickle.dump(rq2_bundle, f)

    print("Knowledge Routing Agent saved.")
    return bundle_path


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
        best_model_name = results_df.iloc[0]["Model"]

        demo = run_routing_demonstration(best_model_name, trained_models, X_test_clean, y_test, encoder)
        impact = analyze_business_impact(demo)
        routing_accuracy = compute_routing_accuracy_by_repository(impact)
        routing_table = generate_routing_decision_table(df)

        save_rq2_routing_bundle(
            df=df,
            X=X,
            X_train=X_train,
            encoder=encoder,
            trained_models=trained_models,
            best_model_name=best_model_name,
            columns_to_drop=cols_dropped
        )

