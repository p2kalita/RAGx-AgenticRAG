"""
Run All Experiments: Generate responses using all four strategies and compare them against reference enterprise answers.

Save Results:
"""

###############################################################
# RQ3 - STEP 19 & 20 : EXPERIMENT RUNNER & RESULTS PERSISTENCE
###############################################################

from typing import Dict, Any, Optional
import os
import io
import pandas as pd
from xgboost import XGBClassifier


def align_and_predict_repositories(
    evaluation_df: pd.DataFrame,
    df: pd.DataFrame,
    X: pd.DataFrame,
    X_train: pd.DataFrame,
    trained_models: Dict[str, Any],
    best_model_name: str,
    encoder: Any,
    clean_xgb_fn=None
) -> pd.DataFrame:
    """
    Generate RQ2 predicted repository labels for all questions in evaluation_df.
    """
    eval_df = evaluation_df.copy()

    # 1. Get Question_IDs from evaluation_df
    evaluation_question_ids = eval_df['Question_ID'].tolist()

    # 2. Filter original X using matching Question_IDs
    original_df_indices_for_evaluation = df[df['Question_ID'].isin(evaluation_question_ids)].index
    X_evaluation_features = X.loc[original_df_indices_for_evaluation].copy()

    # 3. Apply column cleaning if XGBoost model
    best_model = trained_models[best_model_name]
    if clean_xgb_fn is not None and isinstance(best_model, (type(XGBClassifier()), type(XGBClassifier(random_state=42, eval_metric="mlogloss")))):
        X_evaluation_features = clean_xgb_fn(X_evaluation_features)

    # 4. Reindex X_evaluation_features to match X_train columns
    trained_columns = X_train.columns
    X_evaluation_features = X_evaluation_features.reindex(columns=trained_columns, fill_value=0)

    # 5. Predict repositories
    predicted_labels = best_model.predict(X_evaluation_features)

    # 6. Inverse transform
    predicted_repositories_names = encoder.inverse_transform(predicted_labels)

    # 7. Add 'Predicted Repository' column
    predicted_series = pd.Series(predicted_repositories_names, index=eval_df.index)
    eval_df['Predicted Repository'] = predicted_series

    return eval_df


def run_rq3_experiments(
    evaluation_df: pd.DataFrame,
    vector_dbs: Dict[str, Any],
    baseline_fn,
    ground_truth_fn,
    predicted_fn,
    all_repo_fn,
    retrieve_fn,
    sample_n: int = 100
) -> pd.DataFrame:
    """
    Execute all 4 response generation strategies across evaluation questions.
    """
    experiment_results = []
    subset_df = evaluation_df.head(sample_n)
    N = len(subset_df)

    for idx, row in subset_df.iterrows():
        question = row["Question"]
        print(f"Running Question {idx+1}/{N}")

        baseline = baseline_fn(question)

        gt = ground_truth_fn(
            question,
            row["Enterprise Knowledge Required"],
            vector_dbs=vector_dbs,
            retrieve_fn=retrieve_fn
        )

        predicted = predicted_fn(
            question,
            row["Predicted Repository"],
            vector_dbs=vector_dbs,
            retrieve_fn=retrieve_fn
        )

        all_repo = all_repo_fn(
            question,
            vector_dbs=vector_dbs,
            retrieve_fn=retrieve_fn
        )

        experiment_results.append({
            "Question": question,
            "Reference": row.get("Official Answer", row.get("Answer", "")),
            "Baseline": baseline,
            "GroundTruth": gt,
            "Predicted": predicted,
            "AllRepositories": all_repo
        })

    experiment_df = pd.DataFrame(experiment_results)

    try:
        display(experiment_df.head())
    except NameError:
        print(experiment_df.head())

    return experiment_df


def save_experiment_results(
    experiment_df: pd.DataFrame,
    output_path: str = "/content/drive/MyDrive/DataAnalytics_Capstone/rq3_generation_results.csv"
) -> str:
    """
    Save generation experiment results to CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    csv_string = experiment_df.to_csv(index=False)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(csv_string)
    print("Results saved.")
    return output_path


if __name__ == "__main__":
    print("Experiment runner and results persistence module ready.")

