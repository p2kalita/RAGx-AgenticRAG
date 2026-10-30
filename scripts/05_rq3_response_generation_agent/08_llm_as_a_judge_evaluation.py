"""
LLM-as-a-Judge: Each response will be scored on:
- accuracy
- completeness
- enterprise_relevance
- technical_correctness
- actionability
- overall

Response Quality Index (RQI): This gives us a single metric for comparing responses across strategies.
"""

###############################################################
# RQ3 - STEP 21, 22, 23 : LLM-AS-A-JUDGE & RQI CALCULATION
###############################################################

from typing import Dict, Any, Optional
import os
import re
import json
import time
import pandas as pd
from openai import OpenAI

DEFAULT_MODEL = "gpt-4.1"


def evaluate_response(
    question: str,
    reference: str,
    generated: str,
    client: Optional[OpenAI] = None,
    model: str = DEFAULT_MODEL
) -> str:
    """
    Evaluate generated response against reference using LLM-as-a-Judge returning structured JSON scores.
    """
    cli = client if client is not None else OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    prompt = f"""
You are an expert evaluator of enterprise presales responses.

Question:
{question}

Reference Answer:
{reference}

Generated Answer:
{generated}

Evaluate the generated answer against the reference.

Return ONLY JSON.

{{
"accuracy":0,
"completeness":0,
"enterprise_relevance":0,
"technical_correctness":0,
"actionability":0,
"overall":0
}}
"""
    # Uses responses.create (or chat.completions fallback) matching notebook implementation
    try:
        response = cli.responses.create(
            model=model,
            input=prompt
        )
        return response.output_text
    except AttributeError:
        response = cli.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content


def calculate_rqi(
    accuracy: float,
    completeness: float,
    enterprise_relevance: float,
    technical_correctness: float,
    actionability: float
) -> float:
    """
    Calculate the composite Response Quality Index (RQI) as the unweighted mean of all 5 dimensions.
    """
    return (
        accuracy +
        completeness +
        enterprise_relevance +
        technical_correctness +
        actionability
    ) / 5


def parse_llm_json_output_robustly(json_string_candidate: Any) -> Dict[str, Any]:
    """
    Helper function to parse LLM JSON output robustly, stripping markdown fences and trailing tokens.
    """
    if not isinstance(json_string_candidate, str):
        return {"overall": 0, "accuracy": 0, "completeness": 0, "enterprise_relevance": 0, "technical_correctness": 0, "actionability": 0}

    cleaned_string = json_string_candidate.strip()
    cleaned_string = re.sub(r'^```json\s*|\s*```$', '', cleaned_string, flags=re.DOTALL).strip()

    try:
        return json.loads(cleaned_string)
    except json.JSONDecodeError as e:
        if "Extra data" in str(e) and e.pos is not None:
            try:
                return json.loads(cleaned_string[:e.pos])
            except json.JSONDecodeError:
                pass
        print(f"Warning: Failed to parse LLM JSON output. Returning default scores. Error: {e}.")
        return {"overall": 0, "accuracy": 0, "completeness": 0, "enterprise_relevance": 0, "technical_correctness": 0, "actionability": 0}
    except Exception as e:
        print(f"Warning: Unexpected error during JSON parsing: {e}.")
        return {"overall": 0, "accuracy": 0, "completeness": 0, "enterprise_relevance": 0, "technical_correctness": 0, "actionability": 0}


def run_llm_judge(
    experiment_df: pd.DataFrame,
    client: Optional[OpenAI] = None,
    model: str = DEFAULT_MODEL
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute LLM Judge across all strategies and parse evaluation scores into analysis_df.
    """
    print("Running LLM-as-a-Judge evaluation...")
    judge_start_time = time.time()

    exp_df = experiment_df.copy()
    exp_df["Baseline_Judge"] = ""
    exp_df["GroundTruth_Judge"] = ""
    exp_df["Predicted_Judge"] = ""
    exp_df["AllRepositories_Judge"] = ""

    for idx, row in exp_df.iterrows():
        question = row["Question"]
        reference = row["Reference"]

        exp_df.at[idx, "Baseline_Judge"] = evaluate_response(question, reference, row["Baseline"], client, model)
        exp_df.at[idx, "GroundTruth_Judge"] = evaluate_response(question, reference, row["GroundTruth"], client, model)
        exp_df.at[idx, "Predicted_Judge"] = evaluate_response(question, reference, row["Predicted"], client, model)
        exp_df.at[idx, "AllRepositories_Judge"] = evaluate_response(question, reference, row["AllRepositories"], client, model)

    judge_end_time = time.time()
    print(f"LLM-as-a-Judge evaluation completed in {round(judge_end_time - judge_start_time, 2)} seconds.")

    # Prepare analysis dataset
    score_rows = []
    for _, row in exp_df.iterrows():
        b_scores = parse_llm_json_output_robustly(row["Baseline_Judge"])
        gt_scores = parse_llm_json_output_robustly(row["GroundTruth_Judge"])
        p_scores = parse_llm_json_output_robustly(row["Predicted_Judge"])
        all_scores = parse_llm_json_output_robustly(row["AllRepositories_Judge"])

        score_rows.append({
            "Question": row["Question"],
            "Baseline": b_scores.get("overall", calculate_rqi(b_scores.get("accuracy", 0), b_scores.get("completeness", 0), b_scores.get("enterprise_relevance", 0), b_scores.get("technical_correctness", 0), b_scores.get("actionability", 0))),
            "GroundTruth": gt_scores.get("overall", calculate_rqi(gt_scores.get("accuracy", 0), gt_scores.get("completeness", 0), gt_scores.get("enterprise_relevance", 0), gt_scores.get("technical_correctness", 0), gt_scores.get("actionability", 0))),
            "Predicted": p_scores.get("overall", calculate_rqi(p_scores.get("accuracy", 0), p_scores.get("completeness", 0), p_scores.get("enterprise_relevance", 0), p_scores.get("technical_correctness", 0), p_scores.get("actionability", 0))),
            "AllRepositories": all_scores.get("overall", calculate_rqi(all_scores.get("accuracy", 0), all_scores.get("completeness", 0), all_scores.get("enterprise_relevance", 0), all_scores.get("technical_correctness", 0), all_scores.get("actionability", 0)))
        })

    analysis_df = pd.DataFrame(score_rows)
    return exp_df, analysis_df


def save_analysis_dataset(
    analysis_df: pd.DataFrame,
    output_path: str = "/content/drive/MyDrive/DataAnalytics_Capstone/rq3_analysis_df"
) -> str:
    """
    Persist analysis DataFrame to CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    csv_string = analysis_df.to_csv(index=False)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(csv_string)
    print("Analysis saved.")
    return output_path


if __name__ == "__main__":
    print("LLM-as-a-Judge and RQI scoring module ready.")

