"""
Clone the GitHub repository containing the enterprise knowledge repository.

Load the evaluation questions used for the RQ3 experiment.
"""

###############################################################
# RQ3 - STEP 1 : CLONE REPOSITORY & LOAD EVALUATION DATASET
###############################################################

import os
import subprocess
import pandas as pd


def clone_knowledge_repository(
    repo_url: str = "https://github.com/Xquisite-design/enterprise-presales-knowledge-orchestration.git",
    destination_dir: str = "/content/enterprise-presales-knowledge-orchestration"
) -> str:
    """
    Download the enterprise knowledge repository containing public and enterprise documents.

    Parameters
    ----------
    repo_url : str
        Git repository URL.
    destination_dir : str
        Target local directory.

    Returns
    -------
    str
        Path to local clone.
    """
    if not os.path.exists(destination_dir):
        print(f"Cloning {repo_url} into {destination_dir}...")
        subprocess.run(["git", "clone", repo_url, destination_dir], check=True)
        print("Repository downloaded successfully.")
    else:
        print(f"Repository directory {destination_dir} already exists.")
    return destination_dir


def load_evaluation_dataset(
    eval_csv_path: str = "/content/enterprise-presales-knowledge-orchestration/research_results/rq2/evaluation_dataset.csv"
) -> pd.DataFrame:
    """
    Load the evaluation questions used for the RQ3 experiment.

    Parameters
    ----------
    eval_csv_path : str
        Path to evaluation dataset CSV.

    Returns
    -------
    pd.DataFrame
        Evaluation DataFrame containing questions, ground truth repositories, and official answers.
    """
    evaluation_df = pd.read_csv(eval_csv_path)
    print("Evaluation dataset shape:", evaluation_df.shape)
    return evaluation_df


if __name__ == "__main__":
    clone_knowledge_repository()
    eval_df = load_evaluation_dataset()

