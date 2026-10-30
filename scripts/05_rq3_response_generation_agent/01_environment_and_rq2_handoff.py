"""
<a href="https://colab.research.google.com/github/Xquisite-design/enterprise-presales-knowledge-orchestration/blob/main/notebooks/05_RQ3_Response_Generation_Agent.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Part 5 of 5 : RQ3 - Response Generation Agent

Multi-repository vector indexing, retrieval-augmented generation strategies (Baseline, Ground Truth, Predicted Routing, All Repositories),
LLM-as-a-Judge evaluation, and statistical hypothesis testing (ANOVA, Friedman, Pairwise t-tests, Cohen's d).

#PART C: RQ3 - RESPONSE GENERATION AGENT
"""

###############################################################
# SETUP : MOUNT DRIVE, IMPORTS AND LOAD HANDOFF DATA
###############################################################

from typing import Tuple, Dict, Any
import pickle
import time
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier


def mount_drive(mount_point: str = "/content/drive") -> None:
    """
    Mount Google Drive storage for accessing datasets and handoff artifacts.
    """
    from google.colab import drive
    drive.mount(mount_point)


def clean_xgb_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to clean column names for XGBoost compatibility.
    Replaces non-alphanumeric characters with underscores and strips leading/trailing underscores.
    """
    cols = df.columns
    new_cols = []
    for col in cols:
        new_col = re.sub('[^A-Za-z0-9_]+', '_', col)
        new_col = new_col.strip('_')
        new_cols.append(new_col)
    df.columns = new_cols
    return df


def load_rq2_handoff_bundle(
    base_dir: str = "/content/drive/MyDrive/DataAnalytics_Capstone",
    handoff_subpath: str = "handoff",
    bundle_filename: str = "rq2_routing_bundle.pkl"
) -> Dict[str, Any]:
    """
    Load the serialized Knowledge Routing Agent bundle from RQ2 handoff directory.

    Parameters
    ----------
    base_dir : str
        Base storage path.
    handoff_subpath : str
        Directory subpath.
    bundle_filename : str
        Bundle pickle filename.

    Returns
    -------
    Dict[str, Any]
        Dictionary of RQ2 artifacts: df, X, X_train, encoder, trained_models, best_model_name, columns_to_drop.
    """
    handoff_dir = f"{base_dir}/{handoff_subpath}"
    bundle_path = f"{handoff_dir}/{bundle_filename}"

    with open(bundle_path, "rb") as f:
        rq2_bundle = pickle.load(f)

    best_model_name = rq2_bundle["best_model_name"]
    print("Best routing model:", best_model_name)
    return rq2_bundle


if __name__ == "__main__":
    mount_drive()
    bundle = load_rq2_handoff_bundle()

