"""
<a href="https://colab.research.google.com/github/Xquisite-design/enterprise-presales-knowledge-orchestration/blob/main/notebooks/03_RQ1_Enterprise_Knowledge_Requirement.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Part 3 of 5 : RQ1 - Enterprise Knowledge Requirement

Inferential statistics, feature engineering, baseline models, hyperparameter tuning and evaluation for RQ1 (predicting Public Knowledge Only?).

**Run first:** `02_Exploratory_Data_Analysis.ipynb` &nbsp;|&nbsp; **Next:** `04_RQ2_Knowledge_Routing_Agent.ipynb`
"""

###############################################################
# SETUP : MOUNT DRIVE, IMPORTS AND LOAD HANDOFF DATA
###############################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def mount_drive(mount_point: str = "/content/drive") -> None:
    """
    Mount Google Drive storage for accessing datasets and handoff artifacts.
    """
    from google.colab import drive
    drive.mount(mount_point)


def load_cleaned_dataset(
    base_dir: str = "/content/drive/MyDrive/DataAnalytics_Capstone",
    handoff_subpath: str = "handoff",
    filename: str = "cleaned_dataset.pkl"
) -> pd.DataFrame:
    """
    Load cleaned dataset saved by 01_Data_Loading_Quality_Cleaning.ipynb from handoff directory.

    Parameters
    ----------
    base_dir : str
        Base storage path for project artifacts.
    handoff_subpath : str
        Directory subpath designated for intermediate pipeline handoffs.
    filename : str
        Filename of the serialized pickle artifact.

    Returns
    -------
    pd.DataFrame
        Loaded cleaned enterprise presales DataFrame.
    """
    handoff_dir = f"{base_dir}/{handoff_subpath}"
    df = pd.read_pickle(f"{handoff_dir}/{filename}")
    print(df.shape)
    return df


if __name__ == "__main__":
    mount_drive()
    df = load_cleaned_dataset()

