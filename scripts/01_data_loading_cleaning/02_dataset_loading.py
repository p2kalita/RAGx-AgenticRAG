"""
## 1. Dataset loading and inspection

This module handles loading the Curated Enterprise Presales Research Corpus (CEPRC)
from the research Excel workbook into a pandas DataFrame.
"""

import pandas as pd


def load_dataset(
    file_path: str = "/content/drive/MyDrive/DataAnalytics_Capstone/Enterprise_Presales_Knowledge_Research_Dataset_V1.xlsx",
    sheet_name: str = "Enterprise_QA_Dataset",
) -> pd.DataFrame:
    """
    Load the enterprise presales QA dataset from the Excel workbook sheet.

    Parameters
    ----------
    file_path : str
        Path to the Excel dataset file.
    sheet_name : str
        Name of the sheet containing the QA dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the raw enterprise presales QA corpus.
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df


if __name__ == "__main__":
    df = load_dataset()
    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")
