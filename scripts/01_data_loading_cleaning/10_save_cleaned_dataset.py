"""
## Save Cleaned Dataset

Save the cleaned dataset so the following notebooks can load it without repeating the steps above.
"""

###############################################################
# SAVE CLEANED DATASET FOR THE NEXT NOTEBOOKS
###############################################################

import os
import pandas as pd


def save_cleaned_dataset(
    df: pd.DataFrame,
    base_dir: str = "/content/drive/MyDrive/DataAnalytics_Capstone",
    handoff_subpath: str = "handoff",
    output_filename: str = "cleaned_dataset.pkl"
) -> str:
    """
    Save the cleaned dataset to the handoff directory for subsequent downstream notebooks.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset to serialize.
    base_dir : str
        Base storage path for project artifacts.
    handoff_subpath : str
        Directory subpath designated for intermediate pipeline handoffs.
    output_filename : str
        Filename of the serialized pickle archive.

    Returns
    -------
    str
        Full path to the persisted pickle artifact.
    """
    handoff_dir = f"{base_dir}/{handoff_subpath}"
    os.makedirs(handoff_dir, exist_ok=True)
    output_path = f"{handoff_dir}/{output_filename}"

    df.to_pickle(output_path)
    print("Cleaned dataset saved.")
    return output_path


if __name__ == "__main__":
    import importlib.util

    current_dir = os.path.dirname(os.path.abspath(__file__))
    cleaning_path = os.path.join(current_dir, "09_data_cleaning_and_log.py")
    loader_path = os.path.join(current_dir, "02_dataset_loading.py")

    if os.path.exists(loader_path) and os.path.exists(cleaning_path):
        spec_load = importlib.util.spec_from_file_location("step_02_dataset_loading", loader_path)
        mod_load = importlib.util.module_from_spec(spec_load)
        spec_load.loader.exec_module(mod_load)
        df_raw = mod_load.load_dataset()

        spec_clean = importlib.util.spec_from_file_location("step_09_data_cleaning_and_log", cleaning_path)
        mod_clean = importlib.util.module_from_spec(spec_clean)
        spec_clean.loader.exec_module(mod_clean)
        df_cleaned, _ = mod_clean.clean_and_log(df_raw)

        save_cleaned_dataset(df_cleaned)
