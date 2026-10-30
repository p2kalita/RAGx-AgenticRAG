"""
## 2. Feature Engineering

Here, target class is "Enterprise Knowledge Required" with four classes. Lets encode it
"""

###############################################################
# LABEL ENCODING & FEATURE ENGINEERING
###############################################################

from typing import Tuple, List, Optional
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Updated list of columns to drop for X
DEFAULT_COLUMNS_TO_DROP = [
    "Enterprise Knowledge Required", # Target variable for RQ2
    "Repository_Label",            # Encoded target for RQ2
    "Question",                    # Free text
    "Question_ID",                 # Identifier
    "Source URL",                  # URL string
    "Question URL",                # URL string
    "Official Answer",             # Free text
    "Official Answer Summary",     # Free text
    "Official Documentation URL(s)", # Free text/URL
    "Tags",                        # Free text/high cardinality
    "Notes",                       # Free text
    "Publication Date",            # All NaN
    "Accepted Answer Available?",  # All NaN
    "Accepted Answer URL",         # All NaN
    "Public Knowledge Only?"       # This is the target for RQ1, not a feature for RQ2
]


def encode_target_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Fit LabelEncoder on 'Enterprise Knowledge Required' and add 'Repository_Label' to df.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing 'Enterprise Knowledge Required'.

    Returns
    -------
    Tuple[pd.DataFrame, LabelEncoder]
        Updated DataFrame and fitted LabelEncoder instance.
    """
    encoder = LabelEncoder()
    df["Repository_Label"] = encoder.fit_transform(
        df["Enterprise Knowledge Required"]
    )
    mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    print(mapping)
    return df, encoder


def prepare_features_and_target(
    df: pd.DataFrame,
    columns_to_drop: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder, List[str]]:
    """
    Execute feature selection, one-hot encoding, and target encoding.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset.
    columns_to_drop : Optional[List[str]], optional
        Columns to exclude from predictor matrix.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, LabelEncoder, List[str]]
        (X, y, encoder, cols_dropped)
    """
    cols_drop = columns_to_drop if columns_to_drop is not None else DEFAULT_COLUMNS_TO_DROP
    df_encoded, encoder = encode_target_labels(df.copy())

    # Create X by dropping the identified problematic columns from a copy of df
    existing_drop_cols = [c for c in cols_drop if c in df_encoded.columns]
    X = df_encoded.drop(columns=existing_drop_cols).copy()

    # Apply one-hot encoding to the remaining categorical columns
    X = pd.get_dummies(X, drop_first=True)

    # Convert any boolean columns (from get_dummies) to integer (0 or 1)
    # This also ensures all columns are numeric (int/float)
    X = X.astype(int)

    y = df_encoded["Repository_Label"]

    print(X.shape)
    print(y.value_counts())

    return X, y, encoder, cols_drop


if __name__ == "__main__":
    import os
    import importlib.util

    current_dir = os.path.dirname(os.path.abspath(__file__))
    loader_path = os.path.join(current_dir, "01_environment_and_data_loading.py")

    if os.path.exists(loader_path):
        spec_load = importlib.util.spec_from_file_location("step_01_loader", loader_path)
        mod_load = importlib.util.module_from_spec(spec_load)
        spec_load.loader.exec_module(mod_load)
        df = mod_load.load_cleaned_dataset()

        X, y, encoder, cols_dropped = prepare_features_and_target(df)

