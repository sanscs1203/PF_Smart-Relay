"""
utils/io.py
-----------
Shared I/O and configuration helpers used across all detection and
classification scripts (train.py, evaluate.py, predict.py, mcdm.py).

Functions
---------
- load_config()           : load a YAML configuration file
- load_split()            : load a split CSV → (X, y) numpy arrays
- load_model()            : load a fitted model from a .pkl file
- verify_scaler()         : verify scaler.pkl exists on disk
- build_pairwise_matrix() : reconstruct 3×3 AHP matrix from config values

Design note
-----------
These functions contain no business logic — they are pure I/O helpers.
Any module-specific logic (AHP scoring, threshold sweep, MCDM) lives in
the corresponding scripts or utility modules (ahp_weights, shannon_entropy,
monte_carlo).
"""

import joblib
import numpy as np
import pandas as pd
import yaml

from pathlib import Path
from typing import Dict, Tuple    # Python 3.6 compatible type hints

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict:
    """Load a YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to config.yaml (e.g. 'ieee5/config.yaml').

    Returns
    -------
    Dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(csv_path: str, label_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a split CSV and return (X, y) as numpy arrays.

    The CSV must follow the project standard format:
    separator=';', decimal=','.

    Parameters
    ----------
    csv_path : str
        Path to the split CSV (e.g. 'data/splits/ieee5/det_train.csv').
    label_col : str
        Name of the label column to separate as y.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Feature matrix X and label vector y.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}\n"
            "Run utils/preprocess.py and utils/split.py first."
        )
    df = pd.read_csv(path, sep=";", decimal=",")
    X  = df.drop(columns=[label_col]).values
    y  = df[label_col].values
    print(f"[load]  '{path.name}'  →  {X.shape[0]:,} samples, {X.shape[1]} features")
    return X, y


def load_model(models_dir: Path, model_id: str):
    """Load a trained model from a .pkl file.

    Parameters
    ----------
    models_dir : Path
        Directory containing .pkl model files.
    model_id : str
        Model identifier used as filename stem (e.g. 'RF' → 'RF.pkl').

    Returns
    -------
    Fitted sklearn estimator.

    Raises
    ------
    FileNotFoundError
        If the .pkl file does not exist.
    """
    path = models_dir / f"{model_id}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "Run the corresponding train.py first."
        )
    model = joblib.load(path)
    print(f"[load]  Model '{model_id}' loaded from '{path}'")
    return model


def verify_scaler(scaler_path: str) -> None:
    """Verify the pre-fitted StandardScaler exists on disk.

    The split CSVs produced by utils/split.py are already scaled.
    This function is an existence check only — it does NOT load or
    re-apply the scaler, to prevent double-scaling.

    Parameters
    ----------
    scaler_path : str
        Path to scaler.pkl.

    Raises
    ------
    FileNotFoundError
        If scaler.pkl does not exist.
    """
    path = Path(scaler_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Scaler not found: {path}\n"
            "Run utils/split.py first."
        )
    print(f"[scaler] Found at '{path}'  (splits already scaled — no re-scaling)")


# ---------------------------------------------------------------------------
# AHP matrix reconstruction
# ---------------------------------------------------------------------------

def build_pairwise_matrix(pairwise_cfg: Dict) -> np.ndarray:
    """Reconstruct the 2×2 AHP pairwise matrix from config.yaml values.

    The config stores only the three upper-triangle judgements:
        Recall_vs_Specificity

    The full reciprocal matrix (row/col order: Recall, Specificity, ROC_AUC):

        [[1,          R_vs_S   ],
         [1/R_vs_S,   1        ]]

    This function is shared by train.py, evaluate.py, and mcdm.py to
    guarantee that the same AHP weights are used consistently across
    all pipeline stages.

    Parameters
    ----------
    pairwise_cfg : Dict
        Sub-dict from config.yaml: detection.ahp.pairwise_matrix
        Expected keys: Recall_vs_Specificity.

    Returns
    -------
    np.ndarray, shape (2, 2)
        Positive reciprocal pairwise comparison matrix.
    """
    r_s = float(pairwise_cfg["Recall_vs_Specificity"])

    return np.array([
        [1.0,    r_s],   # Recall
        [1/r_s,  1.0],   # Specificity
    ])