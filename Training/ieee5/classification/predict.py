"""
ieee5/classification/predict.py
--------------------------------
Runs inference with the best classification model selected by mcdm.py.

Two operating modes
-------------------
--mode sim  (simulation)
    Input : any CSV with the 12 electrical features + optional label column.
            Typically used with cls_test.csv to verify the pipeline end-to-end.
            Features are already scaled — scaler is NOT applied.
    Output: predictions CSV + optional metrics report (if labels present).

--mode lab  (laboratory validation)
    Input : CSV with the 12 electrical features + label column (known fault
            type recorded during physical lab experiments).
            Features are raw sensor values — scaler IS applied.
    Output: predictions CSV + validation report JSON comparing predictions
            against real lab labels.

In both modes the pipeline is identical:
    1. Load best model from mcdm_result.json → models/<model_id>.pkl
    2. Apply cls_scaler.pkl only in lab mode (transform only — never refit)
    3. Predict fault type labels
    4. If labels present → compute weighted recall + per-class recall

Utility modules used
---------------------
- utils.io.load_config()  : load YAML config
- utils.io.load_model()   : load fitted model from .pkl

Usage (called from project root)
---------------------------------
    # Simulation mode (e.g. verify on test set)
    python3 ieee5/classification/predict.py \\
        --config ieee5/config.yaml \\
        --mode sim \\
        --input data/splits/ieee5/cls_test.csv

    # Laboratory validation mode
    python3 ieee5/classification/predict.py \\
        --config ieee5/config.yaml \\
        --mode lab \\
        --input path/to/lab_dataset.csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib as _jl
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score

# Add project root to path so utils/ is importable regardless of cwd
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.io import load_config, load_model   # shared I/O helpers


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "Va", "Vb", "Vc",
    "phi_Va", "phi_Vb", "phi_Vc",
    "Ia", "Ib", "Ic",
    "phi_Ia", "phi_Ib", "phi_Ic"
]

CLASSIFICATION_LABEL_COL = "label_classif"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_input_csv(
    csv_path: str,
) -> Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray]]:
    """Load an input CSV and separate features from label (if present).

    The CSV must contain the 12 electrical feature columns. The label
    column (label_classification) is optional — if absent, metrics cannot
    be computed but predictions are still produced.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV file.

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray]]
        (original dataframe, feature matrix X, label vector y or None)

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist.
    ValueError
        If any of the 12 expected feature columns are missing.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: '{path}'")

    df = pd.read_csv(path, sep=";", decimal=",")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in input CSV: {missing}")

    X = df[FEATURE_COLS].values

    y = None
    if CLASSIFICATION_LABEL_COL in df.columns:
        y = df[CLASSIFICATION_LABEL_COL].values
        classes, counts = np.unique(y, return_counts=True)
        print(f"[load]  '{path.name}'  →  {X.shape[0]:,} samples  "
              f"(labels present: {len(classes)} fault types)")
    else:
        print(f"[load]  '{path.name}'  →  {X.shape[0]:,} samples  "
              f"(no label column — metrics will not be computed)")

    return df, X, y


def load_scaler(scaler_path: str):
    """Load the pre-fitted StandardScaler from disk.

    Parameters
    ----------
    scaler_path : str
        Path to cls_scaler.pkl.

    Returns
    -------
    StandardScaler
        The fitted scaler instance.

    Raises
    ------
    FileNotFoundError
        If cls_scaler.pkl does not exist.
    """
    path = Path(scaler_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Scaler not found: '{path}'\n"
            "Run utils/split.py first."
        )
    scaler = _jl.load(path)
    print(f"[scaler] Loaded from '{path}'")
    return scaler


def load_best_model_id(results_dir: Path) -> str:
    """Read the best model identifier from mcdm_result.json.

    Parameters
    ----------
    results_dir : Path
        Directory containing mcdm_result.json.

    Returns
    -------
    str
        Model identifier (e.g. 'RF', 'XGB', 'KNN').

    Raises
    ------
    FileNotFoundError
        If mcdm_result.json does not exist.
    """
    path = results_dir / "mcdm_result.json"
    if not path.exists():
        raise FileNotFoundError(
            f"MCDM result not found: '{path}'\n"
            "Run ieee5/classification/mcdm.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        result = json.load(f)

    best_model = result["best_model"]
    print(f"[mcdm]  Best model from mcdm_result.json: '{best_model}'")
    return best_model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict:
    """Compute weighted recall, per-class recall and confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth fault-type labels.
    y_pred : np.ndarray
        Predicted fault-type labels.

    Returns
    -------
    Dict
        weighted_recall, per_class_recall, confusion_matrix, classes.
    """
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))

    weighted_recall = recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    per_class = recall_score(
        y_true, y_pred,
        labels        = classes,
        average       = None,
        zero_division = 0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()

    return {
        "weighted_recall":  round(float(weighted_recall), 6),
        "per_class_recall": {
            cls: round(float(val), 6)
            for cls, val in zip(classes, per_class)
        },
        "confusion_matrix": cm,
        "classes":          classes,
    }


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_predictions_csv(
    df:       pd.DataFrame,
    y_pred:   np.ndarray,
    out_path: Path,
) -> None:
    """Append prediction column to the input dataframe and save as CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Original input dataframe.
    y_pred : np.ndarray
        Predicted fault-type labels.
    out_path : Path
        Destination CSV path.
    """
    out_df = df.copy()
    out_df["pred_fault_type"] = y_pred
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, sep=";", decimal=",", index=False)
    print(f"[save]  Predictions saved → '{out_path}'")


def save_validation_report(
    mode:      str,
    model_id:  str,
    metrics:   Optional[Dict],
    n_samples: int,
    out_path:  Path,
) -> None:
    """Save a validation report as JSON.

    Parameters
    ----------
    mode : str
        'sim' or 'lab'.
    model_id : str
    metrics : Dict or None
        None if no labels were available.
    n_samples : int
    out_path : Path
    """
    report = {
        "mode":      mode,
        "model_id":  model_id,
        "n_samples": n_samples,
        "metrics":   metrics,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[save]  Validation report saved → '{out_path}'")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(config_path: str, mode: str, input_csv: str) -> None:
    """Run classification inference in simulation or laboratory mode.

    Steps
    -----
    1. Load config and best model ID from mcdm_result.json.
    2. Load input CSV → features X and optional labels y.
    3. Apply cls_scaler.pkl only in lab mode (transform only).
    4. Run inference with model.predict().
    5. If labels present → compute weighted recall + per-class recall.
    6. Save predictions CSV and validation report JSON.

    Parameters
    ----------
    config_path : str
        Path to config.yaml.
    mode : str
        'sim' or 'lab'.
    input_csv : str
        Path to input CSV with the 12 electrical features.
    """
    cfg = load_config(config_path)

    network       = cfg["network"]["name"]
    processed_dir = Path(cfg["data"]["processed_dir"])
    cls_cfg       = cfg["classification"]
    models_dir    = Path(cls_cfg["models_dir"])
    results_dir   = Path(cls_cfg["results_dir"])

    print(f"\n{'='*60}")
    print(f"  Classification — predict  [{network}]  mode={mode}")
    print(f"{'='*60}\n")

    # Load best model from MCDM output
    best_model_id = load_best_model_id(results_dir)
    model         = load_model(models_dir, best_model_id)

    # Load input CSV
    df, X, y = load_input_csv(input_csv)

    # Apply scaler only in lab mode — sim CSVs are already scaled
    if mode == "lab":
        scaler   = load_scaler(str(processed_dir / "cls_scaler.pkl"))
        X_scaled = scaler.transform(X)
        print(f"[scaler] Applied to {X_scaled.shape[0]:,} samples")
    else:
        X_scaled = X
        print(f"[scaler] Skipped — sim input is already scaled")

    # Inference
    y_pred = model.predict(X_scaled)
    # Decode if model is XGBoost
    le_path = models_dir / "label_encoder.pkl"
    if best_model_id == "XGB" and le_path.exists():
        le = _jl.load(le_path)
        y_pred = le.inverse_transform(y_pred)

    # Class distribution of predictions
    pred_classes, pred_counts = np.unique(y_pred, return_counts=True)
    print(f"\n[predict] {len(y_pred):,} samples classified:")
    for cls, cnt in zip(pred_classes, pred_counts):
        print(f"  {cls:<12}  {cnt:>5}  ({cnt/len(y_pred):.1%})")

    # Metrics (only if labels available)
    metrics = None
    if y is not None:
        metrics = compute_metrics(y, y_pred)
        print(f"\n  Metrics vs ground truth:")
        print(f"  weighted_recall = {metrics['weighted_recall']:.4f}")
        print(f"\n  Per-class recall:")
        for cls, val in metrics["per_class_recall"].items():
            print(f"    {cls:<12}  {val:.4f}")

    # Save outputs
    input_stem  = Path(input_csv).stem
    pred_path   = results_dir / f"predictions_{mode}_{input_stem}.csv"
    report_path = results_dir / f"validation_{mode}_{input_stem}.json"

    save_predictions_csv(df, y_pred, pred_path)
    save_validation_report(mode, best_model_id, metrics, len(y_pred), report_path)

    print(f"\n[done]  Prediction complete for {network}  (mode={mode}).")
    print(f"        Results → '{results_dir}'\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run fault classification inference (simulation or lab mode)."
    )
    parser.add_argument(
        "--config",
        type     = str,
        required = True,
        help     = "Path to config.yaml (e.g. 'ieee5/config.yaml')",
    )
    parser.add_argument(
        "--mode",
        type     = str,
        required = True,
        choices  = ["sim", "lab"],
        help     = "'sim' for simulation data, 'lab' for laboratory data",
    )
    parser.add_argument(
        "--input",
        type     = str,
        required = True,
        help     = "Path to input CSV with the 12 electrical features",
    )
    args = parser.parse_args()

    try:
        main(args.config, args.mode, args.input)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
