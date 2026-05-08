"""
ieee5/detection/predict.py
---------------------------
Runs inference with the best detection model selected by mcdm.py.

Two operating modes
-------------------
--mode sim  (simulation)
    Input : any CSV with the 12 electrical features + optional label column.
            Typically used with det_test.csv to verify the pipeline end-to-end.
    Output: predictions CSV + optional metrics report (if labels present).

--mode lab  (laboratory validation)
    Input : CSV with the 12 electrical features + label column (known fault
            type recorded during physical lab experiments).
            Features are magnitudes and phase angles derived via FFT from
            real sensor measurements — same 12-column format as simulation.
    Output: predictions CSV + validation report JSON comparing predictions
            against real lab labels.
            The same det_scaler.pkl trained on simulation data is applied
            (scale assumed comparable between simulation and lab).

In both modes the pipeline is identical:
    1. Load best model from mcdm_result.json → models/<model_id>.pkl
    2. Load and apply det_scaler.pkl (transform only — never refit)
    3. Predict labels and probabilities
    4. If labels present → compute metrics and save report

Utility modules used
---------------------
- utils.io.load_config()   : load YAML config
- utils.io.load_model()    : load fitted model from .pkl

Usage (called from project root)
---------------------------------
    # Simulation mode (e.g. verify on test set)
    python3 ieee5/detection/predict.py \\
        --config ieee5/config.yaml \\
        --mode sim \\
        --input data/splits/ieee5/det_test.csv

    # Laboratory validation mode
    python ieee5/detection/predict.py \\
        --config ieee5/config.yaml \\
        --mode lab \\
        --input path/to/lab_dataset.csv
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    recall_score
)

# Add project root to path so utils/ is importable regardless of cwd
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.io import load_config, load_model


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "Va", "Vb", "Vc",
    "phi_Va", "phi_Vb", "phi_Vc",
    "Ia", "Ib", "Ic",
    "phi_Ia", "phi_Ib", "phi_Ic",
]

DETECTION_LABEL_COL = "label_detection"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_input_csv(
    csv_path: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    """Load an input CSV and separate features from label (if present).

    The CSV must contain the 12 electrical feature columns. The label
    column (label_detection) is optional — if absent, metrics cannot
    be computed but predictions are still produced.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV file.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray, np.ndarray | None]
        (original dataframe, feature matrix X, label vector y or None)

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist.
    ValueError
        If any of the 6 expected feature columns are missing.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path, sep=";", decimal=",")

    # Validate feature columns
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in input CSV: {missing}")

    X = df[FEATURE_COLS].values

    # Label column is optional
    y = None
    if DETECTION_LABEL_COL in df.columns:
        y = df[DETECTION_LABEL_COL].values
        print(f"[load]  '{path.name}'  →  {X.shape[0]:,} samples  "
              f"(labels present: fault={y.mean():.1%}  "
              f"no-fault={1-y.mean():.1%})")
    else:
        print(f"[load]  '{path.name}'  →  {X.shape[0]:,} samples  "
              f"(no label column — metrics will not be computed)")

    return df, X, y


def load_scaler(scaler_path: str):
    """Load the pre-fitted StandardScaler from disk.

    Parameters
    ----------
    scaler_path : str
        Path to det_scaler.pkl.

    Returns
    -------
    StandardScaler
        The fitted scaler instance.

    Raises
    ------
    FileNotFoundError
        If det_scaler.pkl does not exist.
    """
    path = Path(scaler_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Scaler not found: {path}\n"
            "Run utils/split.py first."
        )
    scaler = joblib.load(path)
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
        Model identifier (e.g. 'RF', 'SVM', 'MLP').

    Raises
    ------
    FileNotFoundError
        If mcdm_result.json does not exist.
    """
    path = results_dir / "mcdm_result.json"
    if not path.exists():
        raise FileNotFoundError(
            f"MCDM result not found: {path}\n"
            "Run ieee5/detection/mcdm.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        result = json.load(f)

    best_model = result["best_model"]
    w_ahp      = np.array(result["weights"]["ahp"])
    print(f"[mcdm]  Best model from mcdm_result.json: '{best_model}'")
    return best_model, w_ahp


def load_threshold(results_dir: Path, model_id: str) -> float:
    """Read the optimal threshold for the best model from metrics JSON.

    Parameters
    ----------
    results_dir : Path
        Directory containing metrics_<model_id>.json.
    model_id : str
        Model identifier.

    Returns
    -------
    float
        Optimal decision threshold selected on the val set.

    Raises
    ------
    FileNotFoundError
        If metrics_<model_id>.json does not exist.
    """
    path = results_dir / f"metrics_{model_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {path}\n"
            "Run ieee5/detection/evaluate.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    threshold = metrics["optimal_threshold"]
    print(f"[threshold] Optimal threshold for {model_id}: {threshold:.2f}")
    return float(threshold)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(
    model,
    X_scaled:  np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and apply the optimal decision threshold.

    Parameters
    ----------
    model : fitted sklearn estimator
    X_scaled : np.ndarray
        Already-scaled feature matrix.
    threshold : float
        Decision threshold (probability ≥ threshold → fault = 1).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (predicted labels, positive-class probabilities)
    """
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_scaled)[:, 1]
    else:
        scores = model.decision_function(X_scaled)
        y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

    y_pred = (y_prob >= threshold).astype(int)
    return y_pred, y_prob


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    weights:  np.ndarray,
) -> dict:
    """Compute detection metrics against ground-truth labels.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels.
    y_pred : np.ndarray
        Predicted binary labels.
    y_prob : np.ndarray
        Positive-class probabilities.

    Returns
    -------
    dict
        recall, specificity, ahp_score, confusion_matrix.
    """
    recall      = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    ahp_score   = weights[0] * recall + weights[1] * specificity
    cm          = confusion_matrix(y_true, y_pred).tolist()

    return {
        "recall":           round(float(recall),      6),
        "specificity":      round(float(specificity), 6),
        "ahp_score":        round(float(ahp_score),   6),
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_predictions_csv(
    df:        pd.DataFrame,
    y_pred:    np.ndarray,
    y_prob:    np.ndarray,
    out_path:  Path,
) -> None:
    """Append prediction and probability columns to the input dataframe
    and save as CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Original input dataframe.
    y_pred : np.ndarray
        Predicted binary labels.
    y_prob : np.ndarray
        Positive-class probabilities.
    out_path : Path
        Destination CSV path.
    """
    out_df = df.copy()
    out_df["pred_label"]    = y_pred
    out_df["pred_prob_fault"] = np.round(y_prob, 6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, sep=";", decimal=",", index=False)
    print(f"[save]  Predictions saved → '{out_path}'")


def save_validation_report(
    mode:      str,
    model_id:  str,
    threshold: float,
    metrics:   dict | None,
    n_samples: int,
    out_path:  Path,
) -> None:
    """Save a validation report as JSON.

    Parameters
    ----------
    mode : str
        'sim' or 'lab'.
    model_id : str
    threshold : float
    metrics : dict or None
        None if no labels were available.
    n_samples : int
    out_path : Path
    """
    report = {
        "mode":      mode,
        "model_id":  model_id,
        "threshold": round(threshold, 4),
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
    """Run detection inference in simulation or laboratory mode.

    Steps
    -----
    1. Load config, best model ID (mcdm_result.json), optimal threshold.
    2. Load input CSV → features X and optional labels y.
    3. Apply det_scaler.pkl (transform only).
    4. Run inference with optimal threshold.
    5. If labels present → compute metrics.
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
    det_cfg       = cfg["detection"]
    models_dir    = Path(det_cfg["models_dir"])
    results_dir   = Path(det_cfg["results_dir"])

    print(f"\n{'='*60}")
    print(f"  Detection — predict  [{network}]  mode={mode}")
    print(f"{'='*60}\n")

    # Load best model and threshold from MCDM + evaluate outputs
    best_model_id, w_ahp = load_best_model_id(results_dir)
    threshold     = load_threshold(results_dir, best_model_id)
    model         = load_model(models_dir, best_model_id)

    # Load input CSV
    df, X, y = load_input_csv(input_csv)

    # Apply scaler (transform only — same scaler as training)
    if mode == "lab":
        scaler = load_scaler(str(processed_dir / "det_scaler.pkl"))
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X
    print(f"[scaler] Applied to {X_scaled.shape[0]:,} samples")

    # Inference
    y_pred, y_prob = predict(model, X_scaled, threshold)
    n_fault    = int(y_pred.sum())
    n_no_fault = len(y_pred) - n_fault
    print(f"[predict] fault={n_fault}  no-fault={n_no_fault}  "
          f"(threshold={threshold:.2f})")

    # Metrics (only if labels available)
    metrics = None
    if y is not None:
        metrics = compute_metrics(y, y_pred, y_prob, w_ahp)
        print(f"\n  Metrics vs ground truth:")
        print(f"  recall      = {metrics['recall']:.4f}")
        print(f"  specificity = {metrics['specificity']:.4f}")
        print(f"  ahp_score   = {metrics['ahp_score']:.4f}")
        cm = metrics["confusion_matrix"]
        print(f"  confusion matrix:")
        print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
        print(f"    FN={cm[1][0]}  TP={cm[1][1]}")

    # Save outputs
    input_stem  = Path(input_csv).stem
    pred_path   = results_dir / f"predictions_{mode}_{input_stem}.csv"
    report_path = results_dir / f"validation_{mode}_{input_stem}.json"

    save_predictions_csv(df, y_pred, y_prob, pred_path)
    save_validation_report(mode, best_model_id, threshold,
                           metrics, len(y_pred), report_path)

    print(f"\n[done]  Prediction complete for {network}  (mode={mode}).")
    print(f"        Results → '{results_dir}'\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run fault detection inference (simulation or lab mode)."
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