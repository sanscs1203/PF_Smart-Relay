"""
ieee5/classification/evaluate.py
----------------------------------
Evaluates the three trained classification models (RF, XGBoost, kNN)
on the IEEE 5-bus network.

Responsibilities
----------------
- Load each trained model via utils.io.load_model().
- Load cls_val.csv and cls_test.csv via utils.io.load_split().
- Evaluate each model on the val set and the test set using
  weighted recall as the single selection criterion.
- Save per-model JSON reports (val + test metrics + confusion matrix).
- Build and save the decision matrix (n_models × 1) for mcdm.py.

What this script does NOT do
-----------------------------
- Does not optimise a decision threshold  →  not applicable
                                              (multiclass uses predict() directly)
- Does not select the best model          →  mcdm.py
- Does not run Monte Carlo                →  mcdm.py
- Does not generate plots                 →  utils/plots.py

Utility modules used
---------------------
- utils.io.load_config()  : load YAML config
- utils.io.load_split()   : load split CSV → (X, y)
- utils.io.load_model()   : load fitted model from .pkl

Output files (results/)
-----------------------
- metrics_RF.json         : val + test metrics + confusion matrix
- metrics_XGB.json        : idem
- metrics_KNN.json        : idem
- decision_matrix.json    : (n_models × 1) weighted recall on test set

Usage (called from project root)
---------------------------------
    python ieee5/classification/evaluate.py --config ieee5/config.yaml
"""

import argparse
import json
import sys
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
)

# Add project root to path so utils/ is importable regardless of cwd
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.io import load_config, load_split, load_model   # shared I/O helpers


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute weighted recall and confusion matrix.

    Weighted recall accounts for class frequency, making it robust under
    class imbalance. It equals macro recall when classes are balanced.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth fault-type labels.
    y_pred : np.ndarray
        Predicted fault-type labels.

    Returns
    -------
    Dict
        Keys: weighted_recall, confusion_matrix (as nested list),
              classes (as list).
    """
    weighted_recall = recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Per-class recall for diagnostics — surfaces systematic failures
    classes         = sorted(np.unique(np.concatenate([y_true, y_pred])))
    per_class_recall = recall_score(
        y_true, y_pred,
        labels       = classes,
        average      = None,
        zero_division = 0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()

    return {
        "weighted_recall":  round(float(weighted_recall), 6),
        "per_class_recall": {
            cls: round(float(val), 6)
            for cls, val in zip(classes, per_class_recall)
        },
        "confusion_matrix": cm,
        "classes":          classes,
    }


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_metrics_json(
    model_id:     str,
    val_metrics:  Dict,
    test_metrics: Dict,
    results_dir:  Path,
) -> None:
    """Save per-model metrics report as JSON.

    Report structure
    ----------------
    {
      "model_id": "RF",
      "val":  { weighted_recall, per_class_recall, confusion_matrix, classes },
      "test": { weighted_recall, per_class_recall, confusion_matrix, classes }
    }

    Parameters
    ----------
    model_id : str
    val_metrics : Dict
    test_metrics : Dict
    results_dir : Path
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model_id": model_id,
        "val":      val_metrics,
        "test":     test_metrics,
    }
    out = results_dir / f"metrics_{model_id}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  [{model_id}] Metrics saved → '{out}'")


def save_decision_matrix(
    decision_matrix: np.ndarray,
    model_names:     List[str],
    metric_names:    List[str],
    results_dir:     Path,
) -> None:
    """Save the decision matrix for mcdm.py as JSON.

    Shape: (n_models × 1) — weighted recall on test set only.
    Single criterion: no AHP weighting is needed or applied.
    Model selection in mcdm.py is based directly on this value,
    with Monte Carlo robustness as the tiebreaker.

    Parameters
    ----------
    decision_matrix : np.ndarray, shape (n_models, 1)
    model_names : List[str]
    metric_names : List[str]
    results_dir : Path
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_names":     model_names,
        "metric_names":    metric_names,
        "decision_matrix": decision_matrix.tolist(),
    }
    out = results_dir / "decision_matrix.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[decision matrix] Saved → '{out}'")


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_key:   str,
    cfg:         Dict,
    X_val:       np.ndarray,
    y_val:       np.ndarray,
    X_test:      np.ndarray,
    y_test:      np.ndarray,
    models_dir:  Path,
    results_dir: Path,
) -> Dict:
    """Full evaluation pipeline for one model.

    Steps
    -----
    1. Load model from disk (utils.io.load_model).
    2. Predict on val set and compute weighted recall + per-class recall.
    3. Predict on test set and compute weighted recall + per-class recall.
    4. Save JSON report.

    Parameters
    ----------
    model_key : str
        Key in config.yaml classification section.
    cfg : Dict
        Full parsed config dictionary.
    X_val, y_val : np.ndarray
        Validation features and labels.
    X_test, y_test : np.ndarray
        Test features and labels.
    models_dir : Path
        Directory containing .pkl model files.
    results_dir : Path
        Directory for result files.

    Returns
    -------
    Dict
        Keys: model_id, val, test.
    """
    model_id = cfg["classification"][model_key]["model_id"]
    model    = load_model(models_dir, model_id)
    
    # Load LabelEncoder if the model is XGBoost
    le_path = models_dir / "label_encoder.pkl"
    le = joblib.load(le_path) if (model_id == "XGB" and le_path.exists()) else None

    print(f"\n{'─'*60}")
    print(f"  Evaluating : {model_id}")
    print(f"{'─'*60}")

    # Val set
    y_pred_val  = model.predict(X_val)
    if le is not None:
        y_pred_val = le.inverse_transform(y_pred_val)
        
    val_metrics = compute_metrics(y_val, y_pred_val)

    print(f"  [{model_id}] Val  → weighted_recall={val_metrics['weighted_recall']:.4f}")

    # Test set
    y_pred_test = model.predict(X_test)
    if le is not None:
        y_pred_test = le.inverse_transform(y_pred_test)
        
    test_metrics = compute_metrics(y_test, y_pred_test)

    print(f"  [{model_id}] Test → weighted_recall={test_metrics['weighted_recall']:.4f}")

    # Per-class recall on test — surfaces systematic failures
    print(f"  [{model_id}] Per-class recall (test):")
    for cls, val in test_metrics["per_class_recall"].items():
        print(f"    {cls:<12}  {val:.4f}")

    save_metrics_json(model_id, val_metrics, test_metrics, results_dir)

    return {
        "model_id": model_id,
        "val":      val_metrics,
        "test":     test_metrics,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(config_path: str) -> None:
    """Full evaluation pipeline for the classification module.

    Steps
    -----
    1. Load config, val set, test set (utils.io).
    2. Evaluate RF, XGBoost, kNN on val and test.
    3. Build decision matrix (weighted recall on test) and save for mcdm.py.

    Parameters
    ----------
    config_path : str
        Path to config.yaml for the target network.
    """
    cfg = load_config(config_path)

    network     = cfg["network"]["name"]
    splits_dir  = Path(cfg["data"]["splits_dir"])
    cls_cfg     = cfg["classification"]
    models_dir  = Path(cls_cfg["models_dir"])
    results_dir = Path(cls_cfg["results_dir"])
    label_col = "label_classif"

    print(f"\n{'='*60}")
    print(f"  Classification — evaluation pipeline  [{network}]")
    print(f"{'='*60}\n")

    # Load splits (utils.io)
    X_val,  y_val  = load_split(str(splits_dir / "cls_val.csv"),  label_col)
    X_test, y_test = load_split(str(splits_dir / "cls_test.csv"), label_col)

    # Evaluate all three models
    results = []
    for model_key in ["random_forest", "xgboost", "knn"]:
        result = evaluate_model(
            model_key   = model_key,
            cfg         = cfg,
            X_val       = X_val,
            y_val       = y_val,
            X_test      = X_test,
            y_test      = y_test,
            models_dir  = models_dir,
            results_dir = results_dir,
        )
        results.append(result)

    # Build decision matrix for mcdm.py — weighted recall on test only
    model_names     = [r["model_id"] for r in results]
    metric_names    = ["weighted_recall"]
    decision_matrix = np.array([
        [r["test"]["weighted_recall"]]
        for r in results
    ])

    # Summary table
    print(f"\n{'='*60}")
    print("  Test set summary")
    print(f"{'='*60}")
    print(f"  {'Model':<8}  {'Val WRecall':>12}  {'Test WRecall':>12}")
    print(f"  {'-'*38}")
    for r in results:
        print(f"  {r['model_id']:<8}  "
              f"{r['val']['weighted_recall']:>12.4f}  "
              f"{r['test']['weighted_recall']:>12.4f}")

    save_decision_matrix(decision_matrix, model_names, metric_names, results_dir)

    print(f"\n[done]  Evaluation complete for {network}.")
    print(f"        Results → '{results_dir}'\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate RF, XGBoost, and kNN for the fault classification module."
    )
    parser.add_argument(
        "--config",
        type     = str,
        required = True,
        help     = "Path to config.yaml (e.g. 'ieee5/config.yaml')",
    )
    args = parser.parse_args()

    try:
        main(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)