"""
ieee5/detection/evaluate.py
----------------------------
Evaluates the three trained detection models (RF, SVM, MLP) and selects
the optimal decision threshold for each one.

Responsibilities
----------------
- Load each trained model via utils.io.load_model().
- Load det_val.csv and det_test.csv via utils.io.load_split().
- Reconstruct AHP weights via utils.io.build_pairwise_matrix() and
  utils.ahp_weights.ahp_weights() — identical to train.py.
- Sweep thresholds on the val set (range and steps from config.yaml)
  and select the one that maximises the AHP composite score.
- Evaluate each model on the test set with its optimal threshold.
- Save per-model JSON reports (metrics + confusion matrix + threshold).
- Build and save the decision matrix (n_models × n_metrics) for mcdm.py.

What this script does NOT do
-----------------------------
- Does not select the best model         →  mcdm.py
- Does not run Monte Carlo               →  mcdm.py
- Does not generate plots                →  utils/plots.py

Utility modules used
---------------------
- utils.io.load_config()            : load YAML config
- utils.io.load_split()             : load split CSV → (X, y)
- utils.io.load_model()             : load fitted model from .pkl
- utils.io.build_pairwise_matrix()  : reconstruct 2×2 AHP matrix
- utils.ahp_weights.ahp_weights()   : derive AHP priority weights

Output files (results/)
-----------------------
- metrics_RF.json        : val + test metrics, confusion matrix, threshold
- metrics_SVM.json       : idem
- metrics_MLP.json       : idem
- decision_matrix.json   : (n_models × n_metrics) for mcdm.py

Usage (called from project root)
---------------------------------
    python ieee5/detection/evaluate.py --config ieee5/config.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
)

# Add project root to path so utils/ is importable regardless of cwd
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.io import (
    load_config,
    load_split,
    load_model,
    build_pairwise_matrix,
)
from utils.ahp_weights import ahp_weights   # AHP weight computation (Saaty, 1980)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    weights: np.ndarray,
) -> dict:
    """Compute Recall, Specificity and AHP composite score.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels.
    y_pred : np.ndarray
        Predicted binary labels (after threshold application).
    
    weights : np.ndarray, shape (2,)
        AHP priority vector [w_recall, w_spec].

    Returns
    -------
    dict
        Keys: recall, specificity, roc_you, ahp_score,
              confusion_matrix (as nested list).
    """
    recall      = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    ahp_score   = (weights[0] * recall
                   + weights[1] * specificity)
    
    cm          = confusion_matrix(y_true, y_pred).tolist()

    return {
        "recall":           round(float(recall),      6),
        "specificity":      round(float(specificity), 6),
        "ahp_score":        round(float(ahp_score),   6),
        "confusion_matrix": cm,
    }


def get_probabilities(model, X: np.ndarray) -> np.ndarray:
    """Extract positive-class probabilities from a fitted model.

    Parameters
    ----------
    model : fitted sklearn estimator
    X : np.ndarray
        Feature matrix.

    Returns
    -------
    np.ndarray
        Probability scores for the positive class (label = 1).
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # Fallback for SVC with probability=False (should not occur given config)
    scores = model.decision_function(X)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)


# ---------------------------------------------------------------------------
# Threshold optimisation on val set
# ---------------------------------------------------------------------------

def optimise_threshold(
    model:    object,
    X_val:    np.ndarray,
    y_val:    np.ndarray,
    weights:  np.ndarray,
    thr_cfg:  dict,
    model_id: str,
) -> tuple[float, dict]:
    """Sweep thresholds on the val set and return the optimal one.
 
    Selection strategy (two-stage):
      1. Compute AHP reference score at threshold=0.50 (sklearn default).
         This anchors the search — a threshold is only valid if it does
         not degrade the AHP score below this baseline.
      2. Among valid thresholds (AHP >= ahp_reference), maximise Recall.
         This directly minimises FN, which is the primary protection goal.
 
    If no threshold improves AHP over the reference, the one with the
    highest AHP score is selected as fallback.
 
    Sweep range and step count are read from config.yaml:
        detection.threshold.min, .max, .steps
 
    Parameters
    ----------
    model : fitted sklearn estimator
    X_val : np.ndarray
    y_val : np.ndarray
    weights : np.ndarray
        AHP priority vector [w_recall, w_specificity].
    thr_cfg : dict
        Threshold config sub-dict from config.yaml.
    model_id : str
        Short identifier for logging.
 
    Returns
    -------
    tuple[float, dict]
        (optimal_threshold, val_metrics_at_optimal_threshold)
    """
    thresholds = np.linspace(
        thr_cfg["min"],
        thr_cfg["max"],
        thr_cfg["steps"],
    )
 
    y_prob = get_probabilities(model, X_val)
 
    # AHP reference — score at default threshold=0.50
    y_pred_ref   = (y_prob >= 0.50).astype(int)
    metrics_ref  = compute_metrics(y_val, y_pred_ref, weights)
    ahp_ref      = metrics_ref["ahp_score"]
 
    best_thr      = 0.5
    best_recall   = -np.inf
    best_metrics  = metrics_ref
 
    # Fallback — highest AHP regardless of constraint
    fallback_thr     = 0.5
    fallback_score   = ahp_ref
    fallback_metrics = metrics_ref
 
    for thr in thresholds:
        y_pred  = (y_prob >= thr).astype(int)
        metrics = compute_metrics(y_val, y_pred, weights)
 
        if metrics["ahp_score"] > fallback_score:
            fallback_score   = metrics["ahp_score"]
            fallback_thr     = thr
            fallback_metrics = metrics
 
        # Hard constraint — AHP must not fall below reference
        if metrics["ahp_score"] < ahp_ref:
            continue
 
        # Among valid thresholds — maximise Recall
        if metrics["recall"] > best_recall:
            best_recall  = metrics["recall"]
            best_thr     = thr
            best_metrics = metrics
 
    if best_recall == -np.inf:
        print(f"  [{model_id}] ⚠️  No threshold achieved AHP ≥ {ahp_ref:.4f}. "
              f"Falling back to highest AHP ({fallback_score:.4f}).")
        best_thr     = fallback_thr
        best_metrics = fallback_metrics
 
    print(f"  [{model_id}] Optimal threshold : {best_thr:.2f}  "
          f"|  ahp_ref={ahp_ref:.4f}  "
          f"(recall={best_metrics['recall']:.4f}  "
          f"spec={best_metrics['specificity']:.4f}  "
          f"ahp={best_metrics['ahp_score']:.4f})")
 
    return float(best_thr), best_metrics
 

# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_metrics_json(
    model_id:     str,
    threshold:    float,
    val_metrics:  dict,
    test_metrics: dict,
    results_dir:  Path,
) -> None:
    """Save per-model metrics report as JSON.

    Report structure
    ----------------
    {
      "model_id": "RF",
      "optimal_threshold": 0.42,
      "val":  { recall, specificity, ahp_score, confusion_matrix },
      "test": { recall, specificity, ahp_score, confusion_matrix }
    }

    Parameters
    ----------
    model_id : str
    threshold : float
    val_metrics : dict
    test_metrics : dict
    results_dir : Path
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model_id":          model_id,
        "optimal_threshold": round(threshold, 4),
        "val":               val_metrics,
        "test":              test_metrics,
    }
    out = results_dir / f"metrics_{model_id}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  [{model_id}] Metrics saved → '{out}'")


def save_decision_matrix(
    decision_matrix: np.ndarray,
    model_names:     list[str],
    metric_names:    list[str],
    results_dir:     Path,
) -> None:
    """Save the decision matrix for mcdm.py as JSON.

    Shape: (n_models × n_metrics) — test-set scores only.
    This file is the primary input for monte_carlo_robustness()
    in mcdm.py.

    Parameters
    ----------
    decision_matrix : np.ndarray, shape (n_models, n_metrics)
    model_names : list[str]
    metric_names : list[str]
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
    cfg:         dict,
    X_val:       np.ndarray,
    y_val:       np.ndarray,
    X_test:      np.ndarray,
    y_test:      np.ndarray,
    weights:     np.ndarray,
    models_dir:  Path,
    results_dir: Path,
) -> dict:
    """Full evaluation pipeline for one model.

    Steps
    -----
    1. Load model from disk (utils.io.load_model).
    2. Optimise threshold on val set (maximise AHP composite score).
    3. Evaluate on test set using optimal threshold.
    4. Save JSON report.

    Parameters
    ----------
    model_key : str
    cfg : dict
    X_val, y_val : np.ndarray
    X_test, y_test : np.ndarray
    weights : np.ndarray
    models_dir : Path
    results_dir : Path

    Returns
    -------
    dict
        Keys: model_id, threshold, val, test.
    """
    model_id = cfg["detection"][model_key]["model_id"]
    model    = load_model(models_dir, model_id)
    thr_cfg  = cfg["detection"]["threshold"]

    print(f"\n{'─'*60}")
    print(f"  Evaluating : {model_id}")
    print(f"{'─'*60}")

    # Threshold optimisation on val
    threshold, val_metrics = optimise_threshold(
        model, X_val, y_val, weights, thr_cfg, model_id
    )

    # Test set evaluation with optimal threshold 
    y_prob_test  = get_probabilities(model, X_test)
    y_pred_test  = (y_prob_test >= threshold).astype(int)
    test_metrics = compute_metrics(y_test, y_pred_test, weights)

    print(f"  [{model_id}] Test  → "
          f"recall={test_metrics['recall']:.4f}  "
          f"spec={test_metrics['specificity']:.4f}  "
          f"ahp={test_metrics['ahp_score']:.4f}")

    save_metrics_json(model_id, threshold, val_metrics, test_metrics, results_dir)

    return {
        "model_id":  model_id,
        "threshold": threshold,
        "val":       val_metrics,
        "test":      test_metrics,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(config_path: str) -> None:
    """Full evaluation pipeline for the detection module.

    Steps
    -----
    1. Load config, val set, test set (utils.io).
    2. Reconstruct AHP weights (utils.io + utils.ahp_weights).
    3. Evaluate RF, SVM, MLP — threshold on val, final score on test.
    4. Build decision matrix and save for mcdm.py.

    Parameters
    ----------
    config_path : str
        Path to config.yaml for the target network.
    """
    cfg = load_config(config_path)

    network     = cfg["network"]["name"]
    splits_dir  = Path(cfg["data"]["splits_dir"])
    det_cfg     = cfg["detection"]
    models_dir  = Path(det_cfg["models_dir"])
    results_dir = Path(det_cfg["results_dir"])
    label_col   = "label_detection"

    print(f"\n{'='*60}")
    print(f"  Detection — evaluation pipeline  [{network}]")
    print(f"{'='*60}\n")

    # Load splits (utils.io)
    X_val,  y_val  = load_split(str(splits_dir / "det_val.csv"),  label_col)
    X_test, y_test = load_split(str(splits_dir / "det_test.csv"), label_col)

    # AHP weights (utils.io + utils.ahp_weights)
    metric_names    = ["Recall", "Specificity"]
    pairwise_matrix = build_pairwise_matrix(det_cfg["ahp"]["pairwise_matrix"])
    weights, CR     = ahp_weights(pairwise_matrix, metric_names)

    if CR >= 0.10:
        raise ValueError(
            f"AHP pairwise matrix inconsistent (CR={CR:.4f} ≥ 0.10). "
            "Review pairwise_matrix in config.yaml."
        )

    # Evaluate all three models
    results = []
    for model_key in ["random_forest", "svm", "mlp"]:
        result = evaluate_model(
            model_key   = model_key,
            cfg         = cfg,
            X_val       = X_val,
            y_val       = y_val,
            X_test      = X_test,
            y_test      = y_test,
            weights     = weights,
            models_dir  = models_dir,
            results_dir = results_dir,
        )
        results.append(result)

    # Build decision matrix for mcdm.py (test scores only)
    model_names     = [r["model_id"] for r in results]
    decision_matrix = np.array([
        [
            r["test"]["recall"],
            r["test"]["specificity"],
        ]
        for r in results
    ])

    # Summary table
    print(f"\n{'='*60}")
    print("  Test set summary")
    print(f"{'='*60}")
    print(f"  {'Model':<8} {'Threshold':>10} {'Recall':>10} "
          f"{'Spec':>10}  {'AHP':>10}")
    print(f"  {'-'*58}")
    for r in results:
        print(f"  {r['model_id']:<8} {r['threshold']:>10.2f} "
              f"{r['test']['recall']:>10.4f} "
              f"{r['test']['specificity']:>10.4f} "
              f"{r['test']['ahp_score']:>10.4f}")

    save_decision_matrix(decision_matrix, model_names, metric_names, results_dir)

    print(f"\n[done]  Evaluation complete for {network}.")
    print(f"        Results → '{results_dir}'\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate RF, SVM, MLP for the fault detection module."
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
