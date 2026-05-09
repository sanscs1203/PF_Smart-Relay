"""
ieee5/detection/train.py
------------------------
Trains three candidate models (Random Forest, SVM, MLP) for the binary
fault-detection module on the IEEE 5-bus network.

Responsibilities
----------------
- Load det_train.csv and det_val.csv via utils.io.load_split().
- Verify det_scaler.pkl exists via utils.io.verify_scaler().
- Reconstruct the AHP pairwise matrix via utils.io.build_pairwise_matrix()
  and derive priority weights via utils.ahp_weights.ahp_weights().
- Build a GridSearchCV-compatible AHP composite scorer from those weights.
- Run GridSearchCV (5-fold StratifiedKFold) on the training set for each
  model, optimising the AHP composite score.
- Perform an overfitting check (train score vs CV score gap).
- Save each best estimator to models/<model_id>.pkl via joblib.
- Save GridSearchCV results to results/cv_results_<model_id>.csv.
- Save overfitting summary to results/overfit_check.json.

What this script does NOT do
-----------------------------
- Does not optimise the decision threshold  →  evaluate.py
- Does not compute Shannon Entropy weights  →  mcdm.py
- Does not run Monte Carlo analysis         →  mcdm.py
- Does not generate plots                  →  utils/plots.py
- Does not evaluate on the test set        →  evaluate.py

Utility modules used
---------------------
- utils.io.load_config()            : load YAML config
- utils.io.load_split()             : load split CSV → (X, y)
- utils.io.verify_scaler()          : verify det_scaler.pkl exists
- utils.io.build_pairwise_matrix()  : reconstruct 2×2 AHP matrix
- utils.ahp_weights.ahp_weights()   : derive AHP priority weights

Usage (called from project root)
---------------------------------
    python3 ieee5/detection/train.py --config ieee5/config.yaml
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Add project root to path so utils/ is importable regardless of cwd
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.io import (               # shared I/O helpers
    load_config,
    load_split,
    verify_scaler,
    build_pairwise_matrix,
)
from utils.ahp_weights import ahp_weights   # AHP weight computation (Saaty, 1980)

# suppress convergence / UndefinedMetric warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ---------------------------------------------------------------------------
# AHP composite scorer
# ---------------------------------------------------------------------------

def build_ahp_scorer(weights: np.ndarray) -> callable:
    """Build a GridSearchCV-compatible scorer from AHP priority weights.

    The composite score is a weighted sum of two metrics:
        score = w[0]*Recall + w[1]*Specificity
        
    Weight order matches the pairwise matrix row order:
        index 0 → Recall       (dependability)
        index 1 → Specificity  (security)

    Parameters
    ----------
    weights : np.ndarray, shape (2,)
        AHP priority vector from utils.ahp_weights.ahp_weights().

    Returns
    -------
    callable
        Scorer function compatible with GridSearchCV (higher = better).
    """
    w_recall = weights[0]
    w_spec   = weights[1]

    def ahp_score(estimator, X, y) -> float:
        y_pred = estimator.predict(X)

        recall      = recall_score(y, y_pred, pos_label=1, zero_division=0)
        specificity = recall_score(y, y_pred, pos_label=0, zero_division=0)

        return w_recall * recall + w_spec * specificity

    return ahp_score


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(model_key: str, cfg: Dict) -> Tuple:
    """Instantiate a model and its param_grid from config.yaml.

    Parameters
    ----------
    model_key : str
        One of 'random_forest', 'svm', 'mlp'.
    cfg : Dict
        Full parsed config dictionary.

    Returns
    -------
    Tuple[estimator, Dict, str]
        (unfitted model, param_grid dict, model_id string)
    """
    model_cfg = cfg["detection"][model_key]
    model_id  = model_cfg["model_id"]
    
    if model_key == "random_forest":
        model = RandomForestClassifier(
            class_weight = model_cfg["class_weight"],
            random_state = model_cfg["random_state"],
            n_jobs       = model_cfg["n_jobs"],
        )
        grid = dict(model_cfg["param_grid"])
        # YAML null → Python None for max_depth
        grid["max_depth"] = [
            None if v is None else int(v) for v in grid["max_depth"]
        ]
        
    elif model_key == "svm":
        model = SVC(
            probability  = model_cfg["probability"],
            class_weight = model_cfg["class_weight"],
            random_state = model_cfg["random_state"],
        )
        grid = dict(model_cfg["param_grid"])

    elif model_key == "mlp":
        model = MLPClassifier(
            random_state = model_cfg["random_state"],
        )
        grid = dict(model_cfg["param_grid"])
        # YAML list-of-lists → list-of-tuples for hidden_layer_sizes
        grid["hidden_layer_sizes"] = [
            tuple(v) for v in grid["hidden_layer_sizes"]
        ]

    else:
        raise ValueError(f"Unknown model key: '{model_key}'")

    return model, grid, model_id


# ---------------------------------------------------------------------------
# Overfitting check
# ---------------------------------------------------------------------------

def overfitting_check(
    grid_search: GridSearchCV,
    X_train:     np.ndarray,
    y_train:     np.ndarray,
    scorer:      callable,
    model_id:    str,
    threshold:   float = 0.05,
) -> Dict:
    """Compare train score vs best CV score to flag potential overfitting.

    Parameters
    ----------
    grid_search : GridSearchCV
        Fitted GridSearchCV object.
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.
    scorer : callable
        The AHP scorer used during GridSearchCV.
    model_id : str
        Short identifier for logging (e.g. 'RF').
    threshold : float
        Gap above which overfitting is flagged. Default 0.05 (5 pp).

    Returns
    -------
    Dict
        Keys: model_id, train_score, cv_score, gap, overfit_flag.
    """
    best_model  = grid_search.best_estimator_
    train_score = scorer(best_model, X_train, y_train)
    cv_score    = grid_search.best_score_
    gap         = train_score - cv_score
    overfit     = gap > threshold

    status = "⚠️  possible overfit" if overfit else "✓  ok"
    print(f"  [{model_id}] train={train_score:.4f}  cv={cv_score:.4f}  "
          f"gap={gap:.4f}  {status}")

    return {
        "model_id":     model_id,
        "train_score":  round(float(train_score), 6),
        "cv_score":     round(float(cv_score),    6),
        "gap":          round(float(gap),          6),
        "overfit_flag": bool(overfit),
    }


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_model(estimator, models_dir: Path, model_id: str) -> None:
    """Persist a fitted estimator to disk as a .pkl file.

    Parameters
    ----------
    estimator : fitted sklearn estimator
        Best estimator from GridSearchCV (refit on full train set).
    models_dir : Path
        Directory where the .pkl file will be saved.
    model_id : str
        Used as the filename stem (e.g. 'RF' → 'RF.pkl').
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / f"{model_id}.pkl"
    joblib.dump(estimator, out_path)
    print(f"  [{model_id}] Model saved  → '{out_path}'")


def save_cv_results(grid_search: GridSearchCV,
                    results_dir: Path,
                    model_id:    str) -> None:
    """Save GridSearchCV cv_results_ to CSV, sorted by rank.

    Per-split columns (split0_*, split1_*, ...) are dropped to keep
    the file compact — mean and std columns are retained.

    Parameters
    ----------
    grid_search : GridSearchCV
        Fitted GridSearchCV object.
    results_dir : Path
        Directory for result files.
    model_id : str
        Used as part of the output filename.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    df   = pd.DataFrame(grid_search.cv_results_)
    keep = [c for c in df.columns if not c.startswith("split")]
    df   = df[keep].sort_values("rank_test_score")
    out  = results_dir / f"cv_results_{model_id}.csv"
    df.to_csv(out, sep=";", decimal=",", index=False)
    print(f"  [{model_id}] CV results saved → '{out}'")


def save_overfit_report(records: List[Dict], results_dir: Path) -> None:
    """Save the overfitting check summary as JSON.

    Parameters
    ----------
    records : List[Dict]
        One dict per model, output of overfitting_check().
    results_dir : Path
        Directory for result files.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "overfit_check.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"[overfit] Summary saved → '{out}'")


# ---------------------------------------------------------------------------
# Per-model training loop
# ---------------------------------------------------------------------------

def train_model(
    model_key:   str,
    cfg:         Dict,
    X_train:     np.ndarray,
    y_train:     np.ndarray,
    scorer:      callable,
    cv:          StratifiedKFold,
    models_dir:  Path,
    results_dir: Path,
) -> Tuple[GridSearchCV, Dict]:
    """Run GridSearchCV for one model and persist outputs.

    Parameters
    ----------
    model_key : str
        Key in config.yaml detection section.
    cfg : Dict
        Full parsed config dictionary.
    X_train : np.ndarray
        Training feature matrix (already scaled).
    y_train : np.ndarray
        Training labels.
    scorer : callable
        AHP composite scorer.
    cv : StratifiedKFold
        Cross-validation splitter.
    models_dir : Path
        Where to save the .pkl model file.
    results_dir : Path
        Where to save CV results CSV.

    Returns
    -------
    Tuple[GridSearchCV, Dict]
        Fitted GridSearchCV and overfitting check dict.
    """
    model, grid, model_id = build_model(model_key, cfg)

    n_combinations = 1
    for v in grid.values():
        n_combinations *= len(v)

    print(f"\n{'─'*60}")
    print(f"  Training : {model_id}  "
          f"({n_combinations} combinations × "
          f"{cfg['detection']['cv']['n_splits']} folds)")
    print(f"{'─'*60}")

    t0 = time.time()

    gs = GridSearchCV(
        estimator   = model,
        param_grid  = grid,
        scoring     = scorer,
        cv          = cv,
        n_jobs      = -1,
        refit       = True,
        verbose     = 1,
        error_score = "raise",
    )
    gs.fit(X_train, y_train)

    elapsed = time.time() - t0
    print(f"  [{model_id}] Done in {elapsed:.1f}s  "
          f"|  best CV score: {gs.best_score_:.4f}")
    print(f"  [{model_id}] Best params: {gs.best_params_}")

    overfit_info = overfitting_check(gs, X_train, y_train, scorer, model_id)
    save_model(gs.best_estimator_, models_dir, model_id)
    save_cv_results(gs, results_dir, model_id)

    return gs, overfit_info


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(config_path: str) -> None:
    """Full training pipeline for the detection module.

    Steps
    -----
    1. Load config and verify data files (via utils.io).
    2. Reconstruct AHP pairwise matrix (via utils.io) and derive weights
       (via utils.ahp_weights).
    3. Build AHP composite scorer.
    4. Train RF, SVM, MLP via GridSearchCV (5-fold StratifiedKFold).
    5. Run overfitting check per model.
    6. Save models (.pkl), CV results (.csv), overfit report (.json).

    Parameters
    ----------
    config_path : str
        Path to config.yaml for the target network.
    """
    cfg = load_config(config_path)

    network       = cfg["network"]["name"]
    splits_dir    = Path(cfg["data"]["splits_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    det_cfg       = cfg["detection"]
    models_dir    = Path(det_cfg["models_dir"])
    results_dir   = Path(det_cfg["results_dir"])
    label_col     = "label_detection"

    print(f"\n{'='*60}")
    print(f"  Detection — training pipeline  [{network}]")
    print(f"{'='*60}\n")

    # Load data (utils.io)
    X_train, y_train = load_split(str(splits_dir / "det_train.csv"), label_col)
    X_val,   y_val   = load_split(str(splits_dir / "det_val.csv"),   label_col)
    verify_scaler(str(processed_dir / "det_scaler.pkl"))

    print(f"\n  Class balance — train : "
          f"fault={y_train.mean():.1%}  no-fault={1-y_train.mean():.1%}")
    print(f"  Class balance — val   : "
          f"fault={y_val.mean():.1%}  no-fault={1-y_val.mean():.1%}\n")

    # AHP weights (utils.io + utils.ahp_weights)
    metric_names    = ["Recall", "Specificity"]
    pairwise_matrix = build_pairwise_matrix(det_cfg["ahp"]["pairwise_matrix"])
    weights, CR     = ahp_weights(pairwise_matrix, metric_names)

    if CR >= 0.10:
        raise ValueError(
            f"AHP pairwise matrix inconsistent (CR={CR:.4f} ≥ 0.10). "
            "Review pairwise_matrix in config.yaml."
        )

    # AHP composite scorer
    scorer = build_ahp_scorer(weights)

    # Cross-validation splitter
    cv_cfg = det_cfg["cv"]
    cv = StratifiedKFold(
        n_splits     = cv_cfg["n_splits"],
        shuffle      = cv_cfg["shuffle"],
        random_state = cv_cfg["random_state"],
    )

    # Train all three models
    overfit_records = []
    for model_key in ["random_forest", "svm", "mlp"]:
        _, overfit_info = train_model(
            model_key   = model_key,
            cfg         = cfg,
            X_train     = X_train,
            y_train     = y_train,
            scorer      = scorer,
            cv          = cv,
            models_dir  = models_dir,
            results_dir = results_dir,
        )
        overfit_records.append(overfit_info)

    # Overfitting summary
    print(f"\n{'='*60}")
    print("  Overfitting check summary")
    print(f"{'='*60}")
    for r in overfit_records:
        flag = "⚠️" if r["overfit_flag"] else "✓"
        print(f"  {r['model_id']:<6}  train={r['train_score']:.4f}  "
              f"cv={r['cv_score']:.4f}  gap={r['gap']:.4f}  {flag}")

    save_overfit_report(overfit_records, results_dir)

    print(f"\n[done]  Training complete for {network}.")
    print(f"        Models  → '{models_dir}'")
    print(f"        Results → '{results_dir}'\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RF, SVM, and MLP for the fault detection module."
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