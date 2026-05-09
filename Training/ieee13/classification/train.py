"""
ieee13/classification/train.py
------------------------------
Trains three candidate models (Random Forest, XGBoost, kNN) for the
multiclass fault-classification module on the IEEE 13-bus network.

Responsibilities
----------------
- Load cls_train.csv via utils.io.load_split().
- Verify cls_scaler.pkl exists via utils.io.verify_scaler().
- Build a GridSearchCV-compatible weighted-recall scorer.
- Run GridSearchCV (5-fold StratifiedKFold) on the training set for each
  model, optimising weighted recall.
- Perform an overfitting check (train score vs CV score gap).
- Save each best estimator to models/<model_id>.pkl via joblib.
- Save GridSearchCV results to results/cv_results_<model_id>.csv.
- Save overfitting summary to results/overfit_check.json.

What this script does NOT do
-----------------------------
- Does not evaluate on the validation or test set  →  evaluate.py
- Does not run Monte Carlo analysis                →  mcdm.py
- Does not select the best model                   →  mcdm.py
- Does not generate plots                          →  utils/plots.py
- Does not optimise a decision threshold           →  not applicable
                                                      (multiclass uses
                                                       predict() directly)

Utility modules used
---------------------
- utils.io.load_config()   : load YAML config
- utils.io.load_split()    : load split CSV → (X, y)
- utils.io.verify_scaler() : verify cls_scaler.pkl exists

Usage (called from project root)
---------------------------------
    python ieee13/classification/train.py --config ieee13/config.yaml
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Add project root to path so utils/ is importable regardless of cwd
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.io import load_config, load_split, verify_scaler   # shared I/O helpers

# Suppress convergence and UndefinedMetric warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# ---------------------------------------------------------------------------
# Weighted-recall scorer
# ---------------------------------------------------------------------------

def build_weighted_recall_scorer() -> callable:
    """Build a GridSearchCV-compatible weighted-recall scorer.

    Weighted recall accounts for class frequency, making it robust under
    class imbalance (relevant for the IEEE 13-bus network). When classes
    are perfectly balanced, weighted recall equals macro recall.

    Returns
    -------
    callable
        Scorer compatible with GridSearchCV (higher = better).
    """
    def weighted_recall_score(estimator, X, y) -> float:
        y_pred = estimator.predict(X)
        return recall_score(y, y_pred, average="weighted", zero_division=0)

    return weighted_recall_score

# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(model_key: str, cfg: dict) -> tuple:
    """Instantiate a model and its param_grid from config.yaml.

    Parameters
    ----------
    model_key : str
        One of 'random_forest', 'xgboost', 'knn'.
    cfg : dict
        Full parsed config dictionary.

    Returns
    -------
    tuple[estimator, dict, str]
        (unfitted model, param_grid dict, model_id string)
    """
    model_cfg = cfg["classification"][model_key]
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

    elif model_key == "xgboost":
        model = XGBClassifier(
            random_state         = model_cfg["random_state"],
            n_jobs               = model_cfg["n_jobs"],
            use_label_encoder    = False,
            eval_metric          = "mlogloss",
            verbosity            = 0,
        )
        grid = dict(model_cfg["param_grid"])

    elif model_key == "knn":
        # kNN is deterministic — no random_state needed
        model = KNeighborsClassifier(
            n_jobs = model_cfg["n_jobs"],
        )
        grid = dict(model_cfg["param_grid"])

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
    model_id:    str,
    label_encoder: LabelEncoder | None = None,
    threshold:   float = 0.05,
) -> dict:
    """Compare train score vs best CV score to flag potential overfitting.

    Parameters
    ----------
    grid_search : GridSearchCV
        Fitted GridSearchCV object.
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.
    model_id : str
        Short identifier for logging (e.g. 'RF').
    threshold : float
        Gap above which overfitting is flagged. Default 0.05 (5 pp).

    Returns
    -------
    dict
        Keys: model_id, train_score, cv_score, gap, overfit_flag.
    """
    best_model  = grid_search.best_estimator_
    y_pred      = best_model.predict(X_train)
    if label_encoder is not None:
        y_pred = label_encoder.inverse_transform(y_pred)
    train_score = recall_score(
        y_train, y_pred, average="weighted", zero_division=0
    )
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


def save_overfit_report(records: list[dict], results_dir: Path) -> None:
    """Save the overfitting check summary as JSON.

    Parameters
    ----------
    records : list[dict]
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
    cfg:         dict,
    X_train:     np.ndarray,
    y_train:     np.ndarray,
    scorer:      callable,
    cv:          StratifiedKFold,
    models_dir:  Path,
    results_dir: Path,
    label_encoder: LabelEncoder | None = None,
) -> tuple[GridSearchCV, dict]:
    """Run GridSearchCV for one model and persist outputs.

    Parameters
    ----------
    model_key : str
        Key in config.yaml classification section.
    cfg : dict
        Full parsed config dictionary.
    X_train : np.ndarray
        Training feature matrix (already scaled).
    y_train : np.ndarray
        Training labels (fault type strings).
    scorer : callable
        Weighted-recall scorer.
    cv : StratifiedKFold
        Cross-validation splitter.
    models_dir : Path
        Where to save the .pkl model file.
    results_dir : Path
        Where to save CV results CSV.

    Returns
    -------
    tuple[GridSearchCV, dict]
        Fitted GridSearchCV and overfitting check dict.
    """
    model, grid, model_id = build_model(model_key, cfg)

    n_combinations = 1
    for v in grid.values():
        n_combinations *= len(v)

    print(f"\n{'─'*60}")
    print(f"  Training : {model_id}  "
          f"({n_combinations} combinations × "
          f"{cfg['classification']['cv']['n_splits']} folds)")
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
    y_fit = label_encoder.transform(y_train) if label_encoder is not None else y_train
    gs.fit(X_train, y_fit)

    elapsed = time.time() - t0
    print(f"  [{model_id}] Done in {elapsed:.1f}s  "
          f"|  best CV score: {gs.best_score_:.4f}")
    print(f"  [{model_id}] Best params: {gs.best_params_}")

    overfit_info = overfitting_check(gs, X_train, y_train, model_id, label_encoder=label_encoder)
    save_model(gs.best_estimator_, models_dir, model_id)
    save_cv_results(gs, results_dir, model_id)

    return gs, overfit_info


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(config_path: str) -> None:
    """Full training pipeline for the classification module.

    Steps
    -----
    1. Load config and verify data files (via utils.io).
    2. Build weighted-recall scorer.
    3. Train RF, XGBoost, kNN via GridSearchCV (5-fold StratifiedKFold).
    4. Run overfitting check per model.
    5. Save models (.pkl), CV results (.csv), overfit report (.json).

    Parameters
    ----------
    config_path : str
        Path to config.yaml for the target network.
    """
    cfg = load_config(config_path)

    network       = cfg["network"]["name"]
    splits_dir    = Path(cfg["data"]["splits_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    cls_cfg       = cfg["classification"]
    models_dir    = Path(cls_cfg["models_dir"])
    results_dir   = Path(cls_cfg["results_dir"])
    label_col     = "label_classif"

    print(f"\n{'='*60}")
    print(f"  Classification — training pipeline  [{network}]")
    print(f"{'='*60}\n")

    # Load data (utils.io)
    X_train, y_train = load_split(str(splits_dir / "cls_train.csv"), label_col)
    verify_scaler(str(processed_dir / "cls_scaler.pkl"))

    # Class distribution summary
    classes, counts = np.unique(y_train, return_counts=True)
    print("  Class distribution — cls_train:")
    for cls, cnt in zip(classes, counts):
        print(f"    {cls:<12}  {cnt:>5}  ({cnt/len(y_train):.1%})")
    print()

    # LabelEncoder for XGBoost
    le = LabelEncoder()
    le.fit(y_train)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(le, models_dir / "label_encoder.pkl")
    print(f"  LabelEncoder classes: {list(le.classes_)}")
    print(f"  Saved → '{models_dir / 'label_encoder.pkl'}'")

    # Weighted-recall scorer
    scorer = build_weighted_recall_scorer()

    # Cross-validation splitter
    cv_cfg = cls_cfg["cv"]
    cv = StratifiedKFold(
        n_splits     = cv_cfg["n_splits"],
        shuffle      = cv_cfg["shuffle"],
        random_state = cv_cfg["random_state"],
    )

    # Train all three models
    overfit_records = []
    for model_key in ["random_forest", "xgboost", "knn"]:
        _, overfit_info = train_model(
            model_key     = model_key,
            cfg           = cfg,
            X_train       = X_train,
            y_train       = y_train,
            scorer        = scorer,
            cv            = cv,
            models_dir    = models_dir,
            results_dir   = results_dir,
            label_encoder = le if model_key == "xgboost" else None,
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
        description="Train RF, XGBoost, and kNN for the fault classification module."
    )
    parser.add_argument(
        "--config",
        type     = str,
        required = True,
        help     = "Path to config.yaml (e.g. 'ieee13/config.yaml')",
    )
    args = parser.parse_args()

    try:
        main(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
