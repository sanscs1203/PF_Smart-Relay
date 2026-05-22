"""
ieee13/classification/train.py
------------------------------
Trains three candidate models (Random Forest, XGBoost, kNN) for the
multiclass fault-classification module on the IEEE 13-bus network.
Optionally, real laboratory data can be mixed into the training set.

Responsibilities
----------------
- Load cls_train.csv and optionally real CSV with the same 12 features + label.
- Verify cls_scaler.pkl exists and use it to transform real data.
- Build a weighted-recall scorer.
- Run RandomizedSearchCV (5-fold StratifiedKFold) on the training set
  (simulated + optional real) for each model, optimising weighted recall.
- Perform an overfitting check (train score vs CV score gap).
- Save each best estimator to models/<model_id>.pkl.
- Save RandomizedSearchCV results to results/cv_results_<model_id>.csv.
- Save overfitting summary to results/overfit_check.json.

Usage (called from project root):
    # Training with simulation only
    python ieee13/classification/train.py --config ieee13/config.yaml

    # Training with simulation + real data
    python ieee13/classification/train.py --config ieee13/config.yaml --real-data data/real/lab_data.csv
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# Add project root to path so utils/ is importable regardless of cwd
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.io import load_config, load_split, verify_scaler

# Suppress convergence and UndefinedMetric warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# ---------------------------------------------------------------------------
# Helper: load real dataset (same features + label_classif)
# ---------------------------------------------------------------------------
def load_real_dataset(csv_path: str, scaler: StandardScaler) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Real dataset not found: {path}")
    df = pd.read_csv(path, sep=";", decimal=",")

    feature_cols = ["Va", "Vb", "Vc", "Ia", "Ib", "Ic"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Real dataset missing feature columns: {missing}")

    # 1. Seleccionar la etiqueta correcta (específica de fase)
    if "label_classif" in df.columns:
        # Si ya viene con el formato de simulación (poco probable en lab)
        y = df["label_classif"].values
    elif "Fases_en_Falla" in df.columns:
        # Usar directamente la columna que ya tiene "L A", "LL BC", etc.
        y = df["Fases_en_Falla"].astype(str).str.strip().values
        # Filtrar filas sin falla
        mask_sin_falla = np.isin(y, ["Sin_Falla", "N.A", "nan"])
        if mask_sin_falla.any():
            print(f"[real] Descartadas {mask_sin_falla.sum()} filas sin falla")
            df = df[~mask_sin_falla]
            y = y[~mask_sin_falla]
    else:
        raise ValueError("El CSV debe contener 'label_classif' o 'Fases_en_Falla'")

    X = df[feature_cols].values
    X_scaled = scaler.transform(X)

    print(f"[real] Cargadas {len(X_scaled)} muestras desde '{path.name}'")
    classes, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(classes, counts):
        print(f"         {cls:<15} {cnt:>3}")
    return X_scaled, y

# ---------------------------------------------------------------------------
# Weighted-recall scorer
# ---------------------------------------------------------------------------
def build_weighted_recall_scorer() -> callable:
    def weighted_recall_score(estimator, X, y) -> float:
        y_pred = estimator.predict(X)
        return recall_score(y, y_pred, average="weighted", zero_division=0)
    return weighted_recall_score


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_model(model_key: str, cfg: Dict) -> Tuple:
    model_cfg = cfg["classification"][model_key]
    model_id  = model_cfg["model_id"]

    if model_key == "random_forest":
        model = RandomForestClassifier(
            class_weight = model_cfg["class_weight"],
            random_state = model_cfg["random_state"],
            n_jobs       = model_cfg["n_jobs"],
        )
        grid = dict(model_cfg["param_grid"])
        grid["max_depth"] = [None if v is None else int(v) for v in grid["max_depth"]]

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
        model = KNeighborsClassifier(n_jobs = model_cfg["n_jobs"])
        grid = dict(model_cfg["param_grid"])

    else:
        raise ValueError(f"Unknown model key: '{model_key}'")

    return model, grid, model_id


# ---------------------------------------------------------------------------
# Overfitting check
# ---------------------------------------------------------------------------
def overfitting_check(
    random_search: RandomizedSearchCV,
    X_train:       np.ndarray,
    y_train:       np.ndarray,
    model_id:      str,
    label_encoder: Optional[LabelEncoder] = None,
    threshold:     float = 0.05,
) -> Dict:
    best_model  = random_search.best_estimator_
    y_pred      = best_model.predict(X_train)
    if label_encoder is not None:
        y_pred = label_encoder.inverse_transform(y_pred)
    train_score = recall_score(y_train, y_pred, average="weighted", zero_division=0)
    cv_score    = random_search.best_score_
    gap         = train_score - cv_score
    overfit     = gap > threshold
    status = "⚠️  possible overfit" if overfit else "✓  ok"
    print(f"  [{model_id}] train={train_score:.4f}  cv={cv_score:.4f}  gap={gap:.4f}  {status}")
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
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / f"{model_id}.pkl"
    joblib.dump(estimator, out_path)
    print(f"  [{model_id}] Model saved  → '{out_path}'")

def save_cv_results(random_search: RandomizedSearchCV, results_dir: Path, model_id: str) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    df   = pd.DataFrame(random_search.cv_results_)
    keep = [c for c in df.columns if not c.startswith("split")]
    df   = df[keep].sort_values("rank_test_score")
    out  = results_dir / f"cv_results_{model_id}.csv"
    df.to_csv(out, sep=";", decimal=",", index=False)
    print(f"  [{model_id}] CV results saved → '{out}'")

def save_overfit_report(records: List[Dict], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "overfit_check.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"[overfit] Summary saved → '{out}'")


# ---------------------------------------------------------------------------
# Training loop per model
# ---------------------------------------------------------------------------
def train_model(
    model_key:     str,
    cfg:           Dict,
    X_train:       np.ndarray,
    y_train:       np.ndarray,
    scorer:        callable,
    cv:            StratifiedKFold,
    models_dir:    Path,
    results_dir:   Path,
    label_encoder: Optional[LabelEncoder] = None,
) -> Tuple[RandomizedSearchCV, Dict]:
    model, param_dist, model_id = build_model(model_key, cfg)
    model_cfg = cfg["classification"][model_key]
    n_iter = model_cfg.get("n_iter", 20)

    n_combinations = 1
    for v in param_dist.values():
        n_combinations *= len(v)
    n_iter = min(n_iter, n_combinations)

    print(f"\n{'─'*60}")
    print(f"  Training : {model_id}  ({n_iter} sampled iterations from {n_combinations} total × "
          f"{cfg['classification']['cv']['n_splits']} folds)")
    print(f"{'─'*60}")

    t0 = time.time()

    rs = RandomizedSearchCV(
        estimator           = model,
        param_distributions = param_dist,
        n_iter              = n_iter,
        scoring             = scorer,
        cv                  = cv,
        n_jobs              = -1,
        refit               = True,
        verbose             = 2,
        error_score         = "raise",
        random_state        = cfg["classification"].get("random_state", 42),
    )

    y_fit = label_encoder.transform(y_train) if label_encoder is not None else y_train
    rs.fit(X_train, y_fit)

    elapsed = time.time() - t0
    print(f"  [{model_id}] Done in {elapsed:.1f}s  |  best CV score: {rs.best_score_:.4f}")
    print(f"  [{model_id}] Best params: {rs.best_params_}")

    overfit_info = overfitting_check(rs, X_train, y_train, model_id, label_encoder=label_encoder)
    save_model(rs.best_estimator_, models_dir, model_id)
    save_cv_results(rs, results_dir, model_id)

    return rs, overfit_info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(config_path: str, real_data_path: Optional[str] = None) -> None:
    cfg = load_config(config_path)

    network       = cfg["network"]["name"]
    splits_dir    = Path(cfg["data"]["splits_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    cls_cfg       = cfg["classification"]
    models_dir    = Path(cls_cfg["models_dir"])
    results_dir   = Path(cls_cfg["results_dir"])
    label_col     = "label_classif"

    print(f"\n{'='*60}")
    print(f"  Classification — training pipeline (simulation + optional real)  [{network}]")
    print(f"{'='*60}\n")

    # Load simulated training split (already scaled)
    X_train_sim, y_train_sim = load_split(str(splits_dir / "cls_train.csv"), label_col)
    verify_scaler(str(processed_dir / "cls_scaler.pkl"))

    # Load scaler to apply to real data
    scaler = joblib.load(processed_dir / "cls_scaler.pkl")
    print(f"[scaler] Loaded from '{processed_dir / 'cls_scaler.pkl'}'")

    # Combine with real data if provided
    X_train = X_train_sim
    y_train = y_train_sim
    if real_data_path:
        X_real, y_real = load_real_dataset(real_data_path, scaler)
        X_train = np.vstack([X_train, X_real])
        y_train = np.hstack([y_train, y_real])
        print(f"[combined] Training set now has {len(X_train)} samples "
              f"(sim: {len(X_train_sim)} + real: {len(X_real)})")
    else:
        print("[combined] Using simulation data only (no --real-data).")

    # Class distribution summary (training)
    classes, counts = np.unique(y_train, return_counts=True)
    print("\n  Class distribution — training set:")
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

    # Scorer and CV
    scorer = build_weighted_recall_scorer()
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
        description="Train RF, XGBoost, and kNN for fault classification (simulation + optional real data)."
    )
    parser.add_argument(
        "--config",
        type     = str,
        required = True,
        help     = "Path to config.yaml (e.g. 'ieee13/config.yaml')",
    )
    parser.add_argument(
        "--real-data",
        type     = str,
        help     = "Path to CSV with real samples (12 features + label_classif column)",
    )
    args = parser.parse_args()

    try:
        main(args.config, args.real_data)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)