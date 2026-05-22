"""
ieee13/classification/predict.py
--------------------------------
Runs inference with the best classification model selected by mcdm.py.
Filters out 'Sin_Falla' rows. Expects labels in the standard format
(L A, L B, L C, LL AB, LL BC, LL CA, LLG AB, LLG BC, LLG CA, LLL ABC).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.model_selection import StratifiedKFold

sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.io import load_config, load_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLS_6 = ["Va", "Vb", "Vc", "Ia", "Ib", "Ic"]

CLASSIFICATION_LABEL_COL = "label_classif"

def convert_to_native(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native(i) for i in obj]
    else:
        return obj

# ---------------------------------------------------------------------------
# Data loading (filtra Sin_Falla, sin mapeo)
# ---------------------------------------------------------------------------
def load_input_csv(
    csv_path: str,
    filter_sin_falla: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray]]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: '{path}'")

    df = pd.read_csv(path, sep=";", decimal=",")

    feature_cols = FEATURE_COLS_6
    print("[load] Using 6 feature columns (magnitudes only).")
    
    X = df[feature_cols].values

    # Determine label column
    label_col = None
    if CLASSIFICATION_LABEL_COL in df.columns:
        label_col = CLASSIFICATION_LABEL_COL
    elif "Tipo_Falla" in df.columns:
        label_col = "Tipo_Falla"
    elif "Fases_en_Falla" in df.columns:
        label_col = "Fases_en_Falla"

    y = None
    if label_col is not None:
        y = df[label_col].values
        if filter_sin_falla:
            mask = y != "Sin_Falla"
            removed = (~mask).sum()
            if removed > 0:
                print(f"[load] Eliminadas {removed} filas con etiqueta 'Sin_Falla' (solo fallas).")
                X = X[mask]
                y = y[mask]
                df = df.iloc[mask]
        classes, counts = np.unique(y, return_counts=True)
        print(f"[load]  '{path.name}'  →  {X.shape[0]:,} muestras (fallas), {len(classes)} tipos de falla.")
    else:
        print(f"[load]  '{path.name}'  →  {X.shape[0]:,} muestras (sin etiqueta).")

    return df, X, y


def load_scaler(scaler_path: str):
    path = Path(scaler_path)
    if not path.exists():
        raise FileNotFoundError(f"Scaler not found: '{path}'")
    scaler = joblib.load(path)
    print(f"[scaler] Loaded from '{path}'")
    return scaler


def load_best_model_id(results_dir: Path) -> str:
    path = results_dir / "mcdm_result.json"
    if not path.exists():
        raise FileNotFoundError(f"MCDM result not found: '{path}'")
    with open(path, "r") as f:
        result = json.load(f)
    best_model = result["best_model"]
    print(f"[mcdm] Best model: '{best_model}'")
    return best_model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    weighted_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    per_class = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()
    return {
        "weighted_recall": round(float(weighted_recall), 6),
        "per_class_recall": {cls: round(float(val), 6) for cls, val in zip(classes, per_class)},
        "confusion_matrix": cm,
        "classes": classes,
    }


# ---------------------------------------------------------------------------
# Cross‑validation evaluation
# ---------------------------------------------------------------------------
def cross_validate_on_real(model, X: np.ndarray, y: np.ndarray, cv_folds: int = 5, random_state: int = 42) -> Dict:
    classes, counts = np.unique(y, return_counts=True)
    min_count = min(counts)
    if min_count < cv_folds:
        cv_folds = min_count
        print(f"[cv-eval] Reduced folds to {cv_folds} due to low class frequency.")
    if cv_folds < 2:
        raise ValueError("Not enough samples per class for cross‑validation.")

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fold_scores = []
    fold_per_class_recalls = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model_cls = type(model)
        new_model = model_cls(**model.get_params())
        new_model.fit(X_tr, y_tr)

        y_pred = new_model.predict(X_val)
        w_rec = recall_score(y_val, y_pred, average="weighted", zero_division=0)
        fold_scores.append(w_rec)

        classes_fold = np.unique(np.concatenate([y_val, y_pred]))
        per_class = recall_score(y_val, y_pred, labels=classes_fold, average=None, zero_division=0)
        fold_per_class_recalls.append(dict(zip(classes_fold, per_class)))

    mean_weighted = np.mean(fold_scores)
    std_weighted = np.std(fold_scores)

    all_classes = set()
    for d in fold_per_class_recalls:
        all_classes.update(d.keys())
    all_classes = sorted(all_classes)

    per_class_mean = {}
    per_class_std = {}
    for cls in all_classes:
        recalls = [d.get(cls, 0.0) for d in fold_per_class_recalls]
        per_class_mean[cls] = np.mean(recalls)
        per_class_std[cls] = np.std(recalls)

    return {
        "cv_folds": int(cv_folds),
        "weighted_recall": {"mean": float(mean_weighted), "std": float(std_weighted)},
        "per_class_recall": {
            cls: {"mean": float(per_class_mean[cls]), "std": float(per_class_std[cls])}
            for cls in all_classes
        },
    }


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------
def save_predictions_csv(df: pd.DataFrame, y_pred: np.ndarray, out_path: Path) -> None:
    out_df = df.copy()
    out_df["pred_fault_type"] = y_pred
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, sep=";", decimal=",", index=False)
    print(f"[save] Predictions saved → '{out_path}'")

def save_validation_report(mode: str, model_id: str, metrics: Optional[Dict], n_samples: int, out_path: Path) -> None:
    report = {"mode": mode, "model_id": model_id, "n_samples": n_samples, "metrics": metrics}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[save] Validation report saved → '{out_path}'")

def save_cv_results(cv_results: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv_results_native = convert_to_native(cv_results)
    with open(out_path, "w") as f:
        json.dump(cv_results_native, f, indent=2)
    print(f"[cv-eval] Results saved → '{out_path}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(config_path: str, mode: str, input_csv: str, cv_eval: bool) -> None:
    cfg = load_config(config_path)
    network = cfg["network"]["name"]
    processed_dir = Path(cfg["data"]["processed_dir"])
    cls_cfg = cfg["classification"]
    models_dir = Path(cls_cfg["models_dir"])
    results_dir = Path(cls_cfg["results_dir"])

    print(f"\n{'='*60}")
    print(f"  Classification — predict  [{network}]  mode={mode}")
    print(f"{'='*60}\n")

    best_model_id = load_best_model_id(results_dir)
    model = load_model(models_dir, best_model_id)

    filter_sin_falla = (mode == "lab")
    df, X, y = load_input_csv(input_csv, filter_sin_falla=filter_sin_falla)

    if mode == "lab":
        scaler = load_scaler(str(processed_dir / "cls_scaler.pkl"))
        X_scaled = scaler.transform(X)
        print(f"[scaler] Applied to {X_scaled.shape[0]:,} samples")
    else:
        X_scaled = X
        print("[scaler] Skipped — sim input already scaled")

    if mode == "lab" and cv_eval:
        if y is None:
            raise ValueError("Cannot run CV without labels.")
        print("\n[cv-eval] Running stratified cross‑validation on real data...")
        cv_results = cross_validate_on_real(model, X_scaled, y)
        print(f"\n  Weighted recall (CV) = {cv_results['weighted_recall']['mean']:.4f} ± {cv_results['weighted_recall']['std']:.4f}")
        print("\n  Per‑class recall (mean ± std):")
        for cls in sorted(cv_results["per_class_recall"].keys()):
            m = cv_results["per_class_recall"][cls]["mean"]
            s = cv_results["per_class_recall"][cls]["std"]
            print(f"    {cls:<15}  {m:.4f} ± {s:.4f}")
        cv_out = results_dir / f"cv_eval_{Path(input_csv).stem}.json"
        save_cv_results(cv_results, cv_out)
        return

    # Normal inference
    y_pred = model.predict(X_scaled)
    pred_classes, pred_counts = np.unique(y_pred, return_counts=True)
    print(f"\n[predict] {len(y_pred):,} samples classified:")
    for cls, cnt in zip(pred_classes, pred_counts):
        print(f"  {cls:<12}  {cnt:>5}  ({cnt/len(y_pred):.1%})")

    metrics = None
    if y is not None:
        metrics = compute_metrics(y, y_pred)
        print(f"\n  Metrics vs ground truth:")
        print(f"  weighted_recall = {metrics['weighted_recall']:.4f}")
        print("\n  Per-class recall:")
        for cls, val in metrics["per_class_recall"].items():
            print(f"    {cls:<12}  {val:.4f}")

    input_stem = Path(input_csv).stem
    pred_path = results_dir / f"predictions_{mode}_{input_stem}.csv"
    report_path = results_dir / f"validation_{mode}_{input_stem}.json"
    save_predictions_csv(df, y_pred, pred_path)
    save_validation_report(mode, best_model_id, metrics, len(y_pred), report_path)
    print(f"\n[done] Prediction complete for {network}.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=["sim", "lab"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--cv-eval", action="store_true")
    args = parser.parse_args()
    if args.cv_eval and args.mode != "lab":
        print("[ERROR] --cv-eval only for --mode lab", file=sys.stderr)
        sys.exit(1)
    try:
        main(args.config, args.mode, args.input, args.cv_eval)
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)