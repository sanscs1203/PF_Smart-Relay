"""
ieee13/detection/predict.py (con nested CV, matriz de confusión y umbrales)
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

sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.io import load_config, load_model

FEATURE_COLS = ["Va", "Vb", "Vc", "Ia", "Ib", "Ic"]
DETECTION_LABEL_COL = "label_detection"

# ----------------------------------------------------------------------
# Conversión de etiquetas
# ----------------------------------------------------------------------
def to_binary_label(series, source_col: str) -> np.ndarray:
    if source_col == DETECTION_LABEL_COL:
        return series.values
    elif source_col in ("Fases_en_Falla", "Tipo_Falla"):
        return (series != "Sin_Falla").astype(int).values
    else:
        raise ValueError(f"No sé cómo convertir {source_col}")

# ----------------------------------------------------------------------
# Carga de CSV (con filtro opcional de resistencia 100Ω)
# ----------------------------------------------------------------------
def load_input_csv(csv_path: str, filter_resistance: bool = False):
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path, sep=";", decimal=",")
    if filter_resistance and 'Resistencia_Falla' in df.columns:
        resist = pd.to_numeric(df['Resistencia_Falla'].astype(str).str.replace(',','.'), errors='coerce')
        before = len(df)
        df = df[(resist != 100) | (pd.isna(resist))]
        removed = before - len(df)
        if removed > 0:
            print(f"[filter] Eliminadas {removed} filas con resistencia = 100 Ω")
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    X = df[FEATURE_COLS].values
    label_col = None
    if DETECTION_LABEL_COL in df.columns:
        label_col = DETECTION_LABEL_COL
    elif "Fases_en_Falla" in df.columns:
        label_col = "Fases_en_Falla"
    elif "Tipo_Falla" in df.columns:
        label_col = "Tipo_Falla"
    y = None
    if label_col is not None:
        y = to_binary_label(df[label_col], label_col)
        fault_rate = y.mean()
        print(f"[load]  '{path.name}'  → {X.shape[0]} samples (labels from '{label_col}': fault={fault_rate:.1%})")
    else:
        print(f"[load]  '{path.name}'  → {X.shape[0]} samples (no labels)")
    return df, X, y

def load_scaler(scaler_path: str):
    path = Path(scaler_path)
    if not path.exists():
        raise FileNotFoundError(f"Scaler not found: {path}")
    scaler = joblib.load(path)
    print(f"[scaler] Loaded from '{path}'")
    return scaler

def load_best_model_id(results_dir: Path):
    path = results_dir / "mcdm_result.json"
    if not path.exists():
        raise FileNotFoundError(f"MCDM result not found: {path}")
    with open(path) as f:
        result = json.load(f)
    best_model = result["best_model"]
    w_ahp = np.array(result["weights"]["ahp"])
    print(f"[mcdm] Best model: '{best_model}'")
    return best_model, w_ahp

def load_threshold(results_dir: Path, model_id: str) -> float:
    path = results_dir / f"metrics_{model_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with open(path) as f:
        metrics = json.load(f)
    threshold = metrics["optimal_threshold"]
    print(f"[threshold] Optimal threshold for {model_id}: {threshold:.2f}")
    return float(threshold)

# ----------------------------------------------------------------------
# Optimización de umbral con CV simple
# ----------------------------------------------------------------------
def optimize_threshold_cv(model, X: np.ndarray, y: np.ndarray, scaler,
                          cv_folds: int = 5, n_thresholds: int = 41,
                          random_state: int = 42) -> Tuple[float, Dict]:
    from sklearn.model_selection import StratifiedKFold
    thresholds = np.linspace(0.10, 0.90, n_thresholds)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = []
    for th in thresholds:
        fold_scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_val_scaled = scaler.transform(X[val_idx])
            y_prob = model.predict_proba(X_val_scaled)[:, 1]
            y_pred = (y_prob >= th).astype(int)
            recall = recall_score(y[val_idx], y_pred, pos_label=1, zero_division=0)
            spec = recall_score(y[val_idx], y_pred, pos_label=0, zero_division=0)
            gmean = np.sqrt(recall * spec) if (recall * spec) > 0 else 0.0
            fold_scores.append(gmean)
        cv_scores.append(np.mean(fold_scores))
    best_idx = np.argmax(cv_scores)
    best_th = thresholds[best_idx]
    return best_th, {"best_threshold": float(best_th), "best_gmean": float(cv_scores[best_idx])}

# ----------------------------------------------------------------------
# Validación cruzada anidada (nested CV) con matriz de confusión promedio
# ----------------------------------------------------------------------
def nested_cv_evaluate(model, X: np.ndarray, y: np.ndarray, scaler, w_ahp: np.ndarray,
                       outer_folds: int = 5, inner_folds: int = 5,
                       n_thresholds: int = 41, random_state: int = 42) -> Dict:
    from sklearn.model_selection import StratifiedKFold
    outer_skf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
    outer_recalls = []
    outer_specs = []
    outer_gmeans = []
    outer_ahp = []
    outer_thresholds = []
    cm_sum = np.zeros((2, 2), dtype=int)

    for train_idx, test_idx in outer_skf.split(X, y):
        X_train_inner = X[train_idx]
        y_train_inner = y[train_idx]
        X_test_outer = X[test_idx]
        y_test_outer = y[test_idx]

        best_th, _ = optimize_threshold_cv(model, X_train_inner, y_train_inner, scaler,
                                           cv_folds=inner_folds, n_thresholds=n_thresholds,
                                           random_state=random_state)

        X_test_scaled = scaler.transform(X_test_outer)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_prob >= best_th).astype(int)

        recall = recall_score(y_test_outer, y_pred, pos_label=1, zero_division=0)
        spec = recall_score(y_test_outer, y_pred, pos_label=0, zero_division=0)
        gmean = np.sqrt(recall * spec) if recall * spec > 0 else 0.0
        ahp = w_ahp[0] * recall + w_ahp[1] * spec

        outer_recalls.append(recall)
        outer_specs.append(spec)
        outer_gmeans.append(gmean)
        outer_ahp.append(ahp)
        outer_thresholds.append(best_th)

        cm = confusion_matrix(y_test_outer, y_pred)
        cm_sum += cm

    cm_avg = cm_sum / outer_folds
    cm_avg_percent = cm_avg / cm_avg.sum() * 100

    results = {
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "thresholds_optimized": [float(t) for t in outer_thresholds],
        "recall": {"mean": float(np.mean(outer_recalls)), "std": float(np.std(outer_recalls)), "values": outer_recalls},
        "specificity": {"mean": float(np.mean(outer_specs)), "std": float(np.std(outer_specs)), "values": outer_specs},
        "gmean": {"mean": float(np.mean(outer_gmeans)), "std": float(np.std(outer_gmeans)), "values": outer_gmeans},
        "ahp_score": {"mean": float(np.mean(outer_ahp)), "std": float(np.std(outer_ahp)), "values": outer_ahp},
        "confusion_matrix_avg": cm_avg.tolist(),
        "confusion_matrix_avg_percent": cm_avg_percent.tolist(),
    }
    return results

# ----------------------------------------------------------------------
# Predicción simple
# ----------------------------------------------------------------------
def predict(model, X_scaled: np.ndarray, threshold: float):
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_scaled)[:, 1]
    else:
        scores = model.decision_function(X_scaled)
        y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    y_pred = (y_prob >= threshold).astype(int)
    return y_pred, y_prob

def compute_metrics(y_true, y_pred, y_prob, weights):
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    ahp = weights[0] * recall + weights[1] * specificity
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {"recall": round(recall, 6), "specificity": round(specificity, 6),
            "ahp_score": round(ahp, 6), "confusion_matrix": cm}

def save_predictions_csv(df, y_pred, y_prob, out_path):
    out_df = df.copy()
    out_df["pred_label"] = y_pred
    out_df["pred_prob_fault"] = np.round(y_prob, 6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, sep=";", decimal=",", index=False)
    print(f"[save] Predictions saved → '{out_path}'")

def save_validation_report(mode, model_id, threshold, metrics, n_samples, out_path):
    report = {"mode": mode, "model_id": model_id, "threshold": round(threshold, 4),
              "n_samples": n_samples, "metrics": metrics}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[save] Validation report saved → '{out_path}'")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(config_path: str, mode: str, input_csv: str, optimize_threshold: bool, nested_cv: bool):
    cfg = load_config(config_path)
    network = cfg["network"]["name"]
    processed_dir = Path(cfg["data"]["processed_dir"])
    det_cfg = cfg["detection"]
    models_dir = Path(det_cfg["models_dir"])
    results_dir = Path(det_cfg["results_dir"])

    print(f"\n{'='*60}\n  Detection — predict  [{network}]  mode={mode}\n{'='*60}\n")
    best_model_id, w_ahp = load_best_model_id(results_dir)
    model = load_model(models_dir, best_model_id)

    filter_res = (mode == "lab")
    df, X, y = load_input_csv(input_csv, filter_resistance=filter_res)

    if mode == "lab":
        scaler = load_scaler(str(processed_dir / "det_scaler.pkl"))
        X_scaled = scaler.transform(X)
        print(f"[scaler] Applied to {X_scaled.shape[0]} samples")
    else:
        X_scaled = X
        print("[sim] Using already scaled features")

    if optimize_threshold and nested_cv and y is not None:
        print("\n[threshold] Running NESTED CROSS-VALIDATION for unbiased evaluation...")
        nested_results = nested_cv_evaluate(model, X, y, scaler, w_ahp,
                                            outer_folds=5, inner_folds=5,
                                            n_thresholds=41, random_state=42)
        out_path = results_dir / "nested_cv_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(nested_results, f, indent=2)
        print(f"\n[nested-cv] Results saved to '{out_path}'")
        # Métricas
        print(f"  Recall       = {nested_results['recall']['mean']:.4f} ± {nested_results['recall']['std']:.4f}")
        print(f"  Specificity  = {nested_results['specificity']['mean']:.4f} ± {nested_results['specificity']['std']:.4f}")
        print(f"  G-mean       = {nested_results['gmean']['mean']:.4f} ± {nested_results['gmean']['std']:.4f}")
        print(f"  AHP score    = {nested_results['ahp_score']['mean']:.4f} ± {nested_results['ahp_score']['std']:.4f}")
        # Umbral
        th_list = nested_results["thresholds_optimized"]
        th_mean = np.mean(th_list)
        th_std = np.std(th_list)
        print(f"  Threshold    = {th_mean:.4f} ± {th_std:.4f}")
        print(f"  Thresholds per outer fold: {[f'{t:.2f}' for t in th_list]}")
        # Matriz de confusión
        cm_avg = np.array(nested_results["confusion_matrix_avg"])
        cm_pct = np.array(nested_results["confusion_matrix_avg_percent"])
        print("\n  Average confusion matrix (counts, over outer folds):")
        print(f"    TN={cm_avg[0,0]:.1f}  FP={cm_avg[0,1]:.1f}")
        print(f"    FN={cm_avg[1,0]:.1f}  TP={cm_avg[1,1]:.1f}")
        print("\n  Average confusion matrix (percentages):")
        print(f"    TN={cm_pct[0,0]:.1f}%  FP={cm_pct[0,1]:.1f}%")
        print(f"    FN={cm_pct[1,0]:.1f}%  TP={cm_pct[1,1]:.1f}%")
        return

    # Caso normal: usar umbral de evaluate.py
    threshold = load_threshold(results_dir, best_model_id)
    y_pred, y_prob = predict(model, X_scaled, threshold)
    n_fault = int(y_pred.sum())
    n_no_fault = len(y_pred) - n_fault
    print(f"[predict] fault={n_fault}  no-fault={n_no_fault} (threshold={threshold:.2f})")

    metrics = None
    if y is not None:
        metrics = compute_metrics(y, y_pred, y_prob, w_ahp)
        print(f"\n  Metrics vs ground truth:")
        print(f"  recall      = {metrics['recall']:.4f}")
        print(f"  specificity = {metrics['specificity']:.4f}")
        print(f"  ahp_score   = {metrics['ahp_score']:.4f}")
        cm = metrics["confusion_matrix"]
        print(f"  confusion matrix:\n    TN={cm[0][0]}  FP={cm[0][1]}\n    FN={cm[1][0]}  TP={cm[1][1]}")

    input_stem = Path(input_csv).stem
    pred_path = results_dir / f"predictions_{mode}_{input_stem}.csv"
    report_path = results_dir / f"validation_{mode}_{input_stem}.json"
    save_predictions_csv(df, y_pred, y_prob, pred_path)
    save_validation_report(mode, best_model_id, threshold, metrics, len(y_pred), report_path)
    print(f"\n[done] Prediction complete for {network}  (mode={mode})\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=["sim", "lab"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--optimize-threshold", action="store_true")
    parser.add_argument("--nested-cv", action="store_true")
    args = parser.parse_args()
    if args.nested_cv and not args.optimize_threshold:
        print("[ERROR] --nested-cv requires --optimize-threshold", file=sys.stderr)
        sys.exit(1)
    try:
        main(args.config, args.mode, args.input, args.optimize_threshold, args.nested_cv)
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)