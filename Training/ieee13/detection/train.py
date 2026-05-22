"""
ieee13/detection/train.py
Entrenamiento de modelos (RF, SVM, MLP) con datos simulados (splits) + datos reales (opcional).
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.io import (
    load_config,
    load_split,
    verify_scaler,
    build_pairwise_matrix,
)
from utils.ahp_weights import ahp_weights

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ----------------------------------------------------------------------
# Cargar dataset real y aplicar el scaler ya existente (sin reajustar)
# ----------------------------------------------------------------------
def load_real_dataset(csv_path: str, scaler: StandardScaler) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Real dataset not found: {path}")
    df = pd.read_csv(path, sep=";", decimal=",")
    feature_cols = ["Va", "Vb", "Vc", "Ia", "Ib", "Ic"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Real dataset missing features: {missing}")
    X = df[feature_cols].values

    if "label_detection" in df.columns:
        y = df["label_detection"].values
    elif "Tipo_Falla" in df.columns:
        y = (df["Tipo_Falla"] != "Sin_Falla").astype(int).values
    elif "Fases_en_Falla" in df.columns:
        y = (df["Fases_en_Falla"] != "Sin_Falla").astype(int).values
    else:
        raise ValueError("Real dataset must contain 'label_detection', 'Tipo_Falla' or 'Fases_en_Falla'")

    X_scaled = scaler.transform(X)
    print(f"[real] Loaded {len(X_scaled)} samples from '{path.name}' (fault ratio: {y.mean():.1%})")
    return X_scaled, y

# ----------------------------------------------------------------------
# Aumento de datos con ruido
# ----------------------------------------------------------------------
def augment_with_noise(X: np.ndarray, y: np.ndarray,
                       noise_levels: List[float] = [0.005, 0.01],
                       random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    X_aug = [X]
    y_aug = [y]
    for noise_std in noise_levels:
        noise = rng.normal(loc=0.0, scale=noise_std, size=X.shape)
        X_aug.append(X + noise)
        y_aug.append(y)
    X_final = np.vstack(X_aug)
    y_final = np.hstack(y_aug)
    print(f"[augmentation] Original: {len(X)} → Augmented: {len(X_final)}")
    return X_final, y_final

# ----------------------------------------------------------------------
# Scorer AHP
# ----------------------------------------------------------------------
def build_ahp_scorer(weights: np.ndarray):
    w_recall, w_spec = weights[0], weights[1]
    def ahp_score(estimator, X, y):
        y_pred = estimator.predict(X)
        recall = recall_score(y, y_pred, pos_label=1, zero_division=0)
        specificity = recall_score(y, y_pred, pos_label=0, zero_division=0)
        return w_recall * recall + w_spec * specificity
    return ahp_score

# ----------------------------------------------------------------------
# Fábrica de modelos
# ----------------------------------------------------------------------
def build_model(model_key: str, cfg: Dict):
    model_cfg = cfg["detection"][model_key]
    model_id = model_cfg["model_id"]
    if model_key == "random_forest":
        model = RandomForestClassifier(
            class_weight=model_cfg["class_weight"],
            random_state=model_cfg["random_state"],
            n_jobs=model_cfg["n_jobs"]
        )
        grid = dict(model_cfg["param_grid"])
        grid["max_depth"] = [None if v is None else int(v) for v in grid["max_depth"]]
    elif model_key == "svm":
        model = SVC(
            probability=model_cfg["probability"],
            class_weight=model_cfg["class_weight"],
            random_state=model_cfg["random_state"]
        )
        grid = dict(model_cfg["param_grid"])
    elif model_key == "mlp":
        model = MLPClassifier(
            random_state=model_cfg["random_state"],
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
        grid = dict(model_cfg["param_grid"])
        grid["hidden_layer_sizes"] = [tuple(v) for v in grid["hidden_layer_sizes"]]
    else:
        raise ValueError(f"Unknown model key: {model_key}")
    return model, grid, model_id

# ----------------------------------------------------------------------
# Verificación de sobreajuste
# ----------------------------------------------------------------------
def overfitting_check(grid_search, X_train, y_train, scorer, model_id, threshold=0.05):
    best_model = grid_search.best_estimator_
    train_score = scorer(best_model, X_train, y_train)
    cv_score = grid_search.best_score_
    gap = train_score - cv_score
    overfit = gap > threshold
    status = "⚠️ possible overfit" if overfit else "✓ ok"
    print(f"  [{model_id}] train={train_score:.4f}  cv={cv_score:.4f}  gap={gap:.4f}  {status}")
    return {
        "model_id": model_id,
        "train_score": round(float(train_score), 6),
        "cv_score": round(float(cv_score), 6),
        "gap": round(float(gap), 6),
        "overfit_flag": bool(overfit),
    }

# ----------------------------------------------------------------------
# Guardado de modelos y resultados
# ----------------------------------------------------------------------
def save_model(estimator, models_dir: Path, model_id: str):
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / f"{model_id}.pkl"
    joblib.dump(estimator, out_path)
    print(f"  [{model_id}] Model saved → '{out_path}'")

def save_cv_results(grid_search, results_dir: Path, model_id: str):
    results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(grid_search.cv_results_)
    keep = [c for c in df.columns if not c.startswith("split")]
    df = df[keep].sort_values("rank_test_score")
    out = results_dir / f"cv_results_{model_id}.csv"
    df.to_csv(out, sep=";", decimal=",", index=False)
    print(f"  [{model_id}] CV results saved → '{out}'")

def save_overfit_report(records: List[Dict], results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "overfit_check.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"[overfit] Summary saved → '{out}'")

# ----------------------------------------------------------------------
# Entrenamiento de un modelo
# ----------------------------------------------------------------------
def train_model(model_key: str, cfg: Dict,
                X_train: np.ndarray, y_train: np.ndarray,
                scorer, cv, models_dir: Path, results_dir: Path):
    model, grid, model_id = build_model(model_key, cfg)
    n_combinations = 1
    for v in grid.values():
        n_combinations *= len(v)
    print(f"\n{'─'*60}")
    print(f"  Training : {model_id}  ({n_combinations} combinations × {cfg['detection']['cv']['n_splits']} folds)")
    print(f"{'─'*60}")
    t0 = time.time()
    gs = RandomizedSearchCV(
        estimator=model,
        param_distributions=grid,
        n_iter=20,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
        random_state=42,
        error_score="raise"
    )
    gs.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  [{model_id}] Done in {elapsed:.1f}s  |  best CV score: {gs.best_score_:.4f}")
    print(f"  [{model_id}] Best params: {gs.best_params_}")
    overfit_info = overfitting_check(gs, X_train, y_train, scorer, model_id)
    save_model(gs.best_estimator_, models_dir, model_id)
    save_cv_results(gs, results_dir, model_id)
    return gs, overfit_info

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(config_path: str, real_data_path: Optional[str] = None):
    cfg = load_config(config_path)
    network = cfg["network"]["name"]
    splits_dir = Path(cfg["data"]["splits_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    det_cfg = cfg["detection"]
    models_dir = Path(det_cfg["models_dir"])
    results_dir = Path(det_cfg["results_dir"])
    label_col = "label_detection"

    print(f"\n{'='*60}")
    print(f"  Detection — training pipeline (combined simulation + real)  [{network}]")
    print(f"{'='*60}\n")

    # Cargar splits simulados (ya escalados)
    X_train_sim, y_train_sim = load_split(str(splits_dir / "det_train.csv"), label_col)
    X_val, y_val = load_split(str(splits_dir / "det_val.csv"), label_col)
    verify_scaler(str(processed_dir / "det_scaler.pkl"))

    # Cargar el escalador (ajustado en split.py)
    scaler = joblib.load(processed_dir / "det_scaler.pkl")
    print(f"[scaler] Loaded from '{processed_dir / 'det_scaler.pkl'}'")

    # Concatenar datos reales si se proporcionan
    X_train = X_train_sim
    y_train = y_train_sim
    if real_data_path:
        X_real, y_real = load_real_dataset(real_data_path, scaler)
        X_train = np.vstack([X_train, X_real])
        y_train = np.hstack([y_train, y_real])
        print(f"[combined] Training set now has {len(X_train)} samples (sim: {len(X_train_sim)} + real: {len(X_real)})")

    # Aumento de datos con ruido
    X_train, y_train = augment_with_noise(X_train, y_train, noise_levels=[0.005, 0.01])

    print(f"\n  Class balance — train : fault={y_train.mean():.1%}  no-fault={1-y_train.mean():.1%}")
    print(f"  Class balance — val   : fault={y_val.mean():.1%}  no-fault={1-y_val.mean():.1%}\n")

    # Pesos AHP
    metric_names = ["Recall", "Specificity"]
    pairwise_matrix = build_pairwise_matrix(det_cfg["ahp"]["pairwise_matrix"])
    weights, CR = ahp_weights(pairwise_matrix, metric_names)
    if CR >= 0.10:
        raise ValueError(f"AHP inconsistent (CR={CR:.4f})")
    scorer = build_ahp_scorer(weights)

    cv_cfg = det_cfg["cv"]
    cv = StratifiedKFold(
        n_splits=cv_cfg["n_splits"],
        shuffle=cv_cfg["shuffle"],
        random_state=cv_cfg["random_state"]
    )

    overfit_records = []
    for model_key in ["random_forest", "svm", "mlp"]:
        _, overfit_info = train_model(
            model_key=model_key,
            cfg=cfg,
            X_train=X_train,
            y_train=y_train,
            scorer=scorer,
            cv=cv,
            models_dir=models_dir,
            results_dir=results_dir
        )
        overfit_records.append(overfit_info)

    print(f"\n{'='*60}")
    print("  Overfitting check summary")
    print(f"{'='*60}")
    for r in overfit_records:
        flag = "⚠️" if r["overfit_flag"] else "✓"
        print(f"  {r['model_id']:<6}  train={r['train_score']:.4f}  cv={r['cv_score']:.4f}  gap={r['gap']:.4f}  {flag}")
    save_overfit_report(overfit_records, results_dir)

    print(f"\n[done]  Training complete for {network}.")
    print(f"        Models  → '{models_dir}'")
    print(f"        Results → '{results_dir}'\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--real-data", help="Path to CSV with real samples (same features + label)")
    args = parser.parse_args()
    try:
        main(args.config, args.real_data)
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)