"""
ieee13/classification/predict.py
---------------------------------
Ejecuta inferencia con el mejor modelo de clasificación elegido por mcdm.py.

Modos de operación
-------------------
--mode sim
    Entrada : CSV con 6 features eléctricas + columna de etiqueta opcional.
              Las features ya están escaladas — el scaler NO se aplica.
    Salida  : CSV de predicciones + reporte JSON de métricas (si hay etiquetas).

--mode lab  [modo principal para validación con datos reales]
    Entrada : CSV con 6 features eléctricas crudas + columna de etiqueta
              (Fases_en_Falla o label_classif con tipos de falla del lab).
              Las features son valores crudos del sensor — el scaler SÍ se aplica.
    Salida  : CSV de predicciones + reporte JSON + reporte de CV (si --cv-eval).

Mapeo de etiquetas de laboratorio
-----------------------------------
Las etiquetas del laboratorio ('Fases_en_Falla') usan un vocabulario distinto
al de la simulación. Este script aplica un mapeo explícito antes de calcular
métricas:

    Laboratorio       →  Simulación
    ─────────────────────────────────
    Monofásica        →  inferido por fase (L A / L B / L C)
    Bifásica          →  inferido por fases (LL AB / LL BC / LL CA)
    Bifásica          →  igual (variante con espacio)
    Bifásica tierra   →  inferido por fases (LLG AB / LLG BC / LLG CA)
    Bif a tierra      →  igual (variante abreviada)
    Trifásico         →  LLL ABC
    Trifásico         →  igual (variante con espacio)

El mapeo completo usa TANTO 'Fases_en_Falla' COMO 'Tipo_Falla' para resolver
la fase exacta (A, B, C, AB, BC, CA). Si una fila no puede mapearse, la etiqueta
queda como "UNKNOWN" y se excluye del cálculo de métricas.

Validación cruzada (--cv-eval)
--------------------------------
Con --mode lab --cv-eval se ejecuta una validación cruzada estratificada
(LeaveOneOut si n < 10, StratifiedKFold de 2-5 pliegues según min_count) sobre
los datos de laboratorio completos. Cada pliegue:
  1. El modelo ya entrenado predice el fold de validación directamente
     (no se re-entrena — el modelo es fijo, entrenado sobre datos simulados).
  2. Se registra weighted recall + per-class recall por fold.
Esto mide qué tan consistente es la predicción del modelo sobre distintos
subconjuntos del laboratorio, sin refitting.

Uso (desde la raíz del proyecto)
----------------------------------
    # Modo simulación
    python3 ieee13/classification/predict.py \\
        --config ieee13/config.yaml \\
        --mode sim \\
        --input data/splits/ieee13/cls_test.csv

    # Validación en laboratorio (inferencia directa)
    python3 ieee13/classification/predict.py \\
        --config ieee13/config.yaml \\
        --mode lab \\
        --input data/real/lab_data.csv

    # Validación en laboratorio + CV de robustez
    python3 ieee13/classification/predict.py \\
        --config ieee13/config.yaml \\
        --mode lab \\
        --input data/real/lab_data.csv \\
        --cv-eval
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

# Agrega raíz del proyecto al path para importar utils/
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.io import load_config, load_model   # helpers compartidos

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

FEATURE_COLS = ["Va", "Vb", "Vc", "Ia", "Ib", "Ic"]

# Columnas de etiqueta aceptadas (en orden de prioridad)
LABEL_COLS_PRIORITY = ["label_classif", "Fases_en_Falla", "Tipo_Falla"]

# Mapeo de tipo general de falla + fases → etiqueta del modelo simulado
# Clave: (tipo_normalizado, fases_normalizadas)  →  etiqueta simulada
_FAULT_TYPE_MAP: Dict[Tuple[str, str], str] = {
    # Monofásica
    ("monofasica", "a"):   "L A",
    ("monofasica", "b"):   "L B",
    ("monofasica", "c"):   "L C",
    # Bifásica sin tierra
    ("bifasica", "ab"):    "LL AB",
    ("bifasica", "bc"):    "LL BC",
    ("bifasica", "ca"):    "LL CA",
    # Bifásica a tierra
    ("bifasicatierra", "ab"):  "LLG AB",
    ("bifasicatierra", "bc"):  "LLG BC",
    ("bifasicatierra", "ca"):  "LLG CA",
    # Trifásica
    ("trifasico", "abc"):  "LLL ABC",
    # Etiquetas simuladas directas (cuando label_classif ya tiene el formato correcto)
    ("l a", ""):    "L A",
    ("l b", ""):    "L B",
    ("l c", ""):    "L C",
    ("ll ab", ""):  "LL AB",
    ("ll bc", ""):  "LL BC",
    ("ll ca", ""):  "LL CA",
    ("llg ab", ""): "LLG AB",
    ("llg bc", ""): "LLG BC",
    ("llg ca", ""): "LLG CA",
    ("lll abc", ""): "LLL ABC",
}

# Normalización de variantes tipográficas del tipo de falla
_TIPO_NORMALIZE: Dict[str, str] = {
    "monofásica":      "monofasica",
    "monofasica":      "monofasica",
    "bifásica":        "bifasica",
    "bifasica":        "bifasica",
    "bifásica ":       "bifasica",
    "bifasica ":       "bifasica",
    "bifásica tierra": "bifasicatierra",
    "bifasica tierra": "bifasicatierra",
    "bif a tierra":    "bifasicatierra",
    "trifásico":       "trifasico",
    "trifasico":       "trifasico",
    "trifásico ":      "trifasico",
    "trifasico ":      "trifasico",
}

# ---------------------------------------------------------------------------
# Mapeo de etiquetas laboratorio → simulación
# ---------------------------------------------------------------------------

def _extract_phases(fases_str: str) -> str:
    """Extrae las fases involucradas de la cadena 'Fases_en_Falla'.

    Ejemplos:
        'L A'    → 'a'
        'LL AB'  → 'ab'
        'LLG BC' → 'bc'
        'LLL ABC'→ 'abc'
    """
    s = fases_str.strip().upper()
    # Quita prefijos L / LL / LLG / LLL
    for prefix in ["LLL ", "LLG ", "LL ", "L "]:
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    # Normaliza orden AB / BC / CA
    phases = "".join(sorted(set(s.replace(" ", ""))))
    # Reordena a convención AB / BC / CA
    if phases in {"AB", "BA"}:   return "ab"
    if phases in {"BC", "CB"}:   return "bc"
    if phases in {"AC", "CA"}:   return "ca"
    if phases in {"ABC", "ACB", "BAC", "BCA", "CAB", "CBA"}: return "abc"
    return phases.lower()


def map_lab_label(row: pd.Series) -> str:
    """Convierte la etiqueta de laboratorio al formato del modelo simulado.

    Intenta varias estrategias en orden:
      1. Si 'label_classif' ya tiene el formato del modelo → úsalo directamente.
      2. Si 'Fases_en_Falla' tiene el formato del modelo (p.ej. 'L A') → úsalo.
      3. Combina 'Tipo_Falla' (categoría) + 'Fases_en_Falla' (fases) para inferir.
    Retorna "UNKNOWN" si ninguna estrategia funciona.
    """
    # Estrategia 1: label_classif directo
    if "label_classif" in row.index and pd.notna(row.get("label_classif", None)):
        val = str(row["label_classif"]).strip()
        key = (val.lower(), "")
        if key in _FAULT_TYPE_MAP:
            return _FAULT_TYPE_MAP[key]

    # Estrategia 2: Fases_en_Falla en formato simulado directo (e.g. 'L A')
    fases = str(row.get("Fases_en_Falla", "")).strip()
    if fases and fases not in ("Sin_Falla", "N.A", "nan"):
        key = (fases.lower(), "")
        if key in _FAULT_TYPE_MAP:
            return _FAULT_TYPE_MAP[key]

        # Estrategia 3: Tipo_Falla + Fases_en_Falla
        tipo_raw = str(row.get("Tipo_Falla", "")).strip()
        tipo_norm = _TIPO_NORMALIZE.get(tipo_raw, _TIPO_NORMALIZE.get(tipo_raw.lower(), ""))
        if tipo_norm:
            phases = _extract_phases(fases)
            key2 = (tipo_norm, phases)
            if key2 in _FAULT_TYPE_MAP:
                return _FAULT_TYPE_MAP[key2]

    return "UNKNOWN"


def build_mapped_labels(df: pd.DataFrame) -> np.ndarray:
    """Aplica map_lab_label a cada fila del DataFrame."""
    return df.apply(map_lab_label, axis=1).values


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_input_csv(
    csv_path: str,
    mode: str,
) -> Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray]]:
    """Carga el CSV de entrada y separa features de etiquetas.

    Parameters
    ----------
    csv_path : str
        Ruta al archivo CSV.
    mode : str
        'sim' o 'lab'.

    Returns
    -------
    (df_original, X, y_mapped_or_None)
        y es None si no hay columna de etiqueta.
        En mode='lab', y ya está mapeado al vocabulario del modelo simulado.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV de entrada no encontrado: '{path}'")

    df = pd.read_csv(path, sep=";", decimal=",")

    missing_feats = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_feats:
        raise ValueError(f"Columnas de features faltantes: {missing_feats}")

    # Filtrar Sin_Falla en modo lab (clasificación solo sobre fallas)
    if mode == "lab" and "Fases_en_Falla" in df.columns:
        sin_falla_mask = df["Fases_en_Falla"].astype(str).str.strip().isin(
            ["Sin_Falla", "N.A", "nan"]
        )
        n_removed = sin_falla_mask.sum()
        if n_removed > 0:
            print(f"[load] Eliminadas {n_removed} filas 'Sin_Falla' "
                  f"(clasificación solo opera sobre fallas).")
            df = df[~sin_falla_mask].reset_index(drop=True)

    X = df[FEATURE_COLS].values

    # Detectar y construir etiquetas
    y = None
    has_label = any(c in df.columns for c in LABEL_COLS_PRIORITY
                    if c not in ("label_classif",)) or "label_classif" in df.columns

    if has_label:
        if mode == "lab":
            y_mapped = build_mapped_labels(df)
            unknown_mask = y_mapped == "UNKNOWN"
            n_unknown = unknown_mask.sum()
            if n_unknown > 0:
                print(f"[load] ADVERTENCIA: {n_unknown} filas con etiqueta no mapeada "
                      f"(UNKNOWN) — se excluirán de métricas.")
            y = y_mapped
        else:
            # mode == 'sim': usa label_classif directamente
            for col in LABEL_COLS_PRIORITY:
                if col in df.columns:
                    y = df[col].values
                    break

        if y is not None:
            classes, counts = np.unique(y[y != "UNKNOWN"] if mode == "lab" else y,
                                        return_counts=True)
            print(f"[load]  '{path.name}'  →  {X.shape[0]:,} muestras  "
                  f"({len(classes)} tipos de falla):")
            for cls, cnt in zip(classes, counts):
                print(f"         {cls:<15}  {cnt:>3}")
    else:
        print(f"[load]  '{path.name}'  →  {X.shape[0]:,} muestras  "
              f"(sin columna de etiqueta)")

    return df, X, y


def load_scaler(scaler_path: str):
    path = Path(scaler_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Scaler no encontrado: '{path}'\n"
            "Ejecuta utils/split.py primero."
        )
    scaler = joblib.load(path)
    print(f"[scaler] Cargado desde '{path}'")
    return scaler


def load_best_model_id(results_dir: Path) -> str:
    path = results_dir / "mcdm_result.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Resultado MCDM no encontrado: '{path}'\n"
            "Ejecuta ieee13/classification/mcdm.py primero."
        )
    with open(path, "r", encoding="utf-8") as f:
        result = json.load(f)
    best = result["best_model"]
    print(f"[mcdm]  Mejor modelo: '{best}'")
    return best


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Weighted recall + per-class recall + confusion matrix."""
    # Excluye UNKNOWN del cálculo
    mask = y_true != "UNKNOWN"
    y_t = y_true[mask]
    y_p = y_pred[mask]

    if len(y_t) == 0:
        return {"weighted_recall": 0.0, "per_class_recall": {}, "confusion_matrix": [], "classes": []}

    classes = sorted(np.unique(np.concatenate([y_t, y_p])))
    wr = recall_score(y_t, y_p, average="weighted", zero_division=0)
    per_class = recall_score(y_t, y_p, labels=classes, average=None, zero_division=0)
    cm = confusion_matrix(y_t, y_p, labels=classes).tolist()

    return {
        "weighted_recall":  round(float(wr), 6),
        "per_class_recall": {cls: round(float(v), 6) for cls, v in zip(classes, per_class)},
        "confusion_matrix": cm,
        "classes":          classes,
        "n_evaluated":      int(mask.sum()),
        "n_unknown":        int((~mask).sum()),
    }


# ---------------------------------------------------------------------------
# Validación cruzada sobre datos de laboratorio (modelo fijo, sin re-entrenamiento)
# ---------------------------------------------------------------------------

def cross_validate_lab(
    model,
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
) -> Dict:
    """CV sobre datos de lab usando el modelo fijo (sin re-entrenamiento).

    Estrategia:
    - Excluye filas UNKNOWN.
    - Si min_count_per_class == 1 → LeaveOneOut (LOO).
    - Si min_count >= 2 → StratifiedKFold con k = min(5, min_count).

    En cada fold solo se evalúa; el modelo NO se re-entrena.
    Esto mide consistencia de las predicciones del modelo sobre distintos
    subconjuntos del laboratorio.
    """
    mask = y != "UNKNOWN"
    X_clean = X[mask]
    y_clean = y[mask]
    n = len(y_clean)

    if n == 0:
        raise ValueError("No hay muestras válidas (sin UNKNOWN) para CV.")

    classes, counts = np.unique(y_clean, return_counts=True)
    min_count = int(counts.min())

    print(f"\n[cv-eval] {n} muestras válidas, {len(classes)} clases, "
          f"mín. muestras por clase = {min_count}")

    fold_scores = []
    fold_per_class: List[Dict[str, float]] = []

    if min_count == 1:
        # LeaveOneOut: la única opción cuando hay clases con 1 sola muestra
        print("[cv-eval] Usando LeaveOneOut (hay clases con n=1).")
        cv = LeaveOneOut()
        splits = list(cv.split(X_clean))
    else:
        n_folds = min(5, min_count)
        print(f"[cv-eval] Usando StratifiedKFold con k={n_folds}.")
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        splits = list(cv.split(X_clean, y_clean))

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_val_fold = X_clean[val_idx]
        y_val_fold = y_clean[val_idx]

        # El modelo es fijo — solo predict, sin fit
        y_pred_fold = model.predict(X_val_fold)

        wr = recall_score(y_val_fold, y_pred_fold, average="weighted", zero_division=0)
        fold_scores.append(float(wr))

        classes_fold = np.unique(np.concatenate([y_val_fold, y_pred_fold]))
        pc = recall_score(y_val_fold, y_pred_fold,
                          labels=classes_fold, average=None, zero_division=0)
        fold_per_class.append(dict(zip(classes_fold.tolist(), pc.tolist())))

    mean_wr = float(np.mean(fold_scores))
    std_wr  = float(np.std(fold_scores))

    all_classes = sorted({cls for d in fold_per_class for cls in d})
    pc_stats = {}
    for cls in all_classes:
        vals = [d.get(cls, 0.0) for d in fold_per_class]
        pc_stats[cls] = {
            "mean": round(float(np.mean(vals)), 6),
            "std":  round(float(np.std(vals)),  6),
        }

    strategy = "LeaveOneOut" if min_count == 1 else f"StratifiedKFold(k={min(5, min_count)})"

    return {
        "strategy":        strategy,
        "n_folds":         len(splits),
        "n_samples":       n,
        "weighted_recall": {"mean": round(mean_wr, 6), "std": round(std_wr, 6)},
        "fold_scores":     [round(s, 6) for s in fold_scores],
        "per_class_recall": pc_stats,
    }


# ---------------------------------------------------------------------------
# Guardado de resultados
# ---------------------------------------------------------------------------

def save_predictions_csv(df: pd.DataFrame, y_pred: np.ndarray,
                          y_mapped: Optional[np.ndarray], out_path: Path) -> None:
    out = df.copy()
    out["pred_fault_type"] = y_pred
    if y_mapped is not None:
        out["label_mapped"] = y_mapped
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep=";", decimal=",", index=False)
    print(f"[save]  Predicciones → '{out_path}'")


def save_json(data: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Convierte tipos numpy a nativos
    def _native(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        if isinstance(obj, dict):           return {k: _native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):  return [_native(i) for i in obj]
        return obj
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_native(data), f, indent=2, ensure_ascii=False)
    print(f"[save]  Reporte → '{out_path}'")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def main(config_path: str, mode: str, input_csv: str, cv_eval: bool) -> None:
    cfg           = load_config(config_path)
    network       = cfg["network"]["name"]
    processed_dir = Path(cfg["data"]["processed_dir"])
    cls_cfg       = cfg["classification"]
    models_dir    = Path(cls_cfg["models_dir"])
    results_dir   = Path(cls_cfg["results_dir"])

    print(f"\n{'='*60}")
    print(f"  Clasificación — predict  [{network}]  modo={mode}")
    print(f"{'='*60}\n")

    # ── Modelo ──────────────────────────────────────────────────────────────
    best_model_id = load_best_model_id(results_dir)
    model         = load_model(models_dir, best_model_id)

    # LabelEncoder (solo XGBoost)
    le = None
    le_path = models_dir / "label_encoder.pkl"
    if best_model_id == "XGB" and le_path.exists():
        le = joblib.load(le_path)
        print(f"[le]    LabelEncoder cargado ({len(le.classes_)} clases)")

    # ── Datos ────────────────────────────────────────────────────────────────
    df, X, y = load_input_csv(input_csv, mode)

    # ── Escalado ─────────────────────────────────────────────────────────────
    if mode == "lab":
        scaler   = load_scaler(str(processed_dir / "cls_scaler.pkl"))
        X_scaled = scaler.transform(X)
        print(f"[scaler] Aplicado a {X_scaled.shape[0]:,} muestras")
    else:
        X_scaled = X
        print("[scaler] Omitido — datos sim ya escalados")

    # ── Validación cruzada (opcional, solo lab) ───────────────────────────────
    if cv_eval:
        if mode != "lab":
            raise ValueError("--cv-eval solo está disponible en --mode lab")
        if y is None:
            raise ValueError("--cv-eval requiere etiquetas de ground truth en el CSV.")

        print(f"\n{'─'*60}")
        print(f"  CV de robustez (modelo fijo, sin re-entrenamiento)")
        print(f"{'─'*60}")
        cv_results = cross_validate_lab(model, X_scaled, y)

        wr_mean = cv_results["weighted_recall"]["mean"]
        wr_std  = cv_results["weighted_recall"]["std"]
        print(f"\n  Estrategia : {cv_results['strategy']}")
        print(f"  Pliegues   : {cv_results['n_folds']}")
        print(f"  Weighted recall (CV) = {wr_mean:.4f} ± {wr_std:.4f}")
        print(f"\n  Per-class recall (mean ± std):")
        for cls in sorted(cv_results["per_class_recall"]):
            m = cv_results["per_class_recall"][cls]["mean"]
            s = cv_results["per_class_recall"][cls]["std"]
            print(f"    {cls:<15}  {m:.4f} ± {s:.4f}")

        stem    = Path(input_csv).stem
        cv_path = results_dir / f"cv_eval_{mode}_{stem}.json"
        save_json(cv_results, cv_path)

        # También hace inferencia completa tras el CV
        print(f"\n{'─'*60}")
        print("  Continuando con inferencia completa...")
        print(f"{'─'*60}")

    # ── Inferencia ────────────────────────────────────────────────────────────
    y_pred = model.predict(X_scaled)
    if le is not None:
        y_pred = le.inverse_transform(y_pred)

    pred_classes, pred_counts = np.unique(y_pred, return_counts=True)
    print(f"\n[predict] {len(y_pred):,} muestras clasificadas:")
    for cls, cnt in zip(pred_classes, pred_counts):
        print(f"  {cls:<15}  {cnt:>3}  ({cnt/len(y_pred):.1%})")

    # ── Métricas vs ground truth ──────────────────────────────────────────────
    metrics = None
    if y is not None:
        metrics = compute_metrics(y, y_pred)
        print(f"\n  Métricas vs ground truth (n={metrics['n_evaluated']}, "
              f"excluidos UNKNOWN={metrics['n_unknown']}):")
        print(f"  weighted_recall = {metrics['weighted_recall']:.4f}")
        print(f"\n  Per-class recall:")
        for cls, val in metrics["per_class_recall"].items():
            print(f"    {cls:<15}  {val:.4f}")

    # ── Guardar resultados ────────────────────────────────────────────────────
    stem        = Path(input_csv).stem
    pred_path   = results_dir / f"predictions_{mode}_{stem}.csv"
    report_path = results_dir / f"validation_{mode}_{stem}.json"

    save_predictions_csv(df, y_pred, y if mode == "lab" else None, pred_path)
    save_json({
        "mode":      mode,
        "model_id":  best_model_id,
        "n_samples": int(len(y_pred)),
        "metrics":   metrics,
    }, report_path)

    print(f"\n[done]  Clasificación completa para {network}  (modo={mode}).")
    print(f"        Resultados → '{results_dir}'\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inferencia de clasificación de fallas (modo simulación o laboratorio)."
    )
    parser.add_argument("--config", required=True,
                        help="Ruta a config.yaml (ej. 'ieee13/config.yaml')")
    parser.add_argument("--mode",   required=True, choices=["sim", "lab"],
                        help="'sim' para datos simulados, 'lab' para datos de laboratorio")
    parser.add_argument("--input",  required=True,
                        help="Ruta al CSV de entrada con las 6 features eléctricas")
    parser.add_argument("--cv-eval", action="store_true",
                        help="Ejecuta CV de robustez sobre los datos (solo --mode lab)")
    args = parser.parse_args()

    if args.cv_eval and args.mode != "lab":
        print("[ERROR] --cv-eval solo está disponible con --mode lab", file=sys.stderr)
        sys.exit(1)

    try:
        main(args.config, args.mode, args.input, args.cv_eval)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)