"""
utils/preprocess.py
-------------------
Preprocesa el dataset simulado de IEEE 13-bus.
Aplica filtros de resistencia (100 Ω) y eliminación de fallas LLG/LLL con 25 Ω.
Luego submuestreo estratificado (opcional) y balanceo a proporción fault/no-fault ajustable.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# CONSTANTES
# ----------------------------------------------------------------------
FEATURE_COLS = ["Va", "Vb", "Vc", "Ia", "Ib", "Ic"]
NO_FAULT_LABEL = "Sin_Falla"
DETECTION_LABEL_COL = "label_detection"
CLASSIF_LABEL_COL = "label_classif"

# ----------------------------------------------------------------------
# FILTROS
# ----------------------------------------------------------------------
def filter_100_ohm_resistance(df):
    """Elimina filas con resistencia = 100 Ω."""
    if 'Resistencia_Falla' not in df.columns:
        return df
    df = df.copy()
    resist = pd.to_numeric(df['Resistencia_Falla'].astype(str).str.replace(',','.'), errors='coerce')
    before = len(df)
    df = df[(resist != 100) | (pd.isna(resist))]
    print(f"[filter] Eliminadas {before - len(df)} filas con resistencia = 100 Ω")
    return df

def filter_llg_lll_resistance_25(df):
    """Elimina fallas LLG AB, LLG BC, LLG CA, LLL ABC con resistencia = 25 Ω."""
    if 'Resistencia_Falla' not in df.columns or 'Tipo_Falla' not in df.columns:
        return df
    df = df.copy()
    resist = pd.to_numeric(df['Resistencia_Falla'].astype(str).str.replace(',','.'), errors='coerce')
    fault_types = ['LLG AB', 'LLG BC', 'LLG CA', 'LLL ABC']
    mask = df['Tipo_Falla'].isin(fault_types) & (resist == 25)
    before = len(df)
    df = df[~mask]
    print(f"[filter] Eliminadas {before - len(df)} filas con fallas {fault_types} y resistencia=25Ω")
    return df

# ----------------------------------------------------------------------
# SUBMUESTREO ESTRATIFICADO
# ----------------------------------------------------------------------
def stratified_subsample(df, target_fraction, target_col='Tipo_Falla', random_state=42):
    if target_fraction >= 1.0:
        return df
    df = df.copy()
    y = df[target_col].astype(str)
    _, df_sampled = train_test_split(df, test_size=target_fraction, stratify=y, random_state=random_state)
    print(f"[subsample] Reducido de {len(df)} a {len(df_sampled)} filas ({target_fraction*100:.0f}%)")
    return df_sampled

# ----------------------------------------------------------------------
# BALANCEO (ajustable por fault_multiplier)
# ----------------------------------------------------------------------
def balance_detection_data(df, fault_multiplier=1.0, random_state=42):
    """Balancea el dataset de detección a una proporción fault/no-fault = fault_multiplier."""
    np.random.seed(random_state)
    load_cols = [c for c in ["Carga_645", "Carga_671", "Carga_675", "Carga_611"] if c in df.columns]
    df_no_fault = df[df["Tipo_Falla"] == NO_FAULT_LABEL].copy()
    df_fault = df[df["Tipo_Falla"] != NO_FAULT_LABEL].copy()
    n_no_fault = len(df_no_fault)
    n_fault_target = int(n_no_fault * fault_multiplier)
    print(f"\n[balance] Original no-fault: {n_no_fault}, fault: {len(df_fault)}")
    print(f"[balance] Target fault rows: {n_fault_target} (multiplier {fault_multiplier})")
    if len(df_fault) > n_fault_target:
        fault_types = df_fault["Tipo_Falla"].unique()
        samples_per_type = n_fault_target // len(fault_types)
        remainder = n_fault_target % len(fault_types)
        sampled_faults = []
        for ft in fault_types:
            ft_df = df_fault[df_fault["Tipo_Falla"] == ft]
            n_take = samples_per_type + (1 if remainder > 0 else 0)
            remainder -= 1
            if len(ft_df) > n_take:
                sampled_faults.append(ft_df.sample(n=n_take, random_state=random_state))
            else:
                sampled_faults.append(ft_df)
        df_fault_balanced = pd.concat(sampled_faults, ignore_index=True)
    else:
        df_fault_balanced = df_fault
    det = pd.concat([df_no_fault, df_fault_balanced], ignore_index=True)
    det[DETECTION_LABEL_COL] = (det["Tipo_Falla"] != NO_FAULT_LABEL).astype(int)
    det = det.sample(frac=1, random_state=random_state).reset_index(drop=True)
    final_fault = det[DETECTION_LABEL_COL].sum()
    final_no_fault = len(det) - final_fault
    print(f"[balance] Final: no-fault={final_no_fault}, fault={final_fault} (ratio {final_fault/final_no_fault:.2f})")
    out_cols = FEATURE_COLS + [DETECTION_LABEL_COL] + load_cols
    return det[out_cols]

# ----------------------------------------------------------------------
# DATOS PARA CLASIFICACIÓN (solo fallas)
# ----------------------------------------------------------------------
def build_classif_data(df):
    fault_mask = df["Tipo_Falla"] != NO_FAULT_LABEL
    cls = df.loc[fault_mask, FEATURE_COLS].copy()
    cls[CLASSIF_LABEL_COL] = df.loc[fault_mask, "Tipo_Falla"].values
    print(f"[classif] Fault only: {len(cls)} rows, {cls[CLASSIF_LABEL_COL].nunique()} types")
    return cls

# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------
def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_raw_data(raw_csv_path: str):
    df = pd.read_csv(raw_csv_path, sep=";", decimal=",", low_memory=False)
    print(f"[load] Read {len(df):,} rows from '{raw_csv_path}'")
    return df

def save_csv(df, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep=";", decimal=",", index=False)
    print(f"[save] Saved {len(df):,} rows -> '{out_path}'")

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main(config_path, subsample_fraction=1.0, fault_multiplier=1.0):
    config = load_config(config_path)
    network = config["network"]["name"]
    raw_csv = config["data"]["raw_csv"]
    processed_dir = Path(config["data"]["processed_dir"])

    print(f"\n{'='*60}")
    print(f"  Preprocessing pipeline — {network.upper()}")
    print(f"  Subsample fraction: {subsample_fraction}  |  Fault multiplier: {fault_multiplier}")
    print(f"{'='*60}\n")

    # Cargar datos simulados
    df = load_raw_data(raw_csv)
    df = filter_100_ohm_resistance(df)
    df = filter_llg_lll_resistance_25(df)

    # Submuestreo estratificado (si se solicita)
    if subsample_fraction < 1.0:
        df = stratified_subsample(df, target_fraction=subsample_fraction, target_col='Tipo_Falla')

    # Balanceo a la proporción deseada
    det_df = balance_detection_data(df, fault_multiplier=fault_multiplier)

    # Datos para clasificación multiclase (solo fallas)
    cls_df = build_classif_data(df)

    # Guardar
    save_csv(det_df, processed_dir / "detection_data.csv")
    save_csv(cls_df, processed_dir / "classif_data.csv")

    print(f"\n[done] Preprocessing complete.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--subsample-fraction", type=float, default=1.0,
                        help="Fraction of total data to keep (stratified, default=1.0)")
    parser.add_argument("--fault-multiplier", type=float, default=1.0,
                        help="Ratio fault/no-fault in detection dataset (default=1.0 for 1:1)")
    args = parser.parse_args()

    if not (0 < args.subsample_fraction <= 1.0):
        print("[ERROR] subsample-fraction must be in (0,1]", file=sys.stderr)
        sys.exit(1)

    try:
        main(args.config, args.subsample_fraction, args.fault_multiplier)
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)