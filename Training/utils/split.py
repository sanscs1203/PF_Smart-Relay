"""
utils/split.py
--------------
Splits the processed detection and classification datasets into
train / val / test partitions (70 / 15 / 15) with stratification,
fits a StandardScaler on the training features, applies it to all
partitions, and saves everything to data/splits/<network>/.

Scaler fitting policy
---------------------
The scaler is fitted ONLY on the detection train set features, then
reused for the classification splits of the same network. This is
correct because both modules share the same 12 electrical features
and the same physical scale, so a single scaler avoids inconsistency.

The fitted scaler is saved as data/processed/<network>/scaler.pkl so
that evaluate.py and predict.py can load it without re-fitting.

Usage (called from project root):
    python utils/split.py --config ieee5/config.yaml
    python utils/split.py --config ieee13/config.yaml
"""

import argparse
import sys
import joblib
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

# Base features (shared by detection and classification)
BASE_FEATURE_COLS = [
    "Va", "Vb", "Vc",
    "Ia", "Ib", "Ic"
]

# Detection uses 12 features
DETECTION_FEATURE_COLS = BASE_FEATURE_COLS
CLASSIF_FEATURE_COLS   = BASE_FEATURE_COLS

DETECTION_LABEL_COL = "label_detection"
CLASSIF_LABEL_COL   = "label_classif"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ---------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------

def load_config(config_path: str) -> Dict:
    """Load a YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to config.yaml (e.g. 'ieee5/config.yaml').

    Returns
    -------
    Dict
        Parsed configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_processed_csv(file_path: str, label_col: str, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """Load a processed CSV and separate features from the label column.

    Parameters
    ----------
    file_path : str
        Path to the processed CSV file.
    label_col : str
        Name of the label column to separate as y.
    feature_cols : List[str]
        List of feature column names to extract.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X, y) where X contains only the specified feature columns and y is the label.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed file not found: {path}\n"
            "Run utils/preprocess.py first."
        )
    df = pd.read_csv(path, sep=";", decimal=",")
    X  = df[feature_cols]
    y  = df[label_col]
    print(f"[load]  '{path.name}'  →  {len(df):,} rows, {X.shape[1]} features")
    return X, y


def save_split_csv(X: pd.DataFrame, y: pd.Series,
                   label_col: str, out_path: Path) -> None:
    """Concatenate scaled features and label, then save to CSV.

    Parameters
    ----------
    X : pd.DataFrame
        Scaled feature matrix (columns = FEATURE_COLS).
    y : pd.Series
        Label series.
    label_col : str
        Column name for y in the output file.
    out_path : Path
        Destination file path. Parent directories are created if needed.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = X.copy()
    df_out[label_col] = y.values
    df_out.to_csv(out_path, sep=";", decimal=",", index=False)
    print(f"  → saved  {len(df_out):>7,} rows  '{out_path}'")

# ---------------------------------------------------------
# Core split logic
# ---------------------------------------------------------

def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """Split X, y into train / val / test with stratification (70/15/15).

    The split is performed in two steps:
      1. 70% train  |  30% temp
      2. 50% of temp → val  |  50% of temp → test
         (which gives 15% / 15% of the total)

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Label series.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """

    # Train vs temp (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=1 - TRAIN_RATIO,
        stratify=y,
        random_state=random_state,
    )

    # Val vs test (equal halves of the 30%)
    val_fraction_of_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1 - val_fraction_of_temp,
        stratify=y_temp,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def print_split_summary(
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    total: int,
    label_col: str,
) -> None:
    """Print partition sizes and class proportions to stdout.

    Parameters
    ----------
    y_train, y_val, y_test : pd.Series
        Label series for each partition.
    total : int
        Total number of samples in the original dataset.
    label_col : str
        Label column name (used to decide display format).
    """
    for name, y in [("Train", y_train), ("Val  ", y_val), ("Test ", y_test)]:
        pct = 100 * len(y) / total
        if label_col == DETECTION_LABEL_COL:
            # Binary: show fault proportion
            fault_pct = 100 * y.mean()
            print(f"  {name}: {len(y):>7,}  ({pct:4.1f}%)  |  "
                  f"fault={fault_pct:.1f}%  no-fault={100-fault_pct:.1f}%")
        else:
            # Multiclass: just show counts
            print(f"  {name}: {len(y):>7,}  ({pct:4.1f}%)")

# ---------------------------------------------------------
# Scaler
# ---------------------------------------------------------

def fit_and_save_scaler(X_train: pd.DataFrame, scaler_path: Path) -> StandardScaler:
    """Fit a StandardScaler on training features and persist it to disk.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix (unscaled).
    scaler_path : Path
        Destination path for the .pkl file.

    Returns
    -------
    StandardScaler
        The fitted scaler instance.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"[scaler] fitted on train set and saved to '{scaler_path}'")
    return scaler

def apply_scaler(
    scaler: StandardScaler,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply a fitted scaler to all three partitions.

    Parameters
    ----------
    scaler : StandardScaler
        Already-fitted scaler.
    X_train, X_val, X_test : pd.DataFrame
        Unscaled feature matrices.
    feature_cols : List[str]
        Column names for the output DataFrames.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Scaled versions of (X_train, X_val, X_test) as DataFrames,
        preserving original column names.
    """
    def _scale(X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            scaler.transform(X),
            columns=feature_cols,
            index=X.index,
        )

    return _scale(X_train), _scale(X_val), _scale(X_test)

# ---------------------------------------------------------
# Pipeline per module
# ---------------------------------------------------------

def process_module(
    processed_csv: str,
    label_col: str,
    feature_cols: List[str],
    splits_dir: Path,
    prefix: str,
    scaler: Optional[StandardScaler],
    random_state: int,
    scaler_path: Optional[Path] = None,
) -> StandardScaler:
    """Run the full split-and-scale pipeline for one module.

    Parameters
    ----------
    processed_csv : str
        Path to the processed CSV (detection_data.csv or classif_data.csv).
    label_col : str
        Label column name in the processed CSV.
    feature_cols : List[str]
        List of feature column names for this module.
    splits_dir : Path
        Output directory for the six split files.
    prefix : str
        Filename prefix: 'det' for detection, 'cls' for classification.
    scaler : StandardScaler or None
        Pre-fitted scaler. If None, a new one is fitted on this module's
        training set and saved to scaler_path.
    random_state : int
        Random seed.
    scaler_path : Path or None
        Where to save the scaler (only used when scaler is None).

    Returns
    -------
    StandardScaler
        The fitted scaler (either the one passed in or the newly fitted one).
    """
    print(f"\n--- {prefix.upper()} module  ({label_col}) ---")

    # Load
    X, y = load_processed_csv(processed_csv, label_col, feature_cols)
    total = len(y)

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
        X, y, random_state
    )
    print_split_summary(y_train, y_val, y_test, total, label_col)

    # Scaler: fit if not provided
    if scaler is None:
        scaler = fit_and_save_scaler(X_train, scaler_path)
    else:
        print(f"[scaler] reusing existing scaler for {prefix} module")

    # Scale
    X_train_scaled, X_val_scaled, X_test_scaled = apply_scaler(
        scaler, X_train, X_val, X_test, feature_cols  # <-- UPDATED
    )

    # Save splits
    print("[save]  Writing split files...")
    save_split_csv(X_train_scaled, y_train, label_col, splits_dir / f"{prefix}_train.csv")
    save_split_csv(X_val_scaled,   y_val,   label_col, splits_dir / f"{prefix}_val.csv")
    save_split_csv(X_test_scaled,  y_test,  label_col, splits_dir / f"{prefix}_test.csv")

    return scaler

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(config_path: str) -> None:
    """Full split pipeline for one network (ieee5 or ieee13).

    Steps:
      1. Load config.yaml
      2. Split + scale detection data (12 features) → fit det_scaler
      3. Split + scale classification data (14 features) → fit cls_scaler
      4. Save all 6 split CSVs to data/splits/<network>/

    Note: Detection and classification now use separate scalers because
    classification includes 2 additional features (Ir, phi_Ir).

    Parameters
    ----------
    config_path : str
        Path to config.yaml for the target network.
    """
    config       = load_config(config_path)
    network      = config["network"]["name"]
    random_state = config["split"]["random_state"]
    processed_dir = Path(config["data"]["processed_dir"])
    splits_dir    = Path(config["data"]["splits_dir"])

    print(f"{'='*60}")
    print(f"  Split pipeline — {network.upper()}")
    print(f"  Ratios: {int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/{int(TEST_RATIO*100)}"
          f"  |  seed={random_state}")
    print(f"{'='*60}")

    # Detection module — fits its own scaler (12 features)
    det_scaler_path = processed_dir / "det_scaler.pkl"
    process_module(
        processed_csv = str(processed_dir / "detection_data.csv"),
        label_col     = DETECTION_LABEL_COL,
        feature_cols  = DETECTION_FEATURE_COLS,   # <-- 12 features
        splits_dir    = splits_dir,
        prefix        = "det",
        scaler        = None,
        random_state  = random_state,
        scaler_path   = det_scaler_path,
    )

    # Classification module — fits its own scaler (14 features)
    cls_scaler_path = processed_dir / "cls_scaler.pkl"
    process_module(
        processed_csv = str(processed_dir / "classif_data.csv"),
        label_col     = CLASSIF_LABEL_COL,
        feature_cols  = CLASSIF_FEATURE_COLS,     # <-- 14 features
        splits_dir    = splits_dir,
        prefix        = "cls",
        scaler        = None,
        random_state  = random_state,
        scaler_path   = cls_scaler_path,
    )

    print(f"\n[done]  Split pipeline complete for {network.upper()}.")
    print(f"        Splits saved to: '{splits_dir}'")
    print(f"        Detection scaler: '{det_scaler_path}'")
    print(f"        Classification scaler: '{cls_scaler_path}'\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split processed datasets into train/val/test and fit scaler."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config.yaml (e.g. 'ieee5/config.yaml')",
    )
    args = parser.parse_args()

    try:
        main(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)