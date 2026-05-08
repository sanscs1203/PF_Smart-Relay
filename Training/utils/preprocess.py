"""
utils/preprocess.py
-------------------
Loads a raw dataset CSV, validates its structure, and produces two
processed output files:
 
  - detection_data.csv  : features (X) + binary label (fault / no-fault)
  - classif_data.csv    : features (X) + multiclass label (fault type only,
                          Sin_Falla rows excluded)
 
This script does NOT fit or apply any scaler. Scaling is handled
downstream in utils/split.py after the train/val/test split is created,
to prevent data leakage.
 
Usage (called from project root):
    python3 utils/preprocess.py --config ieee5/config.yaml
    python3 utils/preprocess.py --config ieee13/config.yaml
"""
 
import argparse
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------

# Exact features column names
FEATURE_COLS = [
    "Va", "Vb", "Vc",
    "phi_Va", "phi_Vb", "phi_Vc",
    "Ia", "Ib", "Ic",
    "phi_Ia", "phi_Ib", "phi_Ic",
]

# Label used in the raw dataset to indicate no-fault condition
NO_FAULT_LABEL = "Sin_Falla"

# Binary label for detection task
DETECTION_LABEL_COL = "label_detection"

# Multiclass label column name written to classif_data.csv
CLASSIF_LABEL_COL = "label_classif"

# Additional residual current columns for classification dataset
RESIDUAL_CURRENT_COLS = ["Ir", "phi_Ir"]

# Extended feature set for classification (12 original + 2 residual)
CLASSIF_FEATURE_COLS = FEATURE_COLS + RESIDUAL_CURRENT_COLS

# ---------------------------------------------------------------
# I/O Functions
# ---------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load a YAML configuration file.
 
    Parameters
    ----------
    config_path : str
        Path to the config.yaml file (e.g. 'ieee5/config.yaml').
 
    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():    
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
    
def load_raw_data(raw_csv_path: str) -> pd.DataFrame:
    """Read the raw dataset CSV with the project-standard format.
 
    The CSV uses semicolon (;) as separator and comma (,) as decimal.
 
    Parameters
    ----------
    raw_csv_path : str
        Path to the raw CSV file.
 
    Returns
    -------
    pd.DataFrame
        Raw dataframe with all original columns.
    """
    path = Path(raw_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset CSV not found: {path}")
    
    df = pd.read_csv(path, sep=';', decimal=',')
    print(f"[load] Read {len(df):,} rows from '{path}'")
    return df

# ---------------------------------------------------------------
# Validation Functions
# ---------------------------------------------------------------

def validate_dataframe(df: pd.DataFrame, config: dict) -> None:
    """Run basic sanity checks on the raw dataframe.
 
    Checks performed:
      1. All expected feature columns are present.
      2. The fault-type column ('Tipo_Falla') is present.
      3. No NaN values exist in feature or label columns.
      4. All Tipo_Falla values are among the known set.
      5. Class distribution is printed for manual inspection.
 
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe loaded from CSV.
    config : dict
        Parsed config.yaml; used to read the list of valid fault types.
 
    Raises
    ------
    ValueError
        If any critical check fails.
    """
    print("\n[validate] Running dataset validation...")
    
    # Feature columns check
    missing_features = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    print(f" All {len(FEATURE_COLS)} feature columns are present.")
    
    # Label column check
    if 'Tipo_Falla' not in df.columns:
        raise ValueError("Missing label column: 'Tipo_Falla'")
    print(" Label column 'Tipo_Falla' is present.")
    
    # NaN check
    cols_to_check = FEATURE_COLS + ['Tipo_Falla']
    nan_counts = df[cols_to_check].isna().sum()
    cols_with_nans = nan_counts[nan_counts > 0]
    if not cols_with_nans.empty:
        raise ValueError(f"NaN values found in columns:\n{cols_with_nans}")
    print(" No NaN values found in features or labels.")
    
    # Valid fault types check
    valid_types = set(config["data"]["fault_types"] + [NO_FAULT_LABEL])
    unknown_types = set(df["Tipo_Falla"].unique()) - valid_types
    if unknown_types:
        raise ValueError(f"Unknown fault types found in 'Tipo_Falla': {unknown_types}")
    print(f" All fault types in 'Tipo_Falla' are valid: {valid_types}")
    
    # Class distribution
    print("\n Class distribution in 'Tipo_Falla':")
    dist = df["Tipo_Falla"].value_counts()
    for label, count in dist.items():
        pct = 100 * count / len(df)
        print(f"  {label:<15} {count:>7,}  ({pct:5.1f}%)")
    print()
    
# ---------------------------------------------------------------
# Processing Functions
# ---------------------------------------------------------------

def compute_residual_current(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the residual (zero-sequence) current from phase currents.
    
    The residual current is the phasor sum of the three phase currents:
        Ir = Ia + Ib + Ic
    
    This function converts polar coordinates (magnitude, phase) to 
    rectangular, performs the sum, and converts back to polar.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing Ia, Ib, Ic (magnitudes) and 
        phi_Ia, phi_Ib, phi_Ic (phases in degrees).
    
    Returns
    -------
    pd.DataFrame
        Original dataframe with two new columns: 'Ir' and 'phi_Ir'.
    """
    df = df.copy()
    
    # Convert phase angles from degrees to radians
    phi_a_rad = np.deg2rad(df["phi_Ia"])
    phi_b_rad = np.deg2rad(df["phi_Ib"])
    phi_c_rad = np.deg2rad(df["phi_Ic"])
    
    # Convert polar to rectangular (real + imaginary)
    Ia_real = df["Ia"] * np.cos(phi_a_rad)
    Ia_imag = df["Ia"] * np.sin(phi_a_rad)
    
    Ib_real = df["Ib"] * np.cos(phi_b_rad)
    Ib_imag = df["Ib"] * np.sin(phi_b_rad)
    
    Ic_real = df["Ic"] * np.cos(phi_c_rad)
    Ic_imag = df["Ic"] * np.sin(phi_c_rad)
    
    # Phasor sum: Ir = Ia + Ib + Ic
    Ir_real = Ia_real + Ib_real + Ic_real
    Ir_imag = Ia_imag + Ib_imag + Ic_imag
    
    # Convert back to polar
    df["Ir"] = np.sqrt(Ir_real**2 + Ir_imag**2)
    df["phi_Ir"] = np.rad2deg(np.arctan2(Ir_imag, Ir_real))
    
    return df

def build_detection_data(df: pd.DataFrame) -> pd.DataFrame:
    """Create the binary-labeled dataset for the detection module.
 
    Label encoding:
      Sin_Falla → 0  (no fault)
      any other → 1  (fault)
 
    Parameters
    ----------
    df : pd.DataFrame
        Validated raw dataframe.
 
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [FEATURE_COLS..., DETECTION_LABEL_COL]
    """
    det = df[FEATURE_COLS].copy()
    det[DETECTION_LABEL_COL] = (df["Tipo_Falla"] != NO_FAULT_LABEL).astype(int)
    
    n_fault = det[DETECTION_LABEL_COL].sum()
    n_no_fault = len(det) - n_fault
    print(f"[detection]  Total rows : {len(det):,}")
    print(f"             No-fault (0): {n_no_fault:,}")
    print(f"             Fault    (1): {n_fault:,}")
    return det

def build_classif_data(df: pd.DataFrame) -> pd.DataFrame:
    """Create the multiclass-labeled dataset for the classification module.
 
    Only fault rows are included (Sin_Falla rows are excluded).
    The fault-type string is preserved as the label.
    
    This dataset includes the 12 original features plus the computed
    residual current (Ir magnitude and phi_Ir phase).
 
    Parameters
    ----------
    df : pd.DataFrame
        Validated raw dataframe.
 
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [CLASSIF_FEATURE_COLS..., CLASSIF_LABEL_COL]
        Only rows where Tipo_Falla != 'Sin_Falla'.
    """
    fault_mask = df["Tipo_Falla"] != NO_FAULT_LABEL
    cls = df.loc[fault_mask, FEATURE_COLS].copy()
    
    # Compute and add residual current columns
    cls = compute_residual_current(cls)
    
    # Add classification label
    cls[CLASSIF_LABEL_COL] = df.loc[fault_mask, "Tipo_Falla"].values
    
    print(f"[classif]    Total rows (fault only): {len(cls):,}")
    print(f"             Unique fault types      : {cls[CLASSIF_LABEL_COL].nunique()}")
    print(f"             Features                : {len(CLASSIF_FEATURE_COLS)} (12 original + 2 residual)")
    return cls

# ---------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------

def save_csv(df: pd.DataFrame, out_path: str) -> None:
    """Save a processed dataframe to CSV using the project standard format.
 
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save.
    out_path : Path
        Destination file path. Parent directories are created if needed.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use semicolon separator and comma decimal to stay consistent with raw
    df.to_csv(out_path, sep=";", decimal=",", index=False)
    print(f"[save]  Saved {len(df):,} rows → '{out_path}'")

# ---------------------------------------------------------------
# Main function
# ---------------------------------------------------------------

def main(config_path: str) -> None:
    """Full preprocessing pipeline for one network (ieee5 or ieee13).
 
    Steps:
      1. Load config.yaml
      2. Load raw CSV
      3. Validate structure and content
      4. Build detection_data.csv  (binary labels)
      5. Build classif_data.csv    (multiclass labels, fault rows only)
      6. Save both to data/processed/<network>/
 
    Parameters
    ----------
    config_path : str
        Path to the config.yaml for the target network.
    """
    
    # Load config
    config = load_config(config_path)
    network = config["network"]["name"]
    raw_csv = config["data"]["raw_csv"]
    processed_dir = Path(config["data"]["processed_dir"])
    
    print(f"{'='*60}")
    print(f"  Preprocessing pipeline — {network.upper()}")
    print(f"{'='*60}\n")
    
    # Load
    df = load_raw_data(raw_csv)
    
    # Validate
    validate_dataframe(df, config)
    
    # Build processed datasets
    det_df = build_detection_data(df)
    cls_df = build_classif_data(df)
    
    # Save outputs
    save_csv(det_df, processed_dir / "detection_data.csv")
    save_csv(cls_df, processed_dir / "classif_data.csv")
 
    print(f"\n[done]  Preprocessing complete for {network.upper()}.")
    print(f"        Output directory: '{processed_dir}'\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess raw fault dataset into detection and classification CSVs."
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