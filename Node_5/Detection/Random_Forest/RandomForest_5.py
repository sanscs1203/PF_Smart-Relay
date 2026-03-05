# =============================================================================
# RANDOM FOREST - FAULT DETECTION MODEL
# Smart Relay - IEEE 5-Bus Test System
#
# Score based on AHP weights (Saaty, 1980) - CR = 0.0032
# Metrics derived from IEEE C37.100 / IEC 60255:
#   - Recall (0.6483)      → Dependability
#   - Specificity (0.1220) → Security  
#   - ROC-AUC (0.2297)     → Global discrimination capability
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import json
import joblib
from sklearn.metrics import (
    recall_score,
    precision_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    make_scorer
)
from matplotlib.patches import Patch

# =============================================================================
# AHP WEIGHTS (computed previously with CR = 0.0032 ✅)
# =============================================================================
W_RECALL      = 0.6483   # Dependability  (IEEE C37.100)
W_SPECIFICITY = 0.1220   # Security       (IEEE C37.100)
W_AUC         = 0.2297   # Global discrimination capability

# Verify weights sum to 1
assert abs(W_RECALL + W_SPECIFICITY + W_AUC - 1.0) < 0.001, "Weights must sum to 1.0"

# =============================================================================
# PHASE 1: DATA LOADING AND EXPLORATION
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 1: DATA LOADING AND EXPLORATION")
print("=" * 70)

# Load dataset
file = '~/CODES/PF_Smart-Relay/Node_5/DataSet_Nodes_5.csv'
df = pd.read_csv(file, sep=';', decimal=',')

# Fix: convert RF and Factor_Carga to float (handle comma decimals)
df['RF'] = pd.to_numeric(df['RF'], errors='coerce')
df['Factor_Carga'] = pd.to_numeric(df['Factor_Carga'], errors='coerce')
df['Nodo_Falla'] = df['Nodo_Falla'].apply(lambda x: np.nan if x == 'Sin_Falla' else x)

# Define column roles
features = [
    'Va', 'Vb', 'Vc',
    'phi_Va', 'phi_Vb', 'phi_Vc',
    'Ia', 'Ib', 'Ic',
    'phi_Ia', 'phi_Ib', 'phi_Ic'
]

metadata = [
    'RF', 'Factor_Carga', 'Nodo_Falla'
]

label = 'Tipo_Falla'

# --- General dataset information ---
print("=" * 60)
print("GENERAL DATASET INFORMATION")
print("=" * 60)
print(f"Dimensions: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nColumns found:\n{df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")

# --- Verify all expected columns exist ---
expected_columns = features + metadata + [label]
missing_columns = [col for col in expected_columns if col not in df.columns]

if missing_columns:
    print(f"\n⚠️ MISSING COLUMNS: {missing_columns}")
else:
    print("\n✅ All expected columns are present")

# --- Descriptive statistics for features ---
print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS (FEATURES)")
print("=" * 60)
print(df[features].describe().round(2))

# --- Target variable distribution ---
print("\n" + "=" * 60)
print("CLASS DISTRIBUTION (Tipo_Falla)")
print("=" * 60)
conteo = df[label].value_counts()
print(conteo)
print(f"\nTotal classes: {df[label].nunique()}")

# --- Metadata distribution ---
print("\n" + "=" * 60)
print("METADATA DISTRIBUTION")
print("=" * 60)
print(f"\nUnique RF values: {sorted(df['RF'].dropna().unique())}")
print(f"Unique Factor_Carga values: {sorted(df['Factor_Carga'].dropna().unique())}")
print(f"Unique Nodo_Falla values: {sorted(df['Nodo_Falla'].dropna().unique())}")

# =============================================================================
# PHASE 2: PREPROCESSING
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 2: PREPROCESSING")
print("=" * 70)

# --- Create binary label for detection model ---
# 0 = No Fault (normal operation), 1 = Fault (any type)
df['Label_detection'] = df['Tipo_Falla'].apply(
    lambda x: 0 if x == 'Sin_Falla' else 1
)

print("=" * 60)
print("DISTRIBUTION - DETECTION MODEL")
print("=" * 60)
print(df['Label_detection'].value_counts())
print(f"\nClass 1 proportion (Fault):    {df['Label_detection'].mean():.2%}")
print(f"Class 0 proportion (No Fault): {1 - df['Label_detection'].mean():.2%}")

# --- Define features (X) and target (y) ---
X = df[features]
y = df['Label_detection']

print("\n" + "=" * 60)
print("DIMENSIONS")
print("=" * 60)
print(f"Features (X): {X.shape}")
print(f"Label (y):    {y.shape}")

# --- Check class balance ---
ratio = df['Label_detection'].value_counts()
ratio_min_max = ratio.min() / ratio.max()

print("\n" + "=" * 60)
print("CLASS BALANCE CHECK")
print("=" * 60)
print(f"Minority/Majority ratio: {ratio_min_max:.2%}")

if ratio_min_max >= 0.8:
    print("✅ Classes are balanced")
elif ratio_min_max >= 0.5:
    print("⚠️ Moderate imbalance - Consider class_weight='balanced'")
else:
    print("🚨 Severe imbalance - Using class_weight='balanced'")

# --- Train/Test split (80/20) with stratification ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=242,
    stratify=y
)

# Verify split dimensions
print("\n" + "=" * 60)
print("DIMENSIONS AFTER SPLIT")
print("=" * 60)
print(f"Train: {X_train.shape[0]} samples ({X_train.shape[0]/X.shape[0]:.0%})")
print(f"Test:  {X_test.shape[0]} samples ({X_test.shape[0]/X.shape[0]:.0%})")

# Verify class proportions are maintained
print("\n" + "=" * 60)
print("CLASS PROPORTIONS AFTER SPLIT")
print("=" * 60)
print(f"Original dataset: Fault={y.mean():.2%}  |  No Fault={1-y.mean():.2%}")
print(f"Train:            Fault={y_train.mean():.2%}  |  No Fault={1-y_train.mean():.2%}")
print(f"Test:             Fault={y_test.mean():.2%}  |  No Fault={1-y_test.mean():.2%}")

# =============================================================================
# PHASE 3: HYPERPARAMETER OPTIMIZATION (GridSearchCV)
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 3: HYPERPARAMETER OPTIMIZATION (GridSearchCV)")
print("=" * 70)

def ahp_score_func(y_true, y_proba):
    """
    AHP-weighted composite score for model selection.
    Based on IEEE C37.100 / IEC 60255 protection attributes.
    
    Weights (AHP, CR = 0.0032):
        Recall:      0.6483 (Dependability)
        Specificity: 0.1220 (Security)
        ROC-AUC:     0.2297 (Discrimination)
    """
    y_pred = (y_proba >= 0.5).astype(int)

    # Recall → Dependability (IEEE C37.100)
    rec = recall_score(y_true, y_pred)

    # ROC-AUC → Global discrimination capability
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = 0.0

    # Specificity → Security (IEEE C37.100)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Composite score
    score = (
        W_RECALL      * rec  +
        W_SPECIFICITY * spec +
        W_AUC         * auc
    )

    return score

custom_scorer = make_scorer(
    ahp_score_func,
    response_method='predict_proba'
)

# --- Define hyperparameter search space ---
param_grid = {
    'n_estimators':      [100, 200, 300, 500],
    'max_depth':         [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'max_features':      ['sqrt', 'log2']
}

print("=" * 60)
print("GRIDSEARCHCV CONFIGURATION")
print("=" * 60)
total_combinations = 4 * 5 * 3 * 3 * 2
print(f"Total combinations: {total_combinations}")
print(f"Total fits (x5 folds): {total_combinations * 5}")
print(f"\nScoring: AHP composite score (CR = 0.0032)")
print(f"  Recall weight:      {W_RECALL:.4f}")
print(f"  Specificity weight: {W_SPECIFICITY:.4f}")
print(f"  ROC-AUC weight:     {W_AUC:.4f}")

# --- Configure and execute GridSearchCV ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=242)

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(
        class_weight='balanced',
        random_state=242,
        n_jobs=1
    ),
    param_grid=param_grid,
    scoring=custom_scorer,
    cv=cv,
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

print("\nStarting hyperparameter search...\n")
grid_search.fit(X_train, y_train)

print(f"\n✅ Search completed")
print(f"Best AHP Score (CV): {grid_search.best_score_:.4f}")

# =============================================================================
# TOP 5 MODELS - EVALUATION ON TEST SET
# =============================================================================

print("\n" + "=" * 60)
print("TOP 5 MODELS - TEST SET EVALUATION")
print("=" * 60)

results = pd.DataFrame(grid_search.cv_results_)
top5 = results.nsmallest(5, 'rank_test_score')

top5_detailed = []

for i, (_, row) in enumerate(top5.iterrows()):

    # Recreate model with the hyperparameters of this combination
    params = row['params']
    model_temp = RandomForestClassifier(
        **params,
        class_weight='balanced',
        random_state=242,
        n_jobs=-1
    )
    model_temp.fit(X_train, y_train)

    # --- Predictions on test set ---
    y_pred_temp = model_temp.predict(X_test)
    y_proba_temp = model_temp.predict_proba(X_test)[:, 1]

    # --- Compute individual metrics ---
    rec = recall_score(y_test, y_pred_temp)

    try:
        auc = roc_auc_score(y_test, y_proba_temp)
    except ValueError:
        auc = 0.0

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_temp).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # --- AHP composite score ---
    score_final = (
        W_RECALL      * rec  +
        W_SPECIFICITY * spec +
        W_AUC         * auc
    )

    # --- Compute model complexity (for tie-breaking) ---
    n_estimators = params.get('n_estimators', 100)
    max_depth = params.get('max_depth', 999)
    if max_depth is None:
        max_depth = 999  # None means unlimited depth → highest complexity
    complexity = n_estimators * max_depth

    # Store detailed results
    top5_detailed.append({
        'rank': i + 1,
        'params': params,
        'model': model_temp,
        'recall': rec,
        'specificity': spec,
        'auc': auc,
        'ahp_score': score_final,
        'complexity': complexity
    })

    print(f"\n  Model #{i+1}")
    print(f"  {'Metric':<20} {'Value':>10} {'Weight':>10} {'Contribution':>14}")
    print(f"  {'-'*56}")
    print(f"  {'Recall':<20} {rec:>10.4f} {W_RECALL:>10.4f} {W_RECALL*rec:>14.4f}")
    print(f"  {'Specificity':<20} {spec:>10.4f} {W_SPECIFICITY:>10.4f} {W_SPECIFICITY*spec:>14.4f}")
    print(f"  {'ROC-AUC':<20} {auc:>10.4f} {W_AUC:>10.4f} {W_AUC*auc:>14.4f}")
    print(f"  {'-'*56}")
    print(f"  {'AHP SCORE':<20} {score_final:>10.4f}")
    print(f"  Complexity: {complexity}")
    print(f"  Params: {params}")

# =============================================================================
# DETERMINISTIC MODEL SELECTION
# =============================================================================

df_top5 = pd.DataFrame(top5_detailed)

# Round AHP scores to 4 decimals to group truly equivalent models
df_top5['ahp_rounded'] = df_top5['ahp_score'].round(4)
best_ahp_rounded = df_top5['ahp_rounded'].max()

# Among tied models, select the SIMPLEST (lowest complexity) - Occam's razor
tied_models = df_top5[df_top5['ahp_rounded'] == best_ahp_rounded]
best_idx = tied_models['complexity'].idxmin()
best_entry = df_top5.loc[best_idx]

best_model_final = best_entry['model']
best_params_final = best_entry['params']
best_score_final = best_entry['ahp_score']

print("\n" + "=" * 60)
print("BEST MODEL (DETERMINISTIC SELECTION)")
print("=" * 60)
print(f"\n  Selection criteria:")
print(f"  1. Highest AHP score (rounded to 4 decimals)")
print(f"  2. Among ties: lowest complexity (Occam's razor)")
print(f"\n  AHP Score: {best_score_final:.4f}")
print(f"  Complexity: {best_entry['complexity']}")
print(f"  Tied models: {len(tied_models)}")
print(f"\n  Hyperparameters:")
for param, value in best_params_final.items():
    print(f"    {param}: {value}")

rf_optimizado = best_model_final

# =============================================================================
# PHASE 4: EVALUATION OF OPTIMIZED MODEL (Default Threshold)
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 4: EVALUATION OF OPTIMIZED MODEL (Default Threshold)")
print("=" * 70)

y_pred = rf_optimizado.predict(X_test)
y_proba = rf_optimizado.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print("OPTIMIZED RANDOM FOREST - TEST SET RESULTS")
print("=" * 60)

# --- Quick overview of predictions ---
print(f"\nPrediction Overview:")
print(f"  Predicted Fault:     {(y_pred == 1).sum()}")
print(f"  Predicted No Fault:  {(y_pred == 0).sum()}")
print(f"  Actual Fault:        {(y_test == 1).sum()}")
print(f"  Actual No Fault:     {(y_test == 0).sum()}")

# --- Individual metrics ---
recall    = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_proba)
accuracy  = accuracy_score(y_test, y_pred)

# Specificity (True Negative Rate)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

# F1 and F2 scores
f1_score_val = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
f2_score_val = 5 * (precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0

print("\n" + "=" * 60)
print("METRICS (threshold = 0.50)")
print("=" * 60)

# AHP score metrics
print("\n🔴 AHP SCORE METRICS:")
print(f"  Recall (Dependability):        {recall:.4f}")
print(f"  Specificity (Security):        {specificity:.4f}")
print(f"  ROC-AUC (Discrimination):      {roc_auc:.4f}")

# Complementary metrics
print("\n🟡 COMPLEMENTARY METRICS (reported only, not in score):")
print(f"  Precision:                     {precision:.4f}")
print(f"  Accuracy:                      {accuracy:.4f}")
print(f"  F1-Score:                      {f1_score_val:.4f}")
print(f"  F2-Score:                      {f2_score_val:.4f}")

# Confusion matrix
print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)
print(f"""
                     Predicted
                  No Fault    Fault
Actual No Fault      {tn}          {fp}
Actual Fault         {fn}          {tp}
""")
print(f"  True Negatives  (TN): {tn} → No Fault correctly identified")
print(f"  False Positives (FP): {fp} → Unnecessary trip (tolerable)")
print(f"  False Negatives (FN): {fn} → ⚠️ UNDETECTED FAULT (catastrophic)")
print(f"  True Positives  (TP): {tp} → Fault correctly detected")

# AHP Composite Score
score_ahp = (
    W_RECALL      * recall      +
    W_SPECIFICITY * specificity +
    W_AUC         * roc_auc
)

print("\n" + "=" * 60)
print("AHP COMPOSITE SCORE (IEEE C37.100 / IEC 60255)")
print("=" * 60)
print(f"\n  {'Metric':<25} {'Value':>10} {'Weight':>10} {'Contribution':>14}")
print(f"  {'-'*61}")
print(f"  {'Recall (Dependab.)':<25} {recall:>10.4f} {W_RECALL:>10.4f} {W_RECALL*recall:>14.4f}")
print(f"  {'Specificity (Security)':<25} {specificity:>10.4f} {W_SPECIFICITY:>10.4f} {W_SPECIFICITY*specificity:>14.4f}")
print(f"  {'ROC-AUC (Discrim.)':<25} {roc_auc:>10.4f} {W_AUC:>10.4f} {W_AUC*roc_auc:>14.4f}")
print(f"  {'-'*61}")
print(f"  {'AHP TOTAL SCORE':<25} {score_ahp:>10.4f}")
print(f"\n  Weighting method: AHP (Saaty, 1980)")
print(f"  Consistency Ratio: CR = 0.0032 ✅ (< 0.10)")

# =============================================================================
# FEATURE IMPORTANCE ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

importances = rf_optimizado.feature_importances_
feature_names = X.columns.tolist()

df_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\n  Feature Ranking:")
print(f"  {'#':<4} {'Feature':<12} {'Importance':>12} {'Cumulative':>12}")
print(f"  {'-'*42}")

cumulative = 0
for i, (_, row) in enumerate(df_importance.iterrows()):
    cumulative += row['Importance']
    print(f"  {i+1:<4} {row['Feature']:<12} {row['Importance']:>12.4f} {cumulative:>12.4f}")

# Group by type
imp_mag_voltage = df_importance[df_importance['Feature'].isin(['Va', 'Vb', 'Vc'])]['Importance'].sum()
imp_phi_voltage = df_importance[df_importance['Feature'].isin(['phi_Va', 'phi_Vb', 'phi_Vc'])]['Importance'].sum()
imp_mag_currents = df_importance[df_importance['Feature'].isin(['Ia', 'Ib', 'Ic'])]['Importance'].sum()
imp_phi_currents = df_importance[df_importance['Feature'].isin(['phi_Ia', 'phi_Ib', 'phi_Ic'])]['Importance'].sum()

print(f"\n  Contribution by group:")
print(f"  Voltages Magnitude (Va, Vb, Vc):     {imp_mag_voltage:.4f} ({imp_mag_voltage*100:.1f}%)")
print(f"  Voltages Angle (θVa, θVb, θVc):     {imp_phi_voltage:.4f} ({imp_phi_voltage*100:.1f}%)")
print(f"  Currents Magnitude (Ia, Ib, Ic):     {imp_mag_currents:.4f} ({imp_mag_currents*100:.1f}%)")
print(f"  Currents Angle (θIa, θIb, θIc):     {imp_phi_currents:.4f} ({imp_phi_currents*100:.1f}%)")

# =============================================================================
# PHASE 5: DECISION THRESHOLD OPTIMIZATION
# =============================================================================

print("\n" + "=" * 60)
print("PHASE 5: THRESHOLD OPTIMIZATION")
print("=" * 60)

# --- Perform threshold sweep ---
thresholds = np.arange(0.05, 0.96, 0.01)
sweep_results = []

for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, y_pred_t).ravel()
    
    rec_t  = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
    spec_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0.0
    prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0
    
    try:
        auc_t = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc_t = 0.0
    
    ahp_score_t = W_RECALL * rec_t + W_SPECIFICITY * spec_t + W_AUC * auc_t
    
    sweep_results.append({
        'Threshold':   t,
        'Recall':      rec_t,
        'Specificity': spec_t,
        'Precision':   prec_t,
        'ROC_AUC':     auc_t,
        'AHP_Score':   ahp_score_t,
        'TP': tp_t, 'FP': fp_t, 'FN': fn_t, 'TN': tn_t
    })

df_sweep = pd.DataFrame(sweep_results)

# --- Evaluate default threshold ---
default_row = df_sweep[df_sweep['Threshold'].round(2) == 0.50].iloc[0]
recall_default = default_row['Recall']
fn_default = int(default_row['FN'])
fp_default = int(default_row['FP'])

print(f"\n  Default threshold (0.50) performance:")
print(f"    Recall: {recall_default:.4f}")
print(f"    FN: {fn_default}, FP: {fp_default}")

# --- Find optimal threshold ---
# Priority 1: Recall = 1.0 (mandatory for protection)
# Priority 2: Maximum AHP Score (balances all metrics)
# Priority 3: Minimum FP (security)
# Priority 4: Closest to 0.50 (best generalization)

perfect_recall = df_sweep[df_sweep['Recall'] == 1.0]

if len(perfect_recall) > 0:
    print(f"\n  ✅ Found {len(perfect_recall)} thresholds with Recall = 1.0")
    print(f"     Range: [{perfect_recall['Threshold'].min():.2f}, {perfect_recall['Threshold'].max():.2f}]")
    
    # Step 1: Filter by maximum AHP Score
    max_ahp = perfect_recall['AHP_Score'].max()
    candidates = perfect_recall[perfect_recall['AHP_Score'] == max_ahp].copy()
    print(f"\n  Step 1 - Max AHP ({max_ahp:.4f}): {len(candidates)} candidates")
    
    # Step 2: Filter by minimum FP
    min_fp = candidates['FP'].min()
    candidates = candidates[candidates['FP'] == min_fp].copy()
    print(f"  Step 2 - Min FP ({int(min_fp)}): {len(candidates)} candidates")
    
    # Step 3: Pick threshold CLOSEST TO 0.50 (best generalization margin)
    candidates['distance_to_default'] = (candidates['Threshold'] - 0.50).abs()
    best_row = candidates.loc[candidates['distance_to_default'].idxmin()]
    optimal_threshold = best_row['Threshold']
    print(f"  Step 3 - Closest to 0.50: {optimal_threshold:.2f}")
    
    # Show the valid range
    valid_range = candidates['Threshold'].values
    print(f"\n  Valid threshold range: {valid_range.min():.2f} - {valid_range.max():.2f}")
    print(f"  Selected: {optimal_threshold:.2f} (closest to natural boundary 0.50)")
    
    # Report if default was selected
    if optimal_threshold == 0.50:
        print(f"\n  📌 Default threshold (0.50) IS optimal!")
        print(f"     - Achieves Recall = 1.0 ✅")
        print(f"     - Has maximum AHP Score ✅")
        print(f"     - Has minimum FP ✅")
    elif abs(optimal_threshold - 0.50) < 0.1:
        print(f"\n  📌 Optimal threshold ({optimal_threshold:.2f}) is close to default")
        print(f"     - Minor adjustment for improved performance")
    else:
        print(f"\n  ⚠️ Optimal threshold ({optimal_threshold:.2f}) differs significantly from default")
        print(f"     - Default (0.50) had FN={fn_default}, adjusting threshold helps")

else:
    print(f"\n  ⚠️ No threshold achieves Recall = 1.0")
    print(f"     This indicates some faults are inherently hard to detect")
    
    # Analyze FN at default
    fn_mask = (y_test == 1) & (y_proba < 0.5)
    fn_indices = np.where(fn_mask)[0]
    
    if fn_mask.sum() > 0:
        print(f"\n  FN Analysis at default threshold:")
        for i, idx in enumerate(fn_indices[:5]):
            print(f"    FN #{i+1}: P(Fault) = {y_proba[idx]:.4f}")
            # Could add feature analysis here
    
    # Strategy: Maximize Recall, then AHP, then closest to 0.50
    max_recall = df_sweep['Recall'].max()
    candidates = df_sweep[df_sweep['Recall'] == max_recall].copy()
    
    max_ahp = candidates['AHP_Score'].max()
    candidates = candidates[candidates['AHP_Score'] == max_ahp].copy()
    
    candidates['distance_to_default'] = (candidates['Threshold'] - 0.50).abs()
    best_row = candidates.loc[candidates['distance_to_default'].idxmin()]
    optimal_threshold = best_row['Threshold']
    
    print(f"\n  Best achievable Recall: {max_recall:.4f}")
    print(f"  Selected threshold: {optimal_threshold:.2f}")
    print(f"  Remaining FN: {int(best_row['FN'])} (unavoidable with current features)")

# --- Apply optimal threshold ---
y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(y_test, y_pred_optimal).ravel()

recall_opt    = recall_score(y_test, y_pred_optimal)
precision_opt = precision_score(y_test, y_pred_optimal, zero_division=0)
roc_auc_opt   = roc_auc_score(y_test, y_proba)
accuracy_opt  = accuracy_score(y_test, y_pred_optimal)
spec_opt      = tn_opt / (tn_opt + fp_opt) if (tn_opt + fp_opt) > 0 else 0.0

score_ahp_opt = W_RECALL * recall_opt + W_SPECIFICITY * spec_opt + W_AUC * roc_auc_opt

# --- Print results ---
print(f"\n" + "=" * 60)
print(f"RESULTS AT OPTIMAL THRESHOLD ({optimal_threshold:.2f})")
print("=" * 60)

print(f"\n  {'Metric':<20} {'Default (0.50)':>15} {'Optimal ({:.2f})':>15}".format(optimal_threshold))
print(f"  {'-'*52}")
print(f"  {'Recall':<20} {recall_default:>15.4f} {recall_opt:>15.4f}")
print(f"  {'Specificity':<20} {default_row['Specificity']:>15.4f} {spec_opt:>15.4f}")
print(f"  {'AHP Score':<20} {default_row['AHP_Score']:>15.4f} {score_ahp_opt:>15.4f}")
print(f"  {'FN':<20} {fn_default:>15d} {fn_opt:>15d}")
print(f"  {'FP':<20} {fp_default:>15d} {fp_opt:>15d}")

print(f"\n  Confusion Matrix (threshold = {optimal_threshold:.2f}):")
print(f"    TN={tn_opt}  FP={fp_opt}")
print(f"    FN={fn_opt}  TP={tp_opt}")

# Flag for later use
THRESHOLD_CHANGED = (optimal_threshold != 0.50)

# =============================================================================
# PHASE 6: STRATIFIED 5-FOLD CROSS-VALIDATION
# =============================================================================

print("\n" + "=" * 60)
print("PHASE 6: STRATIFIED 5-FOLD CROSS-VALIDATION")
print("=" * 60)

OPTIMAL_THRESHOLD = optimal_threshold
BEST_PARAMS = best_params_final

# Check if optimization actually changed the threshold
THRESHOLD_CHANGED = (abs(OPTIMAL_THRESHOLD - 0.50) > 0.001)

if THRESHOLD_CHANGED:
    THRESHOLDS = {
        'Default (0.50)': 0.50,
        f'Optimized ({OPTIMAL_THRESHOLD:.2f})': OPTIMAL_THRESHOLD
    }
    print(f"\n  Evaluating TWO thresholds:")
    print(f"    • Default: 0.50")
    print(f"    • Optimized: {OPTIMAL_THRESHOLD:.2f}")
else:
    THRESHOLDS = {
        'Default (0.50)': 0.50
    }
    print(f"\n  Optimal threshold = Default (0.50)")
    print(f"  Evaluating single threshold only (no comparison needed)")

print(f"\n  Model: Random Forest")
print(f"  Hyperparameters: {BEST_PARAMS}")
print(f"  Thresholds: {list(THRESHOLDS.values())}")
print(f"\n  AHP Weights (CR = 0.0032):")
print(f"    Recall:      {W_RECALL:.4f}")
print(f"    Specificity: {W_SPECIFICITY:.4f}")
print(f"    ROC-AUC:     {W_AUC:.4f}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=242)
cv_results = {t_name: [] for t_name in THRESHOLDS.keys()}

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    
    print(f"\n  {'─'*50}")
    print(f"  FOLD {fold_idx}/5")
    print(f"  {'─'*50}")
    
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    X_val_fold   = X.iloc[val_idx]
    y_val_fold   = y.iloc[val_idx]
    
    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)}")
    
    model_fold = RandomForestClassifier(
        **BEST_PARAMS,
        class_weight='balanced',
        random_state=242,
        n_jobs=-1
    )
    model_fold.fit(X_train_fold, y_train_fold)
    
    y_proba_fold = model_fold.predict_proba(X_val_fold)[:, 1]
    
    try:
        auc_fold = roc_auc_score(y_val_fold, y_proba_fold)
    except ValueError:
        auc_fold = 0.0
    
    for t_name, t_value in THRESHOLDS.items():
        y_pred_fold = (y_proba_fold >= t_value).astype(int)
        tn_f, fp_f, fn_f, tp_f = confusion_matrix(y_val_fold, y_pred_fold).ravel()
        
        rec_f  = recall_score(y_val_fold, y_pred_fold)
        prec_f = precision_score(y_val_fold, y_pred_fold, zero_division=0)
        spec_f = tn_f / (tn_f + fp_f) if (tn_f + fp_f) > 0 else 0.0
        acc_f  = accuracy_score(y_val_fold, y_pred_fold)
        
        ahp_f = (
            W_RECALL      * rec_f  +
            W_SPECIFICITY * spec_f +
            W_AUC         * auc_fold
        )
        
        fold_result = {
            'Fold':        fold_idx,
            'Recall':      rec_f,
            'Specificity': spec_f,
            'ROC_AUC':     auc_fold,
            'Precision':   prec_f,
            'Accuracy':    acc_f,
            'AHP_Score':   ahp_f,
            'TP': tp_f, 'FP': fp_f, 
            'FN': fn_f, 'TN': tn_f
        }
        cv_results[t_name].append(fold_result)
        
        if t_value == OPTIMAL_THRESHOLD:
            print(f"  Recall={rec_f:.4f} | Spec={spec_f:.4f} | AUC={auc_fold:.4f} | "
                  f"AHP={ahp_f:.4f} | FN={fn_f}")

# --- Summary ---
print(f"\n" + "=" * 60)
print("CROSS-VALIDATION SUMMARY")
print("=" * 60)

for t_name, t_results in cv_results.items():
    df_cv = pd.DataFrame(t_results)
    
    print(f"\n  {t_name}:")
    print(f"  {'─'*45}")
    print(f"  {'Metric':<15} {'Mean':>10} {'± Std':>10}")
    print(f"  {'─'*45}")
    
    for metric in ['Recall', 'Specificity', 'ROC_AUC', 'AHP_Score']:
        mean_val = df_cv[metric].mean()
        std_val  = df_cv[metric].std()
        print(f"  {metric:<15} {mean_val:>10.4f} {std_val:>10.4f}")
    
    total_fn = int(df_cv['FN'].sum())
    total_fp = int(df_cv['FP'].sum())
    print(f"\n  Total FN: {total_fn} | Total FP: {total_fp}")

# Store CV results
df_cv_def = pd.DataFrame(cv_results['Default (0.50)'])

if THRESHOLD_CHANGED:
    df_cv_opt = pd.DataFrame(cv_results[f'Optimized ({OPTIMAL_THRESHOLD:.2f})'])
else:
    df_cv_opt = df_cv_def.copy()  # Same as default
    
    
# =============================================================================
# PHASE 6.5: THRESHOLD VALIDATION (CV-Based Decision)
# =============================================================================

print("\n" + "=" * 60)
print("THRESHOLD VALIDATION (CV-Based Decision)")
print("=" * 60)

if not THRESHOLD_CHANGED:
    # No optimization happened, threshold stays at 0.50
    print(f"\n  ✅ Optimal threshold = Default (0.50)")
    print(f"     No validation needed.")
    FINAL_THRESHOLD = 0.50
    
else:
    # Compare CV performance
    cv_def_fn = int(df_cv_def['FN'].sum())
    cv_def_fp = int(df_cv_def['FP'].sum())
    cv_def_recall = df_cv_def['Recall'].mean()
    cv_def_ahp = df_cv_def['AHP_Score'].mean()
    
    cv_opt_fn = int(df_cv_opt['FN'].sum())
    cv_opt_fp = int(df_cv_opt['FP'].sum())
    cv_opt_recall = df_cv_opt['Recall'].mean()
    cv_opt_ahp = df_cv_opt['AHP_Score'].mean()
    
    print(f"\n  CV Performance Comparison:")
    print(f"  {'Metric':<20} {'Default (0.50)':>15} {'Optimized ({:.2f})':>15}".format(OPTIMAL_THRESHOLD))
    print(f"  {'-'*52}")
    print(f"  {'Recall (mean)':<20} {cv_def_recall:>15.4f} {cv_opt_recall:>15.4f}")
    print(f"  {'AHP Score (mean)':<20} {cv_def_ahp:>15.4f} {cv_opt_ahp:>15.4f}")
    print(f"  {'Total FN':<20} {cv_def_fn:>15d} {cv_opt_fn:>15d}")
    print(f"  {'Total FP':<20} {cv_def_fp:>15d} {cv_opt_fp:>15d}")
    
    # Decision: Optimized is BETTER if:
    # 1. Has equal or fewer FN (Dependability maintained)
    # 2. AND has equal or fewer FP (Security maintained or improved)
    
    optimized_is_better = (cv_opt_fn <= cv_def_fn) and (cv_opt_fp <= cv_def_fp)
    optimized_is_same = (cv_opt_fn == cv_def_fn) and (cv_opt_fp == cv_def_fp)
    
    if optimized_is_same:
        # Both perform equally - prefer 0.50 for simplicity
        print(f"\n  📌 Both thresholds perform IDENTICALLY in CV")
        print(f"     Keeping default (0.50) for simplicity and interpretability")
        FINAL_THRESHOLD = 0.50
        
    elif optimized_is_better:
        print(f"\n  ✅ Optimized threshold ({OPTIMAL_THRESHOLD:.2f}) CONFIRMED by CV!")
        
        if cv_opt_fn < cv_def_fn:
            print(f"     • FN improved: {cv_def_fn} → {cv_opt_fn} ✅")
        if cv_opt_fp < cv_def_fp:
            print(f"     • FP improved: {cv_def_fp} → {cv_opt_fp} ✅")
            
        FINAL_THRESHOLD = OPTIMAL_THRESHOLD
        
    else:
        print(f"\n  ⚠️ Optimized threshold ({OPTIMAL_THRESHOLD:.2f}) REJECTED by CV!")
        print(f"     CV shows worse generalization:")
        
        if cv_opt_fn > cv_def_fn:
            print(f"     • FN increased: {cv_def_fn} → {cv_opt_fn} ❌ (Dependability degraded)")
        if cv_opt_fp > cv_def_fp:
            print(f"     • FP increased: {cv_def_fp} → {cv_opt_fp} ❌ (Security degraded)")
        
        print(f"\n  🔄 REVERTING to default threshold (0.50)")
        FINAL_THRESHOLD = 0.50

# Update optimal_threshold with final decision
optimal_threshold = FINAL_THRESHOLD

# Update metrics if threshold changed
if FINAL_THRESHOLD != OPTIMAL_THRESHOLD:
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(y_test, y_pred_optimal).ravel()
    
    recall_opt    = recall_score(y_test, y_pred_optimal)
    precision_opt = precision_score(y_test, y_pred_optimal, zero_division=0)
    roc_auc_opt   = roc_auc_score(y_test, y_proba)
    accuracy_opt  = accuracy_score(y_test, y_pred_optimal)
    spec_opt      = tn_opt / (tn_opt + fp_opt) if (tn_opt + fp_opt) > 0 else 0.0
    
    score_ahp_opt = W_RECALL * recall_opt + W_SPECIFICITY * spec_opt + W_AUC * roc_auc_opt
    
    # Use default CV results as "optimized" since they're the same
    df_cv_opt = df_cv_def.copy()

print(f"\n  {'='*52}")
print(f"  FINAL THRESHOLD: {optimal_threshold:.2f}")
print(f"  {'='*52}")


# =============================================================================
# PHASE 7: SAVE MODEL AND RESULTS
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 7: SAVING MODEL AND RESULTS")
print("=" * 70)

# Ensure directories exist
import os
os.makedirs('Results', exist_ok=True)
os.makedirs('Images', exist_ok=True)

# --- Save trained model ---
model_path = 'Results/RF_detection_5bus.pkl'
joblib.dump(rf_optimizado, model_path)
print(f"  ✅ Model saved: {model_path}")

# --- Prepare CV dataframes (handle potential threshold revert) ---
df_cv_def = pd.DataFrame(cv_results['Default (0.50)'])

# Get the optimized threshold key that exists in cv_results
opt_threshold_key = f'Optimized ({optimal_threshold:.2f})'
if opt_threshold_key in cv_results:
    df_cv_opt = pd.DataFrame(cv_results[opt_threshold_key])
else:
    # If threshold was reverted, use default
    df_cv_opt = df_cv_def.copy()
    opt_threshold_key = 'Default (0.50)'

# --- Save complete results summary as JSON ---
results_summary = {
    'model': 'Random Forest',
    'task': 'Fault Detection (Binary)',
    'system': 'IEEE 5-Bus',
    'samples': int(X.shape[0]),
    'features': features,
    'hyperparameters': {k: (str(v) if v is None else v) for k, v in best_params_final.items()},
    'optimal_threshold': float(optimal_threshold),
    'ahp_config': {
        'weights': {
            'Recall': W_RECALL,
            'Specificity': W_SPECIFICITY,
            'ROC_AUC': W_AUC
        },
        'CR': 0.0032,
        'pairwise_matrix': {
            'Recall_vs_Specificity': 5,
            'Recall_vs_ROC_AUC': 3,
            'ROC_AUC_vs_Specificity': 2
        }
    },
    'test_set_results': {
        'default_threshold': {
            'threshold': 0.50,
            'recall': float(recall),
            'specificity': float(specificity),
            'roc_auc': float(roc_auc),
            'precision': float(precision),
            'accuracy': float(accuracy),
            'ahp_score': float(score_ahp),
            'confusion_matrix': {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}
        },
        'optimized_threshold': {
            'threshold': float(optimal_threshold),
            'recall': float(recall_opt),
            'specificity': float(spec_opt),
            'roc_auc': float(roc_auc_opt),
            'precision': float(precision_opt),
            'accuracy': float(accuracy_opt),
            'ahp_score': float(score_ahp_opt),
            'confusion_matrix': {'TN': int(tn_opt), 'FP': int(fp_opt), 'FN': int(fn_opt), 'TP': int(tp_opt)}
        }
    },
    'cv_results': {
        'n_folds': 5,
        'default_threshold': {
            'recall': {'mean': float(df_cv_def['Recall'].mean()), 'std': float(df_cv_def['Recall'].std())},
            'specificity': {'mean': float(df_cv_def['Specificity'].mean()), 'std': float(df_cv_def['Specificity'].std())},
            'roc_auc': {'mean': float(df_cv_def['ROC_AUC'].mean()), 'std': float(df_cv_def['ROC_AUC'].std())},
            'ahp_score': {'mean': float(df_cv_def['AHP_Score'].mean()), 'std': float(df_cv_def['AHP_Score'].std())},
            'total_FN': int(df_cv_def['FN'].sum()),
            'total_FP': int(df_cv_def['FP'].sum())
        },
        'optimized_threshold': {
            'recall': {'mean': float(df_cv_opt['Recall'].mean()), 'std': float(df_cv_opt['Recall'].std())},
            'specificity': {'mean': float(df_cv_opt['Specificity'].mean()), 'std': float(df_cv_opt['Specificity'].std())},
            'roc_auc': {'mean': float(df_cv_opt['ROC_AUC'].mean()), 'std': float(df_cv_opt['ROC_AUC'].std())},
            'ahp_score': {'mean': float(df_cv_opt['AHP_Score'].mean()), 'std': float(df_cv_opt['AHP_Score'].std())},
            'total_FN': int(df_cv_opt['FN'].sum()),
            'total_FP': int(df_cv_opt['FP'].sum())
        }
    },
    'feature_importance': {
        'ranking': df_importance.to_dict('records'),
        'currents_magnitude_contribution': float(imp_mag_currents),
        'currents_angle_contribution': float(imp_phi_currents),
        'voltages_magnitude_contribution': float(imp_mag_voltage),
        'voltages_angle_contribution': float(imp_phi_voltage)  # ✅ FIXED
    }
}

results_path = 'Results/RF_detection_5bus_summary.json'
with open(results_path, 'w') as f:
    json.dump(results_summary, f, indent=2)
print(f"  ✅ Summary saved: {results_path}")

# --- Save CV results as CSV ---
df_cv_def.to_csv('Results/RF_cv_default.csv', index=False)
df_cv_opt.to_csv('Results/RF_cv_optimized.csv', index=False)
print("  ✅ CV results saved: Results/RF_cv_default.csv")
print("  ✅ CV results saved: Results/RF_cv_optimized.csv")

# --- Save threshold sweep ---
df_sweep.to_csv('Results/RF_threshold_sweep.csv', index=False)
print("  ✅ Threshold sweep saved: Results/RF_threshold_sweep.csv")

# --- Save feature importance ---
df_importance.to_csv('Results/RF_feature_importance.csv', index=False)
print("  ✅ Feature importance saved: Results/RF_feature_importance.csv")

# =============================================================================
# PHASE 8: FINAL VISUALIZATION AND DOCUMENTATION
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 8: VISUALIZATION AND DOCUMENTATION")
print("=" * 70)

fig = plt.figure(figsize=(20, 14))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.30)

fig.suptitle('Random Forest - Fault Detection Model\n'
             'IEEE 5-Bus Test System | AHP Composite Score (CR = 0.0032)',
             fontsize=16, fontweight='bold', y=1.02)

# --- Panel 1: ROC Curve ---
ax1 = fig.add_subplot(gs[0, 0])

fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')

# Mark default and optimized threshold operating points
y_pred_default = (y_proba >= 0.5).astype(int)
y_pred_optimal = (y_proba >= optimal_threshold).astype(int)

tn_d, fp_d, fn_d, tp_d = confusion_matrix(y_test, y_pred_default).ravel()
tn_o, fp_o, fn_o, tp_o = confusion_matrix(y_test, y_pred_optimal).ravel()

fpr_default = fp_d / (fp_d + tn_d) if (fp_d + tn_d) > 0 else 0
tpr_default = tp_d / (tp_d + fn_d) if (tp_d + fn_d) > 0 else 0
fpr_optimal = fp_o / (fp_o + tn_o) if (fp_o + tn_o) > 0 else 0
tpr_optimal = tp_o / (tp_o + fn_o) if (tp_o + fn_o) > 0 else 0

ax1.scatter(fpr_default, tpr_default, color='gray', s=120, zorder=5,
            marker='o', edgecolors='black', label=f'Default (t=0.50)')
ax1.scatter(fpr_optimal, tpr_optimal, color='green', s=120, zorder=5,
            marker='*', edgecolors='black', label=f'Optimized (t={optimal_threshold:.2f})')

ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=10)
ax1.set_ylabel('True Positive Rate (Recall)', fontsize=10)
ax1.set_title('ROC Curve', fontsize=12, fontweight='bold')
ax1.legend(fontsize=8, loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(-0.02, 1.02)

# --- Panel 2: Confusion Matrix (Optimized Threshold) ---
ax2 = fig.add_subplot(gs[0, 1])

cm = confusion_matrix(y_test, y_pred_optimal)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax2,
            xticklabels=['No Fault', 'Fault'],
            yticklabels=['No Fault', 'Fault'],
            annot_kws={'size': 16, 'fontweight': 'bold'})
ax2.set_xlabel('Predicted', fontsize=10)
ax2.set_ylabel('Actual', fontsize=10)
ax2.set_title(f'Confusion Matrix (threshold = {optimal_threshold:.2f})', 
              fontsize=12, fontweight='bold')

# --- Panel 3: Feature Importance ---
ax3 = fig.add_subplot(gs[0, 2])

# Assign colors by feature type
colors_fi = [
    '#e74c3c' if f in ['phi_Ia', 'phi_Ib', 'phi_Ic'] else      # Current angles (red)
    '#f39c12' if f in ['phi_Va', 'phi_Vb', 'phi_Vc'] else      # Voltage angles (orange)
    '#27ae60' if f in ['Ia', 'Ib', 'Ic'] else                  # Current magnitudes (green)
    '#3498db'                                                   # Voltage magnitudes (blue)
    for f in df_importance['Feature']
]

bars_fi = ax3.barh(
    df_importance['Feature'][::-1],
    df_importance['Importance'][::-1],
    color=colors_fi[::-1],
    edgecolor='white', linewidth=0.5
)

for bar, val in zip(bars_fi, df_importance['Importance'][::-1]):
    ax3.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', fontsize=9)

ax3.set_xlabel('Importance', fontsize=10)
ax3.set_title('Feature Importance (MDI)', fontsize=12, fontweight='bold')

max_importance = df_importance['Importance'].max()
ax3.set_xlim(0, max_importance * 1.25)  # 25% extra space for labels

legend_fi = [
    Patch(facecolor='#e74c3c', label=f'Current Angles ({imp_phi_currents*100:.1f}%)'),   # ✅ FIXED
    Patch(facecolor='#f39c12', label=f'Voltage Angles ({imp_phi_voltage*100:.1f}%)'),
    Patch(facecolor='#27ae60', label=f'Current Magnitudes ({imp_mag_currents*100:.1f}%)'),
    Patch(facecolor='#3498db', label=f'Voltage Magnitudes ({imp_mag_voltage*100:.1f}%)')
]
ax3.legend(handles=legend_fi, fontsize=7, loc='lower right')

# --- Panel 4: Threshold Sweep (Recall vs Specificity) ---
ax4 = fig.add_subplot(gs[1, 0])

ax4.plot(df_sweep['Threshold'], df_sweep['Recall'],
         'r-', linewidth=2, label='Recall (Dependability)')
ax4.plot(df_sweep['Threshold'], df_sweep['Specificity'],
         'b-', linewidth=2, label='Specificity (Security)')
ax4.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Default (0.50)')
ax4.axvline(x=optimal_threshold, color='green', linestyle='--',
            linewidth=2, label=f'Optimized ({optimal_threshold:.2f})')

ax4.set_xlabel('Decision Threshold', fontsize=10)
ax4.set_ylabel('Score', fontsize=10)
ax4.set_title('Threshold Optimization', fontsize=12, fontweight='bold')
ax4.legend(fontsize=8)
ax4.set_xlim(0.05, 0.95)
ax4.set_ylim(-0.05, 1.05)
ax4.grid(True, alpha=0.3)

# --- Panel 5: Cross-Validation Metrics per Fold ---
ax5 = fig.add_subplot(gs[1, 1])

folds_x = np.arange(1, 6)
metrics_cv = ['Recall', 'Specificity', 'ROC_AUC']
colors_cv = ['#F44336', '#2196F3', '#9C27B0']
markers_cv = ['o', 's', '^']

for metric, color, marker in zip(metrics_cv, colors_cv, markers_cv):
    values = df_cv_opt[metric].values
    ax5.plot(folds_x, values, color=color, marker=marker,
             linewidth=2, markersize=8, label=f'{metric} (μ={np.mean(values):.4f})')

ax5.set_xlabel('Fold', fontsize=10)
ax5.set_ylabel('Score', fontsize=10)
ax5.set_title(f'5-Fold CV Metrics (threshold = {optimal_threshold:.2f})',
              fontsize=12, fontweight='bold')
ax5.set_xticks(folds_x)
ax5.set_ylim(0.85, 1.01)
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# --- Panel 6: FN/FP Comparison Default vs Optimized ---
ax6 = fig.add_subplot(gs[1, 2])

categories = ['False Negatives\n(Missed Faults)', 'False Positives\n(False Trips)']
default_counts = [df_cv_def['FN'].sum(), df_cv_def['FP'].sum()]
optimal_counts = [df_cv_opt['FN'].sum(), df_cv_opt['FP'].sum()]

x_bar = np.arange(len(categories))
width_bar = 0.35

bars_d = ax6.bar(x_bar - width_bar/2, default_counts, width_bar,
                  label='Default (0.50)', color='#90CAF9', alpha=0.8,
                  edgecolor='black', linewidth=0.5)
bars_o = ax6.bar(x_bar + width_bar/2, optimal_counts, width_bar,
                  label=f'Optimized ({optimal_threshold:.2f})', color='#4CAF50', alpha=0.8,
                  edgecolor='black', linewidth=0.5)

# Set reasonable y-axis limits
max_count = max(max(default_counts), max(optimal_counts), 1)  # At least 1 to avoid weird scaling
ax6.set_ylim(0, max_count * 1.3)  # 30% padding above max value

# Smart label positioning
for bar in bars_d:
    height = bar.get_height()
    # Position label inside bar if tall enough, otherwise above
    if height > max_count * 0.1:
        ax6.text(bar.get_x() + bar.get_width()/2., height/2,
                 f'{int(height)}', ha='center', va='center',
                 fontsize=12, fontweight='bold', color='white')
    else:
        ax6.text(bar.get_x() + bar.get_width()/2., height + max_count * 0.05,
                 f'{int(height)}', ha='center', va='bottom',
                 fontsize=12, fontweight='bold')

for bar in bars_o:
    height = bar.get_height()
    if height > max_count * 0.1:
        ax6.text(bar.get_x() + bar.get_width()/2., height/2,
                 f'{int(height)}', ha='center', va='center',
                 fontsize=12, fontweight='bold', color='white')
    else:
        ax6.text(bar.get_x() + bar.get_width()/2., height + max_count * 0.05,
                 f'{int(height)}', ha='center', va='bottom',
                 fontsize=12, fontweight='bold')

ax6.set_ylabel('Total Count (across 5 folds)', fontsize=10)
ax6.set_title('FN vs FP Trade-off (5-Fold CV)', fontsize=12, fontweight='bold')
ax6.set_xticks(x_bar)
ax6.set_xticklabels(categories, fontsize=9)
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3, axis='y')

# --- Panel 7: AHP Score Breakdown ---
ax7 = fig.add_subplot(gs[2, :])

models_labels = ['Default (0.50)', f'Optimized ({optimal_threshold:.2f})']

# Compute contributions for both thresholds (CV means)
recall_def_cv = df_cv_def['Recall'].mean()
spec_def_cv = df_cv_def['Specificity'].mean()
auc_def_cv = df_cv_def['ROC_AUC'].mean()

recall_opt_cv = df_cv_opt['Recall'].mean()
spec_opt_cv = df_cv_opt['Specificity'].mean()
auc_opt_cv = df_cv_opt['ROC_AUC'].mean()

contributions = {
    f'Recall × {W_RECALL:.4f}': [W_RECALL * recall_def_cv, W_RECALL * recall_opt_cv],
    f'Specificity × {W_SPECIFICITY:.4f}': [W_SPECIFICITY * spec_def_cv, W_SPECIFICITY * spec_opt_cv],
    f'ROC-AUC × {W_AUC:.4f}': [W_AUC * auc_def_cv, W_AUC * auc_opt_cv],
}

colors_ahp = ['#F44336', '#2196F3', '#9C27B0']
x_ahp = np.arange(len(models_labels))
bottom = np.zeros(len(models_labels))

for (label, values), color in zip(contributions.items(), colors_ahp):
    ax7.barh(x_ahp, values, left=bottom, height=0.5,
             label=label, color=color, alpha=0.8, edgecolor='white')
    for i, (val, bot) in enumerate(zip(values, bottom)):
        if val > 0.02:
            ax7.text(bot + val/2, i, f'{val:.4f}',
                     ha='center', va='center', fontsize=9, fontweight='bold',
                     color='white')
    bottom += values

for i, total in enumerate(bottom):
    ax7.text(total + 0.01, i, f'Total: {total:.4f}',
             ha='left', va='center', fontsize=11, fontweight='bold')

ax7.set_yticks(x_ahp)
ax7.set_yticklabels(models_labels, fontsize=11)
ax7.set_xlabel('AHP Composite Score', fontsize=10)
ax7.set_title('AHP Score Breakdown by Component (5-Fold CV Mean)\n'
              'Weights: Recall=0.6483 | ROC-AUC=0.2297 | Specificity=0.1220 | CR=0.0032',
              fontsize=12, fontweight='bold')
ax7.legend(fontsize=9, loc='lower right', ncol=3)
ax7.set_xlim(0, 1.05)
ax7.grid(True, alpha=0.3, axis='x')

plt.savefig('Images/RF_detection_dashboard.png', dpi=200, bbox_inches='tight')
print("  📊 Dashboard saved: Images/RF_detection_dashboard.png")

# =============================================================================
# FIGURE 2: PROBABILITY DISTRIBUTION ANALYSIS
# =============================================================================

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Probability Distribution Analysis - Random Forest Detection',
              fontsize=14, fontweight='bold')

# --- Panel 1: Histogram ---
ax_p1 = axes2[0]

proba_fault = y_proba[y_test.values == 1]
proba_no_fault = y_proba[y_test.values == 0]

ax_p1.hist(proba_no_fault, bins=20, alpha=0.7, color='#2196F3',
           label=f'No Fault (n={len(proba_no_fault)})', density=True,
           edgecolor='white')
ax_p1.hist(proba_fault, bins=20, alpha=0.7, color='#F44336',
           label=f'Fault (n={len(proba_fault)})', density=True,
           edgecolor='white')

ax_p1.axvline(x=0.5, color='gray', linestyle='--', linewidth=2,
              label='Default (0.50)')
ax_p1.axvline(x=optimal_threshold, color='green', linestyle='--',
              linewidth=2, label=f'Optimized ({optimal_threshold:.2f})')

# Only show recovery zone if thresholds are different
if optimal_threshold != 0.5:
    ax_p1.axvspan(min(optimal_threshold, 0.5), max(optimal_threshold, 0.5), 
                  alpha=0.1, color='red', label='Threshold adjustment zone')

ax_p1.set_xlabel('Predicted P(Fault)', fontsize=11)
ax_p1.set_ylabel('Density', fontsize=11)
ax_p1.set_title('Probability Distribution by Class', fontsize=12)
ax_p1.legend(fontsize=8)
ax_p1.grid(True, alpha=0.3)

# --- Panel 2: Boxplot ---
ax_p2 = axes2[1]

data_box = [proba_no_fault, proba_fault]
bp = ax_p2.boxplot(data_box, labels=['No Fault', 'Fault'],
                    patch_artist=True, widths=0.5)

bp['boxes'][0].set_facecolor('#2196F3')
bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor('#F44336')
bp['boxes'][1].set_alpha(0.7)

ax_p2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5,
              label='Default (0.50)')
ax_p2.axhline(y=optimal_threshold, color='green', linestyle='--',
              linewidth=2, label=f'Optimized ({optimal_threshold:.2f})')

ax_p2.set_ylabel('Predicted P(Fault)', fontsize=11)
ax_p2.set_title('Probability Boxplot by Class', fontsize=12)
ax_p2.legend(fontsize=9)
ax_p2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Images/RF_probability_analysis.png', dpi=150, bbox_inches='tight')
print("  📊 Probability analysis saved: Images/RF_probability_analysis.png")

# =============================================================================
# SUMMARY TABLES FOR THESIS
# =============================================================================

print("\n\n" + "=" * 70)
print("SUMMARY TABLES FOR THESIS DOCUMENTATION")
print("=" * 70)

# --- Prepare CV dataframes safely ---
df_cv_def_final = pd.DataFrame(cv_results['Default (0.50)'])

# Find the optimized threshold key that actually exists in cv_results
available_keys = list(cv_results.keys())
opt_keys = [k for k in available_keys if k.startswith('Optimized')]

if len(opt_keys) > 0:
    # Use the optimized key that exists
    cv_opt_key = opt_keys[0]
    df_cv_opt_final = pd.DataFrame(cv_results[cv_opt_key])
    
    # Extract the threshold value from the key for display
    import re
    match = re.search(r'Optimized \((\d+\.\d+)\)', cv_opt_key)
    cv_opt_threshold_display = float(match.group(1)) if match else optimal_threshold
else:
    # No optimized key found, use default
    df_cv_opt_final = df_cv_def_final.copy()
    cv_opt_threshold_display = 0.50

# If threshold was reverted, both should show the same results
if optimal_threshold == 0.50:
    print("  Note: Threshold was validated as 0.50 (same as default)")
    print("        CV results are identical for both columns.\n")
    df_cv_opt_final = df_cv_def_final.copy()
    cv_opt_threshold_display = 0.50

# --- Table 1: Model Configuration ---
print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│              TABLE 1: MODEL CONFIGURATION                           │
├──────────────────────────┬──────────────────────────────────────────┤
│ Parameter                │ Value                                    │
├──────────────────────────┼──────────────────────────────────────────┤
│ Algorithm                │ Random Forest                            │
│ n_estimators             │ {best_params_final['n_estimators']:<40} │
│ max_depth                │ {str(best_params_final['max_depth']):<40} │
│ min_samples_split        │ {best_params_final['min_samples_split']:<40} │
│ min_samples_leaf         │ {best_params_final['min_samples_leaf']:<40} │
│ max_features             │ {best_params_final['max_features']:<40} │
│ class_weight             │ balanced                                 │
│ Decision threshold       │ {optimal_threshold:<40.2f} │
│ Optimization method      │ GridSearchCV (360 combinations × 5 CV)  │
│ Selection criteria       │ AHP Score + Occam's razor tie-breaking  │
└──────────────────────────┴──────────────────────────────────────────┘
""")

# --- Table 2: Test Set Results ---
print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│          TABLE 2: TEST SET RESULTS (80/20 SPLIT)                    │
├──────────────────────────┬──────────────────┬───────────────────────┤
│ Metric                   │ Default (t=0.50) │ Optimized (t={optimal_threshold:.2f})     │
├──────────────────────────┼──────────────────┼───────────────────────┤
│ Recall (Dependability)   │     {recall:.4f}       │     {recall_opt:.4f}              │
│ Specificity (Security)   │     {specificity:.4f}       │     {spec_opt:.4f}              │
│ ROC-AUC (Discrimination) │     {roc_auc:.4f}       │     {roc_auc_opt:.4f}              │
│ AHP Score                │     {score_ahp:.4f}       │     {score_ahp_opt:.4f}              │
├──────────────────────────┼──────────────────┼───────────────────────┤
│ Precision                │     {precision:.4f}       │     {precision_opt:.4f}              │
│ Accuracy                 │     {accuracy:.4f}       │     {accuracy_opt:.4f}              │
├──────────────────────────┼──────────────────┼───────────────────────┤
│ False Negatives          │        {fn}            │        {fn_opt}                    │
│ False Positives          │        {fp}            │        {fp_opt}                    │
└──────────────────────────┴──────────────────┴───────────────────────┘
""")

# --- Table 3: Cross-Validation Results ---
print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│        TABLE 3: 5-FOLD CROSS-VALIDATION RESULTS                    │
├──────────────────────────┬──────────────────┬───────────────────────┤
│ Metric                   │ Default (t=0.50) │ Optimized (t={optimal_threshold:.2f})     │
│                          │   mean ± std     │   mean ± std          │
├──────────────────────────┼──────────────────┼───────────────────────┤
│ Recall                   │ {df_cv_def_final['Recall'].mean():.4f} ± {df_cv_def_final['Recall'].std():.4f}  │ {df_cv_opt_final['Recall'].mean():.4f} ± {df_cv_opt_final['Recall'].std():.4f}          │
│ Specificity              │ {df_cv_def_final['Specificity'].mean():.4f} ± {df_cv_def_final['Specificity'].std():.4f}  │ {df_cv_opt_final['Specificity'].mean():.4f} ± {df_cv_opt_final['Specificity'].std():.4f}          │
│ ROC-AUC                  │ {df_cv_def_final['ROC_AUC'].mean():.4f} ± {df_cv_def_final['ROC_AUC'].std():.4f}  │ {df_cv_opt_final['ROC_AUC'].mean():.4f} ± {df_cv_opt_final['ROC_AUC'].std():.4f}          │
│ AHP Score                │ {df_cv_def_final['AHP_Score'].mean():.4f} ± {df_cv_def_final['AHP_Score'].std():.4f}  │ {df_cv_opt_final['AHP_Score'].mean():.4f} ± {df_cv_opt_final['AHP_Score'].std():.4f}          │
├──────────────────────────┼──────────────────┼───────────────────────┤
│ Total FN (all folds)     │        {int(df_cv_def_final['FN'].sum())}          │        {int(df_cv_opt_final['FN'].sum())}                    │
│ Total FP (all folds)     │        {int(df_cv_def_final['FP'].sum())}          │        {int(df_cv_opt_final['FP'].sum())}                    │
└──────────────────────────┴──────────────────┴───────────────────────┘
""")

# Rest of your tables (Table 4, etc.)...
# --- Table 4: AHP Weight Justification (Updated) ---
print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│        TABLE 4: AHP WEIGHT JUSTIFICATION                           │
├─────────────────┬──────────┬────────────────────────────────────────┤
│ Metric          │ Weight   │ Justification                          │
├─────────────────┼──────────┼────────────────────────────────────────┤
│ Recall          │  0.6483  │ Dependability (IEEE C37.100)           │
│ (Dependability) │ (64.8%)  │ Highest priority: undetected fault     │
│                 │          │ causes equipment damage / human risk   │
├─────────────────┼──────────┼────────────────────────────────────────┤
│ ROC-AUC         │  0.2297  │ Global discrimination capability       │
│ (Discrimination)│ (23.0%)  │ Threshold-independent performance      │
│                 │          │ Integrates TPR and FPR information     │
├─────────────────┼──────────┼────────────────────────────────────────┤
│ Specificity     │  0.1220  │ Security (IEEE C37.100)                │
│ (Security)      │ (12.2%)  │ Lowest priority: false trip causes     │
│                 │          │ only temporary power interruption      │
├─────────────────┼──────────┼────────────────────────────────────────┤
│ TOTAL           │  1.0000  │                                        │
└─────────────────┴──────────┴────────────────────────────────────────┘

  Pairwise Comparison Matrix (Saaty, 1980):
  ┌─────────────┬────────┬─────────────┬─────────┐
  │             │ Recall │ Specificity │ ROC-AUC │
  ├─────────────┼────────┼─────────────┼─────────┤
  │ Recall      │  1.000 │    5.000    │  3.000  │
  │ Specificity │  0.200 │    1.000    │  0.500  │
  │ ROC-AUC     │  0.333 │    2.000    │  1.000  │
  └─────────────┴────────┴─────────────┴─────────┘

  Consistency Ratio: CR = 0.0032 ✅ (CR << 0.10)
""")


# =============================================================================
# FINAL STATUS
# =============================================================================

print(f"""

{'='*70}
RANDOM FOREST DETECTION MODEL - COMPLETE ✅
{'='*70}

  Files generated:
  │
  ├── Results/
  │   ├── RF_detection_5bus.pkl           (trained model)
  │   ├── RF_detection_5bus_summary.json  (complete config & results)
  │   ├── RF_cv_default.csv               (CV - default threshold)
  │   ├── RF_cv_optimized.csv             (CV - optimized threshold)
  │   ├── RF_threshold_sweep.csv          (all thresholds evaluated)
  │   └── RF_feature_importance.csv       (feature ranking)
  │
  └── Images/
      ├── RF_detection_dashboard.png      (7-panel summary)
      └── RF_probability_analysis.png     (probability distributions)

  Final Threshold: {optimal_threshold:.2f}
  {'(Same as default)' if optimal_threshold == 0.5 else '(Optimized from 0.50)'}

""")