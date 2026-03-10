# =============================================================================
# AHP (Analytic Hierarchy Process) - Saaty (1980)
# Smart Relay - Fault Detection Model Selection
#
# Metrics based on IEEE C37.100 / IEC 60255:
#   - Recall        → Dependability (IEEE C37.100)
#   - Specificity   → Security      (IEEE C37.100)
#   - ROC-AUC       → Global discrimination capability
#
# Speed is REPORTED as complementary but excluded from the score
# because it depends on hardware, not on model quality.
# =============================================================================

import numpy as np

def ahp_weights(matrix, metric_names):
    """
    Computes AHP weights and Consistency Ratio from a pairwise comparison matrix.
    
    Parameters
    ----------
    matrix : np.ndarray
        Square pairwise comparison matrix (Saaty scale 1-9).
    metric_names : list of str
        Names of the criteria/metrics being compared.
    
    Returns
    -------
    weights : np.ndarray
        Normalized priority vector (weights summing to 1).
    CR : float
        Consistency Ratio. Must be < 0.10 for acceptable consistency.
    """
    n = matrix.shape[0]
    
    # Step 1: Compute principal eigenvalue and eigenvector
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_idx = np.argmax(eigenvalues.real)
    principal_eigenvector = eigenvectors[:, max_idx].real
    
    # Step 2: Normalize eigenvector to obtain priority weights
    weights = principal_eigenvector / principal_eigenvector.sum()
    
    # Step 3: Compute Consistency Ratio (CR)
    lambda_max = eigenvalues[max_idx].real
    CI = (lambda_max - n) / (n - 1)
    
    # Random Index table (Saaty, 1980)
    RI_dict = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 
               5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 
               9: 1.45, 10: 1.49}
    RI = RI_dict.get(n, 1.49)
    CR = CI / RI if RI > 0 else 0
    
    # Step 4: Display results
    print("=" * 60)
    print("AHP WEIGHTS (Analytic Hierarchy Process - Saaty, 1980)")
    print("=" * 60)
    
    print(f"\n  Pairwise Comparison Matrix:\n")
    header = "              " + "".join(f"{name:>12s}" for name in metric_names)
    print(header)
    for i, name in enumerate(metric_names):
        row = f"  {name:12s}" + "".join(f"{matrix[i,j]:12.3f}" for j in range(n))
        print(row)
    
    print(f"\n  Principal Eigenvalue (λ_max): {lambda_max:.4f}")
    print(f"  Consistency Index (CI):       {CI:.4f}")
    print(f"  Random Index (RI):            {RI:.4f}")
    print(f"  Consistency Ratio (CR):       {CR:.4f}")
    print(f"  Status: {'✅ CONSISTENT (CR < 0.10)' if CR < 0.10 else '❌ INCONSISTENT (CR >= 0.10)'}")
    
    print(f"\n  Computed Weights:")
    print(f"  {'-'*45}")
    for name, w in zip(metric_names, weights):
        bar = "█" * int(w * 50)
        print(f"  {name:15s}: {w:.4f}  {bar}")
    print(f"  {'-'*45}")
    print(f"  {'Sum':15s}: {weights.sum():.4f}")
    
    return weights, CR


# =============================================================================
# CONFIGURATION - 3 METRICS
# =============================================================================

metric_names = ['Recall', 'Specificity', 'ROC-AUC']

# Pairwise comparison matrix (Saaty scale 1-9)
#
# Recall vs Specificity = 5 (strongly more important)
#   → IEEE C37.100: Dependability > Security.
#   → A FN (undetected fault) causes equipment destruction,
#     fire hazard, human risk.
#   → A FP (false trip) causes only a temporary service
#     interruption that is quickly recoverable.
#
# Recall vs ROC-AUC = 3 (moderately more important)
#   → Recall directly measures the relay's operational performance
#     at the chosen threshold (the actual operating point).
#   → AUC is a global quality indicator across all possible
#     thresholds, useful but less directly operational.
#
# ROC-AUC vs Specificity = 2 (slightly more important)
#   → AUC integrates information from BOTH dimensions (TPR and FPR)
#     providing a more comprehensive view than Specificity alone.
#   → A model with higher AUC will achieve better Recall-Specificity
#     trade-offs during threshold optimization.
#
# Consistency check:
#   If Recall/Specificity = 5 and Recall/AUC = 3,
#   then AUC/Specificity should be ≈ 5/3 ≈ 1.67 → we use 2
#   This is consistent (no contradictions).

comparison_matrix = np.array([
    #  Recall  Specif.  AUC
    [  1,      5,       3    ],  # Recall
    [  1/5,    1,       1/2  ],  # Specificity
    [  1/3,    2,       1    ],  # ROC-AUC
])

# =============================================================================
# RUN
# =============================================================================

weights_ahp, CR = ahp_weights(comparison_matrix, metric_names)