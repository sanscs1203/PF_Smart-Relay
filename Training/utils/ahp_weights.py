"""
ahp_weights.py
-------------------
Module: AHP (Analytic Hierarchy Process) — Saaty (1980)
Purpose: Compute priority weights and Consistency Ratio from a pairwise
         comparison matrix built on the Saaty 1–9 scale.

Intended use: import this module from any MCDM pipeline that requires
              AHP-derived weights (detection module, classification module).

Author  : Smart Relay Project — Fault Detection & Classification
Standard: IEEE C37.100 / IEC 60255 (metric interpretation)
"""

import numpy as np

# Random Index table — Saaty (1980), up to n = 10
_RI_TABLE = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90,
    5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41,
    9: 1.45, 10: 1.49,
}

def ahp_weights(matrix: np.array, metric_names: list[str]) -> tuple[np.ndarray, float]:
    """
    Compute AHP priority weights and Consistency Ratio (CR) from a
    pairwise comparison matrix expressed on the Saaty 1–9 scale.
 
    The function follows the standard AHP procedure:
        1. Compute the principal eigenvalue (λ_max) and its eigenvector.
        2. Normalize the eigenvector to obtain the priority weight vector.
        3. Derive CI = (λ_max − n) / (n − 1).
        4. Derive CR = CI / RI  (RI from Saaty's Random Index table).
 
    A CR < 0.10 indicates acceptable judgement consistency.
 
    Parameters
    ----------
    matrix : np.ndarray, shape (n, n)
        Positive reciprocal pairwise comparison matrix.
        Entry [i, j] represents "how much more important criterion i is
        over criterion j" on Saaty's scale (1 = equal, 9 = extreme).
        The matrix must satisfy: matrix[j, i] = 1 / matrix[i, j].
 
    metric_names : list of str
        Names of the n criteria, in the same order as the matrix rows/columns.
 
    Returns
    -------
    weights : np.ndarray, shape (n,)
        Normalized priority vector. Values are positive and sum to 1.
 
    CR : float
        Consistency Ratio. Should be < 0.10 for a reliable judgement.
 
    Raises
    ------
    ValueError
        If the matrix is not square or its size does not match metric_names.
 
    Examples
    --------
    >>> import numpy as np
    >>> from ahp_weights import ahp_weights
    >>> M = np.array([[1, 5, 3], [1/5, 1, 1/2], [1/3, 2, 1]])
    >>> names = ['Recall', 'Specificity', 'ROC-AUC']
    >>> w, cr = ahp_weights(M, names)
    """
    
    # Validate input
    matrix = np.array(matrix)
    n = matrix.shape[0]
    
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square (n x n).")
    if n != len(metric_names):
        raise ValueError(
            f"Length of metric_names ({len(metric_names)}) must match matrix size ({n})."
        )
         
    # Principal eigenvalue and eigenvector computation
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_idx = np.argmax(eigenvalues.real)
    principal_eigenvector = eigenvectors[:, max_idx].real
    
    # Normalize to get priority weights
    weights = principal_eigenvector / principal_eigenvector.sum()
    
    # Consistency Ratio calculation
    lambda_max = eigenvalues[max_idx].real
    CI = (lambda_max - n) / (n - 1)
    RI = _RI_TABLE.get(n, 1.49)  # Default to RI for n > 10
    CR = CI / RI if RI > 0 else 0
    
    # Console reporting
    report(matrix, metric_names, lambda_max, weights, CI, RI, CR)
    
    return weights, CR

# =============================================================================
# Console reporting function for AHP results
# =============================================================================

def report(matrix: np.ndarray, 
           metric_names: list[str], 
           lambda_max: float, 
           weights: np.ndarray,
           CI: float, 
           RI: float, 
           CR: float) -> None:
    """Print a formatted report of the AHP weights and consistency analysis."""
    
    sep = "=" * 60
    print(sep)
    print("AHP WEIGHTS (Analytic Hierarchy Process - Saaty, 1980)")
    print(sep)
    
    # Pairwise comparison matrix
    col_w = 12
    header = " " * 16 + "".join(f"{name:>{col_w}s}" for name in metric_names)
    print(f"\n  Pairwise Comparison Matrix:\n\n{header}")
    for i, name in enumerate(metric_names):
        row_str = f"  {name:<14s}" + "".join(
            f"{matrix[i, j]:>{col_w}.4f}" for j in range(matrix.shape[1])
        )
        print(row_str)
        
    # Consistency metrics
    print(f"\n  Principal Eigenvalue (λ_max): {lambda_max:.4f}")
    print(f"  Consistency Index (CI):       {CI:.4f}")
    print(f"  Random Index (RI):            {RI:.4f}")
    print(f"  Consistency Ratio (CR):       {CR:.4f}")
    status = "✅ CONSISTENT (CR < 0.10)" if CR < 0.10 else "❌ INCONSISTENT (CR >= 0.10)"
    print(f"  Status: {status}")
    
    # Weight vector summary
    print(f"\n  Priority Weights:\n  {'-' * 44}")
    for name, w in zip(metric_names, weights):
        bar = "█" * int(w * 40)
        print(f"  {name:<15s}: {w:.6f}  {bar}")
    print(f"  {'-' * 44}")
    print(f"  {'Sum':<15s}: {weights.sum():.6f}\n")