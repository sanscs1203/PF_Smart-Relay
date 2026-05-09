"""
monte_carlo.py

 Module: Monte Carlo robustness analysis
 Purpose: Validate that the model ranking produced by the combined
          (AHP × Shannon Entropy) weights is stable under weight
          uncertainty. If the winner keeps winning across thousands of
          perturbed weight vectors, the decision is trustworthy.

 Role in the MCDM pipeline:
   AHP            → subjective weights
   Shannon Entropy → objective weights
   Combined        → w_final = normalize(w_AHP × w_Entropy)
   HERE            → perturb w_final → re-score → check ranking stability

 Perturbation method: Dirichlet distribution
   Each simulation draws a new weight vector from Dirichlet(α), where
   α = w_final * concentration. Higher concentration → perturbations
   stay closer to w_final (less uncertainty). Lower → wider spread.
   Dirichlet guarantees weights are always positive and sum to 1,
   making it the natural distribution for compositional data.

 Outputs:
   - Win rate (%) of each model across all simulations
   - Stability index: fraction of simulations where the full ranking
     matches the nominal ranking
   - Score distributions per model (mean ± std)

 Reference:
   Saltelli, A. et al. (2008). Global Sensitivity Analysis: The Primer.
   Wiley. (Chapter 1 — Monte Carlo methods for uncertainty propagation)

 Author  : Smart Relay Project — Fault Detection & Classification
 Standard: IEEE C37.100 / IEC 60255
 """

import numpy as np
from typing import List, Dict, Tuple


def monte_carlo_robustness(
    decision_matrix: np.ndarray,
    w_combined:      np.ndarray,
    metric_names:    List[str],
    model_names:     List[str],
    n_simulations:   int   = 1000,
    concentration:   float = 10.0,
    random_seed:     int   = 42,
) -> Dict:
    """
    Assess the robustness of the MCDM ranking via Monte Carlo simulation.

    In each simulation a new weight vector is sampled from a Dirichlet
    distribution centered on w_combined. The weighted sum score is
    computed for every model, producing a ranking. After all simulations,
    the function reports how often each model wins and how stable the
    full ranking is.

    Scoring method (weighted sum — WSM)
    ------------------------------------
        score_i = sum_j( w_j * x_ij )
    where x_ij is the normalized performance of model i on metric j
    (column-normalized decision matrix, same as in Shannon Entropy).

    Parameters
    ----------
    decision_matrix : np.ndarray, shape (n_models, n_metrics)
        Performance scores. Same matrix used in shannon_entropy_weights.
        All values must be strictly positive.

    w_combined : np.ndarray, shape (n_metrics,)
        Final combined weights from combined_weights() in shannon_entropy.py.
        These are the nominal weights that Monte Carlo will perturb.

    metric_names : List[str]
        Names of the criteria (columns).

    model_names : List[str]
        Names of the alternatives (rows).

    n_simulations : int, default 1000
        Number of Monte Carlo iterations.

    concentration : float, default 10.0
        Dirichlet concentration parameter (scalar applied to w_combined).
        Controls how tightly samples cluster around w_combined:
          - concentration = 1    → nearly uniform, very high uncertainty
          - concentration = 10   → moderate uncertainty (recommended)
          - concentration = 50   → low uncertainty, small perturbations
          - concentration = 100+ → near-deterministic, stress-test only

    random_seed : int, default 42
        Seed for reproducibility.

    Returns
    -------
    results : Dict with keys:
        'nominal_ranking'  : List[str]
            Model names ordered best → worst using w_combined.
        'nominal_scores'   : np.ndarray, shape (n_models,)
            WSM scores using w_combined (before perturbation).
        'win_rates'        : Dict[str, float]
            Percentage of simulations each model ranked first.
        'stability_index'  : float
            Fraction of simulations where the full ranking matches
            the nominal ranking. Range [0, 1].
            ≥ 0.70 → robust decision.
            < 0.50 → ranking is sensitive to weight uncertainty.
        'score_mean'       : Dict[str, float]
            Mean WSM score per model across all simulations.
        'score_std'        : Dict[str, float]
            Standard deviation of WSM score per model.
        'score_matrix'     : np.ndarray, shape (n_simulations, n_models)
            Full score matrix for custom post-processing or plotting.
        'weight_samples'   : np.ndarray, shape (n_simulations, n_metrics)
            All sampled weight vectors (useful for sensitivity plots).

    Raises
    ------
    ValueError
        If shapes are inconsistent or inputs contain non-positive values.

    Examples
    --------
    >>> import numpy as np
    >>> from utils.monte_carlo import monte_carlo_robustness
    >>> M = np.array([
    ...     [0.97, 0.94, 0.98],
    ...     [0.91, 0.96, 0.95],
    ...     [0.95, 0.89, 0.96],
    ... ])
    >>> w = np.array([0.60, 0.10, 0.30])
    >>> results = monte_carlo_robustness(M, w, ['Recall','Spec','AUC'],
    ...                                  ['RF','SVM','MLP'])
    """

    # Input validation — fix: use float instead of deprecated np.float
    decision_matrix = np.array(decision_matrix, dtype=float)
    w_combined      = np.array(w_combined,      dtype=float)

    n_models, n_metrics = decision_matrix.shape

    if decision_matrix.ndim != 2:
        raise ValueError("decision_matrix must be 2D (n_models, n_metrics).")
    if w_combined.ndim != 1 or len(w_combined) != n_metrics:
        raise ValueError(
            f"w_combined length ({len(w_combined)}) must match "
            f"number of metrics ({n_metrics})."
        )
    if len(metric_names) != n_metrics:
        raise ValueError(
            f"metric_names length ({len(metric_names)}) must match "
            f"number of metrics ({n_metrics})."
        )
    if len(model_names) != n_models:
        raise ValueError(
            f"model_names length ({len(model_names)}) must match "
            f"number of models ({n_models})."
        )
    if np.any(decision_matrix <= 0):
        raise ValueError("All values in decision_matrix must be strictly positive.")
    if concentration <= 0:
        raise ValueError("concentration must be a positive number.")
    if n_simulations < 1:
        raise ValueError("n_simulations must be at least 1.")

    # Normalize decision matrix (column-wise)
    col_sums          = decision_matrix.sum(axis=0)
    normalized_matrix = decision_matrix / col_sums

    # Nominal scores and ranking
    nominal_scores = normalized_matrix @ w_combined
    nominal_order  = np.argsort(nominal_scores)[::-1]
    nominal_ranking = [model_names[i] for i in nominal_order]

    # Dirichlet sampling
    rng            = np.random.default_rng(random_seed)
    alpha          = w_combined * concentration
    weight_samples = rng.dirichlet(alpha, size=n_simulations)

    # Score matrix: (n_simulations, n_models)
    score_matrix = weight_samples @ normalized_matrix.T

    # Winners and rankings per simulation
    winners     = np.argmax(score_matrix, axis=1)
    rank_matrix = np.argsort(score_matrix, axis=1)[:, ::-1]

    # Win rates
    win_counts = np.bincount(winners, minlength=n_models)
    win_rates  = {
        model_names[i]: float(win_counts[i] / n_simulations * 100)
        for i in range(n_models)
    }

    # Stability index
    nominal_ranking_indices = np.array(nominal_order)
    matches_nominal = np.all(rank_matrix == nominal_ranking_indices, axis=1)
    stability_index = float(matches_nominal.sum() / n_simulations)

    # Score statistics
    score_mean = {
        model_names[i]: float(score_matrix[:, i].mean())
        for i in range(n_models)
    }
    score_std = {
        model_names[i]: float(score_matrix[:, i].std())
        for i in range(n_models)
    }

    # Console reporting — fix: function was named _print_report but called as report
    _print_report(
        nominal_ranking, nominal_scores, nominal_order,
        win_rates, stability_index, score_mean, score_std,
        n_simulations, concentration, model_names,
    )

    return {
        "nominal_ranking": nominal_ranking,
        "nominal_scores":  nominal_scores,
        "win_rates":       win_rates,
        "stability_index": stability_index,
        "score_mean":      score_mean,
        "score_std":       score_std,
        "score_matrix":    score_matrix,
        "weight_samples":  weight_samples,
    }


# =============================================================================
# Console reporting
# =============================================================================

def _print_report(
    nominal_ranking: List[str],
    nominal_scores:  np.ndarray,
    nominal_order:   np.ndarray,
    win_rates:       Dict[str, float],
    stability_index: float,
    score_mean:      Dict[str, float],
    score_std:       Dict[str, float],
    n_simulations:   int,
    concentration:   float,
    model_names:     List[str],
) -> None:
    """Print a formatted Monte Carlo robustness summary to stdout."""

    sep = "=" * 62

    print(sep)
    print("  MONTE CARLO ROBUSTNESS ANALYSIS")
    print(f"  Simulations : {n_simulations:,}   "
          f"Concentration : {concentration}")
    print(sep)

    # Nominal ranking
    print(f"\n  Nominal ranking (w_combined, no perturbation):")
    print(f"  {'-' * 50}")
    for rank, name in enumerate(nominal_ranking, start=1):
        idx   = model_names.index(name)
        score = nominal_scores[idx]
        bar   = "█" * int(score * 60)
        print(f"  #{rank}  {name:<10s}  score = {score:.6f}  {bar}")

    # Win rates
    print(f"\n  Win rate per model ({n_simulations:,} simulations):")
    print(f"  {'-' * 50}")
    for name in nominal_ranking:
        wr  = win_rates[name]
        bar = "█" * int(wr / 2)
        print(f"  {name:<10s}  {wr:6.2f}%  {bar}")

    # Score statistics
    print(f"\n  Score statistics (mean ± std):")
    print(f"  {'-' * 50}")
    for name in nominal_ranking:
        print(f"  {name:<10s}  "
              f"{score_mean[name]:.6f} ± {score_std[name]:.6f}")

    # Stability verdict
    pct = stability_index * 100
    if stability_index >= 0.70:
        verdict = "✅  ROBUST   — ranking stable under weight uncertainty"
    elif stability_index >= 0.50:
        verdict = "⚠️   MODERATE — ranking changes in some scenarios"
    else:
        verdict = "❌  UNSTABLE — ranking is sensitive to weight choice"

    print(f"\n  Stability index : {stability_index:.4f}  ({pct:.1f}% of simulations)")
    print(f"  Verdict         : {verdict}\n")