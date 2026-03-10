# =============================================================================
# MULTI-CRITERIA DECISION MAKING (MCDM) FOR MODEL SELECTION
# Smart Relay - Fault Detection
# 
# Methods:
#   1. AHP (Analytic Hierarchy Process) - Subjective weights
#   2. Shannon Entropy - Objective weights
#   3. Monte Carlo Sensitivity Analysis - Robustness validation
#
# Metrics based on IEEE C37.100 / IEC 60255:
#   - Recall        → Dependability
#   - Specificity   → Security
#   - ROC-AUC       → Global discrimination capability
#   - Speed (1/t)   → Operating speed
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# =============================================================================
# METHOD 1: ANALYTIC HIERARCHY PROCESS (AHP) - Saaty (1980)
# =============================================================================

def ahp_weights(matrix, metric_names):
    """
    Computes AHP weights and Consistency Ratio from a pairwise comparison matrix.
    
    Parameters
    ----------
    matrix : np.ndarray
        Square pairwise comparison matrix (Saaty scale 1-9).
        Element (i,j) represents the relative importance of criterion i over j.
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
    
    # --- Step 1: Compute principal eigenvalue and eigenvector ---
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_idx = np.argmax(eigenvalues.real)
    principal_eigenvector = eigenvectors[:, max_idx].real
    
    # --- Step 2: Normalize eigenvector to obtain priority weights ---
    weights = principal_eigenvector / principal_eigenvector.sum()
    
    # --- Step 3: Compute Consistency Ratio (CR) ---
    lambda_max = eigenvalues[max_idx].real
    CI = (lambda_max - n) / (n - 1)  # Consistency Index
    
    # Random Index table (Saaty, 1980) for matrices of size 1 to 10
    RI_dict = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 
               5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 
               9: 1.45, 10: 1.49}
    RI = RI_dict.get(n, 1.49)
    CR = CI / RI if RI > 0 else 0
    
    # --- Step 4: Display results ---
    print("=" * 60)
    print("AHP WEIGHTS (Analytic Hierarchy Process)")
    print("=" * 60)
    print(f"\n  Pairwise Comparison Matrix ({n}x{n}):\n")
    
    # Print matrix with labels
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
    print(f"  {'-'*40}")
    for name, w in zip(metric_names, weights):
        bar = "█" * int(w * 50)
        print(f"  {name:15s}: {w:.4f}  {bar}")
    print(f"  {'-'*40}")
    print(f"  {'Sum':15s}: {weights.sum():.4f}")
    
    return weights, CR


# --- Define metrics and pairwise comparison matrix ---
# Based on IEEE C37.100 / IEC 60255 protection philosophy:
#   - Dependability (Recall) has PRIORITY over Security (Specificity)
#   - A missed fault (FN) is catastrophic vs a false trip (FP) is tolerable

metric_names = ['Recall', 'Specificity', 'ROC-AUC', 'Speed_inv']

# Pairwise comparison matrix (Saaty scale 1-9)
# 
# Justification for each comparison:
#
# Recall vs Specificity = 5 (strongly more important)
#   → IEEE C37.100: Dependability > Security. A FN (undetected fault)
#     causes equipment damage or human risk. A FP only causes
#     a temporary service interruption.
#
# Recall vs ROC-AUC = 3 (moderately more important)
#   → Recall directly measures operational performance of the relay.
#     AUC is a global measure that includes threshold scenarios
#     that will not be used in practice.
#
# Recall vs Speed = 3 (moderately more important)
#   → A correct but slow detection is preferable to a fast but
#     incorrect decision. However, speed remains critical in
#     protection systems (typically < 30-80 ms).
#
# ROC-AUC vs Specificity = 3 (moderately more important)
#   → AUC integrates information from both axes (TPR and FPR),
#     while Specificity only measures one dimension.
#
# Speed vs Specificity = 2 (slightly more important)
#   → Both are secondary to detection, but a slow relay can
#     still cause damage even if it eventually detects correctly.
#
# ROC-AUC vs Speed = 1 (equal importance)
#   → Both are complementary: AUC ensures good discrimination,
#     Speed ensures timely response.

comparison_matrix = np.array([
    #  Recall  Specif.  AUC    Speed
    [  1,      5,       3,     3    ],  # Recall
    [  1/5,    1,       1/3,   1/2  ],  # Specificity
    [  1/3,    3,       1,     1    ],  # ROC-AUC
    [  1/3,    2,       1,     1    ],  # Speed_inv
])

# Compute AHP weights
weights_ahp, CR = ahp_weights(comparison_matrix, metric_names)


# =============================================================================
# METHOD 2: SHANNON ENTROPY - Objective Weights (Data-Driven)
# =============================================================================

def entropy_weights(decision_matrix, metric_names):
    """
    Computes objective weights using Shannon Entropy method.
    
    Logic: A metric with HIGH variation across models has MORE 
    discriminating power → deserves MORE weight. A metric where 
    all models score similarly provides NO useful information 
    for ranking → deserves LESS weight.
    
    Parameters
    ----------
    decision_matrix : pd.DataFrame
        Rows = models/alternatives, Columns = metrics/criteria.
        All values should be positive (metrics in [0,1] recommended).
    metric_names : list of str
        Names of the criteria/metrics.
    
    Returns
    -------
    weights : pd.Series
        Entropy-based weights for each metric (summing to 1).
    entropy : pd.Series
        Entropy value for each metric.
    diversity : pd.Series
        Diversity (1 - entropy) for each metric.
    """
    n_models = len(decision_matrix)
    
    # --- Step 1: Normalize each column to [0,1] range (min-max) ---
    norm = decision_matrix.copy()
    for col in norm.columns:
        col_min = norm[col].min()
        col_max = norm[col].max()
        if col_max - col_min > 0:
            norm[col] = (norm[col] - col_min) / (col_max - col_min)
        else:
            # No variation → metric cannot discriminate
            norm[col] = 1.0
    
    # Add small epsilon to avoid log(0)
    norm = norm + 1e-10
    
    # --- Step 2: Compute proportion of each model per metric ---
    P = norm.div(norm.sum(axis=0), axis=1)
    
    # --- Step 3: Compute Shannon Entropy for each metric ---
    k = 1.0 / np.log(n_models)  # Normalization constant
    entropy = -k * (P * np.log(P)).sum(axis=0)
    
    # --- Step 4: Compute diversity (inverse of entropy) ---
    diversity = 1 - entropy
    
    # --- Step 5: Normalize to obtain weights ---
    weights = diversity / diversity.sum()
    
    # --- Step 6: Display results ---
    print("\n" + "=" * 60)
    print("SHANNON ENTROPY WEIGHTS (Objective / Data-Driven)")
    print("=" * 60)
    print(f"\n  Normalized Decision Matrix:")
    print(f"  {norm.round(4).to_string()}")
    print(f"\n  {'Metric':<15s} {'Entropy':>10s} {'Diversity':>10s} {'Weight':>10s}")
    print(f"  {'-'*45}")
    for metric in metric_names:
        bar = "█" * int(weights[metric] * 50)
        print(f"  {metric:<15s} {entropy[metric]:10.4f} {diversity[metric]:10.4f} {weights[metric]:10.4f}  {bar}")
    print(f"  {'-'*45}")
    print(f"  {'Sum':<15s} {'':>10s} {'':>10s} {weights.sum():10.4f}")
    
    return weights, entropy, diversity


# NOTE: This function is called AFTER having results from all 3 models.
# Example with placeholder data (replace with actual results later):

print("\n\n" + "#" * 60)
print("# EXAMPLE WITH PLACEHOLDER DATA")
print("# Replace with actual model results after training")
print("#" * 60)

example_results = pd.DataFrame({
    'Recall':      [0.95, 0.99, 0.92],
    'Specificity': [0.97, 0.93, 0.98],
    'ROC-AUC':     [0.98, 0.99, 0.97],
    'Speed_inv':   [0.90, 0.60, 0.85],
}, index=['Random_Forest', 'MLP', 'SVM'])

weights_entropy, E, D = entropy_weights(example_results, metric_names)


# =============================================================================
# METHOD 3: MONTE CARLO SENSITIVITY ANALYSIS - Robustness Validation
# =============================================================================

def sensitivity_analysis(results_df, metric_names, n_simulations=10000,
                         recall_dominant=True, seed=42):
    """
    Monte Carlo sensitivity analysis to validate ranking robustness.
    
    Generates thousands of random weight combinations and checks 
    whether the winning model changes. If the best model wins in 
    >80% of simulations, the result is robust regardless of exact weights.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Rows = models, Columns = metrics. Values in [0,1].
    metric_names : list of str
        Names of the metrics (must match column names).
    n_simulations : int
        Number of random weight combinations to test.
    recall_dominant : bool
        If True, constrains Recall to always have the highest weight.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    win_count : dict
        Number of times each model won across all simulations.
    all_scores : np.ndarray
        Score matrix (n_simulations x n_models).
    all_weights : np.ndarray
        Weight matrix (n_simulations x n_metrics).
    """
    np.random.seed(seed)
    n_metrics = results_df.shape[1]
    n_models = results_df.shape[0]
    model_names = results_df.index.tolist()
    
    win_count = {name: 0 for name in model_names}
    all_scores = []
    all_weights = []
    
    for _ in range(n_simulations):
        # --- Generate random weights using Dirichlet distribution ---
        # Dirichlet ensures weights are positive and sum to 1
        raw_weights = np.random.dirichlet(np.ones(n_metrics))
        
        if recall_dominant:
            # Ensure Recall always has the highest weight
            # (consistent with IEEE C37.100 Dependability priority)
            recall_idx = list(results_df.columns).index('Recall')
            max_idx = np.argmax(raw_weights)
            if max_idx != recall_idx:
                # Swap so Recall gets the largest weight
                raw_weights[recall_idx], raw_weights[max_idx] = \
                    raw_weights[max_idx], raw_weights[recall_idx]
        
        # --- Compute weighted score for each model ---
        scores = results_df.values @ raw_weights
        winner = model_names[np.argmax(scores)]
        win_count[winner] += 1
        all_scores.append(scores)
        all_weights.append(raw_weights)
    
    all_scores = np.array(all_scores)
    all_weights = np.array(all_weights)
    
    # --- Display results ---
    print("\n" + "=" * 60)
    print(f"MONTE CARLO SENSITIVITY ANALYSIS ({n_simulations:,} simulations)")
    print("=" * 60)
    print(f"  Constraint: Recall always dominant = {recall_dominant}")
    print(f"  Random seed: {seed}")
    print(f"\n  {'Model':<20s} {'Wins':>8s} {'Percentage':>12s}")
    print(f"  {'-'*50}")
    for name in model_names:
        pct = win_count[name] / n_simulations * 100
        bar = "█" * int(pct / 2)
        print(f"  {name:<20s} {win_count[name]:>8,d} {pct:>10.1f}%  {bar}")
    
    # --- Determine robustness ---
    best_model = max(win_count, key=win_count.get)
    best_pct = win_count[best_model] / n_simulations * 100
    print(f"\n  Best model: {best_model} ({best_pct:.1f}% of simulations)")
    if best_pct >= 80:
        print(f"  ✅ ROBUST: Result holds regardless of exact weight values")
    elif best_pct >= 60:
        print(f"  ⚠️  MODERATE: Result is likely but sensitive to weights")
    else:
        print(f"  ❌ SENSITIVE: Result depends heavily on weight selection")
    
    return win_count, all_scores, all_weights


# --- Run sensitivity analysis with placeholder data ---
win_count, all_scores, all_weights = sensitivity_analysis(
    example_results, metric_names, n_simulations=10000
)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_mcdm_results(weights_ahp, weights_entropy, metric_names,
                      win_count, all_scores, model_names):
    """
    Generates comprehensive visualization of MCDM results.
    
    Produces 4 subplots:
    1. AHP vs Entropy weights comparison
    2. Weight distribution from Monte Carlo simulations
    3. Win percentage per model (sensitivity analysis)
    4. Score distribution per model across simulations
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Multi-Criteria Decision Making (MCDM) Analysis\n'
                 'Smart Relay - Fault Detection Model Selection',
                 fontsize=14, fontweight='bold', y=1.02)
    
    # --- Plot 1: AHP vs Entropy weights comparison ---
    ax1 = axes[0, 0]
    x = np.arange(len(metric_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, weights_ahp, width, 
                     label='AHP (Subjective)', color='#2196F3', alpha=0.8)
    bars2 = ax1.bar(x + width/2, [weights_entropy[m] for m in metric_names], width,
                     label='Entropy (Objective)', color='#FF9800', alpha=0.8)
    
    ax1.set_xlabel('Metric')
    ax1.set_ylabel('Weight')
    ax1.set_title('Weight Comparison: AHP vs Shannon Entropy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names, rotation=15)
    ax1.legend()
    ax1.set_ylim(0, max(max(weights_ahp), max(weights_entropy.values)) * 1.3)
    
    # Add value labels on bars
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    
    # --- Plot 2: Monte Carlo weight distributions ---
    ax2 = axes[0, 1]
    # This will be filled when we have all_weights from Monte Carlo
    # For now, show weight distribution boxplot
    ax2.boxplot([all_weights[:, i] for i in range(len(metric_names))],
                labels=metric_names)
    ax2.set_ylabel('Weight Value')
    ax2.set_title(f'Weight Distribution Across {len(all_weights):,} MC Simulations')
    ax2.tick_params(axis='x', rotation=15)
    
    # --- Plot 3: Win percentage (sensitivity analysis) ---
    ax3 = axes[1, 0]
    models = list(win_count.keys())
    wins = list(win_count.values())
    total = sum(wins)
    percentages = [w/total*100 for w in wins]
    
    colors = ['#4CAF50', '#2196F3', '#FF5722']
    bars = ax3.bar(models, percentages, color=colors[:len(models)], alpha=0.8)
    ax3.set_ylabel('Win Percentage (%)')
    ax3.set_title('Sensitivity Analysis: Model Win Rate')
    ax3.set_ylim(0, 100)
    
    # Add percentage labels
    for bar, pct in zip(bars, percentages):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add robustness threshold line
    ax3.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Robust threshold (80%)')
    ax3.axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='Moderate threshold (60%)')
    ax3.legend(fontsize=8)
    
    # --- Plot 4: Score distribution per model ---
    ax4 = axes[1, 1]
    for i, name in enumerate(model_names):
        ax4.hist(all_scores[:, i], bins=50, alpha=0.6, label=name)
    ax4.set_xlabel('Composite Score')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Score Distribution Across MC Simulations')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('mcdm_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n  📊 Figure saved as 'mcdm_analysis.png'")


# --- Generate visualization ---
plot_mcdm_results(
    weights_ahp=weights_ahp,
    weights_entropy=weights_entropy,
    metric_names=metric_names,
    win_count=win_count,
    all_scores=all_scores,
    model_names=example_results.index.tolist()
)


# =============================================================================
# COMPOSITE SCORE COMPUTATION
# =============================================================================

def compute_composite_scores(results_df, weights, weight_method_name):
    """
    Computes the weighted composite score for each model.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Rows = models, Columns = metrics.
    weights : array-like
        Weights for each metric (must sum to 1).
    weight_method_name : str
        Name of the weighting method for display purposes.
    
    Returns
    -------
    scores : pd.Series
        Composite score for each model.
    """
    # Ensure weights are in the right format
    if isinstance(weights, pd.Series):
        w = weights.values
    else:
        w = np.array(weights)
    
    scores = pd.Series(
        results_df.values @ w,
        index=results_df.index,
        name=f'Score ({weight_method_name})'
    )
    
    # --- Display ranking ---
    print(f"\n{'='*60}")
    print(f"COMPOSITE SCORES - {weight_method_name} Weights")
    print(f"{'='*60}")
    
    ranking = scores.sort_values(ascending=False)
    for rank, (model, score) in enumerate(ranking.items(), 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "  ")
        bar = "█" * int(score * 50)
        print(f"  {medal} #{rank} {model:<20s}: {score:.4f}  {bar}")
    
    return scores


# --- Compute scores with both weighting methods ---
scores_ahp = compute_composite_scores(example_results, weights_ahp, "AHP")
scores_entropy = compute_composite_scores(example_results, weights_entropy, "Entropy")


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def print_summary_table(results_df, weights_ahp, weights_entropy, 
                        scores_ahp, scores_entropy, CR, metric_names):
    """
    Prints a comprehensive summary table for documentation.
    """
    print("\n\n" + "=" * 75)
    print("COMPLETE MCDM SUMMARY")
    print("=" * 75)
    
    # Weight comparison
    print(f"\n  {'Metric':<15s} {'AHP Weight':>12s} {'Entropy Weight':>16s} {'Difference':>12s}")
    print(f"  {'-'*55}")
    for m in metric_names:
        w_ahp = weights_ahp[list(metric_names).index(m)] if isinstance(weights_ahp, np.ndarray) else weights_ahp[m]
        w_ent = weights_entropy[m]
        diff = abs(w_ahp - w_ent)
        print(f"  {m:<15s} {w_ahp:>12.4f} {w_ent:>16.4f} {diff:>12.4f}")
    
    print(f"\n  AHP Consistency Ratio: {CR:.4f} {'✅' if CR < 0.10 else '❌'}")
    
    # Score comparison
    print(f"\n  {'Model':<20s} {'AHP Score':>12s} {'AHP Rank':>10s} {'Entropy Score':>15s} {'Entropy Rank':>14s}")
    print(f"  {'-'*71}")
    
    ahp_rank = scores_ahp.rank(ascending=False).astype(int)
    ent_rank = scores_entropy.rank(ascending=False).astype(int)
    
    for model in results_df.index:
        print(f"  {model:<20s} {scores_ahp[model]:>12.4f} {ahp_rank[model]:>10d} "
              f"{scores_entropy[model]:>15.4f} {ent_rank[model]:>14d}")
    
    # Check ranking agreement
    print(f"\n  Rankings agree: {'✅ YES' if list(ahp_rank) == list(ent_rank) else '⚠️  NO (check sensitivity analysis)'}")


print_summary_table(example_results, weights_ahp, weights_entropy,
                    scores_ahp, scores_entropy, CR, metric_names)

print("\n\n" + "#" * 60)
print("# NOTE: Replace example_results with actual model metrics")
print("# after training Random Forest, MLP, and SVM")
print("#" * 60)