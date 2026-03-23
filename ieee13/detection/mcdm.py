"""
ieee13/detection/mcdm.py
------------------------
Runs the full MCDM pipeline to select the best detection model from the
three candidates (RF, SVM, MLP) evaluated by evaluate.py.

MCDM pipeline
-------------
1. AHP  (subjective weights)  →  utils.ahp_weights.ahp_weights()
2. Monte Carlo robustness check  →  utils.monte_carlo.monte_carlo_robustness()
3. Best model = nominal winner with stability_index ≥ 0.70

Inputs
------
- results/decision_matrix.json  : produced by evaluate.py
- config.yaml                   : AHP pairwise matrix + Monte Carlo settings

Outputs (results/)
------------------
- mcdm_result.json  : weights, ranking, stability index, best model

Utility modules used
---------------------
- utils.io.load_config()                       : load YAML config
- utils.io.build_pairwise_matrix()             : reconstruct 2×2 AHP matrix
- utils.ahp_weights.ahp_weights()             : AHP subjective weights
- utils.monte_carlo.monte_carlo_robustness()  : ranking stability check

Usage (called from project root)
---------------------------------
    python ieee13/detection/mcdm.py --config ieee13/config.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path so utils/ is importable regardless of cwd
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.io import load_config, build_pairwise_matrix
from utils.ahp_weights import ahp_weights
from utils.monte_carlo import monte_carlo_robustness


# ---------------------------------------------------------------------------
# Decision matrix loader
# ---------------------------------------------------------------------------

def load_decision_matrix(results_dir: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """Load the decision matrix produced by evaluate.py.

    Parameters
    ----------
    results_dir : Path
        Directory containing decision_matrix.json.

    Returns
    -------
    tuple[np.ndarray, list[str], list[str]]
        (decision_matrix, model_names, metric_names)

    Raises
    ------
    FileNotFoundError
        If decision_matrix.json does not exist.
    """
    path = results_dir / "decision_matrix.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Decision matrix not found: {path}\n"
            "Run ieee13/detection/evaluate.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    decision_matrix = np.array(payload["decision_matrix"], dtype=float)
    model_names     = payload["model_names"]
    metric_names    = payload["metric_names"]

    print(f"[load]  Decision matrix loaded from '{path}'")
    print(f"        Models  : {model_names}")
    print(f"        Metrics : {metric_names}")
    print(f"        Shape   : {decision_matrix.shape}")
    return decision_matrix, model_names, metric_names

# ---------------------------------------------------------------------------
# Dependability override
# ---------------------------------------------------------------------------
 
def dependability_override(
    decision_matrix: np.ndarray,
    model_names:     list[str],
    metric_names:    list[str],
    nominal_winner:  str,
    spec_margin:     float,
) -> tuple[str, bool, str]:
    """Check if a model dominates the nominal winner on dependability.
 
    A model A dominates the nominal winner B if:
        Recall_A > Recall_B
        AND Specificity_A >= Specificity_B - spec_margin
 
    If multiple models dominate, the one with the highest Recall is
    selected. If no model dominates, the nominal winner is kept.
 
    Parameters
    ----------
    decision_matrix : np.ndarray, shape (n_models, 2)
        Columns: Recall, Specificity (test-set scores).
    model_names : list[str]
        Model identifiers in the same row order as decision_matrix.
    metric_names : list[str]
        Must contain 'Recall' and 'Specificity'.
    nominal_winner : str
        Model selected by the AHP + Monte Carlo ranking.
    spec_margin : float
        Maximum allowed drop in Specificity for a dominant model.
        Read from config.yaml: detection.mcdm.spec_margin.
 
    Returns
    -------
    tuple[str, bool, str]
        (best_model, override_applied, reason)
    """
    recall_idx = metric_names.index("Recall")
    spec_idx   = metric_names.index("Specificity")
 
    winner_row      = model_names.index(nominal_winner)
    winner_recall   = decision_matrix[winner_row, recall_idx]
    winner_spec     = decision_matrix[winner_row, spec_idx]
 
    dominant_model  = None
    dominant_recall = winner_recall
 
    for i, model in enumerate(model_names):
        if model == nominal_winner:
            continue
 
        recall = decision_matrix[i, recall_idx]
        spec   = decision_matrix[i, spec_idx]
 
        if recall > winner_recall and spec >= winner_spec - spec_margin:
            if recall > dominant_recall:
                dominant_recall = recall
                dominant_model  = model
 
    if dominant_model is not None:
        reason = (
            f"Recall({dominant_model})={decision_matrix[model_names.index(dominant_model), recall_idx]:.6f} "
            f"> Recall({nominal_winner})={winner_recall:.6f}  AND  "
            f"Specificity({dominant_model})={decision_matrix[model_names.index(dominant_model), spec_idx]:.6f} "
            f">= Specificity({nominal_winner}) - margin "
            f"({winner_spec:.6f} - {spec_margin} = {winner_spec - spec_margin:.6f})"
        )
        return dominant_model, True, reason
 
    reason = (
        f"No model dominates {nominal_winner} under spec_margin={spec_margin}. "
        f"Nominal winner retained."
    )
    return nominal_winner, False, reason


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_mcdm_result(
    w_ahp:           np.ndarray,
    mc_results:      dict,
    metric_names:    list[str],
    model_names:     list[str],
    best_model:      str,
    override:        bool,
    override_reason: str,
    results_dir:     Path,
) -> None:
    """Save the MCDM result to JSON.
 
    Parameters
    ----------
    w_ahp : np.ndarray
        AHP weights used for model selection.
    mc_results : dict
        Output of monte_carlo_robustness().
    metric_names : list[str]
    model_names : list[str]
    best_model : str
        Final selected model (may differ from nominal winner if override).
    override : bool
        True if dependability override was applied.
    override_reason : str
        Human-readable explanation of the override decision.
    results_dir : Path
    """
    stability = mc_results["stability_index"]
    if stability >= 0.70:
        verdict = "ROBUST"
    elif stability >= 0.50:
        verdict = "MODERATE"
    else:
        verdict = "UNSTABLE"
 
    report = {
        "metric_names": metric_names,
        "model_names":  model_names,
        "weights": {
            "ahp": [round(float(w), 6) for w in w_ahp],
        },
        "ranking": {
            "nominal":         mc_results["nominal_ranking"],
            "nominal_scores":  [round(float(s), 6)
                                for s in mc_results["nominal_scores"]],
            "win_rates":       {k: round(v, 2)
                                for k, v in mc_results["win_rates"].items()},
            "score_mean":      {k: round(v, 6)
                                for k, v in mc_results["score_mean"].items()},
            "score_std":       {k: round(v, 6)
                                for k, v in mc_results["score_std"].items()},
            "stability_index": round(stability, 4),
            "verdict":         verdict,
        },
        "dependability_override": {
            "applied": override,
            "reason":  override_reason,
        },
        "best_model": best_model,
    }
 
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "mcdm_result.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[save]  MCDM result saved → '{out}'")
 
 

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(config_path: str) -> None:
    """Full MCDM pipeline for the detection module.
 
    Steps
    -----
    1. Load config and decision matrix from evaluate.py.
    2. Compute 2×2 AHP weights (Recall, Specificity).
    3. Monte Carlo robustness check using w_ahp.
    4. Dependability override check.
    5. Save result to results/mcdm_result.json.
 
    Parameters
    ----------
    config_path : str
        Path to config.yaml for the target network.
    """
    cfg = load_config(config_path)
 
    network     = cfg["network"]["name"]
    det_cfg     = cfg["detection"]
    results_dir = Path(det_cfg["results_dir"])
    mc_cfg      = det_cfg["monte_carlo"]
    spec_margin = float(det_cfg["mcdm"]["spec_margin"])
 
    print(f"\n{'='*62}")
    print(f"  Detection — MCDM pipeline  [{network}]")
    print(f"{'='*62}\n")
 
    decision_matrix, model_names, metric_names = load_decision_matrix(results_dir)
 
    # Step 1 — AHP weights
    print(f"\n{'─'*62}")
    print("  Step 1 — AHP weights  (Recall, Specificity)")
    print(f"{'─'*62}")
    pairwise_matrix = build_pairwise_matrix(det_cfg["ahp"]["pairwise_matrix"])
    w_ahp, CR       = ahp_weights(pairwise_matrix, metric_names)
 
    if CR >= 0.10:
        raise ValueError(
            f"AHP pairwise matrix inconsistent (CR={CR:.4f} ≥ 0.10). "
            "Review pairwise_matrix in config.yaml."
        )
 
    # Step 2 — Monte Carlo robustness check
    print(f"\n{'─'*62}")
    print("  Step 2 — Monte Carlo robustness analysis")
    print(f"{'─'*62}")
    mc_results = monte_carlo_robustness(
        decision_matrix = decision_matrix,
        w_combined      = w_ahp,
        metric_names    = metric_names,
        model_names     = model_names,
        n_simulations   = mc_cfg["n_simulations"],
        concentration   = mc_cfg["concentration"],
        random_seed     = mc_cfg["random_seed"],
    )
 
    nominal_winner  = mc_results["nominal_ranking"][0]
    stability_index = mc_results["stability_index"]
 
    # Step 3 — Dependability override
    print(f"\n{'─'*62}")
    print(f"  Step 3 — Dependability override  (spec_margin={spec_margin})")
    print(f"{'─'*62}")
    best_model, override, reason = dependability_override(
        decision_matrix = decision_matrix,
        model_names     = model_names,
        metric_names    = metric_names,
        nominal_winner  = nominal_winner,
        spec_margin     = spec_margin,
    )
 
    print(f"  Nominal winner : {nominal_winner}")
    print(f"  Override       : {'✅ YES' if override else '❌ NO'}")
    print(f"  Reason         : {reason}")
 
    print(f"\n{'='*62}")
    print(f"  Best model    : {best_model}")
    print(f"  Stability     : {stability_index:.4f}  "
          f"({'ROBUST' if stability_index >= 0.70 else 'MODERATE' if stability_index >= 0.50 else 'UNSTABLE'})")
    print(f"  Win rate      : {mc_results['win_rates'][nominal_winner]:.1f}%  (nominal)")
    print(f"{'='*62}")
 
    save_mcdm_result(
        w_ahp           = w_ahp,
        mc_results      = mc_results,
        metric_names    = metric_names,
        model_names     = model_names,
        best_model      = best_model,
        override        = override,
        override_reason = reason,
        results_dir     = results_dir,
    )
 
    print(f"\n[done]  MCDM pipeline complete for {network}.")
    print(f"        Results → '{results_dir}'\n")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MCDM pipeline (AHP + Monte Carlo) for the fault "
                    "detection module."
    )
    parser.add_argument(
        "--config",
        type     = str,
        required = True,
        help     = "Path to config.yaml (e.g. 'ieee13/config.yaml')",
    )
    args = parser.parse_args()
 
    try:
        main(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)