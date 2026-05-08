"""
ieee5/classification/mcdm.py
-----------------------------
Runs the MCDM pipeline to select the best classification model from the
three candidates (RF, XGBoost, kNN) evaluated by evaluate.py.

MCDM pipeline
-------------
1. Load decision matrix (weighted recall on test set) from evaluate.py.
2. Select nominal winner — model with highest weighted recall.
3. Monte Carlo robustness check — subsample cls_train N times, refit
   each model, re-evaluate weighted recall, report win rates and
   stability index.
4. Best model = model with highest Monte Carlo win rate.
   If the nominal winner has the highest win rate, it is confirmed.
   If another model dominates in win rate, it overrides the nominal.
   Ties in win rate fall back to the nominal winner.

Design notes
------------
- No AHP: weighted recall is the single criterion — nothing to weight.
- No dependability override: the asymmetric FN/FP logic of detection
  does not apply to multiclass fault classification.
- Monte Carlo perturbs the training dataset (subsample_rate from config),
  not the weight vector. This measures sensitivity of the ranking to
  training data variation, not to subjective weight choices.
- Best hyperparameters are loaded ONCE before the simulation loop to
  avoid redundant I/O on every iteration.

Inputs
------
- results/decision_matrix.json      : produced by evaluate.py
- data/splits/<network>/cls_train.csv : subsampled in each MC simulation
- data/splits/<network>/cls_test.csv  : fixed evaluation set
- config.yaml                        : Monte Carlo settings + model configs

Outputs (results/)
------------------
- mcdm_result.json  : nominal winner, MC win rates, stability, best model

Utility modules used
---------------------
- utils.io.load_config()  : load YAML config
- utils.io.load_split()   : load split CSV → (X, y)
- utils.io.load_model()   : load fitted model from .pkl

Usage (called from project root)
---------------------------------
    python3 ieee5/classification/mcdm.py --config ieee5/config.yaml
"""

import argparse
import json
import sys
import warnings
import joblib
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Add project root to path so utils/ is importable regardless of cwd
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.io import load_config, load_split, load_model   # shared I/O helpers

warnings.filterwarnings("ignore")


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
            f"Decision matrix not found: '{path}'\n"
            "Run ieee5/classification/evaluate.py first."
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
# Monte Carlo robustness — dataset perturbation
# ---------------------------------------------------------------------------

def _refit_model(
    model_key: str,
    params:    dict,
    X_sub:     np.ndarray,
    y_sub:     np.ndarray,
):
    """Refit a model on a training subsample using pre-loaded best params.

    Parameters
    ----------
    model_key : str
        One of 'random_forest', 'xgboost', 'knn'.
    params : dict
        Best hyperparameters extracted from the fitted .pkl model.
        Loaded once before the simulation loop — not reloaded here.
    X_sub : np.ndarray
        Subsampled training features.
    y_sub : np.ndarray
        Subsampled training labels.

    Returns
    -------
    fitted estimator
    """
    if model_key == "random_forest":
        model = RandomForestClassifier(**params)
    elif model_key == "xgboost":
        model = XGBClassifier(**params)
    elif model_key == "knn":
        model = KNeighborsClassifier(**params)
    else:
        raise ValueError(f"Unknown model key: '{model_key}'")

    model.fit(X_sub, y_sub)
    return model


def monte_carlo_cls_robustness(
    cfg:          dict,
    model_keys:   list[str],
    model_names:  list[str],
    X_train:      np.ndarray,
    y_train:      np.ndarray,
    X_test:       np.ndarray,
    y_test:       np.ndarray,
) -> dict:
    """Assess ranking robustness by subsampling the training dataset.

    In each simulation, cls_train is subsampled at subsample_rate,
    each model is refitted on the subsample using its best
    hyperparameters, and weighted recall is evaluated on the fixed
    test set. Win rate and stability index are reported.

    Best hyperparameters are loaded ONCE before the loop to avoid
    redundant I/O across 1,000+ simulations.

    Perturbation method: stratified random subsampling without
    replacement. StratifiedKFold is not used here — each simulation
    is an independent subsample to introduce maximum variability.

    Parameters
    ----------
    cfg : dict
        Full parsed config dictionary.
    model_keys : list[str]
        Keys in the classification section of config.yaml.
    model_names : list[str]
        Short model identifiers in the same order as model_keys.
    X_train : np.ndarray
        Full cls_train feature matrix.
    y_train : np.ndarray
        Full cls_train labels.
    X_test : np.ndarray
        Fixed test feature matrix — never perturbed.
    y_test : np.ndarray
        Fixed test labels.

    Returns
    -------
    dict
        Keys: nominal_ranking, nominal_scores, win_rates,
              stability_index, score_mean, score_std, score_matrix.
    """
    mc_cfg         = cfg["classification"]["monte_carlo"]
    n_simulations  = int(mc_cfg["n_simulations"])
    subsample_rate = float(mc_cfg["subsample_rate"])
    random_state   = int(mc_cfg["random_state"])

    n_models = len(model_keys)
    rng      = np.random.default_rng(random_state)

    # Load LabelEncoder for XGBoost (if present)
    models_dir = Path(cfg["classification"]["models_dir"])
    le_path    = models_dir / "label_encoder.pkl"
    le         = joblib.load(le_path) if le_path.exists() else None

    def _decode(model_key: str, y_pred: np.ndarray) -> np.ndarray:
        if model_key == "xgboost" and le is not None:
            return le.inverse_transform(y_pred)
        return y_pred

    # Load best params ONCE — avoids redundant I/O inside the loop
    best_params = {}
    for model_key in model_keys:
        model_id              = cfg["classification"][model_key]["model_id"]
        best                  = load_model(models_dir, model_id)
        best_params[model_key] = best.get_params()

    # Nominal scores — weighted recall on full test set (recomputed for consistency)
    nominal_scores = np.zeros(n_models)
    for i, model_key in enumerate(model_keys):
        model_id     = cfg["classification"][model_key]["model_id"]
        model        = load_model(models_dir, model_id)
        y_pred       = model.predict(X_test)
        y_pred       = _decode(model_key, y_pred)
        nominal_scores[i] = recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        )

    nominal_order   = np.argsort(nominal_scores)[::-1]
    nominal_ranking = [model_names[i] for i in nominal_order]

    # Monte Carlo simulations
    score_matrix = np.zeros((n_simulations, n_models))

    print(f"\n  Running {n_simulations:,} simulations "
          f"(subsample_rate={subsample_rate:.0%})...")

    for sim in range(n_simulations):
        # Stratified subsample — preserve class distribution
        indices = []
        for cls in np.unique(y_train):
            cls_idx = np.where(y_train == cls)[0]
            n_cls   = max(1, int(len(cls_idx) * subsample_rate))
            sampled = rng.choice(cls_idx, size=n_cls, replace=False)
            indices.append(sampled)
        indices = np.concatenate(indices)

        X_sub = X_train[indices]
        y_sub = y_train[indices]

        for i, model_key in enumerate(model_keys):
            if model_key == "xgboost" and le is not None:
                y_sub_fit = le.transform(y_sub)
            else:
                y_sub_fit = y_sub

            # Use pre-loaded params — no load_model() call inside the loop
            model  = _refit_model(model_key, best_params[model_key], X_sub, y_sub_fit)
            y_pred = model.predict(X_test)
            y_pred = _decode(model_key, y_pred)
            score_matrix[sim, i] = recall_score(
                y_test, y_pred, average="weighted", zero_division=0
            )

        if (sim + 1) % 10 == 0:
            print(f"  [{sim + 1:>4}/{n_simulations}] simulations complete")

    # Win rates
    winners    = np.argmax(score_matrix, axis=1)
    win_counts = np.bincount(winners, minlength=n_models)
    win_rates  = {
        model_names[i]: float(win_counts[i] / n_simulations * 100)
        for i in range(n_models)
    }

    # Stability index — fraction of simulations matching nominal ranking
    rank_matrix     = np.argsort(score_matrix, axis=1)[:, ::-1]
    matches         = np.all(rank_matrix == nominal_order, axis=1)
    stability_index = float(matches.sum() / n_simulations)

    # Score statistics
    score_mean = {
        model_names[i]: float(score_matrix[:, i].mean())
        for i in range(n_models)
    }
    score_std = {
        model_names[i]: float(score_matrix[:, i].std())
        for i in range(n_models)
    }

    _print_mc_report(
        nominal_ranking = nominal_ranking,
        nominal_scores  = nominal_scores,
        nominal_order   = nominal_order,
        win_rates       = win_rates,
        stability_index = stability_index,
        score_mean      = score_mean,
        score_std       = score_std,
        n_simulations   = n_simulations,
        subsample_rate  = subsample_rate,
        model_names     = model_names,
    )

    return {
        "nominal_ranking": nominal_ranking,
        "nominal_scores":  nominal_scores.tolist(),
        "win_rates":       win_rates,
        "stability_index": stability_index,
        "score_mean":      score_mean,
        "score_std":       score_std,
        "score_matrix":    score_matrix,
    }


def _print_mc_report(
    nominal_ranking: list[str],
    nominal_scores:  np.ndarray,
    nominal_order:   np.ndarray,
    win_rates:       dict,
    stability_index: float,
    score_mean:      dict,
    score_std:       dict,
    n_simulations:   int,
    subsample_rate:  float,
    model_names:     list[str],
) -> None:
    """Print a formatted Monte Carlo robustness summary to stdout."""
    sep = "=" * 62

    print(f"\n{sep}")
    print("  MONTE CARLO ROBUSTNESS ANALYSIS  (dataset perturbation)")
    print(f"  Simulations : {n_simulations:,}   "
          f"Subsample rate : {subsample_rate:.0%}")
    print(sep)

    print(f"\n  Nominal ranking (full cls_train, no perturbation):")
    print(f"  {'-'*50}")
    for rank, name in enumerate(nominal_ranking, start=1):
        idx   = model_names.index(name)
        score = nominal_scores[idx]
        bar   = "█" * int(score * 60)
        print(f"  #{rank}  {name:<10s}  weighted_recall = {score:.6f}  {bar}")

    print(f"\n  Win rate per model ({n_simulations:,} simulations):")
    print(f"  {'-'*50}")
    for name in nominal_ranking:
        wr  = win_rates[name]
        bar = "█" * int(wr / 2)
        print(f"  {name:<10s}  {wr:6.2f}%  {bar}")

    print(f"\n  Score statistics (mean ± std):")
    print(f"  {'-'*50}")
    for name in nominal_ranking:
        print(f"  {name:<10s}  "
              f"{score_mean[name]:.6f} ± {score_std[name]:.6f}")

    pct = stability_index * 100
    if stability_index >= 0.70:
        verdict = "✅  ROBUST   — ranking stable under data variation"
    elif stability_index >= 0.50:
        verdict = "⚠️   MODERATE — ranking changes in some scenarios"
    else:
        verdict = "❌  UNSTABLE — ranking sensitive to training data"

    print(f"\n  Stability index : {stability_index:.4f}  ({pct:.1f}% of simulations)")
    print(f"  Verdict         : {verdict}\n")


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_mcdm_result(
    mc_results:       dict,
    model_names:      list[str],
    best_model:       str,
    selection_reason: str,
    results_dir:      Path,
) -> None:
    """Save the MCDM result to JSON.

    Parameters
    ----------
    mc_results : dict
        Output of monte_carlo_cls_robustness().
    model_names : list[str]
    best_model : str
        Final selected model.
    selection_reason : str
        Human-readable explanation of the selection decision.
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
        "model_names":  model_names,
        "metric":       "weighted_recall",
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
        "selection_reason": selection_reason,
        "best_model":       best_model,
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
    """Full MCDM pipeline for the classification module.

    Steps
    -----
    1. Load config, decision matrix, cls_train and cls_test.
    2. Identify nominal winner (highest weighted recall on full test set).
    3. Monte Carlo robustness — subsample cls_train, refit, re-evaluate.
    4. Best model = model with highest Monte Carlo win rate.
       If the nominal winner has the highest win rate, it is confirmed.
       If another model dominates in win rate, it overrides the nominal.
       Ties in win rate fall back to the nominal winner.
    5. Save result to results/mcdm_result.json.

    Parameters
    ----------
    config_path : str
        Path to config.yaml for the target network.
    """
    cfg = load_config(config_path)

    network     = cfg["network"]["name"]
    splits_dir  = Path(cfg["data"]["splits_dir"])
    cls_cfg     = cfg["classification"]
    results_dir = Path(cls_cfg["results_dir"])
    label_col   = "label_classif"
    model_keys  = ["random_forest", "xgboost", "knn"]

    print(f"\n{'='*62}")
    print(f"  Classification — MCDM pipeline  [{network}]")
    print(f"{'='*62}\n")

    # Load decision matrix
    decision_matrix, model_names, metric_names = load_decision_matrix(results_dir)

    # Load training and test data for Monte Carlo
    X_train, y_train = load_split(str(splits_dir / "cls_train.csv"), label_col)
    X_test,  y_test  = load_split(str(splits_dir / "cls_test.csv"),  label_col)

    # Step 1 — Nominal winner (highest weighted recall on full test set)
    print(f"\n{'─'*62}")
    print("  Step 1 — Nominal winner  (weighted recall on test set)")
    print(f"{'─'*62}")
    scores         = decision_matrix[:, 0]
    nominal_idx    = int(np.argmax(scores))
    nominal_winner = model_names[nominal_idx]

    for name, score in zip(model_names, scores):
        marker = "  ←  nominal winner" if name == nominal_winner else ""
        print(f"  {name:<8}  weighted_recall = {score:.6f}{marker}")

    # Step 2 — Monte Carlo robustness
    print(f"\n{'─'*62}")
    print("  Step 2 — Monte Carlo robustness  (dataset perturbation)")
    print(f"{'─'*62}")
    mc_results = monte_carlo_cls_robustness(
        cfg         = cfg,
        model_keys  = model_keys,
        model_names = model_names,
        X_train     = X_train,
        y_train     = y_train,
        X_test      = X_test,
        y_test      = y_test,
    )

    # Step 3 — Best model: win rate como criterio de selección final
    print(f"\n{'─'*62}")
    print("  Step 3 — Best model selection  (win rate)")
    print(f"{'─'*62}")

    top_win_rate   = max(mc_results["win_rates"].values())
    top_by_winrate = [m for m, wr in mc_results["win_rates"].items()
                      if wr == top_win_rate]

    if nominal_winner in top_by_winrate:
        best_model       = nominal_winner
        selection_reason = (
            f"nominal winner confirmed by Monte Carlo "
            f"(win rate={top_win_rate:.1f}%)"
        )
        print(f"  ✅  Nominal winner '{best_model}' confirmed by win rate "
              f"({top_win_rate:.1f}%).")
    else:
        # Empate en win rate también → fallback al nominal winner
        best_model       = top_by_winrate[0]
        selection_reason = (
            f"win rate override: '{best_model}' "
            f"({mc_results['win_rates'][best_model]:.1f}%) > "
            f"'{nominal_winner}' "
            f"({mc_results['win_rates'][nominal_winner]:.1f}%)"
        )
        print(f"  ⚠️  Win rate override applied.")
        print(f"      '{best_model}' ({mc_results['win_rates'][best_model]:.1f}%) "
              f"> '{nominal_winner}' ({mc_results['win_rates'][nominal_winner]:.1f}%)")

    stability_index = mc_results["stability_index"]

    if stability_index < 0.50:
        print(f"\n  ⚠️  WARNING: stability index = {stability_index:.4f} < 0.50.")
        print(f"     The ranking is sensitive to training data variation.")
        print(f"     Review per-class recall in metrics_*.json before deploying.")

    print(f"\n{'='*62}")
    print(f"  Best model       : {best_model}")
    print(f"  Selection reason : {selection_reason}")
    print(f"  Stability        : {stability_index:.4f}  "
          f"({'ROBUST' if stability_index >= 0.70 else 'MODERATE' if stability_index >= 0.50 else 'UNSTABLE'})")
    print(f"  Win rate         : {mc_results['win_rates'][best_model]:.1f}%")
    print(f"{'='*62}")

    save_mcdm_result(
        mc_results       = mc_results,
        model_names      = model_names,
        best_model       = best_model,
        selection_reason = selection_reason,
        results_dir      = results_dir,
    )

    print(f"\n[done]  MCDM pipeline complete for {network}.")
    print(f"        Results → '{results_dir}'\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MCDM pipeline (Monte Carlo) for the fault "
                    "classification module."
    )
    parser.add_argument(
        "--config",
        type     = str,
        required = True,
        help     = "Path to config.yaml (e.g. 'ieee5/config.yaml')",
    )
    args = parser.parse_args()

    try:
        main(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)