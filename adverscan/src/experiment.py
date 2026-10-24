"""
AdverScan — Full Experiment Orchestrator
Runs the complete pipeline: load → train → attack → defend → save results.
"""

import os
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "..", "results")
EPSILONS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]


def _jsonify(obj):
    """Recursively convert numpy scalars/arrays to native Python types."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def run_full_experiment(data_path: str = None, verbose: bool = True) -> dict:
    """
    Execute the full AdverScan experiment and return a results dict.
    All intermediate artefacts are also saved to results/.
    """
    os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)

    from src.data_pipeline import load_data, auto_detect_dataset, get_feature_bounds
    from src.model import train_random_forest, evaluate
    from src.attacker import (train_surrogate, fgsm_attack, pgd_attack,
                               evaluate_attack)
    from src.defender import (adversarial_training, feature_squeezing_detection_rate)

    results = {}

    # ------------------------------------------------------------------ data
    if verbose:
        print("\n[1/6] Loading dataset…")
    if data_path is None:
        data_path, _ = auto_detect_dataset()
    X_train, X_test, y_train, y_test, scaler = load_data(
        data_path, n_features=20
    )
    X_min, X_max = get_feature_bounds(X_train)
    results["dataset"] = {
        "path": os.path.basename(data_path),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
    }

    # ---------------------------------------------------------------- models
    if verbose:
        print("\n[2/6] Training baseline models…")
    rf = train_random_forest(X_train, y_train)
    surrogate = train_surrogate(X_train, y_train)

    # -------------------------------------------------------- baseline eval
    if verbose:
        print("\n[3/6] Baseline evaluation…")
    baseline = evaluate(rf, X_test, y_test)
    results["baseline"] = baseline
    if verbose:
        print(f"  Accuracy: {baseline['accuracy']*100:.1f}%  |  "
              f"Detection Rate: {baseline['detection_rate']*100:.1f}%")

    # -------------------------------------------------------- FGSM sweep
    if verbose:
        print("\n[4/6] FGSM attack sweep…")
    fgsm_results = {}
    for eps in EPSILONS:
        X_adv = fgsm_attack(surrogate, X_test, y_test, eps, X_min, X_max)
        dr = evaluate_attack(rf, X_adv, y_test)
        fgsm_results[eps] = float(dr)
        if verbose:
            drop = f"  ← drop!" if eps > 0 and dr < baseline["detection_rate"] * 0.8 else ""
            print(f"  ε={eps:.2f} → Detection rate: {dr*100:.1f}%{drop}")
    results["fgsm_sweep"] = fgsm_results

    # ----------------------------------------------------------- PGD
    if verbose:
        print("\n[5/6] PGD attack (ε=0.3, 20 steps)…")
    X_pgd = pgd_attack(surrogate, X_test, y_test,
                        epsilon=0.3, alpha=0.03, num_steps=20,
                        X_min=X_min, X_max=X_max)
    pgd_dr = evaluate_attack(rf, X_pgd, y_test)
    results["pgd"] = {"epsilon": 0.3, "alpha": 0.03,
                      "steps": 20, "detection_rate": float(pgd_dr)}
    if verbose:
        print(f"  PGD → Detection rate: {pgd_dr*100:.1f}%")

    # ---------------------------------------------------------- defense
    if verbose:
        print("\n[6/6] Defense evaluation…")

    # Defense 1 — adversarial training
    robust_rf = adversarial_training(
        RandomForestClassifier, X_train, y_train, surrogate,
        epsilon=0.1, X_min=X_min, X_max=X_max,
        n_estimators=100, max_depth=10, n_jobs=-1,
    )
    X_adv_test = fgsm_attack(surrogate, X_test, y_test, 0.1, X_min, X_max)
    adv_train_dr = evaluate_attack(robust_rf, X_adv_test, y_test)
    results["adversarial_training"] = {
        "train_epsilon": 0.1,
        "detection_rate_under_attack": float(adv_train_dr),
    }

    # Defense 2 — feature squeezing
    squeeze_dr = feature_squeezing_detection_rate(rf, X_adv_test, y_test, n_bits=8)
    results["feature_squeezing"] = {
        "n_bits": 8,
        "detection_rate_under_attack": float(squeeze_dr),
    }

    if verbose:
        print(f"  Adversarial Training → {adv_train_dr*100:.1f}% detection under attack")
        print(f"  Feature Squeezing    → {squeeze_dr*100:.1f}% detection under attack")

    # -------------------------------------------------------------- save
    out_path = os.path.join(RESULTS_DIR, "experiment_results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(_jsonify(results), fh, indent=2)

    if verbose:
        print(f"\n  Results saved → {out_path}")

    return results, rf, surrogate, X_train, X_test, y_train, y_test, X_min, X_max
