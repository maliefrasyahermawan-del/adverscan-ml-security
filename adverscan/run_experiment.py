#!/usr/bin/env python3
"""
AdverScan - Entry Point
Run the full adversarial IDS experiment end-to-end.

Usage:
    python run_experiment.py
    python run_experiment.py --data path/to/cicids2017_wednesday.csv
"""

import os
import sys
import time
import json
import argparse
import numpy as np

# Force UTF-8 output so Unicode box-drawing / arrows render on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def _banner():
    print("\n" + "=" * 64)
    print("  AdverScan - Adversarial ML Lab")
    print("  Adversarial Attacks on ML-based Intrusion Detection Systems")
    print("=" * 64)


def _jsonify(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    parser = argparse.ArgumentParser(description="AdverScan experiment runner")
    parser.add_argument("--data", default=None,
                        help="Path to CICIDS2017 CSV (default: auto-detect)")
    args = parser.parse_args()

    _banner()
    t0 = time.time()

    # ---------------------------------------------------------------- 1. Data
    print("\n[1/7] Loading dataset...")
    from src.data_pipeline import load_data, auto_detect_dataset, get_feature_bounds

    data_path = args.data
    if data_path is None:
        data_path, dataset_type = auto_detect_dataset()
    else:
        dataset_type = "user-supplied"

    X_train, X_test, y_train, y_test, scaler = load_data(data_path, n_features=20)
    X_min, X_max = get_feature_bounds(X_train)

    fname = os.path.basename(data_path)
    n_total = len(X_train) + len(X_test)
    n_attack = int((y_test == 1).sum()) + int((y_train == 1).sum())
    print(f"  Dataset : {fname}  ({n_total:,} rows, {n_attack:,} attack, "
          f"{n_total - n_attack:,} benign)")

    # ------------------------------------------------------------ 2. Training
    print("\n[2/7] Training baseline models...")
    from src.model import train_random_forest, evaluate
    from src.attacker import train_surrogate

    rf = train_random_forest(X_train, y_train)
    surrogate = train_surrogate(X_train, y_train)
    print(f"  Random Forest : {rf.n_estimators} trees, max_depth={rf.max_depth}")
    print("  LR Surrogate  : trained as white-box proxy for gradient computation")

    # --------------------------------------------------------- 3. Baseline
    print("\n[3/7] Baseline evaluation (no attack)...")
    baseline = evaluate(rf, X_test, y_test)
    print(f"  Accuracy       : {baseline['accuracy']*100:.1f}%")
    print(f"  Detection Rate : {baseline['detection_rate']*100:.1f}%")
    print(f"  F1 Score       : {baseline['f1']*100:.1f}%")

    # ------------------------------------------------------- 4. FGSM sweep
    print("\n[4/7] FGSM attack sweep (epsilon 0.0 to 0.5)...")
    from src.attacker import fgsm_attack, evaluate_attack

    EPSILONS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    fgsm_results = {}
    for eps in EPSILONS:
        X_adv = fgsm_attack(surrogate, X_test, y_test, eps, X_min, X_max)
        dr = evaluate_attack(rf, X_adv, y_test)
        fgsm_results[eps] = float(dr)
        drop = "  <- drop!" if eps >= 0.1 and dr < baseline["detection_rate"] * 0.85 else ""
        print(f"  eps={eps:.2f} -> Detection rate: {dr*100:.1f}%{drop}")

    # ------------------------------------------------------- 5. PGD
    print("\n[5/7] PGD attack (eps=0.3, alpha=0.03, 20 steps)...")
    from src.attacker import pgd_attack

    X_pgd = pgd_attack(surrogate, X_test, y_test,
                        epsilon=0.3, alpha=0.03, num_steps=20,
                        X_min=X_min, X_max=X_max)
    pgd_dr = evaluate_attack(rf, X_pgd, y_test)
    print(f"  PGD -> Detection rate: {pgd_dr*100:.1f}%")

    # ------------------------------------------------------- 6. Defense
    print("\n[6/7] Defense evaluation...")
    from src.defender import adversarial_training, feature_squeezing_detection_rate
    from sklearn.ensemble import RandomForestClassifier

    # Adversarial training
    print("  Training robust model (adversarial training)...")
    robust_rf = adversarial_training(
        RandomForestClassifier, X_train, y_train, surrogate,
        epsilon=0.1, X_min=X_min, X_max=X_max,
        n_estimators=100, max_depth=10, n_jobs=-1,
    )
    X_adv_test = fgsm_attack(surrogate, X_test, y_test, 0.1, X_min, X_max)
    adv_dr = evaluate_attack(robust_rf, X_adv_test, y_test)
    print(f"  Adv. Training   -> {adv_dr*100:.1f}% detection under attack (eps=0.1)")

    # Feature squeezing
    squeeze_dr = feature_squeezing_detection_rate(rf, X_adv_test, y_test, n_bits=8)
    print(f"  Feat. Squeezing -> {squeeze_dr*100:.1f}% detection under attack (eps=0.1)")

    # ------------------------------------------------------- 7. Plots
    print("\n[7/7] Generating plots...")
    from src.visualizer import generate_all_plots

    results = {
        "baseline": baseline,
        "fgsm_sweep": fgsm_results,
        "pgd": {"epsilon": 0.3, "detection_rate": float(pgd_dr)},
        "adversarial_training": {"detection_rate_under_attack": float(adv_dr)},
        "feature_squeezing": {"detection_rate_under_attack": float(squeeze_dr)},
    }

    generate_all_plots(results, rf, X_test, y_test, surrogate, X_min, X_max)

    # Save JSON
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "experiment_results.json")
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(_jsonify(results), fh, indent=2)

    elapsed = time.time() - t0

    # ----------------------------------------------------------- Summary
    print("\n" + "=" * 64)
    print(f"  Results saved -> results/experiment_results.json")
    print(f"  Plots saved   -> results/plots/  (4 PNG files)")
    print(f"  Runtime       : {elapsed:.1f}s")
    print("=" * 64)
    print()
    print("  Key findings:")
    print(f"    Baseline detection rate       : {baseline['detection_rate']*100:.1f}%")
    print(f"    After FGSM (eps=0.2)          : {fgsm_results.get(0.2, 0)*100:.1f}%")
    print(f"    After PGD  (eps=0.3)          : {pgd_dr*100:.1f}%")
    print(f"    Recovery - Adv. Training      : {adv_dr*100:.1f}%")
    print(f"    Recovery - Feat. Squeezing    : {squeeze_dr*100:.1f}%")
    print()
    print("  Next steps:")
    print("    jupyter notebook  ->  open notebooks/05_full_report.ipynb")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
