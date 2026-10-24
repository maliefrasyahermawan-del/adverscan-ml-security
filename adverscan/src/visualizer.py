"""
AdverScan — Visualizer
Generates and saves all experiment plots.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for headless runs)
import matplotlib.pyplot as plt
import seaborn as sns

_HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(_HERE, "..", "results", "plots")


def _style():
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except OSError:
        try:
            plt.style.use("seaborn-darkgrid")
        except OSError:
            pass
    plt.rcParams.update({"font.size": 12, "figure.dpi": 120})


# ---------------------------------------------------------------------------
# Plot 1 — Detection rate vs epsilon
# ---------------------------------------------------------------------------

def plot_accuracy_vs_epsilon(fgsm_results: dict, pgd_dr: float,
                              baseline_dr: float, adv_train_dr: float,
                              feat_squeeze_dr: float,
                              save: bool = True) -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=(10, 6))

    eps_vals = sorted(fgsm_results.keys())
    dr_vals = [fgsm_results[e] * 100 for e in eps_vals]

    ax.plot(eps_vals, dr_vals, "b-o", linewidth=2.5, markersize=9,
            label="FGSM Attack", zorder=5)

    ax.axhline(baseline_dr * 100, color="#27ae60", linewidth=2, linestyle="-",
               alpha=0.85, label=f"Baseline: {baseline_dr*100:.1f}%")
    ax.axhline(pgd_dr * 100, color="#e74c3c", linewidth=2, linestyle="--",
               label=f"PGD Attack (ε=0.3): {pgd_dr*100:.1f}%")
    ax.axhline(adv_train_dr * 100, color="#e67e22", linewidth=2, linestyle=":",
               label=f"After Adv. Training: {adv_train_dr*100:.1f}%")
    ax.axhline(feat_squeeze_dr * 100, color="#8e44ad", linewidth=2, linestyle="-.",
               label=f"After Feat. Squeezing: {feat_squeeze_dr*100:.1f}%")

    ax.set_xlabel("Perturbation Magnitude ε (normalised feature space)", fontsize=13)
    ax.set_ylabel("Attack Detection Rate (%)", fontsize=13)
    ax.set_title("FGSM Epsilon Sweep — Detection Rate Under Attack\n"
                 "AdverScan  |  ML-based IDS Vulnerability Demo", fontsize=14)
    ax.legend(fontsize=10, loc="lower left")
    ax.set_ylim(0, 110)
    ax.set_xlim(-0.02, 0.52)
    ax.grid(True, alpha=0.4)

    fig.tight_layout()
    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        path = os.path.join(PLOTS_DIR, "accuracy_vs_epsilon.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: accuracy_vs_epsilon.png")
    return fig


# ---------------------------------------------------------------------------
# Plot 2 & 3 — Confusion matrices
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm: np.ndarray, title: str,
                           filename: str, save: bool = True) -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=(7, 6))

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sums > 0, cm.astype(float) / row_sums * 100, 0)

    sns.heatmap(cm_pct, annot=False, cmap="Blues",
                xticklabels=["Benign", "Attack"],
                yticklabels=["Benign", "Attack"],
                ax=ax, vmin=0, vmax=100,
                cbar_kws={"label": "Row %"}, linewidths=0.5)

    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.38, f"{cm_pct[i,j]:.1f}%",
                    ha="center", va="center", fontsize=17, fontweight="bold",
                    color="white" if cm_pct[i, j] > 60 else "black")
            ax.text(j + 0.5, i + 0.65, f"({cm[i,j]:,})",
                    ha="center", va="center", fontsize=11, color="gray")

    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title(title, fontsize=13, pad=10)

    fig.tight_layout()
    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        path = os.path.join(PLOTS_DIR, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {filename}")
    return fig


# ---------------------------------------------------------------------------
# Plot 4 — Defense comparison bar chart
# ---------------------------------------------------------------------------

def plot_defense_comparison(baseline_dr: float, attack_dr: float,
                             adv_train_dr: float, feat_squeeze_dr: float,
                             save: bool = True) -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = ["Baseline\n(No Attack)",
              "Under FGSM\nAttack (ε=0.2)",
              "Adv. Training\nDefense",
              "Feature Squeezing\nDefense"]
    values = [baseline_dr * 100, attack_dr * 100,
              adv_train_dr * 100, feat_squeeze_dr * 100]
    colors = ["#27ae60", "#e74c3c", "#2980b9", "#8e44ad"]

    bars = ax.bar(labels, values, color=colors, edgecolor="black",
                  linewidth=0.8, width=0.55)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.8,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=14, fontweight="bold")

    ax.set_ylabel("Attack Detection Rate (%)", fontsize=13)
    ax.set_title("Defense Effectiveness Comparison\n"
                 "AdverScan  |  Adversarial Robustness of ML-based IDS", fontsize=14)
    ax.set_ylim(0, 118)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.5,
               label="50% reference")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=10)

    fig.tight_layout()
    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        path = os.path.join(PLOTS_DIR, "defense_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: defense_comparison.png")
    return fig


# ---------------------------------------------------------------------------
# Master function — generate all 4 plots
# ---------------------------------------------------------------------------

def generate_all_plots(results: dict,
                        rf_model=None,
                        X_test=None,
                        y_test=None,
                        surrogate=None,
                        X_min=None,
                        X_max=None):
    """Generate and save all 4 experiment plots."""
    from sklearn.metrics import confusion_matrix as sk_cm
    from src.attacker import fgsm_attack

    os.makedirs(PLOTS_DIR, exist_ok=True)

    fgsm_res = results["fgsm_sweep"]
    baseline_dr = results["baseline"]["detection_rate"]
    pgd_dr = results["pgd"]["detection_rate"]
    adv_dr = results["adversarial_training"]["detection_rate_under_attack"]
    sq_dr = results["feature_squeezing"]["detection_rate_under_attack"]

    # Plot 1
    fig1 = plot_accuracy_vs_epsilon(fgsm_res, pgd_dr, baseline_dr, adv_dr, sq_dr)
    plt.close(fig1)

    # Plots 2 & 3 need model + data
    if rf_model is not None and X_test is not None and y_test is not None:
        y_pred_base = rf_model.predict(X_test)
        cm_base = sk_cm(y_test, y_pred_base)
        fig2 = plot_confusion_matrix(
            cm_base,
            "Confusion Matrix — Baseline (No Attack)",
            "confusion_matrix_baseline.png",
        )
        plt.close(fig2)

        # Under FGSM attack ε=0.2
        X_adv_02 = fgsm_attack(surrogate, X_test, y_test, 0.2, X_min, X_max)
        y_pred_adv = rf_model.predict(X_adv_02)
        cm_adv = sk_cm(y_test, y_pred_adv)
        fig3 = plot_confusion_matrix(
            cm_adv,
            "Confusion Matrix — Under FGSM Attack (ε=0.2)\n"
            "False Negatives spike: attack traffic classified as benign",
            "confusion_matrix_under_attack.png",
        )
        plt.close(fig3)

    # Plot 4
    attack_dr_02 = fgsm_res.get(0.2, list(fgsm_res.values())[-2])
    fig4 = plot_defense_comparison(baseline_dr, attack_dr_02, adv_dr, sq_dr)
    plt.close(fig4)

    print("  All 4 plots generated.")
