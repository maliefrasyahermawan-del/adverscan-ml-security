"""
AdverScan — Adversarial Attacker
Implements FGSM and PGD attacks via a Logistic Regression surrogate model.

Why surrogate?  Random Forest is non-differentiable.  We train an LR proxy on
the same data, compute exact gradients against it, then transfer the
perturbation to RF.  This is the standard black-box transferability attack.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Surrogate model
# ---------------------------------------------------------------------------

def train_surrogate(X_train, y_train, C: float = 0.5,
                    random_state: int = 42) -> LogisticRegression:
    """Train a white-box LR surrogate used to derive gradient directions."""
    surrogate = LogisticRegression(
        max_iter=2000, C=C, random_state=random_state, class_weight="balanced"
    )
    surrogate.fit(X_train, y_train)
    return surrogate


# ---------------------------------------------------------------------------
# Gradient computation
# ---------------------------------------------------------------------------

def _attack_gradient_sign(surrogate: LogisticRegression,
                           X: np.ndarray) -> np.ndarray:
    """
    Return the sign vector that decreases p(attack) for each sample.

    For binary LR with sigmoid σ:
        p(attack | x) = σ(w·x + b)
        ∂(-log p(benign)) / ∂x  =  p(attack) · w
    To minimise -log p(benign) (i.e. maximise p(benign)):
        step direction  =  -sign( p(attack) · w )

    Returns:  shape (n_samples, n_features) of {-1, 0, +1}
    """
    w = surrogate.coef_[0]                          # (n_features,)
    p_attack = surrogate.predict_proba(X)[:, 1]    # (n_samples,)
    grad = p_attack[:, None] * w[None, :]           # (n_samples, n_features)
    return -np.sign(grad)


# ---------------------------------------------------------------------------
# FGSM
# ---------------------------------------------------------------------------

def fgsm_attack(surrogate: LogisticRegression,
                X: np.ndarray,
                y: np.ndarray,
                epsilon: float,
                X_min: np.ndarray = None,
                X_max: np.ndarray = None) -> np.ndarray:
    """
    Fast Gradient Sign Method — single-step perturbation.
    Only attack samples (y == 1) are perturbed; benign samples are unchanged.
    """
    X_adv = X.copy()
    if epsilon == 0.0:
        return X_adv

    attack_mask = y == 1
    if attack_mask.sum() == 0:
        return X_adv

    X_atk = X[attack_mask]
    signs = _attack_gradient_sign(surrogate, X_atk)   # (n_atk, n_feat)
    X_adv[attack_mask] = X_atk + epsilon * signs

    if X_min is not None and X_max is not None:
        X_adv = np.clip(X_adv, X_min, X_max)

    return X_adv


# ---------------------------------------------------------------------------
# PGD
# ---------------------------------------------------------------------------

def pgd_attack(surrogate: LogisticRegression,
               X: np.ndarray,
               y: np.ndarray,
               epsilon: float = 0.3,
               alpha: float = 0.03,
               num_steps: int = 20,
               X_min: np.ndarray = None,
               X_max: np.ndarray = None,
               random_state: int = 0) -> np.ndarray:
    """
    Projected Gradient Descent — iterative multi-step attack.
    Stronger than FGSM because it takes many small steps with L-inf projection.
    """
    X_adv = X.copy()
    attack_mask = y == 1
    if attack_mask.sum() == 0 or epsilon == 0.0:
        return X_adv

    rng = np.random.default_rng(random_state)
    X_orig = X[attack_mask].copy()

    # Random initialisation within epsilon-ball
    X_iter = X_orig + rng.uniform(-epsilon * 0.1, epsilon * 0.1,
                                   X_orig.shape).astype(np.float32)

    for _ in range(num_steps):
        signs = _attack_gradient_sign(surrogate, X_iter)
        X_iter = X_iter + alpha * signs
        # Project back onto epsilon-ball around original
        delta = np.clip(X_iter - X_orig, -epsilon, epsilon)
        X_iter = X_orig + delta

    X_adv[attack_mask] = X_iter

    if X_min is not None and X_max is not None:
        X_adv = np.clip(X_adv, X_min, X_max)

    return X_adv


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_attack(model, X_adv: np.ndarray, y_true: np.ndarray) -> float:
    """Detection rate after attack = fraction of attack samples still detected."""
    attack_mask = y_true == 1
    if attack_mask.sum() == 0:
        return 1.0
    y_pred = model.predict(X_adv)
    return float((y_pred[attack_mask] == 1).mean())


def perturbation_stats(X_orig: np.ndarray, X_adv: np.ndarray,
                       y: np.ndarray = None) -> dict:
    """Summary statistics of the applied perturbation."""
    mask = (y == 1) if y is not None else np.ones(len(X_orig), dtype=bool)
    if mask.sum() == 0:
        return {}
    delta = X_adv[mask] - X_orig[mask]
    l2 = np.linalg.norm(delta, axis=1)
    linf = np.abs(delta).max(axis=1)
    return {
        "mean_l2": float(l2.mean()),
        "mean_linf": float(linf.mean()),
        "max_change": float(np.abs(delta).max()),
        "mean_abs_change": float(np.abs(delta).mean()),
        "n_perturbed": int(mask.sum()),
    }
