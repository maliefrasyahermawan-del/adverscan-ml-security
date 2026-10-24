"""
AdverScan — Defense Module
Defense 1: Adversarial Training  — retrain on augmented data
Defense 2: Feature Squeezing     — quantise features to remove fine perturbations
Defense 3: Ensemble Defense      — majority vote over diverse RF models
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Defense 1 — Adversarial Training
# ---------------------------------------------------------------------------

def adversarial_training(
    model_class,
    X_train: np.ndarray,
    y_train: np.ndarray,
    surrogate,
    epsilon: float = 0.1,
    X_min: np.ndarray = None,
    X_max: np.ndarray = None,
    random_state: int = 42,
    **model_kwargs,
):
    """
    Augment training data with FGSM adversarial examples, then retrain.
    The new model is more robust because it has seen perturbed attack traffic.
    """
    from src.attacker import fgsm_attack  # avoid circular import

    X_adv_train = fgsm_attack(surrogate, X_train, y_train, epsilon, X_min, X_max)

    X_aug = np.vstack([X_train, X_adv_train])
    y_aug = np.concatenate([y_train, y_train])

    # Shuffle
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(len(X_aug))
    X_aug, y_aug = X_aug[idx], y_aug[idx]

    model = model_class(random_state=random_state, **model_kwargs)
    model.fit(X_aug, y_aug)
    return model


# ---------------------------------------------------------------------------
# Defense 2 — Feature Squeezing
# ---------------------------------------------------------------------------

def round_to_bits(X: np.ndarray, n_bits: int = 8) -> np.ndarray:
    """
    Quantise each feature to n_bits precision within its observed range.
    Fine-grained adversarial perturbations are rounded away.
    """
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    span = X_max - X_min
    span = np.where(span < 1e-10, 1.0, span)

    levels = 2 ** n_bits
    X_norm = (X - X_min) / span
    X_quantised = np.round(X_norm * levels) / levels
    return X_quantised * span + X_min


def feature_squeezing_predict(model, X: np.ndarray, n_bits: int = 8) -> np.ndarray:
    """Run inference after squeezing features."""
    X_sq = round_to_bits(X, n_bits)
    return model.predict(X_sq)


def feature_squeezing_detection_rate(model, X_adv: np.ndarray,
                                      y_true: np.ndarray,
                                      n_bits: int = 8) -> float:
    """Detection rate on adversarial examples after applying feature squeezing."""
    y_pred = feature_squeezing_predict(model, X_adv, n_bits)
    attack_mask = y_true == 1
    if attack_mask.sum() == 0:
        return 0.0
    return float((y_pred[attack_mask] == 1).mean())


# ---------------------------------------------------------------------------
# Defense 3 — Ensemble Defense (bonus)
# ---------------------------------------------------------------------------

def train_ensemble(X_train: np.ndarray, y_train: np.ndarray,
                   n_models: int = 3, random_state: int = 42):
    """Train N RF models with different hyper-parameters."""
    configs = [
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 150, "max_depth": 8},
        {"n_estimators": 80,  "max_depth": 12},
        {"n_estimators": 120, "max_depth": 6},
    ]
    models = []
    for i, cfg in enumerate(configs[:n_models]):
        m = RandomForestClassifier(
            n_jobs=-1, random_state=random_state + i,
            class_weight="balanced", **cfg
        )
        m.fit(X_train, y_train)
        models.append(m)
    return models


def ensemble_predict(models: list, X: np.ndarray) -> np.ndarray:
    """Majority vote from an ensemble."""
    votes = np.array([m.predict(X) for m in models])   # (n_models, n_samples)
    majority = (votes.sum(axis=0) >= (len(models) / 2)).astype(int)
    return majority
