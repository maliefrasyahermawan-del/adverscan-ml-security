"""
AdverScan — Model Training & Evaluation
Trains Random Forest (primary) and Logistic Regression (surrogate / secondary).
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)


def train_random_forest(X_train, y_train,
                        n_estimators: int = 100,
                        max_depth: int = 10,
                        random_state: int = 42) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train,
                               C: float = 1.0,
                               random_state: int = 42) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, C=C, random_state=random_state,
                               class_weight="balanced")
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test) -> dict:
    """Returns accuracy, precision, recall, f1, detection_rate, confusion_matrix."""
    y_pred = model.predict(X_test)

    attack_mask = y_test == 1
    if attack_mask.sum() > 0:
        detection_rate = float((y_pred[attack_mask] == 1).mean())
    else:
        detection_rate = 0.0

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "detection_rate": detection_rate,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_test": int(len(y_test)),
        "n_attack": int(attack_mask.sum()),
    }


def save_model(model, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(model, path)
    print(f"  Model saved to {path}")


def load_model(path: str):
    return joblib.load(path)
