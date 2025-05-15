import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix
)

def cout_metier(y_true, y_pred, cout_FN=10, cout_FP=1):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn * cout_FN + fp * cout_FP

def find_best_threshold(y_true, y_proba, cout_FN=10, cout_FP=1):
    thresholds = np.linspace(0, 1, 100)
    best_cost = float("inf")
    best_threshold = 0.5
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        cost = cout_metier(y_true, y_pred, cout_FN, cout_FP)
        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold
    return best_threshold

def compute_metrics(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "AUC": roc_auc_score(y_true, y_proba),
        "F1": f1_score(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred)
    }