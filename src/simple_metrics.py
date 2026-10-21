"""
Minimal, pure-NumPy replacements for the sklearn.metrics functions used
in this project (accuracy_score, f1_score with macro averaging,
confusion_matrix). Avoids sklearn/scipy entirely, since their compiled
submodules (scipy.spatial, scipy.sparse, etc.) have repeatedly been
blocked by this machine's Windows Application Control policy -- this
module has no compiled dependencies at all.
"""

import numpy as np


def accuracy_score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, labels=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = sorted(labels)
    n = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm


def f1_score(y_true, y_pred, average="macro"):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))

    f1s = []
    for label in labels:
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    if average == "macro":
        return float(np.mean(f1s))
    raise NotImplementedError("Only average='macro' is implemented (all this project uses).")
