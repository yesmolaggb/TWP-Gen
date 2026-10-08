"""Standard external clustering metrics used by the DuEE experiments."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def contingency(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    true_values = np.unique(y_true)
    pred_values = np.unique(y_pred)
    true_index = {value: index for index, value in enumerate(true_values)}
    pred_index = {value: index for index, value in enumerate(pred_values)}
    matrix = np.zeros((len(true_values), len(pred_values)), dtype=np.int64)
    for true_value, pred_value in zip(y_true, y_pred):
        matrix[true_index[true_value], pred_index[pred_value]] += 1
    return matrix


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    matrix = contingency(y_true, y_pred)
    rows, columns = linear_sum_assignment(-matrix)
    return float(matrix[rows, columns].sum()) / float(len(y_true))


def bcubed(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    matrix = contingency(y_true, y_pred).astype(np.float64)
    count = matrix.sum()
    precision = np.sum(
        matrix * matrix / np.maximum(matrix.sum(axis=0, keepdims=True), 1.0)
    ) / count
    recall = np.sum(
        matrix * matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
    ) / count
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return float(precision), float(recall), float(f1)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1 = bcubed(y_true, y_pred)
    return {
        "ARI": 100.0 * adjusted_rand_score(y_true, y_pred),
        "NMI": 100.0 * normalized_mutual_info_score(
            y_true, y_pred, average_method="arithmetic"
        ),
        "ACC": 100.0 * clustering_accuracy(y_true, y_pred),
        "B3_precision": 100.0 * precision,
        "B3_recall": 100.0 * recall,
        "B3_F1": 100.0 * f1,
    }
