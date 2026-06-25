from __future__ import annotations

import numpy as np


def roc_auc_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    positives = int(y_true.sum())
    negatives = int((1 - y_true).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("ROC-AUC is undefined when one class is missing.")

    order = np.argsort(y_score)
    sorted_scores = y_score[order]
    sorted_true = y_true[order]

    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    start = 0
    n = len(sorted_scores)
    while start < n:
        end = start + 1
        while end < n and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + end - 1) / 2.0 + 1.0
        ranks[start:end] = average_rank
        start = end

    positive_ranks_sum = ranks[sorted_true == 1].sum()
    auc = (positive_ranks_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)
