from __future__ import annotations

from typing import Iterable


def accuracy_score(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    truth = list(y_true)
    pred = list(y_pred)
    if not truth:
        return 0.0
    correct = sum(int(t == p) for t, p in zip(truth, pred))
    return correct / len(truth)


def macro_f1_score(y_true: Iterable[int], y_pred: Iterable[int], num_classes: int) -> float:
    truth = list(y_true)
    pred = list(y_pred)
    if not truth:
        return 0.0

    scores: list[float] = []
    for class_id in range(num_classes):
        tp = sum(1 for t, p in zip(truth, pred) if t == class_id and p == class_id)
        fp = sum(1 for t, p in zip(truth, pred) if t != class_id and p == class_id)
        fn = sum(1 for t, p in zip(truth, pred) if t == class_id and p != class_id)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))

    return sum(scores) / len(scores)
