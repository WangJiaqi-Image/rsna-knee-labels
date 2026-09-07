"""Generic multi-label macro-AUC, matching the RSNA knee competition's scoring metric."""

from __future__ import annotations

import numpy as np


def macro_auc(y, p) -> float:
    """Mean per-column ROC-AUC. A column with only one class present is skipped (nan),
    not scored as 0.5 or 1.0 -- there is no correct AUC to report on a single class.

    y, p: arrays of shape (n_studies, n_targets); y is 0/1 (or thresholded), p is a score.
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    p = np.asarray(p)
    return float(np.nanmean([
        roc_auc_score(y[:, j], p[:, j]) if len(set(y[:, j])) > 1 else np.nan
        for j in range(y.shape[1])
    ]))


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Standard error of an AUC estimate (Hanley & McNeil, 1982).

    An AUC computed from a small gold-annotated set is not precise just because the
    number itself looks confident -- with a handful of positives, the 95% interval
    (+-1.96 * this) is often wider than the gap between two configurations being
    compared. Use this before treating a small-sample AUC difference as a real finding.
    nan when there is nothing to estimate from (auc is nan, or either class is empty).
    """
    if not np.isfinite(auc) or n_pos < 1 or n_neg < 1:
        return float("nan")
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
           + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))
