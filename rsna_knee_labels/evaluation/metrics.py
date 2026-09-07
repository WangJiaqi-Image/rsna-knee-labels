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
