"""Standard conformal baselines for the derived-quantity study (B1).

The manuscript's own method family (constant width, pre/post dispersion, Method A/B)
is compared in ``analyze_piv_uncertainty_study.py``.  A methods journal will also ask
for the *standard* conformal baselines, which are implemented here:

* ``cqr``            - conformalized quantile regression (Romano et al. 2019): fit
  pinball regressions for the lower/upper quantile on a proper-training split, then
  conformalise the interval on a separate calibration split.
* ``cv_plus``        - CV+ / jackknife+-style cross-conformal (Barber et al. 2021):
  K-fold models, absolute-residual scores computed out-of-fold, interval built from the
  union of fold predictions.
* ``quantile_only``  - the same pinball fits *without* conformalisation; included to
  show what the calibration step buys.

All three are given the same features the manuscript's methods can see
(``log sigma``, the point estimate, and a domain indicator), the same proper-training
set, the same calibration set, and are evaluated on the same untouched test split.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EPSILON = 1e-12


def conformal_quantile(scores, alpha):
    scores = np.sort(np.asarray(scores, dtype=np.float64).ravel())
    n = scores.size
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    return float(scores[min(rank, n) - 1])


def _standardise(train, *others):
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return [(item - mean) / scale for item in (train, *others)]


def build_features(scale, point, domain_onehot):
    """Features every method may use: log scale, point estimate, domain indicators."""
    return np.column_stack([np.log(np.asarray(scale, dtype=np.float64) + EPSILON),
                            np.asarray(point, dtype=np.float64),
                            np.asarray(domain_onehot, dtype=np.float64)])


def domain_onehot(domains):
    names = sorted(set(str(item) for item in domains))
    index = {name: i for i, name in enumerate(names)}
    matrix = np.zeros((len(domains), len(names)))
    for row, name in enumerate(domains):
        matrix[row, index[str(name)]] = 1.0
    return matrix, names


@dataclass
class BaselineResult:
    name: str
    coverage: float
    width: float
    q_lo: float
    q_hi: float


def _fit_pinball(X, y, tau):
    from sklearn.linear_model import QuantileRegressor

    model = QuantileRegressor(quantile=tau, alpha=1e-3, solver="highs", fit_intercept=True)
    model.fit(X, y)
    return model


def cqr(X_train, y_train, X_cal, y_cal, X_eval, alpha):
    lo = _fit_pinball(X_train, y_train, alpha / 2.0)
    hi = _fit_pinball(X_train, y_train, 1.0 - alpha / 2.0)
    lo_cal, hi_cal = lo.predict(X_cal), hi.predict(X_cal)
    scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    q = conformal_quantile(scores, alpha)
    lo_eval, hi_eval = lo.predict(X_eval), hi.predict(X_eval)
    lower, upper = lo_eval - q, hi_eval + q
    return lower, upper, q


def quantile_only(X_train, y_train, X_cal, y_cal, X_eval, alpha):
    lo = _fit_pinball(X_train, y_train, alpha / 2.0)
    hi = _fit_pinball(X_train, y_train, 1.0 - alpha / 2.0)
    return lo.predict(X_eval), hi.predict(X_eval), 0.0


def cv_plus(X_train, y_train, X_cal, y_cal, X_eval, alpha, folds=5, seed=0):
    """CV+ with absolute-residual scores (cross-conformal / jackknife+ variant)."""
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(seed)
    n = X_cal.shape[0]
    order = rng.permutation(n)
    assignments = np.zeros(n, dtype=int)
    for fold, index in enumerate(np.array_split(order, folds)):
        assignments[index] = fold

    fold_models = []
    calibration_residual = np.zeros(n)
    for fold in range(folds):
        mask = assignments != fold
        model = LinearRegression().fit(X_cal[mask], y_cal[mask])
        fold_models.append(model)
        held = ~mask
        calibration_residual[held] = np.abs(y_cal[held] - model.predict(X_cal[held]))

    # the proper-training split is folded in as one more model, which only helps
    model_all = LinearRegression().fit(np.vstack([X_train, X_cal]), np.hstack([y_train, y_cal]))
    fold_models.append(model_all)

    predictions = np.stack([model.predict(X_eval) for model in fold_models], axis=0)   # [K, n_eval]
    q = conformal_quantile(calibration_residual, alpha)
    lower = predictions.min(axis=0) - q
    upper = predictions.max(axis=0) + q
    return lower, upper, q


def normalised_dispersion(scale_cal, error_cal, scale_eval, error_eval, point_eval, alpha):
    """The manuscript's normalised split conformal, for the same-split comparison."""
    q = conformal_quantile(np.abs(error_cal) / (scale_cal + EPSILON), alpha)
    half = q * (scale_eval + EPSILON)
    return point_eval - half, point_eval + half, q


def constant_width(error_cal, point_eval, alpha):
    q = conformal_quantile(np.abs(error_cal), alpha)
    return point_eval - q, point_eval + q, q
