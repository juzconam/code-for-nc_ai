"""Split-conformal calibration utilities for derived flow quantities.

Why this module exists
----------------------
The published pipeline converted *candidate dispersion* into an interval by
taking the 5 %/95 % quantiles over ``K`` candidates and then adding an additive
radius fitted on the validation split (``src/evaluation/posterior_calibration``).
That construction has three defects that this module removes:

1. With ``K = 5`` the 5 %/95 % quantiles are the sample minimum and maximum, so
   the "interval" has no coverage interpretation at all.
2. The additive radius is a single number per quantity; it cannot use the
   per-sample scale ``s(x)``, so a quantity whose dispersion is *smaller* than
   its error gets exactly the same width inflation as one whose dispersion is
   larger.
3. The radius is the ``ceil((n+1)(1-alpha))``-th order statistic of the
   non-conformity ``max(lower - y, y - upper)``, which is the right finite-sample
   statistic for the *base* interval, but the base interval is a quantile
   interval whose own coverage is unknown, so the two errors compound.

This module implements the canonical construction:

    R_i = |y_hat(x_i) - y_i| / (s(x_i) + eps)          (calibration scores)
    Q   = R_(ceil((n+1)(1-alpha)))                     (order statistic, not interpolated)
    C(x) = [y_hat(x) - Q (s(x)+eps), y_hat(x) + Q (s(x)+eps)]

with ``s`` a deterministic function of the frozen pipeline.  Under
exchangeability of calibration and test samples,
``P(y in C(x)) >= 1 - alpha - 1/(n+1)``.  ``s(x) = 1`` recovers the
constant-width conformal baseline, which is a useful control: if it matches the
scaled interval then the scale carries no information.

Every coverage number is reported with a Wilson score interval, and the
interval score (Gneiting & Raftery) is reported next to the width so that
"coverage at any price" cannot be mistaken for quality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

Z95 = 1.959963984540054
DEFAULT_EPSILON = 1e-8


# ---------------------------------------------------------------------------
# Finite-sample primitives
# ---------------------------------------------------------------------------
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return the ``ceil((n+1)(1-alpha))``-th smallest score.

    This is the split-conformal quantile.  ``+inf`` is returned when the
    calibration set is too small to support the requested ``alpha`` on finite
    data, which is the honest answer: an interval that cannot be certified must
    be reported as unbounded rather than truncated.

    ``numpy.quantile(..., method="linear")`` must NOT be used here: interpolation
    between order statistics destroys the finite-sample coverage guarantee.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    scores = scores[np.isfinite(scores)]
    n = scores.size
    if n == 0:
        raise ValueError("conformal_quantile requires at least one finite score")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    rank = int(math.ceil((n + 1) * (1.0 - float(alpha))))
    if rank > n:
        return float("inf")
    return float(np.partition(scores, rank - 1)[rank - 1])


def rank_for_alpha(n: int, alpha: float) -> int:
    """The 1-based rank used by :func:`conformal_quantile` (``n+1`` if unbounded)."""
    return int(math.ceil((n + 1) * (1.0 - float(alpha))))


def wilson_interval(successes, n, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    n = int(n)
    if n <= 0:
        return (float("nan"), float("nan"))
    k = float(successes)
    p = k / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(max(p * (1.0 - p) / n + z * z / (4.0 * n * n), 0.0)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def clopper_pearson(successes, n, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial interval via the beta quantile (no SciPy dependency)."""
    n = int(n)
    k = int(successes)
    if n <= 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else _beta_ppf(alpha / 2.0, k, n - k + 1)
    hi = 1.0 if k == n else _beta_ppf(1.0 - alpha / 2.0, k + 1, n - k)
    return (lo, hi)


def _beta_ppf(q: float, a: float, b: float) -> float:
    """Regularised inverse incomplete beta via bisection (small, SciPy-free)."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function ``I_x(a, b)`` (continued fraction)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_cf(a, b, x) / a
    return 1.0 - math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + b * math.log1p(-x) + a * math.log(x)
    ) * _beta_cf(b, a, 1.0 - x) / b


def _beta_cf(a: float, b: float, x: float, max_iter: int = 300) -> float:
    """Lentz's continued fraction for the incomplete beta function."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


# ---------------------------------------------------------------------------
# Interval constructors
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SplitConformal:
    """A fitted split-conformal calibration for one quantity and one method."""

    q: float
    alpha: float
    epsilon: float
    n_calibration: int
    method: str
    unit: str = ""

    @property
    def nominal_coverage(self) -> float:
        return 1.0 - self.alpha

    def radius(self, scale):
        return self.q * (np.asarray(scale, dtype=np.float64) + self.epsilon)


def fit_split_conformal(
    point,
    truth,
    scale=None,
    alpha: float = 0.10,
    epsilon: float = DEFAULT_EPSILON,
    method: str = "scaled",
) -> SplitConformal:
    """Fit the normalised split-conformal quantile ``Q``.

    ``point`` and ``truth`` are calibration point estimates and ground truth;
    ``scale`` is the non-negative per-sample dispersion.  ``scale=None`` gives
    the constant-width baseline (``s(x) = 1``), whose fitted ``Q`` has the
    interpretation of a half-width in the quantity's own units.
    """
    point = np.asarray(point, dtype=np.float64).ravel()
    truth = np.asarray(truth, dtype=np.float64).ravel()
    if point.shape != truth.shape:
        raise ValueError("point and truth must have identical shapes")
    if scale is None:
        scale = np.ones_like(point)
    scale = np.asarray(scale, dtype=np.float64).ravel()
    if scale.shape != point.shape:
        raise ValueError("scale must match point")
    if np.any(scale < 0.0):
        raise ValueError("scale must be non-negative")
    scores = np.abs(point - truth) / (scale + float(epsilon))
    return SplitConformal(
        q=conformal_quantile(scores, alpha),
        alpha=float(alpha),
        epsilon=float(epsilon),
        n_calibration=int(scores.size),
        method=str(method),
    )


def predict_interval(prediction: SplitConformal, point, scale=None) -> tuple[np.ndarray, np.ndarray]:
    point = np.asarray(point, dtype=np.float64).ravel()
    if scale is None:
        scale = np.ones_like(point)
    scale = np.asarray(scale, dtype=np.float64).ravel()
    radius = prediction.radius(scale)
    return point - radius, point + radius


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def interval_score(truth, lower, upper, alpha: float) -> np.ndarray:
    """Gneiting-Raftery interval score for a central ``1-alpha`` interval."""
    truth = np.asarray(truth, dtype=np.float64).ravel()
    lower = np.asarray(lower, dtype=np.float64).ravel()
    upper = np.asarray(upper, dtype=np.float64).ravel()
    width = upper - lower
    penalty_lower = np.clip(lower - truth, 0.0, None)
    penalty_upper = np.clip(truth - upper, 0.0, None)
    return width + (2.0 / float(alpha)) * (penalty_lower + penalty_upper)


@dataclass
class CoverageReport:
    quantity: str
    method: str
    nominal: float
    n: int
    covered: int
    coverage: float
    ci_low: float
    ci_high: float
    mean_width: float
    median_width: float
    mean_interval_score: float
    mean_abs_error: float
    sharpness_ratio: float  # mean_width / mean_abs_error
    q: float = float("nan")
    n_calibration: int = 0
    notes: str = ""

    def as_row(self) -> dict:
        return {
            "quantity": self.quantity,
            "method": self.method,
            "nominal": self.nominal,
            "n": self.n,
            "covered": self.covered,
            "coverage_pct": 100.0 * self.coverage,
            "coverage_ci_low_pct": 100.0 * self.ci_low,
            "coverage_ci_high_pct": 100.0 * self.ci_high,
            "mean_width": self.mean_width,
            "median_width": self.median_width,
            "mean_interval_score": self.mean_interval_score,
            "mean_abs_error": self.mean_abs_error,
            "width_over_error": self.sharpness_ratio,
            "conformal_q": self.q,
            "n_calibration": self.n_calibration,
            "notes": self.notes,
        }


def evaluate_intervals(
    quantity: str,
    method: str,
    truth,
    lower,
    upper,
    alpha: float,
    q: float = float("nan"),
    n_calibration: int = 0,
    notes: str = "",
) -> CoverageReport:
    truth = np.asarray(truth, dtype=np.float64).ravel()
    lower = np.asarray(lower, dtype=np.float64).ravel()
    upper = np.asarray(upper, dtype=np.float64).ravel()
    if not (truth.shape == lower.shape == upper.shape):
        raise ValueError("truth, lower and upper must share a shape")
    covered = ((truth >= lower) & (truth <= upper)).astype(np.float64)
    n = int(truth.size)
    k = int(covered.sum())
    lo, hi = wilson_interval(k, n)
    width = upper - lower
    finite = np.isfinite(lower) & np.isfinite(upper)
    if np.all(finite):
        mean_abs_error = float(np.mean(np.abs(truth - 0.5 * (lower + upper))))
        mean_width = float(np.mean(width))
        median_width = float(np.median(width))
        mean_score = float(np.mean(interval_score(truth, lower, upper, alpha)))
    else:
        # An unbounded interval is the honest outcome when the calibration set is
        # too small for the requested alpha; never report a finite width for it.
        mean_abs_error = (
            float(np.mean(np.abs(truth[finite] - 0.5 * (lower[finite] + upper[finite]))))
            if np.any(finite) else float("nan")
        )
        mean_width = float("inf")
        median_width = float("inf")
        mean_score = float("inf")
        notes = (notes + " unbounded: ceil((n+1)(1-alpha)) exceeds the calibration set").strip()
    sharpness = (
        float(mean_width / mean_abs_error)
        if mean_abs_error > 0 and np.isfinite(mean_width) else float("inf")
    )
    return CoverageReport(
        quantity=quantity,
        method=method,
        nominal=1.0 - float(alpha),
        n=n,
        covered=k,
        coverage=float(k / n) if n else float("nan"),
        ci_low=float(lo),
        ci_high=float(hi),
        mean_width=mean_width,
        median_width=median_width,
        mean_interval_score=mean_score,
        mean_abs_error=mean_abs_error,
        sharpness_ratio=sharpness,
        q=float(q),
        n_calibration=int(n_calibration),
        notes=notes,
    )


def per_group_rows(
    quantity: str,
    method: str,
    truth,
    lower,
    upper,
    groups: Sequence,
    alpha: float,
) -> list[dict]:
    """Marginal coverage broken down by an arbitrary grouping (e.g. domain)."""
    truth = np.asarray(truth, dtype=np.float64).ravel()
    lower = np.asarray(lower, dtype=np.float64).ravel()
    upper = np.asarray(upper, dtype=np.float64).ravel()
    groups = np.asarray(groups).ravel()
    rows = []
    for name in sorted(set(groups.tolist())):
        mask = groups == name
        report = evaluate_intervals(quantity, method, truth[mask], lower[mask], upper[mask], alpha)
        row = report.as_row()
        row["group"] = str(name)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Scale diagnostics: does the dispersion actually see the error?
# ---------------------------------------------------------------------------
def _pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3:
        return float("nan")
    sx, sy = x.std(), y.std()
    if sx <= 0.0 or sy <= 0.0:
        return float("nan")
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))


def _rankdata(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
    # average ties
    sorted_x = x[order]
    i = 0
    while i < sorted_x.size:
        j = i
        while j + 1 < sorted_x.size and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    """Spearman rank correlation without a SciPy dependency."""
    return _pearson(_rankdata(x), _rankdata(y))


def scale_reliability_rows(
    scale,
    abs_error,
    quantity: str,
    method: str,
    bins: int = 5,
) -> list[dict]:
    """Bin samples by predicted dispersion and compare with realised error.

    A dispersion that is informative at all must be monotonically related to the
    absolute error.  If the top-``scale`` quintile has the same error as the
    bottom quintile, the dispersion cannot sharpen any interval.
    """
    scale = np.asarray(scale, dtype=np.float64).ravel()
    abs_error = np.asarray(abs_error, dtype=np.float64).ravel()
    order = np.argsort(scale, kind="mergesort")
    chunks = np.array_split(order, int(max(bins, 1)))
    rows = []
    for index, chunk in enumerate(chunks):
        if chunk.size == 0:
            continue
        rows.append({
            "quantity": quantity,
            "method": method,
            "bin": index + 1,
            "n": int(chunk.size),
            "scale_mean": float(scale[chunk].mean()),
            "abs_error_mean": float(abs_error[chunk].mean()),
            "ratio_error_over_scale": float(
                abs_error[chunk].mean() / max(scale[chunk].mean(), DEFAULT_EPSILON)
            ),
        })
    if rows:
        rows.append({
            "quantity": quantity,
            "method": method,
            "bin": "spearman",
            "n": int(scale.size),
            "scale_mean": float(scale.mean()),
            "abs_error_mean": float(abs_error.mean()),
            "ratio_error_over_scale": spearman(scale, abs_error),
        })
    return rows


def top_bottom_error_ratio(scale, abs_error, quantile: float = 0.2) -> float:
    """Mean error of the highest-scale tail divided by that of the lowest tail."""
    scale = np.asarray(scale, dtype=np.float64).ravel()
    abs_error = np.asarray(abs_error, dtype=np.float64).ravel()
    if scale.size < 5:
        return float("nan")
    count = max(1, int(round(quantile * scale.size)))
    order = np.argsort(scale, kind="mergesort")
    low = abs_error[order[:count]].mean()
    high = abs_error[order[-count:]].mean()
    return float(high / low) if low > 0 else float("inf")


# ---------------------------------------------------------------------------
# Pixel-level community metrics (UNB / UNE)
# ---------------------------------------------------------------------------
def uncertainty_normalized_metrics(error_vectors, sigma_components) -> dict:
    """UNB / UNE of Sciacchitano & Wieneke for per-vector displacement errors.

    ``error_vectors`` has shape ``[..., 2]`` (signed error, ``e = estimate - truth``)
    and ``sigma_components`` the matching non-negative per-component uncertainty
    ``[..., 2]``.  With a well-calibrated Gaussian at 95 % confidence,
    ``UNE ~ sqrt(2/pi) = 0.798`` and ``UNB ~ 0``.

    ``unb`` is the mean of the two component ratios ``e_k / sigma_k``; ``une`` is
    the mean of ``|e| / sqrt(su^2 + sv^2)``.  ``une_rms`` is the RMS variant that
    penalises rare large normalised errors and is therefore the stricter number.
    """
    error_vectors = np.asarray(error_vectors, dtype=np.float64)
    sigma_components = np.asarray(sigma_components, dtype=np.float64)
    if error_vectors.shape != sigma_components.shape:
        raise ValueError("error_vectors and sigma_components must share a shape")
    magnitude = np.linalg.norm(error_vectors, axis=-1)
    sigma_magnitude = np.linalg.norm(sigma_components, axis=-1)
    valid = (
        np.all(sigma_components > 0, axis=-1)
        & np.isfinite(sigma_components).all(axis=-1)
        & np.isfinite(error_vectors).all(axis=-1)
        & (sigma_magnitude > 0)
    )
    if not np.any(valid):
        return {"unb": float("nan"), "une": float("nan"), "une_rms": float("nan"), "n": 0}
    component_ratio = error_vectors[valid] / sigma_components[valid]
    ratio = magnitude[valid] / sigma_magnitude[valid]
    return {
        "unb": float(np.mean(component_ratio)),
        "une": float(np.mean(ratio)),
        "une_rms": float(np.sqrt(np.mean(ratio ** 2))),
        "n": int(valid.sum()),
    }
