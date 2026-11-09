"""Exact paired statistics and deterministic BCa intervals used in the paper.

Ported without numerical changes from the original analysis primitives.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral, Real
from statistics import NormalDist
from typing import Any, Literal

import numpy as np

BCA_RESAMPLES = 10_000
BCA_SEED = 20_260_812
BCA_CONFIDENCE_LEVEL = 0.95


class SeedSetError(ValueError):
    """Raised when an analysis input does not contain the exact seeds."""


@dataclass(frozen=True)
class ObjectiveCall:
    """One chronological objective call.

    ``accepted`` is retained for auditability but does not determine whether a
    valid evaluation enters the best-so-far trajectory.
    """

    call_index: int
    objective_mm: float
    valid: bool
    accepted: bool | None = None


@dataclass(frozen=True)
class WilcoxonResult:
    """Exact two-sided Wilcoxon signed-rank result."""

    n_total: int
    n_nonzero: int
    w_positive: float
    w_negative: float
    statistic: float
    p_value: float
    rank_biserial: float


@dataclass(frozen=True)
class BootstrapInterval:
    """A deterministic paired BCa interval or an explicit undefined result."""

    statistic: Literal["mean", "median"]
    estimate: float | None
    lower: float | None
    upper: float | None
    confidence_level: float
    n_pairs: int
    n_resamples: int
    rng_seed: int
    defined: bool
    reason: str | None
    bias_correction: float | None = None
    acceleration: float | None = None


def _as_seed(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise SeedSetError(f"{label} must be an integer, got {value!r}")
    return int(value)


def _record_seed(record: Any) -> int:
    if isinstance(record, Mapping):
        if "seed" not in record:
            raise SeedSetError("each record must contain a seed field")
        return _as_seed(record["seed"], label="record seed")
    if not hasattr(record, "seed"):
        raise SeedSetError("each record must expose a seed attribute")
    return _as_seed(record.seed, label="record seed")


def validate_exact_seed_set(
    mapping_or_records: Mapping[int, Any] | Iterable[Any],
    expected_seeds: Iterable[int],
) -> dict[int, Any]:
    """Validate and order an exact seed-indexed input.

    A mapping is interpreted as ``seed -> value``. An iterable is interpreted
    as records containing a ``seed`` key or attribute. Missing, duplicate, and
    extra seeds are hard failures. The returned dictionary follows the reported
    expected-seed order.
    """

    expected = tuple(_as_seed(seed, label="expected seed") for seed in expected_seeds)
    if not expected:
        raise SeedSetError("expected seeds must contain at least one seed")
    if len(set(expected)) != len(expected):
        raise SeedSetError("expected seeds contain duplicates")

    observed: dict[int, Any] = {}
    if isinstance(mapping_or_records, Mapping):
        items = (
            (_as_seed(seed, label="mapping seed"), value)
            for seed, value in mapping_or_records.items()
        )
    else:
        records = list(mapping_or_records)
        items = ((_record_seed(record), record) for record in records)

    duplicate_seeds: list[int] = []
    for seed, value in items:
        if seed in observed:
            duplicate_seeds.append(seed)
        else:
            observed[seed] = value
    if duplicate_seeds:
        duplicates = sorted(set(duplicate_seeds))
        raise SeedSetError(f"duplicate seeds: {duplicates}")

    expected_set = set(expected)
    observed_set = set(observed)
    missing = sorted(expected_set - observed_set)
    extra = sorted(observed_set - expected_set)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing seeds: {missing}")
        if extra:
            details.append(f"extra seeds: {extra}")
        raise SeedSetError("seed set mismatch; " + "; ".join(details))

    return {seed: observed[seed] for seed in expected}


def paired_differences(
    treatment_by_seed: Mapping[int, Real],
    comparator_by_seed: Mapping[int, Real],
    expected_seeds: Iterable[int],
) -> tuple[float, ...]:
    """Return treatment-minus-comparator differences in seed order."""

    expected = tuple(expected_seeds)
    treatment = validate_exact_seed_set(treatment_by_seed, expected)
    comparator = validate_exact_seed_set(comparator_by_seed, expected)
    differences = tuple(float(treatment[seed]) - float(comparator[seed]) for seed in expected)
    if not all(math.isfinite(value) for value in differences):
        raise ValueError("paired values and differences must be finite")
    return differences


def _coerce_objective_call(record: ObjectiveCall | Mapping[str, Any]) -> ObjectiveCall:
    if isinstance(record, ObjectiveCall):
        call = record
    elif isinstance(record, Mapping):
        try:
            objective_value = record.get("objective_mm", record.get("score_mm"))
            if objective_value is None:
                raise KeyError("objective_mm or score_mm")
            call = ObjectiveCall(
                call_index=record["call_index"],
                objective_mm=objective_value,
                valid=record["valid"],
                accepted=record.get("accepted"),
            )
        except KeyError as exc:
            raise ValueError(f"objective record missing field: {exc.args[0]}") from exc
    else:
        raise TypeError("objective calls must be ObjectiveCall objects or mappings")

    if isinstance(call.call_index, bool) or not isinstance(call.call_index, Integral):
        raise ValueError("call_index must be an integer")  # noqa: TRY004, preserve original API
    if not isinstance(call.valid, bool):
        raise ValueError("valid must be a boolean")  # noqa: TRY004, preserve original API
    if isinstance(call.objective_mm, bool) or not isinstance(call.objective_mm, Real):
        raise ValueError("objective_mm must be a real number")  # noqa: TRY004, preserve original API
    objective = float(call.objective_mm)
    if not math.isfinite(objective):
        raise ValueError("every objective call must have a finite objective_mm")
    return ObjectiveCall(
        call_index=int(call.call_index),
        objective_mm=objective,
        valid=call.valid,
        accepted=call.accepted,
    )


def chronological_best_so_far(
    calls: Sequence[ObjectiveCall | Mapping[str, Any]],
    *,
    budget: int = 40,
    structural_tail_carry: int = 0,
) -> tuple[float, ...]:
    """Build the valid chronological best curve at an exact call budget.

    Invalid calls remain in the chronology but cannot improve the best value.
    A valid evaluation is eligible even when its optimizer-level ``accepted``
    flag is false. Only a declared single unused SPSA tail may be appended.
    """

    if isinstance(budget, bool) or not isinstance(budget, Integral) or budget <= 0:
        raise ValueError("budget must be a positive integer")
    if structural_tail_carry not in (0, 1) or isinstance(structural_tail_carry, bool):
        raise ValueError("structural tail carry must be exactly 0 or 1")

    normalized = [_coerce_objective_call(record) for record in calls]
    if len(normalized) + structural_tail_carry != budget:
        raise ValueError("objective-call count plus declared structural tail must equal budget")
    expected_indices = list(range(1, len(normalized) + 1))
    actual_indices = [call.call_index for call in normalized]
    if actual_indices != expected_indices:
        raise ValueError("objective call indices must be contiguous and chronological")
    if not normalized or not normalized[0].valid:
        raise ValueError("the initial objective call must be valid")

    best = math.inf
    trajectory: list[float] = []
    for call in normalized:
        if call.valid:
            best = min(best, call.objective_mm)
        trajectory.append(best)
    if structural_tail_carry:
        trajectory.append(best)
    return tuple(trajectory)


def auc40(
    calls: Sequence[ObjectiveCall | Mapping[str, Any]],
    *,
    structural_tail_carry: int = 0,
) -> float:
    """Return the mean chronological valid best-so-far value over 40 calls."""

    trajectory = chronological_best_so_far(
        calls,
        budget=40,
        structural_tail_carry=structural_tail_carry,
    )
    return float(math.fsum(trajectory) / 40.0)


def capped_time_to_threshold(
    calls: Sequence[ObjectiveCall | Mapping[str, Any]],
    *,
    threshold_mm: float,
    structural_tail_carry: int = 0,
) -> int:
    """Return first threshold call, or failure-inclusive ``T=41``."""

    threshold = float(threshold_mm)
    if not math.isfinite(threshold):
        raise ValueError("threshold_mm must be finite")
    trajectory = chronological_best_so_far(
        calls,
        budget=40,
        structural_tail_carry=structural_tail_carry,
    )
    for index, best in enumerate(trajectory, start=1):
        if best <= threshold:
            return index
    return 41


def _average_absolute_ranks(values: Sequence[float]) -> list[Fraction]:
    ranks = [Fraction(0) for _ in values]
    order = sorted(range(len(values)), key=lambda index: abs(values[index]))
    position = 0
    while position < len(order):
        end = position + 1
        tied_value = abs(values[order[position]])
        while end < len(order) and abs(values[order[end]]) == tied_value:
            end += 1
        average_rank = Fraction(position + 1 + end, 2)
        for ordered_index in order[position:end]:
            ranks[ordered_index] = average_rank
        position = end
    return ranks


def exact_wilcoxon_signed_rank(differences: Iterable[Real]) -> WilcoxonResult:
    """Compute the exact two-sided paired Wilcoxon signed-rank test.

    Exact zeros are discarded. Tied nonzero absolute differences receive
    average ranks. The conditional sign distribution is computed by integer
    dynamic programming, with no asymptotic fallback.
    """

    values = tuple(float(value) for value in differences)
    if not values:
        raise ValueError("at least one paired difference is required")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("paired differences must be finite")

    nonzero = tuple(value for value in values if value != 0.0)
    if not nonzero:
        return WilcoxonResult(
            n_total=len(values),
            n_nonzero=0,
            w_positive=0.0,
            w_negative=0.0,
            statistic=0.0,
            p_value=1.0,
            rank_biserial=0.0,
        )

    ranks = _average_absolute_ranks(nonzero)
    w_positive_fraction = sum(
        (rank for rank, value in zip(ranks, nonzero) if value > 0.0),
        start=Fraction(0),
    )
    w_negative_fraction = sum(ranks, start=Fraction(0)) - w_positive_fraction

    scale = math.lcm(*(rank.denominator for rank in ranks))
    integer_ranks = [int(rank * scale) for rank in ranks]
    observed_positive = int(w_positive_fraction * scale)

    counts = {0: 1}
    for rank in integer_ranks:
        updated = counts.copy()
        for subtotal, count in counts.items():
            new_total = subtotal + rank
            updated[new_total] = updated.get(new_total, 0) + count
        counts = updated

    assignments = 1 << len(integer_ranks)
    lower_count = sum(count for total, count in counts.items() if total <= observed_positive)
    upper_count = sum(count for total, count in counts.items() if total >= observed_positive)
    p_value = min(1.0, 2.0 * min(lower_count, upper_count) / assignments)

    w_positive = float(w_positive_fraction)
    w_negative = float(w_negative_fraction)
    rank_total = w_positive + w_negative
    rank_biserial = (w_positive - w_negative) / rank_total
    return WilcoxonResult(
        n_total=len(values),
        n_nonzero=len(nonzero),
        w_positive=w_positive,
        w_negative=w_negative,
        statistic=min(w_positive, w_negative),
        p_value=float(p_value),
        rank_biserial=float(rank_biserial),
    )


def holm_adjust_two(p_values: Mapping[str, Real]) -> dict[str, float]:
    """Apply Holm step-down adjustment to exactly two named contrasts."""

    if len(p_values) != 2:
        raise ValueError("Holm family must contain exactly two contrasts")
    checked: dict[str, float] = {}
    for name, raw_value in p_values.items():
        value = float(raw_value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"p-value for {name!r} must be finite and in [0, 1]")
        checked[name] = value

    ordered = sorted(checked, key=lambda name: checked[name])
    first, second = ordered
    first_adjusted = min(1.0, 2.0 * checked[first])
    second_adjusted = min(1.0, max(first_adjusted, checked[second]))
    adjusted = {first: first_adjusted, second: second_adjusted}
    return {name: adjusted[name] for name in checked}


def _undefined_bca(
    *,
    statistic: Literal["mean", "median"],
    estimate: float | None,
    n_pairs: int,
    reason: str,
    bias_correction: float | None = None,
    acceleration: float | None = None,
) -> BootstrapInterval:
    return BootstrapInterval(
        statistic=statistic,
        estimate=estimate,
        lower=None,
        upper=None,
        confidence_level=BCA_CONFIDENCE_LEVEL,
        n_pairs=n_pairs,
        n_resamples=BCA_RESAMPLES,
        rng_seed=BCA_SEED,
        defined=False,
        reason=reason,
        bias_correction=bias_correction,
        acceleration=acceleration,
    )


def paired_bca_interval(
    differences: Iterable[Real],
    *,
    statistic: Literal["mean", "median"],
) -> BootstrapInterval:
    """Compute the deterministic 95% paired BCa interval.

    Bias correction uses the midrank bootstrap proportion
    ``(count(< estimate) + 0.5*count(== estimate)) / 10000``. Quantiles use
    NumPy's linear interpolation. Degenerate bootstrap distributions,
    undefined jackknife acceleration, and invalid adjusted quantiles return an
    explicit undefined result rather than another interval type.
    """

    if statistic not in ("mean", "median"):
        raise ValueError("statistic must be 'mean' or 'median'")
    values = np.asarray(tuple(float(value) for value in differences), dtype=float)
    if values.ndim != 1:
        raise ValueError("paired differences must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("paired differences must be finite")
    if values.size == 0:
        return _undefined_bca(
            statistic=statistic,
            estimate=None,
            n_pairs=0,
            reason="no_pairs",
        )

    statistic_function = np.mean if statistic == "mean" else np.median
    estimate = float(statistic_function(values))
    if values.size < 2:
        return _undefined_bca(
            statistic=statistic,
            estimate=estimate,
            n_pairs=int(values.size),
            reason="at_least_two_pairs_required",
        )

    rng = np.random.default_rng(BCA_SEED)
    indices = rng.integers(0, values.size, size=(BCA_RESAMPLES, values.size))
    bootstrap_statistics = statistic_function(values[indices], axis=1)
    if np.all(bootstrap_statistics == bootstrap_statistics[0]):
        return _undefined_bca(
            statistic=statistic,
            estimate=estimate,
            n_pairs=int(values.size),
            reason="degenerate_bootstrap_distribution",
        )

    less = np.count_nonzero(bootstrap_statistics < estimate)
    equal = np.count_nonzero(bootstrap_statistics == estimate)
    bias_probability = (less + 0.5 * equal) / BCA_RESAMPLES
    if not 0.0 < bias_probability < 1.0:
        return _undefined_bca(
            statistic=statistic,
            estimate=estimate,
            n_pairs=int(values.size),
            reason="bias_correction_infinite",
        )
    normal = NormalDist()
    bias_correction = normal.inv_cdf(bias_probability)

    jackknife_statistics = np.asarray(
        [statistic_function(np.delete(values, index)) for index in range(values.size)],
        dtype=float,
    )
    centered = float(np.mean(jackknife_statistics)) - jackknife_statistics
    sum_squares = float(np.sum(centered**2))
    if sum_squares == 0.0:
        return _undefined_bca(
            statistic=statistic,
            estimate=estimate,
            n_pairs=int(values.size),
            reason="jackknife_acceleration_undefined",
            bias_correction=bias_correction,
        )
    acceleration = float(np.sum(centered**3) / (6.0 * math.pow(sum_squares, 1.5)))

    tail_probability = (1.0 - BCA_CONFIDENCE_LEVEL) / 2.0
    normal_quantiles = (
        normal.inv_cdf(tail_probability),
        normal.inv_cdf(1.0 - tail_probability),
    )
    adjusted_probabilities: list[float] = []
    for normal_quantile in normal_quantiles:
        shifted = bias_correction + normal_quantile
        denominator = 1.0 - acceleration * shifted
        if denominator == 0.0:
            return _undefined_bca(
                statistic=statistic,
                estimate=estimate,
                n_pairs=int(values.size),
                reason="adjusted_quantile_undefined",
                bias_correction=bias_correction,
                acceleration=acceleration,
            )
        adjusted = normal.cdf(bias_correction + shifted / denominator)
        if not 0.0 < adjusted < 1.0:
            return _undefined_bca(
                statistic=statistic,
                estimate=estimate,
                n_pairs=int(values.size),
                reason="adjusted_quantile_out_of_range",
                bias_correction=bias_correction,
                acceleration=acceleration,
            )
        adjusted_probabilities.append(adjusted)

    lower, upper = np.quantile(
        bootstrap_statistics,
        adjusted_probabilities,
        method="linear",
    )
    return BootstrapInterval(
        statistic=statistic,
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        confidence_level=BCA_CONFIDENCE_LEVEL,
        n_pairs=int(values.size),
        n_resamples=BCA_RESAMPLES,
        rng_seed=BCA_SEED,
        defined=True,
        reason=None,
        bias_correction=float(bias_correction),
        acceleration=acceleration,
    )


def paired_bca_mean_median(
    differences: Iterable[Real],
) -> dict[str, BootstrapInterval]:
    """Return paired BCa intervals for both paired estimands."""

    values = tuple(differences)
    return {
        "mean": paired_bca_interval(values, statistic="mean"),
        "median": paired_bca_interval(values, statistic="median"),
    }
