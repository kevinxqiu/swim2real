"""Deterministic search directions, line search and SPSA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from swim2real.config import PAPER_DEFAULTS
from swim2real.parameters import DIMENSION, denormalize_params, normalize_params

SEED_NAMESPACE = PAPER_DEFAULTS["rng_namespace"]
ALPHAS = tuple(PAPER_DEFAULTS["backtracking_alphas"])
RANDOM_RADII = tuple(PAPER_DEFAULTS["random_radii_normalized"])
SPSA_PROBE_RADIUS = PAPER_DEFAULTS["spsa"]["probe_radius_normalized"]
SPSA_STEP_RADIUS = PAPER_DEFAULTS["spsa"]["step_radius_normalized"]

FloatVector = NDArray[np.float64]


@dataclass(frozen=True)
class BacktrackingCandidate:
    """One planned line-search evaluation."""

    alpha: float
    params: FloatVector


@dataclass(frozen=True)
class CandidateSelection:
    """The incumbent after considering a chronological candidate sequence."""

    params: FloatVector
    loss: float
    candidate_index: int | None
    improved: bool
    alpha: float | None = None


@dataclass(frozen=True)
class RandomRayProposal:
    """A multiscale random proposal and its pre-clipping geometry."""

    params: FloatVector
    preclip_params: FloatVector
    direction: FloatVector
    radius: float


@dataclass(frozen=True)
class SPSAProbePlan:
    """A symmetric SPSA probe pair and this round's remaining LS allowance."""

    incumbent: FloatVector
    delta: FloatVector | None
    h: FloatVector | None
    plus: FloatVector | None
    minus: FloatVector | None
    line_search_budget: int
    unused_budget: int

    @property
    def has_probes(self) -> bool:
        """Whether the budget admitted the mandatory symmetric probe pair."""

        return self.plus is not None and self.minus is not None


@dataclass(frozen=True)
class SPSAProposal:
    """A normalized descent proposal, or an explicit invalid outcome."""

    params: FloatVector | None
    preclip_params: FloatVector | None
    gradient: FloatVector
    valid: bool
    reason: str | None


__all__ = [
    "ALPHAS",
    "DIMENSION",
    "RANDOM_RADII",
    "SEED_NAMESPACE",
    "SPSA_PROBE_RADIUS",
    "SPSA_STEP_RADIUS",
    "BacktrackingCandidate",
    "CandidateSelection",
    "RandomRayProposal",
    "SPSAProbePlan",
    "SPSAProposal",
    "denormalize_params",
    "derive_spsa_proposal",
    "make_rng",
    "normalize_params",
    "plan_backtracking_candidates",
    "plan_spsa_probes",
    "propose_multiscale_random",
    "select_first_improvement",
    "select_spsa_successor",
    "signed_permute_displacement",
]


def _vector(values: ArrayLike, name: str) -> FloatVector:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (DIMENSION,):
        raise ValueError(f"{name} must have exactly {DIMENSION} elements")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.copy()


def _normalized_vector(values: ArrayLike, name: str) -> FloatVector:
    vector = _vector(values, name)
    if np.any(vector < 0.0) or np.any(vector > 1.0):
        raise ValueError(f"{name} must lie coordinatewise in [0, 1]")
    return vector


def _valid_loss(loss: float | None) -> bool:
    if loss is None:
        return False
    try:
        return bool(np.isfinite(float(loss)))
    except (TypeError, ValueError):
        return False


def make_rng(
    problem_seed: int, method_code: int, namespace: int = SEED_NAMESPACE
) -> np.random.Generator:
    """Construct the seeded RNG stream for one problem-method pair.

    The entropy words are the namespace, problem seed and method code.
    """

    entropy_words = []
    for name, value in (("problem_seed", problem_seed), ("method_code", method_code)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        value = int(value)
        if value < 0 or value > np.iinfo(np.uint32).max:
            raise ValueError(f"{name} must be in the uint32 range")
        entropy_words.append(value)
    seed_sequence = np.random.SeedSequence([namespace, *entropy_words])
    return np.random.default_rng(seed_sequence)


def plan_backtracking_candidates(
    incumbent: ArrayLike,
    proposal: ArrayLike,
    remaining_budget: int,
    alphas: Sequence[float] = ALPHAS,
) -> tuple[BacktrackingCandidate, ...]:
    """Return the affordable prefix of alpha ``(1, .5, .25)`` candidates."""

    current = _normalized_vector(incumbent, "incumbent")
    proposed = _vector(proposal, "proposal")
    if not isinstance(remaining_budget, (int, np.integer)):
        raise TypeError("remaining_budget must be an integer")
    if remaining_budget < 0:
        raise ValueError("remaining_budget must be nonnegative")

    return tuple(
        BacktrackingCandidate(
            alpha=alpha,
            params=np.clip(current + alpha * (proposed - current), 0.0, 1.0),
        )
        for alpha in alphas[: min(int(remaining_budget), len(alphas))]
    )


def select_first_improvement(
    incumbent: ArrayLike,
    incumbent_loss: float,
    candidates: Sequence[BacktrackingCandidate],
    candidate_losses: Sequence[float | None],
) -> CandidateSelection:
    """Select the first finite candidate strictly better than the incumbent."""

    current = _normalized_vector(incumbent, "incumbent")
    if not _valid_loss(incumbent_loss):
        raise ValueError("incumbent_loss must be finite")
    if len(candidates) != len(candidate_losses):
        raise ValueError("candidates and candidate_losses must have equal length")

    current_loss = float(incumbent_loss)
    for index, (candidate, loss) in enumerate(zip(candidates, candidate_losses)):
        if not _valid_loss(loss):
            continue
        try:
            params = _normalized_vector(candidate.params, f"candidates[{index}].params")
        except ValueError:
            continue
        numeric_loss = float(loss)
        if numeric_loss < current_loss:
            return CandidateSelection(
                params=params,
                loss=numeric_loss,
                candidate_index=index,
                improved=True,
                alpha=float(candidate.alpha),
            )

    return CandidateSelection(
        params=current,
        loss=current_loss,
        candidate_index=None,
        improved=False,
    )


def propose_multiscale_random(
    incumbent: ArrayLike, rng: np.random.Generator, radii: Sequence[float] = RANDOM_RADII
) -> RandomRayProposal:
    """Draw a uniform discrete radius and a unit-Gaussian random direction."""

    current = _normalized_vector(incumbent, "incumbent")
    gaussian = np.asarray(rng.normal(size=DIMENSION), dtype=np.float64)
    norm = float(np.linalg.norm(gaussian))
    if not np.isfinite(norm) or norm == 0.0:
        raise RuntimeError("RNG produced an invalid zero-norm Gaussian direction")
    direction = gaussian / norm
    radius = radii[int(rng.integers(0, len(radii)))]
    preclip = current + radius * direction
    return RandomRayProposal(
        params=np.clip(preclip, 0.0, 1.0),
        preclip_params=preclip,
        direction=direction,
        radius=radius,
    )


def signed_permute_displacement(displacement: ArrayLike, rng: np.random.Generator) -> FloatVector:
    """Apply the seeded coordinate permutation and independent Rademacher signs."""

    vector = _vector(displacement, "displacement")
    permuted = vector[rng.permutation(DIMENSION)]
    signs = 2.0 * rng.integers(0, 2, size=DIMENSION).astype(np.float64) - 1.0
    return permuted * signs


def plan_spsa_probes(
    incumbent: ArrayLike,
    rng: np.random.Generator,
    remaining_budget: int,
    probe_radius: float = SPSA_PROBE_RADIUS,
    alphas: Sequence[float] = ALPHAS,
) -> SPSAProbePlan:
    """Plan a symmetric SPSA probe pair and an affordable LS prefix.

    With one remaining call, no asymmetric probe is made and that call is marked
    unused. With two calls, the probe pair itself forms the terminal tail and
    remains eligible to update the incumbent.
    """

    current = _normalized_vector(incumbent, "incumbent")
    if not isinstance(remaining_budget, (int, np.integer)):
        raise TypeError("remaining_budget must be an integer")
    if remaining_budget < 0:
        raise ValueError("remaining_budget must be nonnegative")
    remaining = int(remaining_budget)

    if remaining < 2:
        return SPSAProbePlan(
            incumbent=current,
            delta=None,
            h=None,
            plus=None,
            minus=None,
            line_search_budget=0,
            unused_budget=remaining,
        )

    delta = 2.0 * rng.integers(0, 2, size=DIMENSION).astype(np.float64) - 1.0
    h = np.minimum.reduce(
        (
            np.full(DIMENSION, probe_radius),
            current,
            1.0 - current,
        )
    )
    offset = h * delta
    return SPSAProbePlan(
        incumbent=current,
        delta=delta,
        h=h,
        plus=current + offset,
        minus=current - offset,
        line_search_budget=min(len(alphas), remaining - 2),
        unused_budget=0,
    )


def derive_spsa_proposal(
    plan: SPSAProbePlan,
    plus_loss: float | None,
    minus_loss: float | None,
    step_radius: float = SPSA_STEP_RADIUS,
) -> SPSAProposal:
    """Derive the fixed-radius normalized descent proposal from two probes."""

    zero_gradient = np.zeros(DIMENSION, dtype=np.float64)
    if not plan.has_probes or plan.h is None or plan.delta is None:
        return SPSAProposal(
            params=None,
            preclip_params=None,
            gradient=zero_gradient,
            valid=False,
            reason="insufficient_budget",
        )
    if not _valid_loss(plus_loss) or not _valid_loss(minus_loss):
        return SPSAProposal(
            params=None,
            preclip_params=None,
            gradient=zero_gradient,
            valid=False,
            reason="invalid_probe_loss",
        )

    difference = float(plus_loss) - float(minus_loss)
    gradient = zero_gradient.copy()
    active = plan.h > 0.0
    gradient[active] = difference / (2.0 * plan.h[active] * plan.delta[active])
    gradient_norm = float(np.linalg.norm(gradient))
    if not np.isfinite(gradient_norm) or gradient_norm == 0.0:
        return SPSAProposal(
            params=None,
            preclip_params=None,
            gradient=gradient,
            valid=False,
            reason="zero_gradient",
        )

    preclip = plan.incumbent - step_radius * gradient / gradient_norm
    return SPSAProposal(
        params=np.clip(preclip, 0.0, 1.0),
        preclip_params=preclip,
        gradient=gradient,
        valid=True,
        reason=None,
    )


def select_spsa_successor(
    incumbent: ArrayLike,
    incumbent_loss: float,
    chronological_candidates: Sequence[ArrayLike],
    candidate_losses: Sequence[float | None],
) -> CandidateSelection:
    """Choose the lowest finite observed loss, keeping the earliest exact tie."""

    current = _normalized_vector(incumbent, "incumbent")
    if not _valid_loss(incumbent_loss):
        raise ValueError("incumbent_loss must be finite")
    if len(chronological_candidates) != len(candidate_losses):
        raise ValueError("chronological_candidates and candidate_losses must have equal length")

    best_params = current
    best_loss = float(incumbent_loss)
    best_index: int | None = None
    for index, (candidate, loss) in enumerate(zip(chronological_candidates, candidate_losses)):
        if not _valid_loss(loss):
            continue
        try:
            params = _normalized_vector(candidate, f"candidate[{index}]")
        except ValueError:
            continue
        numeric_loss = float(loss)
        if numeric_loss < best_loss:
            best_params = params
            best_loss = numeric_loss
            best_index = index

    return CandidateSelection(
        params=best_params,
        loss=best_loss,
        candidate_index=best_index,
        improved=best_index is not None,
    )
