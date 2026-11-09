"""Physical parameters and normalized coordinates."""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

# fmt: off
_JOINT_NAMES = ["body-tail0", "tail0-tail1", "tail1-tail2", "tail2-tail3", "tail3-tail4"]
PARAM_INFO_16D = {
    "fluidCoef_0": (0.0, 10.0, "Pressure drag on blunt surfaces"),
    "fluidCoef_1": (0.0, 10.0, "Skin friction on slender surfaces"),
    "fluidCoef_2": (0.0, 10.0, "Rotational drag in water"),
    "fluidCoef_3": (0.0, 10.0, "Kutta lift (circulation/fin effect)"),
    "fluidCoef_4": (0.0, 10.0, "Magnus lift (spinning body parts)"),
    "motorArmLength": (0.01, 0.06, "Crank arm driving tendons (m)"),
    "hingeStiffness_0": (0.1, 5.0, f"Joint 0 ({_JOINT_NAMES[0]}) stiffness"),
    "hingeStiffness_1": (0.1, 5.0, f"Joint 1 ({_JOINT_NAMES[1]}) stiffness"),
    "hingeStiffness_2": (0.1, 5.0, f"Joint 2 ({_JOINT_NAMES[2]}) stiffness"),
    "hingeStiffness_3": (0.1, 5.0, f"Joint 3 ({_JOINT_NAMES[3]}) stiffness"),
    "hingeStiffness_4": (0.1, 5.0, f"Joint 4 ({_JOINT_NAMES[4]}) stiffness"),
    "hingeDamping_0": (0.0, 2.0, f"Joint 0 ({_JOINT_NAMES[0]}) damping"),
    "hingeDamping_1": (0.0, 2.0, f"Joint 1 ({_JOINT_NAMES[1]}) damping"),
    "hingeDamping_2": (0.0, 2.0, f"Joint 2 ({_JOINT_NAMES[2]}) damping"),
    "hingeDamping_3": (0.0, 2.0, f"Joint 3 ({_JOINT_NAMES[3]}) damping"),
    "hingeDamping_4": (0.0, 2.0, f"Joint 4 ({_JOINT_NAMES[4]}) damping"),
}
# fmt: on
PARAM_BOUNDS_16D = {key: (lo, hi) for key, (lo, hi, _) in PARAM_INFO_16D.items()}
JOINT_NAMES = tuple(_JOINT_NAMES)
DIMENSION = len(PARAM_INFO_16D)


class ParameterError(ValueError):
    """A parameter vector has an invalid shape or value."""


@dataclass(frozen=True)
class Bounds16D:
    keys: tuple[str, ...]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]

    @classmethod
    def from_submitted_bounds(cls) -> Bounds16D:
        """Return the paper's ordered physical bounds."""
        keys = tuple(PARAM_BOUNDS_16D)
        return cls(
            keys,
            np.array([PARAM_BOUNDS_16D[k][0] for k in keys]),
            np.array([PARAM_BOUNDS_16D[k][1] for k in keys]),
        )

    def to_flat(self, normalized: ArrayLike) -> dict[str, float]:
        """Convert a bounded normalized vector to physical parameters."""
        vector = _vector(normalized)
        if np.any(vector < 0) or np.any(vector > 1):
            raise ParameterError("normalized parameters must lie in [0, 1]")
        raw = self.lower + vector * (self.upper - self.lower)
        return {key: float(raw[i]) for i, key in enumerate(self.keys)}

    def to_normalized(self, flat: dict[str, float]) -> NDArray[np.float64]:
        """Normalize a complete physical parameter mapping, clipping excursions."""
        if set(flat) != set(self.keys):
            raise ParameterError("physical parameters must contain exactly the 16 parameter names")
        if any(isinstance(v, (bool, str)) for v in flat.values()):
            raise ParameterError("physical parameters must be finite numbers")
        raw = _vector([flat[key] for key in self.keys])
        return np.clip((raw - self.lower) / (self.upper - self.lower), 0.0, 1.0)

    def validate_flat(self, flat: dict[str, float]) -> None:
        """Reject incomplete, nonnumeric or out-of-bounds physical values."""
        self.to_normalized(flat)
        raw = np.array([flat[k] for k in self.keys])
        if np.any(raw < self.lower) or np.any(raw > self.upper):
            raise ParameterError("physical parameters exceed their bounds")


def _vector(value: ArrayLike) -> NDArray[np.float64]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (DIMENSION,) or not np.all(np.isfinite(vector)):
        raise ParameterError("expected one finite 16-dimensional vector")
    return vector.copy()


def normalize_params(raw: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> NDArray[np.float64]:
    """Normalize and clip physical coordinates."""
    raw, lo, hi = _vector(raw), _vector(lower), _vector(upper)
    if np.any(hi <= lo):
        raise ParameterError("upper bounds must exceed lower bounds")
    return np.clip((raw - lo) / (hi - lo), 0.0, 1.0)


def denormalize_params(
    normalized: ArrayLike, lower: ArrayLike, upper: ArrayLike
) -> NDArray[np.float64]:
    """Clip normalized coordinates and return physical values."""
    lo, hi = _vector(lower), _vector(upper)
    if np.any(hi <= lo):
        raise ParameterError("upper bounds must exceed lower bounds")
    return lo + np.clip(_vector(normalized), 0.0, 1.0) * (hi - lo)


def initial_vector_for_seed(
    problem_seed: int, bounds: Bounds16D | None = None
) -> NDArray[np.float64]:
    """Draw the paired initialization using Python's seeded random stream."""
    if isinstance(problem_seed, bool) or not isinstance(problem_seed, int) or problem_seed < 0:
        raise ParameterError("problem_seed must be a nonnegative integer")
    bounds = bounds or Bounds16D.from_submitted_bounds()
    rng = random.Random(problem_seed)
    raw = np.asarray(
        [rng.uniform(lo, hi) for lo, hi in zip(bounds.lower, bounds.upper)], dtype=np.float64
    )
    return (raw - bounds.lower) / (bounds.upper - bounds.lower)


def to_simulator_params(flat: dict[str, float], base_params: dict[str, Any]) -> dict[str, Any]:
    """Apply the 16 physical values to a simulator's parameter dictionary."""
    Bounds16D.from_submitted_bounds().validate_flat(flat)
    params = deepcopy(base_params)
    params["fluidCoef"] = [flat[f"fluidCoef_{i}"] for i in range(5)]
    params["motorArmLength"] = flat["motorArmLength"]
    params["hingeStiffness"] = [flat[f"hingeStiffness_{i}"] for i in range(5)]
    params["hingeDamping"] = [flat[f"hingeDamping_{i}"] for i in range(5)]
    return params
