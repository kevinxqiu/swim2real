"""Physical timestamps and motor-frequency checks."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


def estimate_dominant_frequency_hz(signal: ArrayLike, fps: float) -> float:
    """Estimate a periodic signal's dominant frequency with FFT peak interpolation."""

    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or len(values) < 3:
        raise ValueError("frequency signal must be one-dimensional with at least three samples")
    if not np.all(np.isfinite(values)) or not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("frequency signal and FPS must be finite, with positive FPS")
    centered = values - np.mean(values)
    spectrum = np.abs(np.fft.rfft(centered * np.hanning(len(centered))))
    spectrum[0] = 0.0
    peak = int(np.argmax(spectrum))
    if peak == 0 or spectrum[peak] <= 0.0:
        raise ValueError("frequency signal has no non-constant spectral content")
    offset = 0.0
    if peak < len(spectrum) - 1:
        left, center, right = np.log(spectrum[peak - 1 : peak + 2] + np.finfo(float).tiny)
        denominator = left - 2.0 * center + right
        if denominator != 0.0:
            offset = 0.5 * (left - right) / denominator
    return float((peak + offset) * fps / len(centered))


def measured_motor_frequency_hz(log: Mapping[str, ArrayLike], settle_seconds: float = 1.0) -> float:
    """Return steady-state motor frequency from sampled joint angular velocity."""

    try:
        time = np.asarray(log["time"], dtype=np.float64)
        angular_velocity = np.asarray(log["motorAngularVelocity"], dtype=np.float64)
    except KeyError as exc:
        raise ValueError("simulation log lacks measured motor frequency fields") from exc
    if (
        time.ndim != 1
        or angular_velocity.ndim != 1
        or len(time) != len(angular_velocity)
        or len(time) < 2
        or not np.all(np.isfinite(time))
        or not np.all(np.isfinite(angular_velocity))
        or not np.all(np.diff(time) > 0.0)
    ):
        raise ValueError("simulation motor-frequency trace is invalid")
    steady = angular_velocity[time >= time[0] + settle_seconds]
    if len(steady) == 0:
        steady = angular_velocity
    return float(np.median(np.abs(steady)) / (2.0 * np.pi))


def assert_requested_frequency_realized(
    log: Mapping[str, ArrayLike],
    requested_hz: float,
    *,
    absolute_tolerance_hz: float = 0.005,
    relative_tolerance: float = 0.01,
) -> float:
    """Fail closed unless measured motor frequency matches the requested drive."""

    requested = float(requested_hz)
    if not np.isfinite(requested) or requested <= 0.0:
        raise ValueError("requested frequency must be finite and positive")
    measured = measured_motor_frequency_hz(log)
    tolerance = max(float(absolute_tolerance_hz), relative_tolerance * requested)
    if not np.isclose(measured, requested, rtol=0.0, atol=tolerance):
        raise ValueError(
            f"requested frequency {requested:.6f} Hz was not realized; "
            f"measured {measured:.6f} Hz (tolerance {tolerance:.6f} Hz)"
        )
    return measured


def nearest_step_sample_times(dt: float, fps: float, duration: float) -> NDArray[np.float64]:
    """Return nearest simulation-step times for a uniform physical-time grid."""

    values = np.asarray([dt, fps, duration], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("dt, fps, and duration must be finite and positive")
    count = int(np.ceil(duration * fps))
    targets = np.arange(count, dtype=np.float64) / fps
    steps = np.rint(targets / dt).astype(np.int64)
    if len(np.unique(steps)) != len(steps):
        raise ValueError("requested FPS exceeds the simulation time resolution")
    return steps.astype(np.float64) * dt


def resample_markers(
    source_time: ArrayLike,
    markers: ArrayLike,
    target_time: ArrayLike,
) -> NDArray[np.float64]:
    """Linearly resample marker columns onto explicit physical timestamps."""

    source = np.asarray(source_time, dtype=np.float64)
    values = np.asarray(markers, dtype=np.float64)
    target = np.asarray(target_time, dtype=np.float64)
    if source.ndim != 1 or target.ndim != 1 or values.ndim < 2:
        raise ValueError("timestamps must be 1D and markers must have a time axis")
    if values.shape[0] != len(source) or len(source) < 2:
        raise ValueError("marker time axis must match at least two source timestamps")
    if not np.all(np.isfinite(source)) or not np.all(np.diff(source) > 0.0):
        raise ValueError("source timestamps must be finite and strictly increasing")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(target)):
        raise ValueError("marker values and target timestamps must be finite")
    if len(target) > 1 and not np.all(np.diff(target) > 0.0):
        raise ValueError("target timestamps must be strictly increasing")
    if len(target) and (target[0] < source[0] or target[-1] > source[-1]):
        raise ValueError("target timestamps must lie inside the source interval")
    flat = values.reshape(len(source), -1)
    result = np.column_stack(
        [np.interp(target, source, flat[:, index]) for index in range(flat.shape[1])]
    )
    return result.reshape((len(target), *values.shape[1:]))
