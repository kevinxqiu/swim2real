"""Timestamped simulation traces and the simulator interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Trace:
    timestamps: NDArray[np.float64]
    markers: NDArray[np.float64]
    motor_signal: NDArray[np.float64]

    def validate(self) -> None:
        """Require finite xy markers, motor samples and increasing timestamps."""
        n = len(self.timestamps)
        if (
            self.timestamps.shape != (n,)
            or n < 2
            or not np.all(np.isfinite(self.timestamps))
            or not np.all(np.diff(self.timestamps) > 0)
        ):
            raise ValueError("Simulation timestamps must be finite and strictly increasing")
        if self.markers.shape != (n, 11, 2) or not np.all(np.isfinite(self.markers)):
            raise ValueError("Simulation markers must be a finite (frames, 11, 2) array")
        if self.motor_signal.shape != (n,) or not np.all(np.isfinite(self.motor_signal)):
            raise ValueError("Simulation motor signal must match the timestamps")

    def to_log(self) -> dict[str, NDArray[np.float64]]:
        """Return marker and angular-velocity fields used by the metric."""
        self.validate()
        return {
            "time": self.timestamps,
            "markers": self.markers.reshape(-1, 22),
            "motorAngularVelocity": self.motor_signal,
        }


class Simulator(Protocol):
    def run(
        self, params: dict[str, float], frequency_hz: float, duration_s: float, fps: int
    ) -> Trace:
        """Simulate physical parameters at a realized motor frequency."""
        ...
