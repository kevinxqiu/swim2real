"""An analytic swimmer for CPU examples and tests."""

from __future__ import annotations

import numpy as np

from swim2real.config import RunConfig
from swim2real.parameters import Bounds16D
from swim2real.simulators.base import Trace


class ToySimulator:
    def run(
        self, params: dict[str, float], frequency_hz: float, duration_s: float, fps: int
    ) -> Trace:
        """Return an analytic centerline with parameter-dependent motion."""
        if (
            not np.isfinite([frequency_hz, duration_s, fps]).all()
            or min(frequency_hz, duration_s, fps) <= 0
        ):
            raise ValueError("frequency, duration and fps must be finite and positive")
        bounds = Bounds16D.from_submitted_bounds()
        bounds.validate_flat(params)
        x = bounds.to_normalized(params)
        t = np.arange(int(np.ceil(duration_s * fps)), dtype=np.float64) / fps
        marker = np.arange(11, dtype=np.float64) / 10.0
        speed = 0.01 + 0.02 * float(x[:5].mean())
        amplitude = 0.003 + 0.018 * float(x[6:11].mean())
        damping = 0.3 + float(x[11:].mean())
        phase = 2 * np.pi * frequency_hz * t[:, None] - marker[None, :] * (1 + 3 * x[5])
        positions = np.empty((len(t), 11, 2))
        positions[:, :, 0] = marker[None, :] * 0.24 - speed * t[:, None]
        positions[:, :, 1] = (
            amplitude * marker[None, :] ** 2 * np.sin(phase) / (1 + damping * frequency_hz)
        )
        trace = Trace(t, positions, np.full(len(t), 2 * np.pi * frequency_hz))
        trace.validate()
        return trace


class ToyDataset:
    """Synthetic reference trajectories for the explicitly labeled toy demo."""

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.metric = config.metric
        self._raw = {}
        params = Bounds16D.from_submitted_bounds().to_flat(np.full(16, 0.5))
        from swim2real.metrics import align_sim_markers

        for frequency in config.frequencies_hz:
            trace = ToySimulator().run(
                params,
                config.realized_frequencies_hz[frequency],
                config.simulator.duration_s,
                config.simulator.fps,
            )
            self._raw[frequency] = align_sim_markers(trace.to_log(), settings=config.metric)[
                :, : config.metric.max_real_frames
            ]

    def raw_markers(self, frequency_hz: float) -> np.ndarray:
        """Return a copy of the synthetic reference crop."""
        return self._raw[frequency_hz].copy()

    def load_markers(self, frequency_hz: float, retain_forward_motion: bool = True) -> np.ndarray:
        """Center synthetic real markers in the same coordinates as CSV inputs."""
        markers = self.raw_markers(frequency_hz)
        head = markers[: self.metric.head_markers].mean(axis=0, keepdims=True)
        markers -= head[:, 0:1] if retain_forward_motion else head
        return markers
