"""Dataset manifests and marker CSV loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from swim2real.config import MetricConfig, RunConfig, read_json


@dataclass(frozen=True)
class FrequencyData:
    frequency_hz: float
    marker_csv: Path
    real_video: Path | None
    crop_window: tuple[int, int | None]

    def load_raw_markers(self) -> NDArray[np.float64]:
        """Load the marker-major CSV as a (9, frames, 2) array in metres."""
        if not self.marker_csv.is_file():
            raise FileNotFoundError(f"Missing real marker CSV: {self.marker_csv}")
        try:
            raw = np.loadtxt(self.marker_csv, delimiter=",")
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot load marker CSV {self.marker_csv}: {exc}") from exc
        if raw.size % 18 or raw.size < 36 or not np.all(np.isfinite(raw)):
            raise ValueError(
                f"Marker CSV must contain finite 9-marker xy trajectories: {self.marker_csv}"
            )
        markers = raw.reshape(9, -1, 2)
        start, end = self.crop_window
        if start >= markers.shape[1] or (end is not None and end > markers.shape[1]):
            raise ValueError(f"Marker crop window exceeds CSV frames: {self.marker_csv}")
        markers = markers[:, start:end].copy()
        if markers.shape[1] < 2:
            raise ValueError(
                f"Marker crop window must contain at least two frames: {self.marker_csv}"
            )
        return markers


@dataclass(frozen=True)
class Dataset:
    entries: dict[float, FrequencyData]
    metric: MetricConfig

    @classmethod
    def load(cls, path: str | Path, metric: MetricConfig | None = None) -> Dataset:
        """Read explicit frequency-to-file mappings relative to the manifest."""
        source = Path(path).resolve()
        raw = read_json(source)
        if (
            not isinstance(raw, dict)
            or set(raw) != {"frequencies"}
            or not isinstance(raw["frequencies"], list)
        ):
            raise ValueError(f"Dataset manifest must contain a frequencies list: {source}")
        entries = {}
        for row in raw["frequencies"]:
            if (
                not isinstance(row, dict)
                or set(row) - {"frequency_hz", "marker_csv", "real_video", "crop_window"}
                or not {"frequency_hz", "marker_csv"} <= set(row)
            ):
                raise ValueError(f"Invalid frequency entry in {source}")
            freq = row["frequency_hz"]
            if (
                isinstance(freq, bool)
                or not isinstance(freq, (int, float))
                or not np.isfinite(freq)
                or freq <= 0
                or freq in entries
            ):
                raise ValueError(
                    f"Frequency labels must be unique finite positive numbers: {source}"
                )

            def resolve(key: str) -> Path | None:
                value = row.get(key)
                if value is None and key == "real_video":
                    return None
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{key} must name a file in {source}")
                file = (source.parent / Path(value).expanduser()).resolve()
                if not file.is_file():
                    raise FileNotFoundError(f"Missing {key} for {freq:g} Hz: {file}")
                return file

            crop = row.get("crop_window", [0, None])
            if (
                not isinstance(crop, list)
                or len(crop) != 2
                or isinstance(crop[0], bool)
                or not isinstance(crop[0], int)
                or crop[0] < 0
                or (
                    crop[1] is not None
                    and (
                        isinstance(crop[1], bool)
                        or not isinstance(crop[1], int)
                        or crop[1] <= crop[0]
                    )
                )
            ):
                raise ValueError(
                    f"crop_window must be [start_frame, exclusive_end_frame] in {source}"
                )
            entries[float(freq)] = FrequencyData(
                float(freq), resolve("marker_csv"), resolve("real_video"), tuple(crop)
            )
        if not entries:
            raise ValueError(f"Dataset manifest has no frequencies: {source}")
        return cls(entries, metric or RunConfig.defaults().metric)

    def for_frequency(self, frequency_hz: float) -> FrequencyData:
        """Return the explicitly named files for a frequency label."""
        if frequency_hz not in self.entries:
            raise ValueError(f"Dataset has no entry for {frequency_hz:g} Hz")
        return self.entries[frequency_hz]

    def raw_markers(self, frequency_hz: float) -> NDArray[np.float64]:
        """Return the configured crop, limited to the metric's comparison window."""
        return self.for_frequency(frequency_hz).load_raw_markers()[:, : self.metric.max_real_frames]

    def load_markers(
        self, frequency_hz: float, retain_forward_motion: bool = True
    ) -> NDArray[np.float64]:
        """Center pre-rotated real markers on their initial head position."""
        markers = self.raw_markers(frequency_hz)
        head = markers[: self.metric.head_markers].mean(axis=0, keepdims=True)
        markers -= head[:, 0:1] if retain_forward_motion else head
        return markers

    def validate(self, frequencies_hz: tuple[float, ...], *, require_video: bool = False) -> None:
        """Check every requested marker crop and optional prepared real video."""
        for freq in frequencies_hz:
            entry = self.for_frequency(freq)
            entry.load_raw_markers()
            if require_video and (entry.real_video is None or not entry.real_video.is_file()):
                raise FileNotFoundError(f"Missing real_video for {freq:g} Hz in dataset manifest")
