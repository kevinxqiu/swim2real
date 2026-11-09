import numpy as np
import pytest

from swim2real.config import PAPER_DEFAULTS
from swim2real.timebase import nearest_step_sample_times, resample_markers

REAL_VIDEO_FPS = PAPER_DEFAULTS["metric"]["real_fps"]


def test_nearest_step_schedule_has_no_cumulative_drift() -> None:
    times = nearest_step_sample_times(0.001, REAL_VIDEO_FPS, 5.0)
    targets = np.arange(len(times)) / REAL_VIDEO_FPS
    assert np.max(np.abs(times - targets)) <= 0.0005 + 1e-12
    assert set(np.unique(np.diff(times)).round(6)) == {0.016, 0.017}


def test_resampling_recovers_same_continuous_trajectory() -> None:
    source_time = np.arange(626) / 62.5
    source = 0.1 * np.sin(2 * np.pi * 2.25 * source_time)[:, None]
    target_time = np.arange(599) / REAL_VIDEO_FPS
    observed = resample_markers(source_time, source, target_time)[:, 0]
    expected = 0.1 * np.sin(2 * np.pi * 2.25 * target_time)
    assert np.max(np.abs(observed - expected)) < 0.001


def test_resampling_does_not_overshoot_marker_range() -> None:
    source_time = np.arange(4, dtype=float)
    source = np.array([[0.0], [1.0], [1.0], [0.0]])
    target_time = np.linspace(0.0, 3.0, 31)
    observed = resample_markers(source_time, source, target_time)
    assert np.all((0.0 <= observed) & (observed <= 1.0))


def test_requested_frequency_assertion_rejects_clipped_motor_trace() -> None:
    from swim2real.timebase import assert_requested_frequency_realized

    clipped = {
        "time": np.arange(300, dtype=float) / REAL_VIDEO_FPS,
        "motorAngularVelocity": np.full(300, 5.0),
    }
    with pytest.raises(ValueError, match="requested.*realized"):
        assert_requested_frequency_realized(clipped, 1.656)


def test_requested_frequency_assertion_accepts_matching_motor_trace() -> None:
    from swim2real.timebase import assert_requested_frequency_realized

    requested_hz = 1.656
    matching = {
        "time": np.arange(300, dtype=float) / REAL_VIDEO_FPS,
        "motorAngularVelocity": np.full(300, 2 * np.pi * requested_hz),
    }
    assert_requested_frequency_realized(matching, requested_hz)


def test_resampling_rejects_nonmonotone_target_timestamps():
    with pytest.raises(ValueError, match="increasing"):
        resample_markers([0.0, 1.0, 2.0], np.zeros((3, 2)), [1.0, -1.0])
