"""Head-frame alignment and the paper's mean marker L2 metric."""

from __future__ import annotations

import numpy as np

from swim2real.config import MetricConfig, RunConfig
from swim2real.data import Dataset
from swim2real.timebase import resample_markers

DEFAULT_METRIC = RunConfig.defaults().metric
REAL_FPS = DEFAULT_METRIC.real_fps
SIM_FPS = RunConfig.defaults().simulator.fps


def align_sim_markers(
    log: dict, retain_forward_motion: bool = True, settings: MetricConfig = DEFAULT_METRIC
) -> np.ndarray:
    """Rotate the simulated centerline into the head frame."""
    raw = np.asarray(log["markers"], dtype=np.float64)
    if raw.size == 0 or raw.size % 22 or not np.all(np.isfinite(raw)):
        raise ValueError("simulation markers must contain finite 11-marker xy frames")
    all_markers = raw.reshape(-1, 11, 2).transpose(1, 0, 2)
    markers = all_markers[list(settings.sim_centerline_indices)]
    head0 = markers[0]
    shifted = markers - head0
    head_vec = np.mean(shifted[1 : settings.head_markers], axis=0)
    norms = np.linalg.norm(head_vec, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    head_vec /= norms
    rot = np.stack(
        [head_vec[:, 0], head_vec[:, 1], -head_vec[:, 1], head_vec[:, 0]], axis=1
    ).reshape(-1, 2, 2)
    rotated = np.stack(
        [
            np.matmul(rot, shifted[i].reshape(-1, 2, 1)).reshape(-1, 2)
            for i in range(shifted.shape[0])
        ],
        axis=0,
    )
    rotated_head0 = np.matmul(rot[0:1], head0.reshape(-1, 2, 1)).reshape(-1, 2)
    rotated[:, :, 0] += rotated_head0[:, 0]
    head_centroid = rotated[: settings.head_markers].mean(axis=0, keepdims=True)
    if retain_forward_motion:
        rotated -= head_centroid[:, 0:1]
    else:
        rotated -= head_centroid
    return rotated


def _find_phase_offset(sim_aligned: np.ndarray, real: np.ndarray, step: int = 3) -> int:
    """Find the lowest-error temporal offset, keeping the first tie."""
    n_real = real.shape[1]
    n_sim = sim_aligned.shape[1]
    max_offset = n_sim - n_real
    if max_offset <= 0:
        return 0
    best_offset = 0
    best_error = np.inf
    for offset in range(0, max_offset + 1, step):
        sim_window = sim_aligned[:, offset : offset + n_real]
        err = np.mean(np.linalg.norm(sim_window - real, axis=2))
        if err < best_error:
            best_error = err
            best_offset = offset
    fine_start = max(0, best_offset - step)
    fine_end = min(max_offset, best_offset + step)
    for offset in range(fine_start, fine_end + 1):
        sim_window = sim_aligned[:, offset : offset + n_real]
        err = np.mean(np.linalg.norm(sim_window - real, axis=2))
        if err < best_error:
            best_error = err
            best_offset = offset
    return best_offset


def _find_phase_offset_real(sim_aligned: np.ndarray, real: np.ndarray, step: int = 3) -> int:
    """Find the lowest-error temporal offset, keeping the first tie."""
    n_sim = sim_aligned.shape[1]
    n_real = real.shape[1]
    max_offset = n_real - n_sim
    if max_offset <= 0:
        return 0
    best_offset = 0
    best_error = np.inf
    for offset in range(0, max_offset + 1, step):
        real_window = real[:, offset : offset + n_sim]
        err = np.mean(np.linalg.norm(sim_aligned - real_window, axis=2))
        if err < best_error:
            best_error = err
            best_offset = offset
    fine_start = max(0, best_offset - step)
    fine_end = min(max_offset, best_offset + step)
    for offset in range(fine_start, fine_end + 1):
        real_window = real[:, offset : offset + n_sim]
        err = np.mean(np.linalg.norm(sim_aligned - real_window, axis=2))
        if err < best_error:
            best_error = err
            best_offset = offset
    return best_offset


def compute_error(
    sim_log: dict,
    frequency: float,
    sim_fps: float = SIM_FPS,
    real_fps: float = REAL_FPS,
    auto_align: bool = True,
    retain_forward_motion: bool = True,
    *,
    dataset: Dataset,
    settings: MetricConfig | None = None,
) -> dict:
    """Align physical-time trajectories and average marker L2 distances in metres."""
    settings = settings or dataset.metric
    if not np.isfinite([sim_fps, real_fps]).all() or min(sim_fps, real_fps) <= 0:
        raise ValueError("frame rates must be finite and positive")
    real_full = dataset.load_markers(frequency, retain_forward_motion=retain_forward_motion)
    sim_full = align_sim_markers(
        sim_log, retain_forward_motion=retain_forward_motion, settings=settings
    )
    sim_time = np.asarray(sim_log.get("time"), dtype=np.float64)
    if (
        sim_time.ndim != 1
        or len(sim_time) != sim_full.shape[1]
        or len(sim_time) < 2
        or (not np.all(np.isfinite(sim_time)))
        or (not np.all(np.diff(sim_time) > 0.0))
    ):
        raise ValueError("simulation log requires finite, strictly increasing timestamps")
    transient_start = sim_time[0] + settings.transient_s
    transient_index = int(np.searchsorted(sim_time, transient_start, side="left"))
    if transient_index >= len(sim_time) - 1:
        transient_index = 0
    sim_time = sim_time[transient_index:] - sim_time[transient_index]
    sim_full = sim_full[:, transient_index:]
    sim_target = np.arange(int(np.floor(sim_time[-1] * real_fps)) + 1, dtype=np.float64) / real_fps
    sim_full = resample_markers(sim_time, np.moveaxis(sim_full, 1, 0), sim_target)
    sim_full = np.moveaxis(sim_full, 0, 1)
    if not retain_forward_motion:
        sim_full[:, :, 1] -= sim_full[:, :, 1].mean(axis=0, keepdims=True)
        real_full[:, :, 1] -= real_full[:, :, 1].mean(axis=0, keepdims=True)
    sim_usable = sim_full
    if retain_forward_motion:
        sim_usable = sim_usable.copy()
        sim_origin = sim_usable[: settings.head_markers, 0:1].mean(axis=0, keepdims=True)
        sim_usable -= sim_origin
        real_full = real_full.copy()
        real_origin = real_full[: settings.head_markers, 0:1].mean(axis=0, keepdims=True)
        real_full -= real_origin
    n_sim = sim_usable.shape[1]
    n_real = real_full.shape[1]
    compare_len = min(n_sim, n_real)
    if auto_align:
        if n_sim >= n_real:
            best_offset = _find_phase_offset(sim_usable, real_full, step=3)
            sim_window = sim_usable[:, best_offset : best_offset + n_real]
            real_window = real_full
        else:
            best_offset = _find_phase_offset_real(sim_usable, real_full, step=3)
            sim_window = sim_usable
            real_window = real_full[:, best_offset : best_offset + n_sim]
    else:
        best_offset = 0
        sim_window = sim_usable[:, :compare_len]
        real_window = real_full[:, :compare_len]
    n_frames = min(sim_window.shape[1], real_window.shape[1])
    sim_window = sim_window[:, :n_frames]
    real_window = real_window[:, :n_frames]
    l2 = np.linalg.norm(sim_window - real_window, axis=2)
    sim_vel = compute_forward_velocity_sim(sim_log, fps=sim_fps, settings=settings)
    real_vel = compute_forward_velocity_real(frequency, dataset=dataset, fps=real_fps)
    return {
        "mean_l2": float(np.mean(l2)),
        "per_marker_l2": np.mean(l2, axis=1),
        "per_frame_l2": np.mean(l2, axis=0),
        "n_frames": n_frames,
        "offset_frames": best_offset,
        "offset_seconds": best_offset / real_fps,
        "comparison_fps": float(real_fps),
        "sim_markers": sim_window,
        "real_markers": real_window,
        "sim_velocity": sim_vel,
        "real_velocity": real_vel,
        "velocity_error": abs(sim_vel - real_vel),
    }


def compute_forward_velocity_sim(
    log: dict, fps: float = SIM_FPS, settings: MetricConfig = DEFAULT_METRIC
) -> float:
    """Fit forward velocity from simulated head positions and timestamps."""
    raw = np.asarray(log["markers"], dtype=np.float64)
    if raw.size == 0 or raw.size % 22 or not np.all(np.isfinite(raw)):
        raise ValueError("simulation markers must contain finite 11-marker xy frames")
    all_markers = raw.reshape(-1, 11, 2).transpose(1, 0, 2)
    markers = all_markers[list(settings.sim_centerline_indices)]
    head0 = markers[0]
    shifted = markers - head0
    head_vec = np.mean(shifted[1 : settings.head_markers], axis=0)
    norms = np.linalg.norm(head_vec, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    head_vec /= norms
    rot = np.stack(
        [head_vec[:, 0], head_vec[:, 1], -head_vec[:, 1], head_vec[:, 0]], axis=1
    ).reshape(-1, 2, 2)
    rotated_head0 = np.matmul(rot[0:1], head0.reshape(-1, 2, 1)).reshape(-1, 2)
    head_x = rotated_head0[:, 0]
    logged_time = np.asarray(log.get("time"), dtype=np.float64)
    if logged_time.shape == head_x.shape and np.all(np.isfinite(logged_time)):
        time_axis = logged_time - logged_time[0]
    else:
        time_axis = np.arange(len(head_x)) / fps
    coef = np.polyfit(time_axis, -head_x, 1)
    return float(coef[0])


def compute_forward_velocity_real(
    frequency: float, *, dataset: Dataset, fps: float = REAL_FPS
) -> float:
    """Fit forward velocity from the real head-marker centroid."""
    markers = dataset.raw_markers(frequency)
    head_x = markers[: dataset.metric.head_markers].mean(axis=0)[:, 0]
    time_axis = np.arange(len(head_x)) / fps
    return float(np.polyfit(time_axis, -head_x, 1)[0])
