"""Diagnostic overlay videos and their content manifest."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from swim2real.config import PAPER_DEFAULTS, read_json
from swim2real.data import Dataset
from swim2real.providers.base import ProposalProviderError
from swim2real.search.store import atomic_write_json


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def media_manifest(
    media_paths: Sequence[str],
    sweep: Sequence[Mapping[str, Any]],
    worst_frequency: float,
    *,
    fps: float,
    width: int = PAPER_DEFAULTS["media"]["width"],
    height: int = PAPER_DEFAULTS["media"]["height"],
) -> list[dict[str, Any]]:
    if not media_paths:
        return []
    if len(media_paths) != len(sweep) + 1:
        raise ProposalProviderError("VLM media count does not match its diagnostic sweep")
    real_path = Path(media_paths[0])
    manifest = [
        {
            "frequency": float(worst_frequency),
            "kind": "prepared_real_video",
            "name": real_path.name,
            "sha256": sha256_file(real_path),
            "size_bytes": real_path.stat().st_size,
        }
    ]
    for path_value, detail in zip(media_paths[1:], sweep):
        path = Path(path_value)
        sim_markers = np.asarray(detail["sim_markers"], dtype=np.float64)
        real_markers = np.asarray(detail["real_markers"], dtype=np.float64)
        if sim_markers.shape != real_markers.shape:
            raise ProposalProviderError("skeleton source marker arrays differ in shape")
        manifest.append(
            {
                "diagnostic_sha256": str(detail["diagnostic_sha256"]),
                "fps": fps,
                "frame_count": int(sim_markers.shape[1]),
                "frequency": float(detail["frequency"]),
                "height": height,
                "kind": "skeleton_overlay",
                "name": path.name,
                "real_markers_sha256": str(detail["real_markers_sha256"]),
                # This identifier preserves the renderer named in the paper's media records.
                "renderer": "fish.plots.render_skeleton_overlay_v1",
                "sha256": sha256_file(path),
                "sim_markers_sha256": str(detail["sim_markers_sha256"]),
                "size_bytes": path.stat().st_size,
                "width": width,
            }
        )
    return manifest


def render_skeleton_overlay(
    sim_markers: np.ndarray,
    real_markers: np.ndarray,
    output_path: str,
    fps: float = PAPER_DEFAULTS["metric"]["real_fps"],
    width: int = PAPER_DEFAULTS["media"]["width"],
    height: int = PAPER_DEFAULTS["media"]["height"],
) -> str:
    """Render aligned real and simulated marker centerlines as H.264 video."""
    n_frames = min(sim_markers.shape[1], real_markers.shape[1])
    all_pts = np.concatenate(
        [sim_markers[:, :n_frames].reshape(-1, 2), real_markers[:, :n_frames].reshape(-1, 2)],
        axis=0,
    )
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)
    x_pad = (x_max - x_min) * 0.15
    y_pad = (y_max - y_min) * 0.15
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad
    data_aspect = (y_max - y_min) / max(x_max - x_min, 1e-10)
    vid_aspect = height / width
    if data_aspect > vid_aspect:
        x_center = (x_min + x_max) / 2
        half_w = (y_max - y_min) / vid_aspect / 2
        x_min, x_max = (x_center - half_w, x_center + half_w)
    else:
        y_center = (y_min + y_max) / 2
        half_h = (x_max - x_min) * vid_aspect / 2
        y_min, y_max = (y_center - half_h, y_center + half_h)

    def to_px(x, y):
        px = int((x - x_min) / (x_max - x_min) * width)
        py = int((y_max - y) / (y_max - y_min) * height)
        return (px, py)

    if not shutil.which("ffmpeg"):
        raise FileNotFoundError("Overlay videos require ffmpeg on PATH")
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Overlay videos require pip install 'swim2real[sim]'") from exc
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".mp4", dir=Path(output_path).parent)
    os.close(fd)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer for {output_path}")
    for i in range(n_frames):
        frame = np.full((height, width, 3), 30, dtype=np.uint8)
        diff = sim_markers[:, i] - real_markers[:, i]
        l2 = np.mean(np.linalg.norm(diff, axis=1)) * 1000
        for m in range(9):
            px, py = to_px(real_markers[m, i, 0], real_markers[m, i, 1])
            cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)
            if m > 0:
                px0, py0 = to_px(real_markers[m - 1, i, 0], real_markers[m - 1, i, 1])
                cv2.line(frame, (px0, py0), (px, py), (0, 0, 255), 2)
        for m in range(9):
            px, py = to_px(sim_markers[m, i, 0], sim_markers[m, i, 1])
            cv2.circle(frame, (px, py), 5, (255, 100, 0), -1)
            if m > 0:
                px0, py0 = to_px(sim_markers[m - 1, i, 0], sim_markers[m - 1, i, 1])
                cv2.line(frame, (px0, py0), (px, py), (255, 100, 0), 2)
        cv2.putText(frame, "Sim", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
        cv2.putText(frame, "Real", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"L2: {l2:.1f} mm",
            (width - 150, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )
        writer.write(frame)
    writer.release()
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                tmp_path,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-an",
                output_path,
            ],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg failed for {output_path}: {exc.stderr.decode(errors='replace')}"
        ) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return output_path


def build_media(
    sweep: Sequence[Mapping[str, Any]],
    dataset: Dataset,
    directory: Path,
    fps: float,
    *,
    width: int = PAPER_DEFAULTS["media"]["width"],
    height: int = PAPER_DEFAULTS["media"]["height"],
) -> tuple[list[str], float, list[float]]:
    """Copy the worst-frequency real clip and render every diagnostic overlay."""
    worst = float(max(sweep, key=lambda row: row["mean_l2"])["frequency"])
    frequencies = [float(row["frequency"]) for row in sweep]
    source = dataset.for_frequency(worst).real_video
    if source is None or not source.is_file():
        raise FileNotFoundError(f"Missing prepared real video for {worst:g} Hz: {source}")
    names = [
        f"real_f{worst:.2f}{source.suffix.lower()}",
        *[f"skeleton_overlay_f{freq:.2f}.mp4" for freq in frequencies],
    ]
    paths = [str(directory / name) for name in names]
    if directory.exists():
        actual = {p.name for p in directory.iterdir()}
        if actual != {*names, "manifest.json"}:
            raise ValueError(f"Incomplete media directory: {directory}")
        expected = {
            "media": media_manifest(paths, sweep, worst, fps=fps, width=width, height=height),
            "schema_version": 1,
        }
        if read_json(directory / "manifest.json") != expected:
            raise ValueError(f"Media content differs on resume: {directory}")
        if sha256_file(paths[0]) != sha256_file(source):
            raise ValueError(f"Real source video changed: {source}")
        return paths, worst, frequencies
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".media-", dir=directory.parent) as temp:
        staging = Path(temp) / "videos"
        staging.mkdir()
        shutil.copyfile(source, staging / names[0])
        staged = [str(staging / name) for name in names]
        for name, row in zip(names[1:], sweep):
            render_skeleton_overlay(
                row["sim_markers"],
                row["real_markers"],
                str(staging / name),
                fps=fps,
                width=width,
                height=height,
            )
        if any(not Path(p).is_file() or Path(p).stat().st_size == 0 for p in staged):
            raise RuntimeError("Media rendering produced an empty or missing file")
        atomic_write_json(
            staging / "manifest.json",
            {
                "media": media_manifest(staged, sweep, worst, fps=fps, width=width, height=height),
                "schema_version": 1,
            },
        )
        os.rename(staging, directory)
    return paths, worst, frequencies
