import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from swim2real.media import build_media, media_manifest, render_skeleton_overlay
from swim2real.providers.base import _canonical_sha256

FIXTURES = Path(__file__).parent / "fixtures"


def sweep_fixture():
    context = json.loads((FIXTURES / "prompt_context.json").read_text())
    with np.load(FIXTURES / "markers.npz") as saved:
        real = saved["real"][:, :30]
    return [
        {
            **row,
            "diagnostic_sha256": _canonical_sha256(row),
            "real_markers": real,
            "sim_markers": real + (0.001 + i * 0.0001),
        }
        for i, row in enumerate(context["diagnostics"])
    ]


def test_media_manifest_matches_original(tmp_path):
    sweep = sweep_fixture()
    paths = [
        tmp_path / "real_f2.25.mp4",
        *[tmp_path / f"skeleton_overlay_f{row['frequency']:.2f}.mp4" for row in sweep],
    ]
    for path in paths:
        path.write_bytes(b"synthetic media fixture\n")
    assert media_manifest([str(p) for p in paths], sweep, 2.25, fps=60) == json.loads(
        (FIXTURES / "media_manifest.json").read_text()
    )


def test_media_uses_copies_and_checks_sources_on_resume(tmp_path, monkeypatch):
    source = tmp_path / "prepared.mp4"
    source.write_bytes(b"synthetic prepared video")
    dataset = SimpleNamespace(for_frequency=lambda freq: SimpleNamespace(real_video=source))

    def render(sim, real, destination, fps, width, height):
        Path(destination).write_bytes(b"synthetic overlay video")

    monkeypatch.setattr("swim2real.media.render_skeleton_overlay", render)

    def no_link(*args, **kwargs):
        raise AssertionError("Hard links must not be used")

    monkeypatch.setattr(os, "link", no_link)
    paths, worst, freqs = build_media(sweep_fixture(), dataset, tmp_path / "media", 60)
    assert len(paths) == 9 and worst == 2.25
    assert os.stat(source).st_ino != os.stat(paths[0]).st_ino
    assert build_media(sweep_fixture(), dataset, tmp_path / "media", 60)[0] == paths
    source.write_bytes(b"changed source video")
    with pytest.raises(ValueError, match="changed"):
        build_media(sweep_fixture(), dataset, tmp_path / "media", 60)


def test_missing_ffmpeg_is_an_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setattr("swim2real.media.shutil.which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="ffmpeg"):
        render_skeleton_overlay(
            np.ones((9, 5, 2)), np.ones((9, 5, 2)), str(tmp_path / "overlay.mp4")
        )


def test_real_overlay_uses_comparison_timebase_and_configured_size(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg is required for real H.264 overlay encoding")
    cv2 = pytest.importorskip(
        "cv2", reason="OpenCV (swim2real[sim]) is required for overlay encoding"
    )
    from swim2real.config import RunConfig
    from swim2real.evaluator import ObjectiveEvaluator, SimulationObjective
    from swim2real.parameters import Bounds16D, initial_vector_for_seed
    from swim2real.simulators.toy import ToyDataset, ToySimulator

    config = RunConfig.from_dict(
        {
            "method": "multiscale_random_ls",
            "frequencies_hz": [0.5],
            "simulator": {"backend": "toy", "fps": 30, "duration_s": 4.0},
            "metric": {"max_real_frames": 90},
            "media": {"width": 320, "height": 180},
        }
    )
    backend = SimulationObjective(
        ToySimulator(), ToyDataset(config), config, tmp_path / "diagnostics"
    )
    flat = Bounds16D.from_submitted_bounds().to_flat(initial_vector_for_seed(1248))
    call = ObjectiveEvaluator(
        backend.evaluate_frequency,
        frequencies_hz=config.frequencies_hz,
        on_call_start=backend.begin_objective,
    ).evaluate(flat)
    assert call.valid
    detail = backend.details_for(flat, call.call_index)[0.5]
    assert detail["videofps"] == config.metric.real_fps
    assert detail["sim_markers"].shape[1] == 90
    # Use a valid video for both the copied recording and the encoded overlay.
    source = tmp_path / "source.mp4"
    render_skeleton_overlay(
        detail["real_markers"],
        detail["real_markers"],
        str(source),
        fps=config.metric.real_fps,
        width=config.media.width,
        height=config.media.height,
    )
    dataset = SimpleNamespace(for_frequency=lambda freq: SimpleNamespace(real_video=source))
    paths, _, _ = build_media(
        [detail],
        dataset,
        tmp_path / "media",
        config.metric.real_fps,
        width=config.media.width,
        height=config.media.height,
    )
    capture = cv2.VideoCapture(paths[1])
    try:
        assert capture.isOpened()
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        assert count == 90
        assert fps == pytest.approx(config.metric.real_fps, rel=1e-5)
        assert count / fps == pytest.approx(90 / config.metric.real_fps, rel=1e-5)
        assert capture.get(cv2.CAP_PROP_FRAME_WIDTH) == config.media.width
        assert capture.get(cv2.CAP_PROP_FRAME_HEIGHT) == config.media.height
        assert capture.read()[0]
    finally:
        capture.release()
    row = json.loads((tmp_path / "media/manifest.json").read_text())["media"][1]
    assert row["fps"] == config.metric.real_fps and row["frame_count"] == 90
    assert row["width"] == config.media.width and row["height"] == config.media.height
