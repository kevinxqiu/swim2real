import json
from pathlib import Path

import numpy as np
import pytest

from swim2real.config import RunConfig, load_config
from swim2real.data import Dataset
from swim2real.parameters import Bounds16D
from swim2real.simulators.fishcreate import FishCreateSimulator


@pytest.mark.parametrize(
    "overrides",
    [
        {"budget": True},
        {"budget": 0},
        {"seeds": [1, 1]},
        {"seeds": [True]},
        {"seeds": [-1]},
        {"seeds": "1"},
        {"frequencies_hz": [0.5, 0.5]},
        {"frequencies_hz": [0.0]},
        {"frequencies_hz": [0.5, 0.75], "method": "full_vlm_ls"},
        {"backtracking_alphas": [0.5, 1.0]},
        {"backtracking_alphas": [1.1]},
        {"random_radii_normalized": [-0.3]},
        {"spsa": {"probe_radius_normalized": 0.6}},
        {"spsa": {"step_radius_normalized": True}},
        {"bayesopt": {"n_initial_points": 0}},
        {"cmaes": {"populations": [1]}},
        {"simulator": {"fps": 0}},
        {"simulator": {"duration_s": 0}},
        {"simulator": {"backend": "missing"}},
        {"vlm": {"temperature": "0"}},
        {"vlm": {"max_transport_attempts": 0}},
        {"vlm": {"timeout_seconds": -1}},
        {"vlm": {"max_output_tokens": True}},
        {"provider": "missing"},
        {"openai_compat": {"api_key_env": "bad name"}},
        {"openai_compat": {"base_url": "file:///endpoint"}},
        {"openai_compat": {"jpeg_quality": 101}},
        {"openai_compat": {"frames_per_clip": 1}},
        {"metric": {"head_markers": 10}},
        {"metric": {"sim_centerline_indices": [0] * 9}},
        {"use_media": "false"},
        {"unknown": 1},
    ],
)
def test_configuration_rejects_invalid_settings(overrides):
    with pytest.raises((ValueError, TypeError)):
        RunConfig.from_dict(overrides)


def test_config_paths_are_relative_to_source(tmp_path):
    marker = tmp_path / "markers.csv"
    np.savetxt(marker, np.arange(9 * 6 * 2).reshape(9, -1), delimiter=",")
    manifest = tmp_path / "data.json"
    manifest.write_text(
        json.dumps(
            {
                "frequencies": [
                    {"frequency_hz": 0.5, "marker_csv": "markers.csv", "crop_window": [1, 5]}
                ]
            }
        )
    )
    source = tmp_path / "config.json"
    source.write_text(
        json.dumps(
            {"dataset": "data.json", "method": "multiscale_random_ls", "frequencies_hz": [0.5]}
        )
    )
    config = load_config(source)
    assert config.dataset == manifest
    dataset = Dataset.load(manifest, config.metric)
    assert dataset.raw_markers(0.5).shape == (9, 4, 2)
    np.testing.assert_allclose(dataset.load_markers(0.5)[:2, 0].mean(axis=0), 0.0)
    dataset.validate((0.5,))
    with pytest.raises(FileNotFoundError, match="real_video"):
        dataset.validate((0.5,), require_video=True)


def test_missing_files_name_the_source(tmp_path):
    config = RunConfig.from_dict({"dataset": "absent.json"}, tmp_path)
    assert config.dataset == tmp_path / "absent.json"
    with pytest.raises(ValueError, match="absent.json"):
        Dataset.load(config.dataset)
    source = tmp_path / "data.json"
    source.write_text(
        json.dumps({"frequencies": [{"frequency_hz": 0.5, "marker_csv": "absent.csv"}]})
    )
    with pytest.raises(FileNotFoundError, match="absent.csv"):
        Dataset.load(source)


def test_malformed_marker_files_fail_before_simulation(tmp_path):
    marker = tmp_path / "markers.csv"
    marker.write_text("nan,1\n")
    source = tmp_path / "data.json"
    source.write_text(
        json.dumps({"frequencies": [{"frequency_hz": 0.5, "marker_csv": "markers.csv"}]})
    )
    dataset = Dataset.load(source)
    with pytest.raises(ValueError, match="markers.csv"):
        dataset.validate((0.5,))


def test_physical_parameter_validation_rejects_nonfinite_and_out_of_bounds():
    bounds = Bounds16D.from_submitted_bounds()
    flat = bounds.to_flat(np.full(16, 0.5))
    for invalid in [float("nan"), -1.0, True, "1"]:
        with pytest.raises(ValueError):
            bounds.validate_flat({**flat, "fluidCoef_0": invalid})


def test_external_simulator_import_has_no_path_or_cwd_side_effects(tmp_path, monkeypatch):
    import importlib
    import sys

    from swim2real.simulators import fishcreate

    before_path = sys.path.copy()
    before_cwd = Path.cwd()
    importlib.reload(fishcreate)
    assert sys.path == before_path and Path.cwd() == before_cwd
    monkeypatch.delenv("FISHCREATE_ROOT", raising=False)
    with pytest.raises(FileNotFoundError, match="FISHCREATE_ROOT"):
        FishCreateSimulator()
    with pytest.raises(FileNotFoundError, match="_simulate.py"):
        FishCreateSimulator(tmp_path)


def test_example_run_uses_the_canonical_paper_profile(tmp_path):
    from swim2real.cli import _inputs
    from swim2real.config import PAPER_DEFAULTS

    root = Path(__file__).parents[1]
    paper = root / "src/swim2real/paper.json"
    assert paper.is_file() and not paper.is_symlink()
    assert not (root / "configs/paper.json").exists()
    assert json.loads(paper.read_text()) == PAPER_DEFAULTS
    example = json.loads((root / "configs/example_run.json").read_text())
    assert example == {"dataset": "data/dataset.json", "simulator": {"root": "/path/to/fishCREATE"}}
    simulator = tmp_path / "fishcreate"
    (simulator / "Meshes").mkdir(parents=True)
    (simulator / "_simulate.py").write_text("# simulator source\n")
    (simulator / "Meshes/auto_tendonFish.py").write_text("# model source\n")
    marker = tmp_path / "markers.csv"
    markers = np.arange(9 * 400 * 2, dtype=float).reshape(9, -1)
    np.savetxt(marker, markers, delimiter=",")
    video = tmp_path / "prepared.mp4"
    video.write_bytes(b"synthetic input-validation fixture")
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "frequencies": [
                    {"frequency_hz": frequency, "marker_csv": str(marker), "real_video": str(video)}
                    for frequency in PAPER_DEFAULTS["frequencies_hz"]
                ]
            }
        )
    )
    example["dataset"] = str(dataset)
    example["simulator"]["root"] = str(simulator)
    source = tmp_path / "run.json"
    source.write_text(json.dumps(example))
    config = load_config(source)
    assert config.dataset == dataset and config.simulator.root == simulator
    _inputs(config, require_media=True)


def test_external_simulator_timeout_is_a_recorded_frequency_failure(tmp_path, monkeypatch):
    import importlib.util
    import subprocess

    from swim2real.config import PAPER_DEFAULTS
    from swim2real.evaluator import ObjectiveEvaluator, SimulationObjective
    from swim2real.parameters import initial_vector_for_seed
    from swim2real.simulators.toy import ToyDataset

    (tmp_path / "Meshes").mkdir()
    for name in ["_simulate.py", "Meshes/auto_tendonFish.py"]:
        (tmp_path / name).write_text("# simulator source\n")
    timeout = 0.25
    config = RunConfig.from_dict(
        {
            "method": "multiscale_random_ls",
            "frequencies_hz": [0.5],
            "simulator": {"root": str(tmp_path), "timeout_seconds": timeout},
        }
    )
    simulator = FishCreateSimulator(tmp_path, timeout_seconds=config.simulator.timeout_seconds)
    original = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: object() if name == "mujoco" else original(name)
    )

    def time_out(*args, **kwargs):
        assert kwargs["timeout"] == timeout
        raise subprocess.TimeoutExpired(args[0], timeout)

    monkeypatch.setattr(subprocess, "run", time_out)
    backend = SimulationObjective(simulator, ToyDataset(config), config, tmp_path / "diagnostics")
    flat = Bounds16D.from_submitted_bounds().to_flat(initial_vector_for_seed(1248))
    result = ObjectiveEvaluator(
        backend.evaluate_frequency,
        frequencies_hz=config.frequencies_hz,
        on_call_start=backend.begin_objective,
    ).evaluate(flat)
    assert not result.valid and result.failure_count == 1
    assert result.score_mm == PAPER_DEFAULTS["failure_penalty_mm_per_frequency"]
    assert "timeout of 0.25 seconds" in result.frequency_records[0].error_message
    assert (
        json.loads((tmp_path / "diagnostics/call_0001/frequency_0.5/outcome.json").read_text())[
            "record"
        ]["valid"]
        is False
    )
