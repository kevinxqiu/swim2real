import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from swim2real.cli import main, make_parser
from swim2real.parameters import Bounds16D


@pytest.mark.parametrize(
    "command",
    [[], ["calibrate"], ["evaluate"], ["validate-inputs"], ["reproduce"], ["check-numbers"]],
)
def test_all_help_commands_work_without_external_inputs(command, monkeypatch, tmp_path):
    from swim2real.config import PAPER_DEFAULTS

    monkeypatch.setenv("FISHCREATE_ROOT", str(tmp_path / "missing-simulator"))
    monkeypatch.setitem(PAPER_DEFAULTS, "dataset", str(tmp_path / "missing-dataset.json"))
    assert make_parser()
    with pytest.raises(SystemExit) as result:
        main([*command, "--help"])
    assert result.value.code == 0


def test_toy_cli_calibrates_evaluates_and_validates(tmp_path, capsys):
    assert (
        main(
            [
                "calibrate",
                "--simulator",
                "toy",
                "--method",
                "multiscale_random_ls",
                "--budget",
                "3",
                "--output",
                str(tmp_path / "run"),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["calls_completed"] == 3 and result["synthetic_demo"]
    assert (tmp_path / "run" / "config.json").is_file()
    params = tmp_path / "params.json"
    params.write_text(json.dumps(Bounds16D.from_submitted_bounds().to_flat(np.full(16, 0.5))))
    assert main(["evaluate", "--simulator", "toy", "--params", str(params)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(result["frequency_records"]) == 8 and result["valid"]
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"method": "multiscale_random_ls", "simulator": {"backend": "toy"}})
    )
    assert main(["validate-inputs", "--config", str(config)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"]


def test_analysis_hook_is_called_when_module_exists(monkeypatch):
    package = types.ModuleType("swim2real.analysis")
    package.__path__ = []
    module = types.ModuleType("swim2real.analysis.cli")
    import importlib.machinery

    module.__spec__ = importlib.machinery.ModuleSpec(module.__name__, loader=None)
    called = []

    def add(subparsers):
        called.append(True)
        subparsers.add_parser("analyze-test")

    module.add_subcommands = add
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert make_parser().parse_args(["analyze-test"]).command == "analyze-test"
    assert called == [True]


def test_cli_resumes_write_ahead_state_before_call_file(tmp_path, monkeypatch, capsys):
    from swim2real.search import store

    original = store._atomic_write_json

    def interrupt(path, value):
        if path.parent.name == "calls" and path.name == "call_0001.json":
            raise OSError("injected interruption after write-ahead state")
        original(path, value)

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "method": "multiscale_random_ls",
                "budget": 3,
                "simulator": {"backend": "toy", "fps": 30, "duration_s": 3.0},
                "metric": {"max_real_frames": 60},
            }
        )
    )
    arguments = ["calibrate", "--config", str(config), "--output", str(tmp_path / "run")]
    monkeypatch.setattr(store, "_atomic_write_json", interrupt)
    with pytest.raises(SystemExit) as result:
        main(arguments)
    assert result.value.code == 2
    assert (tmp_path / "run" / "state" / "state_0001.json").is_file()
    assert not (tmp_path / "run" / "calls" / "call_0001.json").exists()
    monkeypatch.setattr(store, "_atomic_write_json", original)
    assert main([*arguments, "--resume"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["calls_completed"] == 3


def test_analysis_commands_run_without_dataset_or_simulator(tmp_path, monkeypatch, capsys):
    from swim2real.analysis.numbers import compute_numbers
    from swim2real.analysis.records import load_records
    from swim2real.config import PAPER_DEFAULTS

    monkeypatch.setenv("FISHCREATE_ROOT", str(tmp_path / "absent-simulator"))
    monkeypatch.setitem(PAPER_DEFAULTS, "dataset", str(tmp_path / "absent-dataset.json"))
    records = Path(__file__).parent / "analysis/fixtures/runs.zip"
    values = compute_numbers(load_records(records, allow_partial=True), allow_partial=True)
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps({"values": {key: {"value": value} for key, value in values.items()}})
    )
    numbers = tmp_path / "numbers.json"
    assert (
        main(
            [
                "reproduce",
                "--records",
                str(records),
                "--reference",
                str(reference),
                "--allow-partial",
                "--out-json",
                str(numbers),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["matched"] == len(values)
    assert main(["check-numbers", str(numbers), "--reference", str(reference)]) == 0
    assert json.loads(capsys.readouterr().out)["differing"] == 0


@pytest.mark.parametrize("kind", ["config", "numbers", "records", "number_row"])
def test_cli_reports_malformed_user_json_without_tracebacks(tmp_path, capsys, kind):
    source = tmp_path / "malformed.json"
    source.write_text("null" if kind == "config" else "{}")
    if kind == "config":
        args = ["validate-inputs", "--config", str(source)]
    elif kind in {"numbers", "number_row"}:
        reference = tmp_path / "reference.json"
        if kind == "number_row":
            source.write_text('{"key": 0}')
            reference.write_text('{"values": {"key": null}}')
        else:
            reference.write_text("{}")
        args = ["check-numbers", str(source), "--reference", str(reference)]
    else:
        import zipfile

        root = Path(__file__).parent / "analysis/fixtures/runs.zip"
        archive = tmp_path / "malformed.zip"
        with zipfile.ZipFile(root) as original, zipfile.ZipFile(archive, "w") as output:
            for name in original.namelist():
                value = original.read(name)
                if name.startswith("runs/") and name.endswith(".json"):
                    value = b"{}"
                output.writestr(name, value)
        args = [
            "reproduce",
            "--records",
            str(archive),
            "--out-json",
            str(tmp_path / "numbers.json"),
        ]
    with pytest.raises(SystemExit) as result:
        main(args)
    assert result.value.code == 2
    error = capsys.readouterr().err
    assert error.startswith("swim2real:") and "Traceback" not in error
    if kind != "config":
        assert "Invalid JSON input" in error


def test_evaluate_accepts_calibrated_example_parameters(capsys):
    params = Path(__file__).parents[1] / "configs/example_params.json"
    assert main(["evaluate", "--simulator", "toy", "--params", str(params)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] and result["params"] == json.loads(params.read_text())


def test_example_parameters_match_the_best_recorded_replication_candidate():
    import os
    import zipfile

    root = Path(__file__).parents[1]
    source = Path(os.environ.get("SWIM2REAL_RECORDS", root / "records/runs.zip"))
    if not source.is_file():
        pytest.skip("Set SWIM2REAL_RECORDS to verify example parameters against the full archive")
    with zipfile.ZipFile(source) as archive:
        run = json.loads(archive.read("runs/replication/full_vlm_ls/seed_1290.json"))
        keys = json.loads(archive.read("schema.json"))["parameter_keys"]
    best = min(
        (call for call in run["evaluations"] if call["valid"]),
        key=lambda call: call["mean_error_mm"],
    )
    assert json.loads((root / "configs/example_params.json").read_text()) == dict(
        zip(keys, best["parameters"])
    )
