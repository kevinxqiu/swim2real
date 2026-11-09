import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from swim2real.cli import main
from swim2real.identity import package_identity
from swim2real.search.store import canonical_sha256
from swim2real.simulators.fishcreate import MODEL_FILES, FishCreateSimulator
from swim2real.simulators.toy import ToyDataset, ToySimulator


def test_package_identity_hashes_installed_sources_and_profile():
    import swim2real

    root = Path(swim2real.__file__).parent
    identity = package_identity()
    expected = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".json"}
    }
    assert identity["source_files"] == expected
    assert {"evaluator.py", "paper.json", "search/runner.py"} <= expected.keys()
    assert identity["source_sha256"] == canonical_sha256(expected)


def test_cli_refuses_changed_source_under_same_package_version(tmp_path, monkeypatch, capsys):
    args = [
        "calibrate",
        "--simulator",
        "toy",
        "--method",
        "multiscale_random_ls",
        "--budget",
        "1",
        "--output",
        str(tmp_path / "run"),
    ]
    assert main(args) == 0
    identity = copy.deepcopy(package_identity())
    identity["source_files"]["search/runner.py"] = "0" * 64
    identity["source_sha256"] = canonical_sha256(identity["source_files"])
    monkeypatch.setattr("swim2real.cli.package_identity", lambda: identity)
    with pytest.raises(SystemExit) as result:
        main([*args, "--resume"])
    assert result.value.code == 2
    error = capsys.readouterr().err
    assert "source_files/search/runner.py" in error
    assert "Start a new output folder" in error


@pytest.mark.parametrize("change", ["source", "commit"])
def test_cli_refuses_changed_external_simulator(tmp_path, monkeypatch, capsys, change):
    if not shutil.which("git"):
        pytest.skip("Git is required to test simulator revision identity")
    root = tmp_path / "fishcreate"
    (root / "Meshes").mkdir(parents=True)
    for name in MODEL_FILES:
        (root / name).write_text("# simulator source\n")

    def git(*args):
        return subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                *args,
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    git("init")
    git("add", ".")
    git("commit", "-m", "Initial simulator")
    simulator = FishCreateSimulator(root)
    assert simulator.identity()["git_commit"] == git("rev-parse", "HEAD")
    assert set(simulator.identity()["source_files"]) == set(MODEL_FILES)
    monkeypatch.setattr(simulator, "run", ToySimulator().run)
    monkeypatch.setattr(
        "swim2real.cli._inputs", lambda config, **kwargs: (simulator, ToyDataset(config))
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {"simulator": {"root": str(root)}, "method": "multiscale_random_ls", "budget": 1}
        )
    )
    args = ["calibrate", "--config", str(config), "--output", str(tmp_path / "run")]
    assert main(args) == 0
    if change == "source":
        (root / "_simulate.py").write_text("# changed simulator source\n")
        expected = "code/simulator/source_files/_simulate.py"
    else:
        git("commit", "--allow-empty", "-m", "New simulator revision")
        expected = "code/simulator/git_commit"
    with pytest.raises(SystemExit) as result:
        main([*args, "--resume"])
    assert result.value.code == 2
    error = capsys.readouterr().err
    assert expected in error
    assert "Start a new output folder" in error


def test_external_simulator_without_git_still_hashes_sources(tmp_path, monkeypatch):
    (tmp_path / "Meshes").mkdir()
    for name in MODEL_FILES:
        (tmp_path / name).write_text("# simulator source\n")
    monkeypatch.setattr("swim2real.simulators.fishcreate.shutil.which", lambda name: None)
    identity = FishCreateSimulator(tmp_path).identity()
    assert identity["git_commit"] is None
    assert (
        identity["source_files"]["_simulate.py"]
        == hashlib.sha256(b"# simulator source\n").hexdigest()
    )
