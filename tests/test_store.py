import json
import re
from pathlib import Path

import numpy as np
import pytest

from swim2real.search.store import RunStore, RunStoreError


def _manifest() -> dict:
    return {
        "budget": 40,
        "method": "multiscale_random_ls",
        "problem_seed": 1200,
        "schema_version": 1,
    }


def _call(index: int) -> dict:
    return {
        "call_index": index,
        "incumbent_eligible": True,
        "params": {"x": index / 100},
        "score_mm": 100.0 - index,
        "valid": True,
    }


def _summary(best_index: int, *, structural_tail_carry: int = 0) -> dict:
    best = _call(best_index)
    return {
        "best_call_index": best_index,
        "best_params": best["params"],
        "best_score_mm": best["score_mm"],
        "structural_tail_carry": structural_tail_carry,
    }


def test_new_store_writes_manifest_and_score_free_status(tmp_path: Path) -> None:
    path = tmp_path / "cell"

    store = RunStore.create(path, _manifest())
    store.append_call(
        _call(1),
        np.random.default_rng(4).bit_generator.state,
        algorithm_state={"incumbent": [0.5] * 16, "incumbent_loss": 99.0},
    )

    assert json.loads((path / "manifest.json").read_text()) == _manifest()
    status = json.loads((path / "status.json").read_text())
    assert status == {
        "calls_completed": 1,
        "complete": False,
        "manifest_sha256": status["manifest_sha256"],
        "method": "multiscale_random_ls",
        "problem_seed": 1200,
        "target_budget": 40,
    }
    assert re.fullmatch(r"[0-9a-f]{64}", status["manifest_sha256"])
    assert "score" not in json.dumps(status).lower()


def test_store_refuses_existing_nonempty_directory(tmp_path: Path) -> None:
    path = tmp_path / "cell"
    path.mkdir()
    (path / "owned.txt").write_text("existing")

    with pytest.raises(FileExistsError):
        RunStore.create(path, _manifest())


def test_store_initialization_is_atomic_and_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from swim2real.search import store as store_module

    path = tmp_path / "cell"
    original = store_module._atomic_write_json

    def crash_after_manifest(target: Path, value) -> None:
        original(target, value)
        if target.name == "checkpoint.json":
            raise OSError("injected initialization crash")

    monkeypatch.setattr(store_module, "_atomic_write_json", crash_after_manifest)
    with pytest.raises(OSError, match="initialization crash"):
        RunStore.create(path, _manifest())
    assert not path.exists()

    monkeypatch.setattr(store_module, "_atomic_write_json", original)
    assert RunStore.create(path, _manifest()).calls == []


def test_calls_are_append_only_and_contiguous(tmp_path: Path) -> None:
    store = RunStore.create(tmp_path / "cell", _manifest())
    rng = np.random.default_rng(4)
    store.append_call(_call(1), rng.bit_generator.state)

    with pytest.raises(RunStoreError, match="expected call_index 2"):
        store.append_call(_call(3), rng.bit_generator.state)
    with pytest.raises(FileExistsError):
        store.append_call(_call(1), rng.bit_generator.state)


def test_resume_validates_manifest_and_restores_state(tmp_path: Path) -> None:
    path = tmp_path / "cell"
    rng = np.random.default_rng(99)
    store = RunStore.create(path, _manifest())
    store.append_call(
        _call(1),
        rng.bit_generator.state,
        algorithm_state={"incumbent": [0.5] * 16, "incumbent_loss": 99.0},
    )
    expected_next = rng.normal(size=4)

    resumed = RunStore.resume(path, _manifest())
    restored_rng = np.random.default_rng()
    restored_rng.bit_generator.state = resumed.rng_state

    assert resumed.next_call_index == 2
    assert resumed.algorithm_state["incumbent_loss"] == 99.0
    np.testing.assert_allclose(restored_rng.normal(size=4), expected_next)
    with pytest.raises(RunStoreError, match="manifest mismatch"):
        RunStore.resume(path, {**_manifest(), "budget": 39})


def test_resume_recovers_call_written_before_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "cell"
    rng = np.random.default_rng(123)
    store = RunStore.create(path, _manifest())
    rng.normal(size=5)
    saved_state = json.loads(json.dumps(rng.bit_generator.state))

    def crash_before_checkpoint() -> None:
        raise OSError("injected crash before checkpoint")

    monkeypatch.setattr(store, "_write_checkpoint", crash_before_checkpoint)
    with pytest.raises(OSError, match="injected crash"):
        store.append_call(
            _call(1),
            saved_state,
            algorithm_state={"incumbent": [0.5] * 16, "incumbent_loss": 99.0},
        )

    stale_checkpoint = json.loads((path / "checkpoint.json").read_text())
    assert stale_checkpoint["calls_completed"] == 0
    assert (path / "calls" / "call_0001.json").is_file()
    assert (path / "state" / "state_0001.json").is_file()

    resumed = RunStore.resume(path, _manifest())

    assert resumed.calls == [_call(1)]
    assert resumed.rng_state == saved_state
    assert resumed.algorithm_state == {"incumbent": [0.5] * 16, "incumbent_loss": 99.0}
    recovered_checkpoint = json.loads((path / "checkpoint.json").read_text())
    assert recovered_checkpoint["calls_completed"] == 1
    assert re.fullmatch(r"[0-9a-f]{64}", recovered_checkpoint["latest_state_sha256"])


def test_resume_rejects_call_that_disagrees_with_append_state(tmp_path: Path) -> None:
    path = tmp_path / "cell"
    rng = np.random.default_rng(2)
    store = RunStore.create(path, _manifest())
    store.append_call(_call(1), rng.bit_generator.state)
    tampered = _call(1)
    tampered["score_mm"] = -1.0
    (path / "calls" / "call_0001.json").write_text(json.dumps(tampered))

    with pytest.raises(RunStoreError, match="append-state record"):
        RunStore.resume(path, _manifest())


def test_resume_rejects_checkpoint_state_tamper(tmp_path: Path) -> None:
    path = tmp_path / "cell"
    rng = np.random.default_rng(2)
    store = RunStore.create(path, _manifest())
    store.append_call(_call(1), rng.bit_generator.state)
    checkpoint = json.loads((path / "checkpoint.json").read_text())
    checkpoint["rng_state"]["state"]["state"] += 1
    (path / "checkpoint.json").write_text(json.dumps(checkpoint))

    with pytest.raises(RunStoreError, match="checkpoint state does not match"):
        RunStore.resume(path, _manifest())


def test_finalize_requires_exact_budget_and_is_immutable(tmp_path: Path) -> None:
    manifest = {**_manifest(), "budget": 2}
    store = RunStore.create(tmp_path / "cell", manifest)
    rng = np.random.default_rng(1)
    store.append_call(_call(1), rng.bit_generator.state)
    with pytest.raises(RunStoreError, match="2 calls"):
        store.finalize(_summary(1))
    store.append_call(_call(2), rng.bit_generator.state)

    store.finalize(_summary(2))

    status = json.loads((tmp_path / "cell" / "status.json").read_text())
    assert status["complete"] is True
    assert re.fullmatch(r"[0-9a-f]{64}", status["artifact_manifest_sha256"])
    assert (tmp_path / "cell" / "result.json").exists()
    manifest = json.loads((tmp_path / "cell" / "artifact_manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert {row["path"] for row in manifest["files"]} == {
        "calls/call_0001.json",
        "calls/call_0002.json",
        "checkpoint.json",
        "manifest.json",
        "result.json",
        "state/state_0001.json",
        "state/state_0002.json",
    }
    store.finalize(_summary(2))
    with pytest.raises(RunStoreError, match="does not match"):
        store.finalize(_summary(1))


def test_finalize_allows_only_one_declared_structural_tail(tmp_path: Path) -> None:
    manifest = {**_manifest(), "budget": 2}
    store = RunStore.create(tmp_path / "cell", manifest)
    rng = np.random.default_rng(1)
    store.append_call(_call(1), rng.bit_generator.state)

    store.finalize(_summary(1, structural_tail_carry=1))

    assert json.loads((tmp_path / "cell" / "status.json").read_text())["complete"] is True


def test_resume_repairs_result_written_before_complete_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "cell"
    manifest = {**_manifest(), "budget": 1}
    rng = np.random.default_rng(8)
    store = RunStore.create(path, manifest)
    store.append_call(_call(1), rng.bit_generator.state)
    summary = _summary(1)

    def crash_before_complete_status(*, complete: bool, artifact_manifest_sha256=None) -> None:
        assert complete is True
        assert artifact_manifest_sha256 is not None
        raise OSError("injected crash before complete status")

    monkeypatch.setattr(store, "_write_status", crash_before_complete_status)
    with pytest.raises(OSError, match="injected crash"):
        store.finalize(summary)

    assert (path / "result.json").is_file()
    assert (path / "artifact_manifest.json").is_file()
    assert json.loads((path / "status.json").read_text())["complete"] is False

    resumed = RunStore.resume(path, manifest)

    assert json.loads((path / "status.json").read_text())["complete"] is True
    resumed.finalize(summary)
    assert json.loads((path / "result.json").read_text())["summary"] == summary


def test_artifact_integrity_detects_non_json_artifact_tamper(tmp_path: Path) -> None:
    path = tmp_path / "cell"
    manifest = {**_manifest(), "budget": 1}
    store = RunStore.create(path, manifest)
    (path / "objective_cache").mkdir()
    evidence = path / "objective_cache" / "call_0001.npz"
    evidence.write_bytes(b"authoritative-binary-evidence")
    store.append_call(_call(1), np.random.default_rng(8).bit_generator.state)
    store.finalize(_summary(1))
    evidence.write_bytes(b"tampered-binary-evidence")

    with pytest.raises(RunStoreError, match="artifact manifest"):
        RunStore.resume(path, manifest)


def test_result_without_manifest_remains_incomplete_until_validated_finalize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from swim2real.search import store as store_module

    path = tmp_path / "cell"
    manifest = {**_manifest(), "budget": 1}
    store = RunStore.create(path, manifest)
    store.append_call(_call(1), np.random.default_rng(8).bit_generator.state)
    real_manifest = store_module._write_artifact_manifest

    def crash_before_manifest(*args, **kwargs):
        del args, kwargs
        raise OSError("injected manifest crash")

    monkeypatch.setattr(store_module, "_write_artifact_manifest", crash_before_manifest)
    with pytest.raises(OSError, match="manifest crash"):
        store.finalize(_summary(1))
    assert (path / "result.json").is_file()
    assert not (path / "artifact_manifest.json").exists()
    assert json.loads((path / "status.json").read_text())["complete"] is False

    resumed = RunStore.resume(path, manifest)
    assert json.loads((path / "status.json").read_text())["complete"] is False
    monkeypatch.setattr(store_module, "_write_artifact_manifest", real_manifest)
    resumed.finalize(_summary(1))
    assert json.loads((path / "status.json").read_text())["complete"] is True


def test_resume_rejects_tampered_result_instead_of_repairing_status(tmp_path: Path) -> None:
    path = tmp_path / "cell"
    manifest = {**_manifest(), "budget": 1}
    rng = np.random.default_rng(8)
    store = RunStore.create(path, manifest)
    store.append_call(_call(1), rng.bit_generator.state)
    store.finalize(_summary(1))
    result = json.loads((path / "result.json").read_text())
    result["calls"][0]["score_mm"] = -999.0
    (path / "result.json").write_text(json.dumps(result))

    with pytest.raises(RunStoreError, match="result calls do not match"):
        RunStore.resume(path, manifest)


def test_resume_rejects_result_summary_that_disagrees_with_calls(tmp_path: Path) -> None:
    path = tmp_path / "cell"
    manifest = {**_manifest(), "budget": 1}
    rng = np.random.default_rng(8)
    store = RunStore.create(path, manifest)
    store.append_call(_call(1), rng.bit_generator.state)
    store.finalize(_summary(1))
    result = json.loads((path / "result.json").read_text())
    result["summary"]["best_score_mm"] = -999.0
    (path / "result.json").write_text(json.dumps(result))

    with pytest.raises(RunStoreError, match="result summary"):
        RunStore.resume(path, manifest)


def test_load_calls_rejects_tampered_noncontiguous_state(tmp_path: Path) -> None:
    path = tmp_path / "cell"
    store = RunStore.create(path, _manifest())
    rng = np.random.default_rng(2)
    store.append_call(_call(1), rng.bit_generator.state)
    (path / "calls" / "call_0001.json").unlink()

    with pytest.raises(RunStoreError, match="contiguous"):
        RunStore.resume(path, _manifest())
