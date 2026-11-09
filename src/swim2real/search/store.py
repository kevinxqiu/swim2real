"""Atomic per-run records and resumable search checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Mapping


class RunStoreError(RuntimeError):
    """The stored cell state violates an append-only experiment invariant."""


def _atomic_write_json(path: Path, value: Any) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _strict_json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _changed_fields(stored: Any, current: Any, prefix: str) -> list[str]:
    if isinstance(stored, dict) and isinstance(current, dict):
        return [
            field
            for key in sorted(set(stored) | set(current))
            for field in _changed_fields(stored.get(key), current.get(key), f"{prefix}/{key}")
        ]
    return [prefix] if stored != current else []


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def record_redo(directory: Path, identity: Mapping[str, Any]) -> None:
    """Record another attempt at an interrupted operation without a saved outcome."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = sorted(directory.glob("redo_*.json"))
    for index, path in enumerate(paths, 1):
        expected = {"identity": dict(identity), "redo_index": index, "schema_version": 1}
        if path.name != f"redo_{index:04d}.json" or _read_json(path, label="redo") != expected:
            raise RunStoreError(f"Interrupted-operation record mismatch: {path}")
    _atomic_write_json(
        directory / f"redo_{len(paths) + 1:04d}.json",
        {"identity": dict(identity), "redo_index": len(paths) + 1, "schema_version": 1},
    )


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _read_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_json,
        )
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        raise RunStoreError(f"invalid or missing {label}: {path}") from exc


def _state_snapshot(
    records: list[dict[str, Any]],
    hashes: list[str],
    count: int,
) -> dict[str, Any]:
    if count == 0:
        return {
            "algorithm_state": None,
            "calls_completed": 0,
            "latest_state_sha256": None,
            "rng_state": None,
        }
    record = records[count - 1]
    return {
        "algorithm_state": record["algorithm_state"],
        "calls_completed": count,
        "latest_state_sha256": hashes[count - 1],
        "rng_state": record["rng_state"],
    }


def _load_state_records(directory: Path) -> tuple[list[dict[str, Any]], list[str]]:
    state_dir = directory / "state"
    if not state_dir.is_dir():
        raise RunStoreError("cell has no append-state directory")
    paths = sorted(state_dir.glob("state_*.json"))
    records: list[dict[str, Any]] = []
    hashes: list[str] = []
    previous_hash: str | None = None
    required_fields = {
        "algorithm_state",
        "call",
        "call_index",
        "call_sha256",
        "previous_state_sha256",
        "rng_state",
        "schema_version",
    }
    for expected_index, path in enumerate(paths, start=1):
        if path.name != f"state_{expected_index:04d}.json":
            raise RunStoreError("append-state records are not contiguous")
        loaded = _read_json(path, label="append-state record")
        if not isinstance(loaded, dict) or set(loaded) != required_fields:
            raise RunStoreError(f"append-state record fields are invalid: {path}")
        record = _strict_json_copy(loaded)
        if record["schema_version"] != 1 or isinstance(record["schema_version"], bool):
            raise RunStoreError(f"append-state schema version is invalid: {path}")
        if record["call_index"] != expected_index or isinstance(record["call_index"], bool):
            raise RunStoreError("append-state record index does not match its append position")
        call = record["call"]
        if not isinstance(call, dict) or call.get("call_index") != expected_index:
            raise RunStoreError("append-state call does not match its append position")
        if record["call_sha256"] != _canonical_sha256(call):
            raise RunStoreError(f"append-state call hash mismatch: {path}")
        if record["previous_state_sha256"] != previous_hash:
            raise RunStoreError(f"append-state hash chain mismatch: {path}")
        if record["rng_state"] is not None and not isinstance(record["rng_state"], dict):
            raise RunStoreError(f"append-state RNG state is invalid: {path}")
        if record["algorithm_state"] is not None and not isinstance(
            record["algorithm_state"], dict
        ):
            raise RunStoreError(f"append-state algorithm state is invalid: {path}")
        state_hash = _canonical_sha256(record)
        records.append(record)
        hashes.append(state_hash)
        previous_hash = state_hash
    return records, hashes


def _checkpoint_count(checkpoint: Mapping[str, Any]) -> int:
    count = checkpoint.get("calls_completed")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise RunStoreError("checkpoint call count is invalid")
    return count


def _load_calls_from_state(
    directory: Path,
    records: list[dict[str, Any]],
    *,
    checkpoint_count: int,
) -> list[dict[str, Any]]:
    call_dir = directory / "calls"
    if not call_dir.is_dir():
        raise RunStoreError("cell has no call-record directory")
    paths = sorted(call_dir.glob("call_*.json"))
    for expected_index, path in enumerate(paths, start=1):
        if path.name != f"call_{expected_index:04d}.json":
            raise RunStoreError("call records are not contiguous")
    if len(paths) > len(records):
        raise RunStoreError("call record has no matching append-state record")

    calls: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        path = call_dir / f"call_{index:04d}.json"
        if path.exists():
            call = _read_json(path, label="call record")
            if call != record["call"]:
                raise RunStoreError(f"call record disagrees with append-state record: {path}")
        else:
            # A write-ahead state exists before the call file.  Recovery is safe
            # only for the final record while the checkpoint still names the
            # preceding committed state.  Any other absence is corruption.
            if index != len(records) or checkpoint_count != index - 1:
                raise RunStoreError("call records are not contiguous")
            _atomic_write_json(path, record["call"])
            call = record["call"]
        calls.append(_strict_json_copy(call))

    if len(paths) < len(records) - 1:
        raise RunStoreError("call records are not contiguous")
    return calls


def _status_payload(
    manifest: Mapping[str, Any],
    calls_completed: int,
    *,
    complete: bool,
    artifact_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    payload = {
        "calls_completed": calls_completed,
        "complete": bool(complete),
        "manifest_sha256": _canonical_sha256(manifest),
        "method": manifest["method"],
        "problem_seed": int(manifest["problem_seed"]),
        "target_budget": int(manifest["budget"]),
    }
    if complete:
        if (
            not isinstance(artifact_manifest_sha256, str)
            or len(artifact_manifest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in artifact_manifest_sha256)
        ):
            raise RunStoreError("complete status requires an artifact manifest SHA-256")
        payload["artifact_manifest_sha256"] = artifact_manifest_sha256
    elif artifact_manifest_sha256 is not None:
        raise RunStoreError("incomplete status cannot bind an artifact manifest")
    return payload


def _artifact_manifest_payload(directory: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    excluded = {"artifact_manifest.json", "status.json"}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise RunStoreError(f"cell artifact cannot be a symbolic link: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        if relative in excluded:
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise RunStoreError(f"cell artifact is unreadable: {path}") from exc
        files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
        )
    if not any(row["path"] == "result.json" for row in files):
        raise RunStoreError("artifact manifest requires result.json")
    return {
        "files": files,
        "manifest_sha256": _canonical_sha256(manifest),
        "schema_version": 1,
    }


def _write_artifact_manifest(directory: Path, manifest: Mapping[str, Any]) -> str:
    manifest_path = directory / "artifact_manifest.json"
    expected = _artifact_manifest_payload(directory, manifest)
    if manifest_path.exists():
        stored = _read_json(manifest_path, label="artifact manifest")
        if stored != expected:
            raise RunStoreError("artifact manifest disagrees with the stored cell")
    else:
        _atomic_write_json(manifest_path, expected)
    return _canonical_sha256(expected)


def _validate_artifact_manifest(directory: Path, manifest: Mapping[str, Any]) -> str | None:
    manifest_path = directory / "artifact_manifest.json"
    if not manifest_path.exists():
        return None
    stored = _read_json(manifest_path, label="artifact manifest")
    expected = _artifact_manifest_payload(directory, manifest)
    if stored != expected:
        raise RunStoreError("artifact manifest disagrees with the stored cell")
    return _canonical_sha256(expected)


def validate_artifact_manifest(
    path: str | Path,
    manifest: Mapping[str, Any],
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Validate all cell bytes against the final file manifest."""
    directory = Path(path)
    observed = _validate_artifact_manifest(directory, manifest)
    if observed is None:
        raise RunStoreError("completed cell has no artifact manifest")
    if observed != expected_sha256:
        raise RunStoreError("artifact manifest SHA-256 differs from complete status")
    artifacts = _read_json(directory / "artifact_manifest.json", label="artifact manifest")
    if not isinstance(
        artifacts, dict
    ):  # pragma: no cover - internal validation already rejects this.
        raise RunStoreError("artifact manifest must be a JSON object")
    return artifacts


def _validate_status(
    status: Any,
    manifest: Mapping[str, Any],
    calls_completed: int,
    *,
    result_exists: bool,
    artifact_manifest_sha256: str | None,
) -> bool:
    """Return whether an incomplete status is stale by one recoverable call."""
    if not isinstance(status, dict):
        raise RunStoreError("status must be a JSON object")
    complete = status.get("complete")
    if not isinstance(complete, bool):
        raise RunStoreError("status complete flag is invalid")
    if complete:
        if (
            not result_exists
            or artifact_manifest_sha256 is None
            or status
            != _status_payload(
                manifest,
                calls_completed,
                complete=True,
                artifact_manifest_sha256=artifact_manifest_sha256,
            )
        ):
            raise RunStoreError("complete status does not match the stored cell")
        return False

    expected = _status_payload(manifest, calls_completed, complete=False)
    if status == expected:
        return False
    if not result_exists and calls_completed > 0:
        prior = _status_payload(manifest, calls_completed - 1, complete=False)
        if status == prior:
            return True
    raise RunStoreError("incomplete status does not match the stored cell")


def _validate_result_payload(
    result: Any,
    manifest: Mapping[str, Any],
    calls: list[dict[str, Any]],
) -> None:
    if not isinstance(result, dict) or set(result) != {"calls", "manifest", "summary"}:
        raise RunStoreError("result fields are invalid")
    if result["manifest"] != manifest:
        raise RunStoreError("result manifest does not match manifest.json")
    if result["calls"] != calls:
        raise RunStoreError("result calls do not match append-only call records")
    summary = result["summary"]
    required_summary_fields = {
        "best_call_index",
        "best_params",
        "best_score_mm",
        "structural_tail_carry",
    }
    if not isinstance(summary, dict) or set(summary) != required_summary_fields:
        raise RunStoreError("result summary fields are invalid")
    structural_tail = summary["structural_tail_carry"]
    if structural_tail not in (0, 1) or isinstance(structural_tail, bool):
        raise RunStoreError("structural_tail_carry must be exactly 0 or 1")
    if len(calls) + structural_tail != int(manifest["budget"]):
        raise RunStoreError("result call count does not match the manifest budget")

    best_call: Mapping[str, Any] | None = None
    best_score = math.inf
    for call in calls:
        if call.get("incumbent_eligible") is not True:
            continue
        score = call.get("score_mm")
        if (
            isinstance(score, bool)
            or not isinstance(score, Real)
            or not math.isfinite(float(score))
        ):
            raise RunStoreError("eligible call has an invalid score")
        if float(score) < best_score:
            best_call = call
            best_score = float(score)
    if best_call is None:
        raise RunStoreError("result has no incumbent-eligible call")
    expected_summary = {
        "best_call_index": best_call.get("call_index"),
        "best_params": best_call.get("params"),
        "best_score_mm": best_score,
        "structural_tail_carry": structural_tail,
    }
    if summary != expected_summary:
        raise RunStoreError("result summary does not match the best append-only call")


@dataclass
class RunStore:
    """Own one non-overwriting method/seed result directory."""

    path: Path
    manifest: dict[str, Any]
    calls: list[dict[str, Any]]
    rng_state: dict[str, Any] | None
    algorithm_state: dict[str, Any] | None

    @classmethod
    def create(cls, path: str | Path, manifest: Mapping[str, Any]) -> RunStore:
        directory = Path(path)
        if directory.exists():
            raise FileExistsError(f"Cell directory already exists: {directory}")
        canonical_manifest = _strict_json_copy(dict(manifest))
        _validate_manifest(canonical_manifest)
        directory.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}.init-", dir=directory.parent))
        try:
            (temporary / "calls").mkdir()
            (temporary / "state").mkdir()
            _atomic_write_json(temporary / "manifest.json", canonical_manifest)
            temporary_store = cls(temporary, canonical_manifest, [], None, None)
            temporary_store._write_checkpoint()
            temporary_store._write_status(complete=False)
            os.replace(temporary, directory)
            parent_fd = os.open(directory.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        store = cls(directory, canonical_manifest, [], None, None)
        return store

    @classmethod
    def resume(cls, path: str | Path, expected_manifest: Mapping[str, Any]) -> RunStore:
        directory = Path(path)
        canonical_expected = _strict_json_copy(dict(expected_manifest))
        _validate_manifest(canonical_expected)
        stored_manifest = _read_json(directory / "manifest.json", label="manifest")
        if not isinstance(stored_manifest, dict):
            raise RunStoreError("manifest must be a JSON object")
        if stored_manifest != canonical_expected:
            changes = _changed_fields(
                stored_manifest.get("code"), canonical_expected.get("code"), "code"
            )
            if changes:
                raise RunStoreError(
                    "Run implementation changed: "
                    + ", ".join(changes)
                    + ". Start a new output folder."
                )
            raise RunStoreError(
                "manifest mismatch; refusing to resume a different treatment. Start a new output folder."
            )

        checkpoint = _read_json(directory / "checkpoint.json", label="checkpoint")
        if not isinstance(checkpoint, dict):
            raise RunStoreError("checkpoint must be a JSON object")
        checkpoint_count = _checkpoint_count(checkpoint)
        records, hashes = _load_state_records(directory)
        state_count = len(records)
        if checkpoint_count == state_count:
            if checkpoint != _state_snapshot(records, hashes, state_count):
                raise RunStoreError("checkpoint state does not match the append-state record")
            recovered_checkpoint = False
        elif checkpoint_count == state_count - 1:
            if checkpoint != _state_snapshot(records, hashes, checkpoint_count):
                raise RunStoreError("checkpoint state does not match the prior append-state record")
            recovered_checkpoint = True
        else:
            raise RunStoreError("checkpoint call count does not match append-only call records")
        calls = _load_calls_from_state(
            directory,
            records,
            checkpoint_count=checkpoint_count,
        )
        final_count = len(calls)

        if records:
            rng_state = _strict_json_copy(records[-1]["rng_state"])
            algorithm_state = _strict_json_copy(records[-1]["algorithm_state"])
        else:
            rng_state = None
            algorithm_state = None
        store = cls(directory, stored_manifest, calls, rng_state, algorithm_state)
        if recovered_checkpoint:
            store._write_checkpoint()

        status = _read_json(directory / "status.json", label="status")
        result_path = directory / "result.json"
        result_exists = result_path.exists()
        if result_exists:
            result = _read_json(result_path, label="result")
            _validate_result_payload(result, stored_manifest, calls)
        manifest_sha256 = (
            _validate_artifact_manifest(directory, stored_manifest) if result_exists else None
        )
        stale_status = _validate_status(
            status,
            stored_manifest,
            final_count,
            result_exists=result_exists,
            artifact_manifest_sha256=manifest_sha256,
        )
        if result_exists:
            if manifest_sha256 is not None and status.get("complete") is not True:
                store._write_status(
                    complete=True,
                    artifact_manifest_sha256=manifest_sha256,
                )
        elif stale_status:
            store._write_status(complete=False)
        return store

    @property
    def next_call_index(self) -> int:
        return len(self.calls) + 1

    def append_call(
        self,
        call: Mapping[str, Any],
        rng_state: Mapping[str, Any] | None,
        algorithm_state: Mapping[str, Any] | None = None,
    ) -> None:
        if (self.path / "result.json").exists():
            raise RunStoreError("cannot append to a finalized cell")
        canonical_call = _strict_json_copy(dict(call))
        observed_index = canonical_call.get("call_index")
        if isinstance(observed_index, int) and observed_index < self.next_call_index:
            existing_path = self.path / "calls" / f"call_{observed_index:04d}.json"
            if existing_path.exists():
                raise FileExistsError(f"Call record already exists: {existing_path}")
        if observed_index != self.next_call_index:
            raise RunStoreError(
                f"expected call_index {self.next_call_index}, observed {observed_index!r}"
            )
        budget = int(self.manifest["budget"])
        if self.next_call_index > budget:
            raise RunStoreError(f"cell budget {budget} is already exhausted")

        state_records, state_hashes = _load_state_records(self.path)
        if len(state_records) != len(self.calls):
            raise RunStoreError("in-memory calls do not match append-state records")
        canonical_rng = _strict_json_copy(rng_state) if rng_state is not None else None
        canonical_algorithm = (
            _strict_json_copy(algorithm_state)
            if algorithm_state is not None
            else _strict_json_copy(self.algorithm_state)
        )
        index = self.next_call_index
        state_record = {
            "algorithm_state": canonical_algorithm,
            "call": canonical_call,
            "call_index": index,
            "call_sha256": _canonical_sha256(canonical_call),
            "previous_state_sha256": state_hashes[-1] if state_hashes else None,
            "rng_state": canonical_rng,
            "schema_version": 1,
        }
        state_path = self.path / "state" / f"state_{index:04d}.json"
        call_path = self.path / "calls" / f"call_{index:04d}.json"
        if state_path.exists():
            raise FileExistsError(f"Append-state record already exists: {state_path}")
        if call_path.exists():
            raise FileExistsError(f"Call record already exists: {call_path}")

        # The state record is a write-ahead log.  It contains the exact call and
        # post-call RNG/algorithm states needed to complete either interrupted
        # atomic write without re-evaluating or drawing randomness again.
        _atomic_write_json(state_path, state_record)
        _atomic_write_json(call_path, canonical_call)
        self.calls.append(canonical_call)
        self.rng_state = canonical_rng
        self.algorithm_state = canonical_algorithm
        self._write_checkpoint()
        self._write_status(complete=False)

    def finalize(self, summary: Mapping[str, Any]) -> None:
        canonical_summary = _strict_json_copy(dict(summary))
        structural_tail = canonical_summary.get("structural_tail_carry", 0)
        if structural_tail not in (0, 1) or isinstance(structural_tail, bool):
            raise RunStoreError("structural_tail_carry must be exactly 0 or 1")
        budget = int(self.manifest["budget"])
        if len(self.calls) + structural_tail != budget:
            raise RunStoreError(f"cannot finalize until exactly {budget} calls are stored")
        payload = {
            "manifest": self.manifest,
            "calls": self.calls,
            "summary": canonical_summary,
        }
        _validate_result_payload(payload, self.manifest, self.calls)
        result_path = self.path / "result.json"
        if result_path.exists():
            existing = _read_json(result_path, label="result")
            if existing != payload:
                raise RunStoreError("existing result does not match the requested finalization")
        else:
            _atomic_write_json(result_path, payload)
        artifact_manifest_sha256 = _write_artifact_manifest(self.path, self.manifest)
        self._write_status(
            complete=True,
            artifact_manifest_sha256=artifact_manifest_sha256,
        )

    def checkpoint_algorithm_state(self, algorithm_state: Mapping[str, Any]) -> None:
        """Persist a verified post-decision state without rewriting a call record."""
        canonical = _strict_json_copy(dict(algorithm_state))
        if self.calls:
            records, _ = _load_state_records(self.path)
            if len(records) != len(self.calls) or canonical != records[-1]["algorithm_state"]:
                raise RunStoreError(
                    "algorithm checkpoint disagrees with the final append-state record"
                )
        self.algorithm_state = canonical
        self._write_checkpoint()

    def _write_checkpoint(self) -> None:
        records, hashes = _load_state_records(self.path)
        if len(records) != len(self.calls):
            raise RunStoreError("cannot checkpoint calls without matching append-state records")
        snapshot = _state_snapshot(records, hashes, len(records))
        if (
            snapshot["rng_state"] != self.rng_state
            or snapshot["algorithm_state"] != self.algorithm_state
        ):
            raise RunStoreError("in-memory state does not match append-state records")
        _atomic_write_json(self.path / "checkpoint.json", snapshot)

    def _write_status(
        self,
        *,
        complete: bool,
        artifact_manifest_sha256: str | None = None,
    ) -> None:
        _atomic_write_json(
            self.path / "status.json",
            _status_payload(
                self.manifest,
                len(self.calls),
                complete=complete,
                artifact_manifest_sha256=artifact_manifest_sha256,
            ),
        )


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    required = {"budget", "method", "problem_seed", "schema_version"}
    missing = sorted(required - set(manifest))
    if missing:
        raise RunStoreError(f"manifest is missing required fields: {missing}")
    budget = manifest["budget"]
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise RunStoreError("manifest budget must be a positive integer")


__all__ = ["RunStore", "RunStoreError", "validate_artifact_manifest"]

atomic_write_json = _atomic_write_json
canonical_sha256 = _canonical_sha256
