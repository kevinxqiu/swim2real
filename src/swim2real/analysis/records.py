"""Read and validate compact records without simulator or model dependencies."""

from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .studies import BUDGET, PARAMETER_KEYS, STUDIES


@dataclass
class RecordBundle:
    """One release, with reused runs stored only once."""

    runs: dict[tuple[str, str, int], dict[str, Any]]
    manifests: dict[str, dict[str, Any]]
    assessments: list[dict[str, Any]]
    replays: list[dict[str, Any]]


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _frequency_errors(rows, mean, valid):
    if not rows or len({r["frequency_hz"] for r in rows}) != len(rows):
        raise ValueError("frequency records must be nonempty and unique")
    for row in rows:
        if (
            type(row["valid"]) is not bool
            or not _finite(row["score_mm"])
            or not _finite(row["frequency_hz"])
        ):
            raise ValueError("invalid frequency record")
        if "per_marker_mm" in row:
            errors = row["per_marker_mm"]
            if len(errors) != 9 or not all(_finite(v) for v in errors):
                raise ValueError("expected nine finite marker errors")
            if not math.isclose(math.fsum(errors) / 9, row["score_mm"], abs_tol=1e-8):
                raise ValueError("per-marker errors disagree with frequency mean")
    if not _finite(mean) or not math.isclose(
        math.fsum(r["score_mm"] for r in rows) / len(rows), mean, abs_tol=1e-8
    ):
        raise ValueError("frequency errors disagree with evaluation mean")
    if type(valid) is not bool or valid != all(r["valid"] for r in rows):
        raise ValueError("frequency and evaluation validity disagree")


def validate_run(run: Mapping[str, Any]) -> None:
    """Reject missing calls, inconsistent curves and malformed parameter vectors."""
    study, method, seed = run["study"], run["method"], run["seed"]
    if run["schema_version"] != 1 or study not in STUDIES:
        raise ValueError("unsupported run schema or study")
    definition = STUDIES[study]
    if method not in definition.methods or seed not in definition.seeds or type(seed) is not int:
        raise ValueError("unexpected method or seed")
    if run["budget"] != BUDGET:
        raise ValueError("unexpected evaluation budget")
    calls = run["evaluations"]
    if [c["call_index"] for c in calls] != list(range(1, len(calls) + 1)):
        raise ValueError("evaluation indices must be contiguous and chronological")
    tail = run["structural_tail_carry"]
    if type(tail) is not int or tail not in (0, 1):
        raise ValueError("invalid structural tail")
    complete = run["completion"]["complete"]
    if type(complete) is not bool or len(calls) > BUDGET:
        raise ValueError("invalid completion or call count")
    if complete and (len(calls) + tail != BUDGET or not calls[0]["valid"]):
        raise ValueError("completed run does not satisfy the call budget")
    if tail and method != "spsa_ls":
        raise ValueError("only SPSA may carry a structural unused tail")
    best = math.inf
    for call in calls:
        for field in ("parameters", "normalized_parameters"):
            if len(call[field]) != len(PARAMETER_KEYS) or not all(_finite(v) for v in call[field]):
                raise ValueError("expected sixteen finite parameters")
        _frequency_errors(call["frequency_errors"], call["mean_error_mm"], call["valid"])
        accepted = call["valid"] and call["mean_error_mm"] < best
        if call["accepted"] is not accepted:
            raise ValueError("acceptance disagrees with strict valid improvement")
        if accepted:
            best = call["mean_error_mm"]
        expected = best if math.isfinite(best) else None
        if call["best_so_far_mm"] != expected:
            raise ValueError("stored best-so-far differs from recomputed curve")
    rounds = [p["round_index"] for p in run["model_proposals"]]
    if rounds != sorted(set(rounds)):
        raise ValueError("model proposal rounds must be unique and ordered")
    for proposal in run["model_proposals"]:
        if any(type(n) is not int or n < 0 for n in proposal["token_counts"].values()):
            raise ValueError("token counts must be nonnegative integers")


def load_records(path: str | Path, *, allow_partial: bool = False) -> RecordBundle:
    """Load a directory or runs.zip, checking the scheduled matrix by default."""
    path = Path(path)
    if path.is_dir() and not (path / "schema.json").exists():
        if (path / "records/schema.json").exists():
            path = path / "records"
        elif (path / "runs.zip").exists():
            path = path / "runs.zip"
    if path.is_dir():
        payloads = {
            str(p.relative_to(path)): json.loads(p.read_text()) for p in path.rglob("*.json")
        }
    else:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("duplicate archive member")
            payloads = {
                name: json.loads(archive.read(name)) for name in names if name.endswith(".json")
            }
    schema = payloads.get("schema.json", {})
    if schema.get("schema_version") != 1 or schema.get("parameter_keys") != list(PARAMETER_KEYS):
        raise ValueError("unsupported record schema or parameter order")
    runs = {}
    manifests = {}
    for name, value in payloads.items():
        if name.startswith("runs/"):
            validate_run(value)
            key = (value["study"], value["method"], value["seed"])
            if key in runs:
                raise ValueError("duplicate run identity")
            runs[key] = value
        elif name.startswith("studies/"):
            study = value["study"]
            if study not in STUDIES or study in manifests:
                raise ValueError("unexpected or duplicate study manifest")
            definition = STUDIES[study]
            if value["scheduled_seeds"] != list(definition.seeds) or value["methods"] != list(
                definition.methods
            ):
                raise ValueError("manifest disagrees with study definition")
            if value["shared_failure_seed"] != definition.shared_failure_seed:
                raise ValueError("manifest changes the shared-initialization exception")
            manifests[study] = value
    assessments = payloads.get("assessments/interpolation.json", [])
    seen = set()
    for row in assessments:
        key = (row["method"], row["seed"])
        if key in seen:
            raise ValueError("duplicate interpolation assessment")
        seen.add(key)
        _frequency_errors(row["frequency_errors"], row["mean_error_mm"], row["valid"])
    bundle = RecordBundle(
        runs, manifests, assessments, payloads.get("replays/backtracking.json", [])
    )
    if not allow_partial:
        if set(manifests) != set(STUDIES):
            raise ValueError("release must contain all four study manifests")
        for study, definition in STUDIES.items():
            if study == "interpolation":
                expected = {(m, s) for m in definition.methods for s in definition.seeds}
                if seen != expected:
                    raise ValueError("interpolation assessment matrix is incomplete")
                continue
            for method in definition.methods:
                source = (
                    "replication" if study == "glm" and method == "multiscale_random_ls" else study
                )
                rows = [
                    runs[(source, method, s)]
                    for s in definition.seeds
                    if (source, method, s) in runs
                ]
                if len(rows) != len(definition.seeds):
                    raise ValueError(f"incomplete scheduled matrix: {study}/{method}")
                if (
                    sum(r["completion"]["complete"] for r in rows)
                    != manifests[study]["completion_counts"][method]
                ):
                    raise ValueError("completion count disagrees with manifest")
    return bundle
