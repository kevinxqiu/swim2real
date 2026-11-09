"""Complete, budget-counted frequency sweeps with a per-frequency failure penalty."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from swim2real.config import PAPER_DEFAULTS, read_json
from swim2real.search.store import atomic_write_json, canonical_sha256, record_redo

FREQUENCIES_HZ = tuple(PAPER_DEFAULTS["frequencies_hz"])
FAILURE_PENALTY_MM = PAPER_DEFAULTS["failure_penalty_mm_per_frequency"]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrequencyRecord:
    frequency_hz: float
    score_mm: float
    valid: bool
    error_type: str | None
    error_message: str | None
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON record."""
        return asdict(self)


class CandidateEvaluationFailure(RuntimeError):
    """A simulation or metric failure specific to this candidate."""

    def __init__(self, message: str, *, frequency_record: FrequencyRecord | None = None) -> None:
        super().__init__(message)
        self.frequency_record = frequency_record


@dataclass(frozen=True)
class ObjectiveCall:
    call_index: int
    params: dict[str, Any]
    frequency_records: tuple[FrequencyRecord, ...]
    score_mm: float
    valid: bool
    incumbent_eligible: bool
    success_count: int
    failure_count: int
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return the call and all its ordered frequency records."""
        return {**asdict(self), "frequency_records": [r.to_dict() for r in self.frequency_records]}


class ObjectiveEvaluator:
    """Count one objective call for each complete configured frequency sweep."""

    def __init__(
        self,
        evaluate_frequency: Callable[[dict[str, Any], float], object],
        *,
        frequencies_hz: Sequence[float] = FREQUENCIES_HZ,
        failure_penalty_mm: float = FAILURE_PENALTY_MM,
        budget: int | None = None,
        clock: Callable[[], float] = time.perf_counter,
        start_call_index: int = 0,
        journal_dir: str | Path | None = None,
        on_call_start: Callable[[int, dict[str, Any]], None] | None = None,
        recover_frequency: Callable[[int, dict[str, Any], float], FrequencyRecord | None]
        | None = None,
    ) -> None:
        if not callable(evaluate_frequency):
            raise TypeError("evaluate_frequency must be callable")
        if (
            isinstance(start_call_index, bool)
            or not isinstance(start_call_index, int)
            or start_call_index < 0
        ):
            raise ValueError("start_call_index must be a nonnegative integer")
        if budget is not None and (
            isinstance(budget, bool)
            or not isinstance(budget, int)
            or budget < 1
            or start_call_index > budget
        ):
            raise ValueError("budget must be positive and cover the starting call index")
        self.frequencies_hz = tuple(frequencies_hz)
        if (
            not self.frequencies_hz
            or any(
                isinstance(f, bool) or not isinstance(f, Real) or not math.isfinite(f) or f <= 0
                for f in self.frequencies_hz
            )
            or len(set(self.frequencies_hz)) != len(self.frequencies_hz)
        ):
            raise ValueError("frequencies must be unique finite positive numbers")
        if (
            isinstance(failure_penalty_mm, bool)
            or not isinstance(failure_penalty_mm, Real)
            or not math.isfinite(failure_penalty_mm)
            or failure_penalty_mm <= 0
        ):
            raise ValueError("failure penalty must be finite and positive")
        self.failure_penalty_mm = float(failure_penalty_mm)
        self.budget = budget
        self._evaluate_frequency = evaluate_frequency
        self._clock = clock
        self._call_count = start_call_index
        self._journal_dir = Path(journal_dir) if journal_dir else None
        self._on_call_start = on_call_start
        self._recover_frequency = recover_frequency

    @property
    def call_count(self) -> int:
        """Return the number of completed objective evaluations."""
        return self._call_count

    def _record(self, value: Any, frequency: float) -> FrequencyRecord:
        if not isinstance(value, dict) or set(value) != set(FrequencyRecord.__dataclass_fields__):
            raise ValueError("frequency record has invalid fields")
        record = FrequencyRecord(**value)
        if record.frequency_hz != frequency or not isinstance(record.valid, bool):
            raise ValueError("frequency record has an invalid frequency or validity flag")
        for v in [record.score_mm, record.duration_seconds]:
            if isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v < 0:
                raise ValueError("frequency scores and durations must be finite and nonnegative")
        if record.valid:
            if record.error_type is not None or record.error_message is not None:
                raise ValueError("valid frequency record contains an error")
        elif (
            record.score_mm != self.failure_penalty_mm
            or not isinstance(record.error_type, str)
            or not record.error_type
            or not isinstance(record.error_message, str)
            or not record.error_message
        ):
            raise ValueError("failed frequency record must contain the penalty and an error")
        return record

    def _evaluate_one(self, params: dict[str, Any], frequency: float) -> FrequencyRecord:
        started = self._clock()
        error: Exception | None = None
        try:
            value = self._evaluate_frequency(params, frequency)
        except CandidateEvaluationFailure as exc:
            if exc.frequency_record is not None:
                return self._record(exc.frequency_record.to_dict(), frequency)
            error = exc
            value = None
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value < 0
        ):
            error_type = type(error).__name__ if error else "InvalidResult"
            message = (
                str(error)
                if error
                else f"evaluate_frequency returned {value!r}; expected a finite nonnegative number"
            )
            logger.warning("Candidate failed at %.2f Hz: %s", frequency, message)
            return FrequencyRecord(
                frequency,
                self.failure_penalty_mm,
                False,
                error_type,
                message,
                max(0.0, float(self._clock() - started)),
            )
        return FrequencyRecord(
            frequency, float(value), True, None, None, max(0.0, float(self._clock() - started))
        )

    def evaluate(self, params: Mapping[str, Any]) -> ObjectiveCall:
        """Evaluate every frequency in order, including after candidate failures."""
        if not isinstance(params, Mapping) or not all(isinstance(k, str) for k in params):
            raise ValueError("params must be a JSON object with string keys")
        # Canonical JSON rejects non-finite values before the budget is consumed.
        canonical_sha256(dict(params))
        params = dict(sorted(params.items()))
        if self.budget is not None and self.call_count >= self.budget:
            raise RuntimeError(f"Objective budget of {self.budget} is exhausted")
        result = self._evaluate_call(params, self.call_count + 1)
        self._call_count += 1
        return result

    def _evaluate_call(self, params: dict[str, Any], call_index: int) -> ObjectiveCall:
        directory = self._journal_dir / f"call_{call_index:04d}" if self._journal_dir else None
        intent = {
            "call_index": call_index,
            "frequencies_hz": list(self.frequencies_hz),
            "params": params,
            "failure_penalty_mm": self.failure_penalty_mm,
            "schema_version": 1,
        }
        if directory:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "intent.json"
            if path.exists():
                if read_json(path) != intent:
                    raise ValueError(f"Objective journal candidate or settings mismatch: {path}")
            else:
                atomic_write_json(path, intent)
            if (directory / "result.json").exists():
                return self._load_result(directory, params, call_index)
        if self._on_call_start:
            self._on_call_start(call_index, params)
        started = self._clock()
        rows = []
        for index, frequency in enumerate(self.frequencies_hz, 1):
            sidecar = directory / f"frequency_{index:02d}.json" if directory else None
            pending = directory / f"frequency_{index:02d}.intent.json" if directory else None
            identity = {
                "call_index": call_index,
                "frequency_hz": frequency,
                "params": params,
                "position": index,
                "schema_version": 1,
            }
            if sidecar and sidecar.exists():
                row = self._record(read_json(sidecar), frequency)
                if pending and pending.exists():
                    if read_json(pending) != identity:
                        raise ValueError(f"Frequency journal mismatch: {pending}")
                    pending.unlink()
            else:
                if pending and pending.exists():
                    if read_json(pending) != identity:
                        raise ValueError(f"Frequency journal mismatch: {pending}")
                    recovered = (
                        self._recover_frequency(call_index, params, frequency)
                        if self._recover_frequency
                        else None
                    )
                    if recovered is None:
                        record_redo(directory / "retries" / f"frequency_{index:02d}", identity)
                        logger.warning(
                            "Redoing interrupted objective %d at %.2f Hz", call_index, frequency
                        )
                        row = self._evaluate_one(params, frequency)
                    else:
                        row = self._record(recovered.to_dict(), frequency)
                else:
                    if pending:
                        atomic_write_json(pending, identity)
                    row = self._evaluate_one(params, frequency)
                if sidecar and pending:
                    atomic_write_json(sidecar, row.to_dict())
                    pending.unlink()
            rows.append(row)
        failures = sum(not r.valid for r in rows)
        duration = (
            sum(r.duration_seconds for r in rows)
            if directory
            else max(0.0, float(self._clock() - started))
        )
        result = ObjectiveCall(
            call_index,
            params,
            tuple(rows),
            float(sum(r.score_mm for r in rows) / len(rows)),
            failures == 0,
            failures == 0,
            len(rows) - failures,
            failures,
            duration,
        )
        if directory:
            atomic_write_json(directory / "result.json", result.to_dict())
        return result

    def _load_result(self, directory: Path, params: dict[str, Any], index: int) -> ObjectiveCall:
        value = read_json(directory / "result.json")
        if value["params"] != params or value["call_index"] != index:
            raise ValueError(f"Objective result identity mismatch: {directory}")
        duration = value.get("duration_seconds")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, Real)
            or not math.isfinite(duration)
            or duration < 0
        ):
            raise ValueError(f"Objective duration must be finite and nonnegative: {directory}")
        rows = tuple(
            self._record(read_json(directory / f"frequency_{i:02d}.json"), f)
            for i, f in enumerate(self.frequencies_hz, 1)
        )
        failures = sum(not r.valid for r in rows)
        result = ObjectiveCall(
            index,
            params,
            rows,
            float(sum(r.score_mm for r in rows) / len(rows)),
            failures == 0,
            failures == 0,
            len(rows) - failures,
            failures,
            value["duration_seconds"],
        )
        if result.to_dict() != value:
            raise ValueError(f"Objective result differs from its frequency records: {directory}")
        return result

    def validate_stored_calls(
        self, calls: list[Mapping[str, Any]], *, allow_pending: bool = True
    ) -> None:
        """Check that stored calls agree with their per-frequency journal."""
        if self._journal_dir is None:
            return
        expected = {f"call_{i:04d}" for i in range(1, len(calls) + 1)}
        actual = {p.name for p in self._journal_dir.glob("call_*") if p.is_dir()}
        extra = {f"call_{len(calls) + 1:04d}"} if allow_pending else set()
        if not expected <= actual or not actual <= expected | extra:
            raise ValueError("Objective journal has missing or extra calls")
        for index, call in enumerate(calls, 1):
            result = self._load_result(
                self._journal_dir / f"call_{index:04d}", dict(call["params"]), index
            ).to_dict()
            if {k: call.get(k) for k in result} != result:
                raise ValueError(f"Objective journal disagrees with stored call {index}")


class SimulationObjective:
    """Evaluate a simulator against marker data and cache prompt diagnostics."""

    def __init__(self, simulator: Any, dataset: Any, config: Any, cache_dir: Path) -> None:
        self.simulator = simulator
        self.dataset = dataset
        self.config = config
        self.cache_dir = cache_dir
        self.call_index = 0

    def begin_objective(self, call_index: int, flat: dict[str, Any]) -> None:
        """Bind subsequent frequency diagnostics to their objective call."""
        self.call_index = call_index

    def _directory(self, call_index: int, frequency: float) -> Path:
        return self.cache_dir / f"call_{call_index:04d}" / f"frequency_{frequency:g}"

    def evaluate_frequency(self, flat: dict[str, Any], frequency_hz: float) -> float:
        """Simulate the realized drive condition and compute marker error."""
        import numpy as np

        from swim2real.metrics import compute_error
        from swim2real.providers.base import _flat_digest, _validated_detail
        from swim2real.timebase import assert_requested_frequency_realized

        if self.call_index < 1:
            raise RuntimeError("SimulationObjective requires begin_objective before evaluation")
        started = time.perf_counter()
        directory = self._directory(self.call_index, frequency_hz)
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "outcome.json").exists():
            raise FileExistsError(f"Frequency outcome already exists: {directory}")
        realized = self.config.realized_frequencies_hz[frequency_hz]
        try:
            trace = self.simulator.run(
                flat, realized, self.config.simulator.duration_s, self.config.simulator.fps
            )
            log = trace.to_log()
            assert_requested_frequency_realized(log, realized)
            error = compute_error(
                log,
                frequency_hz,
                sim_fps=self.config.simulator.fps,
                real_fps=self.config.metric.real_fps,
                dataset=self.dataset,
                settings=self.config.metric,
            )
            detail = _validated_detail(
                {
                    "call_index": self.call_index,
                    "flat_sha256": _flat_digest(flat),
                    "frequency": frequency_hz,
                    "mean_l2": error["mean_l2"],
                    "per_marker_l2_mm": (error["per_marker_l2"] * 1000).tolist(),
                    "real_markers": error["real_markers"],
                    "real_velocity": error["real_velocity"],
                    "schema_version": 1,
                    "sim_markers": error["sim_markers"],
                    "sim_velocity": error["sim_velocity"],
                    "videofps": error["comparison_fps"],
                },
                fps=error["comparison_fps"],
                head_markers=self.config.metric.head_markers,
                call_index=self.call_index,
                flat_sha256=_flat_digest(flat),
                frequency=frequency_hz,
            )
            score = float(error["mean_l2"]) * 1000.0
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            record = FrequencyRecord(
                frequency_hz,
                self.config.failure_penalty_mm_per_frequency,
                False,
                type(exc).__name__,
                str(exc) or repr(exc),
                max(0.0, time.perf_counter() - started),
            )
            atomic_write_json(
                directory / "outcome.json", {"params": flat, "record": record.to_dict()}
            )
            raise CandidateEvaluationFailure(str(exc), frequency_record=record) from exc
        # The diagnostic file is complete before its outcome is published.
        import os
        import tempfile

        descriptor, temporary = tempfile.mkstemp(suffix=".npz", dir=directory)
        os.close(descriptor)
        try:
            np.savez_compressed(
                temporary, sim_markers=detail["sim_markers"], real_markers=detail["real_markers"]
            )
            os.replace(temporary, directory / "markers.npz")
        finally:
            Path(temporary).unlink(missing_ok=True)
        serial = {k: v for k, v in detail.items() if k not in {"sim_markers", "real_markers"}}
        atomic_write_json(directory / "diagnostic.json", serial)
        record = FrequencyRecord(
            frequency_hz, score, True, None, None, max(0.0, time.perf_counter() - started)
        )
        atomic_write_json(directory / "outcome.json", {"params": flat, "record": record.to_dict()})
        return score

    def recover_frequency(
        self, call_index: int, flat: dict[str, Any], frequency_hz: float
    ) -> FrequencyRecord | None:
        """Recover a completed backend outcome without rerunning its simulation."""
        path = self._directory(call_index, frequency_hz) / "outcome.json"
        if not path.exists():
            return None
        value = read_json(path)
        if value["params"] != flat:
            raise ValueError(f"Cached frequency parameters differ: {path}")
        return FrequencyRecord(**value["record"])

    def details_for(self, flat: dict[str, float], call_index: int) -> dict[float, dict[str, Any]]:
        """Load every diagnostic for a previously evaluated incumbent."""
        import numpy as np

        from swim2real.providers.base import _flat_digest, _validated_detail

        result = {}
        for frequency in self.config.frequencies_hz:
            directory = self._directory(call_index, frequency)
            detail = read_json(directory / "diagnostic.json")
            with np.load(directory / "markers.npz", allow_pickle=False) as saved:
                detail.update(sim_markers=saved["sim_markers"], real_markers=saved["real_markers"])
            result[frequency] = _validated_detail(
                detail,
                fps=self.config.metric.real_fps,
                head_markers=self.config.metric.head_markers,
                call_index=call_index,
                flat_sha256=_flat_digest(flat),
                frequency=frequency,
            )
        return result
