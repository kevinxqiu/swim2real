import json
import math

import numpy as np
import pytest

from swim2real.evaluator import FREQUENCIES_HZ, CandidateEvaluationFailure, ObjectiveEvaluator


@pytest.mark.parametrize("bad", [None, True, -1.0, math.nan, math.inf, "3"])
def test_invalid_scores_are_penalized_and_never_eligible(bad, tmp_path):
    seen = []

    def backend(params, frequency):
        seen.append(frequency)
        return bad if frequency == 0.75 else 1.0

    ev = ObjectiveEvaluator(backend, budget=1, journal_dir=tmp_path)
    result = ev.evaluate({"x": 1})
    assert seen == list(FREQUENCIES_HZ)
    assert not result.valid and not result.incumbent_eligible
    assert result.score_mm == pytest.approx((1000.0 + 7) / 8)
    assert result.failure_count == 1
    assert len(result.frequency_records) == 8
    with pytest.raises(RuntimeError, match="budget"):
        ev.evaluate({"x": 1})
    resumed = ObjectiveEvaluator(backend, start_call_index=0, journal_dir=tmp_path)
    assert resumed.evaluate({"x": 1}) == result
    assert len(seen) == 8


def test_candidate_failure_finishes_entire_sweep():
    def backend(params, freq):
        if freq == 1.0:
            raise CandidateEvaluationFailure("unstable simulation")
        return 5.0

    result = ObjectiveEvaluator(backend).evaluate({})
    assert len(result.frequency_records) == 8 and result.failure_count == 1


def test_external_failure_is_visible():
    def backend(params, freq):
        raise FileNotFoundError("missing.csv")

    with pytest.raises(FileNotFoundError, match="missing.csv"):
        ObjectiveEvaluator(backend).evaluate({})


def test_journal_recovery_does_not_repeat_completed_frequencies(tmp_path):
    class PowerLoss(BaseException):
        pass

    seen = []
    saved = {}

    def backend(params, freq):
        from swim2real.evaluator import FrequencyRecord

        seen.append(freq)
        if freq == 1.0 and freq not in saved:
            saved[freq] = FrequencyRecord(freq, 3.0, True, None, None, 0.0)
            raise PowerLoss()
        return 3.0

    first = ObjectiveEvaluator(backend, journal_dir=tmp_path, clock=lambda: 0.0)
    with pytest.raises(PowerLoss):
        first.evaluate({"x": 1})
    resumed = ObjectiveEvaluator(
        backend,
        journal_dir=tmp_path,
        clock=lambda: 0.0,
        recover_frequency=lambda index, params, freq: saved[freq],
    )
    result = resumed.evaluate({"x": 1})
    assert result.valid and result.score_mm == 3.0
    assert seen == list(FREQUENCIES_HZ)


def test_interrupted_frequency_is_redone_without_consuming_budget(tmp_path):
    class PowerLoss(BaseException):
        pass

    def backend(params, freq):
        raise PowerLoss()

    first = ObjectiveEvaluator(backend, budget=1, journal_dir=tmp_path)
    with pytest.raises(PowerLoss):
        first.evaluate({})
    assert first.call_count == 0
    resumed = ObjectiveEvaluator(lambda params, freq: 1.0, budget=1, journal_dir=tmp_path)
    assert resumed.evaluate({}).score_mm == 1.0
    assert resumed.call_count == 1
    redo = tmp_path / "call_0001/retries/frequency_01/redo_0001.json"
    assert json.loads(redo.read_text())["identity"]["frequency_hz"] == FREQUENCIES_HZ[0]


def test_interrupted_frequency_rejects_changed_intent(tmp_path):
    def backend(params, freq):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        ObjectiveEvaluator(backend, journal_dir=tmp_path).evaluate({})
    pending = tmp_path / "call_0001/frequency_01.intent.json"
    value = json.loads(pending.read_text())
    value["frequency_hz"] = 7.0
    pending.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Frequency journal mismatch"):
        ObjectiveEvaluator(lambda params, freq: 1.0, journal_dir=tmp_path).evaluate({})


def test_journal_rejects_changed_settings_and_corruption(tmp_path):
    ev = ObjectiveEvaluator(lambda params, freq: 1.0, journal_dir=tmp_path, clock=lambda: 0.0)
    ev.evaluate({"x": 1})
    with pytest.raises(ValueError, match="mismatch"):
        ObjectiveEvaluator(
            lambda params, freq: 1.0, failure_penalty_mm=7.0, journal_dir=tmp_path
        ).evaluate({"x": 1})
    record = tmp_path / "call_0001" / "frequency_01.json"
    payload = json.loads(record.read_text())
    payload["score_mm"] = -1
    record.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="nonnegative"):
        ObjectiveEvaluator(lambda params, freq: 1.0, journal_dir=tmp_path).evaluate({"x": 1})


def test_invalid_params_do_not_consume_budget():
    ev = ObjectiveEvaluator(lambda params, freq: 1.0)
    with pytest.raises(ValueError):
        ev.evaluate({"x": np.nan})
    assert ev.call_count == 0
