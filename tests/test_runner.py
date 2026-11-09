import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from swim2real.config import RunConfig
from swim2real.evaluator import ObjectiveEvaluator
from swim2real.search.runner import (
    CalibrationRunner,
    ExternalProposal,
    RunnerInvariantError,
)

CONFIG_SHA256 = "a" * 64


class SimulatedCrash(RuntimeError):
    pass


class _TickingClock:
    def __init__(self) -> None:
        self.value = -1.0

    def __call__(self) -> float:
        self.value += 1.0
        return self.value


class _InterruptingEvaluator(ObjectiveEvaluator):
    def __init__(self, *, start_call_index: int = 0, interrupt_after: int | None = None):
        def evaluate_frequency(params: dict[str, float], frequency: float) -> float:
            vector = np.asarray([params[key] for key in sorted(params)])
            return float(10.0 + np.sum((vector - 0.25) ** 2) + frequency)

        super().__init__(
            evaluate_frequency,
            clock=_TickingClock(),
            start_call_index=start_call_index,
        )
        self.interrupt_after = interrupt_after

    def evaluate(self, params):
        if self.interrupt_after is not None and self.call_count >= self.interrupt_after:
            raise SimulatedCrash(f"interrupted after {self.call_count} calls")
        return super().evaluate(params)


class _ScriptedEvaluator(ObjectiveEvaluator):
    """Return one preset mean score per candidate, independent of the point."""

    def __init__(self, scores: tuple[float, ...]) -> None:
        self.scores = tuple(float(score) for score in scores)

        def evaluate_frequency(params: dict[str, float], frequency: float) -> float:
            del params, frequency
            return self.scores[self.call_count]

        super().__init__(evaluate_frequency, clock=_TickingClock())


def _synthetic_evaluator() -> ObjectiveEvaluator:
    def evaluate_frequency(params: dict[str, float], frequency: float) -> float:
        vector = np.asarray([params[key] for key in sorted(params)])
        return float(10.0 + np.sum((vector - 0.25) ** 2) + frequency)

    return ObjectiveEvaluator(evaluate_frequency, clock=_TickingClock())


def _runner(
    *,
    method: str,
    budget: int,
    output_dir: Path,
    evaluator: ObjectiveEvaluator,
    resume: bool = False,
    proposal_provider=None,
) -> CalibrationRunner:
    return CalibrationRunner(
        method=method,
        problem_seed=1202,
        budget=budget,
        output_dir=output_dir,
        evaluator=evaluator,
        code_manifest={"parent_commit": "abc", "submodule_commit": "def"},
        config_sha256=CONFIG_SHA256,
        proposal_provider=proposal_provider,
        resume=resume,
    )


def test_random_ls_runs_exact_budget_and_writes_complete_cell(tmp_path: Path) -> None:
    runner = CalibrationRunner(
        method="multiscale_random_ls",
        problem_seed=1200,
        budget=7,
        output_dir=tmp_path / "cell",
        evaluator=_synthetic_evaluator(),
        code_manifest={"parent_commit": "abc", "submodule_commit": "def"},
        config_sha256=CONFIG_SHA256,
    )

    result = runner.run()

    assert len(result["calls"]) == 7
    assert result["summary"]["best_call_index"] in range(1, 8)
    assert (tmp_path / "cell" / "result.json").exists()


def test_spsa_uses_probe_tail_and_structurally_leaves_at_most_one(tmp_path: Path) -> None:
    runner = CalibrationRunner(
        method="spsa_ls",
        problem_seed=1201,
        budget=8,
        output_dir=tmp_path / "cell",
        evaluator=_synthetic_evaluator(),
        code_manifest={"parent_commit": "abc", "submodule_commit": "def"},
        config_sha256=CONFIG_SHA256,
    )

    result = runner.run()

    assert len(result["calls"]) in (7, 8)
    assert result["summary"]["structural_tail_carry"] in (0, 1)
    assert len(result["calls"]) + result["summary"]["structural_tail_carry"] == 8
    phases = [call["phase"] for call in result["calls"]]
    assert "spsa_plus" in phases and "spsa_minus" in phases


@pytest.mark.parametrize("method", ["bayesopt", "cmaes_pop12", "cmaes_pop6"])
def test_optimizer_baselines_run_tiny_budget_without_fallback(
    tmp_path: Path,
    method: str,
) -> None:
    if method in {"bayesopt", "cmaes_pop12", "cmaes_pop6"}:
        pytest.importorskip(
            "skopt" if method == "bayesopt" else "cma",
            reason="Optimizer tests require swim2real[baselines]",
        )
    runner = CalibrationRunner(
        method=method,
        problem_seed=1202,
        budget=4,
        output_dir=tmp_path / method,
        evaluator=_synthetic_evaluator(),
        code_manifest={"parent_commit": "abc", "submodule_commit": "def"},
        config_sha256=CONFIG_SHA256,
    )

    result = runner.run()

    assert len(result["calls"]) == 4
    assert result["summary"]["structural_tail_carry"] == 0


def test_unknown_method_fails_before_creating_output(tmp_path: Path) -> None:
    path = tmp_path / "cell"
    with pytest.raises(RunnerInvariantError, match="unsupported method"):
        CalibrationRunner(
            method="unknown",
            problem_seed=1200,
            budget=4,
            output_dir=path,
            evaluator=_synthetic_evaluator(),
            code_manifest={"parent_commit": "abc", "submodule_commit": "def"},
            config_sha256=CONFIG_SHA256,
        )
    assert not path.exists()


@pytest.mark.parametrize("method", ["full_vlm_ls", "signed_permuted_vlm_ls"])
def test_vlm_arms_use_validated_provider_and_exact_budget(tmp_path: Path, method: str) -> None:
    provider_calls = []

    def provider(incumbent, incumbent_loss, calls, round_index):
        provider_calls.append((incumbent_loss, len(calls), round_index))
        return ExternalProposal(
            params=np.clip(incumbent - 0.1, 0.0, 1.0),
            metadata={"model_version": "fake-001", "selected_parameters": ["fluidCoef_0"]},
        )

    result = CalibrationRunner(
        method=method,
        problem_seed=1203,
        budget=6,
        output_dir=tmp_path / method,
        evaluator=_synthetic_evaluator(),
        code_manifest={"parent_commit": "abc", "submodule_commit": "def"},
        config_sha256=CONFIG_SHA256,
        proposal_provider=provider,
    ).run()

    assert len(result["calls"]) == 6
    assert provider_calls
    assert all(
        call.get("proposal_metadata", {}).get("model_version") == "fake-001"
        for call in result["calls"][1:]
    )


def _fake_vlm_provider(incumbent, incumbent_loss, calls, round_index):
    del incumbent_loss, calls
    return ExternalProposal(
        params=np.clip(incumbent - 0.04 * (round_index + 1), 0.0, 1.0),
        metadata={"model_version": "fake-001", "round": round_index},
    )


def test_vlm_direct_scores_one_candidate_per_call_without_line_search(tmp_path: Path) -> None:
    provider_rounds = []

    def provider(incumbent, incumbent_loss, calls, round_index):
        provider_rounds.append(round_index)
        return ExternalProposal(
            params=np.clip(incumbent - 0.1, 0.0, 1.0),
            metadata={"model_version": "fake-001", "selected_parameters": ["fluidCoef_0"]},
        )

    result = _runner(
        method="vlm_direct",
        budget=6,
        output_dir=tmp_path / "cell",
        evaluator=_synthetic_evaluator(),
        proposal_provider=provider,
    ).run()

    assert len(result["calls"]) == 6
    assert result["summary"]["structural_tail_carry"] == 0
    assert provider_rounds == [0, 1, 2, 3, 4]
    assert [call["round_index"] for call in result["calls"]] == [-1, 0, 1, 2, 3, 4]
    for call in result["calls"][1:]:
        metadata: dict[str, Any] = call["proposal_metadata"]
        assert call["phase"] == "vlm_direct"
        assert metadata["protocol"] == "single_candidate_no_line_search_v1"
        assert metadata["model_version"] == "fake-001"
        assert metadata["selected_parameters"] == ["fluidCoef_0"]
        assert len(metadata["external_proposal_normalized"]) == 16
        assert len(metadata["preclip_proposal_normalized"]) == 16
        assert isinstance(metadata["vlm_direct_accepted"], bool)
        assert "planned_candidates" not in metadata


def test_vlm_direct_accepts_only_strict_improvements(tmp_path: Path) -> None:
    def provider(incumbent, incumbent_loss, calls, round_index):
        del incumbent, incumbent_loss, calls
        return ExternalProposal(
            params=np.full(16, (round_index + 1) / 10.0),
            metadata={"model_version": "fake-001", "round": round_index},
        )

    result = _runner(
        method="vlm_direct",
        budget=5,
        output_dir=tmp_path / "cell",
        evaluator=_ScriptedEvaluator((100.0, 90.0, 95.0, 80.0, 80.0)),
        proposal_provider=provider,
    ).run()

    rounds = result["calls"][1:]
    assert [call["score_mm"] for call in rounds] == [90.0, 95.0, 80.0, 80.0]
    assert [call["proposal_metadata"]["vlm_direct_accepted"] for call in rounds] == [
        True,
        False,
        True,
        False,
    ]
    assert [call["proposal_metadata"]["anchor_loss_mm"] for call in rounds] == [
        100.0,
        90.0,
        90.0,
        80.0,
    ]
    assert [call["proposal_metadata"]["anchor_normalized"] for call in rounds[1:]] == [
        [0.1] * 16,
        [0.1] * 16,
        [0.3] * 16,
    ]
    assert result["summary"]["best_call_index"] == 4
    assert result["summary"]["best_score_mm"] == 80.0


@pytest.mark.parametrize("params", [np.zeros(15), np.full(16, np.nan), np.zeros((4, 4))])
def test_vlm_direct_rejects_invalid_provider_output(tmp_path: Path, params) -> None:
    def provider(incumbent, incumbent_loss, calls, round_index):
        del incumbent, incumbent_loss, calls, round_index
        return ExternalProposal(params=params, metadata={"model_version": "fake-001"})

    with pytest.raises(RunnerInvariantError, match="invalid 16D point"):
        _runner(
            method="vlm_direct",
            budget=4,
            output_dir=tmp_path / "cell",
            evaluator=_synthetic_evaluator(),
            proposal_provider=provider,
        ).run()


def test_vlm_direct_requires_an_validated_proposal_provider(tmp_path: Path) -> None:
    path = tmp_path / "cell"
    with pytest.raises(RunnerInvariantError, match="requires a validated proposal_provider"):
        _runner(
            method="vlm_direct",
            budget=4,
            output_dir=path,
            evaluator=_synthetic_evaluator(),
        )
    assert not path.exists()


def test_vlm_direct_records_its_paper_method_code(tmp_path: Path) -> None:
    result = _runner(
        method="vlm_direct",
        budget=3,
        output_dir=tmp_path / "cell",
        evaluator=_synthetic_evaluator(),
        proposal_provider=_fake_vlm_provider,
    ).run()

    assert RunConfig.defaults().method_codes["vlm_direct"] == 107
    assert result["manifest"]["method_code"] == 107
    assert result["manifest"]["algorithm_seed"] is None


@pytest.mark.parametrize(
    ("method", "budget", "provider"),
    [
        ("multiscale_random_ls", 8, None),
        ("spsa_ls", 9, None),
        ("bayesopt", 6, None),
        ("cmaes_pop6", 10, None),
        ("cmaes_pop12", 15, None),
        ("full_vlm_ls", 7, _fake_vlm_provider),
        ("signed_permuted_vlm_ls", 7, _fake_vlm_provider),
        ("vlm_direct", 7, _fake_vlm_provider),
    ],
)
def test_interrupt_resume_matches_uninterrupted_at_every_call_boundary(
    tmp_path: Path,
    method: str,
    budget: int,
    provider,
) -> None:
    if method in {"bayesopt", "cmaes_pop12", "cmaes_pop6"}:
        pytest.importorskip(
            "skopt" if method == "bayesopt" else "cma",
            reason="Optimizer tests require swim2real[baselines]",
        )
    reference = _runner(
        method=method,
        budget=budget,
        output_dir=tmp_path / f"{method}_reference",
        evaluator=_InterruptingEvaluator(),
        proposal_provider=provider,
    ).run()

    for cut in range(1, len(reference["calls"])):
        interrupted_path = tmp_path / f"{method}_cut_{cut}"
        with pytest.raises(SimulatedCrash):
            _runner(
                method=method,
                budget=budget,
                output_dir=interrupted_path,
                evaluator=_InterruptingEvaluator(interrupt_after=cut),
                proposal_provider=provider,
            ).run()

        resumed = _runner(
            method=method,
            budget=budget,
            output_dir=interrupted_path,
            evaluator=_InterruptingEvaluator(start_call_index=cut),
            proposal_provider=provider,
            resume=True,
        ).run()

        assert resumed == reference


def test_resume_requires_evaluator_count_to_equal_stored_calls(tmp_path: Path) -> None:
    path = tmp_path / "cell"
    with pytest.raises(SimulatedCrash):
        _runner(
            method="multiscale_random_ls",
            budget=6,
            output_dir=path,
            evaluator=_InterruptingEvaluator(interrupt_after=2),
        ).run()

    with pytest.raises(RunnerInvariantError, match="evaluator call_count"):
        _runner(
            method="multiscale_random_ls",
            budget=6,
            output_dir=path,
            evaluator=_InterruptingEvaluator(start_call_index=1),
            resume=True,
        ).run()


@pytest.mark.parametrize("bad_hash", ["", "a" * 63, "g" * 64, "A" * 64])
def test_config_hash_must_be_lowercase_sha256(tmp_path: Path, bad_hash: str) -> None:
    with pytest.raises(RunnerInvariantError, match="config_sha256"):
        CalibrationRunner(
            method="multiscale_random_ls",
            problem_seed=1200,
            budget=4,
            output_dir=tmp_path / "cell",
            evaluator=_synthetic_evaluator(),
            code_manifest={"parent_commit": "abc", "submodule_commit": "def"},
            config_sha256=bad_hash,
        )


def test_manifest_keeps_config_hash_and_immutable_initial_vector(tmp_path: Path) -> None:
    result = _runner(
        method="multiscale_random_ls",
        budget=4,
        output_dir=tmp_path / "cell",
        evaluator=_synthetic_evaluator(),
    ).run()

    assert result["manifest"]["config_sha256"] == CONFIG_SHA256
    canonical_code = json.dumps(
        {"parent_commit": "abc", "submodule_commit": "def"},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    assert result["manifest"]["code_sha256"] == hashlib.sha256(canonical_code).hexdigest()
    assert len(result["manifest"]["initial_normalized"]) == 16
    assert result["manifest"]["algorithm_seed"] is None


@pytest.mark.parametrize("method", ["bayesopt", "cmaes_pop12", "cmaes_pop6"])
def test_only_wrapped_optimizers_consume_and_record_algorithm_seed(
    tmp_path: Path, method: str
) -> None:
    pytest.importorskip(
        "skopt" if method == "bayesopt" else "cma",
        reason="Optimizer tests require swim2real[baselines]",
    )
    result = _runner(
        method=method,
        budget=2,
        output_dir=tmp_path / method,
        evaluator=_synthetic_evaluator(),
    ).run()

    assert isinstance(result["manifest"]["algorithm_seed"], int)
    assert result["manifest"]["algorithm_seed"] != 0


def test_calls_store_complete_round_reconstruction_metadata(tmp_path: Path) -> None:
    result = _runner(
        method="spsa_ls",
        budget=6,
        output_dir=tmp_path / "cell",
        evaluator=_synthetic_evaluator(),
    ).run()

    for call in result["calls"][1:]:
        metadata: dict[str, Any] = call["proposal_metadata"]
        assert len(metadata["anchor_normalized"]) == 16
        assert isinstance(metadata["anchor_loss_mm"], float)
        assert "proposal_normalized" in metadata
        assert metadata["spsa_plan"]["plus"]
        assert metadata["spsa_plan"]["minus"]
