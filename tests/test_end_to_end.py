import json

import numpy as np
import pytest

from swim2real.config import RunConfig
from swim2real.evaluator import ObjectiveEvaluator, SimulationObjective
from swim2real.providers.base import ScriptedProvider
from swim2real.providers.client import CachedClient, ModelResponse
from swim2real.providers.proposal import ModelProposalProvider
from swim2real.search.runner import CalibrationRunner
from swim2real.search.store import RunStore, canonical_sha256
from swim2real.simulators.toy import ToyDataset, ToySimulator


@pytest.mark.parametrize("interrupt_at", [1, 3, 5])
def test_toy_scripted_calibration_resumes_without_extra_simulations(
    tmp_path, interrupt_at, monkeypatch
):
    config = RunConfig.from_dict(
        {
            "budget": 7,
            "simulator": {"backend": "toy", "fps": 30, "duration_s": 3.0},
            "metric": {"max_real_frames": 60},
            "provider": "scripted",
            "use_media": False,
        }
    )
    dataset = ToyDataset(config)
    proposals = [
        np.full(16, 0.5),
        np.zeros(16),
        np.ones(16),
        np.full(16, 0.5),
        np.full(16, 0.25),
        np.full(16, 0.5),
    ]
    simulations = []

    class CountingToy(ToySimulator):
        def run(self, params, frequency_hz, duration_s, fps):
            simulations.append(frequency_hz)
            return super().run(params, frequency_hz, duration_s, fps)

    def run(path, resume=False):
        backend = SimulationObjective(CountingToy(), dataset, config, path / "diagnostics")
        count = len(list((path / "calls").glob("call_*.json"))) if resume else 0
        ev = ObjectiveEvaluator(
            backend.evaluate_frequency,
            budget=config.budget,
            start_call_index=count,
            journal_dir=path / "objective",
            on_call_start=backend.begin_objective,
            recover_frequency=backend.recover_frequency,
        )
        return CalibrationRunner(
            method=config.method,
            problem_seed=1248,
            budget=config.budget,
            output_dir=path,
            evaluator=ev,
            proposal_provider=ScriptedProvider(proposals),
            config_sha256=canonical_sha256(config.to_dict()),
            code_manifest={},
            resume=resume,
            config=config,
        ).run()

    reference = run(tmp_path / "reference")
    simulations.clear()
    original = RunStore.append_call

    def interrupt(store, call, rng_state, algorithm_state=None):
        original(store, call, rng_state, algorithm_state)
        if len(store.calls) == interrupt_at:
            raise KeyboardInterrupt()

    monkeypatch.setattr(RunStore, "append_call", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path / "resumed")
    monkeypatch.setattr(RunStore, "append_call", original)
    resumed = run(tmp_path / "resumed", True)
    assert len(resumed["calls"]) == config.budget
    assert len(simulations) == config.budget * 8
    assert resumed["summary"] == reference["summary"]
    for actual, expected in zip(resumed["calls"], reference["calls"]):
        for key in [
            "call_index",
            "params",
            "normalized_params",
            "phase",
            "round_index",
            "score_mm",
            "valid",
        ]:
            assert actual[key] == expected[key]
    # A completed resume is read-only with respect to objective evaluations.
    assert run(tmp_path / "resumed", True) == resumed
    assert len(simulations) == config.budget * 8
    assert json.loads((tmp_path / "resumed" / "status.json").read_text())["complete"]


@pytest.mark.parametrize("method", ["full_vlm_ls", "signed_permuted_vlm_ls", "vlm_direct"])
@pytest.mark.parametrize(
    "stage", ["simulation", "between_frequencies", "request", "uncached_reply", "cached_reply"]
)
def test_resume_recovers_interrupted_simulation_and_model_stages(
    tmp_path, monkeypatch, stage, method
):
    from swim2real import evaluator
    from swim2real.providers import client, proposal

    config = RunConfig.from_dict(
        {
            "method": method,
            "budget": 4,
            "simulator": {"backend": "toy", "fps": 30, "duration_s": 3.0},
            "metric": {"max_real_frames": 30},
            "use_media": False,
        }
    )
    monkeypatch.setattr(evaluator.time, "perf_counter", lambda: 0.0)
    simulations, requests = [], []
    interrupted = False
    active = False

    def interrupt_once():
        nonlocal interrupted
        if active and not interrupted:
            interrupted = True
            raise KeyboardInterrupt()

    class InterruptingToy(ToySimulator):
        def run(self, params, frequency_hz, duration_s, fps):
            simulations.append(frequency_hz)
            if stage == "simulation" and len(simulations) == 10:
                interrupt_once()
            return super().run(params, frequency_hz, duration_s, fps)

    def transport(prompt, media):
        requests.append(prompt)
        if stage == "request":
            interrupt_once()
        return ModelResponse(
            '{"selected_parameters":["fluidCoef_0"],"proposed_values":{"fluidCoef_0":3}}',
            "scripted-model",
            {"tokens": 1},
        )

    write = evaluator.atomic_write_json

    def write_frequency(path, value):
        write(path, value)
        if (
            stage == "between_frequencies"
            and path.parent.name == "call_0002"
            and path.name == "frequency_02.json"
        ):
            interrupt_once()

    def write_reply(path, value):
        if stage == "uncached_reply" and path.name == "response.json":
            interrupt_once()
        write(path, value)

    parse = proposal.parse_response

    def parse_reply(text):
        if stage == "cached_reply":
            interrupt_once()
        return parse(text)

    monkeypatch.setattr(evaluator, "atomic_write_json", write_frequency)
    monkeypatch.setattr(client, "atomic_write_json", write_reply)
    monkeypatch.setattr(proposal, "parse_response", parse_reply)

    def run(path, resume=False):
        backend = SimulationObjective(
            InterruptingToy(), ToyDataset(config), config, path / "diagnostics"
        )
        ev = ObjectiveEvaluator(
            backend.evaluate_frequency,
            budget=config.budget,
            clock=lambda: 0.0,
            start_call_index=len(list((path / "state").glob("state_*.json"))),
            journal_dir=path / "objective",
            on_call_start=backend.begin_objective,
            recover_frequency=backend.recover_frequency,
        )
        model = CachedClient(
            settings={},
            attempts=1,
            retry_delay=0.0,
            transport=transport,
            retryable=lambda exc: False,
        )
        runner = CalibrationRunner(
            method=method,
            problem_seed=1248,
            budget=config.budget,
            output_dir=path,
            evaluator=ev,
            proposal_provider=ModelProposalProvider(backend, model, config, path),
            config_sha256=canonical_sha256(config.to_dict()),
            code_manifest={},
            resume=resume,
            config=config,
        )
        try:
            return runner.run()
        except KeyboardInterrupt:
            assert ev.call_count == len(list((path / "state").glob("state_*.json")))
            raise

    reference = run(tmp_path / "reference")
    reference_requests = len(requests)
    simulations.clear()
    requests.clear()
    active = True
    path = tmp_path / "resumed"
    with pytest.raises(KeyboardInterrupt):
        run(path)
    assert interrupted
    if stage == "cached_reply":
        moved = tmp_path / "moved"
        path.rename(moved)
        path = moved
    resumed = run(path, resume=True)
    assert resumed == reference
    assert len(resumed["calls"]) == config.budget
    assert len(simulations) == 8 * config.budget + (stage == "simulation")
    assert len(requests) == reference_requests + (stage in {"request", "uncached_reply"})
    redos = list(path.rglob("redo_*.json"))
    assert len(redos) == (stage in {"simulation", "request", "uncached_reply"})
    assert run(path, resume=True) == reference
