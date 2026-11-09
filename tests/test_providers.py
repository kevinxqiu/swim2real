import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from swim2real.config import RunConfig
from swim2real.prompts.validation import parse_response
from swim2real.providers.client import CachedClient, ModelResponse
from swim2real.providers.gemini import GeminiClient
from swim2real.providers.openai_compat import (
    OpenAICompatibleClient,
    apply_frame_notice,
    extract_proposal_json,
    sample_frame_indices,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        "null",
        "1",
        "{}",
        '{"selected_parameters":[[]],"proposed_values":{}}',
        '{"selected_parameters":["fluidCoef_0","fluidCoef_0"],"proposed_values":{"fluidCoef_0":1}}',
        '{"selected_parameters":["fluidCoef_0"],"proposed_values":{"fluidCoef_1":1}}',
        '{"selected_parameters":["fluidCoef_0"],"proposed_values":{"fluidCoef_0":NaN}}',
        '{"selected_parameters":["fluidCoef_0"],"proposed_values":{"fluidCoef_0":true}}',
    ],
)
def test_response_structure_is_validated_before_paper_parser(text):
    with pytest.raises(ValueError):
        parse_response(text)


def test_valid_parser_clips_values_in_the_same_way_as_paper():
    text = '```json\n{"selected_parameters":["fluidCoef_0"],"proposed_values":{"fluidCoef_0":12},"confidence":0.8}\n```'
    selected, values, parsed = parse_response(text)
    assert selected == ["fluidCoef_0"] and values == {"fluidCoef_0": 10.0}
    assert parsed["proposed_values"]["fluidCoef_0"] == 12


def test_client_retries_transport_errors_and_reuses_cached_response(tmp_path):
    calls = []

    def transport(prompt, media):
        calls.append(prompt)
        if len(calls) == 1:
            raise TimeoutError("transport timeout")
        return ModelResponse("valid response", "model-version", {"tokens": 4})

    client = CachedClient(
        settings={"model": "model"},
        attempts=3,
        retry_delay=0.0,
        transport=transport,
        retryable=lambda exc: isinstance(exc, TimeoutError),
    )
    first = client.call("prompt", [], tmp_path, request_identity={"round": 0})
    second = client.call("prompt", [], tmp_path, request_identity={"round": 0})
    assert first == second and len(calls) == 2
    with pytest.raises(ValueError, match="differs"):
        client.call("changed prompt", [], tmp_path, request_identity={"round": 0})
    saved = json.loads((tmp_path / "response.json").read_text())
    saved["response"]["text"] = "changed response"
    (tmp_path / "response.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="mismatch"):
        client.call("prompt", [], tmp_path, request_identity={"round": 0})


def test_client_resends_interrupted_request_and_records_redo(tmp_path):
    class PowerLoss(BaseException):
        pass

    attempts = []

    def transport(prompt, media):
        attempts.append(prompt)
        if len(attempts) == 1:
            raise PowerLoss()
        return ModelResponse("reply", "model", {})

    client = CachedClient(
        settings={}, attempts=1, retry_delay=0.0, transport=transport, retryable=lambda exc: False
    )
    with pytest.raises(PowerLoss):
        client.call("prompt", [], tmp_path, request_identity={})
    assert client.call("prompt", [], tmp_path, request_identity={}).text == "reply"
    assert attempts == ["prompt", "prompt"]
    assert json.loads((tmp_path / "retries/redo_0001.json").read_text())["redo_index"] == 1
    assert not (tmp_path / "pending.json").exists()


def test_client_rejects_corrupt_pending_request(tmp_path):
    def transport(prompt, media):
        raise KeyboardInterrupt()

    client = CachedClient(
        settings={}, attempts=1, retry_delay=0.0, transport=transport, retryable=lambda exc: False
    )
    with pytest.raises(KeyboardInterrupt):
        client.call("prompt", [], tmp_path, request_identity={})
    (tmp_path / "pending.json").write_text(json.dumps({"attempt": 1, "request_sha256": "bad"}))
    with pytest.raises(ValueError, match="pending request mismatch"):
        client.call("prompt", [], tmp_path, request_identity={})


def test_second_model_request_matches_original():
    config = RunConfig.defaults()
    client = OpenAICompatibleClient(config.openai_compat)
    prompt = (FIXTURES / "prompt.txt").read_text()
    assert client.request_body(prompt, [[b"frame one", b"frame two"]]) == json.loads(
        (FIXTURES / "openai_request.json").read_text()
    )
    assert sample_frame_indices(60, 16) == tuple(i * 59 // 15 for i in range(16))
    with pytest.raises(ValueError, match="few frames"):
        sample_frame_indices(10, 16)
    with pytest.raises(ValueError, match="already"):
        apply_frame_notice(apply_frame_notice(prompt, 16), 16)


def test_second_model_has_its_own_decoding_and_transport_settings():
    config = RunConfig.from_dict({"vlm": {"max_output_tokens": 999, "max_transport_attempts": 1}})
    client = OpenAICompatibleClient(config.openai_compat)
    assert client.attempts == 5 and client.retry_delay == 2.0
    assert config.openai_compat.max_output_tokens == 2048
    assert config.openai_compat.base_url == "https://inference-api.nousresearch.com/v1"
    assert config.openai_compat.reasoning_effort == "low"
    assert config.openai_compat.reasoning_style == "reasoning"
    assert config.openai_compat.parse_mode == "lenient_fenced_json"


def test_second_model_retries_with_the_adapter_backoff(tmp_path, monkeypatch):
    client = OpenAICompatibleClient(RunConfig.defaults().openai_compat)
    delays = []

    def timeout(prompt, media):
        raise TimeoutError("transport timeout")

    client.transport = timeout
    monkeypatch.setattr("swim2real.providers.client.time.sleep", delays.append)
    with pytest.raises(TimeoutError):
        client.call("prompt", [], tmp_path, request_identity={})
    assert delays == [2.0, 4.0, 8.0, 16.0]


@pytest.mark.parametrize("choices", [[], [None]])
def test_second_model_rejects_empty_choices_and_null_content(choices, monkeypatch):
    import sys
    import types
    from types import SimpleNamespace

    config = RunConfig.defaults().openai_compat
    monkeypatch.setenv(config.api_key_env, "synthetic-test-key")
    monkeypatch.setitem(sys.modules, "openai", types.ModuleType("openai"))
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=value)) for value in choices]
    )
    client = OpenAICompatibleClient(config)
    client._sdk = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response))
    )
    with pytest.raises(ValueError, match="no choices|empty or null content"):
        client._generate("prompt", [])


def test_diagnostic_velocity_uses_configured_head_markers():
    from swim2real.providers.base import _velocity_from_markers

    markers = np.zeros((9, 5, 2))
    markers[2, :, 0] = np.arange(5) * 3.0
    assert _velocity_from_markers(markers, 1.0, head_markers=3) == pytest.approx(-1.0)
    assert _velocity_from_markers(markers, 1.0, head_markers=2) == 0.0


def test_second_model_json_extraction_respects_string_braces():
    response = 'Some reasoning.\n```json\n{"diagnosis":"curly } brace","selected_parameters":["fluidCoef_0"],"proposed_values":{"fluidCoef_0":1}}\n```'
    extracted = extract_proposal_json(response)
    assert extracted.method == "fenced"
    assert parse_response(extracted.extracted_text)[1] == {"fluidCoef_0": 1.0}
    assert extract_proposal_json('text {"a":"}"} tail').extracted_text == '{"a":"}"}'


def test_gemini_uses_configured_decoding_and_environment_key(monkeypatch, tmp_path):
    config = RunConfig.from_dict(
        {
            "vlm": {
                "model": "test-model",
                "temperature": 0.2,
                "timeout_seconds": 17.0,
                "decoding_seed": 9,
            }
        }
    )
    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    sdk_types = types.ModuleType("google.genai.types")
    captured = {}

    def value(**kwargs):
        return kwargs

    class Part:
        from_text = staticmethod(value)
        from_bytes = staticmethod(value)

    sdk_types.Part = Part
    for name in [
        "HttpOptions",
        "HttpRetryOptions",
        "Content",
        "GenerateContentConfig",
        "ThinkingConfig",
    ]:
        setattr(sdk_types, name, value)

    def generate(**kwargs):
        captured["generate"] = kwargs
        return types.SimpleNamespace(
            text="reply", model_version="test-model-version", usage_metadata=None, response_id="id"
        )

    def client(**kwargs):
        captured["client"] = kwargs
        return types.SimpleNamespace(models=types.SimpleNamespace(generate_content=generate))

    genai.Client = client
    genai.types = sdk_types
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", sdk_types)
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-test-key")
    result = GeminiClient(config.vlm).call("prompt", [], tmp_path, request_identity={})
    assert result.text == "reply"
    assert captured["client"]["api_key"] == "synthetic-test-key"
    assert captured["client"]["http_options"]["timeout"] == 17000
    assert captured["generate"]["model"] == "test-model"
    assert captured["generate"]["config"]["temperature"] == 0.2
    assert captured["generate"]["config"]["seed"] == 9
    for path in tmp_path.glob("*.json"):
        assert "synthetic-test-key" not in path.read_text()


def test_model_proposal_provider_reuses_context_and_caches_reply(tmp_path):
    from swim2real.evaluator import ObjectiveEvaluator, SimulationObjective
    from swim2real.parameters import Bounds16D, initial_vector_for_seed
    from swim2real.providers.proposal import ModelProposalProvider
    from swim2real.simulators.toy import ToyDataset, ToySimulator

    config = RunConfig.from_dict(
        {
            "simulator": {"backend": "toy", "duration_s": 3.0, "fps": 30},
            "metric": {"max_real_frames": 60},
            "use_media": False,
        }
    )
    backend = SimulationObjective(
        ToySimulator(), ToyDataset(config), config, tmp_path / "diagnostics"
    )
    initial = initial_vector_for_seed(1248)
    flat = Bounds16D.from_submitted_bounds().to_flat(initial)
    call = (
        ObjectiveEvaluator(backend.evaluate_frequency, on_call_start=backend.begin_objective)
        .evaluate(flat)
        .to_dict()
    )
    call.update(normalized_params=initial.tolist(), round_index=-1, phase="initial")
    requests = []

    def transport(prompt, media):
        requests.append(prompt)
        return ModelResponse(
            '{"selected_parameters":["fluidCoef_0"],"proposed_values":{"fluidCoef_0":3}}',
            "model",
            {},
        )

    client = CachedClient(
        settings={}, attempts=1, retry_delay=0.0, transport=transport, retryable=lambda exc: False
    )
    provider = ModelProposalProvider(backend, client, config, tmp_path)
    first = provider(initial, call["score_mm"], [call], 0)
    second = provider(initial, call["score_mm"], [call], 0)
    np.testing.assert_array_equal(first.params, second.params)
    assert len(requests) == 1 and first.metadata["proposed_values"] == {"fluidCoef_0": 3.0}
