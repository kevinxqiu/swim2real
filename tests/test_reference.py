import ast
import json
from pathlib import Path

import numpy as np
import pytest

from swim2real import parameters
from swim2real.config import RunConfig
from swim2real.evaluator import ObjectiveEvaluator
from swim2real.parameters import Bounds16D, initial_vector_for_seed
from swim2real.prompts import paper
from swim2real.prompts.validation import build_prompt
from swim2real.providers.base import (
    ExternalProposal,
    _build_prompt_context,
    _flat_digest,
    _validated_detail,
)
from swim2real.search import protocol as p
from swim2real.search.runner import CalibrationRunner
from swim2real.timebase import nearest_step_sample_times, resample_markers

FIXTURES = Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIXTURES / name).read_text())


def assert_nested(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for k, v in expected.items():
            assert_nested(actual[k], v)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_nested(a, b)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-13, abs=1e-13)
    else:
        assert actual == expected


def test_paper_source_definitions_are_byte_identical():
    def definitions(path):
        source = path.read_text()
        found = {}
        for n in ast.parse(source).body:
            if isinstance(n, ast.FunctionDef):
                found[n.name] = ast.get_source_segment(source, n)
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        found[t.id] = ast.get_source_segment(source, n)
        return found

    reference = definitions(FIXTURES / "prompt_builder_source.py")
    actual = {**definitions(Path(paper.__file__)), **definitions(Path(parameters.__file__))}
    for key in [
        "_JOINT_NAMES",
        "PARAM_INFO_16D",
        "_build_subspace_prompt",
        "_parse_subspace_response",
        "_clean_json",
        "_compute_sweep_mean_l2",
    ]:
        assert actual[key] == reference[key], key


@pytest.mark.parametrize("seed", range(1248, 1298))
def test_initialization_matches_reference(seed):
    np.testing.assert_array_equal(
        initial_vector_for_seed(seed), load("protocol.json")["initial"][str(seed)]
    )


def test_protocol_matches_reference():
    data = load("protocol.json")
    current = np.array(data["current"])
    proposal = np.array(data["proposal"])
    bounds = Bounds16D.from_submitted_bounds()
    np.testing.assert_array_equal(p.make_rng(1248, 100).random(8), data["rng"])
    assert bounds.to_flat(current) == data["flat"]
    np.testing.assert_array_equal(bounds.to_normalized(data["flat"]), data["roundtrip"])
    np.testing.assert_array_equal(
        p.normalize_params(np.linspace(-2, 12, 16), bounds.lower, bounds.upper),
        data["normalized_clipped"],
    )
    np.testing.assert_array_equal(
        p.denormalize_params(proposal, bounds.lower, bounds.upper), data["denormalized_clipped"]
    )
    ray = p.propose_multiscale_random(current, p.make_rng(1248, 101))
    for key, value in data["random"].items():
        np.testing.assert_array_equal(getattr(ray, key), value)
    np.testing.assert_array_equal(
        p.signed_permute_displacement(proposal - current, p.make_rng(1248, 102)), data["signed"]
    )
    plan = p.plan_backtracking_candidates(current, proposal, 3)
    for row, golden in zip(plan, data["plan"]):
        assert row.alpha == golden["alpha"]
        np.testing.assert_array_equal(row.params, golden["params"])
    selection = p.select_first_improvement(current, 5.0, plan, [None, 4.0, 3.0])
    for key, value in data["selection"].items():
        np.testing.assert_array_equal(getattr(selection, key), value)
    spsa = p.plan_spsa_probes(current, p.make_rng(1248, 103), 5)
    for key, value in data["spsa"].items():
        np.testing.assert_array_equal(getattr(spsa, key), value)
    derived = p.derive_spsa_proposal(spsa, 70.0, 72.0)
    for key, value in data["spsa_proposal"].items():
        np.testing.assert_array_equal(getattr(derived, key), value)
    successor = p.select_spsa_successor(
        current, 100.0, [spsa.plus, spsa.minus, derived.params], [70.0, 72.0, 68.0]
    )
    for key, value in data["spsa_selection"].items():
        np.testing.assert_array_equal(getattr(successor, key), value)
    np.testing.assert_array_equal(
        nearest_step_sample_times(0.003, 60.0, 0.3), data["nearest_times"]
    )
    np.testing.assert_array_equal(
        resample_markers([0, 0.2, 0.5], np.arange(18).reshape(3, 3, 2), [0, 0.1, 0.3, 0.5]),
        data["resample"],
    )


@pytest.mark.parametrize(
    "method",
    ["full_vlm_ls", "signed_permuted_vlm_ls", "multiscale_random_ls", "spsa_ls", "vlm_direct"],
)
def test_complete_calls_and_scores_match_original(tmp_path, method):
    bounds = Bounds16D.from_submitted_bounds()
    ev = ObjectiveEvaluator(
        lambda flat, freq: float(sum((bounds.to_normalized(flat) - 0.4) ** 2) + freq),
        clock=lambda: 0.0,
    )

    def provider(anchor, loss, calls, index):
        return ExternalProposal(np.clip(anchor + 0.15 * np.cos(np.arange(16) + index), 0, 1), {})

    result = CalibrationRunner(
        method=method,
        problem_seed=1248,
        budget=12,
        output_dir=tmp_path / "run",
        evaluator=ev,
        code_manifest={},
        config_sha256="a" * 64,
        proposal_provider=provider if "vlm" in method else None,
    ).run()
    assert_nested(
        {"calls": result["calls"], "summary": result["summary"]}, load("runs.json")[method]
    )


def test_prompt_context_matches_original():
    context = load("prompt_context.json")
    with np.load(FIXTURES / "markers.npz") as saved:
        real = saved["real"][:, :30]
    sweep = []
    for i, freq in enumerate(RunConfig.defaults().frequencies_hz):
        sim = real + (0.001 + i * 0.0001)
        l2 = np.linalg.norm(sim - real, axis=2)
        detail = {
            "call_index": 1,
            "flat_sha256": _flat_digest(context["current_flat"]),
            "frequency": freq,
            "mean_l2": float(l2.mean()),
            "per_marker_l2_mm": (l2.mean(axis=1) * 1000).tolist(),
            "real_markers": real,
            "real_velocity": 0.02,
            "schema_version": 1,
            "sim_markers": sim,
            "sim_velocity": 0.018,
            "videofps": 60,
        }
        sweep.append(
            _validated_detail(
                detail, fps=60, call_index=1, flat_sha256=detail["flat_sha256"], frequency=freq
            )
        )
    # The context consumes saved diagnostic scalars. LAPACK builds can differ
    # in the last bit of a freshly fitted velocity.
    for row, saved in zip(sweep, context["diagnostics"]):
        assert row["real_marker_velocity"] == pytest.approx(
            saved["real_marker_velocity"], abs=1e-15
        )
        assert row["sim_marker_velocity"] == pytest.approx(saved["sim_marker_velocity"], abs=1e-15)
        row.update(saved)
    actual = _build_prompt_context(
        current_flat=context["current_flat"],
        history=[],
        incumbent_call_index=1,
        media_paths=[],
        param_keys=list(parameters.PARAM_INFO_16D),
        previously_selected=[],
        round_index=0,
        skeleton_frequencies=RunConfig.defaults().frequencies_hz,
        sweep=sweep,
        worst_frequency=2.25,
        fps=60,
    )
    assert actual == context


@pytest.mark.parametrize(
    "treatment,filename",
    [("eight_frequency", "prompt.txt"), ("interpolation", "prompt_interpolation.txt")],
)
def test_rendered_prompt_matches_reference(treatment, filename):
    context = load("prompt_context.json")
    sweep = context["sweep"]
    if treatment == "interpolation":
        sweep = [row for row in sweep if row["frequency"] in [0.5, 1.0, 1.5, 2.0, 2.25]]
    prompt = build_prompt(
        context["current_flat"],
        sweep,
        [],
        0,
        [],
        worst_frequency=2.25,
        skeleton_frequencies=[s["frequency"] for s in sweep],
        treatment=treatment,
    )
    assert prompt == (FIXTURES / filename).read_text()


def test_prompt_scaffold_matches_supplement_template():
    import re

    template = (FIXTURES / "prompt_template.txt").read_text()
    scaffold = template[
        template.index("### Instruction") : template.index(
            "========================================================================\n2. Variant"
        )
    ].rstrip("\n")
    rendered = (FIXTURES / "prompt.txt").read_text().rstrip("\n")
    table = rendered.split(
        "  ---------------------------------------------------------------------------\n", 1
    )[1].split("\n\n### Full-Spectrum")[0]
    sweep = rendered.split(
        "  -------------------------------------------------------------------------\n", 1
    )[1].split("\n\n### Previous")[0]
    video = rendered[rendered.index("\n### Videos") : rendered.index("\n### Output")]
    replacements = {
        "ROUND_NUMBER": "1",
        "PARAMETER_TABLE": table,
        "FULL_SPECTRUM_SWEEP_TABLE": sweep,
        "HISTORY_SECTION": "### Previous Optimization Rounds\n\n(first round)",
        "REPEAT_WARNING": "",
        "VIDEO_SECTION": video,
    }
    scaffold = re.sub(r"<([A-Z_]+):[^>]*>", lambda match: replacements[match[1]], scaffold)
    assert scaffold == rendered


def test_interpolation_wording_matches_supplement():
    addendum = (FIXTURES / "interpolation_prompt_addendum.txt").read_text()
    rendered = (FIXTURES / "prompt_interpolation.txt").read_text()
    for sentence in [
        "averaged across the 5 preregistered training motor frequencies (0.5, 1.0, 1.5, 2.0, and 2.25 Hz).",
        "### Training-Frequency Sweep (5 preregistered frequencies)",
    ]:
        assert sentence in addendum and sentence in rendered
