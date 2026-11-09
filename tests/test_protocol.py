import numpy as np
import pytest

from swim2real.search.protocol import (
    ALPHAS,
    DIMENSION,
    RANDOM_RADII,
    SPSA_PROBE_RADIUS,
    SPSA_STEP_RADIUS,
    denormalize_params,
    derive_spsa_proposal,
    make_rng,
    normalize_params,
    plan_backtracking_candidates,
    plan_spsa_probes,
    propose_multiscale_random,
    select_first_improvement,
    select_spsa_successor,
    signed_permute_displacement,
)


def _center(value: float = 0.5) -> np.ndarray:
    return np.full(DIMENSION, value, dtype=float)


def test_normalization_round_trip_and_clipping() -> None:
    lower = np.linspace(-2.0, 1.0, DIMENSION)
    upper = lower + np.linspace(0.5, 3.0, DIMENSION)
    normalized = np.linspace(0.0, 1.0, DIMENSION)
    raw = lower + normalized * (upper - lower)

    np.testing.assert_allclose(normalize_params(raw, lower, upper), normalized)
    np.testing.assert_allclose(denormalize_params(normalized, lower, upper), raw)

    below_and_above = normalized.copy()
    below_and_above[0] = -0.2
    below_and_above[-1] = 1.4
    clipped_raw = denormalize_params(below_and_above, lower, upper)
    assert clipped_raw[0] == lower[0]
    assert clipped_raw[-1] == upper[-1]


def test_normalization_requires_exactly_16_non_degenerate_dimensions() -> None:
    with pytest.raises(ValueError, match="16"):
        normalize_params(np.zeros(15), np.zeros(15), np.ones(15))
    with pytest.raises(ValueError, match="exceed"):
        normalize_params(_center(), np.zeros(DIMENSION), np.zeros(DIMENSION))


def test_rng_uses_the_paper_seedsequence_derivation() -> None:
    actual = make_rng(problem_seed=7, method_code=13).normal(size=8)
    expected = np.random.default_rng(np.random.SeedSequence([0x52414C32, 7, 13])).normal(size=8)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        make_rng(7, 13).normal(size=8),
        make_rng(7, 13).normal(size=8),
    )
    assert not np.array_equal(
        make_rng(7, 13).normal(size=8),
        make_rng(7, 14).normal(size=8),
    )


@pytest.mark.parametrize(
    ("remaining_budget", "expected_alphas"),
    [
        (0, ()),
        (1, (1.0,)),
        (2, (1.0, 0.5)),
        (3, ALPHAS),
        (20, ALPHAS),
    ],
)
def test_backtracking_is_an_alpha_ordered_budget_prefix(
    remaining_budget: int, expected_alphas: tuple[float, ...]
) -> None:
    incumbent = _center(0.25)
    proposal = _center(0.85)
    candidates = plan_backtracking_candidates(
        incumbent, proposal, remaining_budget=remaining_budget
    )

    assert tuple(candidate.alpha for candidate in candidates) == expected_alphas
    for candidate in candidates:
        np.testing.assert_allclose(
            candidate.params,
            incumbent + candidate.alpha * (proposal - incumbent),
        )


def test_backtracking_clips_each_coordinate() -> None:
    incumbent = _center(0.9)
    proposal = np.linspace(-0.5, 1.5, DIMENSION)
    candidates = plan_backtracking_candidates(incumbent, proposal, remaining_budget=3)

    assert np.all((candidates[0].params >= 0.0) & (candidates[0].params <= 1.0))
    np.testing.assert_allclose(candidates[0].params, np.clip(proposal, 0.0, 1.0))


def test_backtracking_selects_first_strict_valid_improvement() -> None:
    incumbent = _center(0.2)
    candidates = plan_backtracking_candidates(incumbent, _center(0.8), 3)

    selection = select_first_improvement(
        incumbent,
        incumbent_loss=10.0,
        candidates=candidates,
        candidate_losses=[float("nan"), 9.0, 8.0],
    )

    assert selection.improved
    assert selection.candidate_index == 1
    assert selection.alpha == 0.5
    assert selection.loss == 9.0
    np.testing.assert_array_equal(selection.params, candidates[1].params)


def test_backtracking_rejects_ties_regressions_and_invalid_candidates() -> None:
    incumbent = _center(0.2)
    candidates = plan_backtracking_candidates(incumbent, _center(0.8), 3)

    selection = select_first_improvement(
        incumbent,
        incumbent_loss=10.0,
        candidates=candidates,
        candidate_losses=[10.0, float("inf"), 11.0],
    )

    assert not selection.improved
    assert selection.candidate_index is None
    assert selection.alpha is None
    assert selection.loss == 10.0
    np.testing.assert_array_equal(selection.params, incumbent)


def test_backtracking_never_installs_nonfinite_candidate_coordinates() -> None:
    incumbent = _center(0.2)
    candidates = list(plan_backtracking_candidates(incumbent, _center(0.8), 2))
    candidates[0].params[0] = float("nan")

    selection = select_first_improvement(
        incumbent,
        incumbent_loss=10.0,
        candidates=candidates,
        candidate_losses=[1.0, 9.0],
    )

    assert selection.candidate_index == 1
    assert selection.loss == 9.0


def test_multiscale_random_ray_is_deterministic_and_clipped() -> None:
    incumbent = _center(1.0)
    proposal_a = propose_multiscale_random(incumbent, make_rng(4, 101))
    proposal_b = propose_multiscale_random(incumbent, make_rng(4, 101))

    assert proposal_a.radius in RANDOM_RADII
    assert proposal_a.radius == proposal_b.radius
    np.testing.assert_array_equal(proposal_a.direction, proposal_b.direction)
    np.testing.assert_array_equal(proposal_a.params, proposal_b.params)
    assert np.linalg.norm(proposal_a.direction) == pytest.approx(1.0)
    assert np.linalg.norm(proposal_a.preclip_params - incumbent) == pytest.approx(proposal_a.radius)
    assert np.all((proposal_a.params >= 0.0) & (proposal_a.params <= 1.0))
    assert np.any(proposal_a.params != proposal_a.preclip_params)


def test_multiscale_random_draws_gaussian_before_radius() -> None:
    rng = make_rng(4, 101)
    expected_gaussian = rng.normal(size=DIMENSION)
    expected_radius = RANDOM_RADII[int(rng.integers(0, len(RANDOM_RADII)))]

    proposal = propose_multiscale_random(_center(0.5), make_rng(4, 101))

    np.testing.assert_array_equal(
        proposal.direction, expected_gaussian / np.linalg.norm(expected_gaussian)
    )
    assert proposal.radius == expected_radius


def test_signed_permutation_preserves_displacement_invariants() -> None:
    displacement = np.array(
        [0.0, -0.1, 0.2, -0.3, 0.0, 0.5, -0.8, 0.01] + [0.0] * (DIMENSION - 8),
        dtype=float,
    )
    shuffled = signed_permute_displacement(displacement, make_rng(8, 202))

    assert np.count_nonzero(shuffled == 0.0) == np.count_nonzero(displacement == 0.0)
    np.testing.assert_array_equal(np.sort(np.abs(shuffled)), np.sort(np.abs(displacement)))
    assert np.linalg.norm(shuffled) == pytest.approx(np.linalg.norm(displacement))
    np.testing.assert_array_equal(
        shuffled,
        signed_permute_displacement(displacement, make_rng(8, 202)),
    )


def test_signed_permutation_matches_paper_literal_coordinate_mapping() -> None:
    displacement = np.linspace(-0.8, 0.7, DIMENSION, dtype=float)
    expected_rng = make_rng(1200, 202)
    permutation = expected_rng.permutation(DIMENSION)
    signs = 2.0 * expected_rng.integers(0, 2, size=DIMENSION).astype(float) - 1.0

    observed = signed_permute_displacement(displacement, make_rng(1200, 202))

    np.testing.assert_array_equal(observed, signs * displacement[permutation])


def test_spsa_probe_plan_is_symmetric_and_coordinatewise_bounded() -> None:
    incumbent = np.array(
        [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0, 0.3] + [0.5] * (DIMENSION - 8),
        dtype=float,
    )
    plan = plan_spsa_probes(incumbent, make_rng(2, 303), remaining_budget=5)
    expected_h = np.minimum.reduce(
        [
            np.full(DIMENSION, SPSA_PROBE_RADIUS),
            incumbent,
            1.0 - incumbent,
        ]
    )

    assert plan.has_probes
    np.testing.assert_allclose(plan.h, expected_h)
    np.testing.assert_allclose((plan.plus + plan.minus) / 2.0, incumbent)
    np.testing.assert_allclose(plan.plus - incumbent, plan.h * plan.delta)
    np.testing.assert_allclose(incumbent - plan.minus, plan.h * plan.delta)
    assert set(np.unique(plan.delta)) <= {-1.0, 1.0}
    assert np.all((plan.plus >= 0.0) & (plan.plus <= 1.0))
    assert np.all((plan.minus >= 0.0) & (plan.minus <= 1.0))
    assert plan.line_search_budget == 3
    assert plan.unused_budget == 0


@pytest.mark.parametrize(
    ("remaining", "has_probes", "line_search_budget", "unused"),
    [
        (0, False, 0, 0),
        (1, False, 0, 1),
        (2, True, 0, 0),
        (3, True, 1, 0),
        (4, True, 2, 0),
        (5, True, 3, 0),
    ],
)
def test_spsa_tail_uses_two_probes_and_leaves_at_most_one_call(
    remaining: int, has_probes: bool, line_search_budget: int, unused: int
) -> None:
    plan = plan_spsa_probes(_center(), make_rng(3, 303), remaining)

    assert plan.has_probes is has_probes
    assert plan.line_search_budget == line_search_budget
    assert plan.unused_budget == unused
    assert plan.unused_budget <= 1


@pytest.mark.parametrize("total_budget", range(41))
def test_repeated_spsa_plans_consume_all_but_at_most_one_call(
    total_budget: int,
) -> None:
    remaining = total_budget
    rng = make_rng(3, 303)
    while remaining >= 2:
        plan = plan_spsa_probes(_center(), rng, remaining)
        remaining -= 2 + plan.line_search_budget

    assert remaining in (0, 1)


def test_spsa_proposal_uses_coordinatewise_gradient_and_fixed_radius() -> None:
    incumbent = _center()
    plan = plan_spsa_probes(incumbent, make_rng(5, 303), 5)
    proposal = derive_spsa_proposal(plan, plus_loss=1.0, minus_loss=3.0)

    expected_gradient = (1.0 - 3.0) / (2.0 * plan.h * plan.delta)
    np.testing.assert_allclose(proposal.gradient, expected_gradient)
    assert proposal.valid
    assert proposal.reason is None
    assert proposal.params is not None
    assert proposal.preclip_params is not None
    assert np.linalg.norm(proposal.preclip_params - incumbent) == pytest.approx(SPSA_STEP_RADIUS)
    assert np.all((proposal.params >= 0.0) & (proposal.params <= 1.0))


@pytest.mark.parametrize(
    ("plus_loss", "minus_loss", "reason"),
    [
        (4.0, 4.0, "zero_gradient"),
        (float("nan"), 4.0, "invalid_probe_loss"),
        (4.0, float("inf"), "invalid_probe_loss"),
    ],
)
def test_spsa_proposal_invalid_tie_and_loss_behavior(
    plus_loss: float, minus_loss: float, reason: str
) -> None:
    plan = plan_spsa_probes(_center(), make_rng(6, 303), 5)
    proposal = derive_spsa_proposal(plan, plus_loss, minus_loss)

    assert not proposal.valid
    assert proposal.params is None
    assert proposal.reason == reason


def test_spsa_successor_updates_to_best_probe_and_uses_earliest_tie() -> None:
    incumbent = _center(0.5)
    plus = _center(0.6)
    minus = _center(0.4)
    later = _center(0.7)

    selection = select_spsa_successor(
        incumbent,
        incumbent_loss=10.0,
        chronological_candidates=[plus, minus, later],
        candidate_losses=[8.0, 8.0, 9.0],
    )

    assert selection.improved
    assert selection.candidate_index == 0
    assert selection.loss == 8.0
    np.testing.assert_array_equal(selection.params, plus)


def test_spsa_successor_never_installs_invalid_candidate() -> None:
    incumbent = _center(0.5)
    invalid = _center(0.1)
    valid = _center(0.4)

    selection = select_spsa_successor(
        incumbent,
        incumbent_loss=10.0,
        chronological_candidates=[invalid, valid],
        candidate_losses=[float("nan"), 9.0],
    )

    assert selection.candidate_index == 1
    assert selection.loss == 9.0
    np.testing.assert_array_equal(selection.params, valid)


def test_spsa_successor_keeps_incumbent_on_equal_or_invalid_losses() -> None:
    incumbent = _center(0.5)
    selection = select_spsa_successor(
        incumbent,
        incumbent_loss=10.0,
        chronological_candidates=[_center(0.4), _center(0.3)],
        candidate_losses=[10.0, float("inf")],
    )

    assert not selection.improved
    assert selection.candidate_index is None
    assert selection.loss == 10.0
    np.testing.assert_array_equal(selection.params, incumbent)
