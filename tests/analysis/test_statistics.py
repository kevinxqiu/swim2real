"""Golden results were generated with the original analysis module."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from swim2real.analysis import statistics as stats

GOLDEN = json.loads((Path(__file__).parent / "fixtures/statistics_golden.json").read_text())


def assert_fields(actual, expected):
    assert actual.keys() == expected.keys()
    for key, target in expected.items():
        if type(target) is float:
            assert actual[key] == pytest.approx(target, rel=1e-13, abs=1e-13)
        else:
            assert actual[key] == target


@pytest.mark.parametrize("case", GOLDEN["cases"])
def test_original_signed_rank_and_bca(case):
    differences = case["differences"]
    assert_fields(asdict(stats.exact_wilcoxon_signed_rank(differences)), case["wilcoxon"])
    for statistic in ("mean", "median"):
        actual = stats.paired_bca_interval(differences, statistic=statistic)
        assert_fields(asdict(actual), case[f"bca_{statistic}"])


def test_original_curve_auc_threshold_and_holm():
    row = GOLDEN["trajectory"]
    assert stats.chronological_best_so_far(row["calls"]) == tuple(row["curve"])
    assert stats.auc40(row["calls"]) == row["auc40"]
    assert stats.capped_time_to_threshold(row["calls"], threshold_mm=55) == row["calls55"]
    assert stats.holm_adjust_two(GOLDEN["holm"]["input"]) == GOLDEN["holm"]["output"]


def test_structural_spsa_tail_and_unaccepted_valid_call():
    calls = [stats.ObjectiveCall(i, 100.0 if i < 39 else 50.0, True, False) for i in range(1, 40)]
    curve = stats.chronological_best_so_far(calls, structural_tail_carry=1)
    assert curve[-2:] == (50.0, 50.0)
    assert stats.auc40(calls, structural_tail_carry=1) == 97.5
    assert stats.capped_time_to_threshold(calls, threshold_mm=55, structural_tail_carry=1) == 39
    with pytest.raises(ValueError, match="count"):
        stats.auc40(calls)


def test_invalid_initial_and_noncontiguous_calls_are_rejected():
    calls = [stats.ObjectiveCall(i, 100.0, True) for i in range(1, 41)]
    calls[0] = stats.ObjectiveCall(1, 1000.0, False)
    with pytest.raises(ValueError, match="initial"):
        stats.auc40(calls)
    calls[0] = stats.ObjectiveCall(2, 100.0, True)
    with pytest.raises(ValueError, match="contiguous"):
        stats.auc40(calls)


def test_seed_pairing_never_silently_intersects():
    with pytest.raises(stats.SeedSetError, match="missing seeds"):
        stats.paired_differences({1: 1.0}, {1: 2.0, 2: 3.0}, (1, 2))
    with pytest.raises(stats.SeedSetError, match="duplicate"):
        stats.validate_exact_seed_set([{"seed": 1}, {"seed": 1}], (1,))
