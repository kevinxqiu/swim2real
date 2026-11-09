"""Real-record fixture checks and an optional complete-release comparison."""

import copy
import json
import math
import os
from pathlib import Path

import pytest

from swim2real.analysis.cli import check_numbers, main
from swim2real.analysis.numbers import compute_numbers
from swim2real.analysis.records import load_records, validate_run
from swim2real.analysis.studies import STUDIES, complete_seeds

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parent / "fixtures/runs.zip"
REFERENCE = ROOT / "records/paper_numbers.json"


def test_fixture_reproduce_and_check_end_to_end(tmp_path, capsys):
    out_json = tmp_path / "numbers.json"
    out_md = tmp_path / "numbers.md"
    assert (
        main(
            [
                "reproduce",
                "--records",
                str(FIXTURE),
                "--out-json",
                str(out_json),
                "--out-md",
                str(out_md),
                "--reference",
                str(REFERENCE),
                "--allow-partial",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["matched"] == 42
    assert report["differing"] == report["printed_differing"] == 0
    assert report["skipped"] > 0
    assert "first 40" in out_md.read_text()
    assert (
        main(
            [
                "check-numbers",
                str(out_json),
                "--reference",
                str(REFERENCE),
                "--allow-partial",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["matched"] == 42
    # A changed per-run outcome must fail, even in partial mode.
    numbers = json.loads(out_json.read_text())
    numbers[next(iter(numbers))] += 1
    out_json.write_text(json.dumps(numbers))
    assert (
        main(
            [
                "check-numbers",
                str(out_json),
                "--reference",
                str(REFERENCE),
                "--allow-partial",
            ]
        )
        == 1
    )


def test_partial_archive_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="all four"):
        load_records(FIXTURE)


def test_record_validation_recomputes_scores_and_curves():
    bundle = load_records(FIXTURE, allow_partial=True)
    run = copy.deepcopy(next(iter(bundle.runs.values())))
    run["evaluations"][1]["best_so_far_mm"] += 1
    with pytest.raises(ValueError, match="best-so-far"):
        validate_run(run)
    run = copy.deepcopy(next(iter(bundle.runs.values())))
    run["evaluations"][1]["frequency_errors"][0]["score_mm"] += 1
    with pytest.raises(ValueError, match="frequency errors"):
        validate_run(run)


def test_checker_detects_missing_extra_and_printed_precision():
    reference = {
        "values": {"a": {"value": 1.234, "printed": [{"value": "1.2", "format": ".1f"}]}},
        "unrecomputed": [],
    }
    assert check_numbers({}, reference)["differing"] == 1
    assert check_numbers({"a": 1.234, "extra": 1}, reference)["differing"] == 1
    reference["values"]["a"]["printed"][0]["value"] = "1.3"
    report = check_numbers({"a": 1.234}, reference)
    assert report["matched"] == 1
    assert report["printed_differing"] == 1
    assert check_numbers({"a": True}, reference)["differing"] == 1


def test_primary_pairing_does_not_depend_on_descriptive_completion():
    definition = STUDIES["fifteen_seed"]
    rows = {
        (m, s): {"completion": {"complete": m in definition.paired_methods}}
        for m in definition.methods
        for s in definition.seeds
    }
    assert complete_seeds(rows, "fifteen_seed") == definition.seeds


def test_full_release_matches_paper_and_original_summaries():
    configured = os.environ.get("SWIM2REAL_RECORDS")
    archive = Path(configured) if configured else ROOT / "records/runs.zip"
    if not archive.exists():
        if configured:
            pytest.fail("SWIM2REAL_RECORDS points to a missing archive")
        pytest.skip("set SWIM2REAL_RECORDS or place runs.zip in records/ for the full check")
    bundle = load_records(archive)
    assert len(bundle.runs) == 318
    assert len(bundle.assessments) == 32
    assert len(bundle.replays) == 242
    numbers = compute_numbers(bundle)
    reference = json.loads(REFERENCE.read_text())
    report = check_numbers(numbers, reference)
    assert report["differing"] == report["printed_differing"] == 0, report["details"]
    assert report["matched"] == len(reference["values"])
    assert report["printed_matched"] == sum(len(r["printed"]) for r in reference["values"].values())
    checks = json.loads((ROOT / "records/source_checks.json").read_text())
    for group in checks["groups"]:
        for key, target in group["expected"].items():
            actual = numbers[key]
            if target is None or type(target) is bool:
                assert actual is target, key
            else:
                assert math.isclose(actual, target, rel_tol=1e-10, abs_tol=1e-10), (
                    key,
                    actual,
                    target,
                )


def test_top_level_cli_propagates_the_check_status(tmp_path, capsys):
    from swim2real.cli import main as swim2real_main

    out_json = tmp_path / "numbers.json"
    reproduce = ["reproduce", "--records", str(FIXTURE), "--out-json", str(out_json)]
    reproduce += ["--out-md", str(tmp_path / "numbers.md"), "--reference", str(REFERENCE)]
    assert swim2real_main([*reproduce, "--allow-partial"]) == 0
    check = ["check-numbers", str(out_json), "--reference", str(REFERENCE), "--allow-partial"]
    assert swim2real_main(check) == 0
    numbers = json.loads(out_json.read_text())
    numbers[next(iter(numbers))] += 1
    out_json.write_text(json.dumps(numbers))
    capsys.readouterr()
    assert swim2real_main(check) == 1
    # The report is printed once, as JSON, with no trailing status line.
    assert json.loads(capsys.readouterr().out)["differing"] == 1
