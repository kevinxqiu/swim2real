"""Commands for CPU-only reproduction and comparison with the paper."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .numbers import Scalar, compute_numbers, describe_key
from .records import load_records


def default_reference() -> Path:
    """Find the repository's reference, including an installed checkout."""
    for parent in (Path.cwd(), *Path(__file__).resolve().parents):
        candidate = parent / "records/paper_numbers.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("records/paper_numbers.json not found; pass --reference")


def check_numbers(
    numbers: Mapping[str, Scalar], reference: Mapping[str, Any], *, allow_partial: bool = False
) -> dict[str, Any]:
    """Compare full precision and every registered printed value separately."""
    expected = reference["values"]
    result = {
        "matched": 0,
        "differing": 0,
        "skipped": 0,
        "printed_matched": 0,
        "printed_differing": 0,
        "details": [],
    }
    for key, row in expected.items():
        if key not in numbers:
            result["skipped" if allow_partial else "differing"] += 1
            if not allow_partial:
                result["details"].append(f"missing: {key}")
            continue
        actual, target = numbers[key], row["value"]
        if type(target) is bool or target is None:
            equal = actual is target
        else:
            equal = (
                type(actual) in (int, float)
                and math.isfinite(actual)
                and math.isclose(actual, target, rel_tol=1e-10, abs_tol=1e-12)
            )
        result["matched" if equal else "differing"] += 1
        if not equal:
            result["details"].append(f"{key}: {actual!r} != {target!r}")
        for printed in row.get("printed", []):
            equal_print = (
                type(actual) in (int, float)
                and math.isfinite(actual)
                and float(format(actual, printed["format"])) == float(printed["value"])
            )
            result["printed_matched" if equal_print else "printed_differing"] += 1
            if not equal_print:
                result["details"].append(
                    f"printed {key}: {actual!r} does not round to {printed['value']}"
                )
    for extra in sorted(set(numbers) - set(expected)):
        result["differing"] += 1
        result["details"].append(f"unexpected: {extra}")
    result["unrecomputed"] = reference.get("unrecomputed", [])
    result["skipped"] += len(result["unrecomputed"])
    return result


def render_markdown(
    numbers: Mapping[str, Scalar], reference: Mapping[str, Any] | None = None
) -> str:
    """Render definitions, locations, values, and explicit coverage limitations."""
    lines = [
        "# Recomputed paper numbers",
        "",
        (
            "AUC40 averages the first 40 chronological valid best-so-far errors. "
            "Invalid evaluations consume budget and cannot improve the incumbent. "
            "Sample SD uses n minus 1. Paired BCa intervals use 10,000 resamples "
            "with seed 20260812. Exact signed-rank tests discard exact zeros, rank "
            "ties by their average and enumerate the conditional sign distribution. "
            "Holm adjustment is applied to each study's two primary comparisons."
        ),
        "",
        (
            "Per-run values and sensitivity analyses support the reported aggregates; "
            "a blank printed value means the scalar is not separately typeset."
        ),
        "",
        "| Key | Recomputed | Definition | Paper location | Printed |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for key, value in sorted(numbers.items()):
        definition, location = describe_key(key)
        row = reference.get("values", {}).get(key, {}) if reference else {}
        definition = row.get("definition", definition)
        location = row.get("paper_location", location)
        printed = "; ".join(p["value"] for p in row.get("printed", []))
        text = f"{value:.12g}" if type(value) in (int, float) else str(value).lower()
        lines.append(f"| `{key}` | {text} | {definition} | {location} | {printed} |")
    lines.extend(["", "## Values outside these records", ""])
    for item in reference.get("unrecomputed", []) if reference else []:
        lines.append(f"- **{item['paper_location']}**, {item['printed']}: {item['reason']}")
    if reference and reference.get("coverage_notes"):
        lines.extend(["", "## Record coverage", ""])
        lines.extend(f"- {note}" for note in reference["coverage_notes"])
    return "\n".join(lines) + "\n"


def _reproduce(args):
    started = time.perf_counter()
    bundle = load_records(args.records, allow_partial=args.allow_partial)
    numbers = compute_numbers(bundle, allow_partial=args.allow_partial)
    reference_path = args.reference
    if reference_path is None:
        reference_path = default_reference()
    reference = json.loads(reference_path.read_text())
    if args.out_md and args.out_json.resolve() == args.out_md.resolve():
        raise ValueError("JSON and Markdown output paths must differ")
    for target in (args.out_json, args.out_md):
        if target is not None and target.exists():
            raise FileExistsError(f"output already exists: {target}")
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(numbers, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_markdown(numbers, reference))
    report = check_numbers(numbers, reference, allow_partial=args.allow_partial)
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    print(json.dumps(report, indent=2))
    return int(bool(report["differing"] or report["printed_differing"]))


def _check(args):
    numbers = json.loads(args.numbers.read_text())
    reference = json.loads((args.reference or default_reference()).read_text())
    report = check_numbers(numbers, reference, allow_partial=args.allow_partial)
    print(json.dumps(report, indent=2))
    return int(bool(report["differing"] or report["printed_differing"]))


def add_subcommands(subparsers: argparse._SubParsersAction) -> None:
    """Register commands on the package CLI's argparse subparser collection."""
    reproduce = subparsers.add_parser("reproduce", help="recompute the paper's calibration numbers")
    reproduce.add_argument("--records", type=Path, required=True)
    reproduce.add_argument("--out-json", type=Path, required=True)
    reproduce.add_argument("--out-md", type=Path)
    reproduce.add_argument("--reference", type=Path)
    reproduce.add_argument(
        "--allow-partial",
        action="store_true",
        help="compute only per-run fixture values; skip absent keys",
    )
    reproduce.set_defaults(func=_reproduce, handler=_reproduce)
    check = subparsers.add_parser("check-numbers", help="compare recomputed and printed values")
    check.add_argument("numbers", type=Path)
    check.add_argument("--reference", type=Path)
    check.add_argument("--allow-partial", action="store_true")
    check.set_defaults(func=_check, handler=_check)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="swim2real")
    add_subcommands(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
