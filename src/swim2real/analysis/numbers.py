"""Recompute scalar paper results from compact run and assessment records."""

from __future__ import annotations

import re
import statistics as std
from typing import Any, Mapping

import numpy as np

from .records import RecordBundle
from .statistics import (
    ObjectiveCall,
    auc40,
    capped_time_to_threshold,
    chronological_best_so_far,
    exact_wilcoxon_signed_rank,
    holm_adjust_two,
    paired_bca_interval,
)
from .studies import (
    ALPHA,
    BODY_LENGTH_MM,
    BUDGET,
    FAILURE_PENALTY_MM,
    PRIMARY_METHODS,
    REPLAY_BETAS,
    REPLAY_STEPS,
    STUDIES,
    THRESHOLDS_MM,
    complete_seeds,
    primary_seeds,
    study_runs,
)

Scalar = float | int | bool | None


def run_metrics(run: Mapping[str, Any]) -> tuple[dict[str, float | int], tuple[float, ...]]:
    """Compute outcomes from chronological evaluations, never stored summaries."""
    calls = [
        ObjectiveCall(c["call_index"], c["mean_error_mm"], c["valid"], c["accepted"])
        for c in run["evaluations"]
    ]
    tail = run["structural_tail_carry"]
    curve = chronological_best_so_far(calls, budget=BUDGET, structural_tail_carry=tail)
    evaluated_rounds = {c["round_index"] for c in run["evaluations"] if c["round_index"] >= 0}
    proposals = sum(p["round_index"] in evaluated_rounds for p in run["model_proposals"])
    output = {
        "auc40_mm": auc40(calls, structural_tail_carry=tail),
        "final_mm": curve[-1],
        "failed_calls": sum(not c["valid"] for c in run["evaluations"]),
        "failed_frequencies": sum(
            not f["valid"] for c in run["evaluations"] for f in c["frequency_errors"]
        ),
        "proposals": proposals,
    }
    for threshold in THRESHOLDS_MM:
        output[f"calls{threshold:g}"] = capped_time_to_threshold(
            calls, threshold_mm=threshold, structural_tail_carry=tail
        )
    return output, curve


def _describe(out, prefix, values):
    values = list(values)
    out[f"{prefix}.n"] = len(values)
    out[f"{prefix}.mean"] = std.fmean(values)
    out[f"{prefix}.median"] = std.median(values)
    out[f"{prefix}.sd"] = std.stdev(values) if len(values) > 1 else None


def _contrast(out, prefix, differences, *, tests=True):
    differences = list(differences)
    out[f"{prefix}.n"] = len(differences)
    out[f"{prefix}.mean_delta_mm"] = std.fmean(differences)
    out[f"{prefix}.median_delta_mm"] = std.median(differences)
    for name, predicate in (
        ("wins", lambda d: d < 0),
        ("ties", lambda d: d == 0),
        ("losses", lambda d: d > 0),
    ):
        out[f"{prefix}.{name}"] = sum(predicate(d) for d in differences)
    for statistic in ("mean", "median"):
        interval = paired_bca_interval(differences, statistic=statistic)
        for field in ("lower", "upper"):
            out[f"{prefix}.bca_{statistic}.{field}_mm"] = getattr(interval, field)
        out[f"{prefix}.bca_{statistic}.defined"] = interval.defined
    if tests:
        test = exact_wilcoxon_signed_rank(differences)
        for field in (
            "n_nonzero",
            "w_positive",
            "w_negative",
            "statistic",
            "p_value",
            "rank_biserial",
        ):
            out[f"{prefix}.wilcoxon.{field}"] = getattr(test, field)
        out[f"{prefix}.wilcoxon.zeros_discarded"] = test.n_total - test.n_nonzero
        return test.p_value
    return None


def _family(
    out,
    prefix,
    seeds,
    metrics,
    *,
    metric="auc40_mm",
    tests=True,
    adjust=True,
    comparators=None,
):
    comparators = comparators or {
        "random": "multiscale_random_ls",
        "permuted": "signed_permuted_vlm_ls",
    }
    p_values = {}
    for label, comparator in comparators.items():
        differences = [
            metrics[("full_vlm_ls", s)][metric] - metrics[(comparator, s)][metric] for s in seeds
        ]
        p_values[label] = _contrast(out, f"{prefix}.{label}", differences, tests=tests)
    if tests and adjust:
        adjusted = holm_adjust_two(p_values)
        for label, value in adjusted.items():
            out[f"{prefix}.{label}.holm_p"] = value
            out[f"{prefix}.{label}.passes"] = (
                value < ALPHA and out[f"{prefix}.{label}.median_delta_mm"] < 0
            )


def _arms(out, prefix, rows, metrics, methods, seeds=None):
    for method in methods:
        selected = [
            s
            for m, s in rows
            if m == method
            and rows[(m, s)]["completion"]["complete"]
            and (seeds is None or s in seeds)
        ]
        arm = f"{prefix}.{method}"
        out[f"{arm}.n"] = len(selected)
        for metric in ("auc40_mm", "final_mm", "calls55", "calls60", "proposals"):
            _describe(out, f"{arm}.{metric}", [metrics[(method, s)][metric] for s in selected])
        for metric in ("failed_calls", "failed_frequencies"):
            out[f"{arm}.{metric}"] = sum(metrics[(method, s)][metric] for s in selected)
        for threshold in THRESHOLDS_MM:
            out[f"{arm}.calls{threshold:g}.unreached"] = sum(
                metrics[(method, s)][f"calls{threshold:g}"] == BUDGET + 1 for s in selected
            )
        out[f"{arm}.final_above100"] = sum(metrics[(method, s)]["final_mm"] > 100 for s in selected)


def _backtracking(out, rows):
    rounds = 0
    reductions = []
    by_step = {1.0: 0, 0.5: 0, 0.25: 0}
    for (method, _seed), run in rows.items():
        if method != "full_vlm_ls":
            continue
        rounds += len({c["round_index"] for c in run["evaluations"] if c["round_index"] >= 0})
        before = None
        for call in run["evaluations"]:
            if call["accepted"] and before is not None:
                by_step[call["step_size"]] += 1
                reductions.append(before - call["mean_error_mm"])
            before = call["best_so_far_mm"]
    prefix = "backtracking.recorded"
    out[f"{prefix}.rounds"] = rounds
    out[f"{prefix}.accepted"] = len(reductions)
    for step, count in by_step.items():
        out[f"{prefix}.step_{step:g}.accepted"] = count
    smaller = sum(n for step, n in by_step.items() if step < 1)
    out[f"{prefix}.smaller_step.accepted"] = smaller
    out[f"{prefix}.smaller_step.percent"] = 100 * smaller / len(reductions)
    out[f"{prefix}.improvement_median_mm"] = std.median(reductions)


def _replay(out, rounds):
    for beta in REPLAY_BETAS:
        for steps in REPLAY_STEPS:
            prefix = f"backtracking.replay.beta_{beta:g}.K_{steps}"
            improvements = [[] for _ in range(steps)]
            reached = [0] * steps
            for row in rounds:
                evaluations = {c["step_size"]: c for c in row["evaluations"]}
                for rung in range(steps):
                    reached[rung] += 1
                    evaluation = evaluations[beta**rung]
                    if (
                        evaluation["valid"]
                        and evaluation["mean_error_mm"] < row["incumbent_error_mm"]
                    ):
                        improvements[rung].append(
                            row["incumbent_error_mm"] - evaluation["mean_error_mm"]
                        )
                        break
            out[f"{prefix}.rounds"] = len(rounds)
            out[f"{prefix}.calls"] = sum(reached)
            out[f"{prefix}.accepted"] = sum(map(len, improvements))
            for rung, values in enumerate(improvements, 1):
                out[f"{prefix}.rung_{rung}.reached"] = reached[rung - 1]
                out[f"{prefix}.rung_{rung}.accepted"] = len(values)
                out[f"{prefix}.rung_{rung}.improvement_median_mm"] = (
                    std.median(values) if values else None
                )


def _markers(out, rows, seeds):
    for method in PRIMARY_METHODS:
        per_run = []
        for seed in seeds:
            calls = rows[(method, seed)]["evaluations"]
            best = min((c for c in calls if c["valid"]), key=lambda c: c["mean_error_mm"])
            per_run.append(np.mean([r["per_marker_mm"] for r in best["frequency_errors"]], axis=0))
        marker_means = np.mean(per_run, axis=0)
        for marker, value in enumerate(marker_means):
            out[f"replication.markers.{method}.M{marker}_mm"] = float(value)
        for region, value in zip(("head", "mid", "tail"), marker_means.reshape(3, 3).mean(axis=1)):
            out[f"replication.markers.{method}.{region}_mm"] = float(value)


def compute_numbers(bundle: RecordBundle, *, allow_partial: bool = False) -> dict[str, Scalar]:
    """Return a flat stable-key dictionary, including per-run validation values.

    Partial mode emits per-run values only and is intended for the small test
    fixture. It never estimates study results from an incomplete population.
    """
    out = {}
    all_metrics = {}
    all_curves = {}
    for key, run in sorted(bundle.runs.items()):
        if run["completion"]["complete"]:
            metrics, curve = run_metrics(run)
            all_metrics[key] = metrics
            all_curves[key] = curve
            prefix = ".".join((key[0], "runs", key[1], str(key[2])))
            out.update({f"{prefix}.{metric}": value for metric, value in metrics.items()})
    if allow_partial:
        return out
    for study in ("fifteen_seed", "replication", "glm"):
        definition = STUDIES[study]
        rows = study_runs(bundle, study)
        metrics = {}
        curves = {}
        for (method, seed), run in rows.items():
            key = (run["study"], method, seed)
            if key in all_metrics:
                metrics[(method, seed)] = all_metrics[key]
                curves[(method, seed)] = all_curves[key]
        common = complete_seeds(rows, study)
        primary = primary_seeds(rows, study)
        out[f"{study}.scheduled_seeds"] = len(definition.seeds)
        out[f"{study}.common_completed"] = len(common)
        out[f"{study}.primary.n"] = len(primary)
        _arms(out, f"{study}.arms", rows, metrics, definition.methods)
        out[f"{study}.full_final_body_percent"] = (
            100 * out[f"{study}.arms.full_vlm_ls.final_mm.mean"] / BODY_LENGTH_MM
        )
        for method in definition.methods:
            out[f"{study}.arms.{method}.incomplete"] = (
                len(definition.seeds) - out[f"{study}.arms.{method}.n"]
            )
        if definition.shared_failure_seed is not None:
            for method in definition.methods:
                metrics[(method, definition.shared_failure_seed)] = {"auc40_mm": FAILURE_PENALTY_MM}
                curves[(method, definition.shared_failure_seed)] = (FAILURE_PENALTY_MM,) * BUDGET
            out[f"{study}.shared_initialization_failures"] = sum(
                r["completion"]["reason"] == "shared_initialization_failure" for r in rows.values()
            ) // len(definition.methods)
            out[f"{study}.infeasible_initializations"] = sum(
                r["completion"]["reason"] == "infeasible_initialization" for r in rows.values()
            ) // len(definition.methods)
            for method in definition.methods:
                for reason in ("invalid_model_reply", "request_failure"):
                    out[f"{study}.arms.{method}.{reason}"] = sum(
                        r["completion"]["reason"] == reason
                        for (m, _), r in rows.items()
                        if m == method
                    )
        _family(out, f"{study}.primary", primary, metrics)
        if study == "fifteen_seed":
            _backtracking(out, rows)
            direct = out[f"{study}.arms.vlm_direct.proposals.mean"]
            full = out[f"{study}.arms.full_vlm_ls.proposals.mean"]
            out["fifteen_seed.direct_to_full_proposal_ratio"] = direct / full
            full_runs = [rows[("full_vlm_ls", s)] for s in definition.seeds]
            median_run = sorted(
                full_runs, key=lambda r: metrics[("full_vlm_ls", r["seed"])]["final_mm"]
            )[len(full_runs) // 2]
            best = min(
                (c for c in median_run["evaluations"] if c["valid"]),
                key=lambda c: c["mean_error_mm"],
            )
            out["fifteen_seed.calibrated_motion.frequency_error_mm"] = next(
                f["score_mm"] for f in best["frequency_errors"] if f["frequency_hz"] == 1.5
            )
        if study == "replication":
            _family(out, "replication.complete_case", common, metrics, adjust=False)
            _family(
                out,
                "replication.final",
                common,
                metrics,
                metric="final_mm",
                tests=False,
            )
            _arms(
                out,
                "replication.common_arms",
                rows,
                metrics,
                definition.methods,
                common,
            )
            _markers(out, rows, common)
            for label, method in (
                ("random", "multiscale_random_ls"),
                ("permuted", "signed_permuted_vlm_ls"),
            ):
                available = [
                    s
                    for s in definition.seeds
                    if rows[("full_vlm_ls", s)]["completion"]["complete"]
                    and rows[(method, s)]["completion"]["complete"]
                ]
                _contrast(
                    out,
                    f"replication.available_pairs.{label}",
                    [
                        metrics[("full_vlm_ls", s)]["auc40_mm"] - metrics[(method, s)]["auc40_mm"]
                        for s in available
                    ],
                )
            for method in definition.methods:
                primary_values = [metrics[(method, s)]["auc40_mm"] for s in primary]
                _describe(out, f"replication.primary_arms.{method}.auc40_mm", primary_values)
                for call_index in range(BUDGET):
                    out[f"replication.curve.{method}.call_{call_index + 1:02d}_mm"] = std.fmean(
                        curves[(method, s)][call_index] for s in primary
                    )
    assessment_metrics = {}
    for row in bundle.assessments:
        # Preserve the original evaluator's NumPy summation order. BCa midrank
        # bias correction is sensitive to exact ties at machine precision.
        value = float(np.mean([r["score_mm"] for r in row["frequency_errors"]]))
        assessment_metrics[(row["method"], row["seed"])] = {"assessment_mm": value}
        out[f"interpolation.runs.{row['method']}.{row['seed']}.assessment_mm"] = value
    definition = STUDIES["interpolation"]
    out["interpolation.n"] = len(definition.seeds)
    for method in definition.methods:
        _describe(
            out,
            f"interpolation.arms.{method}.assessment_mm",
            [assessment_metrics[(method, s)]["assessment_mm"] for s in definition.seeds],
        )
        out[f"interpolation.arms.{method}.failed_frequencies"] = sum(
            not f["valid"]
            for a in bundle.assessments
            if a["method"] == method
            for f in a["frequency_errors"]
        )
    _family(
        out,
        "interpolation.paired",
        definition.seeds,
        assessment_metrics,
        metric="assessment_mm",
        tests=False,
        comparators={
            "random": "multiscale_random_ls",
            "bayesopt": "bayesopt",
            "reference": "all_frequency_reference",
        },
    )
    _replay(out, bundle.replays)
    return dict(sorted(out.items()))


def describe_key(key: str) -> tuple[str, str]:
    """Provide a definition and paper location for each stable output key."""
    if key == "fifteen_seed.calibrated_motion.frequency_error_mm":
        location = "Calibrated-motion figure caption (fig:calibrated_motion)"
    elif key.endswith(".full_final_body_percent"):
        location = "Section VI-B, calibration error structure"
    elif key.startswith("backtracking."):
        location = "Section VI-A, calibration error and model-query cost"
    elif key.startswith("replication.markers."):
        location = "Section VI-B, calibration error structure"
    elif key.startswith("replication.final."):
        location = "Section V-B, final-error diagnostics"
    elif key.startswith("replication.curve."):
        location = "Replication trajectory figure (fig:replication)"
    elif key.startswith("replication."):
        location = "Table III and Section V-B, fifty-seed replication"
    elif key.startswith("glm."):
        location = "Table IV and Section V-C, second-model experiment"
    elif key.startswith("interpolation."):
        location = "Table V and Section V-D, frequency interpolation"
    else:
        location = "Table II and Section V-A, fifteen-seed study"
    return _key_definition(key), location


def _key_definition(key: str) -> str:
    methods = {
        "full_vlm_ls": "VLM proposals with backtracking",
        "multiscale_random_ls": "multiscale random search",
        "signed_permuted_vlm_ls": "signed-permuted VLM proposals",
        "vlm_direct": "direct VLM updates",
        "bayesopt": "Bayesian optimization",
        "spsa_ls": "SPSA with backtracking",
        "cmaes_pop6": "CMA-ES with population 6",
        "cmaes_pop12": "CMA-ES with population 12",
        "all_frequency_reference": "the all-frequency reference",
    }
    studies = {
        "fifteen_seed": "the fifteen-seed study",
        "replication": "the fifty-seed replication",
        "glm": "the second-model study",
        "interpolation": "the frequency-interpolation study",
    }
    metrics = {
        "auc40_mm": "AUC40, the mean valid best-so-far marker error over all 40 chronological calls (mm), carrying an unused SPSA tail forward",
        "final_mm": "the lowest valid marker error observed by the end of the 40-call budget (mm)",
        "calls55": "the first call with valid best-so-far error at most 55 mm, capped at 41 if unreached",
        "calls60": "the first call with valid best-so-far error at most 60 mm, capped at 41 if unreached",
        "proposals": "the number of model-proposed rounds with at least one evaluated candidate",
        "failed_calls": "the number of invalid objective evaluations, each still consuming one call",
        "failed_frequencies": "the number of invalid single-frequency evaluations",
        "assessment_mm": "the mean terminal marker error at the three held-out frequencies, 0.75, 1.25 and 1.75 Hz (mm)",
    }
    statistics = {
        "mean": "Arithmetic mean of",
        "median": "Median of",
        "sd": "Sample standard deviation (denominator n minus 1) of",
        "n": "Number of observations contributing to",
        "unreached": "Number of runs not reaching the stated threshold in 40 calls, included as 41 in summaries of",
    }
    if key.startswith("backtracking."):
        return _backtracking_definition(key)
    tokens = key.split(".")
    study = tokens[0]
    if study not in studies:
        raise ValueError(f"Unknown paper-number study: {key}")
    study_name = studies[study]
    family = tokens[1]
    if family == "runs":
        method, seed, metric = tokens[2:]
        return f"For {methods[method]}, completed seed {seed} in {study_name}, {metrics[metric]}."
    if family in {"arms", "common_arms", "primary_arms"}:
        method = tokens[2]
        population = f"completed {methods[method]} runs in {study_name}"
        if family == "common_arms":
            population += ", restricted to seeds completed by all three primary methods"
        elif family == "primary_arms":
            population = f"{methods[method]} runs in the primary replication population (jointly completed seeds plus shared-failure seed 1270 at 1000 mm)"
        suffix = tokens[3:]
        if len(suffix) == 2:
            metric, statistic = suffix
            return f"{statistics[statistic]} {metrics[metric]} across {population}."
        metric = suffix[0]
        if metric in metrics:
            return f"Total across {population} of {metrics[metric]}."
        counts = {
            "n": f"Number of {population}",
            "incomplete": f"Number of scheduled {methods[method]} runs in {study_name} that did not complete",
            "final_above100": f"Number of {population} whose final valid best-so-far error exceeds 100 mm",
            "invalid_model_reply": f"Number of scheduled {methods[method]} runs in {study_name} ending with an invalid model reply",
            "request_failure": f"Number of scheduled {methods[method]} runs in {study_name} ending with a model-request failure",
        }
        if metric in counts:
            return counts[metric] + "."
    if family in {"primary", "complete_case", "available_pairs", "final", "paired"}:
        population = {
            "primary": "the primary paired seed population",
            "complete_case": "replication seeds completed by all three primary methods, excluding shared failures",
            "available_pairs": "replication seeds completed by both methods in this contrast",
            "final": "replication seeds completed by all three primary methods, excluding shared failures",
            "paired": "all eight paired terminal assessments in the interpolation study",
        }[family]
        if family == "primary" and study == "replication":
            population += (
                ", including shared-failure seed 1270 at 1000 mm and excluding infeasible seed 1279"
            )
        elif family == "primary":
            population = (
                "all fifteen paired seeds"
                if study == "fifteen_seed"
                else "all twenty-four paired seeds"
            )
        if tokens[2:] == ["n"]:
            return f"Number of seeds in {population} for {study_name}."
        comparator = {
            "random": "multiscale random search",
            "permuted": "signed-permuted VLM proposals",
            "bayesopt": "Bayesian optimization",
            "reference": "the all-frequency reference",
        }[tokens[2]]
        outcome = "final valid best-so-far error" if family == "final" else "AUC40"
        if family == "paired":
            outcome = "terminal held-out marker error"
        context = f"VLM backtracking minus {comparator} {outcome}, paired by seed over {population} in {study_name}"
        return _contrast_definition(".".join(tokens[3:]), context)
    if family == "markers":
        method, marker = tokens[2:]
        region = {
            "head_mm": "head markers M0 to M2",
            "mid_mm": "middle markers M3 to M5",
            "tail_mm": "tail markers M6 to M8",
        }.get(marker, marker.removesuffix("_mm"))
        return f"Mean error of {region} for {methods[method]} (mm), first averaged across frequencies at each seed's lowest-error valid evaluation, then across jointly completed replication seeds."
    if family == "curve":
        method, call = tokens[2:]
        index = int(call.removeprefix("call_").removesuffix("_mm"))
        return f"Mean valid best-so-far error (mm) after chronological call {index} for {methods[method]} over the primary replication population, including shared-failure seed 1270 at 1000 mm."
    special = {
        "scheduled_seeds": f"Number of seeds scheduled for each method in {study_name}",
        "common_completed": f"Number of seeds completed by every primary method in {study_name}",
        "shared_initialization_failures": "Number of replication seeds with a shared initial-point simulation failure, retained in primary contrasts at 1000 mm for every method",
        "infeasible_initializations": "Number of replication seeds with an infeasible shared initial point, excluded from paired analyses",
        "full_final_body_percent": f"Mean final valid best-so-far error of completed VLM-backtracking runs in {study_name}, divided by the 600 mm body length and multiplied by 100",
        "direct_to_full_proposal_ratio": "Mean number of evaluated direct-update model proposals per run divided by the mean for VLM backtracking in the fifteen-seed study",
        "calibrated_motion.frequency_error_mm": "Mean nine-marker error at nominal 1.50 Hz for the lowest-error valid evaluation of the fifteen-seed VLM-backtracking run with median final error",
        "n": "Number of paired seeds assessed at held-out frequencies in the interpolation study",
    }
    suffix = ".".join(tokens[1:])
    if suffix in special:
        return special[suffix] + "."
    raise ValueError(f"Unknown paper-number key: {key}")


def _contrast_definition(suffix: str, context: str) -> str:
    quantities = {
        "n": "Number of paired seeds for",
        "mean_delta_mm": "Arithmetic mean paired difference (mm) for",
        "median_delta_mm": "Median paired difference (mm) for",
        "wins": "Number of strictly negative differences (VLM backtracking wins) for",
        "ties": "Number of exactly zero paired differences for",
        "losses": "Number of strictly positive differences (comparator wins) for",
        "holm_p": "Holm-adjusted two-sided exact signed-rank p-value across the two primary contrasts for",
        "passes": f"Whether the Holm-adjusted p-value is below {ALPHA:g} and the median paired difference is strictly negative for",
        "wilcoxon.n_nonzero": "Number of nonzero paired differences ranked by the exact signed-rank test for",
        "wilcoxon.zeros_discarded": "Number of exactly zero paired differences removed before signed-rank testing for",
        "wilcoxon.w_positive": "Sum of average absolute ranks for strictly positive paired differences in",
        "wilcoxon.w_negative": "Sum of average absolute ranks for strictly negative paired differences in",
        "wilcoxon.statistic": "Smaller positive or negative rank sum used by the exact signed-rank test for",
        "wilcoxon.p_value": "Two-sided exact signed-rank p-value, enumerating the conditional sign distribution with average ranks for ties, for",
        "wilcoxon.rank_biserial": "Positive minus negative rank sum divided by their total (negative favors VLM backtracking) for",
    }
    if suffix in quantities:
        return f"{quantities[suffix]} {context}."
    match = re.fullmatch(r"bca_(mean|median)\.(lower_mm|upper_mm|defined)", suffix)
    if match:
        statistic, field = match.groups()
        method = "the unadjusted 95% paired BCa interval (10,000 resamples, seed 20260812, midrank bias correction)"
        if field == "defined":
            return f"Whether {method} for the {statistic} difference is mathematically defined for {context}."
        bound = field.removesuffix("_mm")
        return f"The {bound} bound (mm) of {method} for the {statistic} paired difference in {context}; null when the interval is undefined."
    raise ValueError(f"Unknown paired-comparison quantity: {suffix}")


def _backtracking_definition(key: str) -> str:
    if key.startswith("backtracking.recorded."):
        suffix = key.removeprefix("backtracking.recorded.")
        context = "the recorded VLM-backtracking rounds in the fifteen-seed study"
        quantities = {
            "rounds": "Number of proposal rounds with at least one evaluated candidate in",
            "accepted": "Number of rounds accepting a valid strict improvement over the incumbent in",
            "smaller_step.accepted": "Number of accepted rounds using step 0.5 or 0.25 in",
            "smaller_step.percent": "Percentage of all accepted rounds that use step 0.5 or 0.25 in",
            "improvement_median_mm": "Median incumbent-error reduction (mm) across accepted rounds in",
        }
        if suffix in quantities:
            return f"{quantities[suffix]} {context}."
        match = re.fullmatch(r"step_(1|0\.5|0\.25)\.accepted", suffix)
        if match:
            return f"Number of rounds whose first valid strict improvement uses step {match[1]} in {context}."
    match = re.fullmatch(
        r"backtracking\.replay\.beta_(0\.25|0\.5|0\.75)\.K_([234])(?:\.rung_([1-4]))?\.(.+)", key
    )
    if match:
        beta, limit, rung, quantity = match.groups()
        context = f"replaying the recorded proposals with steps beta^(r-1), beta={beta}, at most K={limit} evaluations per round, stopping at the first valid strict incumbent improvement"
        if rung is not None:
            context += f", at rung r={rung} (step {float(beta) ** (int(rung) - 1):g})"
        quantities = {
            "rounds": "Number of replayed proposal rounds when",
            "calls": "Total number of candidate evaluations reached when",
            "accepted": "Number of rounds accepting a valid strict incumbent improvement when",
            "reached": "Number of rounds reaching this rung without an earlier accepted improvement when",
            "improvement_median_mm": "Median incumbent-error reduction (mm) among improvements accepted at this rung, or null if none, when",
        }
        if quantity in quantities:
            return f"{quantities[quantity]} {context}."
    raise ValueError(f"Unknown backtracking quantity: {key}")
