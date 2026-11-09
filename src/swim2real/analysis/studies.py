"""Study definitions and populations used by the paper and record exporter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from .records import RecordBundle

PARAMETER_KEYS = (
    *(f"fluidCoef_{i}" for i in range(5)),
    "motorArmLength",
    *(f"hingeStiffness_{i}" for i in range(5)),
    *(f"hingeDamping_{i}" for i in range(5)),
)
PRIMARY_METHODS = (
    "full_vlm_ls",
    "multiscale_random_ls",
    "signed_permuted_vlm_ls",
)
BUDGET = 40
FAILURE_PENALTY_MM = 1000.0
BODY_LENGTH_MM = 600.0
ALPHA = 0.05
THRESHOLDS_MM = (55.0, 60.0)


@dataclass(frozen=True)
class Study:
    """The scheduled matrix and its stated population rule."""

    seeds: tuple[int, ...]
    methods: tuple[str, ...]
    population_rule: str
    shared_failure_seed: int | None = None
    excluded_initialization_seed: int | None = None
    paired_methods: tuple[str, ...] = PRIMARY_METHODS


STUDIES = {
    "fifteen_seed": Study(
        tuple(range(1233, 1248)),
        (
            *PRIMARY_METHODS,
            "bayesopt",
            "spsa_ls",
            "cmaes_pop6",
            "cmaes_pop12",
            "vlm_direct",
        ),
        "All fifteen paired seeds enter the two primary contrasts. Each descriptive "
        "method uses its fifteen completed runs. The direct-update seed 1236 uses "
        "the completed run with a larger output-token limit; this comparison is descriptive.",
    ),
    "replication": Study(
        tuple(range(1248, 1298)),
        PRIMARY_METHODS,
        "Per-method summaries use that method's completed runs. Primary paired "
        "contrasts use seeds completed by all three methods plus seed 1270 at "
        "1000 mm in every method. Seed 1279 has an infeasible shared initialization "
        "and is excluded. Other incomplete runs are excluded without imputation. "
        "Final-error and marker diagnostics use only jointly completed seeds.",
        shared_failure_seed=1270,
        excluded_initialization_seed=1279,
    ),
    "glm": Study(
        (
            1258,
            1259,
            1262,
            1263,
            1264,
            1265,
            1266,
            1267,
            1268,
            1269,
            1271,
            1272,
            1273,
            1274,
            1275,
            1276,
            1277,
            1278,
            1280,
            1281,
            1282,
            1283,
            1284,
            1288,
        ),
        PRIMARY_METHODS,
        "All twenty-four paired seeds enter both contrasts. The complete and "
        "permuted methods use GLM-5.3-Flash; random runs are reused from the "
        "replication on exactly these seeds.",
    ),
    "interpolation": Study(
        tuple(range(1200, 1208)),
        ("full_vlm_ls", "multiscale_random_ls", "bayesopt", "all_frequency_reference"),
        "All eight paired terminal assessments enter the descriptive comparisons. "
        "The training-best parameters are assessed at 0.75, 1.25 and 1.75 Hz. "
        "These conditions are excluded from fitting except in the all-frequency "
        "reference. No significance tests are applied. Training histories are "
        "not included in this archive.",
        paired_methods=(
            "full_vlm_ls",
            "multiscale_random_ls",
            "bayesopt",
            "all_frequency_reference",
        ),
    ),
}
TRAIN_FREQUENCIES_HZ = (0.5, 1.0, 1.5, 2.0, 2.25)
ASSESSMENT_FREQUENCIES_HZ = (0.75, 1.25, 1.75)
REPLAY_BETAS = (0.25, 0.5, 0.75)
REPLAY_STEPS = (2, 3, 4)


def study_runs(bundle: RecordBundle, study: str) -> dict[tuple[str, int], dict[str, Any]]:
    """Return a method/seed index, resolving reused runs without duplication."""
    definition = STUDIES[study]
    rows = {}
    for method in definition.methods:
        source = "replication" if study == "glm" and method == "multiscale_random_ls" else study
        for seed in definition.seeds:
            key = (source, method, seed)
            if key in bundle.runs:
                rows[(method, seed)] = bundle.runs[key]
    return rows


def complete_seeds(
    rows: Mapping[tuple[str, int], Mapping[str, Any]], study: str
) -> tuple[int, ...]:
    """Joint completion is explicit and follows the scheduled seed order."""
    definition = STUDIES[study]
    return tuple(
        seed
        for seed in definition.seeds
        if all(rows[(m, seed)]["completion"]["complete"] for m in definition.paired_methods)
    )


def primary_seeds(rows: Mapping[tuple[str, int], Mapping[str, Any]], study: str) -> tuple[int, ...]:
    """Add only the declared common-failure exception to joint completions."""
    definition = STUDIES[study]
    complete = set(complete_seeds(rows, study))
    return tuple(
        s for s in definition.seeds if s in complete or s == definition.shared_failure_seed
    )
