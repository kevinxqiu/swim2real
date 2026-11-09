# Swim2Real calibration records

Compact numerical records accompanying *Swim2Real: Vision-Language Models
for Simulator Calibration of a Soft Robotic Fish*, Kevin Qiu, Kyle Liam
Walker, Mike Yan Michelis, Marek Cygan and Josie Hughes, IEEE Robotics and
Automation Letters, 2026. The [paper](https://arxiv.org/abs/2603.20827) and
[code](https://github.com/kevinxqiu/swim2real) describe the calibration method.

The records recompute the calibration outcomes on a laptop CPU with NumPy.
The full check takes approximately four seconds on the release test machine.
They are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
with the canonical legal code in [LICENSE](LICENSE). Cite the paper when using
the records and indicate changes to the data.

## Distribution and recomputation

| File | Contents |
| --- | --- |
| `runs.zip` | Approximately 6.1 MB compressed; 318 run histories, four study manifests, 32 terminal interpolation assessments, 242 fixed-incumbent proposal replays, schema, data card and license |
| `SHA256SUMS` | SHA-256 of the archive |
| `MANIFEST.sha256` | SHA-256 of each file inside the archive |

Check `SHA256SUMS` before opening the archive. A directory containing
`runs.zip`, the ZIP itself, and an extracted directory are accepted by the
loader. With the package installed:

```bash
swim2real reproduce --records run-records \
  --out-json numbers.json --out-md numbers.md
swim2real check-numbers numbers.json
```

The analysis module also runs directly from a checkout:

```bash
python3 -m venv .venv
.venv/bin/pip install numpy scipy
unzip runs.zip -d run-records
PYTHONPATH=src .venv/bin/python -m swim2real.analysis.cli reproduce \
  --records run-records --out-json numbers.json --out-md numbers.md
PYTHONPATH=src .venv/bin/python -m swim2real.analysis.cli check-numbers numbers.json
```

Choose new output filenames for each recomputation. The commands use no
network and make no simulator or model requests. The reference and report are
`records/paper_numbers.json` and `records/paper_numbers.md` in the code
repository. The JSON output is a flat dictionary with stable scalar keys.
The checker compares every scalar and separately checks every registered
paper value at its printed precision. Its skipped count names the four
groups of printed values outside these calibration records.

## How the records were made

The objective compares nine aligned centerline markers in real and simulated
swimming. An evaluation runs the candidate parameter vector at the fitting
frequencies and averages marker errors across frequencies. All ordinary
studies fit the eight nominal frequencies 0.50, 0.75, 1.00, 1.25, 1.50,
1.75, 2.00 and 2.25 Hz. Errors are in millimetres. An unsuccessful
single-frequency simulation receives 1000 mm. An evaluation is valid only
when all its frequency simulations are valid.

Each run has a budget of 40 objective evaluations, including initialization,
line-search candidates and unsuccessful evaluations. A valid strict improvement
updates the best-so-far error. AUC40 is the mean of the first 40 best-so-far
values, including evaluations that did not improve it. SPSA may leave one
structurally unused evaluation; its last best value carries forward once.
An incomplete run is never padded into a completed trajectory.

The complete method uses a VLM proposal followed by first-improvement
backtracking at steps 1, 0.5 and 0.25. Random directions share the line search
without a model. Signed-permuted directions retain proposal scale while
reassigning normalized parameter displacements. Direct updates evaluate a
new model proposal on every call after initialization. The other descriptive
optimizers are Bayesian optimization, SPSA with line search and CMA-ES with
population sizes 6 and 12.

| Study | Scheduled seeds | Run histories | Analysis population |
| --- | --- | ---: | --- |
| `fifteen_seed` | 1233 through 1247 | 120 | Fifteen paired seeds for the two primary comparisons; fifteen completed runs for each of eight descriptive methods |
| `replication` | 1248 through 1297 | 150 | Per-method summaries use 44, 48 and 45 completed runs. Paired tests use 41 jointly completed seeds plus the shared failure at seed 1270 |
| `glm` | Twenty-four seeds listed in its manifest | 48 | Complete and permuted GLM-5.3-Flash runs, paired with replication random runs on the same seeds |
| `interpolation` | 1200 through 1207 | 0 | Thirty-two terminal assessments, eight per method; training histories are not included |

Replication seed 1279 has an infeasible shared initialization and is excluded.
Seed 1270 failed identically at initialization in all three methods and
contributes a constant 1000 mm trajectory to each primary comparison. Its
paired difference is zero. All other incomplete runs are excluded without
imputation. Per-method counts exclude both initialization cases. Final-error
and marker diagnostics use only the 41 jointly completed seeds.

The original direct-update study had fourteen completed runs. The released
fifteen-run descriptive set uses the completed seed 1236 run with a larger
model output-token limit, as used for the paper's direct-update row.

GLM-5.3-Flash uses 16 sampled frames per clip. Its random-direction runs are
referenced from the replication rather than stored a second time. Gemini
2.5 Pro supplies the model proposals for the other run-history studies.

Interpolation fits 0.50, 1.00, 1.50, 2.00 and 2.25 Hz, then assesses the
training-best parameters once at 0.75, 1.25 and 1.75 Hz. The all-frequency
reference fits all eight conditions. The recordings were used during
development. Only terminal assessment evidence is included here, so training
AUC40, training failure counts and training selection cannot be independently
recomputed for this study. All printed interpolation outcomes can be
recomputed from its per-frequency assessment errors.

## Layout and fields

All JSON uses schema version 1. Study and method identifiers are stable.

```text
schema.json
studies/{fifteen_seed,replication,glm,interpolation}.json
runs/{study}/{method}/seed_{seed}.json
assessments/interpolation.json
replays/backtracking.json
README.md
LICENSE
```

`schema.json`:

| Field | Meaning |
| --- | --- |
| `schema_version` | Integer format version |
| `parameter_keys` | Order of both sixteen-coordinate parameter vectors |
| `units` | Error, frequency and parameter unit conventions |

The parameter order is five `fluidCoef_0` through `fluidCoef_4` values,
`motorArmLength`, five `hingeStiffness_0` through `hingeStiffness_4` values,
then five `hingeDamping_0` through `hingeDamping_4` values. Raw values use
simulator units; normalized values express each coordinate relative to its
search interval in the calibration code. These vectors are not uniquely
identified material measurements.

Each study manifest:

| Field | Meaning |
| --- | --- |
| `schema_version`, `study` | Format version and study identity |
| `scheduled_seeds`, `methods` | Full scheduled matrix, in analysis order |
| `population_rule` | Plain-language population and exclusion rules |
| `completion_counts` | Completed runs per method, or available terminal assessments for interpolation |
| `shared_failure_seed`, `shared_failure_penalty_mm` | The single declared common-failure exception, otherwise null |
| `excluded_initialization_seed` | The infeasible shared initialization, otherwise null |
| `reused_methods` | Method-to-source-study mapping; the GLM random method refers to `replication` |
| `training_histories_included` | Whether run histories are present for this study |

Each run record:

| Field | Meaning |
| --- | --- |
| `schema_version`, `study`, `method`, `seed` | Format and run identity |
| `budget` | Number of allowed objective evaluations, 40 |
| `completion.complete` | Whether the run finished its budget, allowing the declared SPSA tail |
| `completion.reason` | `completed`, `invalid_model_reply`, `request_failure`, `shared_initialization_failure` or `infeasible_initialization` |
| `structural_tail_carry` | Zero, or one structurally unused SPSA evaluation |
| `evaluations` | Every budget-counted evaluation in chronological order, including invalid calls and initialization |
| `model_proposals` | Model-proposed rounds associated with the stored search trajectory; empty for model-free methods |

Each `evaluations` entry:

| Field | Meaning |
| --- | --- |
| `call_index` | One-based contiguous objective-evaluation index |
| `round_index` | Zero-based proposal or optimizer round; -1 for initialization or an ungrouped optimizer candidate |
| `phase` | Initial evaluation, line-search step, SPSA probe, direct proposal or optimizer generation |
| `step_size` | Line-search multiplier, or 1 for a direct proposal; null for initialization, optimizer candidates and SPSA probes |
| `parameters`, `normalized_parameters` | Raw and normalized sixteen-coordinate evaluated parameter vectors |
| `frequency_errors` | One entry per evaluated frequency |
| `mean_error_mm` | Mean per-frequency marker error, including penalties |
| `valid` | All per-frequency simulations succeeded |
| `accepted` | Valid strict improvement over the previous best; true for a valid initialization |
| `best_so_far_mm` | Minimum valid error through this call, or null if no valid call exists |

Each `frequency_errors` entry, in evaluations, assessments and replays:

| Field | Meaning |
| --- | --- |
| `frequency_hz` | Nominal actuation frequency |
| `score_mm` | Mean nine-marker error, or the failed-frequency penalty |
| `valid` | Whether this frequency simulation succeeded |
| `per_marker_mm` | Optional nine-marker time-mean errors, M0 through M8, at the selected best call for marker diagnostics |

Per-marker values are exported for the complete method's fifteen runs and
for all three methods on the jointly completed replication seeds. The
exporter checks these values against the aligned real and simulated marker
trajectories and checks their means against the recorded objective.

Each `model_proposals` entry:

| Field | Meaning |
| --- | --- |
| `round_index` | Proposal-round index, shared by all steps along that proposal |
| `model_id` | Model identifier returned with the proposal |
| `reply_text` | Verbatim model reply selected for the recorded search round |
| `parsed_proposal` | Parsed `diagnosis`, `selected_parameters`, `proposed_values`, `reasoning` and `confidence`, when present |
| `token_counts` | Recorded integer token counts, using the model response's token-counter names |

Selected replies are stored once per proposal round. Separate unsuccessful
parse or request attempts are not model-proposed search rounds and are not
included. Completion reasons describe terminal failures. Model replies are
generated explanations, not validated physical measurements.

Each interpolation assessment:

| Field | Meaning |
| --- | --- |
| `method`, `seed` | Assessment identity |
| `kind` | `heldout` or `in_fit_reference` |
| `parameters` | Assessed sixteen-coordinate raw parameter vector |
| `source_best_call_index` | Training call selected before assessment |
| `frequency_errors`, `mean_error_mm`, `valid` | Three per-frequency results, their mean and joint validity |

Each fixed-incumbent replay entry:

| Field | Meaning |
| --- | --- |
| `seed`, `round_index` | Recorded complete-method proposal being replayed |
| `incumbent_error_mm` | Incumbent error before that proposal |
| `evaluations` | Results at the union of step sizes needed for nine schedules |
| `evaluations[].step_size` | Multiplier along the fixed recorded direction |
| `evaluations[].mean_error_mm`, `valid`, `frequency_errors` | Evaluation error, joint validity and per-frequency results |

Replays use beta in {0.25, 0.5, 0.75} and two, three or four steps, accepting
the first valid improvement at each fixed incumbent. They do not propagate
newly accepted parameters into later model proposals.

## Statistical conventions and coverage

The paired sign is complete method minus comparator, so negative favors the
complete method. Exact two-sided Wilcoxon signed-rank tests discard exact
zeros and assign average ranks to tied magnitudes. Holm adjustment covers
the two primary comparisons within each of the fifteen-seed, replication
and GLM studies. BCa intervals use 10,000 paired bootstrap resamples, seed
20260812, midrank bias correction and linear NumPy quantiles. Interpolation
and final-error diagnostics are descriptive. All SDs are sample SDs.

`records/source_checks.json` in the code repository independently records
571 expected values read from the original analyzers' summaries. The full
test checks these as well as all reference scalars and 192 printed
values. A six-run real-data fixture permits fast offline tests without the
full archive; partial reproduction requires `--allow-partial` and emits
only per-run values, never an estimate from an incomplete study population.

The measured realized 1.121 Hz frequency and hardware deployment results
require separate recording timebases or hardware trajectories. They are
listed explicitly in the paper-number report. These records describe
optimizer variation on the same fish and recordings. They do not establish
unique parameter identification or accuracy on new recordings.
