"""Budget-counted calibration with deterministic resume."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from swim2real.config import RunConfig
from swim2real.evaluator import ObjectiveCall, ObjectiveEvaluator
from swim2real.parameters import Bounds16D, initial_vector_for_seed
from swim2real.providers.base import ExternalProposal, ProposalProvider
from swim2real.search.baselines import build_cmaes, build_skopt_bo
from swim2real.search.protocol import (
    BacktrackingCandidate,
    SPSAProbePlan,
    derive_spsa_proposal,
    make_rng,
    plan_backtracking_candidates,
    plan_spsa_probes,
    propose_multiscale_random,
    select_first_improvement,
    select_spsa_successor,
    signed_permute_displacement,
)
from swim2real.search.store import RunStore, RunStoreError, atomic_write_json

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class RunnerInvariantError(RuntimeError):
    """A cell cannot satisfy the configured method, budget, or resume invariants."""


def _vector(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    point = _finite_vector(value, name=name)
    if np.any(point < 0.0) or np.any(point > 1.0):
        raise RunnerInvariantError(f"{name} lies outside [0,1]")
    return point


def _finite_vector(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (16,) or not np.all(np.isfinite(point)):
        raise RunnerInvariantError(f"{name} must be one finite 16D vector")
    return point.copy()


def _same_point(first: ArrayLike, second: ArrayLike) -> bool:
    return bool(
        np.allclose(
            np.asarray(first, dtype=np.float64),
            np.asarray(second, dtype=np.float64),
            rtol=0.0,
            atol=1e-14,
        )
    )


def _proposal_diagnostics(
    anchor: ArrayLike,
    proposal: ArrayLike,
    preclip: ArrayLike,
) -> dict[str, Any]:
    anchor_point = _vector(anchor, name="proposal diagnostic anchor")
    proposal_point = _vector(proposal, name="proposal diagnostic proposal")
    preclip_point = _finite_vector(preclip, name="proposal diagnostic preclip")
    clipping = preclip_point - proposal_point
    return {
        "clip_norm": float(np.linalg.norm(clipping)),
        "clipped_coordinate_count": int(np.count_nonzero(np.abs(clipping) > 1e-14)),
        "preclip_proposal_normalized": preclip_point.tolist(),
        "proposal_norm": float(np.linalg.norm(proposal_point - anchor_point)),
    }


class CalibrationRunner:
    """Run or resume exactly one recorded method/problem-seed cell."""

    def __init__(
        self,
        *,
        method: str,
        problem_seed: int,
        budget: int,
        output_dir: str | Path,
        evaluator: ObjectiveEvaluator,
        code_manifest: Mapping[str, Any],
        config_sha256: str,
        proposal_provider: ProposalProvider | None = None,
        resume: bool = False,
        config: RunConfig | None = None,
    ) -> None:
        if method not in (config.method_codes if config else RunConfig.defaults().method_codes):
            raise RunnerInvariantError(f"unsupported method: {method}")
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise RunnerInvariantError("budget must be a positive integer")
        if (
            not isinstance(problem_seed, int)
            or isinstance(problem_seed, bool)
            or not 0 <= problem_seed <= np.iinfo(np.uint32).max
        ):
            raise RunnerInvariantError("problem_seed must be an integer")
        if not isinstance(config_sha256, str) or not _SHA256_PATTERN.fullmatch(config_sha256):
            raise RunnerInvariantError("config_sha256 must be 64 lowercase hexadecimal characters")
        if (
            method in {"full_vlm_ls", "signed_permuted_vlm_ls", "vlm_direct"}
            and proposal_provider is None
        ):
            raise RunnerInvariantError(f"{method} requires a validated proposal_provider")
        if (
            method not in {"full_vlm_ls", "signed_permuted_vlm_ls", "vlm_direct"}
            and proposal_provider is not None
        ):
            raise RunnerInvariantError(f"{method} does not accept a proposal_provider")
        if not isinstance(resume, bool):
            raise RunnerInvariantError("resume must be a boolean")

        self.config = config or RunConfig.from_dict({"budget": budget, "method": method})
        self.method = method
        self.problem_seed = problem_seed
        self.budget = budget
        self.output_dir = Path(output_dir)
        self.evaluator = evaluator
        self.code_manifest = dict(code_manifest)
        try:
            canonical_code = json.dumps(
                self.code_manifest,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise RunnerInvariantError("code_manifest must be strict JSON-serializable") from exc
        self.code_sha256 = hashlib.sha256(canonical_code).hexdigest()
        self.config_sha256 = config_sha256
        self.proposal_provider = proposal_provider
        self.resume = resume
        self.bounds = Bounds16D.from_submitted_bounds()
        self.rng = make_rng(
            self.problem_seed, self.config.method_codes[method], self.config.rng_namespace
        )
        if method in {"bayesopt", "cmaes_pop12", "cmaes_pop6"}:
            self.algorithm_seed = int(self.rng.integers(1, np.iinfo(np.int32).max))
        else:
            self.algorithm_seed = None
        self.initial = initial_vector_for_seed(self.problem_seed, self.bounds)
        self.incumbent = self.initial.copy()
        self.incumbent_loss = float("inf")
        self.best_call_index: int | None = None
        self.store: RunStore | None = None

    def run(self) -> dict[str, Any]:
        """Run to the configured budget or resume an incomplete append-only cell."""

        manifest = self._manifest()
        if self.resume:
            try:
                self.store = RunStore.resume(self.output_dir, manifest)
            except RunStoreError as exc:
                raise RunnerInvariantError(str(exc)) from exc
            if self.evaluator.call_count != len(self.store.calls):
                raise RunnerInvariantError(
                    "resume evaluator call_count must equal stored calls: "
                    f"{self.evaluator.call_count} != {len(self.store.calls)}"
                )
            try:
                self.evaluator.validate_stored_calls(self.store.calls)
            except ValueError as exc:
                raise RunnerInvariantError(str(exc)) from exc
            if self.store.rng_state is not None:
                self.rng.bit_generator.state = self.store.rng_state
            self._restore_global_best()
        else:
            if self.evaluator.call_count != 0:
                raise RunnerInvariantError("a new cell requires an unused ObjectiveEvaluator")
            self.store = RunStore.create(self.output_dir, manifest)

        config_path = self.output_dir / "config.json"
        effective = self.config.to_dict()
        if config_path.exists():
            if json.loads(config_path.read_text()) != effective:
                raise RunnerInvariantError("effective configuration differs on resume")
        else:
            atomic_write_json(config_path, effective)

        if not self.store.calls:
            initial = self._evaluate(self.initial, phase="initial", round_index=-1)
            if not initial.valid:
                raise RunnerInvariantError(
                    "shared initial point is invalid; cell is a technical failure"
                )
        else:
            first = self.store.calls[0]
            if first.get("phase") != "initial" or not _same_point(
                first.get("normalized_params"), self.initial
            ):
                raise RunnerInvariantError("stored initial call does not match recorded manifest")
            if not first.get("valid"):
                raise RunnerInvariantError(
                    "stored shared initial point is invalid; cell is a technical failure"
                )

        if self.method == "multiscale_random_ls":
            self._run_random_ls()
        elif self.method in {"full_vlm_ls", "signed_permuted_vlm_ls"}:
            self._run_vlm_ls()
        elif self.method == "vlm_direct":
            self._run_vlm_direct()
        elif self.method == "spsa_ls":
            self._run_spsa()
        elif self.method == "bayesopt":
            self._run_bayesopt()
        elif self.method in {"cmaes_pop12", "cmaes_pop6"}:
            self._run_cmaes(12 if self.method.endswith("12") else 6)
        else:  # pragma: no cover - constructor guards this
            raise AssertionError("unreachable method")

        tail = self._remaining
        if tail not in (0, 1) or (tail and self.method != "spsa_ls"):
            raise RunnerInvariantError(f"method ended with forbidden structural tail of {tail}")
        summary = {
            "best_call_index": self.best_call_index,
            "best_params": self.bounds.to_flat(self.incumbent),
            "best_score_mm": self.incumbent_loss,
            "structural_tail_carry": tail,
        }
        self.store.finalize(summary)
        return json.loads((self.output_dir / "result.json").read_text(encoding="utf-8"))

    def _manifest(self) -> dict[str, Any]:
        return {
            "algorithm_seed": self.algorithm_seed,
            "budget": self.budget,
            "code": self.code_manifest,
            "code_sha256": self.code_sha256,
            "config_sha256": self.config_sha256,
            "initial_normalized": self.initial.tolist(),
            "method": self.method,
            "method_code": self.config.method_codes[self.method],
            "problem_seed": self.problem_seed,
            "schema_version": 1,
        }

    @property
    def _remaining(self) -> int:
        assert self.store is not None
        return self.budget - len(self.store.calls)

    def _algorithm_state(self) -> dict[str, Any]:
        return {
            "best_call_index": self.best_call_index,
            "incumbent_loss": (self.incumbent_loss if math.isfinite(self.incumbent_loss) else None),
            "incumbent_normalized": self.incumbent.tolist(),
        }

    def _restore_global_best(self) -> None:
        """Recompute the global best from authoritative append-only call records."""

        assert self.store is not None
        self.incumbent = self.initial.copy()
        self.incumbent_loss = float("inf")
        self.best_call_index = None
        for call in self.store.calls:
            if not call.get("incumbent_eligible"):
                continue
            score = float(call["score_mm"])
            if not math.isfinite(score) or score >= self.incumbent_loss:
                continue
            point = _vector(call["normalized_params"], name="stored normalized_params")
            self.incumbent = point
            self.incumbent_loss = score
            self.best_call_index = int(call["call_index"])
        if self.store.calls and self.best_call_index is None:
            raise RunnerInvariantError("stored cell has no valid incumbent")
        if self.store.calls:
            self.store.checkpoint_algorithm_state(self._algorithm_state())

    def _evaluate(
        self,
        point: ArrayLike,
        *,
        phase: str,
        round_index: int,
        metadata: Mapping[str, Any] | None = None,
        metadata_from_call: Callable[[ObjectiveCall], Mapping[str, Any]] | None = None,
    ) -> ObjectiveCall:
        assert self.store is not None
        normalized = _vector(point, name="candidate")
        call = self.evaluator.evaluate(self.bounds.to_flat(normalized))
        if math.isfinite(self.incumbent_loss):
            selection = select_spsa_successor(
                self.incumbent,
                self.incumbent_loss,
                [normalized],
                [call.score_mm if call.incumbent_eligible else None],
            )
            improved = selection.improved
        else:
            improved = call.incumbent_eligible
        if improved:
            self.incumbent = normalized.copy()
            self.incumbent_loss = call.score_mm
            self.best_call_index = call.call_index
        payload = call.to_dict()
        payload.update(
            {
                "normalized_params": normalized.tolist(),
                "phase": phase,
                "round_index": round_index,
            }
        )
        if metadata is not None or metadata_from_call is not None:
            resolved = dict(metadata or {})
            if metadata_from_call is not None:
                resolved.update(metadata_from_call(call))
            payload["proposal_metadata"] = resolved
        self.store.append_call(
            payload,
            self.rng.bit_generator.state,
            self._algorithm_state(),
        )
        return call

    def _round_calls(self, round_index: int) -> list[dict[str, Any]]:
        assert self.store is not None
        return [
            call for call in self.store.calls if int(call.get("round_index", -1)) == round_index
        ]

    def _max_round(self) -> int:
        assert self.store is not None
        rounds = [
            int(call["round_index"])
            for call in self.store.calls
            if int(call.get("round_index", -1)) >= 0
        ]
        return max(rounds, default=-1)

    def _backtracking_metadata(
        self,
        *,
        anchor: ArrayLike,
        anchor_loss: float,
        proposal: ArrayLike,
        max_calls: int,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        anchor_point = _vector(anchor, name="backtracking anchor")
        proposal_point = _vector(proposal, name="backtracking proposal")
        candidates = plan_backtracking_candidates(
            anchor_point,
            proposal_point,
            min(self._remaining, max_calls),
            self.config.backtracking_alphas,
        )
        metadata = dict(extra or {})
        reserved = {
            "anchor_loss_mm",
            "anchor_normalized",
            "planned_candidates",
            "proposal_normalized",
            "protocol",
        }
        if reserved.intersection(metadata):
            raise RunnerInvariantError("proposal details collides with protocol metadata")
        preclip = metadata.pop("preclip_proposal_normalized", proposal_point)
        metadata.update(_proposal_diagnostics(anchor_point, proposal_point, preclip))
        metadata.update(
            {
                "anchor_loss_mm": float(anchor_loss),
                "anchor_normalized": anchor_point.tolist(),
                "planned_candidates": [
                    {"alpha": candidate.alpha, "params": candidate.params.tolist()}
                    for candidate in candidates
                ],
                "proposal_normalized": proposal_point.tolist(),
                "protocol": "first_improvement_backtracking_v1",
            }
        )
        return metadata

    def _validate_round_metadata(
        self,
        calls: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not calls:
            raise RunnerInvariantError("cannot recover metadata from an empty round")
        metadata = calls[0].get("proposal_metadata")
        if not isinstance(metadata, Mapping):
            raise RunnerInvariantError("stored round lacks proposal reconstruction metadata")
        canonical = json.loads(json.dumps(metadata, allow_nan=False, sort_keys=True))
        for call in calls[1:]:
            other = call.get("proposal_metadata")
            if isinstance(other, Mapping) and other.get("protocol") == canonical.get("protocol"):
                for key in (
                    "anchor_loss_mm",
                    "anchor_normalized",
                    "planned_candidates",
                    "proposal_normalized",
                ):
                    if key in other and other[key] != canonical.get(key):
                        raise RunnerInvariantError(f"stored round metadata disagrees on {key}")
        return canonical

    @staticmethod
    def _backtracking_complete(
        calls: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]
    ) -> bool:
        anchor_loss = float(metadata["anchor_loss_mm"])
        selection = select_first_improvement(
            metadata["anchor_normalized"],
            anchor_loss,
            [
                BacktrackingCandidate(float(row["alpha"]), np.asarray(row["params"]))
                for row in metadata["planned_candidates"][: len(calls)]
            ],
            [float(call["score_mm"]) if call.get("incumbent_eligible") else None for call in calls],
        )
        if selection.improved:
            return True
        return len(calls) >= len(metadata["planned_candidates"])

    def _continue_backtracking(
        self,
        *,
        round_index: int,
        phase_prefix: str,
        metadata: Mapping[str, Any],
        existing_calls: Sequence[Mapping[str, Any]],
    ) -> None:
        planned = list(metadata["planned_candidates"])
        if len(existing_calls) > len(planned):
            raise RunnerInvariantError("stored round contains more calls than its configured plan")
        for index, call in enumerate(existing_calls):
            expected = planned[index]
            if not _same_point(call["normalized_params"], expected["params"]):
                raise RunnerInvariantError(
                    "stored backtracking call differs from configured candidate"
                )
            expected_phase = f"{phase_prefix}_alpha_{float(expected['alpha']):g}"
            if call.get("phase") != expected_phase:
                raise RunnerInvariantError("stored backtracking phase order is inconsistent")
        if self._backtracking_complete(existing_calls, metadata):
            return
        anchor_loss = float(metadata["anchor_loss_mm"])
        for candidate in planned[len(existing_calls) :]:
            if not self._remaining:
                break
            point = _vector(candidate["params"], name="stored planned candidate")
            call = self._evaluate(
                point,
                phase=f"{phase_prefix}_alpha_{float(candidate['alpha']):g}",
                round_index=round_index,
                metadata=metadata,
            )
            if select_first_improvement(
                metadata["anchor_normalized"],
                anchor_loss,
                [BacktrackingCandidate(float(candidate["alpha"]), point)],
                [call.score_mm if call.incumbent_eligible else None],
            ).improved:
                break

    def _resume_latest_backtracking(self, phase_prefix: str) -> int:
        latest = self._max_round()
        if latest < 0:
            return 0
        calls = self._round_calls(latest)
        metadata = self._validate_round_metadata(calls)
        if metadata.get("protocol") != "first_improvement_backtracking_v1":
            raise RunnerInvariantError("stored latest round is not a backtracking round")
        self._continue_backtracking(
            round_index=latest,
            phase_prefix=phase_prefix,
            metadata=metadata,
            existing_calls=calls,
        )
        return latest + 1

    def _run_random_ls(self) -> None:
        round_index = self._resume_latest_backtracking("random_ls")
        while self._remaining:
            anchor = self.incumbent.copy()
            anchor_loss = self.incumbent_loss
            proposal = propose_multiscale_random(
                anchor, self.rng, self.config.random_radii_normalized
            )
            metadata = self._backtracking_metadata(
                anchor=anchor,
                anchor_loss=anchor_loss,
                proposal=proposal.params,
                max_calls=len(self.config.backtracking_alphas),
                extra={
                    "preclip_norm": float(np.linalg.norm(proposal.preclip_params - anchor)),
                    "preclip_proposal_normalized": proposal.preclip_params.tolist(),
                    "radius": proposal.radius,
                    "unit_direction": proposal.direction.tolist(),
                },
            )
            self._continue_backtracking(
                round_index=round_index,
                phase_prefix="random_ls",
                metadata=metadata,
                existing_calls=[],
            )
            round_index += 1

    def _run_vlm_ls(self) -> None:
        assert self.proposal_provider is not None
        assert self.store is not None
        phase_prefix = self.method
        round_index = self._resume_latest_backtracking(phase_prefix)
        while self._remaining:
            anchor = self.incumbent.copy()
            anchor_loss = self.incumbent_loss
            external = self.proposal_provider(
                anchor.copy(),
                anchor_loss,
                tuple(self.store.calls),
                round_index,
            )
            external_proposal = np.asarray(external.params, dtype=np.float64)
            if external_proposal.shape != (16,) or not np.all(np.isfinite(external_proposal)):
                raise RunnerInvariantError("proposal_provider returned an invalid 16D point")
            external_preclip = external_proposal.copy()
            external_proposal = np.clip(external_proposal, 0.0, 1.0)
            proposal = external_proposal.copy()
            proposal_preclip = external_preclip.copy()
            extra = dict(external.metadata)
            extra["external_proposal_normalized"] = external_proposal.tolist()
            if self.method == "signed_permuted_vlm_ls":
                displacement = external_proposal - anchor
                randomized = signed_permute_displacement(displacement, self.rng)
                proposal_preclip = anchor + randomized
                proposal = np.clip(proposal_preclip, 0.0, 1.0)
                extra["signed_permuted_displacement"] = randomized.tolist()
            extra["preclip_proposal_normalized"] = proposal_preclip.tolist()
            metadata = self._backtracking_metadata(
                anchor=anchor,
                anchor_loss=anchor_loss,
                proposal=proposal,
                max_calls=len(self.config.backtracking_alphas),
                extra=extra,
            )
            self._continue_backtracking(
                round_index=round_index,
                phase_prefix=phase_prefix,
                metadata=metadata,
                existing_calls=[],
            )
            round_index += 1

    def _run_vlm_direct(self) -> None:
        """Evaluate one VLM proposal per round."""

        assert self.proposal_provider is not None
        assert self.store is not None
        round_index = self._max_round() + 1
        while self._remaining:
            anchor = self.incumbent.copy()
            anchor_loss = self.incumbent_loss
            external = self.proposal_provider(
                anchor.copy(),
                anchor_loss,
                tuple(self.store.calls),
                round_index,
            )
            external_proposal = np.asarray(external.params, dtype=np.float64)
            if external_proposal.shape != (16,) or not np.all(np.isfinite(external_proposal)):
                raise RunnerInvariantError("proposal_provider returned an invalid 16D point")
            proposal_preclip = external_proposal.copy()
            proposal = np.clip(external_proposal, 0.0, 1.0)
            metadata = dict(external.metadata)
            reserved = {
                "anchor_loss_mm",
                "anchor_normalized",
                "proposal_normalized",
                "protocol",
                "vlm_direct_accepted",
            }
            if reserved.intersection(metadata):
                raise RunnerInvariantError("proposal details collides with protocol metadata")
            metadata["external_proposal_normalized"] = proposal.tolist()
            metadata.update(_proposal_diagnostics(anchor, proposal, proposal_preclip))
            metadata.update(
                {
                    "anchor_loss_mm": float(anchor_loss),
                    "anchor_normalized": anchor.tolist(),
                    "proposal_normalized": proposal.tolist(),
                    "protocol": "single_candidate_no_line_search_v1",
                }
            )
            self._evaluate(
                proposal,
                phase="vlm_direct",
                round_index=round_index,
                metadata=metadata,
                metadata_from_call=lambda call, anchor_loss=anchor_loss: {
                    "vlm_direct_accepted": bool(
                        call.incumbent_eligible and call.score_mm < anchor_loss
                    )
                },
            )
            round_index += 1

    @staticmethod
    def _serialize_spsa_plan(plan: SPSAProbePlan) -> dict[str, Any]:
        if not plan.has_probes:
            raise RunnerInvariantError("cannot serialize an empty SPSA probe plan")
        assert plan.delta is not None
        assert plan.h is not None
        assert plan.plus is not None
        assert plan.minus is not None
        return {
            "delta": plan.delta.tolist(),
            "h": plan.h.tolist(),
            "line_search_budget": plan.line_search_budget,
            "minus": plan.minus.tolist(),
            "plus": plan.plus.tolist(),
            "unused_budget": plan.unused_budget,
        }

    @staticmethod
    def _deserialize_spsa_plan(metadata: Mapping[str, Any]) -> SPSAProbePlan:
        try:
            plan = metadata["spsa_plan"]
            return SPSAProbePlan(
                incumbent=_vector(metadata["anchor_normalized"], name="SPSA anchor"),
                delta=_finite_vector(plan["delta"], name="SPSA delta"),
                h=_finite_vector(plan["h"], name="SPSA h"),
                plus=_vector(plan["plus"], name="SPSA plus"),
                minus=_vector(plan["minus"], name="SPSA minus"),
                line_search_budget=int(plan["line_search_budget"]),
                unused_budget=int(plan["unused_budget"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RunnerInvariantError("stored SPSA plan metadata is invalid") from exc

    def _new_spsa_metadata(
        self, anchor: NDArray[np.float64], anchor_loss: float, plan: SPSAProbePlan
    ) -> dict[str, Any]:
        return {
            "anchor_loss_mm": float(anchor_loss),
            "anchor_normalized": anchor.tolist(),
            "proposal_normalized": None,
            "protocol": "spsa_backtracking_v1",
            "spsa_plan": self._serialize_spsa_plan(plan),
        }

    def _spsa_complete(
        self,
        calls: Sequence[Mapping[str, Any]],
        metadata: Mapping[str, Any],
    ) -> bool:
        if len(calls) < 2:
            return False
        plan = self._deserialize_spsa_plan(metadata)
        plus, minus = calls[:2]
        proposal = derive_spsa_proposal(
            plan,
            float(plus["score_mm"]) if plus.get("valid") else None,
            float(minus["score_mm"]) if minus.get("valid") else None,
            self.config.spsa.step_radius_normalized,
        )
        if not proposal.valid or plan.line_search_budget == 0:
            return True
        line_calls = calls[2:]
        anchor_loss = float(metadata["anchor_loss_mm"])
        candidates = plan_backtracking_candidates(
            plan.incumbent,
            proposal.params,
            plan.line_search_budget,
            self.config.backtracking_alphas,
        )
        return (
            len(line_calls) >= plan.line_search_budget
            or select_first_improvement(
                plan.incumbent,
                anchor_loss,
                candidates[: len(line_calls)],
                [
                    float(call["score_mm"]) if call.get("incumbent_eligible") else None
                    for call in line_calls
                ],
            ).improved
        )

    def _continue_spsa_round(
        self,
        round_index: int,
        metadata: Mapping[str, Any],
        existing_calls: Sequence[Mapping[str, Any]],
    ) -> None:
        plan = self._deserialize_spsa_plan(metadata)
        calls = list(existing_calls)
        if len(calls) > 2 + plan.line_search_budget:
            raise RunnerInvariantError("stored SPSA round exceeds its configured plan")
        expected_probe_phases = ("spsa_plus", "spsa_minus")
        expected_probe_points = (plan.plus, plan.minus)
        for index in range(min(2, len(calls))):
            if calls[index].get("phase") != expected_probe_phases[index] or not _same_point(
                calls[index]["normalized_params"], expected_probe_points[index]
            ):
                raise RunnerInvariantError("stored SPSA probe order is inconsistent")

        if not calls:
            self._evaluate(
                plan.plus,
                phase="spsa_plus",
                round_index=round_index,
                metadata=metadata,
            )
            calls.append(self.store.calls[-1])
        if len(calls) == 1:
            self._evaluate(
                plan.minus,
                phase="spsa_minus",
                round_index=round_index,
                metadata=metadata,
            )
            calls.append(self.store.calls[-1])

        plus, minus = calls[:2]
        proposal = derive_spsa_proposal(
            plan,
            float(plus["score_mm"]) if plus.get("valid") else None,
            float(minus["score_mm"]) if minus.get("valid") else None,
            self.config.spsa.step_radius_normalized,
        )
        if not proposal.valid or proposal.params is None or plan.line_search_budget == 0:
            return

        ls_metadata = dict(metadata)
        ls_metadata["proposal_normalized"] = proposal.params.tolist()
        ls_metadata["preclip_proposal_normalized"] = proposal.preclip_params.tolist()
        ls_metadata.update(
            _proposal_diagnostics(
                plan.incumbent,
                proposal.params,
                proposal.preclip_params,
            )
        )
        ls_metadata["spsa_gradient"] = proposal.gradient.tolist()
        candidates = plan_backtracking_candidates(
            plan.incumbent,
            proposal.params,
            plan.line_search_budget,
            self.config.backtracking_alphas,
        )
        ls_metadata["planned_candidates"] = [
            {"alpha": candidate.alpha, "params": candidate.params.tolist()}
            for candidate in candidates
        ]
        line_calls = calls[2:]
        anchor_loss = float(metadata["anchor_loss_mm"])
        for index, call in enumerate(line_calls):
            expected = candidates[index]
            if call.get("phase") != f"spsa_ls_alpha_{expected.alpha:g}" or not _same_point(
                call["normalized_params"], expected.params
            ):
                raise RunnerInvariantError("stored SPSA line-search order is inconsistent")
        if select_first_improvement(
            plan.incumbent,
            anchor_loss,
            candidates[: len(line_calls)],
            [
                float(call["score_mm"]) if call.get("incumbent_eligible") else None
                for call in line_calls
            ],
        ).improved:
            return
        for candidate in candidates[len(line_calls) :]:
            call = self._evaluate(
                candidate.params,
                phase=f"spsa_ls_alpha_{candidate.alpha:g}",
                round_index=round_index,
                metadata=ls_metadata,
            )
            if select_first_improvement(
                plan.incumbent,
                anchor_loss,
                [candidate],
                [call.score_mm if call.incumbent_eligible else None],
            ).improved:
                break

    def _run_spsa(self) -> None:
        latest = self._max_round()
        round_index = 0
        if latest >= 0:
            calls = self._round_calls(latest)
            metadata = calls[0].get("proposal_metadata") if calls else None
            if (
                not isinstance(metadata, Mapping)
                or metadata.get("protocol") != "spsa_backtracking_v1"
            ):
                raise RunnerInvariantError("stored latest round is not an SPSA round")
            if not self._spsa_complete(calls, metadata):
                self._continue_spsa_round(latest, metadata, calls)
            round_index = latest + 1

        while self._remaining >= 2:
            anchor = self.incumbent.copy()
            anchor_loss = self.incumbent_loss
            plan = plan_spsa_probes(
                anchor,
                self.rng,
                self._remaining,
                self.config.spsa.probe_radius_normalized,
                self.config.backtracking_alphas,
            )
            metadata = self._new_spsa_metadata(anchor, anchor_loss, plan)
            self._continue_spsa_round(round_index, metadata, [])
            round_index += 1

    def _run_bayesopt(self) -> None:
        assert self.store is not None
        assert self.algorithm_seed is not None
        optimizer = build_skopt_bo(self.algorithm_seed, self.config.bayesopt)
        initial_call = self.store.calls[0]
        optimizer.tell(self.initial.tolist(), float(initial_call["score_mm"]))
        stored = self.store.calls[1:]
        for call in stored:
            if call.get("phase") != "bayesopt":
                raise RunnerInvariantError("stored BO cell contains a non-BO call")
            expected = np.asarray(optimizer.ask(), dtype=np.float64)
            observed = _vector(call["normalized_params"], name="stored BO point")
            if not _same_point(expected, observed):
                raise RunnerInvariantError("stored BO point does not match deterministic replay")
            optimizer.tell(observed.tolist(), float(call["score_mm"]))

        while self._remaining:
            anchor = self.incumbent.copy()
            anchor_loss = self.incumbent_loss
            point = np.asarray(optimizer.ask(), dtype=np.float64)
            call = self._evaluate(
                point,
                phase="bayesopt",
                round_index=-1,
                metadata={
                    "anchor_loss_mm": anchor_loss,
                    "anchor_normalized": anchor.tolist(),
                    **_proposal_diagnostics(anchor, point, point),
                    "proposal_normalized": point.tolist(),
                    "protocol": "skopt_ask_tell_replay_v1",
                },
            )
            optimizer.tell(point.tolist(), call.score_mm)

    def _run_cmaes(self, population_size: int) -> None:
        assert self.store is not None
        assert self.algorithm_seed is not None
        optimizer = build_cmaes(
            self.initial,
            algorithm_seed=self.algorithm_seed,
            population_size=population_size,
            config=self.config.cmaes,
        )
        stored = self.store.calls[1:]
        cursor = 0
        generation = 0
        while cursor < len(stored) or self._remaining:
            raw_population = optimizer.ask()
            population = [
                np.clip(np.asarray(point, dtype=np.float64), 0.0, 1.0) for point in raw_population
            ]
            existing: list[dict[str, Any]] = []
            while cursor < len(stored) and stored[cursor].get("round_index") == generation:
                existing.append(stored[cursor])
                cursor += 1
            if cursor < len(stored) and not existing:
                raise RunnerInvariantError("stored CMA generations are not contiguous")
            if len(existing) > population_size:
                raise RunnerInvariantError("stored CMA generation exceeds population size")
            for index, call in enumerate(existing):
                if call.get("phase") != f"cmaes_generation_{generation}" or not _same_point(
                    call["normalized_params"], population[index]
                ):
                    raise RunnerInvariantError(
                        "stored CMA point does not match deterministic replay"
                    )

            target_count = min(population_size, len(existing) + self._remaining)
            evaluated = [
                _vector(call["normalized_params"], name="stored CMA point").tolist()
                for call in existing
            ]
            scores = [float(call["score_mm"]) for call in existing]
            for candidate_index in range(len(existing), target_count):
                anchor = self.incumbent.copy()
                anchor_loss = self.incumbent_loss
                raw_point = np.asarray(raw_population[candidate_index], dtype=np.float64)
                point = population[candidate_index]
                call = self._evaluate(
                    point,
                    phase=f"cmaes_generation_{generation}",
                    round_index=generation,
                    metadata={
                        "anchor_loss_mm": anchor_loss,
                        "anchor_normalized": anchor.tolist(),
                        "candidate_index": candidate_index,
                        "population_size": population_size,
                        **_proposal_diagnostics(anchor, point, raw_point),
                        "proposal_normalized": point.tolist(),
                        "protocol": "pycma_generation_replay_v1",
                    },
                )
                evaluated.append(point.tolist())
                scores.append(call.score_mm)
            if target_count == population_size:
                optimizer.tell(evaluated, scores)
            elif self._remaining:
                raise RunnerInvariantError("partial CMA population did not exhaust the budget")
            generation += 1
            if cursor >= len(stored) and not self._remaining:
                break


__all__ = [
    "CalibrationRunner",
    "ExternalProposal",
    "ProposalProvider",
    "RunnerInvariantError",
]
