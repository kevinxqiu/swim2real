"""Proposal interfaces and canonical prompt diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from swim2real.config import PAPER_DEFAULTS


class ProposalProviderError(ValueError):
    """The proposal or its diagnostic context is invalid."""


@dataclass(frozen=True)
class ExternalProposal:
    params: np.ndarray
    metadata: Mapping[str, Any]


class ProposalProvider(Protocol):
    def __call__(
        self,
        incumbent: np.ndarray,
        incumbent_loss: float,
        calls: Sequence[Mapping[str, Any]],
        round_index: int,
    ) -> ExternalProposal:
        """Propose the next normalized parameter vector."""
        ...


class ScriptedProvider:
    """Replay explicit normalized proposals for offline examples and tests."""

    def __init__(self, proposals: Sequence[Sequence[float]]) -> None:
        self.proposals = [np.asarray(p, dtype=np.float64) for p in proposals]
        if not self.proposals or any(
            p.shape != (16,) or not np.isfinite(p).all() for p in self.proposals
        ):
            raise ValueError("Scripted proposals must be finite 16-dimensional vectors")

    def __call__(
        self,
        incumbent: np.ndarray,
        incumbent_loss: float,
        calls: Sequence[Mapping[str, Any]],
        round_index: int,
    ) -> ExternalProposal:
        if round_index >= len(self.proposals):
            raise ValueError(f"Scripted provider has no proposal for round {round_index}")
        return ExternalProposal(self.proposals[round_index].copy(), {})


def _flat_key(flat: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((str(key), float(value)) for key, value in flat.items()))


def _flat_digest(flat: Mapping[str, float]) -> str:
    encoded = json.dumps(dict(_flat_key(flat)), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _velocity_from_markers(
    markers: Any, fps: float, head_markers: int = PAPER_DEFAULTS["metric"]["head_markers"]
) -> float:
    array = np.asarray(markers, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] != 9 or array.shape[2] != 2 or array.shape[1] < 2:
        raise ProposalProviderError("objective diagnostic markers have an invalid shape")
    if not np.all(np.isfinite(array)):
        raise ProposalProviderError("objective diagnostic markers must be finite")
    time_axis = np.arange(array.shape[1], dtype=np.float64) / float(fps)
    head_x = np.mean(array[:head_markers, :, 0], axis=0)
    return float(np.polyfit(time_axis, -head_x, 1)[0])


def _validated_detail(
    detail: Mapping[str, Any],
    *,
    fps: float,
    call_index: int,
    flat_sha256: str,
    frequency: float,
    head_markers: int = PAPER_DEFAULTS["metric"]["head_markers"],
) -> dict[str, Any]:
    if int(detail["schema_version"]) != 1:
        raise ProposalProviderError("cached objective diagnostic schema mismatch")
    if int(detail["call_index"]) != call_index:
        raise ProposalProviderError("cached objective diagnostic call-index mismatch")
    if str(detail["flat_sha256"]) != flat_sha256:
        raise ProposalProviderError("cached objective diagnostic parameter mismatch")
    if not math.isclose(float(detail["videofps"]), fps, rel_tol=0.0, abs_tol=1e-12):
        raise ProposalProviderError("cached objective diagnostic FPS mismatch")
    if not math.isclose(float(detail["frequency"]), frequency, rel_tol=0.0, abs_tol=1e-12):
        raise ProposalProviderError("cached objective diagnostic frequency mismatch")
    real_markers = np.asarray(detail["real_markers"], dtype=np.float64)
    sim_markers = np.asarray(detail["sim_markers"], dtype=np.float64)
    if real_markers.shape != sim_markers.shape:
        raise ProposalProviderError("objective diagnostic marker arrays differ in shape")
    if real_markers.ndim != 3 or real_markers.shape[0] != 9 or real_markers.shape[2] != 2:
        raise ProposalProviderError("objective diagnostic markers have an invalid shape")
    if not np.all(np.isfinite(real_markers)) or not np.all(np.isfinite(sim_markers)):
        raise ProposalProviderError("objective diagnostic markers must be finite")
    l2 = np.linalg.norm(sim_markers - real_markers, axis=2)
    per_marker_l2_mm = np.mean(l2, axis=1) * 1000.0
    mean_l2 = float(np.mean(l2))
    stored_per_marker = np.asarray(detail["per_marker_l2_mm"], dtype=np.float64)
    if stored_per_marker.shape != (9,) or not np.allclose(
        stored_per_marker,
        per_marker_l2_mm,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ProposalProviderError("cached objective diagnostic per-marker error mismatch")
    if not math.isclose(float(detail["mean_l2"]), mean_l2, rel_tol=0.0, abs_tol=1e-12):
        raise ProposalProviderError("cached objective diagnostic mean error mismatch")
    real_velocity = float(detail["real_velocity"])
    sim_velocity = float(detail["sim_velocity"])
    if not math.isfinite(real_velocity) or not math.isfinite(sim_velocity):
        raise ProposalProviderError("cached objective diagnostic velocities must be finite")
    result = dict(detail)
    result.update(
        {
            "call_index": call_index,
            "flat_sha256": flat_sha256,
            "frequency": frequency,
            "mean_l2": mean_l2,
            "per_marker_l2_mm": per_marker_l2_mm.tolist(),
            "real_markers": real_markers,
            "real_markers_sha256": _array_sha256(real_markers),
            "real_marker_velocity": _velocity_from_markers(real_markers, fps, head_markers),
            "real_velocity": real_velocity,
            "schema_version": 1,
            "sim_markers": sim_markers,
            "sim_markers_sha256": _array_sha256(sim_markers),
            "sim_marker_velocity": _velocity_from_markers(sim_markers, fps, head_markers),
            "sim_velocity": sim_velocity,
            "videofps": fps,
        }
    )
    binding = _detail_binding(result)
    binding_sha256 = _canonical_sha256(binding)
    stored_binding = detail.get("diagnostic_sha256")
    if stored_binding is not None and str(stored_binding) != binding_sha256:
        raise ProposalProviderError("cached objective diagnostic content hash mismatch")
    result["diagnostic_sha256"] = binding_sha256
    return result


def _detail_binding(detail: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "call_index": int(detail["call_index"]),
        "flat_sha256": str(detail["flat_sha256"]),
        "frequency": float(detail["frequency"]),
        "mean_l2": float(detail["mean_l2"]),
        "per_marker_l2_mm": [float(value) for value in detail["per_marker_l2_mm"]],
        "real_markers_sha256": str(detail["real_markers_sha256"]),
        "real_marker_velocity": float(detail["real_marker_velocity"]),
        "real_velocity": float(detail["real_velocity"]),
        "schema_version": int(detail["schema_version"]),
        "sim_markers_sha256": str(detail["sim_markers_sha256"]),
        "sim_marker_velocity": float(detail["sim_marker_velocity"]),
        "sim_velocity": float(detail["sim_velocity"]),
        "videofps": detail["videofps"],
    }


def _prompt_context_sha256(context: Mapping[str, Any]) -> str:
    unhashed = dict(context)
    unhashed.pop("treatment_sha256", None)
    return _canonical_sha256(unhashed)


def _build_prompt_context(
    *,
    current_flat: Mapping[str, float],
    history: Sequence[Mapping[str, Any]],
    incumbent_call_index: int,
    media_paths: Sequence[str],
    param_keys: Sequence[str],
    previously_selected: Sequence[Sequence[str]],
    round_index: int,
    skeleton_frequencies: Sequence[float],
    sweep: Sequence[Mapping[str, Any]],
    worst_frequency: float,
    fps: float,
    width: int = PAPER_DEFAULTS["media"]["width"],
    height: int = PAPER_DEFAULTS["media"]["height"],
) -> dict[str, Any]:
    from swim2real.media import media_manifest as _media_manifest

    diagnostic_bindings = [_detail_binding(entry) for entry in sweep]
    context: dict[str, Any] = {
        "current_flat": dict(current_flat),
        "diagnostics": diagnostic_bindings,
        "diagnostics_sha256": _canonical_sha256(diagnostic_bindings),
        "history": [dict(row) for row in history],
        "incumbent_call_index": incumbent_call_index,
        "media": _media_manifest(
            media_paths,
            sweep,
            worst_frequency,
            fps=fps,
            width=width,
            height=height,
        ),
        "param_keys": list(param_keys),
        "previously_selected": [list(values) for values in previously_selected],
        "round_index": int(round_index),
        "skeleton_frequencies": [float(value) for value in skeleton_frequencies],
        "sweep": [_prompt_diagnostic(entry) for entry in sweep],
        "worst_frequency": float(worst_frequency),
    }
    context["treatment_sha256"] = _prompt_context_sha256(context)
    return context


def _prompt_diagnostic(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "frequency": float(entry["frequency"]),
        "mean_l2": float(entry["mean_l2"]),
        "per_marker_l2_mm": [float(value) for value in entry.get("per_marker_l2_mm", [])],
        "real_velocity": float(entry["real_velocity"]),
        "sim_velocity": float(entry["sim_velocity"]),
    }


def _history_from_calls(
    calls: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[list[str]]]:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for call in calls:
        round_index = int(call.get("round_index", -1))
        metadata = call.get("proposal_metadata")
        if (
            round_index >= 0
            and isinstance(metadata, Mapping)
            and metadata.get("selected_parameters")
        ):
            grouped.setdefault(round_index, []).append(call)
    history: list[dict[str, Any]] = []
    selections: list[list[str]] = []
    for round_index in sorted(grouped):
        round_calls = grouped[round_index]
        metadata = dict(round_calls[0]["proposal_metadata"])
        selected = [str(value) for value in metadata["selected_parameters"]]
        before = float(metadata["sweep_before_mm"])
        valid_scores = [float(call["score_mm"]) for call in round_calls if call.get("valid")]
        after = min([before, *valid_scores])
        history.append(
            {
                "best_in_round_mm": min(valid_scores) if valid_scores else None,
                "improvement_mm": after - before,
                "round": round_index,
                "selected_parameters": selected,
                "sweep_after_mm": after,
                "sweep_before_mm": before,
            }
        )
        selections.append(selected)
    return history, selections


def _incumbent_call_index(
    calls: Sequence[Mapping[str, Any]],
    incumbent: np.ndarray,
    incumbent_loss: float,
) -> int:
    eligible = [call for call in calls if call.get("incumbent_eligible") is True]
    if not eligible:
        raise ProposalProviderError("VLM proposal has no chronological incumbent call")
    best = min(eligible, key=lambda call: (float(call["score_mm"]), int(call["call_index"])))
    if not math.isclose(float(best["score_mm"]), float(incumbent_loss), rel_tol=0.0, abs_tol=1e-12):
        raise ProposalProviderError("VLM incumbent loss differs from chronological call history")
    normalized = np.asarray(best.get("normalized_params"), dtype=float)
    if normalized.shape != (16,) or not np.allclose(normalized, incumbent, rtol=0.0, atol=1e-14):
        raise ProposalProviderError(
            "VLM incumbent parameters differ from chronological call history"
        )
    return int(best["call_index"])
