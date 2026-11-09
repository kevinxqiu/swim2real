"""Build a model proposal from the incumbent's complete diagnostic sweep."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from swim2real.config import RunConfig, read_json
from swim2real.media import build_media
from swim2real.parameters import Bounds16D
from swim2real.prompts.validation import build_prompt, parse_response
from swim2real.providers.base import (
    ExternalProposal,
    _build_prompt_context,
    _history_from_calls,
    _incumbent_call_index,
)
from swim2real.providers.client import CachedClient
from swim2real.search.store import atomic_write_json


class ModelProposalProvider:
    """Connect cached model responses to the paper's proposal protocol."""

    def __init__(
        self, backend: Any, client: CachedClient, config: RunConfig, output_dir: Path
    ) -> None:
        self.backend = backend
        self.client = client
        self.config = config
        self.output_dir = output_dir
        self.bounds = Bounds16D.from_submitted_bounds()
        self.model_version: str | None = None

    def __call__(
        self,
        incumbent: np.ndarray,
        incumbent_loss: float,
        calls: Sequence[Mapping[str, Any]],
        round_index: int,
    ) -> ExternalProposal:
        versions = {
            row["proposal_metadata"]["model_version"]
            for row in calls
            if row.get("proposal_metadata", {}).get("model_version")
        }
        if len(versions) > 1:
            raise ValueError("Stored run contains multiple model versions")
        if versions:
            self.model_version = next(iter(versions))
        flat = self.bounds.to_flat(incumbent)
        call_index = _incumbent_call_index(calls, incumbent, incumbent_loss)
        details = self.backend.details_for(flat, call_index)
        if set(details) != set(self.config.frequencies_hz):
            raise ValueError("Prompt diagnostic frequencies differ from the configured objective")
        sweep = [details[f] for f in self.config.frequencies_hz]
        directory = self.output_dir / "vlm_calls" / f"round_{round_index:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        if self.config.use_media:
            media, worst, frequencies = build_media(
                sweep,
                self.backend.dataset,
                directory / "media",
                self.config.metric.real_fps,
                width=self.config.media.width,
                height=self.config.media.height,
            )
        else:
            media = []
            worst = float(max(sweep, key=lambda row: row["mean_l2"])["frequency"])
            frequencies = list(self.config.frequencies_hz)
        history, previous = _history_from_calls(calls)
        limit = self.config.vlm.prompt_history_limit
        context = _build_prompt_context(
            current_flat=flat,
            history=history[-limit:] if limit else [],
            incumbent_call_index=call_index,
            media_paths=media,
            param_keys=self.bounds.keys,
            previously_selected=previous,
            round_index=round_index,
            skeleton_frequencies=frequencies,
            sweep=sweep,
            worst_frequency=worst,
            fps=self.config.metric.real_fps,
            width=self.config.media.width,
            height=self.config.media.height,
        )
        path = directory / "prompt_context.json"
        if path.exists():
            if read_json(path) != context:
                raise ValueError(f"Prompt context differs on resume: {path}")
        else:
            atomic_write_json(path, context)
        prompt = build_prompt(
            flat,
            context["sweep"],
            context["history"],
            round_index,
            previous,
            worst_frequency=worst if media else None,
            skeleton_frequencies=frequencies if media else None,
            treatment=self.config.prompt_treatment,
            training_frequencies=self.config.interpolation_training_frequencies_hz,
        )
        response = self.client.call(
            prompt,
            media,
            directory / "api",
            request_identity={
                "incumbent_call_index": call_index,
                "round_index": round_index,
                "treatment_sha256": context["treatment_sha256"],
            },
        )
        if self.model_version is not None and self.model_version != response.model_version:
            raise ValueError(
                f"Model version changed during the run: {self.model_version} to {response.model_version}"
            )
        self.model_version = response.model_version
        selected, values, parsed = parse_response(response.text)
        proposed = self.bounds.to_normalized({**flat, **values})
        metadata = {
            "selected_parameters": selected,
            "proposed_values": values,
            "parsed_response": parsed,
            "diagnosis": parsed.get("diagnosis", ""),
            "confidence": parsed.get("confidence"),
            "model_version": response.model_version,
            "response_id": response.response_id,
            "usage_metadata": dict(response.usage),
            "response_sha256": hashlib.sha256(response.text.encode()).hexdigest(),
            "round_dir": str(directory.relative_to(self.output_dir)),
            "sweep_before_mm": float(incumbent_loss),
            "preclip_norm": float(np.linalg.norm(proposed - incumbent)),
        }
        return ExternalProposal(proposed, metadata)
