"""Public prompt construction and structured proposal validation."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

from swim2real.config import PAPER_DEFAULTS
from swim2real.parameters import PARAM_BOUNDS_16D, PARAM_INFO_16D
from swim2real.prompts.paper import _build_subspace_prompt, _clean_json, _parse_subspace_response


def parse_response(text: str) -> tuple[list[str], dict[str, float], dict[str, Any]]:
    """Validate JSON structure before invoking the paper's response parser."""
    try:
        parsed = json.loads(_clean_json(text))
    except (ValueError, TypeError) as exc:
        raise ValueError("Provider response must contain a JSON object") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Provider response must be a JSON object")
    selected = parsed.get("selected_parameters")
    values = parsed.get("proposed_values")
    if (
        not isinstance(selected, list)
        or not selected
        or any(not isinstance(k, str) or k not in PARAM_INFO_16D for k in selected)
    ):
        raise ValueError("selected_parameters must list known parameter names")
    if (
        len(set(selected)) != len(selected)
        or not isinstance(values, dict)
        or set(selected) != set(values)
    ):
        raise ValueError("Each selected parameter must have exactly one proposed value")
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in values.values()
    ):
        raise ValueError("proposed_values must contain finite numbers")
    if "confidence" in parsed and (
        isinstance(parsed["confidence"], bool)
        or not isinstance(parsed["confidence"], (int, float))
        or not math.isfinite(parsed["confidence"])
        or not 0 <= parsed["confidence"] <= 1
    ):
        raise ValueError("confidence must be a number in [0, 1]")
    for key in ["diagnosis", "reasoning"]:
        if key in parsed and not isinstance(parsed[key], str):
            raise ValueError(f"{key} must be a string")
    selection, proposed, result = _parse_subspace_response(
        text,
        valid_keys=list(PARAM_INFO_16D),
        param_bounds=PARAM_BOUNDS_16D,
        min_params=1,
        max_params=len(PARAM_INFO_16D),
    )
    if selection is None or proposed is None or result is None:
        raise ValueError("Provider response has no valid parameter proposal")
    return selection, proposed, result


def training_prompt(
    prompt: str,
    frequencies: Sequence[float] = tuple(PAPER_DEFAULTS["interpolation_training_frequencies_hz"]),
) -> str:
    """Apply the paper's exact five-frequency interpolation text treatment."""
    sentence = "averaged across all 8 motor frequencies (0.5-2.25 Hz)."
    heading = "### Full-Spectrum Sweep (all 8 frequencies)"
    if prompt.count(sentence) != 1 or prompt.count(heading) != 1:
        raise ValueError("Prompt must contain one sweep heading and one frequency sentence")
    frequency_list = ", ".join(str(float(value)) for value in frequencies[:-1])
    frequency_list += f", and {float(frequencies[-1])}"
    count = len(frequencies)
    return prompt.replace(
        sentence,
        f"averaged across the {count} preregistered training motor frequencies ({frequency_list} Hz).",
    ).replace(heading, f"### Training-Frequency Sweep ({count} preregistered frequencies)")


def build_prompt(
    current_flat: Mapping[str, float],
    sweep: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
    round_index: int,
    previously_selected: Sequence[Sequence[str]],
    *,
    worst_frequency: float | None = None,
    skeleton_frequencies: Sequence[float] | None = None,
    treatment: str = "eight_frequency",
    training_frequencies: Sequence[float] = tuple(
        PAPER_DEFAULTS["interpolation_training_frequencies_hz"]
    ),
) -> str:
    """Render the paper prompt with the selected frequency treatment."""
    expected = (
        tuple(PAPER_DEFAULTS["frequencies_hz"])
        if treatment == "eight_frequency"
        else tuple(training_frequencies)
    )
    if (
        treatment not in {"eight_frequency", "interpolation"}
        or tuple(row["frequency"] for row in sweep) != expected
    ):
        raise ValueError("Prompt diagnostic frequencies must match the selected treatment")
    prompt = _build_subspace_prompt(
        dict(current_flat),
        list(sweep),
        list(history),
        round_index,
        [list(keys) for keys in previously_selected],
        worst_freq=worst_frequency,
        skeleton_freqs=list(skeleton_frequencies) if skeleton_frequencies else None,
        param_keys=list(PARAM_INFO_16D),
        param_info=PARAM_INFO_16D,
        direct_eval=True,
        unconstrained=True,
    )
    return training_prompt(prompt, training_frequencies) if treatment == "interpolation" else prompt
