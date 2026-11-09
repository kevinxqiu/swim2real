# fmt: off
# ruff: noqa
"""The paper's prompt builder and low-level response parser."""
import json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from swim2real.parameters import PARAM_INFO_16D, PARAM_BOUNDS_16D
PARAM_KEYS = list(PARAM_INFO_16D)
PARAM_INFO = PARAM_INFO_16D
PARAM_BOUNDS = PARAM_BOUNDS_16D

def _clean_json(text: str) -> str:
    """Strip markdown fences from LLM response."""
    text = text.strip()
    for fence in ("```json", "```", "'''json", "'''"):
        if text.startswith(fence):
            text = text[len(fence):]
        if text.endswith(fence.rstrip("json")):
            text = text[: -len(fence.rstrip("json"))]
    return text.strip()

def _compute_sweep_mean_l2(freq_sweep: Optional[List[Dict[str, Any]]]) -> float:
    """Compute mean L2 in mm across all frequencies in a sweep."""
    if not freq_sweep:
        return float("inf")
    return float(np.mean([s["mean_l2"] * 1000.0 for s in freq_sweep]))

def _build_subspace_prompt(
    current_flat: Dict[str, float],
    freq_sweep: List[Dict[str, Any]],
    round_history: List[Dict[str, Any]],
    round_number: int,
    previously_selected: List[List[str]],
    worst_freq: Optional[float] = None,
    skeleton_freqs: Optional[List[float]] = None,
    param_keys: Optional[List[str]] = None,
    param_info: Optional[Dict] = None,
    direct_eval: bool = False,
    unconstrained: bool = False,
    refinement_mode: bool = False,
) -> str:
    """Build prompt asking VLM to select 2-3 parameters to optimize."""
    _keys = param_keys or PARAM_KEYS
    _info = param_info or PARAM_INFO

    # Parameter table.
    param_lines = []
    for k in _keys:
        lo, hi, desc = _info[k]
        param_lines.append(
            f"  {k:<18s} | {current_flat[k]:>9.4f} | [{lo}, {hi}]"
            f"{'':<6s} | {desc}"
        )
    param_table = "\n".join(param_lines)

    # Sweep table with per-marker detail.
    sweep_lines = []
    for s in sorted(freq_sweep, key=lambda x: x["frequency"]):
        pm = s.get("per_marker_l2_mm", [])
        pm_str = ", ".join(str(int(round(v))) for v in pm) if pm else "N/A"
        sweep_lines.append(
            f"      {s['frequency']:.2f}  | {s['mean_l2']*1000:>6.1f} "
            f"| [{pm_str}]"
            f" | {s['sim_velocity']:>8.4f} | {s['real_velocity']:>8.4f}"
        )
    mean_l2 = _compute_sweep_mean_l2(freq_sweep)
    sweep_lines.append(f"      MEAN  | {mean_l2:>6.1f} |")
    sweep_table = "\n".join(sweep_lines)

    # Round history.
    if round_history:
        hist_lines = []
        for rh in round_history:
            params_str = ", ".join(rh["selected_parameters"])
            cand_mm = rh.get("candidate_mm") or rh.get("best_in_round_mm")
            cand_str = f"{cand_mm:>10.1f}" if cand_mm is not None else "       N/A"
            hist_lines.append(
                f"  {rh['round']:>5d} | [{params_str}]"
                f" | {rh['sweep_before_mm']:>10.1f} | {cand_str}"
                f" | {rh['sweep_after_mm']:>10.1f}"
                f" | {rh['improvement_mm']:>+7.1f}"
            )
        history_section = (
            "### Previous Optimization Rounds\n\n"
            "  Round | Subspace Selected | Before | Candidate | After | Change\n"
            "  -----------------------------------------------------------------------\n"
            + "\n".join(hist_lines)
        )
    else:
        history_section = "### Previous Optimization Rounds\n\n(first round)"

    # Video description (only if videos are provided).
    video_section = ""
    if worst_freq is not None:
        skel_freqs = skeleton_freqs or [worst_freq]
        skel_desc_lines = []
        for i, sf in enumerate(skel_freqs):
            skel_desc_lines.append(
                f"{2 + i}. Skeleton overlay at {sf:.2f} Hz "
                f"— sim (blue) vs real (red) centerline"
            )
        skel_desc = "\n".join(skel_desc_lines)
        n_videos = 1 + len(skel_freqs)

        video_section = f"""
### Videos (multi-frequency diagnostic)

You are provided with {n_videos} videos:
1. Real video at {worst_freq:.2f} Hz — perspective-corrected footage
{skel_desc}

The skeleton overlays show sim (blue) vs real (red) 9-marker centerline on dark
background with per-frame L2 error. Compare how the mismatch changes across
frequencies to identify which parameters need adjustment.
"""

    # Repeat warning.
    repeat_warning = ""
    if previously_selected:
        last = previously_selected[-1]
        consec = 0
        for ps in reversed(previously_selected):
            if sorted(ps) == sorted(last):
                consec += 1
            else:
                break
        if consec >= 2:
            repeat_warning = (
                f"\nWARNING: The subspace {last} has been selected "
                f"{consec} times consecutively. You MUST select different "
                "parameters this round.\n"
            )

    # Build selection instruction based on mode.
    if refinement_mode:
        selection_instruction = (
            "A numerical optimizer (CMA-ES) will now fine-tune parameters near "
            "the current solution. Select 3 to 5 parameters that you believe are "
            "most sensitive near the current operating point. Propose starting "
            "values for CMA-ES to refine around."
        )
    elif unconstrained:
        selection_instruction = (
            "Select ANY parameters you believe are responsible for the mismatch "
            "and propose new values for each. You may change as few as 1 or as "
            "many as ALL parameters — use your physical reasoning to decide which "
            "parameters need adjustment and by how much."
        )
    else:
        selection_instruction = (
            "Select exactly 2 or 3 parameters and propose new values for each."
        )

    prompt = f"""\
### Instruction

You are diagnosing a MuJoCo fish simulation that does not match real-world behavior.
Your task is to:
1. Identify which parameters are MOST RESPONSIBLE for the current sim-real mismatch.
2. Propose specific values for those parameters based on your physical reasoning.

{"Your proposed values will be evaluated directly — propose the best values you can based on your physical reasoning."
if direct_eval else
"A numerical optimizer (CMA-ES) will refine your proposed values, so they don't need to be exact — but good starting values significantly accelerate convergence."}

The fish has 9 tracked markers along its centerline (M0=nose to M8=tail tip).
Error is measured as L2 distance between sim and real marker positions in
head-frame coordinates, averaged across all 8 motor frequencies (0.5-2.25 Hz).

This is optimization round {round_number + 1}.

### Current Parameters

  Parameter          |   Current |     Bounds     | Description
  ---------------------------------------------------------------------------
{param_table}

### Full-Spectrum Sweep (all 8 frequencies)

  Freq (Hz) | L2 (mm) | Per-Marker L2 [M0..M8] (mm) | Sim vel  | Real vel
  -------------------------------------------------------------------------
{sweep_table}

{history_section}

{repeat_warning}
{video_section}
### Output

{selection_instruction}
For each parameter, explain your physical reasoning for the proposed value.

All proposed values MUST be within the bounds shown above.

Respond with a single JSON object:
{{
    "diagnosis": "<str: 2-4 sentences on what the data reveals about the mismatch>",
    "selected_parameters": ["<param1>", "<param2>", ...],
    "proposed_values": {{"<param1>": <float>, "<param2>": <float>, ...}},
    "reasoning": "<str: why these parameters and values — include direction of change>",
    "confidence": <float: 0.0 to 1.0>
}}
"""
    return prompt

def _parse_subspace_response(
    raw_text: str,
    valid_keys: Optional[List[str]] = None,
    param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    min_params: int = 2,
    max_params: int = 3,
) -> Tuple[Optional[List[str]], Optional[Dict[str, float]], Optional[Dict[str, Any]]]:
    """Parse VLM response.

    Returns (selected_params, proposed_values, full_parsed) or (None, None, None).
    proposed_values may be None if VLM didn't provide them (backward compat).

    Args:
        min_params: Minimum number of valid params required (default 2).
        max_params: Maximum params to keep (default 3). Set high to unconstrain.
    """
    _valid_keys = set(valid_keys or PARAM_KEYS)
    _bounds = param_bounds or PARAM_BOUNDS
    try:
        cleaned = _clean_json(raw_text)
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None, None, None

    selected = parsed.get("selected_parameters", [])
    if not isinstance(selected, list):
        return None, None, parsed

    # Validate parameter names.
    valid = [k for k in selected if k in _valid_keys]
    if len(valid) < min_params:
        return None, None, parsed

    # Enforce max params.
    if len(valid) > max_params:
        valid = valid[:max_params]

    parsed["selected_parameters"] = valid

    # Extract proposed values (optional — VLM may not provide them).
    proposed_raw = parsed.get("proposed_values", {})
    proposed_values: Optional[Dict[str, float]] = None
    if isinstance(proposed_raw, dict) and proposed_raw:
        proposed_values = {}
        for k in valid:
            if k in proposed_raw:
                try:
                    val = float(proposed_raw[k])
                    lo, hi = _bounds.get(k, (float("-inf"), float("inf")))
                    proposed_values[k] = max(lo, min(hi, val))
                except (TypeError, ValueError):
                    pass
        # Only use proposed values if we got at least one valid entry.
        if not proposed_values:
            proposed_values = None

    return valid, proposed_values, parsed
