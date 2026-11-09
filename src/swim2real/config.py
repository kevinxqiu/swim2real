"""Typed JSON configuration, with paths relative to the configuration file."""

from __future__ import annotations

import copy
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

PAPER_DEFAULTS: dict[str, Any] = json.loads(files("swim2real").joinpath("paper.json").read_text())


def read_json(path: str | Path) -> Any:
    """Read strict JSON and report its source on failure."""
    source = Path(path)

    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        return json.loads(source.read_text(encoding="utf-8"), parse_constant=reject)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read JSON file {source}: {exc}") from exc


def _merge(default: dict[str, Any], supplied: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    if not isinstance(supplied, dict):
        raise ValueError(f"{prefix or 'config'} must be an object")
    result = copy.deepcopy(default)
    for key, value in supplied.items():
        name = f"{prefix}{key}"
        if key not in default:
            raise ValueError(f"Unknown configuration setting: {name}")
        if isinstance(default[key], dict):
            result[key] = _merge(default[key], value, name + ".")
        else:
            result[key] = value
    return result


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str, minimum: float = 0, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (positive and value == minimum):
        raise ValueError(f"{name} must be {'>' if positive else '>='} {minimum}")
    return float(value)


def _path(value: Any, base: Path, name: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a path string")
    path = Path(value).expanduser()
    path = (base / path).resolve() if not path.is_absolute() else path.resolve()
    return path


@dataclass(frozen=True)
class SPSAConfig:
    probe_radius_normalized: float
    step_radius_normalized: float


@dataclass(frozen=True)
class BOConfig:
    acquisition_function: str
    acquisition_optimizer: str
    base_estimator: str
    initial_point_generator: str
    n_initial_points: int


@dataclass(frozen=True)
class CMAConfig:
    populations: tuple[int, ...]
    sigma_normalized: float


@dataclass(frozen=True)
class SimulatorConfig:
    backend: str
    root: Path | None
    duration_s: float
    fps: int
    max_steps: int
    timeout_seconds: float


@dataclass(frozen=True)
class VLMConfig:
    model: str
    temperature: float
    decoding_seed: int
    max_output_tokens: int
    thinking_budget: int
    max_transport_attempts: int
    timeout_seconds: float
    retry_delay_seconds: float
    prompt_history_limit: int
    api_version: str


@dataclass(frozen=True)
class OpenAIConfig:
    model: str
    base_url: str | None
    api_key_env: str
    frames_per_clip: int
    max_side_pixels: int
    jpeg_quality: int
    reasoning_effort: str
    reasoning_style: str
    parse_mode: str
    temperature: float
    decoding_seed: int
    max_output_tokens: int
    max_transport_attempts: int
    timeout_seconds: float
    retry_delay_seconds: float
    retry_multiplier: float
    retry_max_delay_seconds: float


@dataclass(frozen=True)
class MediaConfig:
    width: int
    height: int


@dataclass(frozen=True)
class MetricConfig:
    real_fps: float
    transient_s: float
    max_real_frames: int
    head_markers: int
    sim_centerline_indices: tuple[int, ...]


@dataclass(frozen=True)
class RunConfig:
    frequencies_hz: tuple[float, ...]
    budget: int
    backtracking_alphas: tuple[float, ...]
    random_radii_normalized: tuple[float, ...]
    spsa: SPSAConfig
    bayesopt: BOConfig
    cmaes: CMAConfig
    rng_namespace: int
    method_codes: dict[str, int]
    failure_penalty_mm_per_frequency: float
    seeds: tuple[int, ...]
    method: str
    dataset: Path | None
    simulator: SimulatorConfig
    provider: str
    vlm: VLMConfig
    openai_compat: OpenAIConfig
    metric: MetricConfig
    realized_frequencies_hz: dict[float, float]
    prompt_treatment: str
    use_media: bool
    media: MediaConfig
    interpolation_training_frequencies_hz: tuple[float, ...]
    study_seeds: dict[str, tuple[int, ...]]

    @classmethod
    def defaults(cls) -> RunConfig:
        """Return the paper settings without requiring external inputs."""
        return cls.from_dict({})

    @classmethod
    def from_dict(cls, value: dict[str, Any], base_dir: Path | None = None) -> RunConfig:
        """Merge overrides into the paper profile and validate their types."""
        d = _merge(PAPER_DEFAULTS, value)
        base = base_dir or Path.cwd()

        def numbers(key: str) -> tuple[float, ...]:
            values = d[key]
            if not isinstance(values, list) or not values:
                raise ValueError(f"{key} must be a nonempty list")
            return tuple(_number(v, key, positive=True) for v in values)

        freqs = numbers("frequencies_hz")
        if tuple(sorted(set(freqs))) != freqs:
            raise ValueError("frequencies_hz must be unique and increasing")
        alphas = numbers("backtracking_alphas")
        if any(v > 1 for v in alphas) or tuple(sorted(set(alphas), reverse=True)) != alphas:
            raise ValueError("backtracking_alphas must decrease in (0, 1]")
        radii = numbers("random_radii_normalized")
        seeds = d["seeds"]
        if not isinstance(seeds, list) or not seeds:
            raise ValueError("seeds must be a nonempty list")
        for seed in seeds:
            if _integer(seed, "seed") > 2**32 - 1:
                raise ValueError("seed must fit uint32")
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be unique")
        codes = d["method_codes"]
        for code in codes.values():
            if _integer(code, "method code") > 2**32 - 1:
                raise ValueError("method code must fit uint32")
        if len(set(codes.values())) != len(codes):
            raise ValueError("method codes must be unique")
        if not isinstance(d["method"], str) or d["method"] not in codes:
            raise ValueError(f"Unknown method: {d['method']}")
        if _integer(d["rng_namespace"], "rng_namespace") > 2**32 - 1:
            raise ValueError("rng_namespace must fit uint32")
        sim = d["simulator"]
        if not isinstance(sim["backend"], str) or sim["backend"] not in {"toy", "fishcreate"}:
            raise ValueError("simulator.backend must be toy or fishcreate")
        root = _path(sim["root"] or os.environ.get("FISHCREATE_ROOT"), base, "simulator.root")
        simulator = SimulatorConfig(
            sim["backend"],
            root,
            _number(sim["duration_s"], "duration_s", positive=True),
            _integer(sim["fps"], "fps", 1),
            _integer(sim["max_steps"], "max_steps", 1),
            _number(sim["timeout_seconds"], "simulator.timeout_seconds", positive=True),
        )
        spsa = SPSAConfig(
            *[_number(d["spsa"][k], k, positive=True) for k in SPSAConfig.__dataclass_fields__]
        )
        if spsa.probe_radius_normalized > 0.5:
            raise ValueError("SPSA probe radius must be <= 0.5")
        bo = d["bayesopt"]
        _integer(bo["n_initial_points"], "bayesopt.n_initial_points", 1)
        for key in [
            "acquisition_function",
            "acquisition_optimizer",
            "base_estimator",
            "initial_point_generator",
        ]:
            if not isinstance(bo[key], str) or not bo[key]:
                raise ValueError(f"bayesopt.{key} must be a nonempty string")
        cma = d["cmaes"]
        if not isinstance(cma["populations"], list) or not cma["populations"]:
            raise ValueError("cmaes.populations must be a nonempty list")
        for pop in cma["populations"]:
            _integer(pop, "cmaes population", 2)
        cma_config = CMAConfig(
            tuple(cma["populations"]), _number(cma["sigma_normalized"], "sigma", positive=True)
        )
        if (
            d["method"].startswith("cmaes_pop")
            and int(d["method"].removeprefix("cmaes_pop")) not in cma_config.populations
        ):
            raise ValueError("selected CMA population is not enabled")
        vlm = d["vlm"]
        for k in ["decoding_seed", "thinking_budget", "prompt_history_limit"]:
            _integer(vlm[k], "vlm." + k)
        for k in ["max_output_tokens", "max_transport_attempts"]:
            _integer(vlm[k], "vlm." + k, 1)
        _number(vlm["temperature"], "vlm.temperature")
        _number(vlm["timeout_seconds"], "vlm.timeout_seconds", positive=True)
        _number(vlm["retry_delay_seconds"], "vlm.retry_delay_seconds")
        for k in ["model", "api_version"]:
            if not isinstance(vlm[k], str) or not vlm[k]:
                raise ValueError(f"vlm.{k} must be a nonempty string")
        compat = d["openai_compat"]
        for key in ["decoding_seed", "max_output_tokens", "max_transport_attempts"]:
            _integer(compat[key], "openai_compat." + key, 0 if key == "decoding_seed" else 1)
        for key in [
            "temperature",
            "timeout_seconds",
            "retry_delay_seconds",
            "retry_max_delay_seconds",
        ]:
            _number(compat[key], "openai_compat." + key, positive=key == "timeout_seconds")
        _number(compat["retry_multiplier"], "openai_compat.retry_multiplier", minimum=1.0)
        if not isinstance(d["provider"], str) or d["provider"] not in {
            "gemini",
            "openai_compat",
            "scripted",
        }:
            raise ValueError("provider must be gemini, openai_compat or scripted")
        if compat["base_url"] is not None:
            if not isinstance(compat["base_url"], str):
                raise ValueError("openai_compat.base_url must be a URL string")
            url = urlparse(compat["base_url"])
            if (
                url.scheme not in {"https", "http"}
                or not url.netloc
                or url.username
                or url.password
            ):
                raise ValueError(
                    "openai_compat.base_url must be an HTTP endpoint without credentials"
                )
        if not isinstance(compat["api_key_env"], str) or not re.fullmatch(
            "[A-Za-z_][A-Za-z0-9_]*", compat["api_key_env"]
        ):
            raise ValueError("openai_compat.api_key_env must name an environment variable")
        if not isinstance(compat["model"], str) or not compat["model"]:
            raise ValueError("openai_compat.model must be a nonempty string")
        _integer(compat["frames_per_clip"], "frames_per_clip", 2)
        _integer(compat["max_side_pixels"], "max_side_pixels", 32)
        if not 1 <= _integer(compat["jpeg_quality"], "jpeg_quality", 1) <= 100:
            raise ValueError("jpeg_quality must lie in [1, 100]")
        if compat["reasoning_style"] not in {"reasoning_effort", "reasoning"}:
            raise ValueError("reasoning_style must be reasoning_effort or reasoning")
        if compat["reasoning_effort"] not in {"none", "minimal", "low", "medium", "high", "xhigh"}:
            raise ValueError("unknown reasoning_effort")
        if compat["parse_mode"] not in {"json_first", "lenient_fenced_json"}:
            raise ValueError("unknown parse_mode")
        if not isinstance(d["use_media"], bool):
            raise ValueError("use_media must be boolean")
        if d["prompt_treatment"] not in {"eight_frequency", "interpolation"}:
            raise ValueError("unknown prompt_treatment")
        expected = (
            tuple(PAPER_DEFAULTS["frequencies_hz"])
            if d["prompt_treatment"] == "eight_frequency"
            else numbers("interpolation_training_frequencies_hz")
        )
        if ("vlm" in d["method"] or d["prompt_treatment"] == "interpolation") and freqs != expected:
            raise ValueError("VLM frequencies must match the selected prompt treatment")
        metric = d["metric"]
        _number(metric["real_fps"], "real_fps", positive=True)
        _number(metric["transient_s"], "transient_s")
        _integer(metric["max_real_frames"], "max_real_frames", 2)
        if _integer(metric["head_markers"], "head_markers", 2) > 9:
            raise ValueError("head_markers must be <= 9")
        indices = metric["sim_centerline_indices"]
        if (
            not isinstance(indices, list)
            or len(indices) != 9
            or any(
                isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 11
                for index in indices
            )
            or len(set(indices)) != 9
        ):
            raise ValueError("sim_centerline_indices must contain nine unique indices")
        for index in indices:
            _integer(index, "centerline index")
        metric_config = MetricConfig(
            metric["real_fps"],
            metric["transient_s"],
            metric["max_real_frames"],
            metric["head_markers"],
            tuple(indices),
        )
        realized = {
            float(k): _number(v, "realized frequency", positive=True)
            for k, v in d["realized_frequencies_hz"].items()
        }
        if any(f not in realized for f in freqs):
            raise ValueError("realized frequency mapping is incomplete")
        media = d["media"]
        for key in ["width", "height"]:
            if _integer(media[key], "media." + key, 2) % 2:
                raise ValueError(f"media.{key} must be even for H.264 encoding")
        training = numbers("interpolation_training_frequencies_hz")
        if tuple(sorted(set(training))) != training:
            raise ValueError("interpolation training frequencies must be unique and increasing")
        study_seeds = d["study_seeds"]
        for name, values in study_seeds.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"study_seeds.{name} must be a nonempty list")
            for seed in values:
                if _integer(seed, f"study_seeds.{name}") > 2**32 - 1:
                    raise ValueError("study seed must fit uint32")
            if len(set(values)) != len(values):
                raise ValueError(f"study_seeds.{name} must be unique")
        return cls(
            freqs,
            _integer(d["budget"], "budget", 1),
            alphas,
            radii,
            spsa,
            BOConfig(**bo),
            cma_config,
            d["rng_namespace"],
            codes,
            _number(d["failure_penalty_mm_per_frequency"], "failure penalty", positive=True),
            tuple(seeds),
            d["method"],
            _path(d["dataset"], base, "dataset"),
            simulator,
            d["provider"],
            VLMConfig(**vlm),
            OpenAIConfig(**compat),
            metric_config,
            realized,
            d["prompt_treatment"],
            d["use_media"],
            MediaConfig(**media),
            training,
            {name: tuple(values) for name, values in study_seeds.items()},
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the effective configuration as strict JSON values."""

        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {str(k): convert(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [convert(v) for v in value]
            return value

        return convert(asdict(self))


def load_config(
    path: str | Path | None = None, *, overrides: dict[str, Any] | None = None
) -> RunConfig:
    """Load a run configuration and apply explicit CLI overrides."""
    source = Path(path).resolve() if path is not None else None
    supplied = read_json(source) if source else {}
    if overrides:
        supplied = _merge(_merge(PAPER_DEFAULTS, supplied), overrides)
    return RunConfig.from_dict(supplied, source.parent if source else Path.cwd())
