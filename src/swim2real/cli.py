"""Command-line calibration, evaluation and input validation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

from swim2real import __version__
from swim2real.config import RunConfig, load_config, read_json
from swim2real.data import Dataset
from swim2real.evaluator import ObjectiveEvaluator, SimulationObjective
from swim2real.identity import package_identity
from swim2real.parameters import Bounds16D, initial_vector_for_seed
from swim2real.search.runner import CalibrationRunner
from swim2real.search.store import atomic_write_json, canonical_sha256


def _configuration(args: argparse.Namespace) -> RunConfig:
    overrides = {}
    if getattr(args, "simulator", None):
        overrides["simulator"] = {"backend": args.simulator}
    if getattr(args, "method", None):
        overrides["method"] = args.method
    if getattr(args, "budget", None) is not None:
        overrides["budget"] = args.budget
    return load_config(args.config, overrides=overrides)


def _inputs(config: RunConfig, *, require_media: bool = False) -> tuple[Any, Any]:
    if config.simulator.backend == "toy":
        from swim2real.simulators.toy import ToyDataset, ToySimulator

        simulator = ToySimulator()
        dataset = (
            Dataset.load(config.dataset, config.metric) if config.dataset else ToyDataset(config)
        )
    else:
        from swim2real.simulators.fishcreate import FishCreateSimulator

        simulator = FishCreateSimulator(
            config.simulator.root,
            max_steps=config.simulator.max_steps,
            timeout_seconds=config.simulator.timeout_seconds,
        )
        if config.dataset is None:
            raise ValueError("fishCREATE calibration requires dataset to name a dataset manifest")
        dataset = Dataset.load(config.dataset, config.metric)
    if isinstance(dataset, Dataset):
        dataset.validate(config.frequencies_hz, require_video=require_media and config.use_media)
    elif require_media and config.use_media and config.provider != "scripted":
        raise ValueError(
            "The toy demo requires use_media=false for a model provider, or an explicit dataset with prepared videos"
        )
    return simulator, dataset


def _effective_identity(config: RunConfig, dataset: Any, proposals: Path | None) -> dict[str, Any]:
    from swim2real.media import sha256_file

    identity = {"config": config.to_dict(), "dataset": {}, "proposals": None}
    if isinstance(dataset, Dataset):
        identity["dataset"] = {
            str(f): {
                "marker_csv": sha256_file(dataset.for_frequency(f).marker_csv),
                "real_video": sha256_file(dataset.for_frequency(f).real_video)
                if dataset.for_frequency(f).real_video
                else None,
                "crop_window": dataset.for_frequency(f).crop_window,
            }
            for f in config.frequencies_hz
        }
    if proposals:
        identity["proposals"] = sha256_file(proposals)
    return identity


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    """Run one method and seed, or resume its stored state."""
    config = _configuration(args)
    if args.proposals is not None and config.provider != "scripted":
        raise ValueError("--proposals requires provider=scripted")
    simulator, dataset = _inputs(
        config, require_media="vlm" in config.method and config.provider != "scripted"
    )
    output = args.output.resolve()
    seed = args.seed if args.seed is not None else config.seeds[0]
    initial_vector_for_seed(seed)
    # Resolve optional dependencies before any objective evaluations.
    if config.method == "bayesopt" and importlib.util.find_spec("skopt") is None:
        raise ImportError("Bayesian optimization requires pip install 'swim2real[baselines]'")
    if config.method.startswith("cmaes") and importlib.util.find_spec("cma") is None:
        raise ImportError("CMA-ES requires pip install 'swim2real[baselines]'")
    backend = SimulationObjective(simulator, dataset, config, output / "diagnostics")
    provider = None
    if "vlm" in config.method:
        if config.provider == "scripted":
            from swim2real.providers.base import ScriptedProvider

            if args.proposals is None:
                raise ValueError(
                    "The scripted provider requires --proposals with a JSON array of normalized vectors"
                )
            provider = ScriptedProvider(read_json(args.proposals))
        else:
            from swim2real.providers.proposal import ModelProposalProvider

            if config.provider == "gemini":
                from swim2real.providers.gemini import GeminiClient

                client = GeminiClient(config.vlm)
                key_env = "GEMINI_API_KEY"
                module = "google.genai"
            else:
                from swim2real.providers.openai_compat import OpenAICompatibleClient

                if not config.openai_compat.base_url:
                    raise ValueError("Set openai_compat.base_url in config")
                client = OpenAICompatibleClient(config.openai_compat)
                key_env = config.openai_compat.api_key_env
                module = "openai"
            if not os.environ.get(key_env):
                raise ValueError(f"Set {key_env} before calibration with this provider")
            try:
                available = importlib.util.find_spec(module)
            except ModuleNotFoundError:
                available = None
            if available is None:
                raise ImportError("Model providers require pip install 'swim2real[vlm]'")
            provider = ModelProposalProvider(backend, client, config, output)
    elif args.proposals is not None:
        raise ValueError("--proposals requires a VLM method with provider=scripted")
    completed = len(list((output / "state").glob("state_*.json"))) if args.resume else 0
    evaluator = ObjectiveEvaluator(
        backend.evaluate_frequency,
        frequencies_hz=config.frequencies_hz,
        failure_penalty_mm=config.failure_penalty_mm_per_frequency,
        budget=config.budget,
        start_call_index=completed,
        journal_dir=output / "objective",
        on_call_start=backend.begin_objective,
        recover_frequency=backend.recover_frequency,
    )
    identity = _effective_identity(config, dataset, args.proposals)
    implementation = package_identity()
    implementation["simulator"] = (
        simulator.identity() if config.simulator.backend == "fishcreate" else {"backend": "toy"}
    )
    runner = CalibrationRunner(
        method=config.method,
        problem_seed=seed,
        budget=config.budget,
        output_dir=output,
        evaluator=evaluator,
        code_manifest=implementation,
        config_sha256=canonical_sha256(identity),
        proposal_provider=provider,
        resume=args.resume,
        config=config,
    )
    result = runner.run()
    return {
        "output": str(output),
        "synthetic_demo": config.simulator.backend == "toy" and config.dataset is None,
        "summary": result["summary"],
        "calls_completed": len(result["calls"]),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Score one physical parameter vector across the configured frequencies."""
    config = _configuration(args)
    if args.output and args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    simulator, dataset = _inputs(config)
    params = read_json(args.params)
    Bounds16D.from_submitted_bounds().validate_flat(params)
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(prefix="swim2real-evaluate-") as temp:
        backend = SimulationObjective(simulator, dataset, config, Path(temp))
        evaluator = ObjectiveEvaluator(
            backend.evaluate_frequency,
            frequencies_hz=config.frequencies_hz,
            failure_penalty_mm=config.failure_penalty_mm_per_frequency,
            budget=1,
            on_call_start=backend.begin_objective,
        )
        result = evaluator.evaluate(params).to_dict()
        result["synthetic_demo"] = config.simulator.backend == "toy" and config.dataset is None
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.output, result)
    return result


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Check configured files, marker crops and simulator installation."""
    config = _configuration(args)
    _inputs(config, require_media="vlm" in config.method and config.provider != "scripted")
    return {
        "valid": True,
        "frequencies_hz": list(config.frequencies_hz),
        "simulator": config.simulator.backend,
        "synthetic_demo": config.simulator.backend == "toy" and config.dataset is None,
    }


def make_parser() -> argparse.ArgumentParser:
    """Build the public CLI and load optional analysis subcommands."""
    parser = argparse.ArgumentParser(
        prog="swim2real", description="Calibrate a soft robotic fish simulator."
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text, handler in [
        ("calibrate", "Run or resume one calibration.", calibrate),
        ("evaluate", "Evaluate physical parameters.", evaluate),
        ("validate-inputs", "Validate configuration and data files.", validate_inputs),
    ]:
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument(
            "--config", type=Path, help="JSON run config; defaults to the paper profile."
        )
        sub.add_argument(
            "--simulator", choices=["toy", "fishcreate"], help="Override the simulator backend."
        )
        sub.set_defaults(handler=handler)
        if name == "calibrate":
            sub.add_argument("--method", choices=list(RunConfig.defaults().method_codes))
            sub.add_argument("--seed", type=int)
            sub.add_argument("--budget", type=int)
            sub.add_argument("--output", type=Path, required=True)
            sub.add_argument("--resume", action="store_true")
            sub.add_argument(
                "--proposals", type=Path, help="Scripted normalized proposals as a JSON array."
            )
        elif name == "evaluate":
            sub.add_argument("--params", type=Path, required=True)
            sub.add_argument("--output", type=Path)
    try:
        analysis = importlib.util.find_spec("swim2real.analysis.cli")
    except ModuleNotFoundError as exc:
        if exc.name != "swim2real.analysis":
            raise
        analysis = None
    if analysis is not None:
        from swim2real.analysis.cli import add_subcommands

        add_subcommands(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute a subcommand, printing strict JSON or a clear error."""
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
    except KeyError as exc:
        parser.exit(2, f"{parser.prog}: Invalid JSON input: missing field {exc.args[0]!r}.\n")
    except TypeError as exc:
        parser.exit(2, f"{parser.prog}: Invalid JSON input structure: {exc}\n")
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    # Handlers that print their own report return an exit status instead of data.
    if isinstance(result, int) and not isinstance(result, bool):
        return result
    if result is not None:
        print(json.dumps(result, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
