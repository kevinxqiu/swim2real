import builtins
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from swim2real.config import RunConfig
from swim2real.search.baselines import build_cmaes, build_skopt_bo


def test_baseline_constructors_match_original(monkeypatch):
    reference = json.loads(
        (Path(__file__).parent / "fixtures" / "baseline_construction.json").read_text()
    )
    skopt = types.ModuleType("skopt")
    space = types.ModuleType("skopt.space")
    cma = types.ModuleType("cma")

    def real(lo, hi, name):
        return {"low": lo, "high": hi, "name": name}

    space.Real = real
    skopt.Optimizer = lambda **kwargs: kwargs
    cma.CMAEvolutionStrategy = lambda point, sigma, options: {
        "point": point,
        "sigma": sigma,
        "options": options,
    }
    monkeypatch.setitem(sys.modules, "skopt", skopt)
    monkeypatch.setitem(sys.modules, "skopt.space", space)
    monkeypatch.setitem(sys.modules, "cma", cma)
    assert build_skopt_bo(42) == reference["bo"]
    assert (
        build_cmaes(np.linspace(0.02, 0.98, 16), algorithm_seed=42, population_size=12)
        == reference["cmaes"]
    )


@pytest.mark.parametrize("package", ["skopt", "cma"])
def test_missing_baseline_dependency_raises_without_substitution(monkeypatch, package):
    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == package:
            raise ImportError("missing optional dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(ImportError, match="baselines"):
        if package == "skopt":
            build_skopt_bo(42)
        else:
            build_cmaes(np.full(16, 0.5), algorithm_seed=42, population_size=12)


def test_optimizer_settings_are_configurable(monkeypatch):
    config = RunConfig.from_dict(
        {"bayesopt": {"n_initial_points": 3}, "cmaes": {"sigma_normalized": 0.15}}
    )
    module = types.ModuleType("cma")
    module.CMAEvolutionStrategy = lambda point, sigma, options: sigma
    monkeypatch.setitem(sys.modules, "cma", module)
    assert (
        build_cmaes(np.full(16, 0.5), algorithm_seed=1, population_size=6, config=config.cmaes)
        == 0.15
    )
