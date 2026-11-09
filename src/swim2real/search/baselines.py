"""Optional Bayesian optimization and CMA-ES constructors."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from swim2real.config import BOConfig, CMAConfig, RunConfig
from swim2real.parameters import DIMENSION, Bounds16D


def build_skopt_bo(algorithm_seed: int, config: BOConfig | None = None) -> Any:
    """Build the normalized ask/tell Bayesian optimizer."""
    config = config or RunConfig.defaults().bayesopt
    try:
        from skopt import Optimizer
        from skopt.space import Real
    except ImportError as exc:
        raise ImportError(
            "Bayesian optimization requires pip install 'swim2real[baselines]'"
        ) from exc
    return Optimizer(
        dimensions=[Real(0.0, 1.0, name=f"x{index}") for index in range(DIMENSION)],
        base_estimator=config.base_estimator,
        n_initial_points=config.n_initial_points,
        initial_point_generator=config.initial_point_generator,
        acq_func=config.acquisition_function,
        acq_optimizer=config.acquisition_optimizer,
        random_state=int(algorithm_seed),
    )


def build_cmaes(
    initial: ArrayLike,
    *,
    algorithm_seed: int,
    population_size: int,
    config: CMAConfig | None = None,
) -> Any:
    """Build CMA-ES in bounded normalized coordinates."""
    config = config or RunConfig.defaults().cmaes
    if algorithm_seed == 0:
        raise ValueError("CMA-ES requires a nonzero seed")
    if population_size not in config.populations:
        raise ValueError("CMA-ES population must be enabled in config")
    try:
        import cma
    except ImportError as exc:
        raise ImportError("CMA-ES requires pip install 'swim2real[baselines]'") from exc
    point = np.asarray(initial, dtype=np.float64)
    Bounds16D.from_submitted_bounds().to_flat(point)
    return cma.CMAEvolutionStrategy(
        point.tolist(),
        config.sigma_normalized,
        {
            "bounds": [[0.0] * DIMENSION, [1.0] * DIMENSION],
            "popsize": int(population_size),
            "seed": int(algorithm_seed),
            "verbose": -9,
        },
    )
