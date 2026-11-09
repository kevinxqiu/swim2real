"""Calibration of a soft robotic fish simulator."""

__version__ = "1.0.0"
from swim2real.parameters import PARAM_INFO_16D, Bounds16D, initial_vector_for_seed

__all__ = ["Bounds16D", "PARAM_INFO_16D", "initial_vector_for_seed", "__version__"]
