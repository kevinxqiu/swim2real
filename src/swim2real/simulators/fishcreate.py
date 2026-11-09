"""Lazy adapter for a separately installed fishCREATE model."""

from __future__ import annotations

import hashlib
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import time as _time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from swim2real.config import PAPER_DEFAULTS
from swim2real.parameters import to_simulator_params
from swim2real.simulators.base import Trace

MODEL_FILES = ("_simulate.py", "Meshes/auto_tendonFish.py")


class FishCreateSimulator:
    """Run fishCREATE in an isolated process with its own import path."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        max_steps: int = PAPER_DEFAULTS["simulator"]["max_steps"],
        timeout_seconds: float = PAPER_DEFAULTS["simulator"]["timeout_seconds"],
    ) -> None:
        selected = root or os.environ.get("FISHCREATE_ROOT")
        if not selected:
            raise FileNotFoundError(
                "Set simulator.root in the run config or FISHCREATE_ROOT to a fishCREATE checkout containing _simulate.py and Meshes/auto_tendonFish.py"
            )
        self.root = Path(selected).expanduser().resolve()
        self.max_steps = max_steps
        if not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("fishCREATE timeout_seconds must be finite and positive")
        self.timeout_seconds = timeout_seconds
        self.validate_installation()

    def validate_installation(self) -> None:
        """Check the external model files without importing the simulator."""
        for name in MODEL_FILES:
            if not (self.root / name).is_file():
                raise FileNotFoundError(
                    f"Missing fishCREATE file: {self.root / name}. Set simulator.root or FISHCREATE_ROOT."
                )

    def identity(self) -> dict[str, Any]:
        """Identify the external source files and its Git revision when available."""
        commit = None
        if shutil.which("git"):
            result = subprocess.run(
                ["git", "-C", str(self.root), "rev-parse", "--verify", "HEAD"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                commit = result.stdout.strip()
            elif (self.root / ".git").exists():
                raise RuntimeError(
                    f"Cannot identify fishCREATE Git revision: {result.stderr.strip()}"
                )
        return {
            "backend": "fishcreate",
            "git_commit": commit,
            "source_files": {
                name: hashlib.sha256((self.root / name).read_bytes()).hexdigest()
                for name in (*MODEL_FILES, "Meshes/__init__.py")
                if (self.root / name).is_file()
            },
        }

    def run(
        self, params: dict[str, float], frequency_hz: float, duration_s: float, fps: int
    ) -> Trace:
        """Run the simulator in a subprocess without changing the caller's cwd."""
        import importlib.util

        if importlib.util.find_spec("mujoco") is None:
            raise ImportError("fishCREATE requires pip install 'swim2real[sim]'")
        with tempfile.TemporaryDirectory(prefix="swim2real-") as temporary:
            directory = Path(temporary)
            request = directory / "request.json"
            import json

            request.write_text(
                json.dumps(
                    {
                        "params": params,
                        "frequency_hz": frequency_hz,
                        "duration_s": duration_s,
                        "fps": fps,
                        "max_steps": self.max_steps,
                    },
                    allow_nan=False,
                )
            )
            output = directory / "trace.npz"
            package_root = str(Path(__file__).resolve().parents[2])
            program = "import sys;sys.path.insert(0,sys.argv[1]);from swim2real.simulators.fishcreate import _worker;_worker(sys.argv[2],sys.argv[3])"
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join([str(self.root), env.get("PYTHONPATH", "")])
            try:
                result = subprocess.run(
                    [sys.executable, "-c", program, package_root, str(request), str(output)],
                    cwd=self.root,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"fishCREATE simulation exceeded timeout of {self.timeout_seconds:g} seconds"
                ) from exc
            if result.returncode:
                raise RuntimeError(
                    f"fishCREATE simulation failed: {result.stderr.strip() or result.stdout.strip()}"
                )
            if result.stderr.strip():
                warnings.warn(result.stderr.strip(), RuntimeWarning, stacklevel=2)
            with np.load(output, allow_pickle=False) as saved:
                trace = Trace(saved["timestamps"], saved["markers"], saved["motor_signal"])
            trace.validate()
            return trace


def _worker(request_path: str, output_path: str) -> None:
    from swim2real.config import read_json

    request = read_json(request_path)
    simulation = importlib.import_module("_simulate")
    model = importlib.import_module("Meshes.auto_tendonFish")
    params = to_simulator_params(request["params"], model.SYSTEMPARAMETERS)
    trace = _simulate(
        Path.cwd(),
        (simulation.sim_fish, simulation.SineSignal, model.SYSTEMPARAMETERS, model.generate_xml),
        params,
        request["frequency_hz"],
        request["duration_s"],
        request["fps"],
        request["max_steps"],
    )
    np.savez_compressed(
        output_path,
        timestamps=trace.timestamps,
        markers=trace.markers,
        motor_signal=trace.motor_signal,
    )


def _simulate(
    root: Path,
    modules: tuple[Any, ...],
    params: dict[str, Any],
    frequency: float,
    totaltime: float,
    videofps: int,
    max_steps: int,
    warmuptime: float = 0.0,
) -> Trace:
    import mujoco

    _, SineSignal, _, generate_xml = modules
    signal = SineSignal(frequency=frequency)
    mesh_dir = root / "Meshes"
    with tempfile.NamedTemporaryFile(
        prefix=".swim2real-", suffix=".xml", dir=mesh_dir, delete=False
    ) as xml_handle:
        xml_path = Path(xml_handle.name)
    try:
        generate_xml(params, str(xml_path))
        model = mujoco.MjModel.from_xml_path(str(xml_path))
    finally:
        xml_path.unlink(missing_ok=True)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    initLen = data.ten_length.copy()
    dt = float(model.opt.timestep)
    output_fps = float(videofps)
    if not np.isfinite(output_fps) or output_fps <= 0.0 or output_fps > 1.0 / dt:
        raise ValueError("videofps must be finite, positive, and no faster than the simulator")
    log = {
        "time": [],
        "runtime": [],
        "controls": [],
        "markers": [],
        "tendonLengths": [],
        "motorAngularVelocity": [],
        "motorPower": [],
        "mass": float(sum((model.body_mass[i] for i in range(model.nbody)))),
    }
    counter = 0
    sample_origin: float | None = None
    sample_index = 0
    while counter < max_steps and data.time < totaltime + warmuptime:
        data.ctrl = signal(data.time)
        startTime = _time.time()
        mujoco.mj_step(model, data)
        counter += 1
        if data.time < warmuptime:
            continue
        if sample_origin is None:
            sample_origin = float(data.time)
        target_time = sample_origin + sample_index / output_fps
        if data.time + dt / 2.0 >= target_time:
            log["time"].append(data.time)
            log["controls"].append(data.ctrl.copy())
            log["runtime"].append(_time.time() - startTime)
            log["tendonLengths"].append(data.ten_length.copy())
            motorTorque = data.joint("motor_0").qfrc_actuator[0]
            motorAngularVel = data.joint("motor_0").qvel[0]
            log["motorAngularVelocity"].append(motorAngularVel)
            log["motorPower"].append(motorTorque * motorAngularVel)
            ms = []
            for i in range(10):
                if i == 4:
                    ms.append(data.site("bodyFin_0").xpos[:2].copy())
                ms.append(data.site(f"marker{i}_0").xpos[:2].copy())
            log["markers"].append(np.array(ms).flatten())
            sample_index += 1
    if counter >= max_steps:
        raise RuntimeError(f"Simulation exceeded {max_steps} steps")
    for k in log:
        log[k] = np.array(log[k])
    if (log["tendonLengths"].max() - log["tendonLengths"].min()) / initLen.mean() > 0.1:
        warnings.warn("Simulation tendon lengthening exceeds 10%", RuntimeWarning, stacklevel=2)
    return Trace(
        np.asarray(log["time"]),
        np.asarray(log["markers"]).reshape(-1, 11, 2),
        np.asarray(log["motorAngularVelocity"]),
    )
