# Third-party notices

## fishCREATE simulator (not included)

The MuJoCo model of the fish, with its body meshes, tendon actuation and model
generator, comes from fishCREATE by Mike Yan Michelis. It is not part of this
repository, and the MIT license of this repository does not apply to it.
`swim2real` loads it from the folder named by the `FISHCREATE_ROOT`
environment variable or the simulator setting of the run configuration
(`src/swim2real/simulators/fishcreate.py`).

## Pool recordings (not included)

The real swimming videos and the marker tracks extracted from them were
recorded at EPFL. They are not part of this repository or of the run records.
A calibration reads them through a dataset manifest
(`configs/dataset.example.json`).

## Python packages

The packages below are installed from PyPI and keep their own licenses; none
is redistributed here.

| Package | License | Used for |
| --- | --- | --- |
| numpy | BSD-3-Clause | everything |
| mujoco | Apache-2.0 | simulation |
| opencv-python-headless | Apache-2.0 (OpenCV), MIT (wrapper) | videos and overlays |
| google-genai | Apache-2.0 | Gemini requests |
| openai | Apache-2.0 | the OpenAI-compatible second-model requests |
| scikit-optimize | BSD-3-Clause | Bayesian optimization baseline |
| cma | BSD-3-Clause | CMA-ES baseline |
| pytest, ruff | MIT | tests and linting |

FFmpeg with libx264 must be on the `PATH` for calibrations that send overlay
clips to the VLM. It is not distributed with this repository.

## Model APIs

The calibration calls Gemini 2.5 Pro through Google's Gemini API and, in the
second-model experiment, GLM-5.3 Flash (by Z.ai) through Nous Research's
OpenAI-compatible API.
Using these APIs is subject to the providers' terms. The run records contain
the models' replies from the paper's runs.

## Figures

`docs/images/swim2real_overview.png` and `docs/images/replication_strip.png`
are figures from the article accepted for publication in IEEE Robotics and
Automation Letters (© 2026 IEEE; the full notice is in the README).
`docs/images/swim2real_hero.gif` is a clip from the project page,
https://swim2real.github.io.
