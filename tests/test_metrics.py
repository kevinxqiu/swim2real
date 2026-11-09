import json
from pathlib import Path

import numpy as np
import pytest

from swim2real.config import RunConfig
from swim2real.metrics import compute_error

FIXTURES = Path(__file__).parent / "fixtures"


class ArrayDataset:
    metric = RunConfig.defaults().metric

    def __init__(self, raw):
        self.raw = raw

    def raw_markers(self, frequency_hz):
        return self.raw.copy()

    def load_markers(self, frequency_hz, retain_forward_motion=True):
        raw = self.raw_markers(frequency_hz)
        head = raw[:2].mean(axis=0, keepdims=True)
        raw -= head[:, 0:1] if retain_forward_motion else head
        return raw


@pytest.mark.parametrize("length", [180, 300, 500])
@pytest.mark.parametrize("retain", [True, False])
@pytest.mark.parametrize("align", [True, False])
def test_metrics_match_original(length, retain, align):
    with np.load(FIXTURES / "markers.npz") as saved:
        real = saved["real"]
        raw = np.resize(real, (9, length, 2)).copy() if length > 300 else real[:, :length].copy()
        log = {"time": saved["time"], "markers": saved["markers"]}
    result = compute_error(
        log, 0.75, auto_align=align, retain_forward_motion=retain, dataset=ArrayDataset(raw)
    )
    reference = json.loads((FIXTURES / "metrics.json").read_text())[f"{length}-{retain}-{align}"]
    for key, value in reference.items():
        np.testing.assert_allclose(result[key], value, rtol=1e-12, atol=1e-13, err_msg=key)
