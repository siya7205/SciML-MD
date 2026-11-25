"""msm/bootstrap_msm.py must refuse to run unless explicitly overridden:
its confidence intervals are not valid (see BOOTSTRAP_DEFECTS)."""
import subprocess
import sys

import numpy as np
import pytest
import yaml

from pipeline_compare import ROOT

from msm import bootstrap_msm as bm


def test_function_refuses_by_default(tmp_path):
    np.save(tmp_path / "X.npy", np.zeros((50, 3)))
    with pytest.raises(RuntimeError, match="EXPERIMENTAL"):
        bm.bootstrap_msm(str(tmp_path / "X.npy"), str(tmp_path / "out"))
    assert not (tmp_path / "out").exists(), "must refuse before writing anything"


def test_cli_refuses_by_default(tmp_path):
    np.save(tmp_path / "X.npy", np.zeros((50, 3)))
    r = subprocess.run([sys.executable, str(ROOT / "msm" / "bootstrap_msm.py"),
                        "--features", str(tmp_path / "X.npy"), "--output_dir", str(tmp_path / "out")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "EXPERIMENTAL" in r.stderr
    assert not (tmp_path / "out").exists()


def test_override_lets_it_run(tmp_path):
    """With the override the function gets past the gate (it creates its output
    folder only after the gate). It may still fail on the documented defects -
    defect 4 does on this input - which is exactly why it is gated."""
    rng = np.random.default_rng(0)
    np.save(tmp_path / "X.npy", np.cumsum(rng.normal(size=(600, 5)), axis=0))
    cfg = yaml.safe_load(open(ROOT / "configs" / "pipeline.yaml"))
    cfg["bootstrap"]["n_iterations"] = 1
    yaml.safe_dump(cfg, open(tmp_path / "c.yaml", "w"))
    try:
        bm.bootstrap_msm(str(tmp_path / "X.npy"), str(tmp_path / "out"), str(tmp_path / "c.yaml"),
                         lag_tica=5, dim_tica=2, allow_experimental=True)
    except RuntimeError as e:
        assert "EXPERIMENTAL" not in str(e)
    except (IndexError, ValueError):
        pass                                   # known defects (BOOTSTRAP_DEFECTS)
    assert (tmp_path / "out").exists()
