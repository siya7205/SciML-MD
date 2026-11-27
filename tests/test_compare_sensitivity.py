"""The golden comparison must stay tight enough to catch real regressions while
tolerating cross-platform floating-point rounding. Each case edits a copy of a
golden output tree and checks the verdict of compare_outputs (cross-platform mode).
"""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from pipeline_compare import GOLDEN, compare_outputs

NAME = "md_2pth_tric"


def _copy(tmp_path):
    for sub in ("art", "res"):
        shutil.copytree(GOLDEN / sub / NAME, tmp_path / sub / NAME)
    return tmp_path


def _edit_json(root, f, fn):
    p = root / "res" / NAME / f
    d = json.loads(p.read_text()); fn(d); p.write_text(json.dumps(d))


def _edit_npy(root, f, fn):
    p = root / "art" / NAME / f
    x = np.load(p); x = fn(x.copy()); np.save(p, x)


def _edit_csv(root, col, fn):
    p = root / "res" / NAME / "frame_scores_dynamic.csv"
    d = pd.read_csv(p); d[col] = fn(d[col].to_numpy(float)); d.to_csv(p, index=False)


def _swap_adjacent_residues(d):
    k = sorted(d, key=d.get); a, b = k[50], k[51]; d[a], d[b] = d[b], d[a]


def _shuffle_some(d, frac=0.2):
    k = sorted(d); rng = np.random.default_rng(0)
    pick = rng.choice(len(k), int(frac * len(k)), replace=False)
    vals = [d[k[i]] for i in pick]; rng.shuffle(vals)
    for i, v in zip(pick, vals): d[k[i]] = v


TOLERATED = {
    "tica rounding (5e-6)": lambda r: _edit_npy(r, "tica_coords.npy", lambda x: x + 5e-6),
    "two near-tied residues swap": lambda r: _edit_json(r, "residue_scores_kinetic.json", _swap_adjacent_residues),
    "tiny residue-score rounding": lambda r: _edit_json(r, "residue_scores_dynamic.json",
                                                       lambda d: d.update({k: v + 1e-3 for k, v in d.items()})),
}

CAUGHT = {
    "one state assignment changed": lambda r: _edit_npy(r, "dtraj.npy", lambda x: (x.__setitem__(5, (x[5] + 1) % 20), x)[1]),
    "transition matrix changed": lambda r: _edit_npy(r, "P.npy", lambda x: x * (1 + 1e-3)),
    "tica coordinates off by 1e-3": lambda r: _edit_npy(r, "tica_coords.npy", lambda x: x + 1e-3),
    "suitability verdict changed": lambda r: _edit_json(r, "preflight.json",
                                                       lambda d: d["suitability"].update(verdict="suitable")),
    "plateau RMSD off by 1%": lambda r: _edit_json(r, "preflight.json",
                                                  lambda d: d["equilibration"].update(rmsd_plateau_ang=d["equilibration"]["rmsd_plateau_ang"] * 1.01)),
    "trust headline changed": lambda r: _edit_json(r, "trust.json", lambda d: d.update(headline="Scores are shown.")),
    "frame-score order reversed": lambda r: _edit_csv(r, "component_local_density", lambda x: x[::-1]),
    "withheld channel now has values": lambda r: _edit_csv(r, "component_rarity", lambda x: np.nan_to_num(x, nan=50.0)),
    "residue scores shifted by 3 points": lambda r: _edit_json(r, "residue_scores_kinetic.json",
                                                              lambda d: d.update({k: v + 3 for k, v in d.items()})),
    "20% of residue scores shuffled": lambda r: _edit_json(r, "residue_scores_dynamic.json", _shuffle_some),
    "RMSF profile changed by 1%": lambda r: _edit_json(r, "residue_scores_rmsf.json",
                                                      lambda d: d.update({k: v * 1.01 for k, v in d.items()})),
    "output file missing": lambda r: (r / "res" / NAME / "trust.json").unlink(),
}


@pytest.mark.parametrize("case", sorted(TOLERATED))
def test_platform_rounding_is_tolerated(case, tmp_path):
    root = _copy(tmp_path); TOLERATED[case](root)
    assert compare_outputs(root, GOLDEN, NAME, strict=False) == []


@pytest.mark.parametrize("case", sorted(CAUGHT))
def test_real_change_is_caught(case, tmp_path):
    root = _copy(tmp_path); CAUGHT[case](root)
    assert compare_outputs(root, GOLDEN, NAME, strict=False), f"'{case}' was not detected"
