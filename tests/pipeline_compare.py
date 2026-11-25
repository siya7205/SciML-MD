"""Helpers shared by the launch regression tests.

run_fixture(): run the production pipeline (run_all_proteins.run_pipeline, CLI
defaults) on one trajectory directory.
compare_outputs(): compare two output trees file by file. Strings, booleans,
keys, file sets and array shapes must match exactly; numbers must agree to
rtol=1e-6 / atol=1e-9, so a golden written on one machine survives last-bit
floating-point differences on another while any real change still fails.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIXTURES = ROOT / "tests" / "fixtures"
GOLDEN = ROOT / "tests" / "golden"
RTOL, ATOL = 1e-6, 1e-9


def run_fixture(data_dir, name, out_dir):
    """Run the pipeline on data_dir/name; outputs go to out_dir/{art,res}/name."""
    from run_all_proteins import find_protein_dirs, run_pipeline
    found = {pid: (top, traj) for pid, top, traj in find_protein_dirs(str(data_dir))}
    top, traj = found[name]
    ok = run_pipeline(pdb_id=name, topology_path=top, trajectory_path=traj,
                      artifacts_dir=str(Path(out_dir) / "art"),
                      results_dir=str(Path(out_dir) / "res"))
    assert ok, f"pipeline failed on {name}"
    return Path(out_dir)


def _cmp(a, b, path, errs, ignore_keys):
    if isinstance(a, dict) and isinstance(b, dict):
        ka, kb = set(a) - ignore_keys, set(b) - ignore_keys
        if ka != kb:
            errs.append(f"{path}: keys differ {sorted(ka ^ kb)}")
        for k in ka & kb:
            _cmp(a[k], b[k], f"{path}/{k}", errs, ignore_keys)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            errs.append(f"{path}: length {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                _cmp(x, y, f"{path}[{i}]", errs, ignore_keys)
    elif isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or a is None or b is None:
        if a != b:
            errs.append(f"{path}: {a!r} != {b!r}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not np.isclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True):
            errs.append(f"{path}: {a!r} != {b!r}")
    elif a != b:
        errs.append(f"{path}: {a!r} != {b!r}")


def compare_outputs(got_root, want_root, name, ignore_keys=()):
    """Return a list of human-readable differences (empty = equivalent)."""
    import pandas as pd
    errs, ignore_keys = [], set(ignore_keys)
    for sub in ("art", "res"):
        g, w = Path(got_root) / sub / name, Path(want_root) / sub / name
        fg = sorted(p.name for p in g.iterdir()) if g.exists() else []
        fw = sorted(p.name for p in w.iterdir()) if w.exists() else []
        if fg != fw:
            errs.append(f"{sub}: file sets differ: got {fg}, want {fw}")
        for f in sorted(set(fg) & set(fw)):
            pg, pw = g / f, w / f
            if f.endswith(".json"):
                _cmp(json.loads(pg.read_text()), json.loads(pw.read_text()), f"{sub}/{f}", errs, ignore_keys)
            elif f.endswith(".npy"):
                x, y = np.load(pg, allow_pickle=False), np.load(pw, allow_pickle=False)
                if x.shape != y.shape or x.dtype != y.dtype:
                    errs.append(f"{sub}/{f}: shape/dtype {x.shape}/{x.dtype} != {y.shape}/{y.dtype}")
                elif not np.allclose(x, y, rtol=RTOL, atol=ATOL, equal_nan=True):
                    errs.append(f"{sub}/{f}: values differ (max |d| {np.nanmax(np.abs(x - y)):.3g})")
            elif f.endswith(".csv"):
                x, y = pd.read_csv(pg), pd.read_csv(pw)
                if list(x.columns) != list(y.columns) or x.shape != y.shape:
                    errs.append(f"{sub}/{f}: columns/shape differ")
                elif not np.allclose(x.to_numpy(float), y.to_numpy(float), rtol=RTOL, atol=ATOL, equal_nan=True):
                    errs.append(f"{sub}/{f}: values differ")
            elif pg.read_bytes() != pw.read_bytes():
                errs.append(f"{sub}/{f}: bytes differ")
    return errs
