"""Helpers shared by the launch regression tests.

run_fixture(): run the production pipeline (run_all_proteins.run_pipeline, CLI
defaults) on one trajectory directory.
compare_outputs(): compare two output trees file by file.

Two modes:
  strict          every number within rtol=1e-6 / atol=1e-9. Same-platform
                  guarantee: on the machine (and library versions) that wrote
                  the goldens, outputs reproduce to this precision.
  cross-platform  (default for goldens) what the science depends on must still
                  match exactly - file sets, keys, strings, verdicts, the trust
                  contract, preflight diagnostics, state assignments, transition
                  matrix, RMSF (all at rtol 1e-6). Two kinds of output may move
                  with floating-point rounding, measured macOS ARM vs Linux x86:
                    tICA coordinates            |d| <= 5.2e-6     -> atol 1e-4
                    rank-normalised scores      near-tied entries swap places:
                                                frame scores 1 rank step,
                                                kinetic residue scores 3 steps
                                                -> Spearman >= 0.999 and no entry
                                                   off by more than 5 rank steps;
                                                   NaN (withheld) pattern exact.
Set SCIML_GOLDEN_STRICT=1 to force strict mode.
"""
import os
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIXTURES = ROOT / "tests" / "fixtures"
GOLDEN = ROOT / "tests" / "golden"
RTOL, ATOL = 1e-6, 1e-9

# cross-platform tolerances (see module docstring for the measurements)
FLOAT_ATOL = {"tica_coords.npy": 1e-4}
RANKED = {"frame_scores_dynamic.csv", "residue_scores_dynamic.json",
          "residue_scores_kinetic.json"}
RANK_RHO_MIN, RANK_MAX_STEPS = 0.999, 5


def _strict_default():
    return os.environ.get("SCIML_GOLDEN_STRICT", "") not in ("", "0")


def _ranked_errors(x, y, label):
    """Rank-normalised scores (0-100): ordering must agree, values within a
    few rank steps, and the withheld (NaN) pattern must be identical."""
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape:
        return [f"{label}: shape {x.shape} != {y.shape}"]
    nx, ny = np.isnan(x), np.isnan(y)
    if not np.array_equal(nx, ny):
        return [f"{label}: withheld (NaN) entries differ"]
    x, y = x[~nx], y[~nx]
    if x.size < 2 or np.array_equal(x, y):
        return [] if np.array_equal(x, y) else [f"{label}: values differ"]
    step = 100.0 / x.size
    errs = []
    worst = np.abs(x - y).max()
    if worst > RANK_MAX_STEPS * step + 1e-9:
        errs.append(f"{label}: max |d| {worst:.3g} > {RANK_MAX_STEPS} rank steps ({RANK_MAX_STEPS * step:.3g})")
    if np.ptp(x) > 0 and np.ptp(y) > 0:
        rho = spearmanr(x, y)[0]
        if not rho >= RANK_RHO_MIN:
            errs.append(f"{label}: rank correlation {rho:.4f} < {RANK_RHO_MIN}")
    return errs


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


def compare_outputs(got_root, want_root, name, ignore_keys=(), strict=None):
    """Return a list of human-readable differences (empty = equivalent).
    strict=None follows SCIML_GOLDEN_STRICT (default: cross-platform mode)."""
    import pandas as pd
    strict = _strict_default() if strict is None else strict
    errs, ignore_keys = [], set(ignore_keys)
    for sub in ("art", "res"):
        g, w = Path(got_root) / sub / name, Path(want_root) / sub / name
        fg = sorted(p.name for p in g.iterdir()) if g.exists() else []
        fw = sorted(p.name for p in w.iterdir()) if w.exists() else []
        if fg != fw:
            errs.append(f"{sub}: file sets differ: got {fg}, want {fw}")
        for f in sorted(set(fg) & set(fw)):
            pg, pw = g / f, w / f
            if not strict and f in RANKED:
                if f.endswith(".json"):
                    a, b = json.loads(pg.read_text()), json.loads(pw.read_text())
                    if set(a) != set(b):
                        errs.append(f"{sub}/{f}: keys differ")
                        continue
                    k = sorted(a)
                    errs += _ranked_errors([a[i] for i in k], [b[i] for i in k], f"{sub}/{f}")
                else:
                    x, y = pd.read_csv(pg), pd.read_csv(pw)
                    if list(x.columns) != list(y.columns) or x.shape != y.shape:
                        errs.append(f"{sub}/{f}: columns/shape differ")
                        continue
                    for c in x.columns:
                        if c == "frame":
                            if not x[c].equals(y[c]):
                                errs.append(f"{sub}/{f}[frame]: differ")
                            continue
                        errs += _ranked_errors(x[c], y[c], f"{sub}/{f}[{c}]")
                continue
            if f.endswith(".json"):
                _cmp(json.loads(pg.read_text()), json.loads(pw.read_text()), f"{sub}/{f}", errs, ignore_keys)
            elif f.endswith(".npy"):
                x, y = np.load(pg, allow_pickle=False), np.load(pw, allow_pickle=False)
                if x.shape != y.shape or x.dtype != y.dtype:
                    errs.append(f"{sub}/{f}: shape/dtype {x.shape}/{x.dtype} != {y.shape}/{y.dtype}")
                elif np.issubdtype(x.dtype, np.integer) and not np.array_equal(x, y):
                    errs.append(f"{sub}/{f}: {(x != y).sum()} integer entries differ")
                elif not np.allclose(x, y, rtol=RTOL,
                                     atol=ATOL if strict else FLOAT_ATOL.get(f, ATOL),
                                     equal_nan=True):
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
