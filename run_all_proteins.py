# ============================================================== #
#  Module:      run_all_proteins.py
#  Description: Batch ML pipeline orchestrator — runs all 5 steps for every protein in data/
#  Author:      Siya Jethliya
#  Copyright (c) 2026 SciVizAI — All rights reserved.
# ============================================================== #

#!/usr/bin/env python3
"""
Batch ML pipeline runner for all protein datasets in data/.

For every sub-directory under data/ that contains a topology.pdb **and** a
trajectory file (traj.xtc / traj.dcd / trajectory.xtc / trajectory.dcd),
this script runs the full anomaly-detection pipeline:

    1. Load trajectory & compute MD features
    2. Run tICA (TICA dimensionality reduction)
    3. Cluster conformational states (KMeans)
    4. Build a Markov State Model (MSM)
    5. Compute per-frame and per-residue anomaly signals

Outputs per protein
-------------------
    artifacts/{PDB_ID}/tica_coords.npy
    artifacts/{PDB_ID}/dtraj.npy
    artifacts/{PDB_ID}/P.npy
    artifacts/{PDB_ID}/pi.npy
    results/{PDB_ID}/frame_scores_dynamic.csv
    results/{PDB_ID}/residue_scores_dynamic.json

Usage
-----
    python run_all_proteins.py
    python run_all_proteins.py --data_dir data --lag_tica 10 --n_clusters 20
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Accepted trajectory file names (searched in order)
# ---------------------------------------------------------------------------
TRAJ_NAMES = ("traj.xtc", "traj.dcd", "trajectory.xtc", "trajectory.dcd")


# ---------------------------------------------------------------------------
# Per-stage timing
# ---------------------------------------------------------------------------
class StageTimer:
    """Accumulates wall time per named stage and logs a table at the end.

    `start(name)` closes the previous stage and opens `name`, so stages are
    delimited by marker calls and the pipeline body needs no re-indenting.
    """

    def __init__(self):
        self.t0 = time.perf_counter()
        self._name = None
        self._ts = self.t0
        self.stages = []                     # [(name, seconds)]

    def start(self, name):
        now = time.perf_counter()
        if self._name is not None:
            self.stages.append((self._name, now - self._ts))
        self._name, self._ts = name, now

    def finish(self):
        self.start(None)
        self.stages = [(n, t) for n, t in self.stages if n is not None]
        return self.stages

    def log_table(self, pdb_id):
        stages = self.finish()
        total = time.perf_counter() - self.t0
        log.info("[%s] Stage timings (total %.2fs):", pdb_id, total)
        for name, sec in sorted(stages, key=lambda x: -x[1]):
            log.info("[%s]   %-22s %7.2fs  %5.1f%%", pdb_id, name, sec,
                     100 * sec / total if total > 0 else 0.0)


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


# -------------------------------------------------------------- #
# Function: find_protein_dirs
# -------------------------------------------------------------- #
def find_protein_dirs(data_dir):
    """
    Return all sub-directories of *data_dir* that are valid protein datasets.

    A valid dataset directory must contain:
    - topology.pdb
    - at least one trajectory file matching TRAJ_NAMES

    Args:
        data_dir: Path to the root data directory.

    Returns:
        List of (pdb_id, topology_path, trajectory_path) tuples.
    """
    data_dir = Path(data_dir)
    datasets = []

    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue

        # Prefer the canonical topology written during simulation (post
        # PDBFixer + addHydrogens); fall back to the raw PDB only if absent.
        # The canonical file is the single source of truth for atom identity,
        # so feature extraction stays consistent with the trajectory.
        canonical = subdir / "canonical_topology.pdb"
        topology = canonical if canonical.exists() else subdir / "topology.pdb"
        if not topology.exists():
            continue

        traj = None
        for name in TRAJ_NAMES:
            candidate = subdir / name
            if candidate.exists():
                traj = candidate
                break

        if traj is None:
            log.debug("[%s] No trajectory found — skipping.", subdir.name)
            continue

        datasets.append((subdir.name, topology, traj))

    return datasets


# ---------------------------------------------------------------------------
# ML pipeline steps
# ---------------------------------------------------------------------------


# -------------------------------------------------------------- #
# Function: compute_features
# -------------------------------------------------------------- #
def compute_features(topology_path, trajectory_path, stride=1, return_selection=False):
    """
    Step 1 — Extract MD features from a trajectory.

    Delegates to the project's existing feature-extraction module.

    Args:
        topology_path: Path to topology.pdb.
        trajectory_path: Path to the trajectory file.
        stride: Load every *stride*-th frame.

    Returns:
        X: Feature matrix (n_frames × n_features).
        traj: MDTraj trajectory object.
    """
    from features.compute_md_features import (
        compute_features as _compute_features,
        features_to_matrix,
    )

    feats, traj, selection = _compute_features(str(topology_path), str(trajectory_path),
                                               stride=stride, return_selection=True)
    X, _ = features_to_matrix(feats)
    if return_selection:
        return X, traj, selection
    return X, traj



# -------------------------------------------------------------- #
# Function: run_tica
# -------------------------------------------------------------- #
def run_tica(X, lag=10, dim=5):
    """
    Step 2 — Time-lagged Independent Component Analysis.

    Args:
        X: Feature matrix (T × F).
        lag: TICA lag time in frames.
        dim: Number of tICA components to keep.

    Returns:
        Y: Projected coordinates (T × dim).
        tica_model: Fitted deeptime TICA model.
    """
    from deeptime.decomposition import TICA

    # Clamp lag to avoid exceeding trajectory length
    lag = min(lag, len(X) // 4)
    lag = max(lag, 1)

    tica_model = TICA(lagtime=lag, dim=dim).fit(X).fetch_model()
    Y = tica_model.transform(X)
    return Y, tica_model



# -------------------------------------------------------------- #
# Function: cluster_states
# -------------------------------------------------------------- #
def cluster_states(Y, n_clusters=20, seed=42):
    """
    Step 3 — Cluster tICA coordinates into discrete conformational states.

    Args:
        Y: tICA coordinate matrix (T × dim).
        n_clusters: Number of KMeans clusters.
        seed: Random seed for reproducibility.

    Returns:
        dtraj: Discrete trajectory as integer array (T,).
        kmeans_model: Fitted deeptime KMeans model.
    """
    # O-10 FIX: deeptime KMeans is platform-dependent even with a fixed seed
    # (measured ARI 0.43-0.51 between machines on identical input). We use
    # sklearn's Lloyd implementation with an explicit n_init, then CANONICALLY
    # RELABEL clusters by sorting their centres lexicographically. Canonical
    # relabelling is what actually buys reproducibility: it removes the
    # dependence on the order in which centres happen to be discovered.
    from sklearn.cluster import KMeans as SkKMeans

    n_clusters = min(n_clusters, len(Y) // 2)
    n_clusters = max(n_clusters, 2)

    km = SkKMeans(n_clusters=n_clusters, n_init=10, max_iter=300,
                  algorithm="lloyd", random_state=seed).fit(Y)

    centers = km.cluster_centers_
    order = np.lexsort(tuple(centers[:, i] for i in range(centers.shape[1] - 1, -1, -1)))
    remap = np.empty(n_clusters, dtype=np.int64)
    remap[order] = np.arange(n_clusters)
    dtraj = remap[km.labels_].astype(np.int64)
    km.cluster_centers_ = centers[order]
    return dtraj, km



# -------------------------------------------------------------- #
# Function: build_msm
# -------------------------------------------------------------- #
def build_msm(dtraj, lag=10):
    """
    Step 4 — Build a reversible Maximum Likelihood MSM.

    Args:
        dtraj: Discrete trajectory (T,).
        lag: MSM lag time in frames.

    Returns:
        msm: Fitted deeptime MarkovStateModel.
        P: Transition matrix (n_states × n_states).
        pi: Stationary distribution (n_states,).
    """
    from deeptime.markov.msm import MaximumLikelihoodMSM

    # Clamp lag
    lag = min(lag, len(dtraj) // 4)
    lag = max(lag, 1)

    msm = MaximumLikelihoodMSM(lagtime=lag, reversible=True).fit(dtraj).fetch_model()
    P = msm.transition_matrix
    pi = msm.stationary_distribution
    return msm, P, pi



# -------------------------------------------------------------- #
# Function: compute_anomaly_signals
# -------------------------------------------------------------- #
def compute_anomaly_signals(msm, dtraj, Y, lag_msm=10, k_neighbors=10, window=1):
    """
    Step 5 — Compute per-frame anomaly scores using kinetic + density signals.

    Args:
        msm: Fitted MSM.
        dtraj: Discrete trajectory.
        Y: tICA coordinate matrix.
        lag_msm: MSM lag for transition surprise.
        k_neighbors: Neighbours for local density.
        window: Moving-median smoothing window. DEFAULT 1 = OFF.

            Smoothing is off by default because a moving median averages away
            exactly the transient events this pipeline exists to find. Measured
            in the Phase 1 seeded-anomaly test (validation/phase1_results.json,
            defect D-09): on isolated single-frame anomalies the fused score
            achieves AUROC 0.839 unsmoothed but only 0.550 at window=3 and
            0.543 at window=5.

            Set window >= 3 only when the target is a SUSTAINED rare state and
            frame-to-frame noise is the dominant nuisance.

    Returns:
        frame_scores: Per-frame anomaly score in [0, 100].
        components: Dict of normalised individual signal arrays.
    """
    from scoring.anomaly_v2 import (
        compute_kinetic_signals,
        compute_local_density_signal,
        fuse_signals,
        moving_median,
    )

    # Clamp lag
    lag_msm = min(lag_msm, len(dtraj) // 4)
    lag_msm = max(lag_msm, 1)

    rarity, surprise = compute_kinetic_signals(msm, dtraj, lag_msm)
    density = compute_local_density_signal(Y, k=min(k_neighbors, len(Y) - 1))

    signals = {
        "rarity": rarity,
        "transition_surprise": surprise,
        "local_density": -density,  # invert so low density → high score
    }

    score_raw, components = fuse_signals(signals, method="median", normalize_method="rank")
    score_100 = score_raw * 100.0

    # D-09: smoothing is opt-in. window <= 1 leaves the score untouched.
    if window and window > 1:
        frame_scores = moving_median(score_100, window=window)
    else:
        frame_scores = score_100

    return frame_scores, components



# -------------------------------------------------------------- #
# Function: compute_residue_scores
# -------------------------------------------------------------- #
def _superposed_ca(traj):
    """CA atoms superposed onto frame 0 - shared by the residue-level steps,
    which each used to build an identical copy."""
    ca_idx = traj.topology.select("name CA")
    if len(ca_idx) == 0:
        return None
    return traj.atom_slice(ca_idx).superpose(traj.atom_slice(ca_idx))


def compute_residue_scores(traj, frame_scores, ca_traj=None):
    """
    Aggregate per-frame anomaly scores to per-residue scores.

    Combines two per-residue channels:
      1. rmsf_norm      — normalized RMSF (structural flexibility), and
      2. participation  — per-frame anomaly scores aggregated to residues,
                          weighting each frame's score by how far the residue
                          is displaced from its mean position in that frame
                          ("which residues move in the anomalous frames?").

    The previous implementation blended rmsf_norm with the trajectory-mean
    frame score — a single constant added to every residue — so the residue
    ranking was exactly RMSF (validated: Pearson r = 1.0 on 1VII). It also
    keyed residues by str(residue) (e.g. "MET41"), which collides across
    chains and silently dropped half the residues of multi-chain proteins
    (8H0R: 182 residues -> 91 entries).

    Args:
        traj: MDTraj trajectory object.
        frame_scores: Per-frame anomaly scores (T,) in [0, 100].

    Returns:
        residue_scores: Dict mapping residue key → score in [0, 100].
                        Keys are "NAMEnnn" (e.g. "MET41"); when the same
                        name+number occurs in several chains, all copies are
                        disambiguated as "NAMEnnn_chainK".
    """
    try:
        ca_idx = traj.topology.select("name CA")
        if len(ca_idx) == 0:
            ca_idx = traj.topology.select("protein")

        if len(ca_idx) == 0:
            return {}

        # RMSF of CA atoms (one per residue, selected via "name CA")
        if ca_traj is None or ca_traj.n_atoms != len(ca_idx):
            ca_traj = traj.atom_slice(ca_idx)
            ca_traj = ca_traj.superpose(ca_traj)
        mean_xyz = ca_traj.xyz.mean(axis=0)
        # Per-frame per-residue displacement from mean position [T, R]
        disp = np.sqrt(((ca_traj.xyz - mean_xyz) ** 2).sum(axis=-1))
        rmsf = np.sqrt((disp ** 2).mean(axis=0))
        rmsf_norm = rmsf / (rmsf.max() + 1e-12)

        # Anomaly participation: frame scores aggregated to residues, weighted
        # by that residue's displacement in each frame (same "weighted_mean"
        # scheme as scoring.signals.aggregate_frame_to_residue).
        fs = np.asarray(frame_scores, dtype=np.float64)
        T = min(len(fs), disp.shape[0])
        weighted = fs[:T, None] * disp[:T]
        participation = weighted.sum(axis=0) / (disp[:T].sum(axis=0) + 1e-10)

        # Collision-safe residue keys
        residues = [traj.topology.atom(a).residue for a in ca_idx]
        base_keys = [f"{r.name}{r.resSeq}" for r in residues]
        from collections import Counter
        dup = {k for k, c in Counter(base_keys).items() if c > 1}
        keys = [
            f"{k}_chain{r.chain.index}" if k in dup else k
            for k, r in zip(base_keys, residues)
        ]

        residue_scores = {}
        for i, key in enumerate(keys):
            combined = float(0.5 * rmsf_norm[i] * 100.0 + 0.5 * participation[i])
            residue_scores[key] = round(combined, 4)

        return residue_scores

    except Exception as exc:
        log.warning("Residue score computation failed: %s", exc)
        return {}


# -------------------------------------------------------------- #
# Function: compute_rmsf_residue_json
# -------------------------------------------------------------- #
def compute_rmsf_residue_json(traj, ca_traj=None):
    """
    Per-residue RMSF (Angstroms) keyed like compute_residue_scores.

    Written to results/{PDB_ID}/residue_scores_rmsf.json so that
    tools/export_for_asvs.py can populate rmsf_residue.json (previously the
    file was never produced and the ASVS export was empty).
    """
    try:
        ca_idx = traj.topology.select("name CA")
        if len(ca_idx) == 0:
            return {}
        if ca_traj is None or ca_traj.n_atoms != len(ca_idx):
            ca_traj = traj.atom_slice(ca_idx).superpose(traj.atom_slice(ca_idx))
        mean_xyz = ca_traj.xyz.mean(axis=0)
        rmsf_nm = np.sqrt(((ca_traj.xyz - mean_xyz) ** 2).sum(axis=-1).mean(axis=0))
        rmsf_ang = rmsf_nm * 10.0

        residues = [traj.topology.atom(a).residue for a in ca_idx]
        base_keys = [f"{r.name}{r.resSeq}" for r in residues]
        from collections import Counter
        dup = {k for k, c in Counter(base_keys).items() if c > 1}
        keys = [
            f"{k}_chain{r.chain.index}" if k in dup else k
            for k, r in zip(base_keys, residues)
        ]
        return {k: round(float(v), 4) for k, v in zip(keys, rmsf_ang)}
    except Exception as exc:
        log.warning("RMSF export failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Per-protein pipeline orchestrator
# ---------------------------------------------------------------------------


# -------------------------------------------------------------- #
# Function: run_pipeline
# -------------------------------------------------------------- #
def _run_pipeline_impl(
    pdb_id,
    topology_path,
    trajectory_path,
    artifacts_dir,
    results_dir,
    stride=1,
    lag_tica=10,
    dim_tica=5,
    n_clusters=20,
    lag_msm=10,
    k_neighbors=10,
    window=1,
    seed=42,
    T=None,
):
    """
    Run the complete ML pipeline for a single protein.

    Args:
        pdb_id: PDB identifier string (used for logging and directory names).
        topology_path: Path to topology.pdb.
        trajectory_path: Path to the trajectory file.
        artifacts_dir: Root directory for numpy artifacts.
        results_dir: Root directory for CSV/JSON results.
        stride: Trajectory stride.
        lag_tica: tICA lag time.
        dim_tica: tICA dimensions.
        n_clusters: KMeans clusters.
        lag_msm: MSM lag time.
        k_neighbors: k-NN for local density.
        window: Moving-median smoothing window; 1 = OFF (default, see D-09).
        seed: Random seed.

    Returns:
        True on success, False on failure.
    """
    T = T or StageTimer()
    art_dir = Path(artifacts_dir) / pdb_id
    res_dir = Path(results_dir) / pdb_id
    art_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    T.start("1 features")
    log.info("[%s] ── Step 1/5: Feature extraction", pdb_id)
    try:
        X, traj, selection = compute_features(topology_path, trajectory_path,
                                              stride=stride, return_selection=True)
    except Exception as exc:
        log.error("[%s] Feature extraction failed: %s", pdb_id, exc)
        return False
    n_excluded = selection["n_atoms_total"] - selection["n_atoms_analysed"]
    if selection.get("pbc_repair"):
        r = selection["pbc_repair"]
        log.warning("[%s]   Protein was split by the periodic box (%d frame(s) with cut "
                    "bonds, %d with chains in different images); re-joined before "
                    "analysis.", pdb_id, r["frames_with_cut_bonds"],
                    r["frames_with_separated_chains"])
    if n_excluded:
        log.info("[%s]   Analysing %d protein atoms; excluded %d non-protein atoms (%s)",
                 pdb_id, selection["n_atoms_analysed"], n_excluded,
                 ", ".join(f"{k} {v}" for k, v in list(selection["excluded"].items())[:6]))

    T.start("1b preflight")
    # --- D-10: discard equilibration frames before anything is scored ---
    # Startup relaxation from the deposited structure is not a rare event. An
    # external audit found frames 1-3 of ubiquitin scoring 99.7/100 purely from
    # unrelaxed crystal coordinates.
    try:
        from msm.preflight import (detect_equilibration, assess_suitability,
                                   assess_recurrence)

        eq_start, eq_info = detect_equilibration(traj)
        if eq_start > 0:
            log.warning("[%s]   Discarding %d equilibration frame(s) (%.1f%%); "
                        "plateau RMSD %.2f A", pdb_id, eq_start,
                        100 * eq_info.get("fraction_discarded", 0),
                        eq_info.get("rmsd_plateau_ang", float("nan")))
            traj = traj[eq_start:]
            X = X[eq_start:]

        suit = assess_suitability(traj)
        if suit["verdict"] != "suitable":
            log.warning("[%s]   SYSTEM SUITABILITY: %s", pdb_id, suit["verdict"].upper())
            for reason in suit["reasons"]:
                log.warning("[%s]     - %s", pdb_id, reason)
        # S-1b: does this trajectory ever revisit a conformation? The basin
        # census asks whether states exist; this asks whether they recur, and
        # an MSM needs both (report sections 18-19).
        rec = assess_recurrence(traj)
        if rec.get("verdict") in ("non_recurrent", "weak", "unknown"):
            log.warning("[%s]   RECURRENCE: %s - %s", pdb_id,
                        rec["verdict"].upper(), rec.get("reason", ""))
        pre_info = {"equilibration": eq_info, "suitability": suit,
                    "recurrence": rec}
    except Exception as exc:
        log.warning("[%s]   Preflight checks failed: %s", pdb_id, exc)
        pre_info = {"error": str(exc)}
    pbc_repair = selection.pop("pbc_repair", None)
    if n_excluded or selection["nonstandard_kept"]:
        # Provenance: recorded only when the selection changed something, so
        # protein-only inputs keep a byte-identical preflight.json.
        pre_info["atom_selection"] = selection
    if pbc_repair:
        pre_info["pbc_repair"] = pbc_repair
    suit = pre_info.get("suitability")
    rec = pre_info.get("recurrence")

    n_frames, n_feats = X.shape
    log.info("[%s]   %d frames × %d features", pdb_id, n_frames, n_feats)

    if n_frames < 10:
        log.warning("[%s] Too few frames (%d) — skipping.", pdb_id, n_frames)
        return False

    T.start("2 tICA")
    log.info("[%s] ── Step 2/5: tICA (lag=%d, dim=%d)", pdb_id, lag_tica, dim_tica)
    try:
        Y, _ = run_tica(X, lag=lag_tica, dim=dim_tica)
    except Exception as exc:
        log.error("[%s] tICA failed: %s", pdb_id, exc)
        return False

    np.save(art_dir / "tica_coords.npy", Y)
    log.info("[%s]   tICA shape: %s", pdb_id, Y.shape)

    T.start("3 clustering")
    log.info("[%s] ── Step 3/5: Clustering (n_clusters=%d)", pdb_id, n_clusters)
    try:
        dtraj, _ = cluster_states(Y, n_clusters=n_clusters, seed=seed)
    except Exception as exc:
        log.error("[%s] Clustering failed: %s", pdb_id, exc)
        return False

    np.save(art_dir / "dtraj.npy", dtraj)
    log.info("[%s]   Unique states: %d", pdb_id, len(np.unique(dtraj)))

    T.start("4 MSM")
    log.info("[%s] ── Step 4/5: MSM (lag=%d)", pdb_id, lag_msm)
    try:
        msm, P, pi = build_msm(dtraj, lag=lag_msm)
    except Exception as exc:
        log.error("[%s] MSM failed: %s", pdb_id, exc)
        return False

    np.save(art_dir / "P.npy", P)
    np.save(art_dir / "pi.npy", pi)
    log.info("[%s]   MSM states: %d", pdb_id, msm.n_states)

    n_labels = int(dtraj.max()) + 1
    if msm.n_states < n_labels:
        try:
            from scoring.anomaly_v2 import remap_dtraj_to_active_set

            dtraj_active, _ = remap_dtraj_to_active_set(dtraj, msm)
            n_dropped_frames = int((dtraj_active < 0).sum())
        except Exception:
            n_dropped_frames = -1
        log.warning(
            "[%s]   MSM active set covers %d of %d cluster states; %d frame(s) "
            "lie in disconnected states (treated as maximally rare/surprising).",
            pdb_id, msm.n_states, n_labels, n_dropped_frames,
        )
        # Persist the active-set mapping so downstream consumers of dtraj.npy
        # can remap labels correctly.
        try:
            np.save(art_dir / "state_symbols.npy",
                    np.asarray(msm.count_model.state_symbols))
        except Exception:
            pass

    T.start("5 scoring channels")
    log.info("[%s] ── Step 5/5: Scoring channels", pdb_id)
    try:
        frame_scores, components = compute_anomaly_signals(
            msm, dtraj, Y, lag_msm=lag_msm, k_neighbors=k_neighbors, window=window
        )
    except Exception as exc:
        log.error("[%s] Scoring failed: %s", pdb_id, exc)
        return False

    T.start("6 trust contract")
    # --- OI-34: enforce the trust contract BEFORE anything is written ------
    # CLAIMS.md says pi-derived quantities are meaningless without repeat
    # visits. Until 2026-08-31 the pipeline emitted them anyway. It no longer
    # does: withheld channels are written as NaN and declared in trust.json.
    try:
        from msm.preflight import assess_estimability
        from msm.trust import build_contract, gate_components, WITHHELD

        _tot = (rec or {}).get("total_ns")
        _nres = (suit or {}).get("n_residues")
        est = assess_estimability(dtraj, lag_msm, total_ns=_tot, n_residues=_nres)
        contract = build_contract(pdb_id, suitability=suit, recurrence=rec,
                                  estimability=est, n_frames=int(len(traj)))
        components, gated = gate_components(components, contract)
        if gated:
            log.warning("[%s]   WITHHELD (trust contract): %s",
                        pdb_id, ", ".join(sorted(gated)))
            # the fused score inherits the gap: recompute it from what survived
            live = [v for k, v in components.items()
                    if contract["channels"].get(k, {}).get("status") != WITHHELD]
            if live:
                frame_scores = np.nanmedian(np.vstack(live), axis=0) * 100.0
            else:
                frame_scores = np.full(len(frame_scores), np.nan)
        with open(res_dir / "trust.json", "w") as fh:
            json.dump(contract, fh, indent=2, default=str)
        log.info("[%s]   Trust contract → %s  (%s)", pdb_id,
                 res_dir / "trust.json", contract["headline"])
    except Exception as exc:
        log.error("[%s]   Trust contract FAILED: %s - refusing to write "
                  "ungated scores", pdb_id, exc)
        return False

    T.start("7 write frame scores")
    # --- Save frame scores ---
    import pandas as pd

    from scoring.anomaly_v2 import moving_median as _mm

    scores_df = pd.DataFrame(
        {
            "frame": np.arange(len(frame_scores)),
            # Primary score. Unsmoothed unless --window >= 3 was requested.
            "score_dynamic": frame_scores,
            # Reference smoothed track, always written so downstream consumers can
            # compare. NEVER use this one to detect transient events (D-09).
            "score_dynamic_smoothed_w5": _mm(np.asarray(frame_scores, float), window=5),
            **{f"component_{k}": v * 100.0 for k, v in components.items()},
        }
    )
    frame_csv = res_dir / "frame_scores_dynamic.csv"
    scores_df.to_csv(frame_csv, index=False)
    log.info("[%s]   Frame scores → %s", pdb_id, frame_csv)

    T.start("8 residue scores")
    # --- Save residue scores ---
    ca_traj = _superposed_ca(traj)
    residue_scores = compute_residue_scores(traj, frame_scores, ca_traj=ca_traj)
    residue_json = res_dir / "residue_scores_dynamic.json"
    with open(residue_json, "w") as fh:
        json.dump(residue_scores, fh, indent=2)
    log.info("[%s]   Residue scores → %s", pdb_id, residue_json)

    T.start("9 residue attribution")
    # --- E-1 FIX: RMSF-orthogonal attribution (the reported residue score) ---
    try:
        from scoring.residue_attribution import compute_residue_attribution

        attr = compute_residue_attribution(traj, frame_scores, ca_traj=ca_traj)
        keys = list(residue_scores.keys())
        vals = attr["score"]
        if len(keys) == len(vals):
            with open(res_dir / "residue_scores_kinetic.json", "w") as fh:
                json.dump({k: round(float(v), 4) for k, v in zip(keys, vals)}, fh, indent=2)
        with open(res_dir / "residue_attribution_diagnostics.json", "w") as fh:
            json.dump(attr["diagnostics"], fh, indent=2)
        log.info("[%s]   Kinetic residue attribution: Spearman vs RMSF %.3f "
                 "(legacy score was ~0.94-0.98)", pdb_id,
                 attr["diagnostics"]["spearman_final_vs_rmsf"])
    except Exception as exc:
        log.warning("[%s]   Kinetic residue attribution failed: %s", pdb_id, exc)

    T.start("10 write misc")
    # --- Save preflight record ---
    try:
        with open(res_dir / "preflight.json", "w") as fh:
            json.dump(pre_info, fh, indent=2, default=str)
    except Exception:
        pass

    # --- Save RMSF residue scores (consumed by tools/export_for_asvs.py) ---
    T.start("11 rmsf")
    rmsf_scores = compute_rmsf_residue_json(traj, ca_traj=ca_traj)
    if rmsf_scores:
        with open(res_dir / "residue_scores_rmsf.json", "w") as fh:
            json.dump(rmsf_scores, fh, indent=2)

    log.info(
        "[%s] ✓ Pipeline complete. Mean score: %.1f",
        pdb_id,
        float(frame_scores.mean()),
    )
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_pipeline_worker(max_threads, **kwargs):
    """Process-pool entry point: cap this worker's neighbour-search threads.

    Only the pipeline's own KD-tree threads are capped (SCIML_MAX_THREADS). Their
    count cannot change results. BLAS / OpenMP thread counts are deliberately
    left alone, because changing them can change floating-point reduction order;
    set OMP_NUM_THREADS yourself if you need to limit those.
    """
    import os
    os.environ["SCIML_MAX_THREADS"] = str(max(1, int(max_threads)))
    return run_pipeline(**kwargs)


def run_pipeline(pdb_id, *args, **kwargs):
    """Run the pipeline for one protein and log a per-stage timing table.

    Thin wrapper over `_run_pipeline_impl` (same arguments, same return value).
    """
    T = StageTimer()
    try:
        return _run_pipeline_impl(pdb_id, *args, T=T, **kwargs)
    finally:
        try:
            T.log_table(pdb_id)
        except Exception:                               # timing must never break a run
            pass


# -------------------------------------------------------------- #
# Function: main
# -------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Run the SciML-MD pipeline (screening, MSM, scoring channels, trust "
                    "contract) for every protein in data/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Root data directory containing {PDB_ID}/topology.pdb subdirectories",
    )
    parser.add_argument(
        "--artifacts_dir",
        default="artifacts",
        help="Directory for numpy artifacts (tica_coords, dtraj, P, pi)",
    )
    parser.add_argument(
        "--results_dir",
        default="results",
        help="Directory for CSV/JSON results",
    )
    parser.add_argument("--stride", type=int, default=1, help="Trajectory stride")
    parser.add_argument(
        "--lag_tica", type=int, default=10, help="tICA lag time (frames)"
    )
    parser.add_argument(
        "--dim_tica", type=int, default=5, help="Number of tICA components"
    )
    parser.add_argument(
        "--n_clusters", type=int, default=20, help="Number of KMeans clusters"
    )
    parser.add_argument(
        "--lag_msm", type=int, default=10, help="MSM lag time (frames)"
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=10,
        help="k for k-NN local density signal",
    )
    parser.add_argument(
        "--window", type=int, default=1,
        help="Moving-median smoothing window for frame scores. 1 = OFF (default). "
             "Smoothing suppresses isolated rare frames (defect D-09: AUROC 0.84 -> 0.55 "
             "on single-frame anomalies). Use >= 3 only for sustained rare states."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="Proteins to process in parallel (separate processes). 1 = sequential "
             "(default). Proteins are independent, so results are identical; "
             "memory use scales with jobs because each holds its trajectory.",
    )
    parser.add_argument(
        "--protein",
        default=None,
        help="Process only this PDB ID (useful for debugging)",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("SciML-MD pipeline")
    log.info("=" * 60)

    datasets = find_protein_dirs(args.data_dir)

    if not datasets:
        log.error(
            "No valid protein datasets found under '%s'. "
            "Each dataset must have topology.pdb and a trajectory file.",
            args.data_dir,
        )
        return 1

    if args.protein:
        datasets = [(pid, top, traj) for pid, top, traj in datasets if pid == args.protein]
        if not datasets:
            log.error("Protein '%s' not found in '%s'.", args.protein, args.data_dir)
            return 1

    log.info("Found %d protein dataset(s) to process.", len(datasets))

    n_ok = 0
    n_fail = 0
    failed_ids = []

    def _kwargs():
        return dict(
            artifacts_dir=args.artifacts_dir,
            results_dir=args.results_dir,
            stride=args.stride,
            lag_tica=args.lag_tica,
            dim_tica=args.dim_tica,
            n_clusters=args.n_clusters,
            lag_msm=args.lag_msm,
            k_neighbors=args.k_neighbors,
            window=args.window,
            seed=args.seed,
        )

    if args.jobs > 1 and len(datasets) > 1:
        # Proteins share nothing, so run them in separate processes. Results are
        # collected in input order, so the summary below is deterministic.
        from concurrent.futures import ProcessPoolExecutor

        import os
        n_workers = min(args.jobs, len(datasets))
        threads_per_worker = max(1, (os.cpu_count() or 1) // n_workers)
        log.info("Running %d proteins with %d parallel worker(s), up to %d "
                 "neighbour-search thread(s) each.", len(datasets), n_workers,
                 threads_per_worker)
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                (pdb_id, pool.submit(_run_pipeline_worker, threads_per_worker,
                                     pdb_id=pdb_id,
                                     topology_path=topology,
                                     trajectory_path=trajectory, **_kwargs()))
                for pdb_id, topology, trajectory in datasets
            ]
            for pdb_id, fut in futures:
                try:
                    ok = fut.result()
                except Exception as exc:                 # a crashed worker must not hide the rest
                    log.error("[%s] worker crashed: %s", pdb_id, exc)
                    ok = False
                if ok:
                    n_ok += 1
                else:
                    n_fail += 1
                    failed_ids.append(pdb_id)
    else:
        for i, (pdb_id, topology, trajectory) in enumerate(datasets, 1):
            log.info("")
            log.info("─" * 60)
            log.info("[%d/%d] Processing: %s", i, len(datasets), pdb_id)
            log.info("─" * 60)

            ok = run_pipeline(
                pdb_id=pdb_id,
                topology_path=topology,
                trajectory_path=trajectory,
                artifacts_dir=args.artifacts_dir,
                results_dir=args.results_dir,
                stride=args.stride,
                lag_tica=args.lag_tica,
                dim_tica=args.dim_tica,
                n_clusters=args.n_clusters,
                lag_msm=args.lag_msm,
                k_neighbors=args.k_neighbors,
                window=args.window,
                seed=args.seed,
            )

            if ok:
                n_ok += 1
            else:
                n_fail += 1
                failed_ids.append(pdb_id)

    log.info("")
    log.info("=" * 60)
    log.info("Batch run complete: %d succeeded, %d failed.", n_ok, n_fail)
    if failed_ids:
        log.warning("Failed proteins: %s", ", ".join(failed_ids))
    log.info("Artifacts → %s/", args.artifacts_dir)
    log.info("Results   → %s/", args.results_dir)
    log.info("=" * 60)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
