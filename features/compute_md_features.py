# ============================================================== #
#  Module:      features/compute_md_features.py
#  Description: Physics feature extraction — RMSD, Rg, native contacts, φ/ψ dihedrals per frame
#  Author:      Siya Jethliya
#  Copyright (c) 2026 SciVizAI — All rights reserved.
# ============================================================== #

#!/usr/bin/env python3
"""
Compute MD features from trajectory files.

Features computed:
- RMSD from reference structure
- Radius of gyration
- Number of native contacts
- Phi/Psi backbone dihedral angles (sin/cos encoded)

All features - and everything downstream that uses the returned trajectory -
are computed on PROTEIN atoms only (see select_analysis_atoms). Water, ions,
ligands and other non-protein atoms are excluded at load time.
"""
import logging
from collections import Counter

import numpy as np
import mdtraj as md

log = logging.getLogger(__name__)


# -------------------------------------------------------------- #
# Function: select_analysis_atoms
# -------------------------------------------------------------- #
def select_analysis_atoms(topology):
    """Atoms the pipeline analyses: the protein, and nothing else.

    Kept: every residue mdtraj recognises as protein, plus any other residue that
    carries a protein backbone (atoms N, CA and C), so modified amino acids
    (e.g. selenomethionine, MSE) are not silently dropped.
    Excluded: water, ions, ligands, glycans, lipids, cofactors.

    Why: with explicit solvent, an all-atom RMSD or radius of gyration measures
    the water (a 1 ns solvated ubiquitin run gave a 37 A "plateau RMSD", against
    ~1-2 A for the protein), and equilibration trimming then tracks water
    diffusion. For protein-only trajectories (e.g. ATLAS) this selects every
    atom, so their results are unchanged.

    Returns:
        indices: sorted int array of atom indices to keep.
        info: dict with n_atoms_total, n_atoms_analysed, excluded (residue name ->
              atom count) and nonstandard_kept (residue name -> residue count).
    """
    keep, excluded, nonstd = [], Counter(), Counter()
    for res in topology.residues:
        names = {a.name for a in res.atoms}
        if res.is_protein:
            keep.extend(a.index for a in res.atoms)
        elif {"N", "CA", "C"} <= names:
            keep.extend(a.index for a in res.atoms)
            nonstd[res.name] += 1
        else:
            excluded[res.name] += res.n_atoms
    idx = np.array(sorted(keep), dtype=int)
    info = {
        "n_atoms_total": int(topology.n_atoms),
        "n_atoms_analysed": int(idx.size),
        "excluded": dict(sorted(excluded.items(), key=lambda kv: -kv[1])),
        "nonstandard_kept": dict(nonstd),
    }
    return idx, info



# -------------------------------------------------------------- #
# Function: compute_features
# -------------------------------------------------------------- #
def compute_features(topology_path, trajectory_path, stride=1, reference_frame=0,
                     return_selection=False):
    """
    Compute MD features from a trajectory.
    
    Args:
        topology_path: Path to topology file (PDB)
        trajectory_path: Path to trajectory file (XTC, DCD, etc.)
        stride: Load every stride-th frame
        reference_frame: Frame index for RMSD reference
        return_selection: also return the atom-selection info dict
        
    Returns:
        features: Dictionary of feature arrays
        traj: MDTraj trajectory object, PROTEIN ATOMS ONLY
        selection (only if return_selection): see select_analysis_atoms
    """
    # Load trajectory against the (canonical) topology. Fail fast with a clear,
    # located message if the atom sets disagree — this is exactly where topology
    # drift historically surfaced as the opaque MDTraj error
    # "topology and trajectory files might not contain the same atoms".
    top_full = md.load_topology(topology_path)
    keep, selection = select_analysis_atoms(top_full)
    if keep.size == 0:
        raise ValueError(
            f"No protein atoms found in topology '{topology_path}'. SciML-MD analyses "
            f"protein dynamics; residues present: "
            f"{', '.join(list(selection['excluded'])[:10])}")
    subset = keep.size < top_full.n_atoms
    try:
        if subset:
            # md.load(..., atom_indices=...) does NOT detect a topology/trajectory
            # atom-count mismatch - it silently loads garbage. Check the full atom
            # set against the first frame first; this raises on any mismatch.
            md.load_frame(trajectory_path, 0, top=topology_path)
            # Then load only the protein: same coordinates as load-then-slice,
            # without ever holding the solvent in memory.
            traj = md.load(trajectory_path, top=topology_path, stride=stride,
                           atom_indices=keep)
        else:
            traj = md.load(trajectory_path, top=topology_path, stride=stride)
    except (ValueError, IOError) as exc:
        raise ValueError(
            f"Topology/trajectory mismatch: could not load '{trajectory_path}' "
            f"with topology '{topology_path}'. Feature extraction must use the "
            f"SAME canonical topology produced during simulation "
            f"(canonical_topology.pdb). Original error: {exc}"
        ) from exc

    # Explicit atom-count assertion (defensive; md.load also enforces equality).
    expected = keep.size if subset else top_full.n_atoms
    if traj.n_atoms != expected:
        raise ValueError(
            f"Topology drift detected: trajectory has {traj.n_atoms} atoms but "
            f"topology '{topology_path}' has {expected}. These must match."
        )
    if subset:
        excl = ", ".join(f"{k} {v}" for k, v in list(selection["excluded"].items())[:6])
        log.debug("Analysing %d protein atoms of %d; excluded %d non-protein atoms (%s)",
                 keep.size, top_full.n_atoms, top_full.n_atoms - keep.size, excl)
    if selection["nonstandard_kept"]:
        log.debug("Kept non-standard residues with a protein backbone: %s",
                 ", ".join(f"{k} x{v}" for k, v in selection["nonstandard_kept"].items()))

    n_frames = len(traj)
    
    features = {}
    
    # 1. RMSD from reference frame
    ref = traj[reference_frame]
    features['rmsd'] = md.rmsd(traj, ref)
    
    # 2. Radius of gyration
    features['rg'] = md.compute_rg(traj)
    
    # 3. Native contacts (CA-CA pairs within 8 Angstroms)
    ca_atoms = traj.topology.select("name CA")
    if len(ca_atoms) > 1:
        features['contacts'] = _count_ca_contacts(traj, ca_atoms, cutoff_nm=0.8)
    else:
        features['contacts'] = np.zeros(n_frames)
    
    # 4. Backbone dihedrals (phi, psi)
    try:
        _, phi = md.compute_phi(traj)
        _, psi = md.compute_psi(traj)
    except (ValueError, RuntimeError, KeyError):
        # ValueError: topology issues, RuntimeError: computation failures
        # KeyError: missing atom types
        phi = np.zeros((n_frames, 1))
        psi = np.zeros((n_frames, 1))
    
    # Sin/cos encoding of dihedrals
    if phi.size > 0:
        features['phi_sin'] = np.sin(phi).mean(axis=1)
        features['phi_cos'] = np.cos(phi).mean(axis=1)
    else:
        features['phi_sin'] = np.zeros(n_frames)
        features['phi_cos'] = np.zeros(n_frames)
    
    if psi.size > 0:
        features['psi_sin'] = np.sin(psi).mean(axis=1)
        features['psi_cos'] = np.cos(psi).mean(axis=1)
    else:
        features['psi_sin'] = np.zeros(n_frames)
        features['psi_cos'] = np.zeros(n_frames)
    
    if return_selection:
        return features, traj, selection
    return features, traj



# -------------------------------------------------------------- #
# Function: _max_threads
# -------------------------------------------------------------- #
def _max_threads():
    """Thread budget for per-frame neighbour searches.

    SCIML_MAX_THREADS (set per worker by run_all_proteins --jobs) caps it, so
    parallel proteins do not oversubscribe the CPU. Thread count never changes
    results here: each frame is computed independently and written to its own slot.
    """
    import os
    env = os.environ.get("SCIML_MAX_THREADS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(1, min(8, os.cpu_count() or 1))


# -------------------------------------------------------------- #
# Function: _count_ca_contacts
# -------------------------------------------------------------- #
def _count_ca_contacts(traj, ca_atoms, cutoff_nm=0.8, chunk=64,
                       search_margin_nm=1e-3, max_chunk_bytes=256e6,
                       n_threads=None):
    """Per-frame number of CA-CA pairs with mdtraj distance < `cutoff_nm`.

    EXACTLY equal to the original
        (md.compute_distances(traj, all_ca_pairs) < cutoff_nm).sum(axis=1)
    but without materialising all n_res*(n_res-1)/2 distances per frame, which
    was O(F * R^2) time and memory (9.4 GiB for 500 frames of 3,172 residues).

    How exactness is kept:
      1. A KD-tree (float64) proposes candidate pairs within
         cutoff + `search_margin_nm`. The margin (1e-3 nm) is orders of
         magnitude larger than any float32/float64 disagreement (~1e-6 nm even
         for 30 nm boxes), so every pair mdtraj would count is a candidate.
      2. Candidates are then measured with md.compute_distances itself - the
         same float32 code and periodic handling as before - and thresholded
         with the same strict "<". The KD-tree only decides what to measure,
         never what counts.

    No unit cell, or an orthorhombic one (periodic KD-tree, minimum image), use
    this path. Triclinic cells use the original all-pairs computation, chunked
    over frames so memory stays bounded (correct, but not faster).
    """
    n_frames, n_res = len(traj), len(ca_atoms)
    counts = np.zeros(n_frames, dtype=float)
    ca_atoms = np.asarray(ca_atoms)
    ulen, uang = traj.unitcell_lengths, traj.unitcell_angles
    periodic = ulen is not None
    ortho = periodic and bool(np.allclose(uang, 90.0, atol=1e-3))

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None

    if cKDTree is not None and ((not periodic) or ortho):
        r_search = cutoff_nm + search_margin_nm
        xyz = traj.xyz[:, ca_atoms, :]

        def _candidates(f):
            pts = xyz[f].astype(np.float64)
            if ortho:
                box = np.asarray(ulen[f], dtype=np.float64)
                pts = np.mod(pts, box)
                pts = np.where(pts >= box, 0.0, pts)      # mod rounding to exactly L
                tree = cKDTree(pts, boxsize=box)
            else:
                tree = cKDTree(pts)
            p = tree.query_pairs(r_search, output_type="ndarray").astype(np.int64)
            return p[:, 0] * n_res + p[:, 1]               # i < j, flat key

        n_threads = _max_threads() if n_threads is None else max(1, int(n_threads))
        ex = None
        if n_threads > 1 and n_frames >= 32 and n_res >= 500:
            from concurrent.futures import ThreadPoolExecutor
            ex = ThreadPoolExecutor(max_workers=n_threads)
        try:
            for a in range(0, n_frames, chunk):
                b = min(a + chunk, n_frames)
                keys = list(ex.map(_candidates, range(a, b))) if ex else \
                    [_candidates(f) for f in range(a, b)]
                union = np.unique(np.concatenate(keys)) if keys else np.empty(0, np.int64)
                if union.size == 0:
                    continue                                # no pair close in any frame
                pairs = np.stack([ca_atoms[union // n_res], ca_atoms[union % n_res]], axis=1)
                d = md.compute_distances(traj[a:b], pairs)  # original arithmetic
                counts[a:b] = (d < cutoff_nm).sum(axis=1)
        finally:
            if ex:
                ex.shutdown()
        return counts

    # Triclinic (or no scipy): original computation, chunked over frames.
    ii, jj = np.triu_indices(n_res, k=1)
    pairs = np.stack([ca_atoms[ii], ca_atoms[jj]], axis=1)
    step = max(1, int(max_chunk_bytes // (4 * max(1, len(pairs)))))
    for a in range(0, n_frames, step):
        d = md.compute_distances(traj[a:a + step], pairs)
        counts[a:a + step] = (d < cutoff_nm).sum(axis=1)
    return counts


# -------------------------------------------------------------- #
# Function: features_to_matrix
# -------------------------------------------------------------- #
def features_to_matrix(features, keys=None):
    """
    Convert feature dictionary to a feature matrix.
    
    Args:
        features: Dictionary of feature arrays
        keys: List of feature keys to use (default: all)
        
    Returns:
        X: Feature matrix (n_frames x n_features)
        keys: List of feature keys used
    """
    if keys is None:
        keys = list(features.keys())
    
    X = np.column_stack([features[k] for k in keys])
    return X, keys
