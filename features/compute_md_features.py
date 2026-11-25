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
# Function: repair_periodic_images
# -------------------------------------------------------------- #
def repair_periodic_images(traj, bond_cut_nm=0.5, chunk=256):
    """Re-join a protein that the periodic box has cut apart. In place.

    Raw MD output (e.g. GROMACS without `trjconv -pbc mol`) wraps each atom into
    the box separately, so a protein crossing the box edge is cut in two, and
    chains of a complex can sit in different periodic images. RMSD, radius of
    gyration, superposition and RMSF are then meaningless: on a split ubiquitin
    trajectory the plateau RMSD read 10.8 A instead of 1.2 A, and the
    suitability verdict flipped from unsuitable to marginal.

    Only acts when a problem is detected, so intact trajectories are untouched:
      1. any bond longer than `bond_cut_nm` as stored but shorter under the
         minimum-image convention (i.e. cut by the box) -> make molecules whole;
      2. with several molecules (chains): a chain that touches no already-placed
         chain (largest first), but would after a one-box-vector shift, is moved
         there, per frame. Chains in contact are never moved, so an intact
         complex is untouched however elongated it is.
    Afterwards no bond may still be cut by the box, or this raises ValueError.

    Returns None if nothing was done, else a dict describing the repair.
    """
    if traj.unitcell_vectors is None:
        return None
    top = traj.topology
    bonds = np.array([[a.index, b.index] for a, b in top.bonds], dtype=int)
    if bonds.size == 0:
        log.warning("Topology has no bonds: cannot check for molecules split by "
                    "the periodic box.")
        return None

    def _cut_frames():
        """Frames with a bond cut by the box: long as stored (> bond_cut_nm) but
        short under the minimum-image convention. A bond that is long either way
        (a residue gap bridged by the topology, an odd CONECT record, a stretched
        disulfide) is not a periodic-boundary problem and is ignored here."""
        cut = np.zeros(len(traj), dtype=bool)
        for a in range(0, len(traj), chunk):
            x = traj.xyz[a:a + chunk]
            raw = np.linalg.norm(x[:, bonds[:, 0]] - x[:, bonds[:, 1]], axis=-1)
            long = raw > bond_cut_nm
            if long.any():
                cols = np.where(long.any(axis=0))[0]
                mi = md.compute_distances(traj[a:a + chunk], bonds[cols], periodic=True)
                cut[a:a + chunk] = (long[:, cols] & (mi <= bond_cut_nm)).any(axis=1)
        return cut

    def _place_chains(mols, apply, contact_nm=1.5):
        """Per frame, put each chain in the periodic image where it touches the
        chains already placed (largest first). A chain already in contact is
        never moved; one in contact in no neighbouring image is left alone.
        Returns the boolean mask of frames where a chain was (or would be) moved."""
        from scipy.spatial import cKDTree
        if len(mols) < 2:
            return np.zeros(len(traj), dtype=bool)
        order = sorted(range(len(mols)), key=lambda i: -len(mols[i]))
        idx = [np.array([at.index for at in m]) for m in mols]
        probe = []
        for m, ii in zip(mols, idx):
            ca = np.array([at.index for at in m if at.name == "CA"])
            probe.append(ca if ca.size else ii)
        offs = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
        moved = np.zeros(len(traj), dtype=bool)
        for f in range(len(traj)):
            B = traj.unitcell_vectors[f].astype(np.float64)
            group = traj.xyz[f, probe[order[0]]].astype(np.float64)
            for i in order[1:]:
                p = traj.xyz[f, probe[i]].astype(np.float64)
                tree = cKDTree(group)
                if tree.query(p, distance_upper_bound=contact_nm)[0].min() < contact_nm:
                    group = np.vstack([group, p]); continue
                base = -np.round((p.mean(0) - group.mean(0)) @ np.linalg.inv(B))
                best = None
                for o in offs + base:
                    sh = o @ B
                    d = tree.query(p + sh, distance_upper_bound=contact_nm)[0].min()
                    if d < contact_nm and (best is None or d < best[0]):
                        best = (d, sh)
                if best is not None and np.any(best[1] != 0):
                    moved[f] = True
                    if apply:
                        traj.xyz[f, idx[i]] += best[1].astype(np.float32)
                    p = p + best[1]
                group = np.vstack([group, p])
        return moved

    cut = _cut_frames()
    mols = [sorted(m, key=lambda at: at.index) for m in top.find_molecules()]
    if cut.any():
        traj.make_molecules_whole(inplace=True)
    sep = _place_chains(mols, apply=False)
    if not cut.any() and not sep.any():
        return None
    if sep.any():
        _place_chains(mols, apply=True)
    if _cut_frames().any():
        raise ValueError(
            "The protein is split by the periodic box and could not be re-joined "
            "(bonds still cut at %.1f nm). Make molecules whole before "
            "analysis, e.g. `gmx trjconv -pbc mol`." % bond_cut_nm)
    return {
        "repaired": True,
        "frames_with_cut_bonds": int(cut.sum()),
        "frames_with_separated_chains": int(sep.sum()),
        "n_molecules": len(mols),
        "n_frames": int(len(traj)),
    }


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
    pbc = repair_periodic_images(traj)
    if pbc:
        selection["pbc_repair"] = pbc
        log.debug("Re-joined protein split by the periodic box: %s", pbc)
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
