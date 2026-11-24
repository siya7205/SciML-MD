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
"""
import numpy as np
import mdtraj as md



# -------------------------------------------------------------- #
# Feature cache helpers
# -------------------------------------------------------------- #
# Bump when ANY feature definition changes, so stale caches are never reused.
FEATURE_CODE_VERSION = "1"


def _file_sig(path):
    import os
    st = os.stat(path)
    return f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}"


def _cache_key(topology_path, trajectory_path, stride, reference_frame, n_frames):
    import hashlib
    raw = "||".join([FEATURE_CODE_VERSION, _file_sig(topology_path),
                     _file_sig(trajectory_path), str(stride),
                     str(reference_frame), str(n_frames)])
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_feature_cache(cache_path, key):
    import os
    if not os.path.exists(cache_path):
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as z:
            if str(z["__key__"]) != key:
                return None
            return {k: z[k] for k in z.files if k != "__key__"}
    except Exception:                                   # corrupt cache -> recompute
        return None


def _save_feature_cache(cache_path, key, features):
    import os
    try:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        tmp = str(cache_path) + ".tmp.npz"
        np.savez(tmp, __key__=np.array(key), **features)
        os.replace(tmp, cache_path)                    # atomic: no half-written cache
    except Exception:
        pass                                            # caching must never break a run


# -------------------------------------------------------------- #
# Function: compute_features
# -------------------------------------------------------------- #
def compute_features(topology_path, trajectory_path, stride=1, reference_frame=0,
                     cache_path=None):
    """
    Compute MD features from a trajectory.
    
    Args:
        topology_path: Path to topology file (PDB)
        trajectory_path: Path to trajectory file (XTC, DCD, etc.)
        stride: Load every stride-th frame
        reference_frame: Frame index for RMSD reference
        cache_path: Optional .npz path. If given and it holds features computed from
            the SAME files/stride/code version, they are reused instead of
            recomputed (the trajectory is still loaded, since callers need it).
            Otherwise features are computed and written there.
        
    Returns:
        features: Dictionary of feature arrays
        traj: MDTraj trajectory object
    """
    # Load trajectory against the (canonical) topology. Fail fast with a clear,
    # located message if the atom sets disagree — this is exactly where topology
    # drift historically surfaced as the opaque MDTraj error
    # "topology and trajectory files might not contain the same atoms".
    try:
        traj = md.load(trajectory_path, top=topology_path, stride=stride)
    except (ValueError, IOError) as exc:
        raise ValueError(
            f"Topology/trajectory mismatch: could not load '{trajectory_path}' "
            f"with topology '{topology_path}'. Feature extraction must use the "
            f"SAME canonical topology produced during simulation "
            f"(canonical_topology.pdb). Original error: {exc}"
        ) from exc

    # Explicit atom-count assertion (defensive; md.load also enforces equality).
    top_only = md.load(topology_path)
    if traj.n_atoms != top_only.n_atoms:
        raise ValueError(
            f"Topology drift detected: trajectory has {traj.n_atoms} atoms but "
            f"topology '{topology_path}' has {top_only.n_atoms}. These must match."
        )

    n_frames = len(traj)

    cache_key = None
    if cache_path is not None:
        cache_key = _cache_key(topology_path, trajectory_path, stride,
                               reference_frame, n_frames)
        cached = _load_feature_cache(cache_path, cache_key)
        if cached is not None:
            # md.rmsd centres traj.xyz IN PLACE, and the preflight / residue
            # scoring downstream read those centred coordinates. Reproduce that
            # side effect (cheap: one RMSD pass) so a cache hit is bit-identical
            # to a fresh computation.
            md.rmsd(traj, traj[reference_frame])
            return cached, traj

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
    
    if cache_path is not None:
        _save_feature_cache(cache_path, cache_key, features)

    return features, traj



# -------------------------------------------------------------- #
# Function: _count_ca_contacts
# -------------------------------------------------------------- #
def _count_ca_contacts(traj, ca_atoms, cutoff_nm=0.8, max_chunk_bytes=256e6,
                       n_threads=None):
    """Per-frame number of CA-CA pairs closer than `cutoff_nm`.

    Previously this materialised the full (n_frames x n_res*(n_res-1)/2)
    distance matrix just to threshold and sum it: O(F * R^2) time AND memory
    (9.4 GiB for 500 frames of a 3,172-residue protein). The count only needs
    a neighbour search, which is O(F * R) with a KD-tree.

    The KD-tree path is used where it is exactly equivalent to mdtraj's
    distances: no unit cell, or an orthorhombic one (minimum image). Triclinic
    cells fall back to the original mdtraj computation, run in memory-bounded
    chunks of frames so the result is unchanged but no longer unbounded.

    Frames are independent and the KD-tree work releases the GIL, so the
    KD-tree path runs on `n_threads` threads (default: up to 8 cores; serial
    for small inputs). Results are identical to the serial loop.
    """
    n_frames, n_res = len(traj), len(ca_atoms)
    counts = np.zeros(n_frames, dtype=float)
    ulen, uang = traj.unitcell_lengths, traj.unitcell_angles
    periodic = ulen is not None
    ortho = periodic and bool(np.allclose(uang, 90.0, atol=1e-3))

    if (not periodic) or ortho:
        try:
            from scipy.spatial import cKDTree
        except ImportError:                       # scipy absent -> chunked path
            cKDTree = None
        if cKDTree is not None:
            xyz = traj.xyz[:, ca_atoms, :]              # float32 view-copy; cast per frame

            def _one(f):
                pts = xyz[f].astype(np.float64)
                if ortho:
                    box = np.asarray(ulen[f], dtype=np.float64)
                    pts = np.mod(pts, box)
                    pts = np.where(pts >= box, 0.0, pts)   # guard mod rounding to L
                    tree = cKDTree(pts, boxsize=box)
                else:
                    tree = cKDTree(pts)
                # ordered pairs incl. self-matches -> unordered distinct pairs
                return (tree.count_neighbors(tree, cutoff_nm) - n_res) // 2

            if n_threads is None:
                import os
                n_threads = min(8, os.cpu_count() or 1)
            if n_threads > 1 and n_frames >= 32 and n_res >= 500:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=n_threads) as ex:
                    counts[:] = list(ex.map(_one, range(n_frames)))
            else:
                for f in range(n_frames):
                    counts[f] = _one(f)
            return counts

    # Fallback: original computation, chunked over frames to bound memory.
    ii, jj = np.triu_indices(n_res, k=1)
    pairs = np.stack([ca_atoms[ii], ca_atoms[jj]], axis=1)
    chunk = max(1, int(max_chunk_bytes // (4 * len(pairs))))
    for a in range(0, n_frames, chunk):
        d = md.compute_distances(traj[a:a + chunk], pairs)
        counts[a:a + chunk] = (d < cutoff_nm).sum(axis=1)
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
