"""Proteins split by the periodic box must be re-joined before analysis, and
intact trajectories must never be touched.

Raw MD output can cut a protein at the box edge (atoms wrapped one by one) or
put the chains of a complex in different periodic images. Unrepaired, a split
ubiquitin trajectory read a 10.8 A plateau RMSD instead of 1.2 A and the
suitability verdict flipped from unsuitable to marginal.
"""
import numpy as np
import mdtraj as md
import pytest

from pipeline_compare import FIXTURES, ROOT

from features.compute_md_features import (
    compute_features, repair_periodic_images, select_analysis_atoms)
from msm.preflight import assess_suitability, detect_equilibration


def _wrap_atoms(t, shift=2.0):
    """Wrap every atom into the primary cell separately (cuts the protein)."""
    B = t.unitcell_vectors
    frac = np.einsum("fij,fjk->fik", t.xyz + shift, np.linalg.inv(B))
    frac -= np.floor(frac)
    return md.Trajectory(np.einsum("fik,fkj->fij", frac, B).astype(np.float32), t.topology,
                         time=t.time, unitcell_lengths=t.unitcell_lengths,
                         unitcell_angles=t.unitcell_angles)


def _features_from(t, tmp_path, tag):
    d = tmp_path / tag; d.mkdir()
    t[0].save_pdb(str(d / "topology.pdb")); t.save_dcd(str(d / "traj.dcd"))   # dcd: no rounding
    feats, traj, sel = compute_features(str(d / "topology.pdb"), str(d / "traj.dcd"),
                                        return_selection=True)
    return feats, traj, sel


@pytest.mark.parametrize("name", ["md_1ubq_ortho", "md_2pth_tric"])
def test_split_protein_is_rejoined(name, tmp_path):
    t = md.load(str(FIXTURES / name / "traj.xtc"), top=str(FIXTURES / name / "topology.pdb"))
    fw, tw, sw = _features_from(t, tmp_path, "intact")
    fs, ts, ss = _features_from(_wrap_atoms(t), tmp_path, "split")
    assert "pbc_repair" not in sw
    assert ss["pbc_repair"]["frames_with_cut_bonds"] == len(t)
    # md.rmsd works in float32: even frame 0 against itself reads ~8e-4 nm, and
    # shifting atoms by box vectors moves that noise by up to ~2e-4 nm (0.002 A).
    assert np.allclose(fs["rmsd"], fw["rmsd"], atol=1e-3)
    for k in ("rg", "phi_sin", "phi_cos", "psi_sin", "psi_cos"):
        assert np.allclose(fs[k], fw[k], atol=1e-4), k
    assert np.abs(fs["contacts"] - fw["contacts"]).max() <= 1          # float rounding at 8 A only
    ew, iw = detect_equilibration(tw); es, is_ = detect_equilibration(ts)
    assert ew == es and abs(iw["rmsd_plateau_ang"] - is_["rmsd_plateau_ang"]) < 0.01
    assert assess_suitability(tw)["verdict"] == assess_suitability(ts)["verdict"]


def _two_chain(n_frames=20, seed=0):
    top = md.load_topology(str(ROOT / "sample_data" / "8H0R" / "canonical_topology.pdb"))
    keep, _ = select_analysis_atoms(top)
    ref = md.load(str(ROOT / "sample_data" / "8H0R" / "canonical_topology.pdb")).atom_slice(keep)
    rng = np.random.default_rng(seed)
    x = ref.xyz[0] - ref.xyz[0].min(0) + 1.0
    L = (np.ptp(x, axis=0) + 2.0).astype(np.float32)
    xyz = (x[None] + rng.normal(0, 0.01, (n_frames,) + x.shape)).astype(np.float32)
    return md.Trajectory(xyz, ref.topology, unitcell_lengths=np.tile(L, (n_frames, 1)),
                         unitcell_angles=np.full((n_frames, 3), 90, np.float32))


def test_chain_in_wrong_image_is_moved_back():
    t = _two_chain()
    chain_b = np.array([a.index for a in t.topology.chain(1).atoms])
    jumped = t.slice(range(len(t)), copy=True)
    jumped.xyz[1::2][:, chain_b] += jumped.unitcell_vectors[1::2, 0][:, None, :]
    info = repair_periodic_images(jumped)
    assert info["frames_with_separated_chains"] == len(t) // 2
    assert np.allclose(jumped.xyz, t.xyz, atol=1e-5)


def test_intact_trajectories_are_untouched():
    for name in ("md_1ubq_ortho", "md_2pth_tric"):
        t = md.load(str(FIXTURES / name / "traj.xtc"), top=str(FIXTURES / name / "topology.pdb"))
        before = t.xyz.copy()
        assert repair_periodic_images(t) is None and np.array_equal(t.xyz, before)
    t = _two_chain()
    before = t.xyz.copy()
    assert repair_periodic_images(t) is None and np.array_equal(t.xyz, before)


def test_bound_chains_far_apart_in_tight_box_are_not_moved():
    """Chain centres more than half a box apart, chains touching: must stay."""
    t = _two_chain(n_frames=3)
    a = np.array([x.index for x in t.topology.chain(0).atoms])
    b = np.array([x.index for x in t.topology.chain(1).atoms])
    x = t.xyz.copy()
    x[:, a] -= x[:, a].mean(1, keepdims=True); x[:, b] -= x[:, b].mean(1, keepdims=True)
    x[:, b, 2] += (x[:, a, 2].max(1) - x[:, b, 2].min(1) + 0.35)[:, None]   # end to end
    x -= x.min(1, keepdims=True); x += 0.1
    L = (np.ptp(x.reshape(-1, 3), axis=0) + 0.25).astype(np.float32)
    t = md.Trajectory(x.astype(np.float32), t.topology, unitcell_lengths=np.tile(L, (3, 1)),
                      unitcell_angles=np.full((3, 3), 90, np.float32))
    off = np.abs(t.xyz[0, a].mean(0) - t.xyz[0, b].mean(0))
    assert (off > L / 2).any(), "test setup: centres must be more than half a box apart"
    before = t.xyz.copy()
    assert repair_periodic_images(t) is None and np.array_equal(t.xyz, before)
