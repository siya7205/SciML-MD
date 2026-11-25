"""Invariants the pipeline must satisfy on any machine.

These need no stored reference outputs: each one compares the production code
with an independent definition of the right answer.
  * contact counting == brute-force mdtraj distances (all box types, and pairs
    sitting within float32 rounding of the 8 A cutoff)
  * contact_flux == the original dense implementation
  * adding water and ions to a trajectory does not change any result
  * atom selection: protein kept (incl. modified residues), ions named CA dropped
  * a topology that does not match the trajectory is always rejected
"""
import numpy as np
import mdtraj as md
import pytest

from pipeline_compare import FIXTURES, compare_outputs, run_fixture

from features.compute_md_features import (
    _count_ca_contacts, compute_features, select_analysis_atoms)
from scoring.residue_attribution import contact_flux


# ---------------------------------------------------------------- helpers
def _ca_topology(n):
    top = md.Topology(); ch = top.add_chain()
    for _ in range(n):
        top.add_atom("CA", md.element.carbon, top.add_residue("ALA", ch))
    return top


def _bruteforce_counts(traj, cutoff=0.8):
    ca = traj.topology.select("name CA")
    ii, jj = np.triu_indices(len(ca), k=1)
    pairs = np.stack([ca[ii], ca[jj]], axis=1)
    return (md.compute_distances(traj, pairs) < cutoff).sum(axis=1).astype(float)


def _dense_contact_flux(ca_xyz, anom, base, cutoff_nm=0.8):
    """The original O(n_res^2)-per-frame implementation (d9d2a88), as reference."""
    n_res = ca_xyz.shape[1]
    acc_a = np.zeros((n_res, n_res)); acc_b = np.zeros((n_res, n_res)); n_a = n_b = 0
    for f in range(len(ca_xyz)):
        blk = ca_xyz[f:f + 1]
        c = (np.linalg.norm(blk[:, :, None, :] - blk[:, None, :, :], axis=-1) < cutoff_nm)[0]
        if anom[f]: acc_a += c; n_a += 1
        if base[f]: acc_b += c; n_b += 1
    d = np.abs(acc_a / n_a - acc_b / n_b); np.fill_diagonal(d, 0.0)
    return d.sum(axis=1)


def _boundary_pairs(rng, n, decide):
    """float32 pairs at ~0.8 nm where float32 and float64 disagree on '< 0.8'."""
    out = []
    while len(out) < n:
        a = rng.uniform(0, 5, 3).astype(np.float32)
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        b = (a + u * (0.8 + rng.uniform(-3e-7, 3e-7))).astype(np.float32)
        if decide(a, b) != (np.linalg.norm(a.astype(float) - b.astype(float)) < 0.8):
            out.append([a, b])
    return np.array(out, np.float32)


# ---------------------------------------------------------------- contacts
@pytest.mark.parametrize("box", ["none", "ortho", "tiny_ortho", "triclinic"])
def test_contact_count_matches_bruteforce(box):
    rng = np.random.default_rng({"none": 1, "ortho": 2, "tiny_ortho": 3, "triclinic": 4}[box])
    for _ in range(8):
        R, F = int(rng.integers(3, 80)), int(rng.integers(2, 30))
        X = rng.normal(0, 1.0, (F, R, 3)).astype(np.float32)
        kw = {}
        if box != "none":
            L = {"ortho": 6.0, "tiny_ortho": 1.9, "triclinic": 6.0}[box]
            X = np.mod(X, L).astype(np.float32)
            kw = dict(unitcell_lengths=np.full((F, 3), L, np.float32),
                      unitcell_angles=np.full((F, 3), 70 if box == "triclinic" else 90, np.float32))
        t = md.Trajectory(X, _ca_topology(R), **kw)
        got = _count_ca_contacts(t, t.topology.select("name CA"), chunk=int(rng.choice([1, 7, 64])))
        assert np.array_equal(got, _bruteforce_counts(t))


def test_contact_count_exact_at_cutoff_boundary():
    X = _boundary_pairs(np.random.default_rng(5), 150, lambda a, b: np.linalg.norm(a - b) < 0.8)
    t = md.Trajectory(X, _ca_topology(2))
    assert np.array_equal(_count_ca_contacts(t, t.topology.select("name CA")), _bruteforce_counts(t))


def test_contact_flux_matches_dense_reference():
    rng = np.random.default_rng(6)
    for _ in range(10):
        R, F = int(rng.integers(3, 100)), int(rng.integers(4, 40))
        X = rng.normal(0, 0.5, (F, R, 3)).astype(np.float32)
        anom = rng.random(F) < 0.4; base = ~anom
        if anom.any() and base.any():
            assert np.array_equal(contact_flux(X, anom, base, chunk=int(rng.choice([1, 5, 200]))),
                                  _dense_contact_flux(X, anom, base))


def test_contact_flux_exact_at_cutoff_boundary():
    """Pairs the original counts as contacts although their float64 distance is
    > 0.8 nm, all in the anomalous ensemble, against a contact-free baseline.
    (Mixing both directions of disagreement lets errors cancel - this does not.)"""
    rng = np.random.default_rng(7); pairs = []
    while len(pairs) < 100:
        a = rng.uniform(0, 5, 3).astype(np.float32)
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        b = (a + u * (0.8 + rng.uniform(-3e-7, 3e-7))).astype(np.float32)
        blk = np.stack([a, b])[None]
        dense_contact = (np.linalg.norm(blk[:, :, None, :] - blk[:, None, :, :], axis=-1) < 0.8)[0, 0, 1]
        if dense_contact and np.linalg.norm(a.astype(float) - b.astype(float)) > 0.8:
            pairs.append([a, b])
    X = np.concatenate([np.array(pairs, np.float32), np.array([[[0, 0, 0], [5, 5, 5]]], np.float32)])
    anom = np.ones(len(X), bool); anom[-1] = False
    ref = _dense_contact_flux(X, anom, ~anom)
    assert np.array_equal(ref, [1.0, 1.0])                      # sanity: reference sees the contacts
    assert np.array_equal(contact_flux(X, anom, ~anom), ref)


# ---------------------------------------------------------------- water / atom selection
def _add_solvent(src_dir, dst_dir, n_water=400, n_ca_ions=2, seed=8):
    """Copy a protein trajectory and surround it with diffusing water + Ca2+ ions."""
    t = md.load(str(src_dir / "traj.xtc"), top=str(src_dir / "topology.pdb"))
    top = t.topology.copy(); ch = top.add_chain()
    for _ in range(n_water):
        r = top.add_residue("HOH", ch)
        top.add_atom("O", md.element.oxygen, r)
        top.add_atom("H1", md.element.hydrogen, r)
        top.add_atom("H2", md.element.hydrogen, r)
    for _ in range(n_ca_ions):                       # atom named "CA": the classic trap
        top.add_atom("CA", md.element.calcium, top.add_residue("CA", ch))
    rng = np.random.default_rng(seed); L = t.unitcell_lengths
    o = rng.uniform(0, 1, (t.n_frames, n_water, 1, 3)) * L[:, None, None, :]
    w = (o + np.array([[0, 0, 0], [0.0957, 0, 0], [-0.024, 0.0927, 0]])).reshape(t.n_frames, -1, 3)
    ions = rng.uniform(0, 1, (t.n_frames, n_ca_ions, 3)) * L[:, None, :]
    xyz = np.concatenate([t.xyz, w, ions], axis=1).astype(np.float32)
    s = md.Trajectory(xyz, top, time=t.time, unitcell_lengths=t.unitcell_lengths,
                      unitcell_angles=t.unitcell_angles)
    dst_dir.mkdir(parents=True)
    s[0].save_pdb(str(dst_dir / "topology.pdb")); s.save_xtc(str(dst_dir / "traj.xtc"))
    back = md.load(str(dst_dir / "traj.xtc"), top=str(dst_dir / "topology.pdb"))
    assert np.array_equal(back.xyz[:, :t.n_atoms], t.xyz), "protein coordinates must survive the round trip"


def test_water_and_ions_do_not_change_results(tmp_path):
    name = "md_1ubq_ortho"                                     # orthorhombic box fixture
    _add_solvent(FIXTURES / name, tmp_path / "solv" / name)
    run_fixture(FIXTURES, name, tmp_path / "out_protein")
    run_fixture(tmp_path / "solv", name, tmp_path / "out_solvated")
    errs = compare_outputs(tmp_path / "out_solvated", tmp_path / "out_protein", name,
                           ignore_keys={"atom_selection"})
    assert not errs, "adding solvent changed results:\n" + "\n".join(errs[:20])


def test_atom_selection_rules():
    top = md.Topology(); ch = top.add_chain()
    def res(name, atoms):
        r = top.add_residue(name, ch)
        for a in atoms:
            top.add_atom(a, md.element.get_by_symbol("Se" if a == "SE" else a[0]), r)
    res("ALA", ["N", "CA", "C", "O", "CB"])
    res("MSE", ["N", "CA", "C", "O", "CB", "CG", "SE", "CE"])      # selenomethionine
    res("XYZ", ["N", "CA", "C", "O"])                                # unknown amino acid
    res("CA", ["CA"]); res("NA", ["NA"]); res("HOH", ["O", "H1", "H2"]); res("NAG", ["C1", "N2"])
    idx, info = select_analysis_atoms(top)
    kept = {top.atom(i).residue.name for i in idx}
    assert kept == {"ALA", "MSE", "XYZ"}
    assert set(info["excluded"]) == {"CA", "NA", "HOH", "NAG"}
    assert len(top.subset(idx).select("name CA")) == 3              # no calcium posing as a residue


@pytest.mark.parametrize("name", sorted(p.name for p in FIXTURES.iterdir() if p.is_dir()))
def test_protein_only_fixtures_keep_every_atom(name):
    top = md.load_topology(str(FIXTURES / name / "topology.pdb"))
    idx, info = select_analysis_atoms(top)
    assert idx.size == top.n_atoms and not info["excluded"]


def test_topology_without_protein_is_rejected(tmp_path):
    top = md.Topology(); ch = top.add_chain()
    for _ in range(3):
        r = top.add_residue("HOH", ch); top.add_atom("O", md.element.oxygen, r)
    t = md.Trajectory(np.zeros((2, 3, 3), np.float32) + np.arange(3)[None, :, None], top)
    t[0].save_pdb(str(tmp_path / "w.pdb")); t.save_xtc(str(tmp_path / "w.xtc"))
    with pytest.raises(ValueError, match="No protein atoms"):
        compute_features(str(tmp_path / "w.pdb"), str(tmp_path / "w.xtc"))


def test_mismatched_topology_is_rejected(tmp_path):
    # solvated topology (protein-subset load path) paired with the wrong trajectory
    _add_solvent(FIXTURES / "md_1ubq_ortho", tmp_path / "a" / "x")
    _add_solvent(FIXTURES / "md_1ubq_ortho", tmp_path / "b" / "x", n_water=100)
    with pytest.raises(ValueError, match="mismatch"):
        compute_features(str(tmp_path / "a" / "x" / "topology.pdb"), str(tmp_path / "b" / "x" / "traj.xtc"))
    # protein-only path
    with pytest.raises(ValueError, match="mismatch"):
        compute_features(str(FIXTURES / "md_2pth_tric" / "topology.pdb"),
                         str(FIXTURES / "md_1vii_nobox" / "traj.xtc"))
