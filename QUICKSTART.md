# QUICKSTART — SciML-MD

From a molecular dynamics trajectory to a trust-gated result in four steps.
Every command here is the one the pipeline actually uses; for what the numbers
mean, see [`PIPELINE.md`](PIPELINE.md), and for what may be claimed from them,
[`CLAIMS.md`](CLAIMS.md).

---

## 1. Install

Python 3.11 is recommended (prebuilt `deeptime` and `mdtraj` wheels exist for it).

```bash
conda create -n sciml python=3.11 -y
conda activate sciml
pip install -r requirements.txt
```

If `pip` fails to build `deeptime` or `mdtraj`, install those two from
conda-forge first and re-run `pip`:

```bash
conda install -c conda-forge deeptime mdtraj -y
pip install -r requirements.txt
```

Check the installation (about 15 seconds):

```bash
python -m pytest tests -q
```

## 2. Put your trajectory in place

One folder per system under `data/`:

```
data/
└── MY_PROTEIN/
    ├── topology.pdb    ← required (canonical_topology.pdb is used instead if present)
    └── traj.xtc        ← required (or traj.dcd, trajectory.xtc, trajectory.dcd)
```

The topology must describe exactly the atoms in the trajectory, in the same
order; a mismatch stops the run with an error. You do not need to clean the
trajectory first:

- **Solvent and ligands** — water, ions, ligands and glycans are excluded
  automatically; only protein atoms are analysed.
- **Periodic boundaries** — a protein cut by the box edge, or chains in
  different periodic images, is re-joined automatically (logged as a warning).

No trajectory yet? Section 5 shows how to simulate one.

## 3. Run the pipeline

```bash
python run_all_proteins.py                      # every system in data/
python run_all_proteins.py --protein MY_PROTEIN # one system
python run_all_proteins.py --jobs 4             # several systems in parallel
```

Results are written to `results/MY_PROTEIN/` and `artifacts/MY_PROTEIN/`. The
log ends with a per-stage timing table.

| Option | Default | Meaning |
|---|---|---|
| `--data_dir` | `data` | folder containing one sub-folder per system |
| `--results_dir`, `--artifacts_dir` | `results`, `artifacts` | output locations |
| `--protein` | all | process only this system |
| `--stride` | 1 | load every n-th frame |
| `--lag_tica`, `--dim_tica` | 10, 5 | tICA lag (frames) and dimensions |
| `--n_clusters` | 20 | KMeans microstates |
| `--lag_msm` | 10 | MSM lag (frames) |
| `--k_neighbors` | 10 | neighbours for the local-density channel |
| `--window` | 1 (off) | moving-median smoothing; keep at 1 for transient events |
| `--seed` | 42 | random seed |
| `--jobs` | 1 | systems processed in parallel (results identical) |

## 4. Read the verdict, then export for the viewer

**Start with `results/MY_PROTEIN/trust.json`.** Its `headline` says, in one
sentence, whether this trajectory can support a Markov state model. Channels the
trajectory cannot support are *withheld*, and the reason is stated. A short
trajectory is expected to be withheld: that is the pipeline working, not
failing. `preflight.json` holds the diagnostics behind the verdict
(equilibration trim, basin census, recurrence, plus any atom-selection or
periodic-repair record).

Build the viewer bundle — the only file a visualiser should read:

```bash
python tools/export_for_viewer.py --system MY_PROTEIN   # -> viewer/MY_PROTEIN.json
python tools/export_for_viewer.py --all                 # every system in results/
```

The bundle contains only what may be shown; everything withheld is absent and
explained under `omitted`. See [`HANDOFF.md`](HANDOFF.md) before building on it.

## 5. Optional: simulate a trajectory with OpenMM

```bash
pip install openmm
mkdir -p data/1UBQ && cp sample_data/1UBQ/topology_fixed.pdb data/1UBQ/topology.pdb
python scripts/generate_md_trajectories.py --data_dir data --steps 500000 --report_interval 500
```

This solvates the protein in TIP3P water, minimises it and runs NVT dynamics
(amber14, 2 fs steps): 500,000 steps is 1 ns and 1,000 frames. It writes
`canonical_topology.pdb` and `traj.xtc`, which step 3 picks up as they are. Try
`--steps 5000` first to see how fast your machine is.

**Expect a verdict of "cannot support a Markov state model" for any trajectory
of a few nanoseconds.** State populations converge only after microseconds
(`CLAIMS.md`); short runs are useful for testing the pipeline, not for results.

## 6. Where next

| File | Read it for |
|---|---|
| [`HANDOFF.md`](HANDOFF.md) | building an interface on the output |
| [`PIPELINE.md`](PIPELINE.md) | what every stage computes and what it is evidence of |
| [`CLAIMS.md`](CLAIMS.md) | what may and may not be claimed |
| [`INTEGRATION.md`](INTEGRATION.md) | consuming the raw outputs directly |
| [`tests/README.md`](tests/README.md) | the regression tests and how to update them |
