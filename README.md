

# Ensemble-Anomaly-Maps

**Applicability-domain screening and descriptive analysis for Markov state models of protein trajectories**

Ensemble-Anomaly-Maps is a computational biology pipeline that downloads protein structures from the RCSB PDB, generates short MD trajectories, and applies an unsupervised pipeline (tICA → KMeans → Markov State Model → multi-channel scoring) across an entire dataset of proteins — automatically, in a single command.

> ### ⚠️ Scientific status — read before citing any output
>
> This pipeline was originally documented as a **dynamic hotspot detector**.
> That claim has been **withdrawn**. Four independent experimental designs
> (validation Phases 3, 5, 6 and 7) agree that at the sampling used here
> (≤100 ns), the multi-signal score does not detect conformational anomalies
> better than a four-line `abs(diff)` baseline, and the MSM stationary
> distribution π does not generalise across replicates of the same protein.
>
> **What is supported:** the MSM applicability-domain screen (45% of ATLAS
> trajectories fail it), the per-residue flexibility profile (externally
> verified against published ATLAS values, r = 0.892, slope = 0.966), and the
> non-redundancy of the three scoring channels.
>
> **Building on this pipeline?** Start at [`HANDOFF.md`](HANDOFF.md), then
> [`PIPELINE.md`](PIPELINE.md) for what each stage computes.
>
> **[`CLAIMS.md`](CLAIMS.md) is the authoritative list of what this software may
> and may not be said to do.** Full evidence:
> [`validation/VALIDATION_REPORT.md`](validation/VALIDATION_REPORT.md).
> Sections of this README written before 2026-08-30 still contain withdrawn
> language and are being revised; `CLAIMS.md` overrides them wherever they conflict.

---

## Table of Contents

1. [What This Pipeline Does](#1-what-this-pipeline-does)
2. [Prerequisites](#2-prerequisites)
3. [Installation](#3-installation)
4. [Quick Start — Full Pipeline in 3 Commands](#4-quick-start--full-pipeline-in-3-commands)
5. [Step-by-Step Guide](#5-step-by-step-guide)
   - [Step 1 — Download Protein Structures](#step-1--download-protein-structures)
   - [Step 2 — Generate MD Trajectories](#step-2--generate-md-trajectories)
   - [Step 3 — Run the ML Pipeline](#step-3--run-the-ml-pipeline)
6. [Using Your Own Data](#6-using-your-own-data)
7. [Output Files Explained](#7-output-files-explained)
8. [All Command-Line Options](#8-all-command-line-options)
9. [Repository Structure](#9-repository-structure)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. What This Pipeline Does

```
RCSB PDB API
    │
    ▼
scripts/download_pdb_dataset.py
    │  Downloads ≥50 small proteins  (<150 residues, <3 Å resolution)
    │  Saves: data/{PDB_ID}/topology.pdb
    ▼
scripts/generate_md_trajectories.py
    │  For each protein: add H₂O → minimize → NVT MD (50k steps)
    │  Saves: data/{PDB_ID}/traj.xtc
    ▼
run_all_proteins.py
    │
    ├─ Atom selection      (protein only; water, ions, ligands excluded;
    │                       proteins split by the periodic box re-joined)
    ├─ Equilibration trim  (discard startup relaxation)
    ├─ Diagnostics         (basin census, recurrence — before any model is fitted)
    ├─ Feature extraction  (RMSD, Rg, contacts, φ/ψ dihedrals)
    ├─ tICA → KMeans → MSM (slow coordinates, microstates, kinetic model)
    ├─ Channels            (rarity + transition surprise + local density)
    └─ TRUST CONTRACT      (withholds what this trajectory cannot support)
         │
         ├─ results/{PDB_ID}/trust.json      ← read this first
         ├─ results/{PDB_ID}/preflight.json
         ├─ results/{PDB_ID}/frame_scores_dynamic.csv
         ├─ results/{PDB_ID}/residue_scores_*.json
         └─ artifacts/{PDB_ID}/*.npy
    ▼
tools/export_for_viewer.py
       Saves: viewer/{PDB_ID}.json  (only what may be shown; see HANDOFF.md)
```

---

## 2. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.9 | 3.10 or 3.11 recommended |
| conda | any | Strongly recommended for OpenMM |
| Git | any | To clone the repo |
| Internet access | — | For RCSB downloads |
| RAM | ≥ 8 GB | 16 GB preferred for larger proteins |
| Disk | ≥ 5 GB free | For protein structures + trajectories |

> **Why conda?** OpenMM is most reliably installed via `conda-forge`. It can be installed with pip but may require a working C compiler.

---

## 3. Installation

### 3.1 Clone the repository

```bash
git clone https://github.com/siya7205/ensemble-anomaly-maps.git
cd ensemble-anomaly-maps
```

### 3.2 Create a Python environment

**Option A — conda (recommended)**

```bash
conda create -n anomaly-maps python=3.11 -y
conda activate anomaly-maps
```

**Option B — venv (if you prefer pip-only)**

```bash
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

### 3.3 Install ML and MD dependencies

```bash
# Core ML libraries (tICA, MSM, clustering, anomaly scoring)
pip install -r requirements.txt
```

### 3.4 Install OpenMM for trajectory generation

OpenMM is needed only for **Step 2** (generating trajectories). Skip if you already have `.xtc`/`.dcd` files.

```bash
# Recommended via conda-forge:
conda install -c conda-forge openmm -y

# Alternative via pip (requires a C++ toolchain on some systems):
pip install openmm
```

### 3.5 Verify your installation

```bash
python -c "import deeptime; print('deeptime OK')"
python -c "import mdtraj; print('mdtraj OK')"
python -c "import openmm; print('openmm OK')"   # only if you installed it
```

---

## 4. Quick Start — Full Pipeline in 3 Commands

Run these three commands in order from the repository root. Each step builds on the previous one.

```bash
# 1. Download ≥50 small proteins from RCSB PDB
python scripts/download_pdb_dataset.py

# 2. Generate a short MD trajectory for each protein (needs OpenMM)
python scripts/generate_md_trajectories.py

# 3. Run the full ML pipeline on every protein
python run_all_proteins.py

# 4. Build the viewer bundles
python tools/export_for_viewer.py --all
```

Results land in `results/` and `artifacts/`, viewer bundles in `viewer/`. Start
with each system's `results/{PDB_ID}/trust.json`: its headline says whether the
trajectory can support a Markov state model at all. For your own trajectories,
see [`QUICKSTART.md`](QUICKSTART.md).

---

## 5. Step-by-Step Guide

### Step 1 — Download Protein Structures

```bash
python scripts/download_pdb_dataset.py
```

**What it does:**
- Queries the RCSB PDB Search API with these filters:
  - Polymer entity type = Protein
  - Sequence length < 150 residues  
  - Single protein chain (no complexes)
  - Crystal resolution ≤ 3.0 Å
- Downloads PDB files and saves them as `data/{PDB_ID}/topology.pdb`
- Stops after downloading 50 proteins (configurable)

**What you'll see:**

```
============================================================
RCSB PDB Dataset Downloader
============================================================
[search] Querying RCSB for proteins: residues < 150, resolution ≤ 3.0 Å ...
[search] Found 3847 total matches, retrieved 200 PDB IDs.

[download] Attempting up to 200 proteins ...
  [ok]    1UBQ → data/1UBQ/topology.pdb
  [ok]    1L2Y → data/1L2Y/topology.pdb
  ...
  [progress] downloaded=10, skipped=0, failed=0
  ...
[done] Target of 50 downloads reached.

[summary] downloaded=50, skipped=0, failed=2
```

**Output structure after Step 1:**

```
data/
├── 1UBQ/
│   └── topology.pdb
├── 1L2Y/
│   └── topology.pdb
└── ... (50+ directories)
```

---

### Step 2 — Generate MD Trajectories

```bash
python scripts/generate_md_trajectories.py
```

**What it does** for each `data/{PDB_ID}/topology.pdb`:
1. Loads the PDB structure with OpenMM
2. Adds missing hydrogen atoms
3. Solvates the protein in a TIP3P explicit water box (1 nm padding)
4. Minimizes potential energy
5. Runs 50,000 NVT integration steps at 300 K (Langevin thermostat, 2 fs time step)
6. Saves the trajectory as `data/{PDB_ID}/traj.xtc` (via mdtraj) or `traj.dcd` (fallback)

> **Time estimate:** ~2–10 minutes per protein on a laptop CPU, depending on protein size. Run overnight for large batches.

**What you'll see:**

```
11:00:01 [INFO] Found 50 protein directories.
11:00:01 [INFO] [1UBQ] Loading structure: data/1UBQ/topology.pdb
11:00:02 [INFO] [1UBQ] Adding hydrogens and solvating...
11:00:05 [INFO] [1UBQ] System: 8247 atoms after solvation.
11:00:05 [INFO] [1UBQ] Minimizing energy...
11:00:08 [INFO] [1UBQ] Running 50000 steps...
...
11:08:23 [INFO] [1UBQ] Saved trajectory: data/1UBQ/traj.xtc
```

**Proteins that already have a trajectory are automatically skipped.**

---

### Step 3 — Run the ML Pipeline

```bash
python run_all_proteins.py
```

**What it does** for each protein directory that has both `topology.pdb` and a trajectory:

| Sub-step | Library | What happens |
|---|---|---|
| Feature extraction | mdtraj | Computes RMSD, Rg, native contacts, φ/ψ dihedrals per frame |
| tICA | deeptime | Projects features onto slow collective-motion coordinates |
| KMeans clustering | deeptime | Groups frames into 20 discrete conformational states |
| MSM | deeptime | Builds a reversible Markov State Model over the states |
| Anomaly scoring | scoring/anomaly_v2.py | Fuses rarity + transition surprise + local density signals |

**What you'll see:**

```
11:10:00 [INFO] ════════════════════════════════════════════════════
11:10:00 [INFO] Batch ML Pipeline — Ensemble Anomaly Maps
11:10:00 [INFO] ════════════════════════════════════════════════════
11:10:00 [INFO] Found 50 protein dataset(s) to process.

11:10:00 [INFO] ────────────────────────────────────────────────────
11:10:00 [INFO] [1/50] Processing: 1UBQ
11:10:00 [INFO] ────────────────────────────────────────────────────
11:10:00 [INFO] [1UBQ] ── Step 1/5: Feature extraction
11:10:01 [INFO] [1UBQ]   100 frames × 8 features
11:10:01 [INFO] [1UBQ] ── Step 2/5: tICA (lag=10, dim=5)
11:10:01 [INFO] [1UBQ] ── Step 3/5: Clustering (n_clusters=20)
11:10:02 [INFO] [1UBQ] ── Step 4/5: MSM (lag=10)
11:10:02 [INFO] [1UBQ] ── Step 5/5: Anomaly scoring
11:10:02 [INFO] [1UBQ] ✓ Pipeline complete. Mean score: 47.3

...

11:45:00 [INFO] ════════════════════════════════════════════════════
11:45:00 [INFO] Batch run complete: 49 succeeded, 1 failed.
11:45:00 [INFO] Artifacts → artifacts/
11:45:00 [INFO] Results   → results/
```

---

## 6. Using Your Own Data

If you already have an MD trajectory and don't need Steps 1–2, you can run the pipeline directly on your own data.


Place your data like this:

```
data/
├── MY_PROTEIN/
│   ├── topology.pdb      ← required
│   └── traj.xtc          ← required (or traj.dcd / trajectory.xtc / trajectory.dcd)
└── ANOTHER_PROTEIN/
    ├── topology.pdb
    └── trajectory.xtc
```

Then run:

```bash
python run_all_proteins.py
```

If a `canonical_topology.pdb` is present it is used instead of `topology.pdb`.
The topology must list exactly the atoms of the trajectory, in order; a
mismatch stops the run with an error. Water, ions, ligands and glycans are
excluded automatically (only protein atoms are analysed), and a protein cut by
the periodic box is re-joined automatically, so trajectories do not need to be
cleaned first.

---

## 7. Output Files Explained

After a complete run of all three pipeline steps:

```
ensemble-anomaly-maps/
├── data/                          ← created by Steps 1 & 2
│   ├── 1UBQ/
│   │   ├── topology.pdb           (downloaded PDB structure)
│   │   └── traj.xtc               (generated MD trajectory)
│   └── ...
│
├── artifacts/                     ← created by Step 3
│   ├── 1UBQ/
│   │   ├── tica_coords.npy        tICA-projected coordinates   (T × dim)
│   │   ├── dtraj.npy              Discrete state trajectory    (T,)
│   │   ├── P.npy                  MSM transition matrix        (n_states × n_states)
│   │   └── pi.npy                 Stationary distribution      (n_states,)
│   └── ...
│
├── results/                       ← created by Step 3
│   ├── 1UBQ/
│   │   ├── trust.json                    What may be shown, and why the rest is withheld  ← read first
│   │   ├── preflight.json                Diagnostics behind the verdict (equilibration, basin census,
│   │   │                                 recurrence; atom-selection / periodic-repair records)
│   │   ├── frame_scores_dynamic.csv      Per-frame scores [0–100]; withheld channels are NaN
│   │   ├── residue_scores_dynamic.json   Per-residue scores
│   │   ├── residue_scores_kinetic.json   RMSF-orthogonal residue attribution
│   │   ├── residue_scores_rmsf.json      Per-residue flexibility (RMSF)
│   │   └── residue_attribution_diagnostics.json
│   └── ...
│
└── viewer/                        ← created by tools/export_for_viewer.py
    └── 1UBQ.json                  The only file a visualiser should read (HANDOFF.md)
```

### frame_scores_dynamic.csv

```
frame,score_dynamic,component_rarity,component_transition_surprise,component_local_density
0,42.3,38.1,45.0,43.9
1,55.1,60.2,52.3,52.8
...
```

| Column | Description |
|---|---|
| `frame` | Frame index (0-based) |
| `score_dynamic` | Fused multi-channel score, range [0, 100]. **Descriptive only** — see CLAIMS.md W-2 |
| `component_rarity` | Contribution from state rarity (pi). **Unresolved at <=100 ns** — see CLAIMS.md W-3 |
| `component_transition_surprise` | Contribution from transition surprise. **Carries no timing information** — see CLAIMS.md W-4 |
| `component_local_density` | Contribution from local density signal |

### residue_scores_dynamic.json

```json
{
  "MET1": 34.21,
  "GLN2": 41.85,
  "ILE3": 67.43,
  ...
}
```

Each key is a residue identifier. Values are scores in [0, 100].

> **Interpretation warning (CLAIMS.md W-1, W-5).** These are *descriptive* scores.
> A high value does NOT mean the residue is a functional hotspot, an allosteric
> node, or a hinge — that claim was withdrawn on 2026-08-30 and the residue-level
> correspondence to functional sites is an open question (CLAIMS.md O-1), not a
> result. Do not rank, threshold, or publish these as detections.

---

## 8. All Command-Line Options

### `scripts/download_pdb_dataset.py`

```
python scripts/download_pdb_dataset.py [OPTIONS]

Options:
  --data_dir DIR          Root data directory (default: data)
  --max_residues INT      Max residues per protein (default: 150)
  --max_resolution FLOAT  Max crystal resolution in Å (default: 3.0)
  --target INT            Stop after this many downloads (default: 50)
  --rows INT              Candidates requested from RCSB API (default: 200)
  --delay FLOAT           Pause between downloads in seconds (default: 0.5)
```

**Examples:**

```bash
# Download 100 proteins instead of 50
python scripts/download_pdb_dataset.py --target 100

# Be more selective: only very small, very high-resolution structures
python scripts/download_pdb_dataset.py --max_residues 80 --max_resolution 2.0

# Use a different output directory
python scripts/download_pdb_dataset.py --data_dir my_proteins
```

---

### `scripts/generate_md_trajectories.py`

```
python scripts/generate_md_trajectories.py [OPTIONS]

Options:
  --data_dir DIR          Root data directory (default: data)
  --steps INT             MD integration steps (default: 50000)
  --step_size FLOAT       Time step in picoseconds (default: 0.002)
  --temperature FLOAT     Temperature in Kelvin (default: 300.0)
  --padding FLOAT         Water box padding in nm (default: 1.0)
  --report_interval INT   Frames written every N steps (default: 500)
  --no_xtc                Keep DCD format even if mdtraj is available
  --overwrite             Re-generate trajectory even if one exists
```

**Examples:**

```bash
# Longer simulation (100k steps)
python scripts/generate_md_trajectories.py --steps 100000

# Higher temperature
python scripts/generate_md_trajectories.py --temperature 320

# Re-run everything from scratch (overwrite existing trajectories)
python scripts/generate_md_trajectories.py --overwrite
```

---

### `run_all_proteins.py`

```
python run_all_proteins.py [OPTIONS]

Options:
  --data_dir DIR          Root data directory (default: data)
  --artifacts_dir DIR     Numpy output directory (default: artifacts)
  --results_dir DIR       CSV/JSON output directory (default: results)
  --stride INT            Load every Nth frame (default: 1)
  --lag_tica INT          tICA lag time in frames (default: 10)
  --dim_tica INT          Number of tICA components (default: 5)
  --n_clusters INT        KMeans clusters (default: 20)
  --lag_msm INT           MSM lag time in frames (default: 10)
  --k_neighbors INT       k for local density signal (default: 10)
  --window INT            Moving-median smoothing window (default: 1 = off; smoothing
                          suppresses single-frame events, use >= 3 only for sustained states)
  --seed INT              Random seed (default: 42)
  --jobs INT              Proteins processed in parallel (default: 1; results identical)
  --protein PDB_ID        Process only this protein (for debugging)
```

**Examples:**

```bash
# Run with faster/coarser settings (every 2nd frame, 10 clusters)
python run_all_proteins.py --stride 2 --n_clusters 10

# Run only one protein to test
python run_all_proteins.py --protein 1UBQ

# Use different output directories
python run_all_proteins.py --artifacts_dir my_artifacts --results_dir my_results

# More tICA components for complex proteins
python run_all_proteins.py --dim_tica 8 --lag_tica 20
```

---

## 9. Repository Structure

```
SciML-MD/
│
├── run_all_proteins.py          ← The pipeline (Step 3)
├── batch_runner.py              ← Download → simulate → pipeline, end to end
│
├── scripts/
│   ├── download_pdb_dataset.py  ← RCSB structure downloader (Step 1)
│   └── generate_md_trajectories.py  ← OpenMM trajectory generator (Step 2)
│
├── features/
│   └── compute_md_features.py   ← Atom selection, periodic repair, RMSD, Rg, contacts, φ/ψ
│
├── msm/
│   ├── preflight.py             ← Equilibration, basin census, recurrence, estimability
│   ├── trust.py                 ← The trust contract
│   ├── bootstrap_msm.py         ← Bootstrap uncertainty (see its warning)
│   ├── select_lag_and_dim.py    ← VAMP-2 parameter selection
│   ├── soft_states.py           ← HMM soft state assignments
│   └── validation.py            ← Chapman-Kolmogorov tests
│
├── scoring/
│   ├── anomaly_v2.py            ← Scoring channels and fusion
│   ├── residue_attribution.py   ← RMSF-orthogonal residue attribution
│   └── signals.py               ← RMSF, tICA importance, density
│
├── tools/
│   └── export_for_viewer.py     ← The viewer bundle (Step 4)
│
├── tests/                       ← Regression tests (pytest); fixtures/ and golden/
├── validation/                  ← Validation campaign and VALIDATION_REPORT.md
├── viewer/                      ← Worked example bundles
├── configs/pipeline.yaml        ← Default hyperparameters
├── data/ artifacts/ results/    ← Created at runtime
└── requirements.txt
```

---

## 10. Troubleshooting

### `ModuleNotFoundError: No module named 'deeptime'`

```bash
pip install deeptime>=0.4.0
```

### `ModuleNotFoundError: No module named 'mdtraj'`

```bash
pip install mdtraj
```

### `ModuleNotFoundError: No module named 'openmm'`

```bash
# Recommended (most reliable):
conda install -c conda-forge openmm -y

# Alternative:
pip install openmm
```

### RCSB download returns 0 results

The RCSB Search API is occasionally unavailable. Wait a few minutes and retry. Check your internet connection:

```bash
curl -s "https://search.rcsb.org/rcsbsearch/v2/query" -o /dev/null -w "%{http_code}\n"
# Should print 200 or 422 (both mean the server is reachable)
```

### `Too few frames` warning for a protein

The generated trajectory has fewer than 10 frames. This can happen if the simulation crashed. Re-run Step 2 with `--overwrite`:

```bash
python scripts/generate_md_trajectories.py --overwrite
```

### MSM building fails for a protein

The trajectory may be too short for the requested lag time. Try smaller lag values:

```bash
python run_all_proteins.py --lag_tica 5 --lag_msm 5 --n_clusters 10
```

### Running on a single protein to debug

```bash
# Step 3 only, for one protein:
python run_all_proteins.py --protein 1UBQ
```

### OpenMM is slow on CPU

OpenMM automatically uses a GPU (CUDA or OpenCL) if one is available. On CPU-only machines, reduce step count for faster testing:

```bash
python scripts/generate_md_trajectories.py --steps 5000 --report_interval 100
```

---

## Additional Documentation

| File | Contents |
|---|---|
| [QUICKSTART.md](QUICKSTART.md) | Your own trajectory, from install to viewer bundle |
| [HANDOFF.md](HANDOFF.md) | Building an interface on the output |
| [PIPELINE.md](PIPELINE.md) | What every stage computes and what it is evidence of |
| [CLAIMS.md](CLAIMS.md) | What may and may not be claimed |
| [INTEGRATION.md](INTEGRATION.md) | Consuming the raw outputs |
| [validation/VALIDATION_REPORT.md](validation/VALIDATION_REPORT.md) | The evidence |
| [tests/README.md](tests/README.md) | Regression tests |
