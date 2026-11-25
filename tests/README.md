# tests/

**Validation and testing suite for SciML-MD.**

## Test files

| File | What it validates |
|---|---|
| `test_integration.py` | End-to-end pipeline regression |
| `test_signals.py` | Rarity, transition surprise, local density unit tests |
| `test_reproducibility.py` | Deterministic output across runs |
| `test_scientific_validation.py` | Chapman-Kolmogorov, implied timescales |
| `test_statistical_validation.py` | Bootstrap confidence intervals |
| `test_dataset_validation.py` | Trajectory and topology integrity |
| `test_pipeline_edge_cases.py` | Short trajectories, disconnected states |
| `test_chapter9_evaluation.py` | Thesis Chapter 9 results validation |
| `run_all_validation.py` | Runs all tests, saves to TEST_RESULTS.md |

## Run all tests
```bash
python tests/run_all_validation.py
```

## Run individual test
```bash
pytest tests/test_signals.py -v
```

## Launch regression tests

| File | What it guarantees |
|---|---|
| `test_golden_pipeline.py` | The production pipeline on three real MD fixtures (`tests/fixtures/`: no box, orthorhombic, triclinic) reproduces `tests/golden/`. The goldens were checked byte-identical to the validated `d9d2a88` pipeline. |
| `test_launch_invariants.py` | Invariants with no stored reference: contact counts equal brute-force mdtraj distances (all box types, and pairs at the 8 A cutoff), `contact_flux` equals the original dense code, adding water/ions changes no result, atom-selection rules, mismatched topologies are rejected. |

Numbers are compared at `rtol=1e-6`; strings, verdicts, keys, file sets and
shapes must match exactly.

**If `test_golden_pipeline.py` fails, an output changed.** If that is a bug, fix
it. If the change is intended and reviewed, regenerate and commit the goldens
with the change, saying why in the commit message:

    python tests/golden/update_golden.py
