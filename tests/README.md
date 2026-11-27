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

| `test_compare_sensitivity.py` | The comparison itself: platform-sized rounding is tolerated, and twelve kinds of real regression (one state assignment, a 1% plateau-RMSD change, a withheld channel gaining values, ...) are caught. |

**Two comparison modes.** By default the golden test is *cross-platform*:
file sets, keys, strings, verdicts, the trust contract, preflight diagnostics,
state assignments, transition matrix and RMSF must match (numbers at
`rtol=1e-6`); tICA coordinates may differ by `1e-4`; rank-normalised scores
(frame scores, residue scores) must keep their ordering (Spearman >= 0.999) with
no entry more than 5 rank steps away, and the same entries withheld. These
allowances were measured, not guessed: macOS ARM (numpy 2.4.6) against the Linux
goldens (numpy 2.2.6) moved tICA coordinates by at most 5.2e-6 and swapped
near-tied entries by at most 1 frame / 3 residue rank steps, with everything
the verdict depends on identical.

`SCIML_GOLDEN_STRICT=1 python -m pytest tests` compares every number at
`rtol=1e-6`: use it on the platform the goldens were written on (Linux x86-64,
numpy 2.2.6, scipy 1.15.3, scikit-learn 1.7.2, mdtraj 1.10.3).

**If `test_golden_pipeline.py` fails, an output changed.** If that is a bug, fix
it. If the change is intended and reviewed, regenerate and commit the goldens
with the change, saying why in the commit message:

    python tests/golden/update_golden.py
