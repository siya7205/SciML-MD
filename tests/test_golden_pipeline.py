"""End-to-end regression test: the production pipeline on three real MD
fixtures (no box / orthorhombic / triclinic) must reproduce tests/golden/.

A failure means an output changed. If the change is intended, review it and run
`python tests/golden/update_golden.py`; otherwise it is a regression.
"""
import pytest

from pipeline_compare import FIXTURES, GOLDEN, compare_outputs, run_fixture

NAMES = sorted(p.name for p in FIXTURES.iterdir() if p.is_dir())


@pytest.mark.parametrize("name", NAMES)
def test_pipeline_matches_golden(name, tmp_path):
    run_fixture(FIXTURES, name, tmp_path)
    errs = compare_outputs(tmp_path, GOLDEN, name)
    assert not errs, f"{name}: {len(errs)} difference(s) from golden:\n" + "\n".join(errs[:20])
