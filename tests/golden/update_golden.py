#!/usr/bin/env python3
"""Regenerate the golden outputs used by tests/test_golden_pipeline.py.

Run ONLY when an output change is intended and has been reviewed:
    python tests/golden/update_golden.py
then commit tests/golden/ together with the change that caused it, and say
in the commit message why the outputs changed.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_compare import FIXTURES, GOLDEN, run_fixture  # noqa: E402

for d in sorted(p for p in FIXTURES.iterdir() if p.is_dir()):
    for sub in ("art", "res"):
        shutil.rmtree(GOLDEN / sub / d.name, ignore_errors=True)
    run_fixture(FIXTURES, d.name, GOLDEN)
    print("golden updated:", d.name)
