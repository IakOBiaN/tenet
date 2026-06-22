"""Regression tests that re-run the repository's self-checking etalon scripts.

Each script listed below computes a parameter sweep and asserts its result
against a hard-coded scientific etalon (``assert difference < 0.5``).  Rather
than copy those long etalon arrays here, we run each script as a subprocess and
treat a zero exit code as a pass: the trusted reference numbers stay in one
place (the script itself).

Heavy runs (full scientific chi / hundreds of points) are marked ``slow`` and
only execute with ``pytest --runslow``.  The fast subset runs in seconds and is
meant for the per-change inner loop.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Every etalon script does a full scientific-parameter sweep where ``full()``
# evaluates ``simulate`` ~5x per point, so all of them are minutes-scale -> slow.
# Fast per-change coverage lives in the build_matrix golden tests, not here.
ETALON_SCRIPTS = [
    "mono_tm.py",
    "dimer_tm.py",
    "ising.py",
    "mono.py",
    "binary.py",
    "ising_hierarchical.py",
    "binary_hierarchical.py",
    "multy_test.py",
    "14CHD_Si_simple_tm.py",
    "14CHD_Si_complex_tm.py",
    "14CHD_Si_simple_trg.py",
    "14CHD_Si_complex_trg.py",
    "Pentacene_model_1_simple_tm.py",
    "Pentacene_model_1_complex_tm.py",
    "Pentacene_model_2_tm.py",
    "Pentacene_model_3_trg.py",
]


def _cases():
    return [
        pytest.param(script, marks=[pytest.mark.slow], id=script)
        for script in ETALON_SCRIPTS
    ]


@pytest.mark.parametrize("script", _cases())
def test_etalon_script(script):
    proc = subprocess.run(
        [sys.executable, "-u", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"{script} failed (exit {proc.returncode}).\n"
        f"--- stdout tail ---\n{proc.stdout[-2000:]}\n"
        f"--- stderr tail ---\n{proc.stderr[-2000:]}"
    )
