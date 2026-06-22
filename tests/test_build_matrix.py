"""Fast golden tests for build_matrix (the per-model Boltzmann-weight construction).

For every model we recompute build_matrix and compare a compact fingerprint of
its output against the committed golden (tests/golden_build_matrix.json).  This
is the fast tier: it runs on every change and pins the physics encoding in
Scripts/BuildTensors.py.  Regenerate the golden with::

    python -m tests.generate_golden

Models whose construction is slow (>2s) are marked ``slow`` from the golden and
only run with ``--runslow``.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import Scripts.BuildTensors as bt
from tests.build_cases import BUILD_CASES, fingerprint, make_calc

GOLDEN = json.loads((Path(__file__).resolve().parent / "golden_build_matrix.json").read_text())

RTOL = 1e-9
ATOL = 1e-12
_FIELDS = ("sum", "abs_sum", "max", "min", "checksum")


def _cases():
    cases = []
    for case_id in sorted(BUILD_CASES):
        slow = GOLDEN.get(case_id, {}).get("slow", False)
        marks = [pytest.mark.slow] if slow else []
        cases.append(pytest.param(case_id, marks=marks, id=case_id))
    return cases


@pytest.mark.parametrize("case_id", _cases())
def test_build_matrix_fingerprint(case_id):
    assert case_id in GOLDEN, f"no golden for {case_id}; run: python -m tests.generate_golden"
    expected = GOLDEN[case_id]

    case = BUILD_CASES[case_id]
    calc = make_calc(case)
    matrixes, first_norm = bt.build_matrix(calc, case["T"], list(case["m_par"]))
    actual = fingerprint(matrixes)

    assert len(actual) == len(expected["fingerprint"]), (
        f"{case_id}: matrix count changed "
        f"{len(expected['fingerprint'])} -> {len(actual)}"
    )
    assert np.isclose(first_norm, expected["first_norm"], rtol=RTOL, atol=ATOL), (
        f"{case_id}: first_norm changed {expected['first_norm']} -> {first_norm}"
    )

    for i, (got, exp) in enumerate(zip(actual, expected["fingerprint"])):
        assert got["shape"] == exp["shape"], (
            f"{case_id} matrix[{i}]: shape changed {exp['shape']} -> {got['shape']}"
        )
        for field in _FIELDS:
            assert np.isclose(got[field], exp[field], rtol=RTOL, atol=ATOL), (
                f"{case_id} matrix[{i}]: {field} changed "
                f"{exp[field]!r} -> {got[field]!r}"
            )
