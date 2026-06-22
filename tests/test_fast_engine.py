"""Fast end-to-end engine smoke tests.

One ``simulate()`` per case on the cheap langmuir model (plus one 1D long-range),
covering every RG method and lattice geometry at tiny chi.  Compares ln Z against
the committed golden (tests/golden_fast_engine.json).  This pins the RG algorithms
in Scripts/TensorNetworks.py and runs in ~1s, so it is part of the fast tier.

Regenerate the golden with::

    python -m tests.generate_golden

Tolerance is loose-ish (1e-6) because the truncated SVD (scipy ARPACK ``svds``)
is only reproducible to a few ULP; a real engine regression shifts ln Z far more.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import Scripts.MainScripts as ms
from tests.build_cases import FAST_CASES, make_fast_calc

GOLDEN = json.loads((Path(__file__).resolve().parent / "golden_fast_engine.json").read_text())

RTOL = 1e-6
ATOL = 1e-6


@pytest.mark.parametrize("case_id", sorted(FAST_CASES))
def test_fast_engine_lnz(case_id):
    assert case_id in GOLDEN, f"no golden for {case_id}; run: python -m tests.generate_golden"
    case = FAST_CASES[case_id]
    calc = make_fast_calc(case)
    lnZ = ms.simulate(calc, case["T"], list(case["m_par"]))
    expected = GOLDEN[case_id]["lnZ"]
    assert np.isclose(lnZ, expected, rtol=RTOL, atol=ATOL), (
        f"{case_id}: ln Z changed {expected!r} -> {lnZ!r}"
    )
