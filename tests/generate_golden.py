"""Regenerate golden fixtures from the current code.

Run from the repo root:  python -m tests.generate_golden

Only run this when the current code is known-good (the slow etalon suite is
green); it overwrites the committed golden files:
  * golden_build_matrix.json  - build_matrix fingerprints (construction layer)
  * golden_fast_engine.json   - ln Z of fast end-to-end simulate() cases (engine)
"""
import json
import time
from pathlib import Path

import Scripts.BuildTensors as bt
import Scripts.MainScripts as ms
from tests.build_cases import (
    BUILD_CASES,
    FAST_CASES,
    fingerprint,
    make_calc,
    make_fast_calc,
)

HERE = Path(__file__).resolve().parent
BUILD_GOLDEN = HERE / "golden_build_matrix.json"
ENGINE_GOLDEN = HERE / "golden_fast_engine.json"
SLOW_CONSTRUCT_SECONDS = 2.0


def gen_build_matrix():
    golden = {}
    for case_id, case in BUILD_CASES.items():
        calc = make_calc(case)
        t0 = time.perf_counter()
        matrixes, first_norm = bt.build_matrix(calc, case["T"], list(case["m_par"]))
        dt = time.perf_counter() - t0
        golden[case_id] = {
            "time": round(dt, 4),
            "slow": dt > SLOW_CONSTRUCT_SECONDS,
            "first_norm": float(first_norm),
            "fingerprint": fingerprint(matrixes),
        }
        flag = " (slow)" if golden[case_id]["slow"] else ""
        shapes = [fp["shape"] for fp in golden[case_id]["fingerprint"]]
        print(f"  {case_id:28s} {dt:8.3f}s  shapes={shapes}{flag}")
    BUILD_GOLDEN.write_text(json.dumps(golden, indent=2, sort_keys=True))
    print(f"  -> {len(golden)} models -> {BUILD_GOLDEN.name}\n")


def gen_fast_engine():
    golden = {}
    for case_id, case in FAST_CASES.items():
        calc = make_fast_calc(case)
        t0 = time.perf_counter()
        lnZ = ms.simulate(calc, case["T"], list(case["m_par"]))
        dt = time.perf_counter() - t0
        golden[case_id] = {"lnZ": float(lnZ), "time": round(dt, 4)}
        print(f"  {case_id:20s} {dt:8.3f}s  lnZ={lnZ:.12g}")
    ENGINE_GOLDEN.write_text(json.dumps(golden, indent=2, sort_keys=True))
    print(f"  -> {len(golden)} cases -> {ENGINE_GOLDEN.name}\n")


def main():
    print("build_matrix fingerprints:")
    gen_build_matrix()
    print("fast engine ln Z:")
    gen_fast_engine()


if __name__ == "__main__":
    main()
