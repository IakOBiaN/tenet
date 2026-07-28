"""Registry of models for the build_matrix golden tests.

``build_matrix`` is pure, deterministic construction of the per-model Boltzmann
weight matrices (no SVD, no RG iteration), so a stored fingerprint of its output
pins the physics encoding in ``tenet/BuildTensors.py`` and catches accidental
changes in milliseconds.

We store a compact fingerprint per returned matrix rather than the full arrays:
some models produce huge tensors (six_leg_test is 16**6, Pentacene_model_2 is
3375x3375) that are impractical to keep as fixtures.

Each case fixes everything build_matrix actually reads: ``model``, ``coord``
(the lattice coordination number used to share mu/energy over neighbours),
``constant`` (gas constant R, or 1 for the lattice-model scripts), ``T`` and
``m_par``.  Params mirror the repository's entry scripts so the fingerprint
corresponds to a physically meaningful construction.

Every model in build_matrix has a case here.  ``six_leg_test`` is the only one
that is genuinely expensive to construct (a 16**6 tensor, ~12s and 134 MB), so
generate_golden marks it ``slow`` and it runs only with ``--runslow``.
"""
import numpy as np

import tenet.MainScripts as ms


def fingerprint(matrixes):
    """Compact, reorder-sensitive fingerprint of a list of numpy arrays."""
    out = []
    for m in matrixes:
        m = np.asarray(m, dtype=float)
        flat = m.ravel()
        # index weights make the checksum sensitive to element reordering,
        # not just to the multiset of values
        idx = np.arange(1, flat.size + 1, dtype=float) / max(flat.size, 1)
        out.append(
            {
                "shape": list(m.shape),
                "sum": float(np.sum(flat)),
                "abs_sum": float(np.sum(np.abs(flat))),
                "max": float(np.max(flat)) if flat.size else 0.0,
                "min": float(np.min(flat)) if flat.size else 0.0,
                "checksum": float(np.sum(flat * idx)),
            }
        )
    return out


def make_calc(case):
    calc = ms.CalcConfig()
    calc.model = case["model"]
    calc.coord = case.get("coord", 4)
    calc.constant = case.get("constant", 0.008314)
    # metParam only matters for some build_matrix branches; keep a stable default
    calc.metParam = case.get("metParam", 10)
    return calc


# Pentacene_model_2 takes 48 parameters: three chemical potentials followed by six
# blocks of pair energies.  Built the way Pentacene_model_2_tm.py builds them, so
# the case can be checked against the entry script by eye rather than by counting
# indices into a 48-element literal.
_PM2_E44 = [60.7, 48.8, 38.4]
_PM2_E22 = [199.5, 217.4, 197.8, 101.7, 90.3, 150.5, 15.0, 97.9, 82.1]
_PM2_E33 = [206.3, 295.6, 236.0, 95.7, 84.6, 98.1, 87.3, 109.3, 82.0]
_PM2_E24 = [106.3, 87.9, 59.9, 62.7, 15.0, 68.9]
_PM2_E34 = [164.9, 112.8, 75.1, 62.8, 69.4, 60.6]
_PM2_E23 = [180.2, 182.4, 205.3, 201.3, 91.7, 96.5, 89.5, 94.9, 89.2, 81.8, 91.3, 77.3]


def pentacene_2_m_par(mu):
    """m_par for Pentacene_model_2 at chemical potential ``mu``."""
    mu_2 = -(-526.5 - (-656.79))
    mu_3 = -(-601.73 - (-656.79))
    mu_4 = 0
    return (
        [mu_2 + mu, mu_3 + mu, mu_4 + mu]
        + _PM2_E44 + _PM2_E22 + _PM2_E33 + _PM2_E24 + _PM2_E34 + _PM2_E23
    )


def six_leg_m_par():
    """m_par for six_leg_test (the TPB + Cu model), mirroring multy_test.py."""
    eps = 4.8
    return [6.0, -19.0, -20, -40.0, -35.2 - eps, -45.6 - 3.0 * eps, eps]


# id -> case.  T and m_par mirror the entry scripts (a representative mu is fixed).
BUILD_CASES = {
    "langmuir":                 dict(model="langmuir", coord=4, constant=0.008314, T=120.0, m_par=[4.0, 4.0]),
    "langmuir_m":               dict(model="langmuir_m", coord=6, constant=0.008314, T=120.0, m_par=[10.0, 6.0, 2.0]),
    "binary":                   dict(model="binary", coord=4, constant=0.008314, T=100.0, m_par=[5.0, 10.0, 4.0, 6.0]),
    "ising":                    dict(model="ising", coord=4, constant=1.0, T=2.3, m_par=[0.0, 1.0]),
    "hard-hexagon":             dict(model="hard-hexagon", coord=6, constant=1.0, T=1.0, m_par=[2.0]),
    "TLAT":                     dict(model="TLAT", coord=6, constant=1.0, T=1.0, m_par=[0.0, -1.0, -1.0, -1.0]),
    "dimers":                   dict(model="dimers", coord=3, constant=0.008314, T=200.0, m_par=[10.0, 20.0]),
    "dimers_test":              dict(model="dimers_test", coord=4, constant=0.008314, T=200.0, m_par=[10.0]),
    "1NN":                      dict(model="1NN", coord=4, constant=1.0, T=1.0, m_par=[3.0]),
    "2NN":                      dict(model="2NN", coord=4, constant=1.0, T=1.0, m_par=[3.0]),
    "3NN":                      dict(model="3NN", coord=4, constant=1.0, T=1.0, m_par=[3.0]),
    "4NN":                      dict(model="4NN", coord=4, constant=1.0, T=1.0, m_par=[3.0]),
    "5NN":                      dict(model="5NN", coord=4, constant=1.0, T=1.0, m_par=[3.0]),
    "qstate":                   dict(model="qstate", coord=6, constant=1.0, T=0.05, m_par=[1.0, 1, 12, -1, 1.0]),
    "CHD_simple":               dict(model="CHD_simple", coord=4, constant=0.008314, T=300.0,
                                     m_par=[100.0, -43.0, 28.0, 26.0, 43.0, 0.0, 0.0, 0.0]),
    "CHD_complex":              dict(model="CHD_complex", coord=4, constant=0.008314, T=300.0,
                                     m_par=[100.0, -43.0, 24.0, 8.0, 53.0, 22.0, 29.0, 43.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "Pentacene_model_1_simple": dict(model="Pentacene_model_1_simple", coord=4, constant=0.008314, T=300.0,
                                     m_par=[50.0, 60.7, 48.8, 38.4]),
    "Pentacene_model_1_complex": dict(model="Pentacene_model_1_complex", coord=4, constant=0.008314, T=300.0,
                                     m_par=[-1.3, -40.0, 60.7, 48.8, 38.4, -1.2, 2.0, 1.0, -17.4, 3.0, 1.0, -15.1, 0.0, 0.0]),
    "Pentacene_model_2":        dict(model="Pentacene_model_2", coord=4, constant=0.008314, T=300.0,
                                     m_par=pentacene_2_m_par(500.0)),
    "six_leg_test":             dict(model="six_leg_test", coord=6, constant=0.008314, T=640.0,
                                     m_par=six_leg_m_par()),
    "Pentacene_model_3":        dict(model="Pentacene_model_3", coord=4, constant=0.008314, T=300.0,
                                     m_par=[88.7, 50.0, 31.9, 19.8, 5.0, 40.1, 101.4, 6.0, 60.7, -2.4, -17.4, -15.1, 60.7, 0.0, 0.0]),
    "1D_long-range":            dict(model="1D_long-range", coord=2, constant=1.0, T=0.1, m_par=[5.0, [1.0, 0.57735, 0.5], 0.0]),
    "2D_long-range":            dict(model="2D_long-range", coord=6, constant=1.0, T=0.1, m_par=[5.0, [1.0, 0.57735, 0.5], 0.0]),
    "2D_long-range_V":          dict(model="2D_long-range_V", coord=6, constant=1.0, T=0.8, m_par=[5.0, [1.2, 1.0]]),
}


def make_fast_calc(case):
    """Build a CalcConfig for a fast end-to-end engine case."""
    calc = ms.CalcConfig()
    calc.model = case.get("model", "langmuir")
    calc.method = case["method"]
    calc.lattice = case["lattice"]
    calc.gen_tensor = case.get("gen_tensor", "default")
    calc.metModification = case.get("metModification", "default")
    calc.metParam = case.get("metParam", 8)
    calc.constant = case.get("constant", 0.008314)
    calc.coord = case.get("coord", 4)
    calc.iterations = case.get("iterations", 12)
    calc.join_tensors = case.get("join_tensors", [1, 1])
    return calc


# Fast end-to-end engine smoke tests: one simulate() per case on the cheap
# langmuir ("mono") model (and one 1D long-range), exercising every RG method and
# lattice geometry at tiny chi with few iterations.  This is the fast tier that
# pins tenet/TensorNetworks.py; the etalon scripts (slow tier) are the physics
# validation.  simulate() returns ln Z per node, a basis-independent scalar.
FAST_CASES = {
    "trg_square":        dict(method="trg",   lattice="square",     gen_tensor="default",  metParam=8, T=120.0, m_par=[4.0, 4.0]),
    "btrg_square":       dict(method="btrg",  lattice="square",     gen_tensor="default",  metParam=8, T=120.0, m_par=[4.0, 4.0]),
    "hotrg_square":      dict(method="hotrg", lattice="square",     gen_tensor="default",  metParam=8, T=120.0, m_par=[4.0, 4.0]),
    "tm_square":         dict(method="tm",    lattice="square",     gen_tensor="default",  metParam=4, T=120.0, m_par=[4.0, 4.0]),
    "trg_triangular":    dict(method="trg",   lattice="triangular", gen_tensor="default",  metParam=8, T=120.0, m_par=[4.0, 4.0]),
    "trg_hexagonal":     dict(method="trg",   lattice="hexagonal",  gen_tensor="to_square", metParam=8, T=120.0, m_par=[4.0, 4.0]),
    "htn_fshl":          dict(method="htn",   lattice="FSHL",       metParam=1, iterations=8, constant=1.0, T=120.0, m_par=[4.0, 4.0]),
    "htn_diamond":       dict(method="htn",   lattice="diamond",    metParam=1, iterations=8, constant=1.0, T=120.0, m_par=[4.0, 4.0]),
    "tm_1d_long_range":  dict(method="tm",    lattice="square",     gen_tensor="default", model="1D_long-range",
                              metParam=1, constant=1.0, T=0.1, m_par=[5.0, [1.0, 0.57735, 0.5], 0.0]),
}
