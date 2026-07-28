"""The engine must not write derived state back onto the caller's CalcConfig.

``simulate`` and ``build_matrix`` derive configuration while they run - the
coordination number for a triangular lattice, the node count for models that
merge sites, and for the 2D long-range models even the lattice itself.  Those
writes used to land on the caller's object, so reusing one ``CalcConfig`` across
runs carried state over: a triangular run left ``coord = 6`` behind and the next
square run on the same object silently used the wrong coordination number.

``thermodynamics`` calls ``simulate`` up to five times on one config, so this is
the ordinary path, not an exotic one.
"""
import tenet.MainScripts as ms

T = 120.0
M_PAR = [4.0, 4.0]


def _calc(lattice):
    calc = ms.CalcConfig()
    calc.model = "langmuir"
    calc.method = "trg"
    calc.lattice = lattice
    calc.metParam = 8
    calc.iterations = 12
    return calc


def test_simulate_leaves_the_config_untouched():
    calc = _calc("triangular")
    before = dict(vars(calc))
    ms.simulate(calc, T, M_PAR)
    assert vars(calc) == before


def test_square_run_is_unaffected_by_an_earlier_triangular_run():
    fresh = ms.simulate(_calc("square"), T, M_PAR)

    calc = _calc("triangular")
    ms.simulate(calc, T, M_PAR)
    calc.lattice = "square"

    # bit-exact: the truncated SVD is seeded deterministically
    assert ms.simulate(calc, T, M_PAR) == fresh


def test_default_join_tensors_is_not_shared_between_configs():
    first = ms.CalcConfig()
    first.join_tensors[0] = 99
    assert ms.CalcConfig().join_tensors == [1, 1]
