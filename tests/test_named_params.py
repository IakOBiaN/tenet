"""Models declare their parameters, so callers can pass them by name.

``m_par`` used to be a bare list of up to ten floats whose meaning changed from
model to model, and a misplaced entry produced a plausible wrong answer rather
than an error.  Each model now declares its parameter names in ``@register``,
and a caller may pass either a list (as the entry scripts do) or a dict.
"""
import ast
import inspect
import re

import pytest

import tenet.MainScripts as ms
import tenet.models as mdl


def _body_source(build):
    """Source of a model builder with its decorators, docstring and comments gone."""
    tree = ast.parse(inspect.getsource(build))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    stmts = fn.body
    if isinstance(stmts[0], ast.Expr) and isinstance(stmts[0].value, ast.Constant):
        stmts = stmts[1:]
    src = "\n".join(ast.unparse(s) for s in stmts)
    return "\n".join(line.split("#")[0] for line in src.splitlines())


@pytest.mark.parametrize("model", sorted(mdl.MODELS))
def test_declaration_covers_every_index_the_body_reads(model):
    """A spec that drifts shorter than the code silently loses a parameter."""
    names = mdl.get_params(model)
    used = [int(i) for i in re.findall(r"m_par\[(\d+)\]", _body_source(mdl.MODELS[model]))]
    if used:
        assert max(used) < len(names), (
            f"{model} reads m_par[{max(used)}] but declares only {len(names)} parameters"
        )


@pytest.mark.parametrize("model", sorted(mdl.MODELS))
def test_declared_names_are_unique(model):
    names = mdl.get_params(model)
    assert len(set(names)) == len(names), f"{model} declares a duplicate parameter name"


def test_named_and_positional_agree():
    calc = ms.CalcConfig()
    calc.model = "langmuir"
    calc.metParam = 8
    calc.iterations = 12
    positional = ms.simulate(calc, 120.0, [4.0, 4.0])
    named = ms.simulate(calc, 120.0, {"mu": 4.0, "eps": 4.0})
    assert named == positional


def test_unknown_parameter_is_rejected():
    with pytest.raises(ValueError) as excinfo:
        mdl.make_params("langmuir", {"mu": 4.0, "epsilon": 4.0})
    message = str(excinfo.value)
    assert "'epsilon'" in message
    # the message has to say what the model does accept, or it is not actionable
    assert "mu" in message and "eps" in message


def test_missing_names_default_to_zero():
    params = mdl.make_params("langmuir", {"mu": 4.0})
    assert params.mu == 4.0
    assert params.eps == 0.0


def test_short_list_is_padded_and_long_list_truncated():
    """Entry scripts habitually pass trailing zeros past what the model reads."""
    assert mdl.make_params("langmuir", [4.0]).eps == 0.0
    padded = mdl.make_params("langmuir", [4.0, 5.0, 0, 0, 0, 0])
    assert (padded.mu, padded.eps) == (4.0, 5.0)
    assert len(padded) == 2


def test_positional_and_named_access_agree():
    params = mdl.make_params("binary", [1.0, 2.0, 3.0, 4.0])
    assert (params[0], params[1], params[2], params[3]) == (1.0, 2.0, 3.0, 4.0)
    assert (params.mu_A, params.mu_B, params.eps_AA, params.eps_BB) == (1.0, 2.0, 3.0, 4.0)


def test_thermodynamics_accepts_named_parameters():
    calc = ms.CalcConfig()
    calc.model = "langmuir"
    calc.metParam = 8
    calc.iterations = 12
    by_index = ms.thermodynamics(calc, 120.0, [4.0, 4.0], coverage=True, mu_index=0)
    by_name = ms.thermodynamics(
        calc, 120.0, {"mu": 4.0, "eps": 4.0}, coverage=True, mu_index="mu"
    )
    assert by_name["coverage"] == by_index["coverage"]
