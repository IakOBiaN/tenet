"""Per-model construction of the Boltzmann weight matrices.

``build_matrix`` in :mod:`tenet.BuildTensors` is only a dispatcher: it pads the
parameter list, looks the model up here, and applies the shared epilogue that
divides by R*T and exponentiates.  Everything model-specific lives in one of the
modules imported at the bottom of this file.

A model is a function ``(calc, temp, m_par) -> list of numpy arrays`` returning
the *raw* weight matrices, before the epilogue.  Register it under the name that
goes into ``CalcConfig.model``, declaring its parameters in order::

	@register("langmuir", params = ["mu", "eps"])
	def langmuir(calc, temp, m_par):
		neigbours = calc.coord
		...
		return matrixes

The declared names are what makes a model usable without reading its source: a
caller may pass ``m_par`` positionally, as the entry scripts do, or by name::

	simulate(calc, T, [4.0, 4.0])
	simulate(calc, T, {"mu": 4.0, "eps": 4.0})

Both arrive at the model as a :class:`Params`, which supports ``m_par[0]`` and
``m_par.mu`` alike.  Names that a model does not declare are rejected, which is
the point - a mistyped or misplaced parameter used to be silently absorbed by a
positional list.

``temp`` is part of the contract even though most models ignore it - temperature
normally enters through the epilogue, not the construction.

How many matrices to return, and what each one means, is a convention shared
with ``build_tensor``: one entry when every bond is equivalent, otherwise
[right, bottom], or [right-up, right, right-bottom], or [right-up, right,
right-bottom, bottom], depending on the lattice.  A model may append extra
weights beyond that - ``langmuir_m`` returns a fourth, rank-3 tensor that the
``_m`` construction variants read as ``matrixes[-1]``.

A model may write derived configuration back onto ``calc``: ``calc.nodes`` when
it merges several lattice sites into one tensor, or ``calc.lattice`` when the
network geometry it produces differs from the one the caller asked for.
``simulate`` passes in a private copy of the config, so those writes stay inside
a single run.

To add a model: write the builder in the module for its family (or a new module,
imported at the bottom here), then add a case to ``tests/build_cases.py`` and
regenerate the golden.
"""

#Stand-in energy for "this configuration is forbidden": large and negative, so
#the Boltzmann weight underflows to zero in the epilogue.  Deliberately not
#math.inf - the models do arithmetic on these entries before they are exponentiated.
inf = -1e8

#model name -> builder function
MODELS = {}
#model name -> tuple of parameter names, in positional order
PARAMS = {}


class Params:
	"""Model parameters, reachable by name or by position.

	Positional access exists because the model bodies were written against a
	plain list and still index it; new code should use the names.
	"""

	def __init__(self, names, values):
		self.__dict__["_by_name"] = dict(zip(names, values))
		self.__dict__["_values"] = list(values)

	def __getattr__(self, name):
		try:
			return self.__dict__["_by_name"][name]
		except KeyError:
			raise AttributeError("no model parameter named " + repr(name)) from None

	def __getitem__(self, index):
		return self.__dict__["_values"][index]

	def __len__(self):
		return len(self.__dict__["_values"])

	def __repr__(self):
		items = ", ".join(k + "=" + repr(v) for k, v in self.__dict__["_by_name"].items())
		return "Params(" + items + ")"


def register(name, params):
	"""Register a model builder under the name used in ``CalcConfig.model``.

	``params`` lists the model's parameters in positional order, so that
	``m_par[i]`` inside the body and ``params[i]`` here mean the same thing.
	"""
	def decorator(build):
		if name in MODELS:
			raise ValueError("model " + repr(name) + " is already registered")
		MODELS[name] = build
		PARAMS[name] = tuple(params)
		return build
	return decorator


def get_model(name):
	"""Return the builder registered for ``name``, or None if there is none."""
	return MODELS.get(name)


def get_params(name):
	"""Return the declared parameter names for ``name``, or None if unregistered."""
	return PARAMS.get(name)


def make_params(name, m_par):
	"""Build a :class:`Params` for model ``name`` from a list or a dict.

	A short list is padded with zeros and a long one is truncated, which is what
	build_matrix's old blanket pad-to-ten did and what the entry scripts rely on -
	several of them pass a few trailing zeros past the parameters the model reads.
	A dict, by contrast, is checked: an undeclared key is an error, since catching
	exactly that mistake is why the names exist.
	"""
	names = PARAMS[name]
	if isinstance(m_par, Params):
		return m_par
	if isinstance(m_par, dict):
		unknown = sorted(set(m_par) - set(names))
		if unknown:
			raise ValueError(
				"model " + repr(name) + " has no parameter(s) " + ", ".join(repr(u) for u in unknown)
				+ "; declared parameters are: " + ", ".join(names)
			)
		values = [m_par.get(key, 0.0) for key in names]
	else:
		values = list(m_par[:len(names)])
		values += [0.0] * (len(names) - len(values))
	return Params(names, values)


#Imported for their registration side effect, and kept at the bottom so that
#register() above already exists when they run.
from tenet.models import (  # noqa: E402,F401
	adsorption,
	exclusion,
	lattice_gases,
	long_range,
	long_range_v,
	pentacene_2,
	six_leg,
	spin,
)
