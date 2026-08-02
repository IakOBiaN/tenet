"""Per-model construction of the Boltzmann weight matrices.

``build_matrix`` in :mod:`tenet.BuildTensors` is only a dispatcher: it pads the
parameter list, looks the model up here, and applies the shared epilogue that
divides by R*T and exponentiates.  Everything model-specific lives in one of the
modules imported at the bottom of this file.

A model is a function ``(calc, temp, m_par) -> list of numpy arrays`` returning
the *raw* weight matrices, before the epilogue.  Register it under the name that
goes into ``CalcConfig.model``::

	@register("langmuir")
	def langmuir(calc, temp, m_par):
		neigbours = calc.coord
		...
		return matrixes

``temp`` is part of the contract even though most models ignore it - temperature
normally enters through the epilogue, not the construction.

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


def register(name):
	"""Register a model builder under the name used in ``CalcConfig.model``."""
	def decorator(build):
		if name in MODELS:
			raise ValueError("model " + repr(name) + " is already registered")
		MODELS[name] = build
		return build
	return decorator


def get_model(name):
	"""Return the builder registered for ``name``, or None if there is none."""
	return MODELS.get(name)


#Imported for their registration side effect, and kept at the bottom so that
#register() above already exists when they run.
from tenet.models import (  # noqa: E402,F401
	adsorption,
	exclusion,
	lattice_gases,
	pentacene_2,
	six_leg,
	spin,
)
