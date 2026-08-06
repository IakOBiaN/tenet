"""Driving a parameter sweep and checking it against a stored reference.

Every entry script in the repository does the same five things: step a parameter,
call :func:`~tenet.MainScripts.thermodynamics` at each step, print a row, collect
one observable, and compare the result against an etalon measured when the code
was known good.  Only the first two are about physics; the rest was copied from
script to script, which is how two of them ended up reporting a third script's
name on failure.

The physical constants and the etalon array stay in the script - they are the
content, and they should be readable where they are used.
"""
import inspect
import timeit
from pathlib import Path

import numpy as np

import tenet.MainScripts as ms

GREEN = "\033[92m {}\033[00m"


def run_sweep(calc, values, *, T, m_par, observables = ("coverage", ),
		etalon = None, tolerance = 0.5, compare = None, label = None,
		header = None, show_time = True, quiet = False, **thermo_kwargs):
	"""Run thermodynamics() across ``values``, printing a row per point.

	``T`` and ``m_par`` are each either a constant or a callable of the swept
	value, so the same helper drives a chemical-potential sweep at fixed
	temperature, a temperature sweep at fixed parameters, and the scripts that
	step a coupling constant instead.

	``observables`` names what to request and, in that order, what to print after
	the swept value; ``show_time`` appends the elapsed seconds, which some of the
	scripts print and some do not.  ``header`` is True for a generated column
	line, or a sequence of column names to print verbatim.  Anything
	``thermodynamics`` accepts - ``mu_index``, ``dmu``, ``dT`` - passes through as
	a keyword.

	With ``etalon`` given, the collected ``compare`` observable (the first one by
	default) is checked against it: the sum of absolute deviations must stay
	under ``tolerance``, the same criterion the scripts have always used.
	``label`` names the script in the pass/fail line and defaults to the file
	calling this function, so it cannot disagree with reality.

	Returns the collected observables as a dict of lists.
	"""
	if label is None:
		label = Path(inspect.stack()[1].filename).name
	compare = compare or observables[0]
	if compare not in observables:
		raise ValueError("compare=" + repr(compare) + " is not among the requested observables")

	requested = {name: True for name in observables if name != "grand_potential"}
	collected = {name: [] for name in observables}

	if header is True:
		print("Parameter", *observables, *(["Time"] if show_time else []))
	elif header:
		print(*header)

	start = timeit.default_timer()
	for value in values:
		obs = ms.thermodynamics(
			calc,
			T(value) if callable(T) else T,
			m_par(value) if callable(m_par) else m_par,
			**requested,
			**thermo_kwargs,
		)
		for name in observables:
			collected[name].append(obs.get(name, 0.0))
		if not quiet:
			row = [value] + [obs.get(name, 0.0) for name in observables]
			if show_time:
				row.append(timeit.default_timer() - start)
			print(*row)

	if etalon is not None:
		check_etalon(collected[compare], etalon, label, tolerance)
	return collected


def check_etalon(result, etalon, label, tolerance = 0.5):
	"""Assert that ``result`` reproduces ``etalon``, then report it the usual way."""
	if len(result) != len(etalon):
		raise AssertionError(
			"ERROR! Test " + label + " produced " + str(len(result))
			+ " points but its etalon has " + str(len(etalon))
			+ "; the sweep range and the etalon disagree."
		)
	difference = sum(np.abs(np.array(result) - np.array(etalon)))
	assert difference < tolerance, "ERROR! Test " + label + " is broken now!"
	print("Test " + label + " is", end = "")
	print(GREEN.format("OK"), "!", sep = "")
