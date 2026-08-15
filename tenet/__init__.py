"""TeNeT - tensor network renormalization for the thermodynamics of lattice models.

Computes the grand potential ln Z per site of 1D/2D lattice models through
several renormalization-group schemes (TRG, BTRG, HOTRG, transfer matrix, and
hierarchical lattices) and takes finite-difference derivatives of it to obtain
coverage, entropy, susceptibility and heat capacity.

	import tenet

	calc = tenet.CalcConfig()
	calc.model = "langmuir"
	calc.method = "trg"
	calc.metParam = 16

	obs = tenet.thermodynamics(calc, T = 120.0, m_par = {"mu": 4.0, "eps": 4.0},
			coverage = True)

The models live in :mod:`tenet.models`, one module per family; each declares its
parameters by name, so ``tenet.models.get_params(name)`` says what a model takes.
"""
from tenet.MainScripts import CalcConfig, simulate, thermodynamics
from tenet.sweep import check_etalon, run_sweep

__version__ = "1.0.0"

__all__ = [
	"CalcConfig",
	"check_etalon",
	"run_sweep",
	"simulate",
	"thermodynamics",
]
