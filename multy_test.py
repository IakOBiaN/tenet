import tenet.MainScripts as ms
from tenet.sweep import run_sweep


calc = ms.CalcConfig()

calc.method = "trg"
calc.metModification = "default"
calc.model = "six_leg_test"
calc.lattice = "triangular"
calc.gen_tensor = "six_leg_tensor"
calc.metParam = 48
#model params
T = 640#39.69208563868
eps = 4.8
mu_Cu = -19.0
w2 = -20
w3 = -40.0
w3_1 = -35.2 - eps
w4 = -45.6 - 3.0 * eps
mu_TPB = 6.0

"""eps = 1.7
mu_Cu = -19.0
w2 = -20
w3 = -35.4
w3_1 = -31.5 - eps
w4 = -37.7 - 3.0 * eps
mu_TPB = 6.0"""

#fff = open("an_calc_prov.dat", "w")

etalon_coverage = [0.666830385986863, 0.6677578428100706, 0.6700237836368839, 0.6736219881364036, 0.6780650915418285, 0.68272407905412, 0.6874497260243384, 0.6907369973878019]

run_sweep(
	calc,
	ms.np.arange(300, 1000.01, 100),
	T = lambda T: T,
	m_par = lambda T: {"mu_TPB": mu_TPB, "mu_Cu": mu_Cu, "w2": w2, "w3": w3,
			"w3_1": w3_1, "w4": w4, "eps": eps},
	observables = ("coverage", ),
	mu_index = ["mu_TPB", "mu_Cu"],
	dmu = 1e-1,
	show_time = False,
	etalon = etalon_coverage,
)
