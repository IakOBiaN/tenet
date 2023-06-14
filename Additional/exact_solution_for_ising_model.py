from math import log, sinh, cosh, cos, pi, sqrt, exp
from scipy import integrate
from scipy.misc import derivative
import numpy as np

def exact(T = 1.0, J = 1.0, lattice = "square"):

	if lattice == "square":
		f = lambda phi2, phi1: log(cosh(2*J/T)**2-sinh(2*J/T)*(cos(phi1)-cos(phi2)))
		result = integrate.dblquad(f, 0, 2*pi, lambda x: 0, lambda x: 2*pi, epsabs = 1e-13, epsrel = 1e-13)
		res = log(2) + result[0]/(8*pi**2)
	elif lattice == "honeycomb":
		f = lambda phi2, phi1: log(1/2*(cosh(2*J/T)**3 +1-sinh(2*J/T)**2*(cos(phi1)+cos(phi2)+cos(phi1 + phi2))))
		result = integrate.dblquad(f, -pi, pi, lambda x: -pi, lambda x: pi, epsabs = 1e-13, epsrel = 1e-13)
		res = log(2) + result[0]/(16*pi**2)
	else:
		L = J/(T)
		k = (exp(4*L)-1)/(exp(4*L)+1)**2
		f = lambda phi2, phi1: log(1-2*k+2*k*cos(phi1)+2*k*cos(phi2)+2*k*cos(pi-phi1-phi2))
		result = integrate.dblquad(f, 0, 2*pi, lambda x: 0, lambda x: 2*pi, epsabs = 1e-13, epsrel = 1e-13)
		res = log(exp(3*L)+exp(-L)) + result[0]/(8*pi**2)

	return (res*T, result[1])

def heat_capasity(T = 1.0, J = 1.0, lattice = "square"):
	result = derivative(lambda x: exact(x, J, lattice)[0], T, n=2, dx=1e-5)
	return (result, )

T1 = 2.0/log(1+sqrt(2))
T2 = 4.0/log(3)
T3 = 4.0/log(3) + 1e-7

T = 1.0
#print(exact(T3, 1.0, "triangular"))
for J in np.arange(0.15, 0.85, 0.01):
	print(J, exact(T, J, "square")[0])
