import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, epsilon_0, mu_0, hbar, Boltzmann, e, physical_constants
from scipy.special import voigt_profile

import pint 
from arc import Rubidium85

# Constants
def make_unit_registry():
    ureg = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
    Q_ = ureg.Quantity
    ureg.define('uK = micro*kelvin = microkelvin')
    return ureg, Q_
si, Q_ = make_unit_registry()

atom = Rubidium85()
c = c * si.m / si.s
mu_0 = mu_0 * si.N / si.A**2
epsilon_0 = epsilon_0 * si.F / si.m**-1
hbar = hbar * si.J * si.s
bohr_radius = physical_constants['Bohr radius'][0] * si.m 
e = e * si.coulomb

G_F3 = 0.333


Isat_D1 = Q_(4.48e-3,'W')/Q_(1e-4,'m^2')   # 3.6 mW/cm^2
Gamma_D1 = 2*pi*Q_(5.75e6,'Hz') # rad/s
Sigma0_D1 = 1.005 * 10**-9 * (si.centimeter)**2

Isat_D2 = Q_(2.503e-3,'W')/Q_(1e-4,'m^2')   # 3.6 mW/cm^2
Gamma_D2 = 2*pi*Q_(6.07e6,'Hz') # rad/s
Sigma0_D2 = 1.937 * 10**-9 * (si.centimeter)**2

# MOT Parameters

mot_radius = Q_(5,'mm')
def gaussian_density_3d(x,y,z, N, sigma):
    n0 = N / ((2 * np.pi)**(3/2) * sigma**3)
    return n0 * np.exp(-0.5 * ((x/sigma)**2 + (y/sigma)**2 + (z/sigma)**2))

def Zeeman_shift(x, y, z, GF, mF, Bgrad):
    B_field = np.sqrt((Bgrad*x/2)**2 + (Bgrad*y/2)**2 + (Bgrad*z)**2)
    # frequency shift
    delta_nu = muB * GF * mF * B_field / (2 * pi * hbar)
    return delta_nu

# Geometric Parameters

probe_beam_radius = 0.0005 * si.m 
probe_beam_power = 1 * si.uW

# Absorption Parameters

doppler_width = wavevector * v_thermal / (2 * np.pi)

absorption_780 = voigt_profile(detuning, sigma = )