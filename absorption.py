import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, epsilon_0, mu_0, hbar, Boltzmann, e, physical_constants
from scipy.special import voigt_profile

import pint 

# Constants
def make_unit_registry():
    ureg = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
    Q_ = ureg.Quantity
    ureg.define('uK = micro*kelvin = microkelvin')
    return ureg, Q_
si, Q_ = make_unit_registry()

mass_Rb = 84.9117893 * si.amu
c = c * si.m / si.s
mu_0 = mu_0 * si.N / si.A**2
epsilon_0 = epsilon_0 * si.F / si.m**-1
hbar = hbar * si.J * si.s
bohr_radius = physical_constants['Bohr radius'][0] * si.m 
Boltzmann = Boltzmann * si.J / si.K
e = e * si.coulomb

G_F3 = 0.333

wavelength_795 = 794.97914 * si.nm
wavevector_795 = 2 * pi / wavelength_795

Isat_D1 = Q_(4.48e-3,'W')/Q_(1e-4,'m^2')   # 3.6 mW/cm^2
Gamma_D1 = 2*pi*Q_(5.75e6,'Hz') # rad/s
Sigma0_D1 = 1.005 * 10**-9 * (si.centimeter)**2

wavelength_780 = 780.241368 * si.nm
wavevector_780 = 2 * pi / wavelength_780

Isat_D2 = Q_(3.6e-3,'W')/Q_(1e-4,'m^2')   # 3.6 mW/cm^2
Gamma_D2 = 2*pi*Q_(6.07e6,'Hz') # rad/s
Sigma0_D2 = 1.937 * 10**-9 * (si.centimeter)**2

v_thermal = lambda T, m: np.sqrt(Boltzmann * T / m)

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
probe_beam_intensity = probe_beam_power / (pi * probe_beam_radius**2)

MOT_beam_radius = 10 * si.cm
MOT_beam_power = 7.5 * si.mW
MOT_beam_intensity = MOT_beam_power / (pi * MOT_beam_radius**2)

def spatial_intensity(x,y,P, beam_radius):
    I0 = 2 * P / (pi * beam_radius**2)
    return I0 * np.exp(-2 * (x**2 + y**2) / beam_radius**2)

# Absorption Parameters

# MAY NEED TO EXPLICITLY MODEL EXCITED STATE POPULATIONS TOO
# THIS DOES NOT CURRENTLY DO THAT

doppler_width = lambda wavevector, T: wavevector * v_thermal(T, mass_Rb) / (2 * np.pi)

absorption_780 = lambda detuning, T: voigt_profile(detuning, sigma = doppler_width(wavevector_780, T), gamma = Gamma_D2)
absorption_795 = lambda detuning, T: voigt_profile(detuning, sigma = doppler_width(wavevector_795, T), gamma = Gamma_D1)

def scattering_rate(detuning, I, Isat, Gamma):
    s = I / Isat
    rate = (Gamma / 2) * (s / (1 + s + ((2 * detuning) / Gamma)**2))
    return rate

def steady_state_F3_pop(detune_cooling, loss_coefficient, I_780, repump_prop, Isat780, Gamma780):
    cooling_rate = scattering_rate(detune_cooling, I_780, Isat780, Gamma780)
    repump_rate = scattering_rate(0, repump_prop * I_780, Isat780, Gamma780)
    return (repump_rate) / (repump_rate + (loss_coefficient * cooling_rate))


### SIMULATION PARAMETERS ###
# (things that can be tuned)

temperature = 100 * si.uK
N_atoms = 1e6

repump_prob = 0.05
loss_coefficient = 0.01

# Simulation
# z-axis is along the direction of absorption

z = np.linspace(-10,10,1000) * si.cm
x = np.linspace(-5,5,1000) * si.cm
y = np.linspace(-5,5,1000) * si.cm

X, Y, Z = np.meshgrid(x, y, z)



