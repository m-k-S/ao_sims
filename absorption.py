import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, epsilon_0, mu_0, hbar, Boltzmann, e, physical_constants
from scipy.special import voigt_profile

from tqdm import tqdm

import pint 

# Constants
def make_unit_registry():
    ureg = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
    Q_ = ureg.Quantity
    return ureg, Q_
si, Q_ = make_unit_registry()

mass_Rb = 84.9117893 * si.amu
c = c * si.m / si.s
mu_0 = mu_0 * si.N / si.A**2
epsilon_0 = epsilon_0 * si.F / si.m**-1
hbar = hbar * si.J * si.s
bohr_radius = physical_constants['Bohr radius'][0] * si.m 
muB = physical_constants['Bohr magneton'][0] * si.J/si.T
Boltzmann = Boltzmann * si.J / si.K
e = e * si.coulomb

G_F3 = 0.333

wavelength_795 = 794.97914 * si.nm
wavevector_795 = 2 * np.pi / wavelength_795

Isat_D1 = Q_(3.6e-3,'W')/Q_(1e-4,'m^2')   # 3.6 mW/cm^2
Gamma_D1 = Q_(5.75e6,'Hz') # rad/s
Sigma0_D1 = 1.005 * 10**-9 * (si.centimeter)**2

wavelength_780 = 780.241368 * si.nm
wavevector_780 = 2 * np.pi / wavelength_780

Isat_D2 = Q_(1.6e-3,'W')/Q_(1e-4,'m^2')   # 1.6 mW/cm^2
Gamma_D2 = Q_(6.07e6,'Hz') # rad/s
Sigma0_D2 = 1.937 * 10**-9 * (si.centimeter)**2

v_thermal = lambda T, m: np.sqrt(Boltzmann * T / m)
doppler_width = lambda wavevector, T: wavevector * v_thermal(T, mass_Rb) / (2 * np.pi)

# MOT Parameters

mot_radius = Q_(5,'mm')
def gaussian_density_3d(x,y,z, N, sigma):
    n0 = N / ((2 * np.pi)**(3/2) * sigma**3)
    return n0 * np.exp(-0.5 * ((x/sigma)**2 + (y/sigma)**2 + (z/sigma)**2))

def Zeeman_shift(x, y, z, GF, mF, Bgrad):
    B_field = np.sqrt((Bgrad*x/2)**2 + (Bgrad*y/2)**2 + (Bgrad*z)**2)
    # frequency shift
    delta_nu = muB * GF * mF * B_field / (2 * np.pi * hbar)
    return delta_nu

# Geometric Parameters

probe_beam_radius = 0.0005 * si.m 
probe_beam_power = 1 * si.uW
probe_beam_intensity = probe_beam_power / (np.pi * probe_beam_radius**2)

MOT_beam_radius = 10 * si.cm
MOT_beam_power = 7.5 * si.mW
MOT_beam_intensity = MOT_beam_power / (np.pi * MOT_beam_radius**2)

def spatial_intensity(x, y, P, beam_radius):
    I0 = 2 * P / (np.pi * beam_radius**2)
    return I0 * np.exp(-2 * (x**2 + y**2) / beam_radius**2)

# Absorption Parameters

# MAY NEED TO EXPLICITLY MODEL EXCITED STATE POPULATIONS TOO
# THIS DOES NOT CURRENTLY DO THAT

absorption_780 = lambda detuning, T: voigt_profile(detuning, sigma = doppler_width(wavevector_780, T), gamma = Gamma_D2)
absorption_795 = lambda detuning, T: voigt_profile(detuning, sigma = doppler_width(wavevector_795, T), gamma = Gamma_D1)

def scattering_rate(detuning, I, Isat, Gamma):
    s = (I / Isat).to('dimensionless')
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

repump_prop = 0.05
loss_coefficient = 0.01

detuning_795 = 0.0 * si.Hz            # probe detuning (relative to your chosen D1 reference)
detuning_cooling = (-12.0 * si.MHz)   # D2 cooling detuning
detuning_repump  = (0.0 * si.MHz)     # D2 repump near resonant
Bgrad = 0.12 * si.T/si.m              # quadrupole gradient
mF_probe = +1                         # effective m_F for Zeeman shift on D1

# Derived from simulation parameters:

Doppler_sigma_795 = doppler_width(wavevector_795, temperature)

# Simulation
# z-axis is along the direction of absorption

z = np.linspace(-10,10,1000) * si.cm
x = np.linspace(-5,5,1000) * si.cm
y = np.linspace(-5,5,1000) * si.cm

X, Y, Z = np.meshgrid(x, y, z)

I_probe_map = spatial_intensity(X, Y, probe_beam_power, probe_beam_radius)
I = I_probe_map.copy()
I_MOT_map = spatial_intensity(X, Y, MOT_beam_power, MOT_beam_radius)
I_repump_map = spatial_intensity(X, Y, repump_prop * MOT_beam_power, MOT_beam_radius)

# Beer–Lambert march along z
x_vals = X[0,:,0] if X.ndim==3 else x
y_vals = Y[:,0,0] if Y.ndim==3 else y
z_vals = Z[0,0,:]

dz = (z_vals[1] - z_vals[0]).to('m')

x_m = x_vals.to('m').magnitude
y_m = y_vals.to('m').magnitude
I0_xy = I_probe_map[:, :, 0].to('W/m^2').magnitude  # take z-slice
Pin = np.trapezoid(np.trapezoid(I0_xy, x_m, axis=1), y_m, axis=0) * si.W # scalar W

print (Pin)
print (probe_beam_power)
assert(Pin == probe_beam_power)

for k, zi in tqdm(enumerate(z_vals)):
    z_m = zi.to('m').magnitude

    # Local Zeeman shift across the transverse plane (Hz)
    dZ = Zeeman_shift(X, Y, zi, G_F3, mF_probe, Bgrad)

    # Local D2 rates (use maps; if you keep them uniform, replace Ic_map/Ir_map by scalars)
    Rc = scattering_rate(detuning_cooling, I_MOT_map, Isat_D2, Gamma_D2)
    Rr = scattering_rate(detuning_repump,  I_repump_map, Isat_D2, Gamma_D2)

    # Steady-state ground population fraction in F=3 from D2 competition
    pF3 = Rr / (Rr + (loss_coefficient * Rc) + (1e-30 * si.Hz))  # same as steady_state_F3_pop()

    # Extra homogeneous broadening on D1 from D2 scattering (Hz, HWHM contribution)
    gamma_extra = 0.5 * (Rc + Rr)                    # simple, effective decoherence term

    # Local probe saturation
    s_probe = I / Isat_D1

    # Power-broadened Lorentzian HWHM (Hz), including D2-induced extra width
    gammaL = (Gamma_D1/2.0) * np.sqrt(1.0 + s_probe) + gamma_extra

    # Voigt profile at each (x,y): V(Δ, σ_G, γ_L) with Δ = detuning_795 - dZ
    Delta_xy = (detuning_795 - dZ).to('Hz').magnitude
    V_xy = voigt_profile(Delta_xy, Doppler_sigma_795.to('Hz').magnitude, gammaL.to('Hz').magnitude)  # Hz^-1
    V_xy = Q_(V_xy, 'Hz^-1')

    # Effective D1 cross-section at each (x,y): σ = σ0 * (π γL)/(1+s) * V
    sigma_xy = Sigma0_D1 * (np.pi * gammaL) / (1.0 + s_probe) * V_xy  # m^2

    # Weight by F=3 population only (assuming probe tuned to an F=3-centered D1 component).
    # If you want full F→F′ hyperfine sum, replace this line by your multiline sum.
    sigma_xy *= pF3

    # Absorb this slice
    n_slice = gaussian_density_3d(X[:,:,k], Y[:,:,k], zi, N_atoms, mot_radius)
    alpha = (n_slice * sigma_xy).to('1/m')
    I *= np.exp(- (alpha * dz).to('dimensionless').magnitude)

Pout = np.trapezoid(np.trapezoid(I, x_vals.to('m').magnitude, axis=1), y_vals.to('m').magnitude, axis=0)
OD = -np.log((Pout + (1e-30 * si.W/si.m**2))/(Pin + (1e-30 * si.W/si.m**2)))
print("OD_795 =", float(OD))



