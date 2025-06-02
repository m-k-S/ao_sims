import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, h, k, epsilon_0, atomic_mass
from scipy.special import voigt_profile as voigt_func

# Constants for Sr88
wavelength_689 = 689e-9  # m (1S0 -> 3P1 red transition)
wavelength_461 = 461e-9  # m (1S0 -> 1P1 blue transition)
atomic_weight_sr88 = 87.9056  # amu
sr88_mass = atomic_weight_sr88 * atomic_mass  # kg

atom_number = 10e7
temperature = 0.01  # K (1000 uK)
mot_diameter = 0.005  # m 

cross_section_689 = 3 * wavelength_689**2 / (2 * np.pi)
def optical_depth_MOT(diameter, atom_number, cross_section):
    density = (3/4) * atom_number / (np.pi * (diameter / 2)**3)
    return density * cross_section * diameter

optical_depth = optical_depth_MOT(mot_diameter, atom_number, cross_section_689)
print(f"Optical depth of Sr88 MOT for 689nm light: {optical_depth:.3f}")

doppler_fwhm = 2 * np.sqrt(2 * np.log(2) * k * temperature / sr88_mass) * (1 / wavelength_689)
print(f"Doppler broadening (FWHM) at 2 mK: {doppler_fwhm/1e3:.3f} kHz")

natural_linewidth_689 = 7.5e3  # Hz 
print(f"Natural linewidth: {natural_linewidth_689/1e3:.3f} kHz")

freq_range = np.linspace(-500, 500, 1000)  # kHz
on_resonance_freq = 0  # kHz (exactly on resonance)
blue_detuned_freq = 1000  # kHz (blue-detuned by 200 kHz)

gamma = natural_linewidth_689 / 2 / 1e3  # Natural linewidth in kHz (HWHM)
sigma = doppler_fwhm / 2.355 / 1e3  # Doppler width in kHz (standard deviation)

absorption_profile = np.zeros_like(freq_range)
for i, freq in enumerate(freq_range):
    absorption_profile[i] = voigt_func(freq, sigma, gamma)

absorption_profile = absorption_profile / np.max(absorption_profile)

transmission = np.exp(-optical_depth * absorption_profile)

on_resonance_idx = np.argmin(np.abs(freq_range - on_resonance_freq))
blue_detuned_idx = np.argmin(np.abs(freq_range - blue_detuned_freq))

on_resonance_transmission = transmission[on_resonance_idx]
blue_detuned_transmission = transmission[blue_detuned_idx]

print(f"\nTransmission at resonance (0 kHz): {on_resonance_transmission:.8f}")
print(f"Transmission at blue-detuned frequency (+{blue_detuned_freq} kHz): {blue_detuned_transmission:.8f}")
print(f"Transmission ratio (blue-detuned/resonance): {blue_detuned_transmission/on_resonance_transmission:.2f}x higher")

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(freq_range, absorption_profile, 'b-', label='Absorption profile')
plt.axvline(x=on_resonance_freq, color='r', linestyle='--', label='On resonance')
plt.axvline(x=blue_detuned_freq, color='g', linestyle='--', label=f'Blue-detuned (+{blue_detuned_freq} kHz)')
plt.ylabel('Normalized absorption')
plt.title('Sr88 MOT Absorption Profile (689nm, 1S0 -> 3P1 transition)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(freq_range, transmission, 'r-', label='Transmission')
plt.axvline(x=on_resonance_freq, color='r', linestyle='--')
plt.axvline(x=blue_detuned_freq, color='g', linestyle='--')
plt.plot(on_resonance_freq, on_resonance_transmission, 'ro', markersize=8, 
         label=f'On resonance: {on_resonance_transmission:.8f}')
plt.plot(blue_detuned_freq, blue_detuned_transmission, 'go', markersize=8, 
         label=f'Blue-detuned: {blue_detuned_transmission:.8f}')
    
plt.xlabel('Frequency detuning (kHz)')
plt.ylabel('Transmission')
plt.title(f'Transmission through Sr88 MOT (Optical depth: {optical_depth:.1f})')
plt.legend()
plt.grid(True)

plt.tight_layout()

# Create a table of transmission values at different detunings
print("\nTransmission values at different detunings:")
print("-" * 40)
print(f"{'Detuning (kHz)':>15} | {'Transmission':>15}")
print("-" * 40)

detunings = [-500, -400, -300, -200, -100, -50, -25, 0, 25, 50, 100, 200, 300, 400, 500]

for detuning in detunings:
    idx = np.argmin(np.abs(freq_range - detuning))
    trans_value = transmission[idx]
    print(f"{detuning:>15.1f} | {trans_value:>15.8f}")

plt.show()
