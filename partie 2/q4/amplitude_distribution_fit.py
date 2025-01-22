import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rayleigh, gamma, lognorm, ks_1samp

# Charger le fichier signal (assurez-vous qu'il est dans un dossier "data" au même niveau que ce script)
import os
signal_path = os.path.join(os.path.dirname(__file__), "data", "signal")
signal = np.fromfile(signal_path, dtype=np.float32)

# Paramètres
fs = 1000  # Fréquence d'échantillonnage en Hz
N = len(signal)  # Nombre d'échantillons
freq_range = 100  # Plage de fréquences d'intérêt (0-100 Hz)

# Calcul de la TFD
def compute_fft(signal, fs):
    freqs = np.fft.fftfreq(len(signal), d=1/fs)
    fft_vals = np.abs(np.fft.fft(signal))
    return freqs[:len(signal)//2], fft_vals[:len(signal)//2]

# Calcul de la TFD
freqs, fft_vals = compute_fft(signal, fs)

# Sélection des indices correspondant aux fréquences entre 0 et 100 Hz
indices = np.where((freqs >= 0) & (freqs <= freq_range))[0]
selected_amplitudes = fft_vals[indices]

# Calcul des statistiques des amplitudes
mean_amplitude = np.mean(selected_amplitudes)
variance = np.var(selected_amplitudes)

print(f"Moyenne des amplitudes (0-100 Hz) : {mean_amplitude:.4f}")
print(f"Variance des amplitudes (0-100 Hz) : {variance:.4f}")

# Tracé de l'histogramme des amplitudes
plt.figure(figsize=(10, 6))
plt.hist(selected_amplitudes, bins=30, alpha=0.7, color="blue", edgecolor="black", density=True, label="Histogramme des amplitudes")

# Ajustement à une distribution Rayleigh
sigma_rayleigh = np.sqrt(variance / 2)
x = np.linspace(0, np.max(selected_amplitudes), 500)
rayleigh_pdf = rayleigh.pdf(x, scale=sigma_rayleigh)
plt.plot(x, rayleigh_pdf, color="red", lw=2, label="Rayleigh")

# Ajustement à une distribution Gamma
shape_gamma, loc_gamma, scale_gamma = gamma.fit(selected_amplitudes, floc=0)  # On fixe loc=0
gamma_pdf = gamma.pdf(x, a=shape_gamma, scale=scale_gamma)
plt.plot(x, gamma_pdf, color="green", lw=2, label="Gamma")

# Ajustement à une distribution Log-normale
shape_lognorm, loc_lognorm, scale_lognorm = lognorm.fit(selected_amplitudes, floc=0)  # On fixe loc=0
lognorm_pdf = lognorm.pdf(x, s=shape_lognorm, scale=scale_lognorm)
plt.plot(x, lognorm_pdf, color="orange", lw=2, label="Log-normale")

# Ajout des titres et des légendes
plt.title("Histogramme des amplitudes (0-100 Hz) avec ajustements")
plt.xlabel("Amplitude")
plt.ylabel("Densité de probabilité")
plt.legend()
plt.grid()
plt.show()

# Comparaison des ajustements avec le test KS (Kolmogorov-Smirnov)
ks_rayleigh = ks_1samp(selected_amplitudes, lambda data: rayleigh.cdf(data, scale=sigma_rayleigh))
ks_gamma = ks_1samp(selected_amplitudes, lambda data: gamma.cdf(data, a=shape_gamma, scale=scale_gamma))
ks_lognorm = ks_1samp(selected_amplitudes, lambda data: lognorm.cdf(data, s=shape_lognorm, scale=scale_lognorm))

print("Tests d'ajustement avec Kolmogorov-Smirnov :")
print(f"Rayleigh : p-value = {ks_rayleigh.pvalue:.4f}")
print(f"Gamma : p-value = {ks_gamma.pvalue:.4f}")
print(f"Log-normale : p-value = {ks_lognorm.pvalue:.4f}")

# Affichage des paramètres ajustés
print("\nParamètres des distributions :")
print(f"Rayleigh : sigma = {sigma_rayleigh:.4f}")
print(f"Gamma : shape = {shape_gamma:.4f}, scale = {scale_gamma:.4f}")
print(f"Log-normale : shape = {shape_lognorm:.4f}, scale = {scale_lognorm:.4f}")