import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, dimpulse, freqz

# Définition des coefficients du filtre de Chebyshev
numerator = [0.0154, 0.0461, 0.0154]
denominator = [1, -1.9903, 1.5717, -0.458]

# Paramètres
noise_variance = 0.5
N_values = [128, 256, 512]  # Différentes tailles d'échantillons
threshold = 10 ** (-6)  # Seuil pour -60 dB

# Génération d'un bruit blanc
num_samples = max(N_values)
bruit = np.random.normal(0, np.sqrt(noise_variance), num_samples)

# Génération du signal filtré (signal1)
signal1 = lfilter(numerator, denominator, bruit)

# Fonction pour calculer le périodogramme
def periodogram(x, N):
    fft_x = np.fft.fft(x, n=N)
    Pxx = (np.abs(fft_x) ** 2) / N
    freqs = np.fft.fftfreq(N, d=1)
    return freqs[:N // 2], Pxx[:N // 2]  # Retourne les fréquences positives uniquement

# Calcul de la réponse fréquentielle et de la DSP théorique
def calculate_theoretical_psd(numerator, denominator, sigma2, N):
    w, h = freqz(numerator, denominator, worN=N)  # Réponse fréquentielle du filtre
    G_squared = np.abs(h)**2  # Module au carré de G(f)
    P_theoretical = sigma2 * G_squared  # DSP théorique
    return w / (2 * np.pi), P_theoretical  # Fréquences normalisées et DSP

# Calcul de la réponse impulsionnelle
def plot_impulse_response(numerator, denominator):
    t, h = dimpulse((numerator, denominator, 1))  # 1 = période d'échantillonnage
    h = np.squeeze(h)
    plt.stem(range(len(h)), h, basefmt=" ")
    plt.title("Réponse impulsionnelle du filtre")
    plt.xlabel("Échantillons")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

# Calcul des fréquences et de la DSP théorique
frequencies, P_theoretical = calculate_theoretical_psd(numerator, denominator, noise_variance, max(N_values))

# Affichage de la réponse impulsionnelle
plot_impulse_response(numerator, denominator)

# Affichage des périodogrammes avec DSP théorique
plt.figure(figsize=(14, 7))

for i, N in enumerate(N_values):
    # Périodogramme pour le signal filtré
    f_signal, Pxx_signal = periodogram(signal1[:N], N)
    valid_indices_signal = Pxx_signal > threshold

    # Affichage du périodogramme du signal filtré avec la DSP théorique
    plt.subplot(2, len(N_values), i + 1)
    plt.semilogy(frequencies[:len(Pxx_signal)], P_theoretical[:len(Pxx_signal)], label="DSP Théorique", color="blue", linestyle="dotted")
    plt.semilogy(f_signal[valid_indices_signal], Pxx_signal[valid_indices_signal], label=f'Signal filtré (N={N})')
    plt.title(f'Périodogramme signal (N={N})')
    plt.xlabel('Fréquence normalisée')
    plt.ylabel('Densité spectrale (log)')
    plt.legend()

    # Périodogramme pour le bruit blanc
    f_noise, Pxx_noise = periodogram(bruit[:N], N)
    valid_indices_noise = Pxx_noise > threshold

    # Affichage du périodogramme du bruit blanc avec la DSP théorique
    plt.subplot(2, len(N_values), i + 1 + len(N_values))
    plt.semilogy(frequencies[:len(Pxx_noise)], np.full(len(Pxx_noise), noise_variance), label="DSP Théorique", color="blue", linestyle="dotted")
    plt.semilogy(f_noise[valid_indices_noise], Pxx_noise[valid_indices_noise], label=f'Bruit blanc (N={N})', color='r')
    plt.title(f'Périodogramme bruit (N={N})')
    plt.xlabel('Fréquence normalisée')
    plt.ylabel('Densité spectrale (log)')
    plt.legend()

    # Impression des biais et variances
    biais_signal = np.mean(Pxx_signal) - np.mean(P_theoretical[:len(Pxx_signal)])
    variance_signal = np.var(Pxx_signal)
    biais_noise = np.mean(Pxx_noise) - noise_variance
    variance_noise = np.var(Pxx_noise)

    print(f"N = {N}:")
    print(f"  Biais (Signal): {biais_signal:.5f}, Variance (Signal): {variance_signal:.5f}")
    print(f"  Biais (Bruit): {biais_noise:.5f}, Variance (Bruit): {variance_noise:.5f}")

plt.tight_layout()
plt.show()