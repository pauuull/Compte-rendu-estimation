import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

# Paramètres communs
fs = 1024  # Fréquence d'échantillonnage (Hz)
N = 7500  # Nombre total d'échantillons
t = np.arange(N) / fs  # Axe temporel
sigma = 0.08  # Écart-type du bruit
A1 = np.sqrt(2)  # Amplitude sinusoïde 1
A2 = np.sqrt(2)/100  # Amplitude sinusoïde 2
f1 = 140  # Fréquence de la première sinusoïde (Hz)
f2 = 180  # Fréquence de la deuxième sinusoïde (Hz)

# 1. Génération des signaux
# Bruit blanc
bruit = np.random.normal(0, sigma, N)

# Signal2
signal2 = A1 * np.sin(2 * np.pi * f1 * t) + A2 * np.sin(2 * np.pi * f2 * t) + bruit

# Signal1 : Filtre de Chebyshev appliqué sur le bruit
# Coefficients du filtre de Chebyshev (donnés dans l'énoncé)
b = [0.0154, 0.0461, 0.0461, 0.0154]
a = [1, -1.9903, 1.5717, -0.458]
signal1 = lfilter(b, a, bruit)

# 2. Fonctions pour calculer les périodogrammes
def periodogram(signal, fs):
    N = len(signal)
    freqs = np.fft.fftfreq(N, d=1/fs)[:N//2]
    fft_vals = np.fft.fft(signal)[:N//2]
    Pxx = (np.abs(fft_vals) ** 2) / N
    return freqs, Pxx

def averaged_periodogram(signal, fs, segment_length):
    K = len(signal) // segment_length
    Pxx_segments = []
    for k in range(K):
        segment = signal[k * segment_length:(k + 1) * segment_length]
        freqs, Pxx_segment = periodogram(segment, fs)
        Pxx_segments.append(Pxx_segment)
    Pxx_avg = np.mean(Pxx_segments, axis=0)
    return freqs, Pxx_avg

# 3. Périodogrammes moyennés pour différentes tailles de tranches
segment_lengths = [512, 256, 128, 64]

# Liste des signaux à traiter
signals = {
    "Signal1 (filtré)": signal1,
    "Bruit blanc": bruit,
    "Signal2 (combiné)": signal2
}

# Calcul de la DSP exacte pour chaque signal
def exact_dsp(signal_type, freqs, fs):
    if signal_type == "Signal1 (filtré)":
        # Réponse fréquentielle du filtre Chebyshev
        w = 2 * np.pi * freqs / fs
        z = np.exp(1j * w)
        H = (0.0154 + 0.0461 / z + 0.0461 / z**2 + 0.0154 / z**3) / \
            (1 - 1.9903 / z + 1.5717 / z**2 - 0.458 / z**3)
        return np.abs(H)**2 * sigma**2  # |H(f)|^2 * puissance du bruit
    elif signal_type == "Bruit blanc":
        return np.full_like(freqs, sigma**2)  # DSP constante
    elif signal_type == "Signal2 (combiné)":
        # Contribution du bruit
        dsp = np.full_like(freqs, sigma**2)
        # Ajout des pics des sinusoïdes
        bandwidth = fs / len(freqs)  # Bande autour des fréquences
        dsp[np.abs(freqs - f1) < bandwidth] += A1**2 / 2  # Pic à f1 = 140 Hz
        dsp[np.abs(freqs - f2) < bandwidth] += A2**2 / 2  # Pic à f2 = 180 Hz
        return dsp
    else:
        raise ValueError("Signal type inconnu")


# Affichage des périodogrammes simples et moyennés pour les différents signaux
for signal_name, signal in signals.items():
    plt.figure(figsize=(12, 7))

    # Périodogramme simple
    freqs_simple, Pxx_simple = periodogram(signal[:1050], fs)
    plt.plot(freqs_simple, 10 * np.log10(Pxx_simple), '--', label='Périodogramme simple (1050 points)')

    # Périodogramme moyenné
    for segment_length in segment_lengths:
        freqs, Pxx_avg = averaged_periodogram(signal, fs, segment_length)
        plt.plot(freqs, 10 * np.log10(Pxx_avg), label=f'Moyenné (tranches de {segment_length} points)')

    # Modèle exact
    freqs = np.fft.fftfreq(1050, d=1/fs)[:1050//2]
    dsp_exact = exact_dsp(signal_name, freqs, fs)
    plt.plot(freqs, 10 * np.log10(dsp_exact), label='Modèle exact', color='black', linestyle='dotted', linewidth=5) 
    
    # Mise en forme
    plt.title(f"Comparaison pour {signal_name}")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.legend()
    plt.grid()
    plt.show()