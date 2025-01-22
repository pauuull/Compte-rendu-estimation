import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

# Paramètres communs
fs = 1024  # Fréquence d'échantillonnage (Hz)
N = 7500  # Nombre total d'échantillons
t = np.arange(N) / fs  # Axe temporel
sigma = 0.08  # Écart-type du bruit
A1 = np.sqrt(2)  # Amplitude sinusoïde 1
A2 = np.sqrt(2) / 100  # Amplitude sinusoïde 2
f1 = 140  # Fréquence de la première sinusoïde (Hz)
f2 = 180  # Fréquence de la deuxième sinusoïde (Hz)

# 1. Génération des signaux
# Bruit blanc
bruit = np.random.normal(0, sigma, N)

# Signal2
signal2 = A1 * np.sin(2 * np.pi * f1 * t) + A2 * np.sin(2 * np.pi * f2 * t) + bruit

# Signal1 : Filtre de Chebyshev appliqué sur le bruit
b = [0.0154, 0.0461, 0.0461, 0.0154]
a = [1, -1.9903, 1.5717, -0.458]
signal1 = lfilter(b, a, bruit)

# Fonctions pour calculer périodogrammes et fenêtres
def apply_window(segment, window_type="hamming"):
    if window_type == "hamming":
        window = np.hamming(len(segment))
    elif window_type == "hann":
        window = np.hanning(len(segment))
    elif window_type == "rectangular":
        window = np.ones(len(segment))
    elif window_type == "blackman":
        window = np.blackman(len(segment))
    else:
        raise ValueError("Type de fenêtre inconnu")
    return segment * window

def periodogram(signal, fs):
    N = len(signal)
    freqs = np.fft.fftfreq(N, d=1/fs)[:N//2]
    fft_vals = np.fft.fft(signal)[:N//2]
    Pxx = (np.abs(fft_vals) ** 2) / N
    return freqs, Pxx

def averaged_periodogram_with_window(signal, fs, segment_length, window_type="hamming"):
    K = len(signal) // segment_length
    Pxx_segments = []
    for k in range(K):
        segment = signal[k * segment_length:(k + 1) * segment_length]
        segment_windowed = apply_window(segment, window_type)
        freqs, Pxx_segment = periodogram(segment_windowed, fs)
        Pxx_segments.append(Pxx_segment)
    Pxx_avg = np.mean(Pxx_segments, axis=0)
    return freqs, Pxx_avg

def exact_dsp(signal_type, freqs, fs):
    if signal_type == "Signal1 (filtré)":
        w = 2 * np.pi * freqs / fs
        z = np.exp(1j * w)
        H = (0.0154 + 0.0461 / z + 0.0461 / z**2 + 0.0154 / z**3) / \
            (1 - 1.9903 / z + 1.5717 / z**2 - 0.458 / z**3)
        return np.abs(H)**2 * sigma**2
    elif signal_type == "Bruit blanc":
        return np.full_like(freqs, sigma**2)
    elif signal_type == "Signal2":
        dsp = np.full_like(freqs, sigma**2)
        dsp[np.abs(freqs - f1) < fs / len(freqs)] += 1
        dsp[np.abs(freqs - f2) < fs / len(freqs)] += 0.0001
        return dsp
    else:
        raise ValueError("Signal type inconnu")

# 3. Calculs et affichages
signals = {
    "Signal1 (filtré)": signal1,
    "Bruit blanc": bruit,
    "Signal2": signal2
}

segment_lengths = [512, 256, 128, 64]
window_types = ["hamming", "hann", "rectangular", "blackman"]

for signal_name, signal in signals.items():
    plt.figure(figsize=(14, 7))
    for i, window_type in enumerate(window_types):
        plt.subplot(2, 2, i + 1)
        
        # Périodogramme simple pour la fenêtre courante
        freqs_simple, Pxx_simple = periodogram(apply_window(signal[:1050], window_type), fs)
        plt.plot(freqs_simple, 10 * np.log10(Pxx_simple), linestyle='--', label=f"Périodogramme simple")

        # Périodogrammes moyennés
        for segment_length in segment_lengths:
            freqs, Pxx_avg = averaged_periodogram_with_window(signal, fs, segment_length, window_type)
            plt.plot(freqs, 10 * np.log10(Pxx_avg), label=f'{segment_length} points')

        # Modèle exact
        freqs_exact = np.fft.fftfreq(segment_lengths[0], d=1/fs)[:segment_lengths[0]//2]
        dsp_exact = exact_dsp(signal_name, freqs_exact, fs)
        plt.plot(freqs_exact, 10 * np.log10(dsp_exact), linestyle='dotted', color='black', label='Modèle exact')

        plt.title(f"Fenêtre {window_type.capitalize()}")
        plt.xlabel("Fréquence (Hz)")
        plt.ylabel("Amplitude (dB)")
        plt.legend(loc="best")
        plt.grid()

    plt.suptitle(f"Périodogrammes moyennés et simples pour {signal_name}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
