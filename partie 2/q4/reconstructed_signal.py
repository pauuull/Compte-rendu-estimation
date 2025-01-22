import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, find_peaks
import os

# Construire le chemin relatif vers le fichier "signal"
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")
signal_path = os.path.join(data_dir, "signal")

# Charger le fichier signal
signal = np.fromfile(signal_path, dtype=np.float32)
# Paramètres
fs = 1000  # Fréquence d'échantillonnage en Hz
lowcut = 10  # Borne basse du passe-bande
highcut = 480  # Borne haute du passe-bande
window_size = 20  # Taille de la fenêtre pour calculer la moyenne locale
x = 5  # Facteur de seuil

N = len(signal)  # Nombre d'échantillons
t = np.arange(N) / fs  # Axe temporel

# Fonction de filtrage passe-bande
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Filtrage du signal
filtered_signal = bandpass_filter(signal, lowcut, highcut, fs)

# Fonction pour calculer la TFD
def compute_fft(signal, fs):
    freqs = np.fft.fftfreq(len(signal), d=1/fs)[:len(signal)//2]
    fft_vals = np.abs(np.fft.fft(signal)[:len(signal)//2])
    return freqs, fft_vals

# Fonction pour calculer la DSP
def compute_psd(signal, fs):
    freqs, psd = welch(signal, fs=fs, nperseg=1024)
    return freqs, psd

# Calcul des spectres (TFD)
freqs_filtered, fft_vals_filtered = compute_fft(filtered_signal, fs)

# Identification des pics
peaks, _ = find_peaks(fft_vals_filtered)
significant_peaks = []

for peak in peaks:
    lower_bound = max(0, peak - window_size // 2)
    upper_bound = min(len(fft_vals_filtered), peak + window_size // 2)
    local_mean = np.mean(fft_vals_filtered[lower_bound:upper_bound])
    if fft_vals_filtered[peak] >= x * local_mean:
        significant_peaks.append(peak)

# Reconstruction du signal
reconstructed_signal = np.zeros_like(filtered_signal)
for peak in significant_peaks:
    amplitude = fft_vals_filtered[peak] / len(filtered_signal)
    frequency = freqs_filtered[peak]
    phase = 0  # Suppose une phase initiale nulle
    reconstructed_signal += amplitude * np.cos(2 * np.pi * frequency * t)

# Calcul de la DSP du signal reconstruit
freqs_psd_reconstructed, psd_reconstructed = compute_psd(reconstructed_signal, fs)

# Calcul de la DSP du signal filtré
freqs_psd_filtered, psd_filtered = compute_psd(filtered_signal, fs)

# Affichage des résultats
plt.figure(figsize=(12, 7))

# Signal filtré
plt.subplot(4, 1, 1)
plt.plot(t, filtered_signal, label="Signal filtré")
plt.title("Signal filtré (300-480 Hz)")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# Spectre et pics significatifs
plt.subplot(4, 1, 2)
plt.plot(freqs_filtered, fft_vals_filtered, label="Spectre filtré")
plt.scatter(freqs_filtered[significant_peaks], fft_vals_filtered[significant_peaks], color="red", label="Pics significatifs")
plt.title("Spectre filtré (TFD)")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# Signal reconstruit
plt.subplot(4, 1, 3)
plt.plot(t, reconstructed_signal, label="Signal reconstruit")
plt.title("Signal reconstruit à partir des pics significatifs")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# DSP des signaux
plt.subplot(4, 1, 4)
plt.semilogy(freqs_psd_filtered, psd_filtered, label="DSP du signal filtré", alpha=0.7)
plt.semilogy(freqs_psd_reconstructed, psd_reconstructed, label="DSP du signal reconstruit", alpha=0.7)
plt.title("Comparaison des DSP (signal filtré vs signal reconstruit)")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("DSP")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()