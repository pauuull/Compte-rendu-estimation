import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit
import os

# Construire le chemin relatif vers le fichier "signal"
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")
signal_path = os.path.join(data_dir, "signal")

# Charger le fichier signal
signal = np.fromfile(signal_path, dtype=np.float32)

# Paramètres
fs = 1000  # Fréquence d'échantillonnage en Hz 
lowcut = 1  # Borne basse du passe-bande
highcut = 480  # Borne haute du passe-bande
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

# Calcul de la TFD
freqs_filtered, fft_vals_filtered = compute_fft(filtered_signal, fs)

# Fonction de modélisation : plateau suivi d'une décroissance exponentielle
def noise_model_decay(f, a, b, d):
    """
    Modèle :
    - Plateau constant (a) jusqu'à une fréquence b.
    - Décroissance exponentielle contrôlée par le paramètre d au-delà de b.
    """
    return np.where(f <= b, a, a * np.exp(-d * (f - b)))


# Moyenne glissante sur la TFD
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

window_size = 20  # Taille de la fenêtre pour la moyenne glissante
fft_smoothed = moving_average(fft_vals_filtered, window_size)

# Ajustement du modèle 
popt, _ = curve_fit(
    noise_model_decay,
    freqs_filtered,
    fft_smoothed,
    p0=[55, 100, 0.01],  # Guesses initiaux pour [a, b, d]
)

# Paramètres ajustés
a_fit, b_fit, d_fit = popt

# Tracé des résultats
plt.figure(figsize=(10, 6))
plt.plot(freqs_filtered, fft_vals_filtered, label="TFD originale", alpha=0.5)
plt.plot(freqs_filtered, fft_smoothed, label="TFD lissée (moyenne glissante)", color="orange")
plt.plot(freqs_filtered, noise_model_decay(freqs_filtered, *popt),
         label=f"Modèle ajusté : plateau={a_fit:.2f}, seuil={b_fit:.2f} Hz, vitesse={d_fit:.4f}", color="red")
plt.title("Modélisation de l'amplitude moyenne du bruit avec vitesse de décroissance")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude moyenne")
plt.legend()
plt.grid()
plt.show()