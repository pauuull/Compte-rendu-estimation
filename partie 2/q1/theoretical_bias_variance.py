import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, freqz, windows

# Définition des coefficients du filtre de Chebyshev
numerator = [0.0154, 0.0461, 0.0154]
denominator = [1, -1.9903, 1.5717, -0.458]

# Fonction pour calculer le périodogramme
def periodogram(x, N):
    fft_x = np.fft.fft(x, n=N)
    Pxx = (np.abs(fft_x) ** 2) / N
    freqs = np.fft.fftfreq(N, d=1)
    return freqs[:N // 2], Pxx[:N // 2]  # Fréquences positives uniquement

# Calcul de la réponse fréquentielle et de la DSP théorique
def calculate_theoretical_psd(numerator, denominator, sigma2, N):
    w, h = freqz(numerator, denominator, worN=N)  # Réponse fréquentielle du filtre
    G_squared = np.abs(h)**2  # Module au carré de G(f)
    P_theoretical = sigma2 * G_squared  # DSP théorique
    return w / (2 * np.pi), P_theoretical  # Fréquences normalisées et DSP

# Fonction pour calculer le biais théorique en appliquant une convolution
def calculate_theoretical_bias(theoretical_psd, N):
    # Générer la fenêtre triangulaire W_T
    W_T = windows.triang(N)  # Fenêtre triangulaire
    W_T /= np.sum(W_T)  # Normalisation pour que l'intégrale soit égale à 1
    
    # Approximer δ(ν) par une impulsion de Dirac
    delta = np.zeros_like(W_T)
    delta[len(delta) // 2] = 1  # Créer un pic unitaire au centre de la fenêtre
    
    # Calcul de la fenêtre différencielle W_T - δ(ν)
    diff_window = W_T - delta  # Différence de fenêtres
    
    # Appliquer la convolution entre Γ_x(ν) et (W_T - δ(ν))
    bias_theoretical = np.convolve(theoretical_psd, diff_window, mode='same')  # Convolution
    
    return bias_theoretical


# Paramètres pour les simulations
N = 512 
sigma2 = 0.5  # Variance du bruit blanc
num_simulations = 1000 

# Calcul des fréquences et de la DSP théorique
frequencies, P_theoretical = calculate_theoretical_psd(numerator, denominator, sigma2, N)

# Calcul du biais théorique
bias_theoretical = calculate_theoretical_bias(P_theoretical, N)

# Stockage des périodogrammes pour calculer E[P(f)] et V[P(f)]
Pxx_all = []

# Lancer les simulations
for sim in range(num_simulations):
    # Générer un bruit blanc
    bruit = np.random.normal(0, np.sqrt(sigma2), N)
    
    # Générer le signal filtré
    signal1 = lfilter(numerator, denominator, bruit)
    
    # Périodogramme pour le signal filtré
    _, Pxx_signal = periodogram(signal1, N)
    Pxx_all.append(Pxx_signal)

# Convertir en tableau pour un calcul facile
Pxx_all = np.array(Pxx_all)  # Matrice (num_simulations x len(freqs))

# Calcul de la moyenne et de la variance pour chaque fréquence
Pxx_mean = np.mean(Pxx_all, axis=0)  # Moyenne des périodogrammes
Pxx_variance = np.var(Pxx_all, axis=0)  # Variance des périodogrammes

# Calcul du biais pour chaque fréquence
Pxx_bias = Pxx_mean - P_theoretical[:len(Pxx_mean)]

# Tracer les courbes
plt.figure(figsize=(12, 8))

# DSP théorique
plt.subplot(3, 1, 1)
plt.plot(frequencies[:len(P_theoretical)], P_theoretical, label="DSP Théorique", color="blue")
plt.title("DSP Théorique")
plt.xlabel("Fréquence normalisée")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()

# Biais moyen
plt.subplot(3, 1, 2)
plt.plot(frequencies[:len(Pxx_bias)], Pxx_bias, label="Biais moyen de l'estimateur", color="orange")
plt.plot(frequencies[10:len(bias_theoretical)], bias_theoretical[10:], label="Biais Théorique", color="red", linestyle="dotted")
plt.title("Biais moyen de l'estimateur par fréquence")
plt.xlabel("Fréquence normalisée")
plt.ylabel("Biais")
plt.grid()
plt.legend()

# Variance
plt.subplot(3, 1, 3)
plt.plot(frequencies[:len(Pxx_variance)], Pxx_variance, label="Variance de l'estimateur", color="green")
plt.title("Variance de l'estimateur par fréquence")
plt.xlabel("Fréquence normalisée")
plt.ylabel("Variance")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()