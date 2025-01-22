import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm  # Pour superposer une densité normale

def estimateur_1k(x, k, N, i):
    x_segment = x[i:i+N] 
    return 1 / (N - abs(k)) * sum([x_segment[n + abs(k)] * x_segment[n] for n in range(N - abs(k))])

def estimateur_2k(x, k, N, i):
    x_segment = x[i:i+N] 
    return 1 / N * sum([x_segment[n + abs(k)] * x_segment[n] for n in range(N - abs(k))])

def gamma_hat(n, x, N, M, i):
    if n == 1:
        return [estimateur_1k(x, k, N, i) for k in range(-M, M + 1)]
    else:
        return [estimateur_2k(x, k, N, i) for k in range(-M, M + 1)]

# Paramètres
N = 1024
M = 512
i = 0
variance = 0.1

# Génération du signal gaussien
signal = np.sqrt(variance) * np.random.randn(N + i + M)

# Estimation de l'autocorrélation
gamma_bt = gamma_hat(1, signal, N, M, i)  
gamma_bartlett = gamma_hat(2, signal, N, M, i) 
k_range = np.arange(-M, M + 1)

# Moments du signal
mean_signal = np.mean(signal)
variance_signal = np.var(signal)

# Tracé des résultats
plt.figure(figsize=(12, 7))

# 1. Histogramme du signal et densité gaussienne théorique
plt.subplot(3, 1, 1)
plt.hist(signal, bins=50, density=True, alpha=0.6, color="blue", label="Histogramme du signal")
x = np.linspace(min(signal), max(signal), 1000)
plt.plot(x, norm.pdf(x, loc=0, scale=np.sqrt(variance)), color="red", label="Densité gaussienne théorique")
plt.title(f"Distribution du signal (Moyenne={mean_signal:.2f}, Variance={variance_signal:.2f})")
plt.xlabel("Amplitude")
plt.ylabel("Densité")
plt.legend()

# 2. Estimation par Blackman-Tukey
plt.subplot(3, 1, 2)
plt.plot(k_range, gamma_bt, label="Blackman-Tukey", color="green")
plt.title("Estimation de l'autocorrélation (Blackman-Tukey)")
plt.xlabel("Lag (k)")
plt.ylabel("Autocorrélation estimée")
plt.legend()

# 3. Estimation par Bartlett
plt.subplot(3, 1, 3)
plt.plot(k_range, gamma_bartlett, label="Bartlett", color="red")
plt.title("Estimation de l'autocorrélation (Bartlett)")
plt.xlabel("Lag (k)")
plt.ylabel("Autocorrélation estimée")
plt.legend()

plt.tight_layout()
plt.show()