import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm  # Pour superposer une densité normale

# Définitions des estimateurs biaisé et sans biais
def estimateur_blackman_tukey(x, k, N, i):
    x_segment = x[i:i+N]
    return 1 / (N - abs(k)) * np.sum([x_segment[n + abs(k)] * x_segment[n] for n in range(N - abs(k))])

def estimateur_bartlett(x, k, N, i):
    x_segment = x[i:i+N]
    return 1 / N * np.sum([x_segment[n + abs(k)] * x_segment[n] for n in range(N - abs(k))])

def gamma_hat(estimator_type, x, N, M, i):
    if estimator_type == "Blackman-Tukey":
        return [estimateur_blackman_tukey(x, k, N, i) for k in range(-M, M + 1)]
    elif estimator_type == "Bartlett":
        return [estimateur_bartlett(x, k, N, i) for k in range(-M, M + 1)]

# Paramètres
N = 1024  # Longueur de la fenêtre d'échantillons
M = 512   # Horizon pour l'autocorrélation
i = 0     # Décalage initial
variance = 0.1  # Variance du bruit gaussien

# Génération du signal gaussien centré
signal = np.sqrt(variance) * np.random.randn(N + i + M)

# Estimations des autocorrélations
gamma_bt = gamma_hat("Blackman-Tukey", signal, N, M, i)
gamma_bartlett = gamma_hat("Bartlett", signal, N, M, i)
k_range = np.arange(-M, M + 1)

# Moments du signal
mean_signal = np.mean(signal)
variance_signal = np.var(signal)

# Tracé des résultats
plt.figure(figsize=(12, 7))

# 1. Histogramme du signal avec densité gaussienne théorique
plt.subplot(3, 1, 1)
plt.hist(signal, bins=50, density=True, alpha=0.6, color="blue", label="Histogramme du signal")
x = np.linspace(min(signal), max(signal), 1000)
plt.plot(x, norm.pdf(x, loc=0, scale=np.sqrt(variance)), color="red", label="Densité gaussienne théorique")
plt.title(f"Distribution du signal (Moyenne={mean_signal:.2e}, Variance={variance_signal:.2e})")
plt.xlabel("Amplitude")
plt.ylabel("Densité")
plt.legend()

# 2. Estimation par Blackman-Tukey
plt.subplot(3, 1, 2)
plt.plot(k_range, gamma_bt, label="Blackman-Tukey", color="green")
plt.axhline(y=gamma_bt[M], color="black", linestyle="--", label=f"γ(0)={gamma_bt[M]:.2e}")
plt.title("Estimation de l'autocorrélation (Blackman-Tukey)")
plt.xlabel("Lag (k)")
plt.ylabel("Autocorrélation estimée")
plt.legend()

# 3. Estimation par Bartlett
plt.subplot(3, 1, 3)
plt.plot(k_range, gamma_bartlett, label="Bartlett", color="red")
plt.axhline(y=gamma_bartlett[M], color="black", linestyle="--", label=f"γ(0)={gamma_bartlett[M]:.2e}")
plt.title("Estimation de l'autocorrélation (Bartlett)")
plt.xlabel("Lag (k)")
plt.ylabel("Autocorrélation estimée")
plt.legend()

plt.tight_layout()
plt.show()
