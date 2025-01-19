import numpy as np
import matplotlib.pyplot as plt

def autocorr_blackman_tukey(x, M):
    N = len(x)
    gamma_hat = [1 / (N - abs(k)) * np.sum(x[:N - abs(k)] * x[abs(k):]) for k in range(-M, M + 1)]
    return np.array(gamma_hat)

def dsp_blackman_tukey(gamma_hat, M):
    gamma_hat_reshaped = np.zeros(2 * M + 1)
    gamma_hat_reshaped[:M + 1] = gamma_hat[M:]
    gamma_hat_reshaped[M+1:] = gamma_hat[:M]
    dft = np.fft.fft(gamma_hat_reshaped)
    return np.real(dft[:M + 1])

# Paramètres
N = 1000
M = 500
variance = 0.1
K = 1000

signal = np.sqrt(variance) * np.random.randn(N * K).reshape(K, N)

frequences = np.linspace(0, 0.5, M + 1)

# Calcul des DSP pour chaque tranche
dsps = np.array([dsp_blackman_tukey(autocorr_blackman_tukey(signal[i], M), M) for i in range(K)])

# Moyenne et Variance
dsp_moyenne = np.mean(dsps, axis=0)
dsp_variance = np.var(dsps, axis=0)

# Affichage
plt.figure(figsize=(14, 7))

# Affichage pour une tranche particulière
plt.subplot(2, 2, 1)
dsp_example = dsp_blackman_tukey(autocorr_blackman_tukey(signal[0], M), M)
plt.plot(frequences, dsp_example, label="DSP estimée (exemple)")
plt.title("DSP estimée (une tranche)")
plt.xlabel("Fréquence réduite")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# Moyenne de la DSP
plt.subplot(2, 2, 2)
plt.plot(frequences, dsp_moyenne, label="Moyenne de la DSP")
plt.title("Moyenne de la DSP estimée")
plt.xlabel("Fréquence réduite")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# Variance de la DSP
plt.subplot(2,2, 3)
plt.plot(frequences, dsp_variance, label="Variance de la DSP", color="orange")
plt.title("Variance de la DSP estimée")
plt.xlabel("Fréquence réduite")
plt.ylabel("Variance")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()