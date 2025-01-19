import numpy as np
import matplotlib.pyplot as plt

# Define biased and unbiased estimators
def estimateur_1k(x, k, N):
    """Unbiased autocorrelation estimator."""
    return 1 / (N - abs(k)) * sum([x[n + abs(k)] * x[n] for n in range(N - abs(k))])

def estimateur_2k(x, k, N):
    """Biased autocorrelation estimator."""
    return 1 / N * sum([x[n + abs(k)] * x[n] for n in range(N - abs(k))])

def gamma_hat(n, x, N, M):
    """Autocorrelation estimation based on the chosen method (1=unbiased, 2=biased)."""
    if n == 1:
        return [estimateur_1k(x, k, N) for k in range(-M, M + 1)]
    else:
        return [estimateur_2k(x, k, N) for k in range(-M, M + 1)]

# Parameters
N = 300      # Number of samples per segment
M = 250      # Maximum lag
K = 20       # Number of segments
variance = 0.1  # Variance of the white Gaussian noise

# Generate signal
signal = np.sqrt(variance) * np.random.randn(N * K).reshape(K, N)  # Multiple segments
signal_single = signal.flatten()  # Full signal (single segment for comparison)

# Theoretical gamma
gamma_th = np.array([variance if k == 0 else 0 for k in range(-M, M + 1)])

# Estimation for multiple segments (averaged results)
estimations_1_moy = np.array([gamma_hat(1, signal[j], N, M) for j in range(K)])
estimations_2_moy = np.array([gamma_hat(2, signal[j], N, M) for j in range(K)])

# Calculate mean, variance, and EQM
mean_1_moy = np.mean(estimations_1_moy, axis=0)
mean_2_moy = np.mean(estimations_2_moy, axis=0)
variance_1_moy = np.var(estimations_1_moy, axis=0)
variance_2_moy = np.var(estimations_2_moy, axis=0)
eqm_1_moy = (mean_1_moy - gamma_th)**2 + variance_1_moy
eqm_2_moy = (mean_2_moy - gamma_th)**2 + variance_2_moy

# Estimation for the full signal (single segment)
estimations_1_single = gamma_hat(1, signal_single, N * K, M)
estimations_2_single = gamma_hat(2, signal_single, N * K, M)

# Variance and EQM for single segment
variance_1_single = np.var(estimations_1_single)
variance_2_single = np.var(estimations_2_single)
eqm_1_single = (estimations_1_single - gamma_th)**2
eqm_2_single = (estimations_2_single - gamma_th)**2

# Visualization
k_range = np.arange(-M, M + 1)

plt.figure(figsize=(18, 5))

# Bias
plt.subplot(1, 3, 1)
plt.plot(k_range, mean_1_moy - gamma_th, label="Blackman-Tukey (Unbiased)", color="green")
plt.plot(k_range, mean_2_moy - gamma_th, label="Bartlett (Biased)", color="red")
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.title(f"Biais avec moyennage (K = {K})")
plt.xlabel("Lag (k)")
plt.ylabel("Biais")
plt.legend()

# Variance
plt.subplot(1, 3, 2)
plt.plot(k_range, variance_1_moy, label="Blackman-Tukey (Unbiased)", color="green")
plt.plot(k_range, variance_2_moy, label="Bartlett (Biased)", color="red")
plt.title(f"Variance avec moyennage (K = {K})")
plt.xlabel("Lag (k)")
plt.ylabel("Variance")
plt.legend()

# EQM
plt.subplot(1, 3, 3)
plt.plot(k_range, eqm_1_moy, label="Blackman-Tukey (Unbiased)", color="green")
plt.plot(k_range, eqm_2_moy, label="Bartlett (Biased)", color="red")
plt.title(f"EQM avec moyennage (K = {K})")
plt.xlabel("Lag (k)")
plt.ylabel("EQM")
plt.legend()

plt.tight_layout()
plt.show()
