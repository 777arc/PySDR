import numpy as np
import matplotlib.pyplot as plt

# This script generates a figure showing the Cramer-Rao Lower Bound (CRLB) on
# time-delay estimation accuracy as a function of SNR, for three different
# signal bandwidths.  The bound on the variance of any unbiased delay estimate is
#
#     var(tau_hat) >= 1 / (8 * pi^2 * beta^2 * T * gamma)
#
# where beta is the RMS (Gabor) bandwidth of the signal, T is the integration
# time, and gamma is an effective SNR factor.  We convert the resulting delay
# standard deviation into a ranging error in meters (multiplying by the speed of
# light) since that is the more intuitive quantity for localization.

c = 3e8           # speed of light [m/s]
T = 100e-6        # integration time [s]

snr_db = np.linspace(0, 30, 200)   # SNR sweep [dB]
gamma = 10 ** (snr_db / 10)        # effective SNR factor (linear)

# For band-limited noise that is flat from -B/2 to +B/2, the RMS bandwidth is
# beta = B / sqrt(12).  We show three signal bandwidths.
bandwidths = [1e6, 10e6, 50e6]   # signal bandwidths [Hz]

fig, ax = plt.subplots(figsize=(8, 5))
for B in bandwidths:
    beta = B / np.sqrt(12)                                   # RMS bandwidth [Hz]
    var_tau = 1.0 / (8 * np.pi**2 * beta**2 * T * gamma)     # delay variance [s^2]
    range_std = c * np.sqrt(var_tau)                         # ranging error [m]
    ax.semilogy(snr_db, range_std, label=f'{B/1e6:.0f} MHz bandwidth')

ax.set_xlabel('SNR [dB]')
ax.set_ylabel('Ranging error (CRLB) [m]')
ax.grid(True, which='both', alpha=0.4)
ax.legend()
ax.set_xlim(snr_db[0], snr_db[-1])
fig.savefig('../_images/tdoa_cramer_rao.svg', bbox_inches='tight')
plt.show()
