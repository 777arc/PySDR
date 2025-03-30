import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.animation import FuncAnimation

sample_rate = 1e6
d = 0.5 # half wavelength spacing
N = 10000 # number of samples to simulate
t = np.arange(N)/sample_rate # time vector

# more complex scenario taken from DOA code
Nr = 8 # 8 elements
theta1 = 30 / 180 * np.pi # convert to radians
theta2 = 60 / 180 * np.pi
theta3 = -60 / 180 * np.pi
s1 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta1)).reshape(-1,1) # 8x1
s2 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta2)).reshape(-1,1)
s3 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta3)).reshape(-1,1)
# we'll use 3 different frequencies
if False:
    soi = np.exp(2j*np.pi*0.01e6*t).reshape(1,-1) # 1xN
else:
    gold_code = np.array([-1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, 1], dtype=complex)
    soi_samples_per_symbol = 8
    soi = np.repeat(gold_code, soi_samples_per_symbol) # Gold code is 31 bits, so 31*8 = 248 samples
    num_sequence_repeats = int(N / soi.shape[0]) + 1 # number of times to repeat the sequence
    soi = np.tile(soi, num_sequence_repeats) # repeat the sequence
    soi = soi[:N] # trim to N samples
    soi = soi.reshape(1, -1) # 1xN

tone2 = np.exp(2j*np.pi*0.02e6*t).reshape(1,-1)
tone3 = np.exp(2j*np.pi*0.03e6*t).reshape(1,-1)

print(np.var(soi))
print(np.var(tone2))
print(np.var(tone3))

r = s1 @ soi + s2 @ tone2 + 0.1 * s3 @ tone3
n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
r = r + 0.05*n # 8xN


# theta is the direction of interest, in radians, and r is our received signal
def w_mvdr(theta, r):
    s = np.exp(-2j * np.pi * d * np.arange(r.shape[0]) * np.sin(theta)) # steering vector in the desired direction theta
    s = s.reshape(-1,1) # make into a column vector (size 3x1)
    R = np.cov(r) # Calc covariance matrix. gives a Nr x Nr covariance matrix of the samples
    Rinv = np.linalg.pinv(R) # 3x3. pseudo-inverse tends to work better than a true inverse
    w = (Rinv @ s)/(s.conj().T @ Rinv @ s) # MVDR/Capon equation! numerator is 3x3 * 3x1, denominator is 1x3 * 3x3 * 3x1, resulting in a 3x1 weights vector
    return w

def power_mvdr(theta, r):
    s = np.exp(-2j * np.pi * d * np.arange(r.shape[0]) * np.sin(theta)) # steering vector in the desired direction theta_i
    s = s.reshape(-1,1) # make into a column vector (size 3x1)
    #R = (r @ r.conj().T)/r.shape[1] # Calc covariance matrix. gives a Nr x Nr covariance matrix of the samples
    R = np.cov(r)
    Rinv = np.linalg.pinv(R) # 3x3. pseudo-inverse tends to work better than a true inverse
    return 1/(s.conj().T @ Rinv @ s).squeeze()

theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # 1000 different thetas between -180 and +180 degrees
results = []
for theta_i in theta_scan:
    #w = w_mvdr(theta_i, r) # 3x1
    #r_weighted = w.conj().T @ r # apply weights
    #power_dB = 10*np.log10(np.var(r_weighted)) # power in signal, in dB so its easier to see small and large lobes at the same time
    #results.append(power_dB)
    results.append(10*np.log10(power_mvdr(theta_i, r))) # compare to using equation for MVDR power, should match, SHOW MATH OF WHY THIS HAPPENS!
results -= np.max(results) # normalize

if False:
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta_scan, results) # MAKE SURE TO USE RADIAN FOR POLAR
    ax.set_theta_zero_location('N') # make 0 degrees point up
    ax.set_theta_direction(-1) # increase clockwise
    ax.set_rlabel_position(30)  # Move grid labels away from other labels
    ax.set_thetamin(-90)
    ax.set_thetamax(90) 
    plt.show()
    #fig.savefig('../_images/doa_complex_scenario.svg', bbox_inches='tight')

w_soi = w_mvdr(theta1, r)
w_soi = w_soi.reshape(-1) # make into a row vector
print(w_soi)

N_fft = 1024
w_soi = np.conj(w_soi)
w_padded = np.concatenate((w_soi, np.zeros(N_fft - Nr))) # zero pad to N_fft elements to get more resolution in the FFT
w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) # magnitude of fft in dB
w_fft_dB -= np.max(w_fft_dB) # normalize to 0 dB at peak
theta_bins = np.arcsin(np.linspace(-1, 1, N_fft)) # in radians
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta_bins, w_fft_dB) # MAKE SURE TO USE RADIAN FOR POLAR
ax.set_theta_zero_location('N') # make 0 degrees point up
ax.set_theta_direction(-1) # increase clockwise
ax.set_rlabel_position(55)  # Move grid labels away from other labels
ax.set_thetamin(-90) # only show top half
ax.set_thetamax(90)
ax.set_ylim([-30, 1]) # because there's no noise, only go down 30 dB
plt.show()

