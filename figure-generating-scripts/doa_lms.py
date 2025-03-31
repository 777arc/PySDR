import numpy as np
import matplotlib.pyplot as plt

sample_rate = 1e6
d = 0.5 # half wavelength spacing
N = 10000 # number of samples to simulate
t = np.arange(N)/sample_rate # time vector

Nr = 8 # elements
theta_soi = 30 / 180 * np.pi # convert to radians
theta2    = 60 / 180 * np.pi
theta3   = -60 / 180 * np.pi
s1 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_soi)).reshape(-1,1) # 8x1
s2 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta2)).reshape(-1,1)
s3 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta3)).reshape(-1,1)


if False:
    # SOI is a tone
    soi = np.exp(2j*np.pi*0.01e6*t).reshape(1,-1) # 1xN
else:
    # SOI is a gold code, repeated
    
    # Length 31
    #gold_code = np.array([-1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, 1], dtype=complex)
    
    # Length 127
    gold_code = np.array([-1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1])
    
    soi_samples_per_symbol = 8
    soi = np.repeat(gold_code, soi_samples_per_symbol) # Gold code is 31 bits, so 31*8 = 248 samples
    num_sequence_repeats = int(N / soi.shape[0]) + 1 # number of times to repeat the sequence
    soi = np.tile(soi, num_sequence_repeats) # repeat the sequence to fill simulated time
    soi = soi[:N] # trim to N samples
    soi = soi.reshape(1, -1) # 1xN

# Interference, eg tone jammers, from different directions from the SOI
tone2 = np.exp(2j*np.pi*0.02e6*t).reshape(1,-1)
tone3 = np.exp(2j*np.pi*0.03e6*t).reshape(1,-1)

# Simulate received signal
r = s1 @ soi + s2 @ tone2 + s3 @ tone3
n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
r = r + 0.05*n # 8xN

# LMS, not knowing the direction of SOI but knowing the soi signal itself
# TODO add in a random time delay to start of soi
mu = 0.001 # LMS step size
w_lms = np.random.randn(Nr, 1) + 1j*np.random.randn(Nr, 1) # random weights

# Loop through received samples
error_log = []
for i in range(N):
    r_sample = r[:, i].reshape(-1, 1) # 8x1
    soi_sample = soi[0, i] # scalar
    y = w_lms.conj().T @ r_sample # apply the weights (output is a scalar)
    y = y.squeeze() # make it a scalar
    error = soi_sample - y
    error_log.append(np.abs(error))
    w_lms += mu * error * r_sample # weights are still 8x1
    w_lms /= np.linalg.norm(w_lms) # normalize

plt.figure("Error Log")
plt.plot(error_log)
plt.xlabel('Iteration')
plt.ylabel('Mean Square Error')
plt.grid()

# Visualize the weights
N_fft = 1024
w_lms = w_lms.reshape(-1) # make into a row vector
w_lms = w_lms.conj()
w_padded = np.concatenate((w_lms, np.zeros(N_fft - Nr))) # zero pad to N_fft elements to get more resolution in the FFT
w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) # magnitude of fft in dB
w_fft_dB -= np.max(w_fft_dB) # normalize to 0 dB at peak
theta_bins = np.arcsin(np.linspace(-1, 1, N_fft)) # in radians
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta_bins, w_fft_dB)
ax.plot([theta_soi, theta_soi], [-30, 1],'g--')
ax.plot([theta2, theta2], [-30, 1],'r--')
ax.plot([theta3, theta3], [-30, 1],'r--')
ax.set_theta_zero_location('N') # type: ignore # make 0 degrees point up
ax.set_theta_direction(-1) # type: ignore # increase clockwise
ax.set_rlabel_position(55) # type: ignore # Move grid labels away from other labels
ax.set_thetamin(-90) # type: ignore # only show top half
ax.set_thetamax(90) # type: ignore
ax.set_ylim((-30, 1)) # because there's no noise, only go down 30 dB
plt.show()

