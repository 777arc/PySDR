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
    # SOI is BPSK
    sps = 8
    num_symbols = int(N/sps)
    bits = np.random.randint(0, 2, num_symbols) # Our data to be transmitted, 1's and 0's
    bpsk = np.array([])
    for bit in bits:
        pulse = np.zeros(sps)
        pulse[0] = bit*2-1 # set the first value to either a 1 or -1
        bpsk = np.concatenate((bpsk, pulse)) # add the 8 samples to the signal
    num_taps = 101 # for our RRC filter
    beta = 0.35
    t_bpsk = np.arange(num_taps) - (num_taps-1)//2
    h = np.sinc(t_bpsk/sps) * np.cos(np.pi*beta*t_bpsk/sps) / (1 - (2*beta*t_bpsk/sps)**2)
    bpsk = np.convolve(bpsk, h) # Filter our signal, in order to apply the pulse shaping
    soi = bpsk[0:N] # bspk will be a few samples too long because of pulse shaping filter
    soi = soi.reshape(1, -1) # 1xN

# Interference, eg tone jammers, from different directions from the SOI
tone2 = np.exp(2j*np.pi*0.02e6*t).reshape(1,-1)
tone3 = np.exp(2j*np.pi*0.03e6*t).reshape(1,-1)

# Simulate received signal
r = s1 @ soi + s2 @ tone2 + s3 @ tone3
#r = s1 @ soi
n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
r = r + 5*n # 8xN

mu = 0.05
w = np.random.randn(Nr, 1) + 1j*np.random.randn(Nr, 1) # random weights
#w = np.zeros((Nr, 1), dtype=np.complex128) # zero weights

'''
# Loop through received samples
error_log = []
for i in range(N):
    r_sample = r[:, i].reshape(-1, 1) # 8x1
    y = w.conj().T @ r_sample # apply the weights (output is a scalar)
    y = y.squeeze() # make it a scalar
    error = y - y*np.abs(y)**2 # Constant Modulus error
    error_log.append(np.abs(error))
    w += mu * r_sample * error # Constant Modulus update. weights are still 8x1. other code had conj(error) but it screwed it up for me
    w /= np.linalg.norm(w) # normalize
'''


# Use all samples every iteration
error_log = []
for _ in range(100):
    y = w.conj().T @ r # apply the weights (output is 1xN)
    error = np.conj(soi - y) # 1xN
    error = y - y*np.abs(y)**2 # Constant Modulus error
    error_log.append(np.mean(np.abs(error)))
    w += mu * r @ error.conj().T # weights are still 8x1
    w /= np.linalg.norm(w) # normalize


plt.figure("Error Log")
plt.plot(error_log)
plt.xlabel('Iteration')
plt.ylabel('Mean Square Error')
plt.grid()

# Visualize the weights
N_fft = 1024
w = w.reshape(-1) # make into a row vector
w = w.conj()
w_padded = np.concatenate((w, np.zeros(N_fft - Nr))) # zero pad to N_fft elements to get more resolution in the FFT
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
