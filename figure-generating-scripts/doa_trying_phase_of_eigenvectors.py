# TLDR- works fine for 1 signal, but if there are more then the signals need to be at different amplitudes

import numpy as np
import matplotlib.pyplot as plt

sample_rate = 1e6
d = 0.5 # half wavelength spacing
N = 10000 # number of samples to simulate
t = np.arange(N)/sample_rate # time vector

Nr = 8 # elements
theta1 =  15 / 180 * np.pi
theta2 =  60 / 180 * np.pi
theta3 = -50 / 180 * np.pi

tone1 = 1.0 * np.exp(2j*np.pi*0.0173e6*t).reshape(1,-1)
tone2 = 0.5 * np.exp(2j*np.pi*0.0257e6*t).reshape(1,-1)
tone3 = 0.25 * np.exp(2j*np.pi*0.0312e6*t).reshape(1,-1)

s1 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta1)).reshape(-1,1) # 8x1
s2 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta2)).reshape(-1,1)
s3 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta3)).reshape(-1,1)

# Simulate received signal
r = s1 @ tone1 + s2 @ tone2 + s3 @ tone3
n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
r = r + 0.00001*n # 8xN

# Compute covariance matrix
R = r @ r.conj().T / N

# Eigenvalue decomposition
w, V = np.linalg.eig(R)
idx = np.argsort(np.abs(w))[::-1]
w = w[idx]
V = V[:, idx]

if False:
    # Plot eigenvalues
    plt.figure()
    plt.plot(np.sort(np.abs(w))[::-1], 'k*')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.show()

for i in range(3):
    phases = np.angle(V[:, i])
    # find phase between adjacent elements
    phase_diffs = []
    for i in range(len(phases)-1):
        phase_diffs.append(phases[i+1] - phases[i])
    phase_diffs = np.array(phase_diffs)
    phase_diffs = np.mod(phase_diffs + np.pi, 2*np.pi) - np.pi # make them all between -np.pi and np.pi
    print(phase_diffs)
    phase_diff = np.mean(phase_diffs)

    # Convert to AoA
    result = np.arcsin(phase_diff / (-2*np.pi*d))
    print(np.rad2deg(result))
