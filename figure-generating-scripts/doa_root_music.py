import numpy as np

sample_rate = 1e6
N = 10000  # number of samples to simulate
d = 0.5    # half wavelength spacing
Nr = 8     # number of array elements
t = np.arange(N) / sample_rate

# Simulate three signals at 20, 25, and -40 degrees (same scenario as MUSIC section)
theta1 = 20 / 180 * np.pi
theta2 = 25 / 180 * np.pi
theta3 = -40 / 180 * np.pi
s1 = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta1)).reshape(-1, 1)
s2 = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta2)).reshape(-1, 1)
s3 = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta3)).reshape(-1, 1)
tone1 = np.exp(2j * np.pi * 0.01e6 * t).reshape(1, -1)
tone2 = np.exp(2j * np.pi * 0.02e6 * t).reshape(1, -1)
tone3 = np.exp(2j * np.pi * 0.03e6 * t).reshape(1, -1)
X = s1 @ tone1 + s2 @ tone2 + 0.1 * s3 @ tone3
n = np.random.randn(Nr, N) + 1j * np.random.randn(Nr, N)
X = X + 0.05 * n  # 8xN

num_expected_signals = 3

# Same eigendecomposition as MUSIC
R = np.cov(X)
w, v = np.linalg.eig(R)
eig_val_order = np.argsort(np.abs(w))
v = v[:, eig_val_order]
V = v[:, :Nr - num_expected_signals]  # noise subspace eigenvectors

# Build the Root MUSIC polynomial from diagonals of noise-subspace projection
D = V @ V.conj().T
p = np.zeros(2*Nr - 1, dtype=np.complex128)
for k in range(2*Nr - 1):
    p[k] = np.sum(np.diag(D, k - (Nr - 1)))

# Find roots, keep those inside the unit circle, pick the num_expected_signals roots closest to the unit circle
roots = np.roots(p[::-1])  # np.roots expects highest-degree coefficient first
roots = roots[np.abs(roots) <= 1.0]
roots = roots[np.argsort(-np.abs(roots))]  # sort closest-to-unit-circle first
doa_roots = roots[:num_expected_signals]

# Convert roots to angles in degrees
doas_deg = np.sort(np.arcsin(np.angle(doa_roots) / (2 * np.pi * d)) * 180 / np.pi)

# Print results
print("Estimated DOAs (degrees):", doas_deg)
print("True DOAs (degrees):      [-40.  20.  25.]")