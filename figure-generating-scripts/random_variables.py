import numpy as np
import matplotlib.pyplot as plt

# Generate 10,000 samples from standard Gaussian
x = np.random.randn(10000)

# Create histogram to visualize the distribution
plt.hist(x, bins=50, density=True, alpha=0.7, edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Gaussian Distribution (μ=0, σ²=1)')
plt.grid(True)
plt.show()

# Simulation parameters
N = 10000

# Generate standard Gaussian random variables (mean=0, var=1)
x = np.random.randn(N)

# Create different random variables by scaling and shifting
y1 = x                        # mean=0, var=1
y2 = 2 * x                    # mean=0, var=4
y3 = x + 3                    # mean=3, var=1
y4 = 0.5 * x - 1              # mean=-1, var=0.25

# Verify properties
signals = [y1, y2, y3, y4]
labels = ['y1: x', 'y2: 2x', 'y3: x+3', 'y4: 0.5x-1']

for i, (sig, label) in enumerate(zip(signals, labels)):
    print(f"{label}")
    print(f"  Sample mean: {np.mean(sig):.3f}")
    print(f"  Sample variance: {np.var(sig):.3f}")
    print()

# Plot histograms
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for i, (sig, label, ax) in enumerate(zip(signals, labels, axes)):
    ax.hist(sig, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax.set_title(label)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.grid(True)

plt.tight_layout()
plt.show()

# Complex Gaussian noise demonstration
n_complex = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2)

print("Complex Gaussian Noise (unit power):")
print(f"  Real part variance: {np.var(np.real(n_complex)):.3f}")
print(f"  Imag part variance: {np.var(np.imag(n_complex)):.3f}")
print(f"  Total variance: {np.var(n_complex):.3f}")

# Plot on IQ plane
plt.figure(figsize=(6, 6))
plt.plot(np.real(n_complex[:1000]), np.imag(n_complex[:1000]), '.', alpha=0.3)
plt.xlabel('In-phase (I)')
plt.ylabel('Quadrature (Q)')
plt.title('Complex Gaussian Noise on IQ Plane')
plt.grid(True)
plt.axis('equal')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.show()