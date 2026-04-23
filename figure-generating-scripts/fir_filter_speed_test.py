import numpy as np
from scipy.signal import convolve, fftconvolve, oaconvolve, lfilter
import matplotlib.pyplot as plt
import timeit

# Signal length (held constant across methods)
N = 10000
rng = np.random.default_rng(0)
x = rng.standard_normal(N) + 1j * rng.standard_normal(N) # complex signal

tap_sizes = [11, 101, 1001, 10001]

methods = {
    'np.convolve':                 lambda h, x: np.convolve(h, x),
    'scipy.signal.convolve':       lambda h, x: convolve(h, x),
    'scipy.signal.fftconvolve':    lambda h, x: fftconvolve(h, x),
    'scipy.signal.oaconvolve':     lambda h, x: oaconvolve(h, x),
    'scipy.signal.lfilter':        lambda h, x: lfilter(h, 1, x),
}

TIME_BUDGET = 0.1 # seconds per (method, tap-size) measurement

print(f'Input signal length: {N} complex samples\n')
header = f"{'method':<28}" + ''.join(f'{"taps="+str(t):>14}' for t in tap_sizes)
print(header)
print('-' * len(header))

results = {name: [] for name in methods}
for num_taps in tap_sizes:
    h = rng.standard_normal(num_taps) # real-valued FIR taps
    for name, fn in methods.items():
        # Warm-up doubles as a timing probe to pick a sensible trial count
        t0 = timeit.default_timer()
        fn(h, x)
        probe = timeit.default_timer() - t0
        trials = max(1, min(50, int(TIME_BUDGET / max(probe, 1e-4))))
        t = timeit.timeit(lambda: fn(h, x), number=trials) / trials
        results[name].append(t * 1e3) # ms per call

for name, times in results.items():
    row = f'{name:<28}' + ''.join(f'{t:>12.3f} ms' for t in times)
    print(row)

# Verify all methods produce (numerically) the same filtered output for one tap size
print('\nSanity check (taps=101): max abs error vs np.convolve')
h = rng.standard_normal(101)
ref = np.convolve(h, x)
for name, fn in methods.items():
    y = fn(h, x)
    # upfirdn / lfilter return different lengths than full convolution
    m = min(len(ref), len(y))
    err = np.max(np.abs(ref[:m] - y[:m]))
    print(f'  {name:<28} {err:.2e}')

# Plot results on a log-log scale so all 4 tap sizes and all methods are visible
plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(9, 6))
for name, times in results.items():
    plt.loglog(tap_sizes, times, 'o-', label=name)
plt.xlabel('Number of taps')
plt.ylabel('Time per call (ms)')
plt.title(f'FIR filter methods, input signal length = {N} complex samples')
plt.grid(True, which='both', ls=':')
plt.legend()
plt.tight_layout()
plt.show()
