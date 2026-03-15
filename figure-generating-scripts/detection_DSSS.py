import numpy as np
import matplotlib.pyplot as plt

# Barker 11 sequence: +1, -1, +1, +1, -1, +1, +1, +1, -1, -1, -1
barker11 = np.array([1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1])
samples_per_chip = 100

# Upsample the sequence to simulate continuous-ish time
sig = np.repeat(barker11, samples_per_chip)

offsets = np.linspace(-1.5, 1.5, 500) # Fractional chip offsets
peaks = []

for offset in offsets:
    # Shift the signal by a fractional number of chips (converted to samples)
    shift_samples = int(offset * samples_per_chip)
    if shift_samples > 0:
        shifted_sig = np.pad(sig, (shift_samples, 0))[:len(sig)]
    elif shift_samples < 0:
        shifted_sig = np.pad(sig, (0, abs(shift_samples)))[abs(shift_samples):]
    else:
        shifted_sig = sig
        
    # Compute normalized correlation at zero lag for this specific offset
    correlation = np.vdot(sig, shifted_sig) / np.vdot(sig, sig)
    peaks.append(np.abs(correlation))

plt.figure(figsize=(10, 5))
plt.plot(offsets, peaks, label='Normalized Correlation', color='blue', linewidth=2)
plt.axvline(0, color='red', linestyle='--', alpha=0.5, label='Perfect Alignment')
plt.title('DSSS Correlation Peak vs. Fractional Chip Timing Offset')
plt.xlabel('Offset (Fraction of a Chip)')
plt.ylabel('Normalized Correlation Peak Magnitude')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend()
plt.savefig('../_images/detection_dsss.svg', bbox_inches='tight')
plt.show()


