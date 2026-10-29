import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Animation of a boxcar pulse widening in time, and its sinc spectrum narrowing

sample_rate = 100.0  # Hz
N = 4096             # number of time samples / FFT size
t = (np.arange(N) - N // 2) / sample_rate  # centered time axis, in seconds
f = np.fft.fftshift(np.fft.fftfreq(N, d=1 / sample_rate))

A = 1.0
# Pulse width sweeps from short to wide, in seconds. Fewer frames = smaller gif.
widths = np.linspace(0.2, 3.0, 40)

filenames = []
for i, T in enumerate(widths):
    x = np.where(np.abs(t) <= T / 2, A, 0.0)
    X = np.fft.fftshift(np.fft.fft(x))
    X_mag = np.abs(X) / sample_rate

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    plt.subplots_adjust(wspace=0.3)

    ax1.plot(t, x, '-')
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Boxcar Pulse in Time")
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-0.2, 1.3)
    ax1.grid()
    ax1.text(-2.8, 1.05, f"Pulse length = {T:.2f} s", color='r', fontsize=14)

    ax2.plot(f, X_mag, '-')
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Magnitude")
    ax2.set_title("Sinc-Shaped Spectrum")
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-0.3, 3.2)
    ax2.grid()

    filename = '/tmp/boxcar_sinc_' + str(i) + '.png'
    print(i)
    # fixed (non-'tight') bbox so every frame has identical pixel dimensions,
    # which is required for subrectangles=True below
    fig.savefig(filename, dpi=72)
    filenames.append(filename)
    plt.close(fig)

# Create looping animated gif. Quantize each frame to a 64-color palette and let
# Pillow's optimize=True store only the pixels that change between frames (the
# static axes/labels/grid), which shrinks the file a lot.
frames = [Image.open(fn).convert('RGB').quantize(colors=48) for fn in filenames]
frames[0].save('../_images/boxcar_sinc_animation.gif', save_all=True,
               append_images=frames[1:], duration=int(1000 / 15), loop=0,
               optimize=True, disposal=2)
