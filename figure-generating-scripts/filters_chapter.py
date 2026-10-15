# %%
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import firwin2, convolve, fftconvolve, lfilter

# %%
num_taps = 51
cut_off = 3000  # Hz
sample_rate = 32000  # Hz

# create our low pass filter
h = signal.firwin(num_taps, cut_off, fs=sample_rate)

# plot the impulse response
plt.plot(h, '.-')
plt.show()

# plot the frequency response
H = np.abs(np.fft.fft(h, 1024))  # take the 1024-point FFT and magnitude
H = np.fft.fftshift(H)  # make 0 Hz in the center
w = np.linspace(-sample_rate / 2, sample_rate / 2, len(H))  # x axis
plt.plot(w, H, '.-')
plt.show()

# (h was found using the first code snippet)

# Shift the filter in frequency by multiplying by exp(j*2*pi*f0*t)
f0 = 10e3  # amount we will shift
Ts = 1.0 / sample_rate  # sample period
t = np.arange(0.0, Ts * len(h), Ts)  # time vector. args are (start, stop, step)
exponential = np.exp(2j * np.pi * f0 * t)  # this is essentially a complex sine wave

h_band_pass = h * exponential  # do the shift

# plot impulse response
plt.subplot(121)
plt.plot(np.real(h_band_pass), '.-')
plt.plot(np.imag(h_band_pass), '.-')
plt.legend(['real', 'imag'], loc=1)

# plot the frequency response
H = np.abs(np.fft.fft(h_band_pass, 1024))  # take the 1024-point FFT and magnitude
H = np.fft.fftshift(H)  # make 0 Hz in the center
w = np.linspace(-sample_rate / 2, sample_rate / 2, len(H))  # x axis
plt.subplot(122)
plt.plot(w, H, '.-')
plt.xlabel('Frequency [Hz]')
plt.show()

# %%
# comparing different convolution methods
# Create a test signal, we'll use Gaussian noise
_sample_rate = 1e6 # Hz
_N = 1000 # samples to simulate
_x = np.random.randn(_N) + 1j * np.random.randn(_N)

# Create an FIR filter, same one as 2nd example above
freqs = [0, 100e3, 110e3, 190e3, 200e3, 300e3, 310e3, 500e3]
gains = [1, 1,     0,     0,     0.5,   0.5,   0,     0]
h2 = firwin2(101, freqs, gains, fs=_sample_rate)

# Apply filter using the four different methods
x_numpy = np.convolve(h2, _x)
x_scipy = convolve(h2, _x) # scipys convolve
x_fft_convolve = fftconvolve(h2, _x)
x_lfilter = lfilter(h2, 1, _x) # 2nd arg is always 1 for FIR filters

# Prove they are all giving the same output
print(x_numpy[0:2])
print(x_scipy[0:2])
print(x_fft_convolve[0:2])
print(x_lfilter[0:2])

# %%
# Make plot comparing methods using different taps, this takes a while to run (even longer in the web-based marimo notebook) so it's off by default
if False:
    #_taps_sizes = np.logspace(1, 3, num=100) # 10**x is the start/stop
    _taps_sizes = np.linspace(10, 10000, num=100)
    _N = 100000 # signal length
    #_N = 1000 # signal length
    _x = np.random.randn(_N) + 1j * np.random.randn(_N) # complex signal
    _results = np.zeros((4, len(_taps_sizes)))
    for _method_i in range(4):
        for _i in range(len(_taps_sizes)):
            _h = np.random.randn(int(_taps_sizes[_i])) # real taps
            _num_trials = int(10000 / _taps_sizes[_i])
            #_num_trials = int(1000000 / _taps_sizes[_i])
            _start_t = time.time()
            if _method_i == 0:
                for _ii in range(_num_trials):
                    _ = np.convolve(_h, _x)
            elif _method_i == 1:
                for _ii in range(_num_trials):
                    _ = convolve(_h, _x)
            elif _method_i == 2:
                for _ii in range(_num_trials):
                    _ = fftconvolve(_h, _x)
            elif _method_i == 3:
                for _ii in range(_num_trials):
                    _ = lfilter(_h, 1, _x)
            _results[_method_i, _i] = (time.time() - _start_t)/_num_trials*1e3 # in ms

    # Lots of folks use semilogx, but its nice to see the linear aspect
    plt.plot(_taps_sizes, _results[0,:], '.-')
    plt.plot(_taps_sizes, _results[1,:], '.-')
    plt.plot(_taps_sizes, _results[2,:], '.-')
    plt.plot(_taps_sizes, _results[3,:], '.-')
    plt.title('Input Signal Length: ' + str(_N) + ' samples')
    plt.xlabel('Number of taps')
    plt.ylabel('Time per call (ms)')
    plt.legend(['np.convolve', 'scipy.signal.convolve', 'scipy.signal.fftconvolve', 'scipy.signal.lfilter'])
    #if os.path.isdir('../_images'):
    #    plt.savefig('../_images/convolve_comparison_'+str(_N)+'.svg', bbox_inches='tight')
    plt.show()

# %%
_sample_rate = 1e6

# (h2 was created above)

# Simulate signal comprising of Gaussian noise
_N = 100000 # signal length
_x = np.random.randn(_N) + 1j * np.random.randn(_N) # complex signal

# Save PSD of the input signal
PSD_input = 10*np.log10(np.fft.fftshift(np.abs(np.fft.fft(_x))**2)/len(_x))

# Apply filter
_x = fftconvolve(_x, h2, 'same')

# Look at PSD of the output signal
PSD_output = 10*np.log10(np.fft.fftshift(np.abs(np.fft.fft(_x))**2)/len(_x))
f = np.linspace(-_sample_rate/2/1e6, _sample_rate/2/1e6, len(PSD_output))
plt.plot(f, PSD_input, alpha=0.8)
plt.plot(f, PSD_output, alpha=0.8)
plt.xlabel('Frequency [MHz]')
plt.ylabel('PSD [dB]')
plt.axis([_sample_rate/-2/1e6, _sample_rate/2/1e6, -40, 20])
plt.legend(['Input', 'Output'], loc=1)
plt.grid()
#if os.path.isdir('../_images'):
#    plt.savefig('../_images/fftconvolve.svg', bbox_inches='tight')
plt.show()
