import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import adi

# Spikes after FM demod
# 3.579545e6 Color NTSC
# 4.43361875e6 Color PAL and SECAM
# 6.5025e6 audio carrier (I think the audio is transmitted separately but this is where it would show up after FM demodding the whole thing)
# apparently the audio might show up at 5.5, 6.0, or 6.5 MHz

sample_rate = 10e6

if True:
    sdr = adi.Pluto("ip:192.168.2.1")
    sdr.sample_rate = int(sample_rate)
    sdr.rx_lo = int(5925e6)
    sdr.gain_control_mode_chan0 = "slow_attack"
    sdr.rx_buffer_size = 500000

    # Flush buffer
    for _ in range(10):
        sdr.rx()

    x = sdr.rx()
    x = x.astype(np.complex64)
    x.tofile("/tmp/ntsc_pal_bw.iq")

    # PSD of raw RF
    PSD = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x)))**2 / (len(x)*sample_rate))
    f = np.linspace(sample_rate/-2, sample_rate/2, len(PSD))
    plt.plot(f / 1e6, PSD)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("PSD [dB]")
    plt.show()
    exit()

    # If the signal is clearly there, switch the True to False above

else:
    samples_to_process = 500000
    offset = 100
    filename = '/tmp/ntsc_pal_bw.iq'
    #filename = '/mnt/c/Users/marclichtman/Downloads/ntsc_remy_10MHz_5925Hz_cf32.iq'
    x = np.fromfile(filename, dtype=np.complex64, count=samples_to_process, offset=offset*8)
    print(len(x))

samples_per_line = 508
lines_per_frame = 525
refresh_Hz = 30.0/1.001 # almost exactly 29.97 # not exactly 30 Hz!! makes difference
samples_per_frame = samples_per_line * lines_per_frame // 2 # samples per frame. WHY DO I NEED THE /2?
print("Samples per frame:", samples_per_frame)
line_Hz = refresh_Hz * lines_per_frame
print("Line rate (Hz):", line_Hz)

x_demod = np.angle(x[1:] * np.conj(x[:-1])) # FM demodulation

# Filter out audio from demodded signal
h = signal.firwin(301, 3e6, fs=sample_rate) # for the 10 Mhz recording
x_demod = np.convolve(x_demod, h, 'same')

# Resample luma and chroma to exactly L samples per line
resampling_rate = samples_per_line / (sample_rate / line_Hz)
resampling_rate *= 1.00003 # fixes the drift, not 100% sure where it comes from, perhaps sample clock offset
x_demod = signal.resample(x_demod, int(len(x_demod)*resampling_rate))
print("Resampling rate:", resampling_rate)

# reshape into 2D
x_demod = x_demod[:len(x_demod) - (len(x_demod) % samples_per_line)] # trim to multiple of samples_per_line
frame = x_demod.reshape(-1, samples_per_line) # type: ignore

# Normalize to 0-255 and convert to uint8
frame_norm = frame - np.min(frame)
frame_norm = frame_norm / np.max(frame_norm)
frame_uint8 = (frame_norm * 255).astype(np.uint8)

# Display as single image with fixed scaling
plt.imshow(frame_uint8, cmap='gray', aspect='auto', vmin=0, vmax=255)
plt.axis('off')
plt.show()
