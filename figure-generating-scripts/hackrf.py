from python_hackrf import pyhackrf  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import time

# Testing the hackrf_transfer output (doesnt involve using hackrf python API)
if False:
    center_freq = 100e6
    sample_rate = 10e6
    samples = np.fromfile('out.iq', dtype=np.int8)
    samples = samples[::2] + 1j * samples[1::2]
    print(len(samples))
    print(samples[0:10])
    print(np.max(samples))

    fft_size = 2048
    num_rows = len(samples) // fft_size
    spectrogram = np.zeros((num_rows, fft_size))
    for i in range(num_rows):
        spectrogram[i, :] = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples[i * fft_size:(i+1) * fft_size]))) ** 2)
    extent = [(center_freq + sample_rate / -2) / 1e6, (center_freq + sample_rate / 2) / 1e6, len(samples) / sample_rate, 0]

    plt.figure(0)
    plt.imshow(spectrogram, aspect='auto', extent=extent)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Time [s]")

    plt.figure(1)
    plt.plot(np.real(samples[0:10000]))
    plt.plot(np.imag(samples[0:10000]))
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend(["Real", "Imaginary"])

    plt.show()
    exit()

# Testing that the python bindings installed
if False:
    from python_hackrf import pyhackrf  # type: ignore
    pyhackrf.pyhackrf_init()
    sdr = pyhackrf.pyhackrf_open()
    sdr.pyhackrf_set_sample_rate(10e6)
    sdr.pyhackrf_set_antenna_enable(False)
    sdr.pyhackrf_set_freq(100e6)
    sdr.pyhackrf_set_amp_enable(False)
    sdr.pyhackrf_set_lna_gain(30) # LNA gain - 0 to 40 dB in 8 dB steps
    sdr.pyhackrf_set_vga_gain(50) # VGA gain - 0 to 62 dB in 2 dB steps
    sdr.pyhackrf_close()
    
    exit()


# These settings should match the hackrf_transfer example used in the textbook, and the resulting waterfall should look about the same
recording_time = 1  # seconds
center_freq = 100e6  # Hz
sample_rate = 10e6
baseband_filter = 7.5e6
lna_gain = 30 # 0 to 40 dB in 8 dB steps
vga_gain = 50 # 0 to 62 dB in 2 dB steps


pyhackrf.pyhackrf_init()
sdr = pyhackrf.pyhackrf_open()

allowed_baseband_filter = pyhackrf.pyhackrf_compute_baseband_filter_bw_round_down_lt(baseband_filter) # calculate the supported bandwidth relative to the desired one

sdr.pyhackrf_set_sample_rate(sample_rate)
sdr.pyhackrf_set_baseband_filter_bandwidth(allowed_baseband_filter)
sdr.pyhackrf_set_antenna_enable(False)  # It seems this setting enables or disables power supply to the antenna port. False by default. the firmware auto-disables this after returning to IDLE mode

sdr.pyhackrf_set_freq(center_freq)
sdr.pyhackrf_set_amp_enable(False)  # False by default
sdr.pyhackrf_set_lna_gain(lna_gain)  # LNA gain - 0 to 40 dB in 8 dB steps
sdr.pyhackrf_set_vga_gain(vga_gain)  # VGA gain - 0 to 62 dB in 2 dB steps

print(f'center_freq: {center_freq} sample_rate: {sample_rate} baseband_filter: {allowed_baseband_filter}')

num_samples = int(recording_time * sample_rate)
samples = np.zeros(num_samples, dtype=np.complex64)
last_idx = 0

def rx_callback(device, buffer, buffer_length, valid_length):  # this callback function always needs to have these four args
    global samples, last_idx

    accepted = valid_length // 2
    accepted_samples = buffer[:valid_length].astype(np.int8) # -128 to 127
    accepted_samples = accepted_samples[0::2] + 1j * accepted_samples[1::2]  # Convert to complex type (de-interleave the IQ)
    accepted_samples /= 128 # -1 to +1
    samples[last_idx: last_idx + accepted] = accepted_samples

    last_idx += accepted

    return 0

samples = samples[100000:] # get rid of the first 100k samples just to be safe, due to transients

sdr.set_rx_callback(rx_callback)
sdr.pyhackrf_start_rx()
print('is_streaming', sdr.pyhackrf_is_streaming())

time.sleep(recording_time)

sdr.pyhackrf_stop_rx()
sdr.pyhackrf_close()

fft_size = 2048
num_rows = len(samples) // fft_size
spectrogram = np.zeros((num_rows, fft_size))
for i in range(num_rows):
    spectrogram[i, :] = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples[i * fft_size:(i+1) * fft_size]))) ** 2)
extent = [(center_freq + sample_rate / -2) / 1e6, (center_freq + sample_rate / 2) / 1e6, len(samples) / sample_rate, 0]

plt.figure(0)
plt.imshow(spectrogram, aspect='auto', extent=extent) # type: ignore
plt.xlabel("Frequency [MHz]")
plt.ylabel("Time [s]")

plt.figure(1)
plt.plot(np.real(samples[0:10000]))
plt.plot(np.imag(samples[0:10000]))
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend(["Real", "Imaginary"])

plt.show()