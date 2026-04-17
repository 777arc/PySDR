import numpy as np
import adi
import matplotlib.pyplot as plt

sample_rate = 10e6 # Hz
center_freq = 5925e6 # Hz
num_samps = int(10e6) # number of samples returned per call to rx()

sdr = adi.Pluto('ip:192.168.20.1')
#sdr.gain_control_mode_chan0 = 'manual'
#sdr.rx_hardwaregain_chan0 = 70.0 # dB
sdr.gain_control_mode_chan0 = "fast_attack"
sdr.rx_lo = int(center_freq)
sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(sample_rate) # filter width, just set it to the same as sample rate for now
sdr.rx_buffer_size = num_samps

x = sdr.rx() # receive samples off Pluto
x = np.asarray(x) # purely for type hinting and linting
print(np.max(x))

# Spectrogram
if False:
    fft_size = 1024
    num_rows = len(x) // fft_size # // is an integer division which rounds down
    spectrogram = np.zeros((num_rows, fft_size))
    for i in range(num_rows):
        spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)
    # Time starts at the top and goes down, eg sample x[0] will be part of the top row displayed
    plt.imshow(spectrogram, aspect='auto', extent = [sample_rate/-2/1e6, sample_rate/2/1e6, len(x)/sample_rate, 0]) # type: ignore
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Time [s]")
    plt.show()

# PSD
if True:
    psd = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[0:10000])))**2)
    plt.plot(np.linspace(-sample_rate/2, sample_rate/2, len(psd))/1e6, psd)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Power [dB]")
    plt.show()

# Save to file
if True:
    x = np.asarray(x, dtype=np.complex64)
    x.tofile("/tmp/pluto_samples.iq")