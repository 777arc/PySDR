if False:
    from bladerf import _bladerf

    sdr = _bladerf.BladeRF()

    print("Device info:", _bladerf.get_device_list()[0])
    print("libbladeRF version:", _bladerf.version()) # v2.5.0
    print("Firmware version:", sdr.get_fw_version()) # v2.4.0
    print("FPGA version:", sdr.get_fpga_version())   # v0.15.0

    rx_ch = sdr.Channel(0) # ch 0 or 1
    print("sample_rate_range:", rx_ch.sample_rate_range)
    print("bandwidth_range:", rx_ch.bandwidth_range)
    print("frequency_range:", rx_ch.frequency_range)
    print("gain_modes:", rx_ch.gain_modes)
    print("manual gain range:", sdr.get_gain_range(0)) # ch 0 or 1

# --------------

from bladerf import _bladerf
import numpy as np
import matplotlib.pyplot as plt

sdr = _bladerf.BladeRF()
rx_ch = sdr.Channel(0) # ch 0 or 1

sample_rate = 10e6
center_freq = 100e6
gain = 50 # -15 to 60 dB
num_samples = int(1e6)

rx_ch.frequency = center_freq
rx_ch.sample_rate = sample_rate
rx_ch.bandwidth = sample_rate/2
rx_ch.gain_mode = _bladerf.GainMode.Manual
rx_ch.gain = gain

# Setup synchronous stream
sdr.sync_config(layout = _bladerf.ChannelLayout.RX_X1, # or RX_X2
                fmt = _bladerf.Format.SC16_Q11, # int16s
                num_buffers    = 16,
                buffer_size    = 8192,
                num_transfers  = 8,
                stream_timeout = 3500)

# Create receive buffer
bytes_per_sample = 4 # don't change this, it will always use int16s
buf = bytearray(1024 * bytes_per_sample)

# Enable module
print("Starting receive")
rx_ch.enable = True

# Receive loop
x = np.zeros(num_samples, dtype=np.complex64) # storage for IQ samples
num_samples_read = 0
while True:
    if num_samples > 0 and num_samples_read == num_samples:
        break
    elif num_samples > 0:
        num = min(len(buf) // bytes_per_sample, num_samples - num_samples_read)
    else:
        num = len(buf) // bytes_per_sample
    sdr.sync_rx(buf, num) # Read into buffer
    samples = np.frombuffer(buf, dtype=np.int16)
    samples = samples[0::2] + 1j * samples[1::2] # Convert to complex type
    samples /= 2048.0 # Scale to -1 to 1 (its using 12 bit ADC)
    x[num_samples_read:num_samples_read+num] = samples[0:num] # Store buf in samples array
    num_samples_read += num

print("Stopping")
rx_ch.enable = False
print(x[0:10]) # look at first 10 IQ samples
print(np.max(x)) # if this is close to 1, you are overloading the ADC, and should reduce the gain

# --------------

# Create spectrogram
fft_size = 2048
num_rows = len(x) // fft_size # // is an integer division which rounds down
spectrogram = np.zeros((num_rows, fft_size))
for i in range(num_rows):
    spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)
extent = [(center_freq + sample_rate/-2)/1e6, (center_freq + sample_rate/2)/1e6, len(x)/sample_rate, 0]
plt.imshow(spectrogram, aspect='auto', extent=extent)
plt.xlabel("Frequency [MHz]")
plt.ylabel("Time [s]")
plt.savefig("../_images/bladerf-waterfall.svg", bbox_inches='tight')
plt.show()

# --------------

'''
if  ( verbosity == "VERBOSE" ):  _bladerf.set_verbosity( 0 )
elif( verbosity == "DEBUG" ):    _bladerf.set_verbosity( 1 )
elif( verbosity == "INFO" ):     _bladerf.set_verbosity( 2 )
elif( verbosity == "WARNING" ):  _bladerf.set_verbosity( 3 )
elif( verbosity == "ERROR" ):    _bladerf.set_verbosity( 4 )
elif( verbosity == "CRITICAL" ): _bladerf.set_verbosity( 5 )
elif( verbosity == "SILENT" ):   _bladerf.set_verbosity( 6 )
'''