from python_hackrf import pyhackrf # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import time

pyhackrf.pyhackrf_init()
sdr = pyhackrf.pyhackrf_open()
#print(sdr.pyhackrf_get_clkin_status()) # can be used to check if you are feeding it a 10 MHz clock
buffer_size = sdr.pyhackrf_get_transfer_buffer_size()
print("buffer_size:", buffer_size) # this is how many samples you'll get with each call to rx_callback

sdr.pyhackrf_set_sample_rate(1e6)
sdr.pyhackrf_set_baseband_filter_bandwidth(1e6)
sdr.pyhackrf_set_antenna_enable(False) # not sure what this does

sdr.pyhackrf_set_freq(100e6)
sdr.pyhackrf_set_amp_enable(False)
sdr.pyhackrf_set_lna_gain(30) # LNA gain- 0 to 40 dB in 8 dB steps
sdr.pyhackrf_set_vga_gain(30) # bandband gain- 0 to 62 dB in 2 dB steps

#sdr.pyhackrf_set_txvga_gain(0)


samples = np.zeros(buffer_size, dtype=np.uint8)

def rx_callback(buffer, buffer_length, valid_length): # this callback function always needs to have these three args
    global samples
    # each call, buffer will be a 1D numpy array filled with np.uint8 (valid_length of them)
    if True:
        samples = buffer[0:valid_length] - 127.5
    else:
        samples = buffer[0:valid_length].astype(np.complex64)
        samples -= 127.5
        samples /= 127.5 # scale to -1 to 1
        samples = samples[::2] + 1j * samples[1::2]
    print(len(samples), valid_length//2)
    print(samples[0:4])
    print(np.max(samples))
    print()
    return 0 # return 0 if it wants to be called again, anything else will stop rx_callback from being called again

sdr.set_rx_callback(rx_callback)

print(sdr.pyhackrf_is_streaming())
sdr.pyhackrf_start_rx()
print(sdr.pyhackrf_is_streaming())

time.sleep(3)

sdr.pyhackrf_stop_rx()

sdr.pyhackrf_close()

plt.plot(samples.real)
plt.show()