import numpy as np
import adi
import matplotlib.pyplot as plt
import time

sample_rate = 20e6 # Hz
center_freq = 100e6 # Hz

sdr = adi.Pluto("ip:192.168.1.174")
sdr.sample_rate = int(sample_rate)
print("Sample rate:   ", sample_rate)
sdr.rx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
sdr.rx_lo = int(center_freq)
sdr.rx_buffer_size = 1000000 # this is the buffer the Pluto uses to buffer samples

num_rx = 20
start_time = time.time()
for i in range(num_rx):
    samples = sdr.rx() # receive samples off Pluto
print("Rate in python:", (num_rx * sdr.rx_buffer_size)/(time.time() - start_time))