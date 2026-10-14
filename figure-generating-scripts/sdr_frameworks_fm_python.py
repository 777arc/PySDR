import numpy as np
from scipy.signal import firwin, lfilter
from scipy.io import wavfile

# Baseline FM receiver used in the SDR Frameworks chapter, uses the same recording as the RDS chapter
# https://github.com/777arc/498x/blob/master/fm_rds_250k_1Msamples.iq?raw=true

sample_rate = 250e3
x = np.fromfile('fm_rds_250k_1Msamples.iq', dtype=np.complex64)

# FM demod: the audio is the change in phase from one sample to the next
x = np.angle(x[1:] * np.conj(x[:-1]))

# Low-pass filter to keep only the mono audio (0 to 15 kHz), then decimate by 5 to get 50 kHz
taps = firwin(101, 15e3, fs=sample_rate)
x = lfilter(taps, 1.0, x)
x = x[::5]

# Normalize, convert to 16-bit ints, and save as a WAV file
x = x / np.max(np.abs(x))
wavfile.write('fm.wav', int(sample_rate/5), (x * 32767).astype(np.int16))
