# %%
import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# If the recording isn't in the current directory it gets downloaded from GitHub, which is also how the web-based marimo notebook gets it
filename = 'ntsc_remy_10MHz_5925Hz_500ksamples_cf32.iq'
if os.path.exists(filename):
    x = np.fromfile(filename, dtype=np.complex64)
else:
    x = np.frombuffer(urllib.request.urlopen('https://raw.githubusercontent.com/777arc/PySDR/refs/heads/master/figure-generating-scripts/' + filename).read(), dtype=np.complex64)

# %%
sample_rate = 10e6
color_subcarrier_freq = 3.579545e6 # NTSC. higher than luma carrier, not relative to center freq
# color_subcarrier_freq = 4.43361875e6 # PAL and SECAM
relative_audio_subcarrier_freq = 3.5e6 # the audio might show up at 5.5, 6.0, or 6.5 MHz

# NTSC constants
samples_per_line = 508
lines_per_frame = 525
refresh_Hz = 30.0/1.001 # almost exactly 29.97 # not exactly 30 Hz!! makes difference

# PAL constants
#samples_per_line = 512
#lines_per_frame = 625 # (576 visible lines)
#refresh_Hz = 25

samples_per_frame = samples_per_line * lines_per_frame // 2 # NTSC's vertical sync repeats every field (half-frame), not every full frame
print("Samples per frame:", samples_per_frame)
line_Hz = refresh_Hz * lines_per_frame

# %%
# PSD of raw RF
plt.rcParams['svg.fonttype'] = 'none'
_PSD = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x)))**2 / (len(x)*sample_rate))
_f = np.linspace(sample_rate/-2, sample_rate/2, len(_PSD))
_n = 4096
_step = len(_PSD) // _n
_PSD_plot = _PSD[:_step*_n].reshape(_n, _step).mean(axis=1)
_f_plot = _f[:_step*_n].reshape(_n, _step).mean(axis=1)
fig1, _ax = plt.subplots()
_ax.plot(_f_plot / 1e6, _PSD_plot)
_ax.grid()
_ax.set_xlim(-4, 4)
_ax.set_ylim(-30, 15)
_ax.set_xlabel("Frequency [MHz]")
_ax.set_ylabel("PSD [dB]")
#if os.path.isdir('../_images'):
#    fig1.savefig('../_images/fpv_psd_raw_rf.svg', bbox_inches='tight')
fig1

# %%
x_demod = np.angle(x[1:] * np.conj(x[:-1])) # FM demodulation

# %%
# PSD of FM demodulated signal
plt.rcParams['svg.fonttype'] = 'none'
_PSD = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x_demod)))**2 / (len(x_demod)*sample_rate))
_f = np.linspace(sample_rate/-2, sample_rate/2, len(_PSD))
_n = 2**14
_step = len(_PSD) // _n
_PSD_plot = _PSD[:_step*_n].reshape(_n, _step).mean(axis=1)
_f_plot = _f[:_step*_n].reshape(_n, _step).mean(axis=1)
fig2, _ax = plt.subplots()
_ax.plot(_f_plot / 1e6, _PSD_plot)
_ax.grid()
_ax.set_xlim(0, 4.7)
_ax.set_ylim(-110, -30)
_ax.set_xlabel("Frequency [MHz]")
_ax.set_ylabel("PSD [dB]")
#if os.path.isdir('../_images'):
#    fig2.savefig('../_images/fpv_psd_after_fm_demod.svg', bbox_inches='tight')
fig2

# %%
# PSD of FM demodulated signal, zooming into the low freq harmonics
_PSD = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x_demod)))**2 / (len(x_demod)*sample_rate))
_f = np.linspace(sample_rate/-2, sample_rate/2, len(_PSD))
fig3, _ax = plt.subplots()
_ax.plot(_f / 1e6, _PSD)
_ax.grid()
_ax.set_xlim(0, 0.1)
_ax.set_ylim(-70, -20)
_ax.set_xlabel("Frequency [MHz]")
_ax.set_ylabel("PSD [dB]")
#if os.path.isdir('../_images'):
#    fig3.savefig('../_images/fpv_psd_after_fm_demod_harmomics.svg', bbox_inches='tight')
fig3

# %%
# nice shot of the start of a frame and several lines
_offset = 146880
_length = 20000
fig4, _ax = plt.subplots(figsize=(12, 3))
_ax.plot(x_demod[_offset:_offset+_length])
_ax.set_xlabel("Sample")
_ax.set_ylabel("Amplitude")
#if os.path.isdir('../_images'):
#    fig4.savefig('../_images/fpv_time_domain.svg', bbox_inches='tight')
fig4

# %%
# zoomed into 1 line
_offset = 400
_length = samples_per_line + 150
fig5, _ax = plt.subplots(figsize=(12, 3))
_ax.plot(x_demod[_offset:_offset+_length])
_ax.set_xlabel("Sample")
_ax.set_ylabel("Amplitude")
#if os.path.isdir('../_images'):
#    fig5.savefig('../_images/fpv_time_domain_one_line.svg', bbox_inches='tight')
fig5

# %%
# Filter out audio from demodded signal
h = signal.firwin(301, 3e6, fs=sample_rate) # for the 10 Mhz recording
x_video = np.convolve(x_demod, h, 'same')

# Resample luma and chroma to exactly L samples per line
resampling_rate = samples_per_line / (sample_rate / line_Hz)
resampling_rate *= 1.00003 # fixes the drift, not 100% sure where it comes from, perhaps sample clock offset
x_video = signal.resample(x_video, int(len(x_video)*resampling_rate))
print("Resampling rate:", resampling_rate)

# crop to 1 frames worth of samples
if True:
    manually_tuned_offset = 122250 # for both frame sync and horizontal sync
    x_video = x_video[manually_tuned_offset:manually_tuned_offset+samples_per_frame]

# reshape into 2D
x_video = x_video[:len(x_video) - (len(x_video) % samples_per_line)] # trim to multiple of samples_per_line
frame = x_video.reshape(-1, samples_per_line) # type: ignore

# Normalize to 0-255 and convert to uint8
frame_norm = frame - np.min(frame)
frame_norm = frame_norm / np.max(frame_norm)
frame_uint8 = (frame_norm * 255).astype(np.uint8)

# Display as single image with fixed scaling
fig6, _ax = plt.subplots()
_ax.imshow(frame_uint8, cmap='gray', aspect='auto', vmin=0, vmax=255)
_ax.axis('off')
#if os.path.isdir('../_images'):
#    fig6.savefig('../_images/fpv_image_no_sync.svg', bbox_inches='tight')
#    fig6.savefig('../_images/fpv_image_one_frame.svg', bbox_inches='tight')
fig6

# %%
# Kept in its own cell, otherwise marimo would display the figures twice
plt.show()
