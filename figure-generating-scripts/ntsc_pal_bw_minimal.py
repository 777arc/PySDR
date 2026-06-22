import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

filename = 'ntsc_remy_10MHz_5925Hz_500ksamples_cf32.iq'
x = np.fromfile(filename, dtype=np.complex64)
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

# PSD of raw RF
if False:
    plt.rcParams['svg.fonttype'] = 'none'
    PSD = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x)))**2 / (len(x)*sample_rate))
    f = np.linspace(sample_rate/-2, sample_rate/2, len(PSD))
    n = 4096
    step = len(PSD) // n
    PSD_plot = PSD[:step*n].reshape(n, step).mean(axis=1)
    f_plot = f[:step*n].reshape(n, step).mean(axis=1)
    plt.plot(f_plot / 1e6, PSD_plot)
    plt.grid()
    plt.xlim(-4, 4)
    plt.ylim(-30, 15)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("PSD [dB]")
    plt.savefig('../_images/fpv_psd_raw_rf.svg', bbox_inches='tight')
    exit()

x_demod = np.angle(x[1:] * np.conj(x[:-1])) # FM demodulation

# PSD of FM demodulated signal
if False:
    plt.rcParams['svg.fonttype'] = 'none'
    PSD = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x_demod)))**2 / (len(x_demod)*sample_rate))
    f = np.linspace(sample_rate/-2, sample_rate/2, len(PSD))
    n = 2**14
    step = len(PSD) // n
    PSD_plot = PSD[:step*n].reshape(n, step).mean(axis=1)
    f_plot = f[:step*n].reshape(n, step).mean(axis=1)
    plt.plot(f_plot / 1e6, PSD_plot)
    plt.grid()
    plt.xlim(0, 4.7)
    plt.ylim(-110, -30)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("PSD [dB]")
    plt.savefig('../_images/fpv_psd_after_fm_demod.svg', bbox_inches='tight')
    plt.show()
    exit()

# PSD of FM demodulated signal, zooming into the low freq harmonics
if False:
    PSD = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x_demod)))**2 / (len(x_demod)*sample_rate))
    f = np.linspace(sample_rate/-2, sample_rate/2, len(PSD))
    plt.plot(f / 1e6, PSD)
    plt.grid()
    plt.xlim(0, 0.1)
    plt.ylim(-70, -20)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("PSD [dB]")
    plt.savefig('../_images/fpv_psd_after_fm_demod_harmomics.svg', bbox_inches='tight')
    plt.show()


if False: # nice shot of the start of a frame and several lines
    offset = 146880
    length = 20000
    plt.figure(figsize=(12, 3))
    plt.plot(x_demod[offset:offset+length])
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.savefig('../_images/fpv_time_domain.svg', bbox_inches='tight')
    plt.show()

if False: # zoomed into 1 line
    offset = 400
    length = samples_per_line + 150
    plt.figure(figsize=(12, 3))
    plt.plot(x_demod[offset:offset+length])
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.savefig('../_images/fpv_time_domain_one_line.svg', bbox_inches='tight')
    plt.show()

# Filter out audio from demodded signal
h = signal.firwin(301, 3e6, fs=sample_rate) # for the 10 Mhz recording
x_demod = np.convolve(x_demod, h, 'same')

# Resample luma and chroma to exactly L samples per line
resampling_rate = samples_per_line / (sample_rate / line_Hz)
resampling_rate *= 1.00003 # fixes the drift, not 100% sure where it comes from, perhaps sample clock offset
x_demod = signal.resample(x_demod, int(len(x_demod)*resampling_rate))
print("Resampling rate:", resampling_rate)

# crop to 1 frames worth of samples
if True:
    manually_tuned_offset = 122250 # for both frame sync and horizontal sync
    x_demod = x_demod[manually_tuned_offset:manually_tuned_offset+samples_per_frame]

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
#plt.savefig('../_images/fpv_image_no_sync.svg', bbox_inches='tight')
#plt.savefig('../_images/fpv_image_one_frame.svg', bbox_inches='tight')
plt.show()
