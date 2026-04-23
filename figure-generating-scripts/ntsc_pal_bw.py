import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from PIL import Image

# Spikes after FM demod
# 3.579545e6 Color NTSC
# 4.43361875e6 Color PAL and SECAM
# 6.5025e6 audio carrier (I think the audio is transmitted separately but this is where it would show up after FM demodding the whole thing)
# apparently the audio might show up at 5.5, 6.0, or 6.5 MHz

if True:
    format_type = 'ntsc'
    samples_to_process = 500000

    # ATSC recording from 2022 GNU Radio Conference CTF-
    # https://ctf-2022.gnuradio.org/files/5d51c1bb8774333af7e87ecf19f8b664/never_the_same_color.sigmf-meta
    # https://ctf-2022.gnuradio.org/files/bccb3de9c758a0760146aa86e610fa02/never_the_same_color.sigmf-data
    #ntsc_example = '/mnt/c/Users/marclichtman/Downloads/never_the_same_color.sigmf-data' # cf32, 8M sample rate
    
    # ntsc_example = '/mnt/c/Users/marclichtman/Downloads/signal_recordings/RunCamNightEagle3V2_rushfpv.sigmf-data' #ci16, 40M. NO COLOR!
    # offset = 66200 # to manually sync to the horizontal pulse, so lines all start on the left
    # x = np.fromfile(ntsc_example, dtype=np.int16, count=samples_to_process, offset=offset)
    # x = x.astype(np.float32) / 32768.0
    # x = x[::2] + 1j*x[1::2]

    # ntsc_example = '/mnt/c/Users/marclichtman/Downloads/signal_recordings/color.sigmf-data' # 40M sample rate, cf32
    ntsc_example = '/mnt/c/Users/marclichtman/Downloads/ntsc_remy_10MHz_5925Hz_cf32.iq'
    sample_rate = 10e6
    x = np.fromfile(ntsc_example, dtype=np.complex64, count=samples_to_process)
    print(len(x))

    #sample_rate = 40e6
    #fc = 441e6 # taken from metadata file
    color_subcarrier_freq = 3.579545e6 # higher than luma carrier, not relative to center freq
    relative_audio_subcarrier_freq = 3.5e6
else:
    format_type = 'pal'
    samples_to_process = 100000000

    '''
    pal_example = '/mnt/c/Users/marclichtman/Downloads/SDRSharp_20170122_171736Z_179100000Hz_IQ.wav' # used in this SIGIDWIKI entry https://www.sigidwiki.com/wiki/PAL_Broadcast#google_vignette
    x = read(pal_example)
    sample_rate = x[0]
    print("Sample Rate:", sample_rate)
    fc = 179.1e6 # taken from filename
    sample_offset = 200 + 512*55 # in samples. specific to recording
    '''

    #pal_example = '/mnt/c/Users/marclichtman/Downloads/signal_recordings/foxeer_rushfpv.sigmf-data' #ci16, 40M. PAL Color
    # pal_example = '/mnt/c/Users/marclichtman/Downloads/signal_recordings/3390MHz_40MS_10m_cable_4000m_distance+LNA30.sigmf-data'
    # x = np.fromfile(pal_example, dtype=np.int16, count=samples_to_process)
    # x = x.astype(np.float32)
    # x = x[::2] + 1j*x[1::2]
    # x /= np.max(np.abs(x))
    # sample_rate = 40e6
    
    #pal_example = '/tmp/pal_color_hacktv.fc32' # ./hacktv -o file:/mnt/d/pal_color_hacktv.fc32 -t float -m i /mnt/c/Users/marclichtman/Downloads/Free_Test_Data_1.21MB_MKV.mkv -s 16000000 --filter
    #pal_example = '/mnt/d/pal_color_hacktv_colourbars.fc32' # same as above but used test:colourbars instead of mkv file
    #sample_rate = 16e6

    x = np.fromfile(pal_example, dtype=np.complex64, count=samples_to_process)

    sample_offset = 15 + 0*55 # in samples. specific to recording
    
    color_subcarrier_freq = 4.43361875e6 # higher than luma carrier, not relative to center freq
    relative_audio_subcarrier_freq = 3.5e6 # relative to luma carrier, leave positive even if its negative

    # if plotting R-Y and B-Y on complex plane, phase is the hue of the color, and its magnitude is the saturation
    # the tx inserts a snippet of the subcarrier just after the horizontal sync pulse, known as the color burst
    # in PAL, the phase of the R-Y component is inverted on alternate lines, hence "Phase Alternating Line"
    # i.e., the imaginary part (R-Y) will be negative every other line. it also lets the rx know whether its receiving an even or odd line at any given time
    # the phase of the colour burst alternates between 135º and -135º relative to B-Y

if format_type == 'ntsc':
    samples_per_line = 508
    lines_per_frame = 525
    refresh_Hz = 30.0/1.001 # almost exactly 29.97 # not exactly 30 Hz!! makes difference
else: # PAL
    samples_per_line = 512
    lines_per_frame = 625 # (576 visible lines)
    refresh_Hz = 25

samples_per_frame = samples_per_line * lines_per_frame // 2 # samples per frame. WHY DO I NEED THE /2?
print("Samples per frame:", samples_per_frame)
line_Hz = refresh_Hz * lines_per_frame
print("Line rate (Hz):", line_Hz)

# PSD of raw RF
if False:
    PSD = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x)))**2 / (len(x)*sample_rate))
    f = np.linspace(sample_rate/-2, sample_rate/2, len(PSD))
    plt.plot(f / 1e6, PSD)
    plt.xlim(-4, 4)
    plt.ylim(-100, -50)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("PSD [dB]")
    plt.show()

x_demod = np.angle(x[1:] * np.conj(x[:-1])) # FM demodulation

# PSD of FM demodulated signal
if False:
    PSD = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x_demod)))**2 / (len(x_demod)*sample_rate))
    f = np.linspace(sample_rate/-2, sample_rate/2, len(PSD))
    plt.plot(f / 1e6, PSD)
    plt.xlim(0, 7)
    plt.ylim(-110, -35)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("PSD [dB]")
    plt.show()
    exit()

# Spectrogram of FM demodded signal
if False:
    fft_size = 1024
    num_rows = len(x_demod) // fft_size # // is an integer division which rounds down
    spectrogram = np.zeros((num_rows, fft_size))
    for i in range(num_rows):
        spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x_demod[i*fft_size:(i+1)*fft_size])))**2)
    plt.imshow(spectrogram, aspect='auto', extent = [sample_rate/-2/1e6, sample_rate/2/1e6, len(x)/sample_rate, 0])
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Time [s]")
    plt.show()

# Filter out audio from demodded signal
#h = signal.firwin(301, 6e6, fs=sample_rate) # LPF up to 6 MHz, audio is centered at 6.5 MHz but is pretty narrow
h = signal.firwin(301, 3e6, fs=sample_rate) # for the 10 Mhz recording
x_demod = np.convolve(x_demod, h, 'same')

if False: # nice shot of a single line using foxeer_rushfpv
    offset = 2515
    length = 4000
    plt.plot(x_demod[offset:offset+length])
    plt.xlabel("Sample")
    plt.show()
    exit()

# h = signal.firwin(301, 3e6, fs=sample_rate) # LPF
# x_demod = np.convolve(x_demod, h, 'same')

# Resample luma and chroma to exactly L samples per line
resampling_rate = samples_per_line / (sample_rate / line_Hz)
resampling_rate *= 1.00003 # fixes the drift, not 100% sure where it comes from, perhaps sample clock offset
x_demod = signal.resample(x_demod, int(len(x_demod)*resampling_rate))
print("Resampling rate:", resampling_rate)

# Optionally, crop to 1 frames worth of samples
if False:
    manually_tuned_offset = 122250 # for both frame sync and horizontal sync
    x_demod = x_demod[manually_tuned_offset:manually_tuned_offset+samples_per_frame]
    
# Time domain plot 
if False:
    plt.plot(x_demod)
    plt.xlabel("Sample")
    plt.show()

# Save the vertical sync signal as a template to correlate with later
if False:
    v_sync_template = np.asarray(x_demod[117386:121712])
    print(type(v_sync_template[0])) # np.float64
    v_sync_template.tofile("/tmp/vertical_sync_template.iq")
    plt.plot(x_demod[117386:121712])
    plt.xlabel("Sample")
    plt.show()

# Correlate entire signal against the v-sync template, then sync to frame
if False:
    template = np.fromfile("/tmp/vertical_sync_template.iq", dtype=np.float64)
    correlation = np.abs(np.correlate(x_demod, template, mode='full'))**2
    # plt.plot(correlation)
    # plt.xlabel("Sample")
    # plt.ylabel("Correlation")
    # plt.show()

    # Sync to the start of frame we detected
    frame_start = np.argmax(correlation[:150000]) # try to get one of the first ones
    frame_start += 20
    print("Argmax in first 150000 samples at:", frame_start)
    x_demod = x_demod[frame_start:frame_start+samples_per_frame]

# Autocorrelation, to try to file spikes at certain lags (254 for vertical sync, 508 samples per line)
if False:
    n_autocorr = 650
    autocorr = np.array([np.dot(x_demod[:len(x_demod)-lag], x_demod[lag:]) for lag in range(n_autocorr)])
    plt.plot(np.arange(n_autocorr), np.abs(autocorr))
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.show()
    exit()


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
plt.title(f'{format_type.upper()} B&W')
plt.show()
