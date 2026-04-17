import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

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
    ntsc_example = '/tmp/pluto_samples.iq'
    sample_rate = 10e6
    x = np.fromfile(ntsc_example, dtype=np.complex64, count=samples_to_process)

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
    L = 508 # samples per line
    lines_per_frame = 525
    refresh_Hz = 29.97 # not exactly 30 Hz!! makes difference.  it's actually 30/1.001
else: # PAL
    L = 512 # samples per line
    lines_per_frame = 625 # (576 visible lines)
    refresh_Hz = 25

samples_per_frame = L * lines_per_frame # samples per frame
print("Samples per frame:", samples_per_frame)
line_Hz = refresh_Hz * lines_per_frame

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
h = signal.firwin(301, 4e6, fs=sample_rate) # for the 10 Mhz recording
x_demod = np.convolve(x_demod, h, 'same')

if False: # nice shot of a single line using foxeer_rushfpv
    offset = 2515
    length = 4000
    plt.plot(x_demod[offset:offset+length])
    plt.xlabel("Sample")
    plt.show()
    exit()

h = signal.firwin(301, 3e6, fs=sample_rate) # LPF
x_luma = np.convolve(x_demod, h, 'same')

# Resample luma and chroma to exactly L samples per line
resampling_rate = L / (sample_rate / line_Hz)
x_luma = signal.resample(x_luma, int(len(x_luma)*resampling_rate))
print("Resampling rate:", resampling_rate)

# Time domain plot 
if False:
    plt.plot(x_luma)
    plt.xlabel("Sample")
    plt.show()

# reshape into lines
num_lines = len(x_luma) // L
x_luma = x_luma[:num_lines * L]
frame = x_luma.reshape(num_lines, L) # type: ignore

# Display as single image
plt.imshow(frame, cmap='gray', aspect='auto')
plt.axis('off')
plt.title(f'{format_type.upper()} B&W')
plt.show()

