'''
- I'm not sure about PAL, but for NTSC it would be more complicated than black & white but not insurmountable.
- There's some filtering needed to get good separation of the colour information, and you have to lock on to the phase of the colour burst.
- It looks like PAL works the same way. There's a 10-cycle colour burst that you lock onto and use that to regenerate the carrier needed to demodulate the colour signal.
- FPV ones defniitely have color
- NTSC is US, PAL is EU, SECAM is france and russia
- Clayton doesnt think it would be too hard, specs are short https://antiqueradio.org/art/NTSC%20Signal%20Specifications.pdf
- clayton wrote straight python for generating color NTSC https://github.com/argilo/grcon22/tree/main/ntsc
- Dani's GRC and ipynb solution for claytons challenge - https://destevez.net/2022/10/grcon22-capture-the-flag/ and https://github.com/daniestevez/grcon22-ctf/blob/main/Never_the_same_color/NTSC.ipynb
- sigmf recording should be here https://ctf-2022.gnuradio.org/
- writeup of how the color system works - https://www.csse.canterbury.ac.nz/greg.ewing/c64_pal_decoder/PAL/PAL-Video-Encoding.html
- another open source code for color tx only is hacktv, uses hackrf
- post of people asking for color to be added to SDRAngel https://github.com/f4exb/sdrangel/issues/21
- Dani says Gonzalo, the author of SigDigger, has done some experiments with color PAL, but not NTSC, although the two are similar
- NTSC is basically the same idea as PAL but without the alternating phase (and with different timing ofc)
- note from a random github issue post - I used the term "PAL" to qualify the frame synchronization scheme which is the one used in PAL (and NTSC and SECAM by the way) and that's all that is PAL there. To implement color is a lot of work because you have to mimic the rather complex analog circuitry that best works in... analog. This implies locking the color oscillator (a NCO then) in frequency and phase with the small burst at the start of the line. Then you need to filter the luminance (LPF) and the chrominance (BPF) apart. You then take the chrominance signal and decode its phase using the reference of the color oscillator. Then you reassemble everything to get the RGB according to the particular standard (PAL, NTSC, SECAM). I beleieve this is a lot of work for just analog color. Now anyone who is up at it is welcome but I will certainly concentrate on more valuable features. In particular it would be more immediately interesting to add support for digital TV.
- it looks like only the audio is FM modulated, not the video!  the video is essentially just AM
- old TV used PAL/NTSC modulated with VSB (hence asymetric PSD), but FPV cameras in 5.8 GHz band use PAL/NTSC FM modulated
- In vestigial sideband (VSB), the full upper sideband of bandwidth (4.0 MHz) is tx, but only 0.75 MHz of the lower sideband is tx, along with a carrier
- They use VSB and not SSB because they have a DC component and SSB would filter that out
- remember that hacktv can also produce fm modulated pal, as well as the different variants of pal
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.io.wavfile import read
from PIL import Image

'''
flowgraph is
1. Freq Xling FIR Filter using shift of -1.75e6, and FIR filter with cutoff of 3e6, width of 200e3, no decimation
2. Complex to magnitude
3. Polyphase resampler of rate 508/(samp_rate/15734.25)

Python code adapted from Dani's GRCon '22 CTF solution https://destevez.net/2022/10/grcon22-capture-the-flag/
A big thanks to Dani, Gonzalo, and Clayton for their help and resources
'''

if False:
    format_type = 'ntsc'
    # ATSC recording from 2022 GNU Radio Conference CTF-
    # https://ctf-2022.gnuradio.org/files/5d51c1bb8774333af7e87ecf19f8b664/never_the_same_color.sigmf-meta
    # https://ctf-2022.gnuradio.org/files/bccb3de9c758a0760146aa86e610fa02/never_the_same_color.sigmf-data
    ntsc_example = '/mnt/d/never_the_same_color.sigmf-data' # cf32, 8M sample rate
    samples_to_process = 10000000
    x = np.fromfile(ntsc_example, dtype=np.complex64, count=samples_to_process)
    sample_rate = 8e6
    fc = 441e6 # taken from metadata file
    # For now only process first 1M samples
    print(len(x))
    x = x[0:samples_to_process]
    sample_offset = 70 # in samples. specific to recording
    color_subcarrier_freq = 3.579545e6 # higher than luma carrier, not relative to center freq
    relative_audio_subcarrier_freq = 3.5e6
else:
    samples_to_process = 10000000
    format_type = 'pal'
    #pal_example = '/mnt/d/SDRSharp_20170122_171736Z_179100000Hz_IQ.wav' # used in this SIGIDWIKI entry https://www.sigidwiki.com/wiki/PAL_Broadcast#google_vignette
    #x = read(pal_example)
    #sample_rate = x[0]
    #print("Sample Rate:", sample_rate)
    #fc = 179.1e6 # taken from filename
    #sample_offset = 200 + 512*55 # in samples. specific to recording
    pal_example2 = '/mnt/d/pal_color_hacktv.fc32' # ./hacktv -o file:/mnt/d/pal_color_hacktv.fc32 -t float -m i /mnt/c/Users/marclichtman/Downloads/Free_Test_Data_1.21MB_MKV.mkv -s 16000000 --filter
    sample_rate = 16e6
    x = np.fromfile(pal_example2, dtype=np.complex64, count=samples_to_process)
    sample_offset = 15 + 0*55 # in samples. specific to recording
    color_subcarrier_freq = 4.43361875e6 # higher than luma carrier, not relative to center freq
    relative_audio_subcarrier_freq = 3.5e6 # relative to luma carrier, leave positive even if its negative

    # if plotting R-Y and B-Y on complex plane, phase is the hue of the color, and its magnitude is the saturation
    # the tx inserts a snippet of the subcarrier just after the horizontal sync pulse, known as the color burst
    # in PAL, the phase of the R-Y component is inverted on alternate lines, hence "Phase Alternating Line"
    # i.e., the imaginary part (R-Y) will be negative every other line. it also lets the rx know whether its receiving an even or odd line at any given time
    # the phase of the colour burst alternates between 135º and -135º relative to B-Y

# Find luma subcarrier by looking for peak in PSD but only below 0 Hz, as there's also an audio and chroma subcarrier
PSD = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x)))**2)
f = np.linspace(-sample_rate/2, sample_rate/2, len(PSD))
luma_subcarrier_freq = f[np.argmax(PSD[0:len(PSD)//2])]
print("Max peak at:", luma_subcarrier_freq/1e6, "MHz")
if False:
    plt.plot(f, PSD)
    plt.show()
    exit()

# Center on luma subcarrier
x = x * np.exp(-2j*np.pi*luma_subcarrier_freq*np.arange(len(x))/sample_rate)

# Notch out audio signal
#x = np.convolve(x, signal.firwin(301, [relative_audio_subcarrier_freq - 50e3, relative_audio_subcarrier_freq + 50e3], fs=sample_rate), 'same')

if False:
    PSD = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x)))**2)
    f = np.linspace(-sample_rate/2, sample_rate/2, len(PSD))
    plt.plot(f, PSD)
    plt.show()
    exit()


# Extract chroma component
x_chroma = x * np.exp(-2j*np.pi*color_subcarrier_freq*np.arange(len(x))/sample_rate)
x_chroma = np.convolve(x_chroma, signal.firwin(301, 1e6, fs=sample_rate), 'same')
if False:
    offset = 1000000
    x_chroma = np.abs(x_chroma)
    plt.plot(x_chroma[offset:offset+3000])
    plt.show()
    exit()
if False:
    PSD = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x_chroma)))**2)
    f = np.linspace(-sample_rate/2, sample_rate/2, len(PSD))
    plt.plot(f, PSD)
    plt.show()
    exit()
if False:
    fft_size = 512
    num_rows = len(x_chroma) // fft_size
    spectrogram = np.zeros((num_rows, fft_size))
    for i in range(num_rows):
        spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x_chroma[i*fft_size:(i+1)*fft_size])))**2)
    plt.imshow(spectrogram, aspect='auto', extent = [sample_rate/-2/1e6, sample_rate/2/1e6, len(x_chroma)/sample_rate, 0])
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Time [s]")
    plt.show()
    exit()

# Using a manually tuned threshold, zero out all samples except the chroma burst
threshold = 0.65
delay_till_burst = 12 # samples between thresh and start of burst
burst_len = 45 # samples FIXME CONVERT TO SECONDS AND CALC BASED ON SAMPLE RATE
burst_indxs = np.where(np.diff((x > 0.65).astype(int)) == -1)[0] # need to use original signal for detection of each pixel start
x_chroma_burst = np.zeros_like(x)
for i in burst_indxs:
    x_chroma_burst[i+delay_till_burst:i+delay_till_burst+burst_len] = x_chroma[i+delay_till_burst:i+delay_till_burst+burst_len]
if False:
    offset = 1000000
    plt.plot(np.abs(x_chroma_burst[offset:offset+10000]), '.-')
    plt.show()
    exit()

# Fine freq sync W/ PSD FIXME will need to run on small set of pixels for actual transmissions
PSD = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x_chroma_burst)))**2)
f = np.linspace(-sample_rate/2, sample_rate/2, len(PSD))
if False:
    plt.plot(f, PSD)
    plt.show()
    exit()
max_freq = f[np.argmax(PSD[np.abs(f - (-30)).argmin():np.abs(f - 30).argmin()]) + np.abs(f - (-30)).argmin()]
print("max_freq", max_freq, "Hz")
x_chroma_burst = x_chroma_burst * np.exp(-2j*np.pi*max_freq*np.arange(len(x))/sample_rate)
if False:
    plt.plot(x_chroma_burst[0:100000].real, x_chroma_burst[0:100000].imag, '.')
    plt.show()
    exit()

burst_phases = [] # goes along with burst_indxs
for i in burst_indxs:
    burst = x_chroma_burst[i+delay_till_burst+13:i+delay_till_burst+burst_len-6] # hand-tuned to only include meat of burst
    if np.max(np.abs(burst)) > 0.02:
        if False:
            plt.plot(burst.real, burst.imag, '.')
            plt.axis([-0.1, 0.1, -0.1, 0.1])
            plt.show()
        if np.var(burst) > 1e-4:
            print("Cluster wasnt tight")
        else:
            burst_phase = np.mean(np.angle(burst)) # grab average phase of all the good samples
            burst_phases.append(burst_phase)
    else:
        print("Low amplitude burst")

# The phase of the color burst alternates between 135º and -135º (225 deg) relative to B-Y
''' i dont think this actually works, the higher one might wrap around 360deg
# Figure out if we're starting on an even or odd line
while burst_phases[0] >= 2*np.pi:
    burst_phases[0] -= 2*np.pi
while burst_phases[0] < 0:
    burst_phases[0] += 2*np.pi
while burst_phases[1] >= 2*np.pi:
    burst_phases[1] -= 2*np.pi
while burst_phases[1] < 0:
    burst_phases[1] += 2*np.pi
print(burst_phases[0] - burst_phases[1]) # should be either positive or negative roughly pi/2 (1.57)
if (burst_phases[0] - burst_phases[1]) > 0:
    print("Starting on even line")
else:
    print("Starting on odd line")
'''

x = np.abs(x) # Take magnitude
if False:
    offset = 1000000
    plt.plot(x[offset:offset+3000])
    plt.show()
    exit()

if format_type == 'ntsc':
    L = 508 # samples per line
    lines_per_frame = 525
    refresh_Hz = 29.97 # not exactly 30 Hz!! makes difference.  it's actually 30/1.001
else: # PAL
    L = 512 # samples per line
    lines_per_frame = 625 # (576 visible lines)
    refresh_Hz = 25

N = L * lines_per_frame # samples per frame
line_Hz = refresh_Hz * lines_per_frame

# Resample to exactly L samples per line
x = signal.resample(x, int(len(x)*L/(sample_rate/line_Hz)))
print("Resampling rate:", L/(sample_rate/line_Hz))

# Manually perform frame sync, for now
x = x[sample_offset:]

num_frames = int(len(x) / N)
plt.ion()
plt.figure(figsize=(15, 9))
for i in range(num_frames):
    y = x[i*N:][:N]
    z = 1-y[:y.size//L*L].reshape(-1, L)
    # deinterlace
    w = np.empty_like(z)
    a = w[::2].shape[0]
    w[::2] = z[:a]
    w[1::2] = z[a:]
    
    if True:
        plt.imshow(w, aspect=0.6, cmap='gray')
        plt.show()
        ## Trick for updating in realtime
        plt.draw()
        plt.pause(0.001)
        plt.clf()

    #im = Image.fromarray(np.round(255*w).astype('uint8'))
    #im.save(f'/tmp/{i:04d}.png')
