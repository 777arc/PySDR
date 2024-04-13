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
- http://martin.hinner.info/vga/pal.html
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
    #pal_example2 = '/mnt/d/pal_color_hacktv.fc32' # ./hacktv -o file:/mnt/d/pal_color_hacktv.fc32 -t float -m i /mnt/c/Users/marclichtman/Downloads/Free_Test_Data_1.21MB_MKV.mkv -s 16000000 --filter
    pal_example2 = '/mnt/d/pal_color_hacktv_colourbars.fc32' # same as above but used test:colourbars instead of mkv file
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
if False: # nice shot of a single line
    offset = 100000
    length = 2000
    plt.plot(np.abs(x[offset:offset+length]))
    plt.show()
    exit()

# Notch out audio signal
#x = np.convolve(x, signal.firwin(301, [relative_audio_subcarrier_freq - 50e3, relative_audio_subcarrier_freq + 50e3], fs=sample_rate), 'same')

if False:
    PSD = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x)))**2)
    f = np.linspace(-sample_rate/2, sample_rate/2, len(PSD))
    plt.plot(f, PSD)
    plt.show()
    exit()

# Find start of frame TODO: currently assumes recording starts during the transition period
gap_between_lines = 1024 # samples
threshold = 0.65 # can be same as other threshold
burst_indxs = np.where(np.diff((np.abs(x) > threshold).astype(int)) == 1)[0] # indx of rising edges
start_of_frame = burst_indxs[np.where(np.diff(burst_indxs) == gap_between_lines)[0][0] + 1]
print("Start of frame:", start_of_frame)
x = x[start_of_frame:] # cut off end of prev frame
if False:
    print(np.diff(burst_indxs)[0:30])
    plt.plot(np.abs(x[0:10000]))
    for i in burst_indxs[0:15]:
        plt.plot([i, i], [0, 1], 'r:')
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

# Using a manually tuned threshold, find the start of each burst, using the combined luma+chroma (last time it needs to be used)
threshold = 0.65
burst_indxs = np.where(np.diff((np.abs(x) > threshold).astype(int)) == -1)[0] # need to use abs() of original signal that includes luma and chroma for detection of each pixel start
if False: # look at a single line and the threshold
    offset = 100000
    length = 2000
    plt.plot(np.abs(x[offset:offset+length]))
    plt.plot([0, length], [threshold, threshold])
    plt.show()
    exit()
print("Bursts found:", len(burst_indxs))


# Extract luma component - filter and take magnitude
x_luma = np.convolve(x, signal.firwin(301, 3e6, fs=sample_rate), 'same')
x_luma = np.abs(x_luma) # Take magnitude
if False:
    offset = 1000000
    plt.plot(x_luma[offset:offset+3000])
    plt.show()
    exit()


# Resample luma and chroma to exactly L samples per line
resampling_rate = L/(sample_rate/line_Hz)
x_luma = signal.resample(x_luma, int(len(x_luma)*resampling_rate))
x_chroma = signal.resample(x_chroma, int(len(x_chroma)*resampling_rate))
print(sample_rate, L*line_Hz)
print("Resampling rate:", resampling_rate)
sample_rate = L*line_Hz # update sample rate
# Update burst indexes to match new sample rate
resampled_burst_indxs = []
for i in burst_indxs:
    resampled_burst_indxs.append(int(i * resampling_rate))
# At this point, the diff between resampled_burst_indxs should be exactly 512 (L) if all is well, would be a good time to meausre accuracy of sync
burst_indxs = resampled_burst_indxs


# Extract just the chroma bursts, and store freq offsets of each burst
delay_till_burst = 6 # samples between thresh and start of burst FIXME CONVERT TO SECONDS AND CALC BASED ON SAMPLE RATE
burst_len = 22 # samples FIXME CONVERT TO SECONDS AND CALC BASED ON SAMPLE RATE
x_chroma_burst = np.zeros_like(x)
chroma_freq_offsets = [] # corresponds to burst_indxs
for i in burst_indxs:
    burst_slice = x_chroma[i+delay_till_burst:i+delay_till_burst+burst_len]
    x_chroma_burst[i+delay_till_burst:i+delay_till_burst+burst_len] = burst_slice
    PSD = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(burst_slice, 1024)))**2) # FFT of chroma burst with padding
    f = np.linspace(-sample_rate/2, sample_rate/2, 1024)
    burst_max_freq = f[np.argmax(PSD)]
    chroma_freq_offsets.append(burst_max_freq) 
    if False:
        print("Burst max freq:", burst_max_freq, "Hz")
        plt.plot(f, PSD)
        plt.show()
        exit()
if False: # look at chroma burst
    offset = 1000000
    plt.plot(np.abs(x_chroma_burst[offset:offset+10000]), '.-')
    plt.show()
    exit()

# Find color burst phase shift
burst_phases = [] # goes along with burst_indxs
new_burst_indxs = []
new_chroma_freq_offsets = []
for i in range(len(burst_indxs)):
    burst = x_chroma_burst[burst_indxs[i]+delay_till_burst+13:burst_indxs[i]+delay_till_burst+burst_len-6] # hand-tuned to only include meat of burst
    if np.max(np.abs(burst)) > 0.02:
        if False:
            plt.plot(burst.real, burst.imag, '.')
            plt.axis([-0.1, 0.1, -0.1, 0.1])
            plt.show()
        if np.var(burst) > 1e-4:
            print("Cluster wasnt tight") # FIXME i cant just not include these or it will throw off frame timing
        else:
            burst_phase = np.mean(np.angle(burst)) # grab average phase of all the good samples
            burst_phases.append(burst_phase)
            new_burst_indxs.append(burst_indxs[i])
            new_chroma_freq_offsets.append(chroma_freq_offsets[i])
    else:
        print("Low amplitude burst")
burst_indxs = new_burst_indxs
chroma_freq_offsets = new_chroma_freq_offsets
# Make sure the starts of bursts make sense sequentially
if False:
    plt.plot(burst_indxs, np.ones(len(burst_indxs)), '.')
    plt.show()
    exit()

# Figure out if we're starting on an even or odd line, The phase of the color burst alternates between 135º (2.35619 rad) and -135º (225 deg or 3.927 rad) relative to B-Y
while burst_phases[0] >= 2*np.pi:
    burst_phases[0] -= 2*np.pi
while burst_phases[0] < 0:
    burst_phases[0] += 2*np.pi
while burst_phases[1] >= 2*np.pi:
    burst_phases[1] -= 2*np.pi
while burst_phases[1] < 0:
    burst_phases[1] += 2*np.pi
#print(burst_phases[0], burst_phases[1], "radians")
#print(burst_phases[0]*180/np.pi, burst_phases[1]*180/np.pi, "degrees")
#print("diff:", burst_phases[0] - burst_phases[1]) # should be either positive or negative roughly pi/2 (1.57)
if (burst_phases[0] - burst_phases[1]) > 0:
    if (burst_phases[0] - burst_phases[1]) < np.pi: # this means the two are on different sides of the 0 deg axis and we need to reverse
        print("Starting on even line")
    else:
        print("Starting on odd line~")
        burst_indxs = burst_indxs[1:] # remove first one so we always start on even
        burst_phases = burst_phases[1:]
else:
    if (burst_phases[1] - burst_phases[0]) < np.pi:
        print("Starting on odd line")
        burst_indxs = burst_indxs[1:] # remove first one so we always start on even
        burst_phases = burst_phases[1:]
    else:
        print("Starting on even line~")

# Calc phase correction that needs to be applied to chroma.  we can assume we start on an even line
correction_phases = [] # in radians
for i in range(len(burst_phases)):
    current_phase = burst_phases[i] # should be 225 deg (3.927 rad) for even, +135 deg (2.35619 rad) for odd
    corrected_phase = current_phase 
    if i % 2 == 0:
        corrected_phase = current_phase - 225/180*np.pi
    else:
        corrected_phase = current_phase - 135/180*np.pi
    # Get between 0 and 360
    while corrected_phase >= 2*np.pi:
        corrected_phase -= 2*np.pi
    while corrected_phase < 0:
        corrected_phase += 2*np.pi
    correction_phases.append(corrected_phase)
if False:
    plt.plot(correction_phases)
    plt.show()
    exit()

# each burst is a line
line_i = 0
start_burst_offset = 32 
end_burst_offset = 6
frame = np.zeros((lines_per_frame//2, L, 3)) # only showing even lines
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
# (optional) for better image and multipath mitigation, the chrominance for the current line is averaged with a copy of the chrominance from the previous line with R-Y inverted again. This cancels out the phase error, at the expense of a slight change in saturation, which is much less noticeable
for ii in range(len(burst_indxs)):
    reference_level = x_luma[burst_indxs[ii] + 20] # manually tweaked to be in the middle of the luma reference burst
    y = x_luma[burst_indxs[ii]+start_burst_offset:burst_indxs[ii]+L+end_burst_offset]
    y /= reference_level
    x_chroma_slice = x_chroma[burst_indxs[ii]+start_burst_offset:burst_indxs[ii]+L+end_burst_offset] # manually tweaked till chroma burst was gone
    # Freq shift using the offsets we found earlier
    x_chroma_slice *= np.exp(-2j*np.pi*chroma_freq_offsets[ii]*np.arange(len(x_chroma_slice))/sample_rate)
    # Correct phase
    x_chroma_slice *= np.exp(1j*correction_phases[ii])
    I = x_chroma_slice.real
    Q = x_chroma_slice.imag

    # IQ plot of one line
    if line_i == 20:
        ax1.plot(I, Q, '.')
        ax1.axhline(y=0, color='k')
        ax1.axvline(x=0, color='k')
        ax1.axis([-0.1, 0.1, -0.1, 0.1])

    if ii % 2 == 0: # every other line, r-y is negative
        Q *= -1

    # hand-tweaked for now
    I *= 4.5 # till the max in the frame is about 1.0 for the colourtest video (needed to bump blue to 1.1 for some reason to make it look good)
    Q *= 5.5

    b = y + 2.029 * I
    r = y + 1.14 * Q
    g = y - 0.396 * I - 0.581 * Q

    # Figure out why this is needed
    r = 1 - r
    b = 1 - b
    g = 1 - g
    
    # Code only works for even lines (at least for PAL at the moment)
    if line_i < lines_per_frame//2: # even lines
        frame[line_i, 0:len(y), 0] = r
        frame[line_i, 0:len(y), 1] = g
        frame[line_i, 0:len(y), 2] = b
    else: # odd lines
        pass

    line_i += 1
    ii += 1
    if line_i == lines_per_frame - 18: # this one was manually adjusted until there was no shifting between frames
        print("max red:", np.max(frame[:, :, 0]))
        print("max green:", np.max(frame[:, :, 1]))
        print("max blue:", np.max(frame[:, :, 2]))
        ax2.imshow(frame, aspect=0.6)
        plt.show()
        plt.draw()
        plt.pause(0.5)
        ax1.cla()
        ax2.cla()
        line_i = 0
