import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# this part came from pulse shaping exersize
num_symbols = 100
sps = 8
fs = 1e6  # assume our sample rate is 1 MHz
bits = np.random.randint(0, 2, num_symbols)  # Our data to be transmitted, 1's and 0's
pulse_train = np.array([])
for bit in bits:
    pulse = np.zeros(sps)
    pulse[0] = bit * 2 - 1  # set the first value to either a 1 or -1
    pulse_train = np.concatenate((pulse_train, pulse))  # add the 8 samples to the signal
# optionally, turn it into qpsk
if False:
    bits = np.random.randint(0, 2, num_symbols)  # Our data to be transmitted, 1's and 0's
    pulse_train_q = np.array([])
    for bit in bits:
        pulse = np.zeros(sps)
        pulse[0] = bit * 2 - 1  # set the first value to either a 1 or -1
        pulse_train_q = np.concatenate((pulse_train_q, pulse))  # add the 8 samples to the signal
    pulse_train = pulse_train + 1j * pulse_train_q
fig, ax = plt.subplots(1, figsize=(8, 2))  # 7 is nearly full width
plt.plot(pulse_train, '.-')
plt.grid(True)
#fig.savefig('/tmp/time-sync-original-data.svg', bbox_inches='tight')

original_data = pulse_train  # save for plotting later

# Create our raised-cosine filter
num_taps = 101
beta = 0.35
Ts = sps  # Assume sample rate is 1 Hz, so sample period is 1, so *symbol* period is 8
t = np.arange(-51, 52)  # remember it's not inclusive of final number
h = np.sinc(t / Ts) * np.cos(np.pi * beta * t / Ts) / (1 - (2 * beta * t / Ts) ** 2)
# Plot filter taps
# plt.figure(1)
# plt.plot(t, h, '.')
# plt.grid(True)


# Filter our signal, in order to apply the pulse shaping
samples = np.convolve(pulse_train, h)
fig, ax = plt.subplots(1, figsize=(7, 3))  # 7 is nearly full width
symbols_to_plot = 10
plt.plot(samples[0:symbols_to_plot * sps + (num_taps - 1) // 2], '.-')
for i in range(symbols_to_plot):
    plt.plot([i * sps + num_taps // 2 + 1, i * sps + num_taps // 2 + 1], [min(samples), max(samples)], 'g')
plt.grid(True)
#fig.savefig('/tmp/time-sync-pulse-shaped.svg', bbox_inches='tight')

if True:
    # Create and apply fractional delay filter
    delay = 0.4  # fractional delay, in samples
    N = 21  # number of taps
    n = np.arange(N)  # 0,1,2,3...
    h = np.sinc(n - (N - 1) / 2 - delay)  # calc filter taps
    h *= np.hamming(N)  # window the filter to make sure it decays to 0 on both sides
    h /= np.sum(h)  # normalize to get unity gain, we don't want to change the amplitude/power

    fig, ax = plt.subplots(1, figsize=(7, 3))  # 7 is nearly full width
    ax.plot(samples[100 + 0:100 + 30 - (N - 1) // 2], '.-')

    samples = np.convolve(samples, h)  # apply filter

    ax.plot(samples[100 + (N - 1) // 2:100 + 30], '.-')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Real part of signal')
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # x axis tick labels were showing up as fractions for some reason
    plt.grid(True)
    #fig.savefig('/tmp/fractional-delay-filter.svg', bbox_inches='tight')

if True:
    # apply a freq offset
    fo = 100  # simulate freq offset
    Ts = 1 / fs  # calc sample period
    t = np.arange(0, Ts * len(samples), Ts)  # create time vector

    fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 4))  # 7 is nearly full width
    fig.tight_layout(pad=2.0)  # add space between subplots
    ax1.plot(np.real(samples), '.-')
    ax1.plot(np.imag(samples), '.-')
    ax1.set_title('Before Freq Offset')
    ax1.legend(['I', 'Q'], loc=1)

    samples = samples * np.exp(1j * 2 * np.pi * fo * t)  # perform freq shift

    ax2.plot(np.real(samples), '.-')
    ax2.plot(np.imag(samples), '.-')
    ax2.set_title('After Freq Offset')
    ax2.legend(['I', 'Q'], loc=1)
    #fig.savefig('/tmp/sync-freq-offset.svg', bbox_inches='tight')

if False:
    # COARSE FREQ SYNC PART

    # plot before
    fig, axes_1 = plt.subplots(1, figsize=(7, 2))
    psd = np.fft.fftshift(np.abs(np.fft.fft(samples)))
    f = np.linspace(-fs / 2.0, fs / 2.0, len(psd))
    axes_1.plot(f, psd)
    #fig.savefig('/tmp/coarse-freq-sync-before.svg', bbox_inches='tight')

    samples_sq = samples ** 2
    psd = np.fft.fftshift(np.abs(np.fft.fft(samples_sq)))
    f = np.linspace(-fs / 2.0, fs / 2.0, len(psd))
    max_freq = f[np.argmax(psd)]
    fig, (axes_1, axes_2) = plt.subplots(2, figsize=(7, 4))
    # top
    axes_1.plot(f, psd)
    from matplotlib.patches import Rectangle

    axes_1.add_patch(Rectangle((-5000, 0), 55000, np.max(psd) * 1.03, ls='--', lw=1.0, ec='red', alpha=1, facecolor='none'))
    axes_1.annotate('Zoomed in below', (60000, np.max(psd) * 0.8), color='r')
    # bottom
    axes_2.plot(f, psd, '.-')
    axes_2.axis([-5000, 50000, 0, 900])  # zoom in so we can visually see max
    axes_2.annotate(str(round(max_freq, 2)) + ' Hz', (max_freq, 1.05 * np.max(psd)))
    #fig.savefig('/tmp/coarse-freq-sync.svg', bbox_inches='tight')

    # Shift by negative of estimated frequency
    samples = samples * np.exp(-1j * 2 * np.pi * max_freq * t / 2.0)  # remember we have to divide max_freq by 2.0 because we had squared
    # Now all that's left is a small amount of freq shift, which costas loop will fix
    plt.show()
    exit()


##### TIME SYNC PART
def rail(x):
    if x >= 0:
        return 1.0
    else:
        return -1.0


# Interpolate samples
samples_interpolated = signal.resample_poly(samples, 16, 1)
fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 4))  # 7 is nearly full width
fig.tight_layout(pad=2.0)  # add space between subplots
ax1.plot(samples[100:105], '.-')
ax1.set_title('Before Interpolation')
ax2.plot(samples_interpolated[100 * 16:105 * 16], '.-')
ax2.set_title('After Interpolation')
#fig.savefig('/tmp/time-sync-interpolated-samples.svg', bbox_inches='tight')

''' REAL VERSION
mu = 0 # initial estimate of phase of sample
i_in = 0 # input index
i_out = 0 # output index
last_sample = 0 # initial condition
out = np.zeros(num_symbols + 30, dtype=np.complex) # output will be 1 sample per symbol, we add extra room at the end for good measure
while i_out < len(samples) and i_in < len(samples):
    out[i_out] = samples_interpolated[i_in*16 + int(mu*16)] # grab what we think is the "best" sample
    last_sample = out[i_out]
    mm_val = rail(last_sample) * out[i_out] - rail(out[i_out]) * last_sample
    mu += sps + 0.3*mm_val # the multiplier can be tweaked to change how fast it reacts, higher value will make it work faster
    i_in += int(np.floor(mu)) # round down to nearest int since we are using it as an index
    mu = mu - np.floor(mu) # remove the integer part of mu
    i_out += 1 # increment output index
    print(i_out)
'''

# COMPLEX VERSION
''' (made into one-liner)
def rail_complex(x): # takes in a complex sample
    I = 0.0
    Q = 0.0
    if(np.real(x) > 0):
        I = 1.0
    if(np.imag(x) > 0):
        Q = 1.0
    return I + 1j*Q
'''

mu = 0  # initial estimate of phase of sample
out = np.zeros(len(samples) + 10, dtype=np.complex64)
out_rail = np.zeros(len(samples) + 10, dtype=np.complex64)  # stores values, each iteration we need the previous 2 values plus current value
i_in = 0  # input samples index
i_out = 2  # output index (let first two outputs be 0)
while i_out < len(samples) and i_in < len(samples):
    if True:  # interpolated case
        out[i_out] = samples_interpolated[i_in * 16 + int(mu * 16)]  # grab what we think is the "best" sample
    else:
        out[i_out] = samples[i_in + int(mu)]  # grab what we think is the "best" sample
    out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j * int(np.imag(out[i_out]) > 0)
    x = (out_rail[i_out] - out_rail[i_out - 2]) * np.conj(out[i_out - 1])
    y = (out[i_out] - out[i_out - 2]) * np.conj(out_rail[i_out - 1])
    mm_val = np.real(y - x)
    # mm_val = min(mm_val, 4.0) # For the sake of there being less code to explain
    # mm_val = max(mm_val, -4.0)
    mu += sps + 0.3 * mm_val
    i_in += int(np.floor(mu))  # round down to nearest int since we are using it as an index
    mu = mu - np.floor(mu)  # remove the integer part of mu
    i_out += 1  # increment output index
out = out[2:i_out]  # remove the first two, and anything after i_out (that was never filled out)

if False:  # when we disabled both imperfections and turned OFF interpolator
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(8, 5))  # 7 is nearly full width
    fig.tight_layout(pad=1.0)  # add space between subplots
    ax1.plot(original_data, '.-')
    ax2.plot(np.real(samples[6 * 8:-7 * 8]), '.-')
    ax3.plot(np.real(out[6:-7]), '.-')
    #fig.savefig('/tmp/time-sync-output.svg', bbox_inches='tight')
else:
    fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 3.5))  # 7 is nearly full width
    ax1.plot(original_data, '.-')
    ax2.plot(np.real(out[6:-7]), '.-')
    ax2.plot(np.imag(out[6:-7]), '.-')
    #fig.savefig('/tmp/time-sync-output2.svg', bbox_inches='tight')

if False:
    # Plot constellation of before and after time sync
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))  # 7 is nearly full width
    ax1.plot(np.real(samples), np.imag(samples), '.')
    ax1.axis([-2, 2, -2, 2])
    ax1.set_title('Before Time Sync')
    ax1.grid()
    ax2.plot(np.real(out[32:-8]), np.imag(out[32:-8]), '.')  # leave out the ones at beginning, before sync finished
    ax2.axis([-2, 2, -2, 2])
    ax2.set_title('After Time Sync')
    ax2.grid()
    #fig.savefig('/tmp/time-sync-constellation.svg', bbox_inches='tight')

if False:  # Animated version of the above REMEMBER TO INCREASE NUMBER OF SAMPLES FOR THIS PLOT, I USED 300
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    line, = ax.plot([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], '.')
    ax.axis([-2, 2, -2, 2])

    # Add zeros at the beginning so that when gif loops it has a transition period
    temp_out = np.concatenate((np.zeros(50), out))


    def update(i):
        print(i)
        line.set_xdata([np.real(temp_out[i:i + 5])])
        line.set_ydata([np.imag(temp_out[i:i + 5])])
        return line, ax


    anim = FuncAnimation(fig, update, frames=np.arange(0, len(out - 5)), interval=20)
    anim.save('/tmp/time-sync-constellation-animated.gif', dpi=80, writer='imagemagick')
    exit()

# COSTAS LOOP.  THIS ONE IS FROM GNURADIO's COSTAS LOOP IMPL, AND REMEMBER IT INHERITS CONTROL LOOP WHICH IS IN GR-BLOCKS
samples = out  # copy samples from output of timing sync


# For QPSK
def phase_detector_4(sample):
    if sample.real > 0:
        a = 1.0
    else:
        a = -1.0
    if sample.imag > 0:
        b = 1.0
    else:
        b = -1.0
    return a * sample.imag - b * sample.real

'''convert in-line
# For BPSK
def phase_detector_2(sample):
    return sample.real * sample.imag
'''

N = len(samples)
phase = 0
freq = 0
loop_bw = 0.05  # This is what to adjust, to make the feedback loop faster or slower (which impacts stability)
damping = np.sqrt(2.0) / 2.0  # Set the damping factor for a critically damped system
alpha = (4 * damping * loop_bw) / (1.0 + (2.0 * damping * loop_bw) + loop_bw ** 2)
beta = (4 * loop_bw ** 2) / (1.0 + (2.0 * damping * loop_bw) + loop_bw ** 2)
print("alpha:", alpha)
print("beta:", beta)
out = np.zeros(N, dtype=np.complex64)
freq_log = []
for i in range(N):
    out[i] = samples[i] * np.exp(-1j * phase)  # adjust the input sample by the inverse of the estimated phase offset

    if True:
        error = np.real(out[i]) * np.imag(out[i])  # This is the error formula for 2nd order Costas Loop (e.g. for BPSK)
    else:
        error = phase_detector_4(out[i])

    # Limit error to the range -1 to 1
    # error = min(error, 1.0) # left out for sake of reducing code.  didnt seem to get anywhere near 1 or -1
    # error = max(error, -1.0)

    # Advance the loop (recalc phase and freq offset)
    freq += (beta * error)
    freq_log.append(freq / 50.0 * fs)  # see note at bottom
    phase += freq + (alpha * error)

    # Adjust phase so its always between 0 and 2pi, recall that phase wraps around every 2pi
    while phase >= 2 * np.pi:
        phase -= 2 * np.pi
    while phase < 0:
        phase += 2 * np.pi

    # Limit frequency to range -1 to 1
    # freq = min(freq, 1.0) # didnt get anywhere near 1 or -1 in this example, leaving out for sake of understanding code
    # freq = max(freq, -1.0)

# Output, energy should be only in real portion, and samples should be ready for demodulator
# TO MAKE IT LOOK A LITTLE NICER I CAN COMMENT OUT THE PART WHERE I CORRECT THE FREQ OFFSET
fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 5))  # 7 is nearly full width
fig.tight_layout(pad=2.0)  # add space between subplots
ax1.plot(np.real(samples), '.-')
ax1.plot(np.imag(samples), '.-')
ax1.set_title('Before Costas Loop')
ax2.plot(np.real(out), '.-')
ax2.plot(np.imag(out), '.-')
ax2.set_title('After Costas Loop')
#fig.savefig('/tmp/costas-loop-output.svg', bbox_inches='tight')

# Show freq offset being honed in on
fig, ax = plt.subplots(figsize=(7, 3))  # 7 is nearly full width
# For some reason you have to divide the steady state freq by 50,
#   to get the fraction of fs that the fo is... 
#   and changing loop_bw doesn't matter
ax.plot(freq_log, '.-')
ax.set_xlabel('Sample')
ax.set_ylabel('Freq Offset')
#fig.savefig('/tmp/costas-loop-freq-tracking.svg', bbox_inches='tight')

plt.show()
