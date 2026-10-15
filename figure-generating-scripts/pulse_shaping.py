# %%
import os
import numpy as np
import matplotlib.pyplot as plt

# %%
# Inter-Symbol-Interference (ISI), a few raised-cosine pulses before being combined
sps = 32

# Create our raised-cosine filter
num_taps = 1001
beta = 0.35
Ts = sps # Assume sample rate is 1 Hz, so sample period is 1, so *symbol* period is 8
t = np.arange(num_taps//-2, num_taps//2) # remember it's not inclusive of final number
h = np.sinc(t/Ts) * np.cos(np.pi*beta*t/Ts) / (1 - (2*beta*t/Ts)**2)

N = 2000
fig, (ax) = plt.subplots(1, 1, figsize=(10, 4))
for delay in np.arange(-2,3):
    x = np.zeros(N)
    x[N//2 + delay*sps] = 1
    x_shaped = np.convolve(x, h, 'same')
    t = (np.arange(N) - N/2)/sps
    ax.plot(t, x_shaped, '-')

ax.axis([-7, 7, -0.3, 1.2])
ax.set_xlabel("Time")
ax.set_ylabel("Pulses (before being combined)")

#ax.xaxis.set_major_locator(MaxNLocator(integer=True))
major_ticks = np.arange(-6, 7)
ax.set_xticks(major_ticks)

plt.grid(True)
#if os.path.isdir('../_images'):
#    fig.savefig('../_images/pulse_train.svg', bbox_inches='tight')
plt.show()

# %%
# Raised-cosine impulse response for several roll-off factors
_fig, (ax1) = plt.subplots(1, 1, figsize=(10, 3))
plt.subplots_adjust(wspace=0.4)

_sps = 100
_t = np.linspace(-4 * _sps, 4 * _sps, 1000)
leg = []
for _beta in [0, 0.25, 0.5, 1]:
    _h = np.sinc(_t / _sps) * np.cos(np.pi * _beta * _t / _sps) / (1 - (2 * _beta * _t / _sps) ** 2)
    ax1.plot(_t, _h)
    leg.append('β = ' + str(_beta))

plt.legend(leg, fontsize=14)
ax1.text(max(_t) * 1.12, -0.05, 'Time', fontsize=14)
ax1.text(-4 * _sps - 15, -0.1, '-4T', fontsize=14)
ax1.text(-3 * _sps - 15, -0.1, '-3T', fontsize=14)
ax1.text(-2 * _sps - 15, -0.1, '-2T', fontsize=14)
ax1.text(-1 * _sps - 15, -0.1, '-T', fontsize=14)
ax1.text(0 * _sps - 15, -0.1, '0', fontsize=14)
ax1.text(1 * _sps - 15, -0.1, '1T', fontsize=14)
ax1.text(2 * _sps - 15, -0.1, '2T', fontsize=14)
ax1.text(3 * _sps - 15, -0.1, '3T', fontsize=14)
ax1.text(4 * _sps - 15, -0.1, '4T', fontsize=14)
ax1.text(-15, 1.1, 'h(t)', fontsize=14)

# set the x-spine (see below for more info on `set_position`)
ax1.spines['left'].set_position('zero')

# turn off the right spine/ticks
ax1.spines['right'].set_color('none')
ax1.yaxis.tick_left()

# set the y-spine
ax1.spines['bottom'].set_position('zero')

# turn off the top spine/ticks
ax1.spines['top'].set_color('none')
ax1.xaxis.tick_bottom()

# Turn off tick numbering/labels
ax1.set_xticklabels([])
ax1.set_yticklabels([])

plt.show()

#if os.path.isdir('../_images'):

#    fig.savefig('../_images/raised_cosine.svg', bbox_inches='tight')

# %%
# Raised-cosine frequency response for several roll-off factors
_fig, (_ax1) = plt.subplots(1, 1, figsize=(10, 3))
plt.subplots_adjust(wspace=0.4)

_sps = 100
_leg = []
for _beta in [0.01, 0.25, 0.5, 1]:
    f = np.linspace(-1.5 / _sps, 1.5 / _sps, 1000)
    H = []
    for fi in f:
        if np.abs(fi) <= (1 - _beta) / 2 / _sps:
            H.append(1)
        elif (np.abs(fi) <= (1 + _beta) / 2 / _sps) and (np.abs(fi) > (1 - _beta) / 2 / _sps):
            H.append(0.5 * (1 + np.cos(np.pi * _sps / _beta * (np.abs(fi) - (1 - _beta) / (2 * _sps)))))
        else:
            H.append(0)
    _ax1.plot(f, H)
    if _beta == 0.01:
        _beta = 0
    _leg.append('β = ' + str(_beta))

plt.legend(_leg, fontsize=14)
_ax1.text(max(f) * 1.12, -0.05, 'Freq', fontsize=14)
_ax1.text(-1.5 / _sps - 0.0015, -0.1, '-3/2T', fontsize=14)
_ax1.text(-1 / _sps - 0.0015, -0.1, '-1/T', fontsize=14)
_ax1.text(-0.5 / _sps - 0.0015, -0.1, '-1/2T', fontsize=14)
_ax1.text(0, -0.1, '0', fontsize=14)
_ax1.text(1.5 / _sps - 0.0015, -0.1, '3/2T', fontsize=14)
_ax1.text(1 / _sps - 0.0005, -0.1, '1/T', fontsize=14)
_ax1.text(0.5 / _sps - 0.0015, -0.1, '1/2T', fontsize=14)
_ax1.text(0, 1.1, 'H(f)', fontsize=14)

# set the x-spine (see below for more info on `set_position`)
_ax1.spines['left'].set_position('zero')

# turn off the right spine/ticks
_ax1.spines['right'].set_color('none')
_ax1.yaxis.tick_left()

# set the y-spine
_ax1.spines['bottom'].set_position('zero')

# turn off the top spine/ticks
_ax1.spines['top'].set_color('none')
_ax1.xaxis.tick_bottom()

# Turn off tick numbering/labels
_ax1.set_xticklabels([])
_ax1.set_yticklabels([])

plt.show()

#if os.path.isdir('../_images'):

#    fig.savefig('../_images/raised_cosine_freq.svg', bbox_inches='tight')

# %%
# Roll-off factor, a wider view of the impulse response
_fig, (_ax1) = plt.subplots(1, 1, figsize=(10, 3))
plt.subplots_adjust(wspace=0.4)

_sps = 100
_t = np.linspace(-10 * _sps, 10 * _sps, 1000)
_leg = []
for _beta in [0, 0.25, 0.5, 1]:
    _h = np.sinc(_t / _sps) * np.cos(np.pi * _beta * _t / _sps) / (1 - (2 * _beta * _t / _sps) ** 2)
    _ax1.plot(_t, _h)
    _leg.append('β = ' + str(_beta))

plt.legend(_leg, fontsize=14)
# for T in range(-10, 11):
#    ax1.text(T*sps - 15, -0.1, str(T) + 'T', fontsize=10)

_ax1.text(-15, 1.1, 'h(t)', fontsize=14)
_ax1.text(max(_t) * 1.12, -0.05, 'Time', fontsize=14)

# set the x-spine (see below for more info on `set_position`)
_ax1.spines['left'].set_position('zero')

# turn off the right spine/ticks
_ax1.spines['right'].set_color('none')
_ax1.yaxis.tick_left()

# set the y-spine
_ax1.spines['bottom'].set_position('zero')

# turn off the top spine/ticks
_ax1.spines['top'].set_color('none')
_ax1.xaxis.tick_bottom()

# Turn off tick numbering/labels
_ax1.set_xticklabels([])
_ax1.set_yticklabels([])

plt.show()

#if os.path.isdir('../_images'):

#    fig.savefig('../_images/rrc_rolloff.svg', bbox_inches='tight')

# %%
# Python Exercise, pulse shaping BPSK with a raised-cosine filter
num_symbols = 10
_sps = 8

#bits = np.random.randint(0, 2, num_symbols) # Our data to be transmitted, 1's and 0's
bits = [0, 1, 1, 1, 1, 0, 0, 0, 1, 1]

_x = np.array([])
for bit in bits:
    pulse = np.zeros(_sps)
    pulse[0] = bit*2-1 # set the first value to either a 1 or -1
    _x = np.concatenate((_x, pulse)) # add the 8 samples to the signal

# Create our raised-cosine filter
_num_taps = 101
_beta = 0.35
_Ts = _sps # Assume sample rate is 1 Hz, so sample period is 1, so *symbol* period is 8
_t = np.arange(_num_taps) - (_num_taps-1)//2
print(_t)
print(len(_t))
_h = np.sinc(_t/_Ts) * np.cos(np.pi*_beta*_t/_Ts) / (1 - (2*_beta*_t/_Ts)**2)

# Filter our signal, in order to apply the pulse shaping
_x_shaped = np.convolve(_x, _h)
_fig, _ax = plt.subplots(1,1,figsize=(10,4))
plt.plot(_x_shaped, '.-')
for i in range(num_symbols):
    xpos = i*_sps+_num_taps//2+1
    ypos = _x_shaped[xpos]
    plt.arrow(x=xpos, y=0, dx=0, dy=ypos, color="red", ls=(0, (5, 5)))
plt.grid(True)
_ax.set_yticks([-1, 0, +1], minor=False)
plt.show()
#fig.savefig('../_images/pulse_shaping_python3.svg', bbox_inches='tight')

# %%
# Eye Diagrams
np.random.seed(5)

# BPSK + RRC pulse shaping, same style of setup as the chapter's Python exercise
_num_symbols = 400
_sps = 32
_beta = 0.35
span = 8  # filter span in symbols (each side)

# Random BPSK symbols
symbols = np.random.randint(0, 2, _num_symbols) * 2 - 1

# Upsample (impulses spaced by sps)
_x = np.zeros(_num_symbols * _sps)
_x[::_sps] = symbols

# Raised-cosine filter (Tx + Rx combined for this illustration)
_t = np.arange(-span * _sps, span * _sps + 1) / _sps
_h = np.sinc(_t) * np.cos(np.pi * _beta * _t) / (1 - (2 * _beta * _t) ** 2 + 1e-20)
_h /= np.max(np.convolve(np.ones(1), _h))  # keep peaks near +/-1

y = np.convolve(_x, _h, mode='same')

# Build the eye diagram: chop the signal into 2-symbol-wide slices and overlay
span_samps = 2 * _sps
n_traces = 250
_fig, _ax = plt.subplots(1, 1, figsize=(8, 4))
start = span * _sps  # skip filter transient
for _i in range(n_traces):
    s = start + _i * _sps
    seg = y[s:s + span_samps]
    if len(seg) < span_samps:
        break
    _ax.plot(np.arange(span_samps) / _sps - 1, seg, color='#1f77b4', alpha=0.15, linewidth=1, rasterized=True)

# Mark the ideal sampling instant (center of the eye)
_ax.axvline(0, color='k', linestyle='--', linewidth=1)
_ax.text(0.02, 1.35, 'ideal sample time', fontsize=11)

_ax.set_xlabel('Time (symbol periods)', fontsize=12)
_ax.set_ylabel('Amplitude', fontsize=12)
_ax.set_xlim(-1, 1)
_ax.set_ylim(-1.6, 1.6)
_ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
#if os.path.isdir('../_images'):
#    fig.savefig('../_images/eye_diagram.svg', bbox_inches='tight', dpi=100)

# %%
# OQPSK and MSK
# Parameters
_num_symbols = 20000
_span = 6         # filter span in symbols (each side)

# Generate QPSK symbols
_bits = np.random.randint(0, 4, _num_symbols)
qpsk_symbols = np.exp(1j * (np.pi/4 + _bits * np.pi/2)).astype(complex)  # points at 45°, 135°, 225°, 315°

def rc_filter(sps, beta=0.35):
    """Raised-cosine filter, beta is the roll-off factor"""
    t = np.arange(-span * sps, span * sps + 1) / sps  # in symbol periods
    return np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t)**2 + 1e-20)

def half_sine_filter(sps):
    """Half-sine pulse shape"""
    t = np.arange(sps)
    return np.sin(np.pi * t / sps)

def qpsk(symbols, h, sps):
    upsampled = np.zeros(len(symbols) * sps, dtype=complex)
    upsampled[::sps] = symbols
    return np.convolve(upsampled, h, mode='same')

def oqpsk(symbols, h, sps):
    # Delay Q impulses by half a symbol before filtering so the pulse shaping filter handles the ramp-up naturally (no post-filter roll/zero-fill artifact)
    half = sps // 2
    I_up = np.zeros(len(symbols) * sps)
    Q_up = np.zeros(len(symbols) * sps)
    I_up[::sps] = np.real(symbols)
    Q_up[half::sps] = np.imag(symbols)
    I_filt = np.convolve(I_up, h, mode='same')
    Q_filt = np.convolve(Q_up, h, mode='same')
    return I_filt + 1j * Q_filt

def plot_magnitude(signal, sps, mode, N=10):
    """Plots I, Q and the magnitude of the first N symbols"""
    fig, axes = plt.subplots(2, 1, figsize=(7, 4), tight_layout=True)
    axes[0].plot(signal.real[:N*sps], label='I')
    axes[0].plot(signal.imag[:N*sps], label='Q', alpha=0.7)
    axes[0].axhline(1, color='gray', linestyle='--', linewidth=1)
    axes[0].axhline(-1, color='gray', linestyle='--', linewidth=1)
    start = -1 if mode == 'CPFSK' else 0 # the -1 is only for CPFSK to align things
    for x in range(start, N * sps, sps):
        axes[0].axvline(x, color='gray', linestyle='--', linewidth=1)
    if mode == 'OQPSK' or mode == 'MSK' or mode == 'CPFSK':
        for x in range(sps // 2 + start, N * sps, sps):
            axes[0].axvline(x, color='blue', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].set_title(mode)
    axes[0].legend()
    axes[1].plot(np.abs(signal[:N*sps]))
    axes[1].set_ylabel('Magnitude')
    axes[1].set_xlabel('Sample Index (Time)')
    axes[1].set_ylim(bottom=0, top=1.2)
    axes[1].grid(True)
    return fig

def plot_psd(signal, title):
    psd = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(signal)))**2)
    psd -= np.max(psd)  # Normalize to 0 dB max
    f = np.linspace(-0.5, 0.5, len(psd))
    fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)
    ax.plot(f, psd)
    ax.set_ylim(bottom=-80, top=5)
    ax.set_title(title)
    ax.set_xlabel('Normalized Frequency (cycles/sample)')
    ax.set_ylabel('PSD [dB]')
    ax.grid(True)
    return fig

# %%
# QPSK with raised-cosine pulse shaping
_sps = 32 # samples per symbol
signal_qpsk = qpsk(qpsk_symbols, rc_filter(_sps), _sps)
fig_qpsk = plot_magnitude(signal_qpsk, _sps, 'QPSK')
#if os.path.isdir('../_images'):
#    fig_qpsk.savefig('../_images/qpsk_magnitude.svg', bbox_inches='tight')
fig_qpsk

# %%
# OQPSK with raised-cosine pulse shaping
_sps = 32 # samples per symbol
signal_oqpsk = oqpsk(qpsk_symbols, rc_filter(_sps), _sps)
fig_oqpsk = plot_magnitude(signal_oqpsk, _sps, 'OQPSK')
#if os.path.isdir('../_images'):
#    fig_oqpsk.savefig('../_images/oqpsk_magnitude.svg', bbox_inches='tight')
fig_oqpsk

# %%
# MSK, which is OQPSK with the half-sine pulse shape instead of raised-cosine
_sps = 32 # samples per symbol
signal_msk = oqpsk(qpsk_symbols, half_sine_filter(_sps), _sps)
fig_msk = plot_magnitude(signal_msk, _sps, 'MSK')
#if os.path.isdir('../_images'):
#    fig_msk.savefig('../_images/msk_magnitude.svg', bbox_inches='tight')
fig_msk

# %%
# PSDs, these use 8 samples per symbol instead of 32, so the signal takes up more of the plot
fig_rc_psd = plot_psd(oqpsk(qpsk_symbols, rc_filter(8), 8), 'QPSK or OQPSK with RC Pulse Shaping')
#if os.path.isdir('../_images'):
#    fig_rc_psd.savefig('../_images/qpsk_psd.svg', bbox_inches='tight')
fig_msk_psd = plot_psd(oqpsk(qpsk_symbols, half_sine_filter(8), 8), 'MSK')
#if os.path.isdir('../_images'):
#    fig_msk_psd.savefig('../_images/msk_psd.svg', bbox_inches='tight')
plt.show()

# %%
# Power Spectral Density comparison: MSK vs QPSK with Raised Cosine (α=0.35) pulse shaping, using the closed-form equations
# MSK uses a half-sinusoidal pulse shape, giving a naturally compact spectrum.  QPSK with raised-cosine filtering is shown for comparison.


# ── Parameters ──────────────────────────────────────────────────────
Rb = 1.0            # Bit rate (normalised to 1)
Tb = 1.0 / Rb       # Bit period
_Ts = 2 * Tb          # Symbol period (2 bits per QPSK symbol)
alpha = 0.35         # Roll-off factor for raised-cosine filter
_N = 8192             # FFT length for smooth curves
fmax = 3.0           # Max normalised frequency (f·Tb) to display

_f = np.linspace(-fmax, fmax, _N)
# Avoid exact zeros that cause division issues
f_safe = np.where(_f == 0, 1e-30, _f)

# ── 1.  MSK Power Spectral Density (closed-form) ───────────────────
# MSK PSD:  S(f) = (16·Tb / π²) · [ cos(2π·f·Tb) / (1 - 16·f²·Tb²) ]²
# Normalised so that Eb = Tb (energy per bit = bit period for unit power)
numerator = np.cos(2 * np.pi * f_safe * Tb)
denominator = 1.0 - 16.0 * (f_safe * Tb) ** 2

# Handle the removable singularities at f·Tb = ±0.25
psd_msk = np.where(
    np.abs(np.abs(_f * Tb) - 0.25) < 1e-8,
    (16.0 * Tb / np.pi**2) * (np.pi / 8.0) ** 2,   # L'Hôpital limit
    (16.0 * Tb / np.pi**2) * (numerator / denominator) ** 2,
)

# ── 2.  QPSK with Raised-Cosine filter PSD ─────────────────────────
# The raised-cosine spectrum (frequency domain) for roll-off α:
#   H(f) = { Ts,                                           |f| <= (1-α)/(2Ts)
#          { Ts/2 · [1 + cos(π·Ts/α·(|f| - (1-α)/(2Ts)))], passband edge
#          { 0,                                             |f| > (1+α)/(2Ts)  }
# PSD of QPSK = Es · |H(f)|²  (Es = symbol energy, set to 2·Tb for fair comparison)

f1 = (1 - alpha) / (2 * _Ts)   # passband edge
f2 = (1 + alpha) / (2 * _Ts)   # stopband edge

H_rc = np.zeros_like(_f)
abs_f = np.abs(_f)

# Flat region
mask_flat = abs_f <= f1
H_rc[mask_flat] = _Ts

# Roll-off region
mask_roll = (abs_f > f1) & (abs_f <= f2)
H_rc[mask_roll] = (_Ts / 2.0) * (
    1.0 + np.cos((np.pi * _Ts / alpha) * (abs_f[mask_roll] - f1))
)

# PSD of shaped QPSK (energy per symbol Es = 2·Eb = 2·Tb)
Es = 2 * Tb
psd_qpsk_rc = Es * (H_rc / _Ts) ** 2   # normalise H so peak PSD matches

# ── 3.  Unfiltered QPSK (rectangular pulse) for reference ──────────
# S(f) = 2·Tb · sinc²(f·Ts)
psd_qpsk_rect = 2 * Tb * np.sinc(_f * _Ts) ** 2

# ── Normalise all to 0 dB peak ─────────────────────────────────────
psd_msk_dB       = 10 * np.log10(psd_msk / psd_msk.max() + 1e-30)
psd_qpsk_rc_dB   = 10 * np.log10(psd_qpsk_rc / psd_qpsk_rc.max() + 1e-30)
psd_qpsk_rect_dB = 10 * np.log10(psd_qpsk_rect / psd_qpsk_rect.max() + 1e-30)

# ── Plot ────────────────────────────────────────────────────────────
_fig, _ax = plt.subplots(figsize=(10, 6))

_ax.plot(_f, psd_msk_dB,       linewidth=2.2, label="MSK", color="#E63946")
_ax.plot(_f, psd_qpsk_rc_dB,   linewidth=2.2, label=f"QPSK  (RC α = {alpha})", color="#457B9D")
_ax.plot(_f, psd_qpsk_rect_dB, linewidth=1.4, label="QPSK  (rectangular)", color="#457B9D",
        linestyle="--", alpha=0.55)

_ax.set_xlim(-fmax, fmax)
_ax.set_ylim(-50, 3)
_ax.set_xlabel("Normalized Frequency, fTₛ", fontsize=13)
_ax.set_ylabel("Power Spectral Density [dB]", fontsize=13)
_ax.set_title("MSK vs QPSK - Spectral Comparison", fontsize=14, fontweight="bold")
_ax.legend(fontsize=12, loc="upper right")
_ax.grid(True, alpha=0.3, linewidth=0.6)

plt.tight_layout()
#if os.path.isdir('../_images'):
#    plt.savefig('../_images/msk_vs_qpsk_spectrum.svg', dpi=150)
plt.show()

# %%
# MSK generated as Continuous-Phase FSK (CPFSK)
_num_symbols = 20000
_sps = 32 # samples per symbol
_bits = np.random.randint(0, 2, _num_symbols)
_symbols = 2 * _bits - 1 # map {0,1} → {-1, +1}

# Build the instantaneous frequency deviation
mod_index = 0.5
_t = np.arange(_num_symbols * _sps / 2) / (_sps / 2)
freq_dev = np.zeros(_num_symbols * _sps // 2)
for k, a in enumerate(_symbols):
    freq_dev[k * _sps // 2 : (k + 1) * _sps // 2] = a * mod_index / 2.0

phase = 2.0 * np.pi * np.cumsum(freq_dev) / (_sps / 2) # accumulate phase
signal_cpfsk = np.exp(1j * phase)
fig_cpfsk = plot_magnitude(signal_cpfsk, _sps, 'CPFSK')
#if os.path.isdir('../_images'):
#    fig_cpfsk.savefig('../_images/cpfsk_magnitude.svg', bbox_inches='tight')
fig_cpfsk
