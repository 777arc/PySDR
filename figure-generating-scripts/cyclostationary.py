import numpy as np
import matplotlib.pyplot as plt

######################
# Simulate Rect BPSK #
######################

N = 100000 # number of samples to simulate
f_offset = 0.2 # Hz normalized
sps = 20 # cyclic freq (alpha) will be 1/sps or 0.05 Hz normalized

symbols = np.random.randint(0, 2, int(np.ceil(N/sps))) * 2 - 1 # random 1's and -1's
bpsk = np.repeat(symbols, sps)  # repeat each symbol sps times to make rectangular BPSK
bpsk = bpsk[:N]  # clip off the extra samples
bpsk = bpsk * np.exp(2j * np.pi * f_offset * np.arange(N)) # Freq shift up the BPSK, this is also what makes it complex
noise = np.random.randn(N) + 1j*np.random.randn(N) # complex white Gaussian noise
samples = bpsk + 0.1*noise  # add noise to the signal

if True:
    # Plot PSD
    PSD = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples)/N))**2)
    f = np.linspace(-0.5, 0.5, len(PSD))
    plt.plot(f, PSD)
    plt.xlabel('Frequency')
    plt.ylabel('PSD')
    plt.grid()
    plt.savefig('../_images/psd_of_bpsk_used_for_caf.svg', bbox_inches='tight')
    plt.show()
    exit()

##############
# Direct CAF #
##############

# CAF only at the correct alpha
correct_alpha = 1/sps
taus = np.arange(-100, 100)
CAF = np.zeros(len(taus), dtype=complex)
for i in range(len(taus)):
    CAF[i] = np.sum(np.roll(samples, -1*taus[i]//2) *
                    np.conj(np.roll(samples, taus[i]//2)) *
                    np.exp(-2j * np.pi * correct_alpha * np.arange(N)))
plt.figure(0)
plt.plot(taus, np.real(CAF))
plt.xlabel('Tau')
plt.ylabel('CAF (real part)')
plt.savefig('../_images/caf_at_correct_alpha.svg', bbox_inches='tight')

# Used in SCF section
plt.figure(2)
f = np.linspace(-0.5, 0.5, len(taus))
SCF = np.fft.fftshift(np.fft.fft(CAF))
plt.plot(f, np.abs(SCF))
plt.xlabel('Frequency')
plt.ylabel('SCF')
plt.grid()
plt.savefig('../_images/fft_of_caf.svg', bbox_inches='tight')

# CAF at all alphas
alphas = np.arange(0, 0.5, 0.005)
CAF = np.zeros((len(alphas), len(taus)), dtype=complex)
for j in range(len(alphas)):
    for i in range(len(taus)):
        CAF[j, i] = np.sum(np.roll(samples, -1*taus[i]//2) *
                        np.conj(np.roll(samples, taus[i]//2)) *
                        np.exp(-2j * np.pi * alphas[j] * np.arange(N)))
plt.figure(1)
plt.plot(alphas, np.average(np.abs(CAF), axis=1))
plt.xlabel('Alpha')
plt.ylabel('CAF Power')
plt.savefig('../_images/caf_avg_over_alpha.svg', bbox_inches='tight')

plt.show()






##### EVERYTHING BEYOND THIS POINT ISNT IN THE CHAPTER YET ##########

from scipy.linalg import norm
import time

fs = 1  # sample rate in Hz
N = int(2**14)  # number of samples to simulate



def generate_bspk(sps, f_offset):
    # Our data to be transmitted, 1's and 0's
    bits = np.random.randint(0, 2, int(np.ceil(N/sps)))
    # bits = np.ones(int(np.ceil(N/sps)))
    bpsk = np.array([])
    for bit in bits:
        pulse = np.zeros(sps)
        pulse[0] = bit*2-1  # set the first value to either a 1 or -1
        bpsk = np.concatenate((bpsk, pulse))  # add the 8 samples to the signal
    num_taps = 101  # for our RRC filter
    if False:  # RC pulse shaping
        beta = 0.249
        t = np.arange(num_taps) - (num_taps-1)//2
        h = np.sinc(t/sps) * np.cos(np.pi*beta*t/sps) / (1 - (2*beta*t/sps)**2)
    else:  # rect pulses
        h = np.ones(sps)
    # Filter our signal, in order to apply the pulse shaping
    bpsk = np.convolve(bpsk, h, 'same')
    bpsk = bpsk[0:N]  # clip off the extra samples

    # Freq shift up the BPSK
    return bpsk * np.exp(2j*np.pi*f_offset*np.arange(len(bpsk))/fs)


noise = np.random.randn(N) + 1j*np.random.randn(N)
samples = generate_bspk(10, 0.0*fs) + 0.01*noise


#######
# SCF # based on https://github.com/fchirono/cyclostationarity_analysis
#######
if False:
    fft_size = 1024
    # range of alphas to test, dont include a=0
    alphas = np.linspace(0.01, 1, 50)*fs
    Syy = np.zeros((len(alphas), fft_size))  # cyclic spectral density
    rho_y = np.zeros((len(alphas), fft_size),
                     dtype='complex')  # cyclic coherence
    N_blocks = len(samples) // fft_size
    df = fs/fft_size
    freq = np.linspace(0, fs-df, fft_size)-fs/2

    # Spectral Correlation Density (SCD) function
    Sxx = np.zeros((N_blocks, len(alphas), fft_size))
    Syy = np.zeros((N_blocks, len(alphas), fft_size))
    Sxy = np.zeros((N_blocks, len(alphas), fft_size), dtype='complex')

    for n in range(N_blocks):
        n_start = n*fft_size
        t_block = np.linspace(n_start/fs, (n_start+fft_size)/fs, fft_size)
        x_block = samples[n_start: n_start+fft_size]

        # calculate spectra for alpha values in 'alphas'
        for a, alpha in enumerate(alphas):
            # applies frequency shift of +/- alpha/2, then take FFT of data blocks
            u_f = np.fft.fft(x_block*np.exp(-1j*np.pi*alpha*t_block))
            v_f = np.fft.fft(x_block*np.exp(+1j*np.pi*alpha*t_block))

            # calculates auto- and cross-power spectra
            Sxx[n, a, :] = (u_f * u_f.conj()).real
            Syy[n, a, :] = (v_f * v_f.conj()).real
            Sxy[n, a, :] = (u_f * v_f.conj())

    # Normalize
    Sxx *= 1/(fft_size*fs)
    Syy *= 1/(fft_size*fs)
    Sxy *= 1/(fft_size*fs)

    # apply FFT shift
    Sxx = np.fft.fftshift(Sxx, axes=(-1))
    Syy = np.fft.fftshift(Syy, axes=(-1))
    Sxy = np.fft.fftshift(Sxy, axes=(-1))

    # average over blocks
    Sxx_avg = Sxx.sum(axis=0)/N_blocks
    Syy_avg = Syy.sum(axis=0)/N_blocks
    S = Sxy.sum(axis=0)/N_blocks  # cyclic spectral density
    S = np.abs(S)

    rho_y = S/np.sqrt(Sxx_avg*Syy_avg)  # cyclic coherence

    # replace entries outside the principal domain in freq-alpha plane by NaNs (i.e. diamond shape given by |f| > (fs-|alpha|)/2 )
    for a, alpha in enumerate(alphas):
        freqs_outside = (np.abs(freq) > (fs - np.abs(alpha))/2)
        S[a, freqs_outside] = np.nan
        rho_y[a, freqs_outside] = np.nan


#######
# SCF # based on https://www.mathworks.com/matlabcentral/fileexchange/60561-fast_sc-x-window_len-alpha_max-fs-opt
#######
# Author: J. Antoni, September 2016
# Reference:
# J. Antoni, G. Xin, N. Hamzaoui, "Fast Computation of the Spectral Correlation",
# Mechanical Systems and Signal Processing, Elsevier, 2017.
if False:
    alpha_max = fs*0.49999  # in Hz
    # used in SFTF, gives more resolution in freq (not cyclic) domain
    window_len = int(2**7)

    Wind = np.hanning(window_len)

    # Set value of overlap
    R = np.fix(fs/2/alpha_max)  # block shift
    R = int(np.max((1, np.min((R, np.fix(0.25*window_len))))))  # type: ignore
    print("R:", R)

    Nv = window_len - R  # block overlap
    dt = R/fs  # time resolution of STFT (in s)

    # STFT computation prep
    R = window_len - Nv  # block shift
    fft_size = int(np.fix((len(samples)-Nv)/(window_len-Nv)))
    print("fft_size:", fft_size)

    def CPS_STFT_zoom(alpha0, STFT):
        NF, NT = np.shape(STFT)
        window_len = 2*(NF-1)
        alphas = np.arange(fft_size)/fft_size/dt
        # Computation "cross-frequency" cyclic modulation spectrum
        fk = int(np.round(alpha0/fs*window_len))
        alpha0 = fk/window_len*fs
        if fk >= 0:
            S = np.vstack((STFT[fk:NF, :], np.zeros((fk, NT)))) * np.conj(STFT)
        else:
            S = np.vstack(
                (np.conj(STFT[-fk:NF, :]), np.zeros((-fk, NT)))) * STFT
        S = np.fft.fft(S, fft_size, axis=1) / NT
        S = S / np.sum(np.square(Wind)) / fs  # Calibration
        # Removal of aliased cyclic frequencies
        ak = int(np.round(alpha0 * dt * fft_size))
        S[:, (int(np.ceil(fft_size/2))+ak):fft_size] = 0
        # Phase correction
        S = S * np.tile(np.exp(-2j*np.pi*np.argmax(Wind)
                        * (alphas-alpha0)/fs), (NF, 1))
        return S, alphas, alpha0

    def Shift_Window_STFT_zoom(W0, a0):
        a0 = int(a0)
        # Circular shift with linear interpolation for non-integer shifts
        a0_floor = int(np.floor(a0))
        a0_ceil = int(np.ceil(a0))
        if a0_floor == a0_ceil:
            W = np.roll(W0, a0)
        else:
            W = np.roll(W0, a0_floor)*(1-(a0-a0_floor)) + \
                np.roll(W0, a0_ceil)*(a0-a0_floor)
        # Removal of aliased cyclic frequencies
        W[int(np.ceil(len(W0)/2)+np.round(a0)):len(W0)] = 0
        return W

    # compute short-time Fourier transform (STFT)
    index = np.zeros(2, dtype=int)
    index[0] = 0
    index[1] = window_len
    STFT = np.zeros((window_len//2+1, fft_size), dtype=complex)
    for k in range(fft_size):
        Xw = np.fft.fft(np.hanning(window_len) *
                        samples[index[0]:index[1]], window_len)
        STFT[:, k] = Xw[0:window_len//2+1]
        index = index + R

    # spectral coherence function (COH) prep
    NF = STFT.shape[0]
    Nw_coh = 2*(NF-1)  # window length

    # Whitening the STFT for computing the spectral coherence
    Sx = np.mean(np.square(np.abs(STFT)), 1)  # Mean power spectral density
    STFT = STFT * np.tile(1/np.sqrt(Sx), (STFT.shape[1], 1)).T

    # Computation of the cyclic modulation spectrum
    S, alphas, unused = CPS_STFT_zoom(0, STFT)

    # Computation the "zooming" window
    WSquared = np.square(Wind)
    # set origin of time to the centre of symmetry (maximum value) of the window
    Iw = np.argmax(Wind)
    W1 = np.zeros(fft_size)
    W2 = np.zeros(fft_size)
    n = np.arange(1, Iw+1)/fs
    for k in range(fft_size):
        W1[k] = WSquared[Iw] + 2 * np.sum(WSquared[np.arange(Iw-1, -1, -1)] * np.cos(
            2*np.pi*n*(alphas[k])))  # "positive" frequencies
        W2[k] = WSquared[Iw] + 2 * np.sum(WSquared[np.arange(Iw-1, -1, -1)] * np.cos(
            2*np.pi*n*(alphas[k]-1/dt)))  # "negative" frequencies (aliased)
    W0 = W1 + W2  # Note: sum(W2) = max(W)

    W = np.copy(W0)
    W[int(np.ceil(fft_size/2)):fft_size] = 0  # truncate negative frequencies
    Fa = 1/dt  # cyclic sampling frequency in Hz
    K = int(np.fix(Nw_coh/2*Fa/fs))  # Number of scans

    # Computation and summation of "cross-frequency" cyclic modulation spectra
    for k in range(1, K+1):
        # positive cyclic frequencies
        Stemp, alphas, alpha0 = CPS_STFT_zoom(k/Nw_coh*fs, STFT)
        Wtemp = Shift_Window_STFT_zoom(W0, alpha0/Fa*fft_size)
        S[:, 1:fft_size] = S[:, 1:fft_size] + Stemp[:, 1:fft_size]
        W[1:fft_size] = W[1:fft_size] + Wtemp[1:fft_size]

        # negative cyclic frequencies
        Stemp, alphas, alpha0 = CPS_STFT_zoom(-k / Nw_coh*fs, STFT)
        Wtemp = Shift_Window_STFT_zoom(W0, alpha0/Fa*fft_size)
        S[:, 1:fft_size] = S[:, 1:fft_size] + Stemp[:, 1:fft_size]
        W[1:fft_size] = W[1:fft_size] + Wtemp[1:fft_size]

    Winv = np.ones(fft_size)
    I = np.nonzero(W < 0.5*W[0])[0]
    Winv[0:I[0]] = 1/W[0:I[0]]
    Winv[I[0] + 0:fft_size] = 1/W[I[0]]

    S = S * np.tile(Winv, (NF, 1)) * np.sum(np.square(Wind))
    S = S/np.mean(S[:, 0])

    I = np.nonzero(alphas <= alpha_max)[0]
    alphas = alphas[I]
    S = S[:, I]
    S = np.abs(S)
    S = S[:, 1:]
    alphas = alphas[1:]
    S = S/np.max(np.max(S))  # normalize
    S = S.T  # to match the other method

    # Max pooling in alpha axis
    print(S.shape)
    pool_size = 10
    S = np.array([np.max(S[i:i+pool_size], axis=0)
                 for i in range(0, len(S), pool_size)])
    print(S.shape)

##########################
# Time smoothing SCF # based on https://www.mathworks.com/matlabcentral/fileexchange/48909-cyclic-spectral-analysis
##########################

if False:
    alphas = np.arange(0.05, 0.5, 0.005)
    Nw = 256  # window length
    N = len(samples)  # signal length
    Noverlap = int(2/3*Nw)  # block overlap
    num_windows = int((N - Noverlap) / (Nw - Noverlap))  # Number of windows
    window = np.hanning(Nw)

    S = np.zeros((Nw, len(alphas)), dtype=complex)
    for ii in range(len(alphas)):  # Loop over cyclic frequencies
        print("alpha:", alphas[ii])
        neg = samples * np.exp(-1j*np.pi*alphas[ii]*np.arange(N))
        pos = samples * np.exp(1j*np.pi*alphas[ii]*np.arange(N))

        # Cross Cyclic Power Spectrum
        for i in range(num_windows):
            pos_slice = window * pos[i*(Nw-Noverlap):i*(Nw-Noverlap)+Nw]
            neg_slice = window * neg[i*(Nw-Noverlap):i*(Nw-Noverlap)+Nw]
            S[:, ii] += np.fft.fft(neg_slice) * np.conj(np.fft.fft(pos_slice))

    S = np.abs(S)

    plt.imshow(S, aspect='auto', extent=(
        float(np.min(alphas)), float(np.max(alphas)), fs, 0.0))
    plt.xlabel('Cyclic Frequency [Hz]')
    plt.ylabel('Frequency [Normalized Hz]')
    plt.show()



