import numpy as np
import matplotlib.pyplot as plt
import time
import scipy
import scipy.signal



######################
# Simulate Rect BPSK #
######################

if True:
    N = 100000 # number of samples to simulate
    f_offset = 0.2 # Hz normalized
    sps = 20 # cyclic freq (alpha) will be 1/sps or 0.05 Hz normalized

    # BPSK
    symbols = np.random.randint(0, 2, int(np.ceil(N/sps))) * 2 - 1 # random 1's and -1's
    # QPSK
    #symbols = (np.random.randint(0, 2, int(np.ceil(N/sps))) * 2 - 1) + 1j*(np.random.randint(0, 2, int(np.ceil(N/sps))) * 2 - 1) # QPSK
    bpsk = np.repeat(symbols, sps)  # repeat each symbol sps times to make rectangular BPSK
    bpsk = bpsk[:N]  # clip off the extra samples
    bpsk = bpsk * np.exp(2j * np.pi * f_offset * np.arange(N)) # Freq shift up the BPSK, this is also what makes it complex
    noise = np.random.randn(N) + 1j*np.random.randn(N) # complex white Gaussian noise
    samples = bpsk + 0.1*noise  # add noise to the signal

if False:
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


##############################################
# BPSK with Pulse Shaping (replaces samples) #
##############################################

if False:
    N = 100000 # number of samples to simulate
    f_offset = 0.2 # Hz normalized
    sps = 20 # cyclic freq (alpha) will be 1/sps or 0.05 Hz normalized
    num_symbols = int(np.ceil(N/sps))
    symbols = np.random.randint(0, 2, num_symbols) * 2 - 1 # random 1's and -1's
    #symbols = (np.random.randint(0, 2, num_symbols) * 2 - 1) + 1j*(np.random.randint(0, 2, num_symbols) * 2 - 1) # QPSK

    pulse_train = np.zeros(num_symbols * sps, dtype=complex)
    pulse_train[::sps] = symbols # easier explained by looking at an example output
    #print(pulse_train[0:96].astype(int))

    # Raised-Cosine Filter for Pulse Shaping
    beta = 0.3 # rolloff parameter (avoid exactly 0.2, 0.25, 0.5, and 1.0)
    #beta = 0.6 # 2nd fig
    #beta = 0.9 # 3rd fig
    num_taps = 101 # somewhat arbitrary
    t = np.arange(num_taps) - (num_taps-1)//2
    h = np.sinc(t/sps) * np.cos(np.pi*beta*t/sps) / (1 - (2*beta*t/sps)**2) # RC equation
    bpsk = np.convolve(pulse_train, h, 'same') # apply the pulse shaping
    
    bpsk = bpsk[:N]  # clip off the extra samples
    
    # Need to plot it before the freq shift up
    if False:
        plt.plot(bpsk[0:sps*10])
        for i in range(10):
            plt.text(i*sps, int(pulse_train[i*sps]), str(int((pulse_train[i*sps] + 1)/2)), ha='left', fontsize=16) # show 1's and 0's
        plt.grid()
        plt.axis((0, sps*10, -2, 2))
        plt.xlabel("Sample Index")
        plt.ylabel("Sample Value (I)")
        plt.savefig('../_images/pulse_shaped_BSPK.svg', bbox_inches='tight')
        plt.show()
        exit()
    
    bpsk = bpsk * np.exp(2j * np.pi * f_offset * np.arange(N)) # Freq shift up the BPSK, this is also what makes it complex
    noise = np.random.randn(N) + 1j*np.random.randn(N) # complex white Gaussian noise
    samples = bpsk + 0.1*noise  # add noise to the signal



###################################################
# Multiple overlapping signals (replaces samples) #
###################################################
if False:
    N = 1000000 # number of samples to simulate

    def fractional_delay(x, delay):
        N = 21 # number of taps
        n = np.arange(-N//2, N//2) # ...-3,-2,-1,0,1,2,3...
        h = np.sinc(n - delay) # calc filter taps
        h *= np.hamming(N) # window the filter to make sure it decays to 0 on both sides
        h /= np.sum(h) # normalize to get unity gain, we don't want to change the amplitude/power
        return np.convolve(x, h, 'same') # apply filter

    # Signal 1, Rect BPSK
    sps = 20
    f_offset = 0.2
    signal1 = np.repeat(np.random.randint(0, 2, int(np.ceil(N/sps))) * 2 - 1, sps)
    signal1 = signal1[:N] * np.exp(2j * np.pi * f_offset * np.arange(N))
    signal1 = fractional_delay(signal1, 0.12345)

    # Signal 2, Pulse-shaped BPSK
    sps = 20
    f_offset = -0.1
    beta = 0.35
    symbols = np.random.randint(0, 2, int(np.ceil(N/sps))) * 2 - 1
    pulse_train = np.zeros(int(np.ceil(N/sps)) * sps)
    pulse_train[::sps] = symbols
    t = np.arange(101) - (101-1)//2
    h = np.sinc(t/sps) * np.cos(np.pi*beta*t/sps) / (1 - (2*beta*t/sps)**2)
    signal2 = np.convolve(pulse_train, h, 'same')
    signal2 = signal2[:N] * np.exp(2j * np.pi * f_offset * np.arange(N))
    signal2 = fractional_delay(signal2, 0.52634)

    # Signal 3, Pulse-shaped QPSK
    sps = 4
    f_offset = 0.2
    beta = 0.21
    data = x_int = np.random.randint(0, 4, int(np.ceil(N/sps))) # 0 to 3
    data_degrees = data*360/4.0 + 45 # 45, 135, 225, 315 degrees
    symbols = np.cos(data_degrees*np.pi/180.0) + 1j*np.sin(data_degrees*np.pi/180.0)
    pulse_train = np.zeros(int(np.ceil(N/sps)) * sps, dtype=complex)
    pulse_train[::sps] = symbols
    t = np.arange(101) - (101-1)//2
    h = np.sinc(t/sps) * np.cos(np.pi*beta*t/sps) / (1 - (2*beta*t/sps)**2)
    signal3 = np.convolve(pulse_train, h, 'same')
    signal3 = signal3[:N] * np.exp(2j * np.pi * f_offset * np.arange(N))
    signal3 = fractional_delay(signal3, 0.3526)

    # Add noise
    noise = np.random.randn(N) + 1j*np.random.randn(N)
    samples = 0.5*signal1 + signal2 + 1.5*signal3 + 0.1*noise

    if False:
        PSD = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples)/N))**2)
        f = np.linspace(-0.5, 0.5, len(PSD))
        plt.plot(f, PSD)
        plt.xlabel('Frequency')
        plt.ylabel('PSD')
        plt.grid()
        plt.savefig('../_images/psd_of_multiple_signals.svg', bbox_inches='tight')
        plt.show()
        exit()

###########################
# OFDM (replaces samples) #
###########################

# Adapted from https://dspillustrations.com/pages/posts/misc/python-ofdm-example.html
if False:
    from scipy.signal import resample
    N = 100000 # number of samples to simulate
    num_subcarriers = 64
    cp_len = num_subcarriers // 4 # length of the cyclic prefix in symbols, in this case 25% of the starting OFDM symbol
    print("CP length in samples", cp_len*2) # remember there is 2x interpolation at the end
    print("OFDM symbol length in samples", (num_subcarriers+cp_len)*2) # remember there is 2x interpolation at the end
    num_symbols = int(np.floor(N/(num_subcarriers+cp_len))) // 2 # remember the interpolate by 2
    print("Number of OFDM symbols:", num_symbols)

    qpsk_mapping = {
        (0,0) : 1+1j,
        (0,1) : 1-1j,
        (1,0) : -1+1j,
        (1,1) : -1-1j,
    }
    bits_per_symbol = 2

    samples = np.empty(0, dtype=np.complex64)
    for _ in range(num_symbols):
        data = np.random.binomial(1, 0.5, num_subcarriers*bits_per_symbol) # 1's and 0's
        data = data.reshape((num_subcarriers, bits_per_symbol)) # group into subcarriers
        symbol_freq = np.array([qpsk_mapping[tuple(b)] for b in data]) # remember we start in the freq domain with OFDM
        symbol_time = np.fft.ifft(symbol_freq)
        symbol_time = np.hstack([symbol_time[-cp_len:], symbol_time]) # take the last CP samples and stick them at the start of the symbol
        samples = np.concatenate((samples, symbol_time)) # add symbol to samples buffer

    samples = resample(samples, len(samples)*2) # interpolate by 2x
    samples = samples[:N] # clip off the few extra samples

    # Add noise
    SNR_dB = 5
    n = np.sqrt(np.var(samples) * 10**(-SNR_dB/10) / 2) * (np.random.randn(N) + 1j*np.random.randn(N))
    samples = samples + n

############
# Resample #
############

if False:
    from scipy.signal import resample_poly
    print(len(samples))
    samples = resample_poly(samples, 10, 9) # samples, up, down
    print(len(samples))


##############
# Direct CAF #
##############

if True:
    # CAF only at the correct alpha
    alpha_of_interest = 1/sps # equates to 0.05 Hz
    #alpha_of_interest = 0.08 # INCORRECT ALPHA FOR SAKE OF PLOT
    taus = np.arange(-50, 51)
    CAF = np.zeros(len(taus), dtype=complex)
    for i in range(len(taus)):
        CAF[i] = (np.exp(1j * np.pi * alpha_of_interest * taus[i]) * # This term is to make up for the fact we're shifting by 1 sample at a time, and only on one side
                  np.sum(samples * np.conj(np.roll(samples, taus[i])) * 
                         np.exp(-2j * np.pi * alpha_of_interest * np.arange(N))))
    plt.figure(0)
    plt.plot(taus, np.real(CAF))
    plt.xlabel('Tau')
    plt.ylabel('CAF (real part)')
    #plt.savefig('../_images/caf_at_correct_alpha.svg', bbox_inches='tight')
    #plt.savefig('../_images/caf_at_incorrect_alpha.svg', bbox_inches='tight')

    # Used in SCF section
    plt.figure(1)
    SCF = np.fft.fftshift(np.fft.fft(CAF, 2048))
    f = np.linspace(-0.5, 0.5, len(SCF))
    SCF_magnitude = np.abs(SCF)
    plt.plot(f, SCF_magnitude)
    plt.xlabel('Frequency')
    plt.ylabel('SCF Magnitude')
    plt.grid()
    #plt.savefig('../_images/fft_of_caf.svg', bbox_inches='tight')
    
    #plt.show()
    #exit()

    # CAF at all alphas (this will take a while!)
    alphas = np.arange(0, 0.5, 0.005)
    CAF = np.zeros((len(alphas), len(taus)), dtype=complex)
    for j in range(len(alphas)):
        for i in range(len(taus)):
            CAF[j, i] = (np.exp(1j * np.pi * alphas[j] * taus[i]) *
                         np.sum(samples * np.conj(np.roll(samples, taus[i])) * 
                                np.exp(-2j * np.pi * alphas[j] * np.arange(N))))
    CAF_magnitudes = np.average(np.abs(CAF), axis=1) # at each alpha, calc power in the CAF
    plt.figure(2)
    plt.plot(alphas, CAF_magnitudes)
    plt.xlabel('Alpha')
    plt.ylabel('CAF Power')
    #plt.savefig('../_images/caf_avg_over_alpha.svg', bbox_inches='tight')

    plt.show()
    exit()


# Freq smoothing
if False:
    start_time = time.time()

    alphas = np.arange(0, 0.3, 0.001)
    if False: # For OFDM example
        #alphas = np.arange(0, 0.5+0.0001, 0.0001) # enable max pooling for this one
        alphas = np.arange(0, 0.02+0.0001, 0.0001)
    Nw = 256 # window length
    N = len(samples) # signal length
    window = np.hanning(Nw)

    X = np.fft.fftshift(np.fft.fft(samples)) # FFT of entire signal
    
    num_freqs = int(np.ceil(N/Nw)) # freq resolution after decimation
    SCF = np.zeros((len(alphas), num_freqs), dtype=complex)
    for i in range(len(alphas)):
        shift = int(alphas[i] * N/2)
        SCF_slice = np.roll(X, -shift) * np.conj(np.roll(X, shift))
        #SCF[i, :shift] = 0 # do we even need this one?
        SCF[i, :] = np.convolve(SCF_slice, window, mode='same')[::Nw] # apply window and decimate by Nw
    SCF = np.abs(SCF)

    # null out alpha= 0, 1, -1 which is just the PSD of the signal, it throws off the dynamic range
    SCF[np.argmin(np.abs(alphas)), :] = 0 
    SCF[np.argmin(np.abs(alphas - 1)), :] = 0
    SCF[np.argmin(np.abs(alphas - (-1))), :] = 0

    print("Time taken:", time.time() - start_time)

    print("SCF shape", SCF.shape)
    # Max pooling in cyclic domain
    if False:
        import skimage.measure
        SCF = skimage.measure.block_reduce(SCF, block_size=(16, 1), func=np.max) # type: ignore
        print("Shape of SCF:", SCF.shape)

    extent = (-0.5, 0.5, float(np.max(alphas)), float(np.min(alphas)))
    plt.imshow(SCF, aspect='auto', extent=extent, vmax=np.max(SCF)/2)
    plt.xlabel('Frequency [Normalized Hz]')
    plt.ylabel('Cyclic Frequency [Normalized Hz]')
    #plt.savefig('../_images/scf_freq_smoothing.svg', bbox_inches='tight')
    #plt.savefig('../_images/scf_freq_smoothing_ofdm.svg', bbox_inches='tight') # for OFDM example
    #plt.savefig('../_images/scf_freq_smoothing_ofdm_zoomed_in.svg', bbox_inches='tight') # for OFDM example 2
    #plt.savefig('../_images/scf_freq_smoothing_pulse_shaped_bpsk.svg', bbox_inches='tight')
    #plt.savefig('../_images/scf_freq_smoothing_pulse_shaped_bpsk2.svg', bbox_inches='tight')
    #plt.savefig('../_images/scf_freq_smoothing_pulse_shaped_bpsk3.svg', bbox_inches='tight')
    #plt.savefig('../_images/scf_freq_smoothing_pulse_multiple_signals.svg', bbox_inches='tight') # I ADDED ANNOTATIONS TO THIS ONE!!!!
    plt.show()
    exit()


# CONJUGATE VERSION Freq smoothing
if False:
    # For multiple signals
    #samples = signal1 + signal2 + signal3 + 0.1*noise
    #samples = samples[0:100000]

    alphas = np.arange(-1, 1, 0.0025) # Conj SCF should be calculated from -1 to +1
    Nw = 256 # window length
    N = len(samples) # signal length
    window = np.hanning(Nw)

    X = np.fft.fftshift(np.fft.fft(samples)) # FFT of entire signal
    
    num_freqs = int(np.ceil(N/Nw)) # freq resolution after decimation
    SCF = np.zeros((len(alphas), num_freqs), dtype=complex)
    for i in range(len(alphas)):
        shift = int(np.round(alphas[i] * N/2))
        SCF_slice = np.roll(X, -shift) * np.flip(np.roll(X, -shift - 1)) # THIS LINE IS THE ONLY DIFFERENCE
        SCF[i, :] = np.convolve(SCF_slice, window, mode='same')[::Nw]
    SCF = np.abs(SCF)

    print("SCF shape", SCF.shape)
    # Max pooling in cyclic domain
    if False:
        import skimage.measure
        SCF = skimage.measure.block_reduce(SCF, block_size=(16, 1), func=np.max) # type: ignore
        print("Shape of SCF:", SCF.shape)

    extent = (-0.5, 0.5, float(np.min(alphas)), float(np.max(alphas)))
    plt.imshow(SCF, aspect='auto', extent=extent, vmax=np.max(SCF)/2, origin='lower')
    #plt.imshow(SCF, aspect='auto', extent=extent, vmax=1.5e8, origin='lower')
    plt.xlabel('Frequency [Normalized Hz]')
    plt.ylabel('Cyclic Frequency [Normalized Hz]')
    plt.colorbar()
    #plt.savefig('../_images/scf_conj_rect_bpsk.svg', bbox_inches='tight')
    #plt.savefig('../_images/scf_conj_pulseshaped_bpsk.svg', bbox_inches='tight')
    #plt.savefig('../_images/scf_conj_rect_qpsk.svg', bbox_inches='tight')
    #plt.savefig('../_images/scf_conj_rect_qpsk_scaled.svg', bbox_inches='tight')
    #plt.savefig('../_images/scf_conj_multiple_signals.svg', bbox_inches='tight') # THIS ONE WAS EDITING IN INKSCAPE!
    plt.show()
    exit()

# Freq smoothing Coherence Function
if True:
    alphas = np.arange(0, 0.3, 0.001)
    Nw = 256 # window length
    N = len(samples) # signal length
    window = np.hanning(Nw)

    X = np.fft.fftshift(np.fft.fft(samples)) # FFT of entire signal
    
    num_freqs = int(np.ceil(N/Nw)) # freq resolution after decimation
    SCF = np.zeros((len(alphas), num_freqs), dtype=complex)
    COH = np.zeros((len(alphas), num_freqs), dtype=complex)
    for i in range(len(alphas)):
        shift = int(alphas[i] * N/2)
        SCF_slice = np.roll(X, -shift) * np.conj(np.roll(X, shift))
        SCF[i, :] = np.convolve(SCF_slice, window, mode='same')[::Nw] # apply window and decimate by Nw
        COH_slice = SCF_slice / np.sqrt(np.roll(X, -shift) * np.roll(X, shift))
        COH[i, :] = np.convolve(COH_slice, window, mode='same')[::Nw] # apply the same windowing + decimation
    SCF = np.abs(SCF)
    COH = np.abs(COH)

    # null out alpha=0 for both so that it doesnt hurt our dynamic range and ability to see the non-zero alphas
    SCF[np.argmin(np.abs(alphas)), :] = 0
    COH[np.argmin(np.abs(alphas)), :] = 0

    print("SCF shape", SCF.shape)

    extent = (-0.5, 0.5, float(np.max(alphas)), float(np.min(alphas)))
    fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(10, 5))
    ax0.imshow(SCF, aspect='auto', extent=extent, vmax=np.max(SCF)/2)
    ax0.set_xlabel('Frequency [Normalized Hz]')
    ax0.set_ylabel('Cyclic Frequency [Normalized Hz]')
    ax0.set_title('Regular SCF')

    ax1.imshow(COH, aspect='auto', extent=extent, vmax=np.max(COH)/2)
    ax1.set_xlabel('Frequency [Normalized Hz]')
    ax1.set_title('Spectral Coherence Function (COH)')

    #plt.savefig('../_images/scf_coherence.svg', bbox_inches='tight')
    #plt.savefig('../_images/scf_coherence_pulse_shaped.svg', bbox_inches='tight')
    plt.show()
    exit()

# Time smoothing, based on https://www.mathworks.com/matlabcentral/fileexchange/48909-cyclic-spectral-analysis
if False:
    start_time = time.time()

    alphas = np.arange(0, 0.3, 0.001)
    Nw = 256 # window length
    N = len(samples) # signal length
    Noverlap = int(2/3*Nw) # block overlap
    num_windows = int((N - Noverlap) / (Nw - Noverlap)) # Number of windows
    window = np.hanning(Nw)

    SCF = np.zeros((len(alphas), Nw), dtype=complex)
    for ii in range(len(alphas)): # Loop over cyclic frequencies
        neg = samples * np.exp(-1j*np.pi*alphas[ii]*np.arange(N))
        pos = samples * np.exp( 1j*np.pi*alphas[ii]*np.arange(N))
        for i in range(num_windows):
            pos_slice = window * pos[i*(Nw-Noverlap):i*(Nw-Noverlap)+Nw]
            neg_slice = window * neg[i*(Nw-Noverlap):i*(Nw-Noverlap)+Nw]
            SCF[ii, :] += np.fft.fft(neg_slice) * np.conj(np.fft.fft(pos_slice)) # Cross Cyclic Power Spectrum
    SCF = np.fft.fftshift(SCF, axes=1) # shift the RF freq axis
    SCF = np.abs(SCF)
    SCF[0, :] = 0 # null out alpha=0 which is just the PSD of the signal, it throws off the dynamic range

    print("Time taken:", time.time() - start_time)

    extent = (-0.5, 0.5, float(np.max(alphas)), float(np.min(alphas)))
    plt.imshow(SCF, aspect='auto', extent=extent, vmax=np.max(SCF)/2)
    plt.xlabel('Frequency [Normalized Hz]')
    plt.ylabel('Cyclic Frequency [Normalized Hz]')
    #plt.savefig('../_images/scf_time_smoothing.svg', bbox_inches='tight')
    plt.show()
    exit()




# FAM, based on EL Da Costa https://apps.dtic.mil/sti/pdfs/ADA311555.pdf
# note, its not processing all samples in x, it's just processing the first N
# https://github.com/avian2/spectrum-sensing-methods/blob/master/sensing/utils.py
# https://github.com/phwl/cyclostationary/blob/master/analysis/src/cyclostationary.py
if False:
    N = 2**14
    x = samples[0:N]
    Np = 512 # Number of input channels, should be power of 2
    L = Np//4 # Offset between points in the same column at consecutive rows in the same channelization matrix. It should be chosen to be less than or equal to Np/4
    num_windows = (len(x) - Np) // L + 1
    Pe = int(np.floor(int(np.log(num_windows)/np.log(2))))
    P = 2**Pe
    N = L*P
    print("P:", P, " N:", N, " Pe:", Pe, " Np:", Np, " L:", L)

    # channelization
    xs = np.zeros((num_windows, Np), dtype=complex)
    for i in range(num_windows):
        xs[i,:] = x[i*L:i*L+Np]
    xs2 = xs[0:P,:]

    # windowing
    xw = xs2 * np.tile(np.hanning(Np), (P,1))

    # first FFT
    XF1 = np.fft.fftshift(np.fft.fft(xw))

    # freq shift down
    f = np.arange(Np)/float(Np) - 0.5
    f = np.tile(f, (P, 1))
    t = np.arange(P)*L
    t = t.reshape(-1,1) # make it a column vector
    t = np.tile(t, (1, Np))
    XD = XF1 * np.exp(-2j*np.pi*f*t)

    # main calcs
    SCF = np.zeros((2*N, Np))
    Mp = N//Np//2
    for k in range(Np):
        for l in range(Np):
            XF2 = np.fft.fftshift(np.fft.fft(XD[:,k]*np.conj(XD[:,l]))) # second FFT
            i = (k + l) // 2
            a = int(((k - l) / Np + 1) * N)
            SCF[a-Mp:a+Mp, i] = np.abs(XF2[(P//2-Mp):(P//2+Mp)])**2

    # Get rid of negative alphas
    SCF = SCF[N:,:]

    # Max pooling in cyclic domain
    print("Shape of SCF:", SCF.shape)
    if False:
        import skimage.measure
        SCF = skimage.measure.block_reduce(SCF, block_size=(16, 1), func=np.max) # type: ignore
        print("Shape of SCF:", SCF.shape)

    SCF[0, :] = 0 # null out alpha=0 which is just the PSD of the signal, it throws off the dynamic range

   
    plt.figure(0)
    extent = (-0.5, 0.5, 1, 0)
    plt.imshow(SCF, aspect='auto', extent=extent, vmax=np.max(SCF)/4)
    plt.xlabel('Frequency [Normalized Hz]')
    plt.ylabel('Cyclic Frequency [Normalized Hz]')
    plt.savefig('../_images/scf_fam.svg', bbox_inches='tight')

    # Zoom in
    plt.figure(1)
    extent = (0, 0.5, 1/8, 0)
    plt.imshow(SCF[0:N//8,Np//2:], aspect='auto', extent=extent, vmax=np.max(SCF)/4)
    plt.xlabel('Frequency [Normalized Hz]')
    plt.ylabel('Cyclic Frequency [Normalized Hz]')
    plt.savefig('../_images/scf_fam_zoomedin.svg', bbox_inches='tight')

    plt.figure(2, figsize=(10, 5))
    plt.plot(np.linspace(0, 1, SCF.shape[0]), np.average(SCF, axis=1))
    plt.grid()
    plt.xlabel('Cyclic Frequency [Normalized Hz]')
    plt.ylabel('SCF Power')
    plt.savefig('../_images/scf_fam_1d.svg', bbox_inches='tight')

    plt.show()
    exit()

''' My first attempt
    x = samples
    fs = 1
    df = 0.005 # freq res. Make sure that DF is much bigger than DALPHA in order to have a reliable estimate.
    dalpha = 0.0005 # cyclic res

    Np = int(2**(np.ceil(np.log2(fs/df)))) # Number of input channels, defined by the desired frequency # Np=fs/df, where fs is the original data sampling rate.  It must be a power of 2 to avoid truncation or zero-padding in the FFT routines；
    print("Np:", Np)

    L = Np//4 # Offset between points in the same column at consecutive rows in the same channelization matrix. It should be chosen to be less than or equal to Np/4;
    print("L:", L)

    P = int(2**(np.ceil(np.log2(fs/dalpha/L)))) # Number of rows formed in the channelization matrix, defined by the desired cyclic frequency resolution (dalpha) as follows:
    print("P:", P) 

    N = P * L; # Total number of points in the input data.
    print("N:", N)

    # Slice up the data using overlap defined by L
    X = np.zeros((Np, P), dtype=complex)
    for k in range(P):
        X[:, k] = x[k*L:k*L+Np]
    print("shape of X:", X.shape)

    # Apply window
    X = np.matmul(np.diagflat(np.hamming(Np)), X)

    # First FFT
    XF1 = np.fft.fftshift(np.fft.fft(X))
    upper = XF1[:, P//2:P]
    lower = XF1[:, 0:P//2]
    XF1 = np.concatenate((upper, lower), axis=1)
    
    # Downconversion
    exp_mat = np.zeros((Np, P), dtype=complex)
    for k in range(Np//-2, Np//2):
        for m in range(P):
            exp_mat[k + Np//2, m] = np.exp(-2j*np.pi*k*m*L/Np)
    XD = XF1 * exp_mat # elementwise
    
    XD = np.conj(XD.conj().T) # not sure why this is needed
    
    # Multiplication
    XM = np.zeros((P, Np**2), dtype=complex)
    for k in range(Np):
        for l in range(Np):
            XM[:, (k-1) * Np + l] = XD[:, k] * np.conj(XD[:, l]) # elementwise. 99% sure -1 needs to be there

    # Second FFT
    XF2 = np.fft.fftshift(np.fft.fft(XM))
    upper = XF2[:, Np**2//2:Np**2]
    lower = XF2[:, 0:Np**2//2]
    XF2 = np.concatenate((upper, lower), axis=1)
    print(XF2.shape)

    XF2 = XF2[P//4:3*P//4, :]
    print(XF2.shape)

    if True:
        SCF = np.zeros((2*N, Np)) # first dim is cylcic freq, second is freq
        for k1 in range(P//2):
            for k2 in range(Np**2):
                if np.remainder(k2, Np) == 0:
                    l = Np//2
                else:
                    l = np.remainder(k2,Np) - Np//2
                k = int(np.ceil(k2/Np)) - Np//2
                p = k1 - P/4
                alpha = (k-l)/Np + p/L/P
                #print("alpha:", alpha)
                f = (k+l)/2/Np
                if alpha < -1 or alpha > 1:
                    k2 += 1
                elif f < -.5 or f > 0.5:
                    k2 += 1
                else:
                    kk = int(Np * (f + 0.5))
                    ll = int(N*(alpha + 1))
                    SCF[ll, kk] = np.abs(XF2[k1, k2])
    else:
        SCF = np.abs(XF2)

    # Max pooling
    print("Shape of SCF:", SCF.shape)
    import skimage.measure
    SCF = skimage.measure.block_reduce(SCF, (20,1), np.max)
    print("Shape of SCF:", SCF.shape)

    alphas = np.arange(-1, 1, 1/N)

    #SCF[0, :] = 0 # null out alpha=0 which is just the PSD of the signal, it throws off the dynamic range

    extent = (-0.5, 0.5, float(np.max(alphas)), float(np.min(alphas)))
    plt.imshow(SCF, aspect='auto', extent=extent, vmax=np.max(SCF)/2)
    plt.xlabel('Frequency [Normalized Hz]')
    plt.ylabel('Cyclic Frequency [Normalized Hz]')
    #plt.savefig('../_images/scf_fam.svg', bbox_inches='tight')
    plt.show()
    exit()
    '''

#######################
# Non-CSP Alternative #
#######################

# Run using the multiple signals scenario
if False:
    samples_list = [signal1 + 0.1*noise,
                    signal2 + 0.1*noise,
                    signal3 + 0.1*noise,
                    signal1 + signal2 + signal3 + 0.1*noise]
    labels = ['Signal 1', 'Signal 2', 'Signal 3', 'Combined']
    fig, axs = plt.subplots(4, 1, figsize=(5, 8))
    for i in range(4):
        samples = samples_list[i]
        samples_mag = np.abs(samples) # note that samples * np.conj(samples) is pretty much the same
        magnitude_metric = np.abs(np.fft.fft(samples_mag))
        magnitude_metric = magnitude_metric[:len(magnitude_metric)//2] # only need half because input is real
        magnitude_metric[0] = 0 # null out the DC component

        # Plot
        f = np.linspace(0, 0.5, len(samples)//2)
        axs[i].plot(f, magnitude_metric)
        axs[i].grid()
        axs[i].set_xlabel('Frequency [Hz]')
        axs[i].set_ylabel(labels[i])
    plt.savefig('../_images/non_csp_metric.svg', bbox_inches='tight')
    plt.show()

    ''' ALSO INCLUDED THE RF ESTIMATION METHODS
    samples_squared = samples**2
    squared_metric = np.abs(np.fft.fftshift(np.fft.fft(samples_squared)))/len(samples)
    squared_metric[len(squared_metric)//2] = 0 # null out the DC component

    samples_quartic = samples**4
    quartic_metric = np.abs(np.fft.fftshift(np.fft.fft(samples_quartic)))/len(samples)
    quartic_metric[len(quartic_metric)//2] = 0 # null out the DC component

    fig, [ax0, ax1, ax2] = plt.subplots(1, 3, figsize=(10, 3))

    f = np.linspace(0, 0.5, len(samples)//2)
    ax0.plot(f, magnitude_metric)
    ax0.grid()
    ax0.set_xlabel('Frequency [Hz]')
    ax0.set_title('Magnitude Metric')

    f_squared = np.linspace(-0.25, 0.25, len(samples))
    ax1.plot(f_squared, squared_metric)
    ax1.grid()
    ax1.set(ylim=(0, 1.1))
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_title('Squared Metric')

    f_quartic = np.linspace(-0.125, 0.125, len(samples))
    ax2.plot(f_quartic, quartic_metric)
    ax2.grid()
    ax2.set(ylim=(0, 1.1))
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_title('Quartic Metric')
    
    #plt.savefig('../_images/non_csp_metric1.svg', bbox_inches='tight')
    #plt.savefig('../_images/non_csp_metric2.svg', bbox_inches='tight')
    #plt.savefig('../_images/non_csp_metric3.svg', bbox_inches='tight')
    #plt.savefig('../_images/non_csp_metricALL.svg', bbox_inches='tight')
    plt.show()
    '''
    exit()


########
# CONJ #
########

if True:
    sps = 20
    f_offset = 0.2
    beta = 0.35
    # BPSK
    symbols = np.random.randint(0, 2, int(np.ceil(N/sps))) * 2 - 1
    pulse_train = np.zeros(int(np.ceil(N/sps)) * sps)
    pulse_train[::sps] = symbols
    t = np.arange(101) - (101-1)//2
    h = np.sinc(t/sps) * np.cos(np.pi*beta*t/sps) / (1 - (2*beta*t/sps)**2)
    bpsk = np.convolve(pulse_train, h, 'same')
    bpsk = bpsk[:N] * np.exp(2j * np.pi * f_offset * np.arange(N))
    bpsk = bpsk + 0.1*(np.random.randn(N) + 1j*np.random.randn(N))
    # QPSK
    data = x_int = np.random.randint(0, 4, int(np.ceil(N/sps))) # 0 to 3
    data_degrees = data*360/4.0 + 45 # 45, 135, 225, 315 degrees
    symbols = np.cos(data_degrees*np.pi/180.0) + 1j*np.sin(data_degrees*np.pi/180.0)
    pulse_train = np.zeros(int(np.ceil(N/sps)) * sps, dtype=complex)
    pulse_train[::sps] = symbols
    t = np.arange(101) - (101-1)//2
    h = np.sinc(t/sps) * np.cos(np.pi*beta*t/sps) / (1 - (2*beta*t/sps)**2)
    qpsk = np.convolve(pulse_train, h, 'same')
    qpsk = qpsk[:N] * np.exp(2j * np.pi * f_offset * np.arange(N))
    qpsk = qpsk + 0.1*(np.random.randn(N) + 1j*np.random.randn(N))

    # CAF only at the correct alpha
    correct_alpha = 1/sps # equates to 0.05 Hz
    taus = np.arange(-100, 101) # -100 to +100 in steps of 1
    CAF_bpsk = np.zeros(len(taus), dtype=complex)
    CAF_qpsk = np.zeros(len(taus), dtype=complex)
    for i in range(len(taus)):
        # normal
        #CAF_bpsk[i] = np.sum(bpsk * np.conj(np.roll(bpsk, taus[i])) * np.exp(-2j * np.pi * correct_alpha * np.arange(N)))
        #CAF_qpsk[i] = np.sum(qpsk * np.conj(np.roll(qpsk, taus[i])) * np.exp(-2j * np.pi * correct_alpha * np.arange(N)))
        # conj
        CAF_bpsk[i] = np.sum(bpsk * np.roll(bpsk, taus[i]) * np.exp(-2j * np.pi * correct_alpha * np.arange(N)))
        CAF_qpsk[i] = np.sum(qpsk * np.roll(qpsk, taus[i]) * np.exp(-2j * np.pi * correct_alpha * np.arange(N)))

    fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(9, 3))
    # increase width between subplots
    plt.subplots_adjust(wspace=0.3)
    ax0.plot(taus, np.real(CAF_bpsk))
    ax0.plot(taus, np.imag(CAF_bpsk))
    ax0.set_xlabel('Tau')
    ax0.set_ylabel('CAF of BPSK')
    ax1.plot(taus, np.real(CAF_qpsk))
    ax1.plot(taus, np.imag(CAF_qpsk))
    ax1.set_xlabel('Tau')
    ax1.set_ylabel('CAF of QPSK')
    ax1.legend(['Real', 'Imaginary'])
    #plt.savefig('../_images/caf_at_correct_alpha_conj.svg', bbox_inches='tight')
    plt.show()
    exit()


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
    num_taps = 101  # for our RC filter
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


