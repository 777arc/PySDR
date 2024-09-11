if False:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import signal

    num_taps = 51
    cut_off = 3000  # Hz
    sample_rate = 32000  # Hz

    # create our low pass filter
    h = signal.firwin(num_taps, cut_off, fs=sample_rate)

    # plot the impulse response
    plt.plot(h, '.-')
    plt.show()

    # plot the frequency response
    H = np.abs(np.fft.fft(h, 1024))  # take the 1024-point FFT and magnitude
    H = np.fft.fftshift(H)  # make 0 Hz in the center
    w = np.linspace(-sample_rate / 2, sample_rate / 2, len(H))  # x axis
    plt.plot(w, H, '.-')
    plt.show()

    # (h was found using the first code snippet)

    # Shift the filter in frequency by multiplying by exp(j*2*pi*f0*t)
    f0 = 10e3  # amount we will shift
    Ts = 1.0 / sample_rate  # sample period
    t = np.arange(0.0, Ts * len(h), Ts)  # time vector. args are (start, stop, step)
    exponential = np.exp(2j * np.pi * f0 * t)  # this is essentially a complex sine wave

    h_band_pass = h * exponential  # do the shift

    # plot impulse response
    plt.subplot(121)
    plt.plot(np.real(h_band_pass), '.-')
    plt.plot(np.imag(h_band_pass), '.-')
    plt.legend(['real', 'imag'], loc=1)

    # plot the frequency response
    H = np.abs(np.fft.fft(h_band_pass, 1024))  # take the 1024-point FFT and magnitude
    H = np.fft.fftshift(H)  # make 0 Hz in the center
    w = np.linspace(-sample_rate / 2, sample_rate / 2, len(H))  # x axis
    plt.subplot(122)
    plt.plot(w, H, '.-')
    plt.xlabel('Frequency [Hz]')
    plt.show()

# comparing different convolution methods
if False:
    import numpy as np
    from scipy.signal import firwin2, convolve, fftconvolve, lfilter

    # Create a test signal, we'll use Gaussian noise
    sample_rate = 1e6 # Hz
    N = 1000 # samples to simulate
    x = np.random.randn(N) + 1j * np.random.randn(N)

    # Create an FIR filter, same one as 2nd example above
    freqs = [0, 100e3, 110e3, 190e3, 200e3, 300e3, 310e3, 500e3]
    gains = [1, 1,     0,     0,     0.5,   0.5,   0,     0]
    h2 = firwin2(101, freqs, gains, fs=sample_rate)

    # Apply filter using the four different methods
    x_numpy = np.convolve(h2, x)
    x_scipy = convolve(h2, x) # scipys convolve
    x_fft_convolve = fftconvolve(h2, x)
    x_lfilter = lfilter(h2, 1, x) # 2nd arg is always 1 for FIR filters

    # Prove they are all giving the same output
    print(x_numpy[0:2])
    print(x_scipy[0:2])
    print(x_fft_convolve[0:2])
    print(x_lfilter[0:2])

# Make plot comparing methods using different taps
if False:
    import numpy as np
    from scipy.signal import firwin2, convolve, fftconvolve, lfilter
    import matplotlib.pyplot as plt
    import time
    #taps_sizes = np.logspace(1, 3, num=100) # 10**x is the start/stop
    taps_sizes = np.linspace(10, 10000, num=100)
    N = 100000 # signal length
    #N = 1000 # signal length
    x = np.random.randn(N) + 1j * np.random.randn(N) # complex signal
    results = np.zeros((4, len(taps_sizes)))
    for method_i in range(4):
        for i in range(len(taps_sizes)):
            h = np.random.randn(int(taps_sizes[i])) # real taps
            num_trials = int(10000 / taps_sizes[i])
            #num_trials = int(1000000 / taps_sizes[i])
            start_t = time.time()
            if method_i == 0:
                for ii in range(num_trials):
                    _ = np.convolve(h, x)
            elif method_i == 1:
                for ii in range(num_trials):
                    _ = convolve(h, x)
            elif method_i == 2:
                for ii in range(num_trials):
                    _ = fftconvolve(h, x)
            elif method_i == 3:
                for ii in range(num_trials):
                    _ = lfilter(h, 1, x)
            results[method_i, i] = (time.time() - start_t)/num_trials*1e3 # in ms
    
    # Lots of folks use semilogx, but its nice to see the linear aspect
    plt.plot(taps_sizes, results[0,:], '.-')
    plt.plot(taps_sizes, results[1,:], '.-')
    plt.plot(taps_sizes, results[2,:], '.-')
    plt.plot(taps_sizes, results[3,:], '.-')
    plt.title('Input Signal Length: ' + str(N) + ' samples')
    plt.xlabel('Number of taps')
    plt.ylabel('Time per call (ms)')
    plt.legend(['np.convolve', 'scipy.signal.convolve', 'scipy.signal.fftconvolve', 'scipy.signal.lfilter'])
    plt.savefig('../_images/convolve_comparison_'+str(N)+'.svg', bbox_inches='tight')
    plt.show()

if True:
    import numpy as np
    from scipy.signal import firwin2, fftconvolve
    import matplotlib.pyplot as plt
    import time
    sample_rate = 1e6
    freqs = [0, 100e3, 110e3, 190e3, 200e3, 300e3, 310e3, 500e3]
    gains = [1, 1,     0,     0,     0.5,   0.5,   0,     0]
    h2 = firwin2(101, freqs, gains, fs=sample_rate)

    # COPY THE CODE STARTING HERE
   
    # (h2 was created above)

    # Simulate signal comprising of Gaussian noise
    N = 100000 # signal length
    x = np.random.randn(N) + 1j * np.random.randn(N) # complex signal

    # Save PSD of the input signal
    PSD_input = 10*np.log10(np.fft.fftshift(np.abs(np.fft.fft(x))**2)/len(x))

    # Apply filter
    x = fftconvolve(x, h2, 'same')

    # Look at PSD of the output signal
    PSD_output = 10*np.log10(np.fft.fftshift(np.abs(np.fft.fft(x))**2)/len(x))
    f = np.linspace(-sample_rate/2/1e6, sample_rate/2/1e6, len(PSD_output))
    plt.plot(f, PSD_input, alpha=0.8)
    plt.plot(f, PSD_output, alpha=0.8)
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('PSD [dB]')
    plt.axis([sample_rate/-2/1e6, sample_rate/2/1e6, -40, 20])
    plt.legend(['Input', 'Output'], loc=1)
    plt.grid()
    plt.savefig('../_images/fftconvolve.svg', bbox_inches='tight')
    plt.show()