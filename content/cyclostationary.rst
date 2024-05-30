.. _freq-domain-chapter:

##################################
Cyclostationary Signal Processing
##################################

Co-authored by `Sam Brown <https://www.linkedin.com/in/samuel-brown-vt/>`_

1-2 sentence highlight (incl any words we want SEO to pick up, eg CSP)

****************
Introduction
****************

Cyclostationary signal processing (CSP) is a set of techniques for exploiting the cyclostationary property found in many real-world communication signals. These are signals such as modulated signals like AM/FM/TV broadcast, cellular, and WiFi as well as radar signals, and other signals that exhibit periodicity in their statistics. A large swath of traditional signal processing techniques are based on the assumption that the signal is stationary, i.e., the statistics of the signal like the mean, variance and higher-order moments do not change over time. However, many real-world signals are cyclostationary, i.e., the statistics of the signal change *periodically* over time. CSP techniques exploit this cyclostationary property to improve the performance of signal processing algorithms. For example, CSP techniques can be used to detect the presence of signals in noise, perform modulation recognition, and separate signals that are overlapping in time and frequency.

************************************************
The Cyclic Autocorrelation Function (CAF)
************************************************

A good place to start understanding CSP is the cyclic autocorrelation function (CAF). The CAF is an extension of the traditional autocorrelation function to cyclostationary signals. As a refresher, the autocorrelation of a random process is the expected product of two time instants of the process and is defined as: :math:`R_x(t_1, t_2) = E[x(t_1)x^*(t_2)]`. Intuitively, it represents the degree to which a signal exhibits repetitive behavior. This can alternatively be written as :math:`R_x(t, \tau) = E[x(t+\tau/2)x^*(t-\tau/2)]` where :math:`\tau` is the delay between the two signals and :math:`t` is the midpoint. Some signals exhibit the property where their autocorrelation does not depend upon the midpoint :math:`t` and only on the delay :math:`\tau`. These signals are stationary of order 2 or just stationary. For cyclostationary signals, however, the midpoint does matter, meaning that the autocorrelation depends on both the delay and the midpoint, a function of two lag parameters.

Cyclostationary signals possess a periodic or almost periodic autocorrelation, and the CAF is the set of Fourier series coefficients that describe this periodicity. In other words, the CAF is the amplitude and phase of the harmonics present in a signal's autocorrelation, giving it the following form: :math:`R_x(\tau, \alpha) = \int_{-\infty}^{\infty} x(t)x^*(t-\tau)e^{-j2\pi \alpha t}dt`. It can be seen that the CAF is a function of two variables, the delay :math:`\tau` and the cycke frequency :math:`\alpha`.

CAF in Python

Example CAF output for rect BPSK

************************************************
The Spectral Correlation Function (SCF)
************************************************

* Discuss the Cyclic Wiener Relationship (says that the CAF and the SCF are Fourier transforms of each other)
* Discuss generalization of the power spectral density
* Frequency smoothing and time smoothing methods
* Include some illustrations of the SCF for simple cyclostationary signals like BPSK and QPSK with rect and SRRC pulse shapes

SCF in Python (FFT of prev CAF code)

Example SCF output for rect BPSK

Interactive Javascript App

***************************
Time Smoothing Method (TSM)
***************************

talk about the importance of the window length because it determines the resolution

********************************
Frequency Smoothing Method (FSM)
********************************

talk about the importance of the window length because it determines the resolution

.. code-block:: python

 alphas = np.arange(0.05, 0.5, 0.005)
 Nw = 256 # window length
 N = len(samples) # signal length
 Noverlap = int(2/3*Nw) # block overlap
 num_windows = int((N - Noverlap) / (Nw - Noverlap)) # Number of windows
 window = np.hanning(Nw)
 
 S = np.zeros((Nw, len(alphas)), dtype=complex)
 for ii in range(len(alphas)): # Loop over cyclic frequencies
     neg = samples * np.exp(-1j*np.pi*alphas[ii]*np.arange(N))
     pos = samples * np.exp( 1j*np.pi*alphas[ii]*np.arange(N))
     for i in range(num_windows):
         pos_slice = window * pos[i*(Nw-Noverlap):i*(Nw-Noverlap)+Nw]
         neg_slice = window * neg[i*(Nw-Noverlap):i*(Nw-Noverlap)+Nw]
         S[:, ii] += np.fft.fft(neg_slice) * np.conj(np.fft.fft(pos_slice)) # Cross Cyclic Power Spectrum
 S = np.abs(S)
 
 plt.imshow(S, aspect='auto', extent=(float(np.min(alphas)), float(np.max(alphas)), fs, 0.0))
 plt.xlabel('Cyclic Frequency [Hz]')
 plt.ylabel('Frequency [Normalized Hz]')
 plt.show()


********************************
Pulse-Shaped BPSK
********************************


********************************
SNR and Number of Symbols
********************************


********************************
QPSK and Other Signals
********************************

********************************
Multiple Overlapping Signals
********************************

********************************
Spectral Coherence Function
********************************

********************************
Conjugate Versions
********************************

***********************************************
Strip Spectral Correlation Analyzer (SSCA)
***********************************************

********************************
FFT Accumulation Method (FAM)
********************************


********************************
Python Example TO REMOVE
********************************

The following example demonstrates how to compute the SCF of a cyclostationary signal using the `cspy` package. The example generates a random cyclostationary signal, computes the SCF using the `scf` function, and plots the SCF using the `plot_scf` function.


.. code-block:: python

 ##### Generate the Spectral Correlation Function #####
 
 a_res = 0.005
 a_vals = np.arange(-1, 1, a_res)
 smoothing_len = 2048
 window = np.hanning(smoothing_len)
 
 X = np.fft.fft(signal)
 X = np.fft.fftshift(X)
 
 SCF = np.zeros((len(a_vals), num_samples))
 SCF_conj = np.zeros((len(a_vals), num_samples))
 
 for i, a in enumerate(a_vals):
     SCF[i, :] = np.roll(X, -int(np.floor(a*num_samples/2)))*np.conj(np.roll(X, int(np.floor(a*num_samples/2))))
     SCF[i, :abs(round(a*num_samples/2))] = 0
     SCF[i, -abs(round(a*num_samples/2))-1:] = 0
     SCF[i, :] = np.convolve(SCF[i, :], window, mode='same')
     
     SCF_conj[i, :] = np.roll(X, int(np.floor(a*num_samples/2))-1)*np.flip(np.roll(X, int(np.floor(a*num_samples/2))))
     SCF_conj[i, :abs(round(a*num_samples/2))] = 0
     SCF_conj[i, -abs(round(a*num_samples/2))-1:] = 0
     SCF_conj[i, :] = np.convolve(SCF_conj[i, :], window, mode='same')
 
 ##### Plot the Spectral Correlation Function #####
 
 dym_range_dB = 20
 max_val = np.max(SCF[np.where(a_vals > a_res),:])
 linear_scale = True
 
 plt.set_cmap("viridis")
 
 plt.figure(figsize=(10, 5))
 plt.subplot(1, 2, 1)
 if linear_scale:
     plt.imshow(np.abs(SCF), aspect='auto', extent=[-0.5, 0.5, -1, 1],
            vmax=max_val)
 else:
     plt.imshow(10*np.log10(np.abs(SCF)), aspect='auto', extent=[-0.5, 0.5, -1, 1],
             vmax=10*np.log10(max_val), vmin=10*np.log10(max_val)-dym_range_dB)
 
 plt.ylim([0, 0.5])
 plt.xlabel("Normalized Frequency")
 plt.ylabel("Cycle Frequency")
 plt.colorbar()
 plt.title("Non-Conjugate SCF")
 
 max_val = np.max(SCF_conj)
 
 plt.subplot(1, 2, 2)
 if linear_scale:
     plt.imshow(np.abs(SCF_conj), aspect='auto', extent=[-0.5, 0.5, -1, 1],
            vmax=max_val)
 else:
     plt.imshow(10*np.log10(np.abs(SCF_conj)), aspect='auto', extent=[-0.5, 0.5, -1, 1], 
             vmax=10*np.log10(max_val), vmin=10*np.log10(max_val)-dym_range_dB)
 plt.xlabel("Normalized Frequency")
 plt.ylabel("Cycle Frequency")
 plt.ylim([-0.5, 0.5])
 plt.colorbar()
 plt.title("Conjugate SCF")
 plt.tight_layout()
 
 plt.show()

****************
Further Reading
****************

https://cyclostationary.blog/
