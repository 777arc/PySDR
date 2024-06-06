.. _freq-domain-chapter:

##########################
Cyclostationary Processing
##########################

.. raw:: html

 <span style="display: table; margin: 0 auto; font-size: 20px;">Co-authored by <a href="https://www.linkedin.com/in/samuel-brown-vt">Sam Brown</a></span>

In this chapter we introduce Cyclostationary signal processing (CSP), a relatively niche area of RF signal processing that is used to analyze or detect (often in very low SNR!) signals that exhibit cyclostationary properties, such as most modern digitial modulation schemes.  We cover the Cyclic Autocorrelation Function (CAF), Spectral Correlation Function (SCF), Spectral Coherence Function, conjugate versions of these functions, and efficient implementations using Python examples.

****************
Introduction
****************

Cyclostationary signal processing (a.k.a., CSP or simply cyclostationary processing) is a set of techniques for exploiting the cyclostationary property found in many real-world communication signals. These are signals such as modulated signals like AM/FM/TV broadcast, cellular, and WiFi as well as radar signals, and other signals that exhibit periodicity in their statistics. A large swath of traditional signal processing techniques are based on the assumption that the signal is stationary, i.e., the statistics of the signal like the mean, variance and higher-order moments do not change over time. However, many real-world signals are cyclostationary, i.e., the statistics of the signal change *periodically* over time. CSP techniques exploit this cyclostationary property, and can be used to detect the presence of signals in noise, perform modulation recognition, and separate signals that are overlapping in both time and frequency.

Talk about how for single carrier signals its really just an autocorrelation with an extra shift, and at the right shift the main lobe of each pulse will line up with the sidelobe of the same pulse.  And for OFDM it is the same thing but with the cyclic prefix added on to each symbol.  Explanations of CSP in textbooks and other resources tend to be very math-heavy, but we will try to keep things as simple as possible.

************************************************
The Cyclic Autocorrelation Function (CAF)
************************************************

A good place to start understanding CSP is the cyclic autocorrelation function (CAF). The CAF is an extension of the traditional autocorrelation function to cyclostationary signals. As a refresher, the autocorrelation of a random process is the expected product of two time instants of the process and is defined as: :math:`R_x(t_1, t_2) = E[x(t_1)x^*(t_2)]`. Intuitively, it represents the degree to which a signal exhibits repetitive behavior. This can alternatively be written as :math:`R_x(t, \tau) = E[x(t+\tau/2)x^*(t-\tau/2)]` where :math:`\tau` is the delay between the two signals and :math:`t` is the midpoint. Some signals exhibit the property where their autocorrelation does not depend upon the midpoint :math:`t` and only on the delay :math:`\tau`. These signals are stationary of order 2 or just stationary. For cyclostationary signals, however, the midpoint does matter, meaning that the autocorrelation depends on both the delay and the midpoint, a function of two lag parameters.

Cyclostationary signals possess a periodic or almost periodic autocorrelation, and the CAF is the set of Fourier series coefficients that describe this periodicity. In other words, the CAF is the amplitude and phase of the harmonics present in a signal's autocorrelation, giving it the following form: 

.. math::
    R_x(\tau, \alpha) = \lim_{T\rightarrow\infty} \frac{1}{T} \int_{-T/2}^{T/2} x(t + \tau/2)x^*(t - \tau/2)e^{-j2\pi \alpha t}dt.

It can be seen that the CAF is a function of two variables, the delay :math:`\tau` (tau) and the cycle frequency :math:`\alpha`.

In Python, the CAF at a given alpha and tau value can be computed using the following code snippet (we'll fill out the surrounding code shortly):

.. code-block:: python
 
 np.sum(np.roll(samples, -1*tau//2) *
        np.conj(np.roll(samples, tau//2)) *
        np.exp(-2j * np.pi * alpha * np.arange(len(samples))))

In order to play with the CAF, we first need to simulate an example signal. For now we will use a rectangular BPSK signal (i.e., BPSK without pulse-shaping applied) with 20 samples per symbol, added to some AWGN.  We will apply a frequency offset to the BPSK signal, so that later we can show off how cyclostationary processing can be used to estimate the frequency offset as well as the cyclic frequency.  The following code snippet simulates the IQ samples we will use for the remainder of the next two sections:

.. code-block:: python

 N = 100000 # number of samples to simulate
 f_offset = 0.2 # Hz normalized
 sps = 20 # cyclic freq (alpha) will be 1/sps or 0.05 Hz normalized
 
 symbols = np.random.randint(0, 2, int(np.ceil(N/sps))) * 2 - 1 # random 1's and -1's
 bpsk = np.repeat(symbols, sps)  # repeat each symbol sps times to make rectangular BPSK
 bpsk = bpsk[:N]  # clip off the extra samples
 bpsk = bpsk * np.exp(2j * np.pi * f_offset * np.arange(N)) # Freq shift up the BPSK, this is also what makes it complex
 noise = np.random.randn(N) + 1j*np.random.randn(N) # complex white Gaussian noise
 samples = bpsk + 0.1*noise  # add noise to the signal

Just for fun let's look at the power spectral density (FFT) of the signal itself, *before* any CSP is performed:

.. image:: ../_images/psd_of_bpsk_used_for_caf.svg
   :align: center 
   :target: ../_images/psd_of_bpsk_used_for_caf.svg
   :alt: PSD of BPSK used for CAF

It has the 0.2 Hz frequency shift that we applied, and the samples per symbol of 20 leads to a fairly narrow signal.  Because we did not apply pulse shaping, the signal tapers off very slowly in frequency.

Now we will compute the CAF at the correct alpha, and over a range of tau values (we'll use tau from -100 to +100 as a starting point).  The correct alpha in our case is simply the samples per symbol inverted, or 0.05 Hz.  Keep in mind we are using normalized Hz, which essentially means our sample rate is 1 and all our frequencies will be between -0.5 and +0.5 Hz.  To generate the CAF in Python, we will loop over tau:

.. code-block:: python

 correct_alpha = 1/sps
 taus = np.arange(-100, 100)
 CAF = np.zeros(len(taus), dtype=complex)
 for i in range(len(taus)):
     CAF[i] = np.sum(np.roll(samples, -1*taus[i]//2) *
                     np.conj(np.roll(samples, taus[i]//2)) *
                     np.exp(-2j * np.pi * correct_alpha * np.arange(N)))

Let's plot the real part of :code:`CAF` using :code:`plt.plot(taus, np.real(CAF))`:

.. image:: ../_images/caf_at_correct_alpha.svg
   :align: center 
   :target: ../_images/caf_at_correct_alpha.svg
   :alt: CAF at correct alpha

It looks a little funky, but keep in mind that tau is still in the time domain, and the pattern we see above will make more sense after we study the SCF in the next section.

One thing we can do is calculate the CAF over a range of alphas, and at each alpha we can find the power in the CAF, by taking its magnitude and taking either the sum or average (doesn't make a difference in this case).  Then if we plot these powers over alpha, we should see spikes at the cyclic frequencies within our signal.  The following code adds the for loop, and uses an alpha step size of 0.005 Hz (note that this will take a long time to run!):

.. code-block:: python

 alphas = np.arange(0, 0.5, 0.005)
 CAF = np.zeros((len(alphas), len(taus)), dtype=complex)
 for j in range(len(alphas)):
     for i in range(len(taus)):
         CAF[j, i] = np.sum(np.roll(samples, -1*taus[i]//2) *
                         np.conj(np.roll(samples, taus[i]//2)) *
                         np.exp(-2j * np.pi * alphas[j] * np.arange(N)))
 plt.plot(alphas, np.average(np.abs(CAF), axis=1))
 plt.xlabel('Alpha')
 plt.ylabel('CAF Power')

.. image:: ../_images/caf_avg_over_alpha.svg
   :align: center 
   :target: ../_images/caf_avg_over_alpha.svg
   :alt: CAF average over alpha

Not only do we see the expected spike at 0.05 Hz, but we also see a spike at integer multiples of 0.05 Hz.  This is because the CAF is a Fourier series, and the harmonics of the fundamental frequency are present in the CAF, especially when we are looking at PSK/QAM signals without pulse shaping.

While the CAF is interesting, it is really just an intermediate step to reach our end-goal; the Spectral Correlation Function (SCF), which we will discuss next.

************************************************
The Spectral Correlation Function (SCF)
************************************************

Just as the CAF shows us the periodicity in the autocorrelation of a signal, the SCF shows us the periodicity in the power spectral density (PSD) of a signal. The autocorrelation and the PSD are in fact a Fourier Transform pair, and it therefore it should not come as a surprise that the CAF and the SCF are also a Fourier Transform pair.

* Discuss the Cyclic Wiener Relationship (says that the CAF and the SCF are Fourier transforms of each other)
* Discuss generalization of the power spectral density
* Frequency smoothing and time smoothing methods
* Include some illustrations of the SCF for simple cyclostationary signals like BPSK and QPSK with rect and SRRC pulse shapes

First let's look at the SCF at the correct alpha (0.05 Hz) for our rectangular BPSK signal.  All we need to do is take the FFT of the CAF and plot the magnitude.  The following code snippet goes along with the CAF code we wrote earlier when computing just one alpha:

.. code-block:: python

 f = np.linspace(-0.5, 0.5, len(taus))
 SCF = np.fft.fftshift(np.fft.fft(CAF))
 plt.plot(f, np.abs(SCF))
 plt.xlabel('Frequency')
 plt.ylabel('SCF')

.. image:: ../_images/fft_of_caf.svg
   :align: center 
   :target: ../_images/fft_of_caf.svg
   :alt: FFT of CAF

Note that we can see the 0.2 Hz frequency offset that we applied when simulating the BPSK signal (this has nothing to do with the cyclic frequency or samples per symbol). 

Below is an interactive JavaScript app that implements an SCF, so that you can play around with different signal and SCF parameters.  The frequency of the signal is a fairly straightforward knob, and shows how well the SCF can identify RF frequency.  Try adding pulse shaping by unchecking the Rectangular Pulse option, and play around with different rolloff values.  Note that using the default alpha-step, not all samples per symbols will lead to a visible spike in the SCF.  You can try lowering alpha-step, although it will increase the processing time. 

.. raw:: html

    <form id="mainform" name="mainform">
        <label>Samples to Simulate </label>
        <select id="N">
            <option value="1024">1024</option>
            <option value="2048">2048</option>
            <option value="4096">4096</option>
            <option value="8192" selected="selected">8192</option>
            <option value="16384">16384</option>
            <option value="32768">32768</option>
            <option value="65536">65536</option>
            <option value="131072">131072</option>
            <option value="262144">262144</option>
        </select>
        <br />
        <label>Frequency [normalized Hz] </label>
        <input type="range" id="freq" value="0.2" min="-0.5" max="0.5" step="0.05">
        <span id="freq_display">0.2</span>
        <br />
        <label>Samples per Symbol [int] </label>
        <input type="range" id="sps" value="20" min="4" max="30" step="1">
        <span id="sps_display">20</span>
        <br />
        <label>RC Rolloff [0 to 1] </label>
        <input type="number" id="rolloff" value="0.5" min="0" max="1" step="0.0001">
        <label>Rectangular Pulses </label>
        <input type="checkbox" id="rect" checked>
        <br />
        <label>Alpha Start </label>
        <input type="number" id="alpha_start" value="0" min="0" max="100" step="0.0001">
        <br />
        <label>Alpha Stop </label>
        <input type="number" id="alpha_stop" value="0.3" min="0" max="1" step="0.0001">
        <br />
        <label>Alpha Step </label>
        <input type="number" id="alpha_step" value="0.001" min="0.0001" max="0.1" step="0.0001">
        <br />
        <label>Noise Level </label>
        <input type="number" id="noise" value="0.001" min="0" max="10" step="0.0001">
        <br />
        <button type="submit" id="submit_button">Submit</button>
    </form>
    <form id="resetform" name="resetform">
        <button type="submit" id="submit_button">Reset</button>
    </form>
    <canvas id="scf_canvas"></canvas>
    <script>cyclostationary_app()</script>
    </body>



********************************
Frequency Smoothing Method (FSM)
********************************

the number of samples ends up determining your freq domain resolution

talk about how window length impacts things, since it doesnt really change the resolution, just the window size used in the convolve

point out how even though there is only 1 FFT, you still need to do a ton of convolves

.. code-block:: python

    alphas = np.arange(0, 0.3, 0.001)
    Nw = 256 # window length
    N = len(samples) # signal length
    window = np.hanning(Nw)

    X = np.fft.fftshift(np.fft.fft(samples)) # FFT of entire signal

    SCF = np.zeros((len(alphas), N), dtype=complex)
    for i in range(len(alphas)):
        shift = int(alphas[i] * N/2)
        SCF[i, :] = np.roll(X, -shift) * np.conj(np.roll(X, shift))
        SCF[i, :] = np.convolve(SCF[i, :], window, mode='same')
    SCF = np.abs(SCF)
    SCF[0, :] = 0 # null out alpha=0 which is just the PSD of the signal, it throws off the dynamic range

    SCF = SCF[:, ::Nw//2] # decimate by Nw/2 in the freq domain to reduce pixels

    extent = (-0.5, 0.5, float(np.max(alphas)), float(np.min(alphas)))
    plt.imshow(SCF, aspect='auto', extent=extent, vmax=np.max(SCF)/2)
    plt.xlabel('Frequency [Normalized Hz]')
    plt.ylabel('Cyclic Frequency [Normalized Hz]')
    plt.show()

.. image:: ../_images/scf_freq_smoothing.svg
   :align: center 
   :target: ../_images/scf_freq_smoothing.svg
   :alt: SCF with the Frequency Smoothing Method (FSM), showing cyclostationary signal processing


***************************
Time Smoothing Method (TSM)
***************************

talk about the importance of the window length because it determines the resolution

note the addition of an overlap parameter

point out that the javascript app in the SCF section actually uses the TSM method, with 0 overlap for speed sake

.. code-block:: python

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

    extent = (-0.5, 0.5, float(np.max(alphas)), float(np.min(alphas)))
    plt.imshow(SCF, aspect='auto', extent=extent, vmax=np.max(SCF)/2)
    plt.xlabel('Frequency [Normalized Hz]')
    plt.ylabel('Cyclic Frequency [Normalized Hz]')
    plt.show()

.. image:: ../_images/scf_time_smoothing.svg
   :align: center 
   :target: ../_images/scf_time_smoothing.svg
   :alt: SCF with the Time Smoothing Method (TSM), showing cyclostationary signal processing

Looks the same as the FSM!

********************************
Pulse-Shaped BPSK
********************************


********************************
SNR and Number of Symbols
********************************


********************************
QPSK and QAM
********************************

********************************
OFDM
********************************

********************************
Multiple Overlapping Signals
********************************

********************************
Spectral Coherence Function
********************************

The coherence version of the SCF, sometimes refered to as COH, is simply a normalized version of the SCF

********************************
Conjugates
********************************

***********************************************
Strip Spectral Correlation Analyzer (SSCA)
***********************************************

The FSM and TSM techniques presented earlier work great, especially when you want to calculate a specific set of cyclic frequencies (note how both implementations involve looping over cyclic frequency as the outer loop). However, there is an even more efficient SCF implementation known as the Strip Spectral Correlation Analyzer (SSCA), which inherently calculates the full set of cyclic frequencies (at a certain resolution).  

Note, code may be at the end of https://apps.dtic.mil/sti/pdfs/ADA311555.pdf

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
