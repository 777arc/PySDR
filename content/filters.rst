.. _filters-chapter:

#############
Filters
#############

In this chapter we learn about digital filters using Python.  We cover types of filters (FIR/IIR and low-pass/high-pass/band-pass/band-stop), how filters are represented digitally, and how they are designed.  We finish with an introduction to pulse shaping, which we further explore in the :ref:`pulse-shaping-chapter` chapter.

*************************
Filter Basics
*************************

Filters are used in many disciplines. For example, image processing makes heavy use of 2D filters, where the input and output are images.  You might use a filter every morning to make your coffee, which filters out solids from liquid.  In DSP, filters are primarily used for:

1. Separation of signals that have been combined (e.g., extracting the signal you want)
2. Removal of excess noise after receiving a signal
3. Restoration of signals that have been distorted in some way (e.g., an audio equalizer is a filter)

There are certainly other uses for filters, but this chapter is meant to introduce the concept rather than explain all the ways filtering can happen.

You may think we only care about digital filters; this textbook explores DSP, after all. However, it's important to know that a lot of filters will be analog, like those in our SDRs placed before the analog-to-digital converter (ADC) on the receive side. The following image juxtaposes a schematic of an analog filter circuit with a flowchart representation of a digital filtering algorithm.

.. image:: ../_images/analog_digital_filter.png
   :scale: 70 % 
   :align: center
   :alt: Analog vs digital filters
   
In DSP, where the input and output are signals, a filter has one input signal and one output signal:

.. tikz:: [font=\sffamily\Large, scale=2]
   \definecolor{babyblueeyes}{rgb}{0.36, 0.61, 0.83}
   \node [draw,
    color=white,
    fill=babyblueeyes,
    minimum width=4cm,
    minimum height=2.4cm
   ]  (filter) {Filter};
   \draw[<-, very thick] (filter.west) -- ++(-2,0) node[left,align=center]{Input\\(time domain)} ;
   \draw[->, very thick] (filter.east) -- ++(2,0) node[right,align=center]{Output\\(time domain)};   
   :libs: positioning
   :xscale: 80

You cannot feed two different signals into a single filter without adding them together first or doing some other operation.  Likewise, the output will always be one signal, i.e., a 1D array of numbers.

There are four basic types of filters: low-pass, high-pass, band-pass, and band-stop. Each type modifies signals to focus on different ranges of frequencies within them. The plots below demonstrate how frequencies in signals are filtered for each type, presented first with just positive frequencies (easier to understand), then also including negative.

.. image:: ../_images/filter_types.png
   :scale: 70 % 
   :align: center
   :alt: Filter types, including low-pass, high-pass, band-pass, and band-stop filtering in the frequency domain


.. START OF FILTER TYPES TIKZ
.. raw:: html

   <table><tbody><tr><td>

.. This draw the lowpass filter
.. tikz:: [font=\sffamily\large]    
   \draw[->, thick] (-5,0) -- (5,0) node[below]{Frequency};
   \draw[->, thick] (0,-0.5) node[below]{0 Hz} -- (0,5) node[left=1cm]{\textbf{Low-Pass}};
   \draw[red, thick, smooth] plot[tension=0.5] coordinates{(-5,0) (-2.5,0.5) (-1.5,3) (1.5,3) (2.5,0.5) (5,0)};
   :xscale: 100

.. raw:: html

   </td><td  style="padding: 0px">

.. this draws the highpass filter
.. tikz:: [font=\sffamily\large]    
   \draw[->, thick] (-5,0) -- (5,0) node[below]{Frequency};
   \draw[->, thick] (0,-0.5) node[below]{0 Hz} -- (0,5) node[left=1cm]{\textbf{High-Pass}};
   \draw[red, thick, smooth] plot[tension=0.5] coordinates{(-5,3) (-2.5,2.5) (-1.5,0.3) (1.5,0.3) (2.5,2.5) (5,3)};
   :xscale: 100

.. raw:: html

   </td></tr><tr><td>

.. this draws the bandpass filter
.. tikz:: [font=\sffamily\large]    
   \draw[->, thick] (-5,0) -- (5,0) node[below]{Frequency};
   \draw[->, thick] (0,-0.5) node[below]{0 Hz} -- (0,5) node[left=1cm]{\textbf{Band-Pass}};
   \draw[red, thick, smooth] plot[tension=0.5] coordinates{(-5,0) (-4.5,0.3) (-3.5,3) (-2.5,3) (-1.5,0.3) (1.5, 0.3) (2.5,3) (3.5, 3) (4.5,0.3) (5,0)};
   :xscale: 100

.. raw:: html

   </td><td>

.. and finally the bandstop filter
.. tikz:: [font=\sffamily\large]    
   \draw[->, thick] (-5,0) -- (5,0) node[below]{Frequency};
   \draw[->, thick] (0,-0.5) node[below]{0 Hz} -- (0,5) node[left=1cm]{\textbf{Band-Stop}};
   \draw[red, thick, smooth] plot[tension=0.5] coordinates{(-5,3) (-4.5,2.7) (-3.5,0.3) (-2.5,0.3) (-1.5,2.7) (1.5, 2.7) (2.5,0.3) (3.5, 0.3) (4.5,2.7) (5,3)};   
   :xscale: 100

.. raw:: html

   </td></tr></tbody></table>

.. .......................... end of filter plots in tikz


Each filter permits certain frequencies to remain from a signal while blocking other frequencies.  The range of frequencies a filter lets through is known as the "passband", and "stopband" refers to what is blocked.  In the case of the low-pass filter, it passes low frequencies and stops high frequencies, so 0 Hz will always be in the passband.  For a high-pass and band-pass filter, 0 Hz will always be in the stopband.

Do not confuse these filtering types with filter algorithmic implementation (e.g., IIR vs FIR).  The most common type by far is the low-pass filter (LPF) because we often represent signals at baseband.  LPF allows us to filter out everything "around" our signal, removing excess noise and other signals.

*************************
Filter Representation
*************************

For most filters we will see (known as FIR, or Finite Impulse Response, type filters), we can represent the filter itself with a single array of floats.  For filters symmetrical in the frequency domain, these floats will be real (versus complex), and there tends to be an odd number of them.  We call this array of floats "filter taps".  We often use :math:`h` as the symbol for filter taps.  Here is an example of a set of filter taps, which define one filter:

.. code-block:: python

    h =  [ 9.92977939e-04  1.08410297e-03  8.51595307e-04  1.64604862e-04
     -1.01714338e-03 -2.46268845e-03 -3.58236429e-03 -3.55412543e-03
     -1.68583512e-03  2.10562324e-03  6.93100252e-03  1.09302641e-02
      1.17766532e-02  7.60955496e-03 -1.90555639e-03 -1.48306750e-02
     -2.69313236e-02 -3.25659606e-02 -2.63400086e-02 -5.04184562e-03
      3.08099470e-02  7.64264738e-02  1.23536693e-01  1.62377258e-01
      1.84320776e-01  1.84320776e-01  1.62377258e-01  1.23536693e-01
      7.64264738e-02  3.08099470e-02 -5.04184562e-03 -2.63400086e-02
     -3.25659606e-02 -2.69313236e-02 -1.48306750e-02 -1.90555639e-03
      7.60955496e-03  1.17766532e-02  1.09302641e-02  6.93100252e-03
      2.10562324e-03 -1.68583512e-03 -3.55412543e-03 -3.58236429e-03
     -2.46268845e-03 -1.01714338e-03  1.64604862e-04  8.51595307e-04
      1.08410297e-03  9.92977939e-04]

Example Use-Case
########################

To learn how filters are used, let's look an an example where we tune our SDR to a frequency of an existing signal, and we want to isolate it from other signals.  Remember that we tell our SDR which frequency to tune to, but the samples that the SDR captures are at baseband, meaning the signal will display as centered around 0 Hz. We will have to keep track of which frequency we told the SDR to tune to.  Here is what we might receive:

.. image:: ../_images/filter_use_case.png
   :scale: 70 % 
   :align: center
   :alt: GNU Radio frequency domain plot of signal of interest and an interfering signal and noise floor

Because our signal is already centered at DC (0 Hz), we know we want a low-pass filter.  We must choose a "cutoff frequency" (a.k.a. corner frequency), which will determine when the passband transitions to stopband.  Cutoff frequency will always be in units of Hz.  In this example, 3 kHz seems like a good value:

.. image:: ../_images/filter_use_case2.png
   :scale: 70 % 
   :align: center 

However, the way most low-pass filters work, the negative frequency boundary will be -3 kHz as well.  I.e., it’s symmetrical around DC (later on you will see why).  Our cutoff frequencies will look something like this (the passband is the area in between):

.. image:: ../_images/filter_use_case3.png
   :scale: 70 % 
   :align: center 

After creating and applying the filter with a cutoff freq of 3 kHz, we now have:

.. image:: ../_images/filter_use_case4.png
   :scale: 70 % 
   :align: center 
   :alt: GNU Radio frequency domain plot of signal of interest and an interfering signal and noise floor, with interference filtered out

This filtered signal will look confusing until you recall that our noise floor *was* at the green line around -65 dB.  Even though we can still see the interfering signal centered at 10 kHz, we have *severely* decreased the power of that signal. It's now below where the noise floor was!  We also removed most of the noise that existed in the stopband.

In addition to cutoff frequency, the other main parameter of our low-pass filter is called the "transition width".  Transition width, also measured in Hz, instructs the filter how quickly it has to go between the passband and stopband since an instant transition is impossible.

Let's visualize transition width.  In the diagram below, the :green:`green` line represents the ideal response for transitioning between a passband and stopband, which essentially has a transition width of zero.  The :red:`Red` line demonstrates the result of a realistic filter, which has some ripple and a certain transition width.

.. image:: ../_images/realistic_filter.png
   :scale: 100 % 
   :align: center
   :alt: Frequency response of a low-pass filter, showing ripple and transition width

You might be wondering why we wouldn't just set the transition width as small as possible.  The reason is mainly that a smaller transition width results in more taps, and more taps means more computations--we will see why shortly.  A 50-tap filter can run all day long using 1% of the CPU on a Raspberry Pi.  Meanwhile, a 50,000 tap filter will cause your CPU to explode!
Typically we use a filter designer tool, then see how many taps it outputs, and if it's way too many (e.g., more than 100) we increase the transition width.  It all depends on the application and hardware running the filter, of course.

In the filtering example above, we had used a cutoff of 3 kHz and a transition width of 1 kHz (it's hard to actually see the transition width just looking at these screenshots).  The resulting filter had 77 taps.

Back to filter representation.  Even though we might show the list of taps for a filter, we usually represent filters visually in the frequency domain.  We call this the "frequency response" of the filter, and it shows us the behavior of the filter in frequency. Here is the frequency response of the filter we were just using:

.. image:: ../_images/filter_use_case5.png
   :scale: 100 % 
   :align: center 

Note that what I'm showing here is *not* a signal--it's just the frequency domain representation of the filter.  That can be a little hard to wrap your head around at first, but as we look at examples and code, it will click.

A given filter also has a time domain representation; it’s called the "impulse response" of the filter because it is what you see in the time domain if you take an impulse and put it through the filter. (Google "Dirac delta function" for more info about what an impulse is). For a FIR type filter, the impulse response is simply the taps themselves.  For that 77 tap filter we used earlier, the taps are:

.. code-block:: python

    h =  [-0.00025604525581002235, 0.00013669139298144728, 0.0005385575350373983,
    0.0008378280326724052, 0.000906112720258534, 0.0006353431381285191,
    -9.884083502996931e-19, -0.0008822851814329624, -0.0017323142383247614,
    -0.0021665366366505623, -0.0018335371278226376, -0.0005912294145673513,
    0.001349081052467227, 0.0033936649560928345, 0.004703888203948736,
    0.004488115198910236, 0.0023609865456819534, -0.0013707970501855016,
    -0.00564080523326993, -0.008859002031385899, -0.009428252466022968,
    -0.006394983734935522, 4.76480351940553e-18, 0.008114570751786232,
    0.015200719237327576, 0.018197273835539818, 0.01482443418353796,
    0.004636279307305813, -0.010356673039495945, -0.025791890919208527,
    -0.03587324544787407, -0.034922562539577484, -0.019146423786878586,
    0.011919975280761719, 0.05478153005242348, 0.10243935883045197,
    0.1458890736103058, 0.1762896478176117, 0.18720689415931702,
    0.1762896478176117, 0.1458890736103058, 0.10243935883045197,
    0.05478153005242348, 0.011919975280761719, -0.019146423786878586,
    -0.034922562539577484, -0.03587324544787407, -0.025791890919208527,
    -0.010356673039495945, 0.004636279307305813, 0.01482443418353796,
    0.018197273835539818, 0.015200719237327576, 0.008114570751786232,
    4.76480351940553e-18, -0.006394983734935522, -0.009428252466022968,
    -0.008859002031385899, -0.00564080523326993, -0.0013707970501855016,
    0.0023609865456819534, 0.004488115198910236, 0.004703888203948736,
    0.0033936649560928345, 0.001349081052467227, -0.0005912294145673513,
    -0.0018335371278226376, -0.0021665366366505623, -0.0017323142383247614,
    -0.0008822851814329624, -9.884083502996931e-19, 0.0006353431381285191,
    0.000906112720258534, 0.0008378280326724052, 0.0005385575350373983,
    0.00013669139298144728, -0.00025604525581002235]

And even though we haven't gotten into filter design yet, here is the Python code that generated that filter:

.. code-block:: python

    import numpy as np
    from scipy import signal
    import matplotlib.pyplot as plt

    num_taps = 51 # it helps to use an odd number of taps
    cut_off = 3000 # Hz
    sample_rate = 32000 # Hz

    # create our low pass filter
    h = signal.firwin(num_taps, cut_off, nyq=sample_rate/2)

    # plot the impulse response
    plt.plot(h, '.-')
    plt.show()

Simply plotting this array of floats gives us the filter's impulse response:

.. image:: ../_images/impulse_response.png
   :scale: 100 % 
   :align: center
   :alt: Example of impulse response of a filter, plotting the taps in the time domain

And here is the code that was used to produce the frequency response, shown earlier.  It's a little more complicated because we have to create the x-axis array of frequencies. 

.. code-block:: python

    # plot the frequency response
    H = np.abs(np.fft.fft(h, 1024)) # take the 1024-point FFT and magnitude
    H = np.fft.fftshift(H) # make 0 Hz in the center
    w = np.linspace(-sample_rate/2, sample_rate/2, len(H)) # x axis
    plt.plot(w, H, '.-')
    plt.show()

Real vs. Complex Filters
########################

The filter I showed you had real taps, but taps can also be complex.  Whether the taps are real or complex doesn't have to match the signal you put through it, i.e., you can put a complex signal through a filter with real taps and vice versa.  When the taps are real, the filter's frequency response will be symmetrical around DC (0 Hz).  Typically we use complex taps when we need asymmetry, which does not happen too often.

.. draw real vs complex filter
.. tikz:: [font=\sffamily\Large,scale=2] 
   \definecolor{babyblueeyes}{rgb}{0.36, 0.61, 0.83}   
   \draw[->, thick] (-5,0) node[below]{$-\frac{f_s}{2}$} -- (5,0) node[below]{$\frac{f_s}{2}$};
   \draw[->, thick] (0,-0.5) node[below]{0 Hz} -- (0,1);
   \draw[babyblueeyes, smooth, line width=3pt] plot[tension=0.1] coordinates{(-5,0) (-1,0) (-0.5,2) (0.5,2) (1,0) (5,0)};
   \draw[->,thick] (6,0) node[below]{$-\frac{f_s}{2}$} -- (16,0) node[below]{$\frac{f_s}{2}$};
   \draw[->,thick] (11,-0.5) node[below]{0 Hz} -- (11,1);
   \draw[babyblueeyes, smooth, line width=3pt] plot[tension=0] coordinates{(6,0) (11,0) (11,2) (11.5,2) (12,0) (16,0)};
   \draw[font=\huge\bfseries] (0,2.5) node[above,align=center]{Example Low-Pass Filter\\with Real Taps};
   \draw[font=\huge\bfseries] (11,2.5) node[above,align=center]{Example Low-Pass Filter\\with Complex Taps};

As an example of complex taps, let's go back to the filtering use-case, except this time we want to receive the other interfering signal (without having to re-tune the radio).  That means we want a band-pass filter, but not a symmetrical one. We only want to keep (a.k.a "pass") frequencies between around 7 kHz to 13 kHz (we don't want to also pass -13 kHz to -7 kHz):

.. image:: ../_images/filter_use_case6.png
   :scale: 70 % 
   :align: center 

One way to design this kind of filter is to make a low-pass filter with a cutoff of 3 kHz and then frequency shift it.  Remember that we can frequency shift x(t) (time domain) by multiplying it by :math:`e^{j2\pi f_0t}`.  In this case :math:`f_0` should be 10 kHz, which shifts our filter up by 10 kHz. Recall that in our Python code from above, :math:`h` was the filter taps of the low-pass filter.  In order to create our band-pass filter we just have to multiply those taps by :math:`e^{j2\pi f_0t}`, although it involves creating a vector to represent time based on our sample period (inverse of the sample rate):

.. code-block:: python

    # (h was found using the first code snippet)

    # Shift the filter in frequency by multiplying by exp(j*2*pi*f0*t)
    f0 = 10e3 # amount we will shift
    Ts = 1.0/sample_rate # sample period
    t = np.arange(0.0, Ts*len(h), Ts) # time vector. args are (start, stop, step)
    exponential = np.exp(2j*np.pi*f0*t) # this is essentially a complex sine wave

    h_band_pass = h * exponential # do the shift

    # plot impulse response
    plt.figure('impulse')
    plt.plot(np.real(h_band_pass), '.-')
    plt.plot(np.imag(h_band_pass), '.-')
    plt.legend(['real', 'imag'], loc=1)

    # plot the frequency response
    H = np.abs(np.fft.fft(h_band_pass, 1024)) # take the 1024-point FFT and magnitude
    H = np.fft.fftshift(H) # make 0 Hz in the center
    w = np.linspace(-sample_rate/2, sample_rate/2, len(H)) # x axis
    plt.figure('freq')
    plt.plot(w, H, '.-')
    plt.xlabel('Frequency [Hz]')
    plt.show()

The plots of the impulse response and frequency response are shown below:

.. image:: ../_images/shifted_filter.png
   :scale: 60 % 
   :align: center 

Because our filter is not symmetrical around 0 Hz, it has to use complex taps. Therefore we need two lines to plot those complex taps.  What we see in the left-hand plot above is still the impulse response.  Our frequency response plot is what really validates that we created the kind of filter we were hoping for, where it will filter out everything except the signal centered around 10 kHz.  Once again, remember that the plot above is *not* an actual signal: it's just a representation of the filter.  It can be very confusing to grasp because when you apply the filter to the signal and plot the output in the frequency domain, in many cases it will look roughly the same as the filter's frequency response itself.

If this subsection added to the confusion, don't worry, 99% of the time you'll be dealing with simple low pass filters with real taps anyway. 

.. _convolution-section:

***********
Convolution
***********

We will take a brief detour to introduce the convolution operator. Feel free to skip this section if you are already familiar with it.

Adding two signals together is one way of combining two signals into one. In the :ref:`freq-domain-chapter` chapter we explored how the linearity property applies when adding two signals together.  Convolution is another way to combine two signals into one, but it is very different than simply adding them.  The convolution of two signals is like sliding one across the other and integrating.  It is *very* similar to a cross-correlation, if you are familiar with that operation.  In fact it is equivalent to a cross-correlation in many cases.  We typically use the :code:`*` symbol to refer to a convolution, especially in math equations.

I believe the convolution operation is best learned through examples.  In this first example, we convolve two square pulses together:

.. image:: ../_images/rect_rect_conv.gif
   :scale: 90 % 
   :align: center 
   
We have two input signals (one red, one blue), and then the output of the convolution is displayed in black.  You can see that the output is the integration of the two signals as one slides across the other.  Because it's just a sliding integration, the result is a triangle with a maximum at the point where both square pulses lined up perfectly.  

Let's look at a few more convolutions:

.. image:: ../_images/rect_fat_rect_conv.gif
   :scale: 90 % 
   :align: center 

|

.. image:: ../_images/rect_exp_conv.gif
   :scale: 90 % 
   :align: center 

|

.. image:: ../_images/gaussian_gaussian_conv.gif
   :scale: 90 % 
   :align: center 

Note how a Gaussian convolved with a Gaussian is another Gaussian, but with a wider pulse and lower amplitude.

Because of this "sliding" nature, the length of the output is actually longer than the input.  If one signal is :code:`M` samples and the other signal is :code:`N` samples, the convolution of the two can produce :code:`N+M-1` samples.  However, functions such as :code:`numpy.convolve()` have a way to specify whether you want the whole output (:code:`max(M, N)` samples) or just the samples where the signals overlapped completely (:code:`max(M, N) - min(M, N) + 1` if you were curious).  No need to get caught up in this detail. Just know that the length of the output of a convolution is not just the length of the inputs.

So why does convolution matter in DSP?  Well for starters, to filter a signal, we can simply take the impulse response of that filter and convolve it with the signal.  FIR filtering is simply a convolution operation.

.. image:: ../_images/filter_convolve.png
   :scale: 70 % 
   :align: center 

It might be confusing because earlier we mentioned that convolution takes in two *signals* and outputs one.  We can treat the impulse response like a signal, and convolution is a math operator after all, which operates on two 1D arrays.  If one of those 1D arrays is the filter's impulse response, the other 1D array can be a piece of the input signal, and the output will be a filtered version of the input.

Let's see another example to help this click.  In the example below, the triangle will represent our filter's impulse response, and the :green:`green` signal is our signal being filtered.

.. image:: ../_images/convolution.gif
   :scale: 70 % 
   :align: center 

The :red:`red` output is the filtered signal.  

Question: What type of filter was the triangle?

.. raw:: html

   <details>
   <summary>Answers</summary>

It smoothed out the high frequency components of the green signal (i.e., the sharp transitions of the square) so it acts as a low-pass filter.

.. raw:: html

   </details>


Now that we are starting to understand convolution, I will present the mathematical equation for it.  The asterisk (*) is typically used as the symbol for convolution:

.. math::

 (f * g)(t) = \int f(\tau) g(t - \tau) d\tau
 
In this above expression, :math:`g(t)` is the signal or input that is flipped and slides across :math:`f(t)`, but :math:`g(t)` and :math:`f(t)` can be swapped and it's still the same expression.  Typically, the shorter array will be used as :math:`g(t)`.  Convolution is equal to a cross-correlation, defined as :math:`\int f(\tau) g(t+\tau)`, when :math:`g(t)` is symmetrical, i.e., it doesn't change when flipped about the origin.


*************************
Filter Implementation
*************************

We aren't going to dive too deeply into the implementation of filters. Rather, I focus on filter design (you can find ready-to-use implementations in any programming language anyway).  For now, here is one take-away:  to filter a signal with an FIR filter, you simply convolve the impulse response (the array of taps) with the input signal.  (Don't worry, a later section explains convolution.) In the discrete world we use a discrete convolution (example below).  The triangles labeled as b's are the taps.  In the flowchart, the squares labeled :math:`z^{-1}` above the triangles signify to delay by one time step.

.. image:: ../_images/discrete_convolution.png
   :scale: 80 % 
   :align: center
   :alt: Implementation of a finite impulse response (FIR) filter with delays and taps and summations

You might be able to see why we call them filter "taps" now, based on the way the filter itself is implemented. 

FIR vs IIR
##############

There are two main classes of digital filters: FIR and IIR

1. Finite impulse response (FIR)
2. Infinite impulse response (IIR)

We won't get too deep into the theory, but for now just remember: FIR filters are easier to design and can do anything you want if you use enough taps.  IIR filters are more complicated with the potential to be unstable, but they are more efficient (use less CPU and memory for the given filter). If someone just gives you a list of taps, it's assumed they are taps for an FIR filter.  If they start mentioning "poles", they are talking about IIR filters.  We will stick with FIR filters in this textbook.

Below is an example frequency response, comparing an FIR and IIR filter that do almost exactly the same filtering; they have a similar transition-width, which as we learned will determine how many taps are required.  The FIR filter has 50 taps and the IIR filter has 12 poles, which is like having 12 taps in terms of computations required.

.. image:: ../_images/FIR_IIR.png
   :scale: 70 % 
   :align: center
   :alt: Comparing finite impulse response (FIR) and infinite impulse response (IIR) filters by observing frequency response

The lesson is that the FIR filter requires vastly more computational resources than the IIR to perform roughly the same filtering operation.

Here are some real-world examples of FIR and IIR filters that you may have used before.

If you perform a "moving average" across a list of numbers, that's just an FIR filter with taps of 1's:
- h = [1 1 1 1 1 1 1 1 1 1] for a moving average filter with a window size of 10.  It also happens to be a low-pass type filter; why is that?  What's the difference between using 1's and using taps that decay to zero?

.. raw:: html

   <details>
   <summary>Answers</summary>

A moving average filter is a low-pass filter because it smooths out "high frequency" changes, which is usually why people will use one.  The reason to use taps that decay to zero on both ends is to avoid a sudden change in the output, like if the signal being filtered was zero for a while and then suddenly jumped up.

.. raw:: html

   </details>

Now for an IIR example.  Have any of you ever done this: 

    x = x*0.99 + new_value*0.01

where the 0.99 and 0.01 represent the speed the value updates (or rate of decay, same thing).  It's a convenient way to slowly update some variable without having to remember the last several values.  This is actually a form of low-pass IIR Filter.  Hopefully you can see why IIR filters have less stability than FIR.  Values never fully go away!

*************************
FIR Filter Design
*************************

In practice, most people will use a filter designer tool or a function in code (e.g., Python/SciPy) that designs the filter.  We will start by showing off what can be done within Python, then move on to 3rd party tools.  Our focus will be on FIR filters, as they are by far the most commonly used in DSP.

Within Python
#################

As part of designing the filter, which involves generating the filter taps for our desired response, we must identify the type of filter (low-pass, high-pass, band-pass, or band-stop), the cutoff frequency/frequencies, the number of taps, and optionally the transition width.

There are two main functions in SciPy we use to design FIR filters, both using what is referred to as the window method.  First, there is :code:`scipy.signal.firwin()` which is the most straightforward; it will provide the taps for a linear phase FIR filter.  The function needs to be provided the number of taps and cutoff frequency (for low-pass/high-pass) and two cutoff frequencies for band-pass/band-stop.  Optionally, you can provide the transition width.  If you provide the sample rate via :code:`fs` then the units of your cutoff frequency and transition width is in Hz, but if you don't provide it then they will be in units of normalized Hz (0 to 1 Hz).  The :code:`pass_zero` parameter is :code:`True` by default, but if you want a high-pass or band-pass filter then you must set it to :code:`False`; it indicates whether 0 Hz should be included in the passband.  It is recommended to use an odd number of taps, and 101 taps is a good starting point.  For example, let's generate a band-pass filter from 100 kHz to 200 kHz with a sample rate of 1 MHz:

.. code-block:: python

   from scipy.signal import firwin
   sample_rate = 1e6
   h = firwin(101, [100e3, 200e3], pass_zero=False, fs=sample_rate)
   print(h)

The second function is :code:`scipy.signal.firwin2()`, which is more flexible and can be used to design filters with custom frequency responses, as you provide it a list of frequencies and the desired gain at each frequency.   It also requires the number of taps, and supports the same :code:`fs` parameter as mentioned above.  For example, let's generate a filter with a low-pass region up to 100 kHz, and a separate band-pass region from 200 kHz to 300 kHz, but at half the gain as the low-pass region, and we'll use a transition width of 10 kHz:

.. code-block:: python

   from scipy.signal import firwin2
   sample_rate = 1e6
   freqs = [0, 100e3, 110e3, 190e3, 200e3, 300e3, 310e3, 500e3]
   gains = [1, 1,     0,     0,     0.5,   0.5,   0,     0]
   h2 = firwin2(101, freqs, gains, fs=sample_rate)
   print(h2)

To actually use the FIR filter on a signal, there are several options, and they all involve performing a convolution operation between the samples we want to filter, and the filter taps we generated above:

- :code:`np.convolve`
- :code:`scipy.signal.convolve`
- :code:`scipy.signal.fftconvolve`
- :code:`scipy.signal.lfilter`

The convolve-based functions above all have a :code:`mode` parameter that takes the options: :code:`'full'`, :code:`'valid'`, or :code:`'same'`.  The difference is the size of the output, because when performing a convolution, as we saw earlier in this chapter, there are transients created at the very beginning and end.  The :code:`'valid'` option will include no transients, but the output will be slightly smaller in size than the signal fed into the function.  The :code:`'same'` option will give an output the same size as the input signal, which is useful when keeping track of time or other time-domain signal features.  Lastly, the :code:`'full'` option will include all the transients; it outputs the entire convolution result.

We will now use all four functions on the firwin2 taps we created above, and using a test signal that is made with white Gaussian noise. Note that :code:`lfilter` has an extra argument (the 2nd one) which will always be 1 for an FIR filter.

.. code-block:: python

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

The above code shows basic usage of these four methods, but you may be wondering which one is best.  The plots below show all four methods using a range of tap sizes, on an input signal of 1k samples and 100k samples respectively.  It was run on an Intel Core i9-10900K.

.. image:: ../_images/convolve_comparison_1000.svg
   :align: center 
   :target: ../_images/convolve_comparison_1000.svg

.. image:: ../_images/convolve_comparison_100000.svg
   :align: center 
   :target: ../_images/convolve_comparison_100000.svg

As you can see, :code:`scipy.signal.convolve` actually switches its method to FFT-based automatically at a certain input size.  Either way, :code:`fftconvolve` is the clear winner for these size taps and inputs, which represent fairly typical sizes in RF applications.  A lot of the code within PySDR actually uses :code:`np.convolve:` simply because it's one less import and the performance difference is negligible for low data rate or non-real-time applications.

Lastly, we will show the output in the frequency domain, so we can finally check whether the firwin2 method gave us a filter that matched our design parameters.  Starting from the code above that gave us :code:`h2`:

.. code-block:: python

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

We can see that the bandpass portion is 3 dB lower than the lowpass portion:

.. image:: ../_images/fftconvolve.svg
   :align: center 
   :target: ../_images/fftconvolve.svg

As an aside, there is one more obscure option for applying the filter to a signal, called :code:`scipy.signal.filtfilt`, that performs "zero-phase filtering", which helps preserve features in a filtered time waveform exactly where they occur in the unfiltered signal.  It does this by applying the filter taps twice, first in the forward direction and then in reverse.  So the frequency response is going to be a squared version of what you would normally get.  For more information see https://www.mathworks.com/help/signal/ref/filtfilt.html or https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html.

3rd Party Tools
#######################

You can also use tools outside of Python to design a custom FIR filter.  For students I recommend this easy-to-use web app by Peter Isza that will show you impulse and frequency response: http://t-filter.engineerjs.com.  Using the default values, at the time of writing this at least, it's set up to design a low-pass filter with a passband from 0 to 400 Hz and stopband from 500 Hz and up.  The sample rate is 2 kHz, so the max frequency we can "see" is 1 kHz.

.. image:: ../_images/filter_designer1.png
   :scale: 70 % 
   :align: center 

Click the "Design Filter" button to create the taps and plot the frequency response.

.. image:: ../_images/filter_designer2.png
   :scale: 70 % 
   :align: center 

Click "Impulse Response" text above the graph to see the impulse response, which is a plot of the taps since this is an FIR filter.

.. image:: ../_images/filter_designer3.png
   :scale: 70 % 
   :align: center 

This app even includes the C++ source code to implement and use this filter.  The web app does not include any way to design IIR filters, which are in general much more difficult to design.


****************************
Arbitrary Frequency Response
****************************

Now we will consider one way to design an FIR filter ourselves in Python, starting with the desired frequency domain response, and working backwards to find the impulse response. Ultimately that is how our filter is represented (by its taps).

You start by creating a vector of your desired frequency response.  Let's design an arbitrarily shaped low-pass filter shown below:

.. image:: ../_images/filter_design1.png
   :scale: 70 % 
   :align: center 

The code used to create this filter is fairly simple:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    H = np.hstack((np.zeros(20), np.arange(10)/10, np.zeros(20)))
    w = np.linspace(-0.5, 0.5, 50)
    plt.plot(w, H, '.-')
    plt.show()


:code:`hstack()` is one way to concatenate arrays in numpy.  We know it will lead to a filter with complex taps. Why?

.. raw:: html

   <details>
   <summary>Answer</summary>

It's not symmetrical around 0 Hz.

.. raw:: html

   </details>

Our end goal is to find the taps of this filter so we can actually use it.  How do we get the taps, given the frequency response?  Well, how do we convert from the frequency domain back to the time domain?  Inverse FFT (IFFT)!  Recall that the IFFT function is almost exactly the same as the FFT function.  We also need to IFFTshift our desired frequency response before the IFFT, and then we need yet another IFFshift after the IFFT (no, they don't cancel themselves out, you can try).  This process might seem confusing. Just remember that you always should FFTshift after an FFT and IFFshift after an IFFT.

.. code-block:: python

    h = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(H)))
    plt.plot(np.real(h))
    plt.plot(np.imag(h))
    plt.legend(['real','imag'], loc=1)
    plt.show()

.. image:: ../_images/filter_design2.png
   :scale: 90 % 
   :align: center 

We will use these taps shown above as our filter.  We know that the impulse response is plotting the taps, so what we see above *is* our impulse response.  Let's take the FFT of our taps to see what the frequency domain actually looks like.  We will do a 1,024 point FFT to get a high resolution:

.. code-block:: python

    H_fft = np.fft.fftshift(np.abs(np.fft.fft(h, 1024)))
    plt.plot(H_fft)
    plt.show()

.. image:: ../_images/filter_design3.png
   :scale: 70 % 
   :align: center 

See how the frequency response not very straight... it doesn't match our original very well, if you recall the shape that we initially wanted to make a filter for.  A big reason is because our impulse response isn't done decaying, i.e., the left and right sides don't reach zero.  We have two options that will allow it to decay to zero:

**Option 1:** We "window" our current impulse response so that it decays to 0 on both sides.  It involves multiplying our impulse response with a "windowing function" that starts and ends at zero.

.. code-block:: python

    # After creating h using the previous code, create and apply the window
    window = np.hamming(len(h))
    h = h * window

.. image:: ../_images/filter_design4.png
   :scale: 70 % 
   :align: center 


**Option 2:** We re-generate our impulse response using more points so that it has time to decay.  We need to add resolution to our original frequency domain array (called interpolating).

.. code-block:: python

    H = np.hstack((np.zeros(200), np.arange(100)/100, np.zeros(200)))
    w = np.linspace(-0.5, 0.5, 500)
    plt.plot(w, H, '.-')
    plt.show()
    # (the rest of the code is the same)

.. image:: ../_images/filter_design5.png
   :scale: 60 % 
   :align: center 

.. image:: ../_images/filter_design6.png
   :scale: 70 % 
   :align: center 


.. image:: ../_images/filter_design7.png
   :scale: 50 % 
   :align: center 

Both options worked.  Which one would you choose?  The second method resulted in more taps, but the first method resulted in a frequency response that wasn't very sharp and had a falling edge wasn't very steep.  There are numerous ways to design a filter, each with their own trade-offs along the way. Many consider filter design an art.

*************************
Intro to Pulse Shaping
*************************

We will briefly introduce a very interesting topic within DSP, pulse shaping. We will consider the topic in depth in its own chapter later, see :ref:`pulse-shaping-chapter`. It is worth mentioning alongside filtering because pulse shaping is ultimately a type of filter, used for a specific purpose, with special properties.

As we learned, digital signals use symbols to represent one or more bits of information.  We use a digital modulation scheme like ASK, PSK, QAM, FSK, etc., to modulate a carrier so information can be sent wirelessly.  When we simulated QPSK in the :ref:`modulation-chapter` chapter, we only simulated one sample per symbol, i.e., each complex number we created was one of the points on the constellation--it was one symbol.  In practice we normally generate multiple samples per symbol, and the reason has to do with filtering.

We use filters to craft the "shape" of our symbols because the shape in the time domain changes the shape in the frequency domain.  The frequency domain informs us how much spectrum/bandwidth our signal will use, and we usually want to minimize it.  What is important to understand is that the spectral characteristics (frequency domain) of the baseband symbols do not change when we modulate a carrier; it just shifts the baseband up in frequency while the shape stays the same, which means the amount of bandwidth it uses stays the same.  When we use 1 sample per symbol, it's like transmitting square pulses. In fact BPSK using 1 sample per symbol *is* just a square wave of random 1's and -1's:

.. image:: ../_images/bpsk.svg
   :align: center 
   :target: ../_images/bpsk.svg

And as we have learned, square pulses are not efficient because they use an excess amount of spectrum:

.. image:: ../_images/square-wave.svg
   :align: center 

So what we do is we "pulse shape" these block-looking symbols so that they take up less bandwidth in the frequency domain.  We "pulse shape" by using a low-pass filter because it discards the higher frequency components of our symbols.  Below shows an example of symbols in the time (top) and frequency (bottom) domain, before and after a pulse-shaping filter has been applied:

.. image:: ../_images/pulse_shaping.png
   :scale: 70 % 
   :align: center 

|

.. image:: ../_images/pulse_shaping_freq.png
   :scale: 90 % 
   :align: center
   :alt: Demonstration of pulse shaping of an RF signal to reduce occupied bandwidth

Note how much quicker the signal drops off in frequency. The sidelobes are ~30 dB lower after pulse shaping; that's 1,000x less!  And more importantly, the main lobe is narrower, so less spectrum is used for the same amount of bits per second.

For now, be aware that common pulse-shaping filters include:

1. Raised-cosine filter
2. Root raised-cosine filter
3. Sinc filter
4. Gaussian filter

These filters generally have a parameter you can adjust to decrease the bandwidth used.  Below demonstrates the time and frequency domain of a raised-cosine filter with different values of :math:`\beta`, the parameter that defines how steep the roll-off is.

.. image:: ../_images/pulse_shaping_rolloff.png
   :scale: 40 % 
   :align: center 

You can see that a lower value of :math:`\beta` reduces the spectrum used (for the same amount of data). However, if the value is too low then the time domain symbols take longer to decay to zero. Actually when :math:`\beta=0` the symbols never fully decay to zero, which means we can't transmit such symbols in practice.  A :math:`\beta` value around 0.35 is common.

You will learn a lot more about pulse shaping, including some special properties that pulse shaping filters must satisfy, in the :ref:`pulse-shaping-chapter` chapter.





