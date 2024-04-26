.. _doa-chapter:

####################################
DOA & Beamforming
####################################

In this chapter we cover the concepts of beamforming, direction-of-arrival (DOA), and phased arrays in general.  Techniques such as Capon and MUSIC are discussed, using Python simulation examples. We cover beamforming vs. DOA and go through the two different types of phased arrays (Passive electronically scanned array or PESA and Active electronically scanned array or AESA).

*********************
Beamforming Overview
*********************

A phased array, a.k.a. electronically steered array, is an array/collection of antennas that can be used on the transmit or receive side in both communications and radar systems. You'll find phased arrays on ground-based systems, airborne, and on satellites.  We typically refer to the antennas that make up an array as elements, and sometimes the array is called a "sensor" instead.  These array elements are most often omnidirectional antennas, equally spaced in either a line or across two dimensions. 

Beamforming is a signal processing operation used with antenna arrays to create a *spatial* filter; it filters out signals from all directions except the desired direction(s).  Beamforming can be used to increase SNR of desired signals, null out interferers, shape beam patterns, or even transmit/receive multiple data streams at the same time and frequency.  As part of beamforming we use weights (a.k.a. coefficients) applied to each element of an array, either digitally or in analog circuitry.  We manipulate the weights to form the beam(s) of the array, hence the name beamforming!  We can steer these beams (and nulls) extremely fast; much faster than mechanically gimballed antennas, which can be thought of as an alternative to phased arrays.  We'll typically discuss beamforming within the context of a communications link, where the receiver aims to receive one or more signals at the greatest possible SNR.  Arrays also have a huge role in radar, where the goal is to detect and track targets.

.. image:: ../_images/doa_complex_scenario.svg
   :align: center 
   :target: ../_images/doa_complex_scenario.svg
   :alt: Diagram showing a complex scenario of multiple signals arriving at an array

Beamforming approaches can be broken down into three categories, namely: conventional, adaptive, and blind. Conventional beamforming is most useful when you already know the direction of arrival of the signal of interest, and the beamforming process involves choosing weights to maximize the array gain in that direction.  This can be used on both the receive or transmit side of a communication system.  Adaptive beamforming, on the other hand, typically involves adjusting the weights based on the beamformer's input, to optimize some criteria (e.g., nulling out an interferer, having multiple main beams, etc.).  Due to the closed loop and adaptive nature, adaptive beamforming is typically just used on the receive side, so the "beamformer's input" is simply your received signal, and adaptive beamforming involves adjusting the weights based on the statistics of that received data.

******************************
Direction-of-Arrival Overview
******************************

Direction-of-Arrival (DOA) within DSP/SDR refers to the process of using an array of antennas to detect and estimate the directions of arrival of one or more signals received by that array (versus beamforming, which is focused on the process of receiving a signal while rejecting as much noise and interference).  Although DOA certainly falls under the beamforming topic umbrella, the two terms can get confusing.  Some techniques such as Conventional and MVDR beamforming can apply to both DOA and beamforming, because the same technique used for beamforming is used to perform DOA by sweeping the angle of interest and performing the beamforming operation at each angle, then looking for peaks in the result (each peak is a signal, but we don't know whether it is the signal of interest, an interferer, or even a multipath bounce from the signal of interest). You can think of these DOA techniques as a wrapper around a specific beamformer.  Other beamformers are unable to be simply wrapped into a DOA routine, such as due to extra inputs that won't be available within the context of DOA.  There are also DOA techniques such as MUSIC and ESPIRT which are strictly for the purpose of DOA and are not beamformers.  Because most beamforming techniques assume you know the angle of arrival of the signal of interest, if the target is moving, or the array is moving, you will have to continuously perform DOA as an intermediate step, even if your primary goal is to receive and demodulate the signal of interest.

Phased arrays and beamforming/DOA find use in all sorts of applications, although you will most often see them used in multiple forms of radar, newer WiFi standards, mmWave communication within 5G, satellite communications, and jamming. Generally, any applications that require a high-gain antenna, or require a rapidly moving high-gain antenna, are good candidates for phased arrays.

******************
Types of Arrays
******************

Phased arrays can be broken down into three types:

1. **Analog**, a.k.a. passive electronically scanned array (PESA) or traditional phased arrays, where analog phase shifters are used to steer the beam.  On the receive side, all elements are summed after phase shifting (and optionally, adjustable gain) and turned into a signal channel which is downconverted and received.  On the transmit side the inverse takes place; a single digital signal is outputted from the digital side, and on the analog side phase shifters and gain stages are used to produce the output going to each antenna.  These digital phase shifters will have a limited number of bits of resolution, and control latency.
2. **Digital**, a.k.a. active electronically scanned array (AESA), where every single element has its own RF front end, and the beamforming is done entirely in the digital domain.  This is the most expensive approach, as RF components are expensive, but it provides much more flexibility and speed than PESAs.  Digital arrays are popular with SDRs, although the number of receive or transmit channels of the SDR limits the number of elements in your array.
3. **Hybrid**, where the array consists of many subarrays that individually resemble analog arrays, where each subarray has its own RF front-end just like with digital arrays.  This is the most common approach for modern phased arrays, as it provides the best of both worlds.

Note that the terms PESA and AESA are mainly just used in the context of radar, and there is some ambiguity when it comes to exactly what constitutes a PESA or AESA.  Therefore, using the terms analog/digital/hybrid array is clearer and can be applied to any type of application.

A real-world example of each type is shown below:

.. image:: ../_images/beamforming_examples.svg
   :align: center 
   :target: ../_images/beamforming_examples.svg
   :alt: Example of phased arrays including Passive electronically scanned array (PESA), Active electronically scanned array (AESA), Hybrid array, showing Raytheon's MIM-104 Patriot Radar, ELM-2084 Israeli Multi-Mission Radar, Starlink User Terminal, aka Dishy

In addition to these three types, there is also the geometry of the array.  The simplest geometry is the uniform linear array (ULA) where the antennas are in a straight line with equal spacing (i.e., in 1 dimension).  ULAs suffer from a 180-degree ambiguity, which we will talk about later, and one solution is to place the antennas in a circle, which we call a uniform circular array (UCA).  Lastly, for 2D beams, we usually use a uniform rectangular array (URA), where the antennas are in a grid pattern.

In this chapter we focus on digital arrays, as they are more suited towards simulation and DSP, but the concepts carry over to analog and hybrid.  In the next chapter we get hands-on with the "Phaser" SDR from Analog Devices that has a 10 GHz 8-element analog array with phase and gain shifters, connected to a Pluto and Raspberry Pi.  We will also focus on the ULA geometry because it provides the simplest math and code, but all of the concepts carry over to other geometries, and at the end of the chapter we touch upon the UCA.

*******************
SDR Requirements
*******************

Analog phased arrays involve one phase shifter (and often one adjustable gain stage) per channel/element, implemented in analog RF circuitry.  This means an analog phased array is a dedicated piece of hardware that must go alongside an SDR, or be purpose-built for a specific application.  On the other hand, any SDR that contains more than one channel can be used as a digital array with no extra hardware, as long as the channels are phase coherent and sampled using the same clock, which is typically the case for SDRs that have multiple receive channels onboard.  There are many SDRs that contain **two** receive channels, such as the Ettus USRP B210 and Analog Devices Pluto (the 2nd channel is exposed using a uFL connector on the board itself).  Unfortunately, going beyond two channels involves entering the $10k+ segment of SDRs, at least as of 2023, such as the Ettus USRP N310 or the Analog Devices QuadMXFE (16 channels).  The main challenge is that low-cost SDRs are typically not able to be "chained" together to scale the number of channels.  The exception is the KerberosSDR (4 channels) and KrakenSDR (5 channels) which use multiple RTL-SDRs sharing an LO to form a low-cost digital array; the downside being the very limited sample rate (up to 2.56 MHz) and tuning range (up to 1766 MHz).  The KrakenSDR board and example antenna configuration is shown below.

.. image:: ../_images/krakensdr.jpg
   :align: center 
   :alt: The KrakenSDR
   :target: ../_images/krakensdr.jpg

In this chapter we don't use any specific SDRs; instead we simulate the receiving of signals using Python, and then go through the DSP used to perform beamforming/DOA for digital arrays.

**************************************
Intro to Matrix Math in Python/NumPy
**************************************

Python has many advantages over MATLAB, such as being free and open-source, diversity of applications, vibrant community, indices start from 0 like every other language, use within AI/ML, and there seems to be a library for anything you can think of.  But where it falls short is how matrix manipulation is coded/represented (computationally/speed-wise, it's plenty fast, with functions implemented under the hood efficiently in C/C++).  It doesn't help that there are multiple ways to represent matrices in Python, with the :code:`np.matrix` method being deprecated in favor of :code:`np.ndarray`.  In this section we provide a brief primer on doing matrix math in Python using NumPy, so that when we get to the DOA examples you'll be more comfortable.

Let's start by jumping into the most annoying part of matrix math in NumPy; vectors are treated as 1D arrays, so there's no way to distinguish between a row vector and column vector (it will be treated as a row vector by default), whereas in MATLAB a vector is a 2D object.  In Python you can create a new vector using :code:`a = np.array([2,3,4,5])` or turn a list into a vector using :code:`mylist = [2, 3, 4, 5]` then :code:`a = np.asarray(mylist)`, but as soon as you want to do any matrix math, orientation matters, and these will be interpreted as row vectors.  Trying to do a transpose on this vector, e.g. using :code:`a.T`, will **not** change it to a column vector!  The way to make a column vector out of a normal vector :code:`a` is to use :code:`a = a.reshape(-1,1)`.  The :code:`-1` tells NumPy to figure out the size of this dimension automatically, while keeping the second dimension length 1.  What this creates is technically a 2D array but the second dimension is length 1, so it's still essentially 1D from a math perspective. It's only one extra line, but it can really throw off the flow of matrix math code.

Now for a quick example of matrix math in Python; we will multiply a :code:`3x10` matrix with a :code:`10x1` matrix.  Remember that :code:`10x1` means 10 rows and 1 column, known as a column vector because it is just one column.  From our early school years we know this is a valid matrix multiplication because the inner dimensions match, and the resulting matrix size is the outer dimensions, or :code:`3x1`.  We will use :code:`np.random.randn()` to create the :code:`3x10` and :code:`np.arange()` to create the :code:`10x1`, for convenience:

.. code-block:: python

 A = np.random.randn(3,10) # 3x10
 B = np.arange(10) # 1D array of length 10
 B = B.reshape(-1,1) # 10x1
 C = A @ B # matrix multiply
 print(C.shape) # 3x1
 C = C.squeeze() # see next subsection
 print(C.shape) # 1D array of length 3, easier for plotting and other non-matrix Python code

After performing matrix math you may find your result looks something like: :code:`[[ 0.  0.125  0.251  -0.376  -0.251 ...]]` which clearly has just one dimension of data, but if you go to plot it you will either get an error or a plot that doesn't show anything.  This is because the result is technically a 2D array, and you need to convert it to a 1D array using :code:`a.squeeze()`.  The :code:`squeeze()` function removes any dimensions of length 1, and comes in handy when doing matrix math in Python.  In the example given above, the result would be :code:`[ 0.  0.125  0.251  -0.376  -0.251 ...]` (notice the missing second brackets), which can be plotted or used in other Python code that expects something 1D.

When coding matrix math the best sanity check you can do is print out the dimensions (using :code:`A.shape`) to verify they are what you expect. Consider sticking the shape in the comments after each line for future reference, and so it's easy to make sure dimensions match when doing matrix or elementwise multiplication.

Here are some common operations in both MATLAB and Python, as a sort of cheat sheet to reference:

.. list-table::
   :widths: 35 25 40
   :header-rows: 1

   * - Operation
     - MATLAB
     - Python/NumPy
   * - Create (Row) Vector, size :code:`1 x 4`
     - :code:`a = [2 3 4 5];`
     - :code:`a = np.array([2,3,4,5])`
   * - Create Column Vector, size :code:`4 x 1`
     - :code:`a = [2; 3; 4; 5];` or :code:`a = [2 3 4 5].'`
     - :code:`a = np.array([[2],[3],[4],[5]])` or |br| :code:`a = np.array([2,3,4,5])` then |br| :code:`a = a.reshape(-1,1)`
   * - Create 2D Matrix
     - :code:`A = [1 2; 3 4; 5 6];`
     - :code:`A = np.array([[1,2],[3,4],[5,6]])`
   * - Get Size
     - :code:`size(A)`
     - :code:`A.shape`
   * - Transpose a.k.a. :math:`A^T`
     - :code:`A.'`
     - :code:`A.T`
   * - Complex Conjugate Transpose |br| a.k.a. Conjugate Transpose |br| a.k.a. Hermitian Transpose |br| a.k.a. :math:`A^H`
     - :code:`A'`
     - :code:`A.conj().T` |br| |br| (unfortunately there is no :code:`A.H` for ndarrays)
   * - Elementwise Multiply
     - :code:`A .* B`
     - :code:`A * B` or :code:`np.multiply(a,b)`
   * - Matrix Multiply
     - :code:`A * B`
     - :code:`A @ B` or :code:`np.matmul(A,B)`
   * - Dot Product of two vectors (1D)
     - :code:`dot(a,b)`
     - :code:`np.dot(a,b)` (never use np.dot for 2D)
   * - Concatenate
     - :code:`[A A]`
     - :code:`np.concatenate((A,A))`

*********************
Steering Vector
*********************

To get to the fun part we have to get through a little bit of math, but the following section has been written so that the math is relatively straightforward and has diagrams to go along with it, only the most basic trig and exponential properties are used.  It's important to understand the basic math behind what we'll do in Python to perform DOA.

Consider a 1D three-element uniformly spaced array:

.. image:: ../_images/doa.svg
   :align: center 
   :target: ../_images/doa.svg
   :alt: Diagram showing direction of arrival (DOA) of a signal impinging on a uniformly spaced antenna array, showing boresight angle and distance between elements or apertures

In this example a signal is coming in from the right side, so it's hitting the right-most element first.  Let's calculate the delay between when the signal hits that first element and when it reaches the next element.  We can do this by forming the following trig problem, try to visualize how this triangle was formed from the diagram above.  The segment highlighted in red represents the distance the signal has to travel *after* it has reached the first element, before it hits the next one.

.. image:: ../_images/doa_trig.svg
   :align: center 
   :target: ../_images/doa_trig.svg
   :alt: Trig associated with direction of arrival (DOA) of uniformly spaced array

If you recall SOH CAH TOA, in this case we are interested in the "adjacent" side and we have the length of the hypotenuse (:math:`d`), so we need to use a cosine:

.. math::
  \cos(90 - \theta) = \frac{\mathrm{adjacent}}{\mathrm{hypotenuse}}

We must solve for adjacent, as that is what will tell us how far the signal must travel between hitting the first and second element, so it becomes adjacent :math:`= d \cos(90 - \theta)`.  Now there is a trig identity that lets us convert this to adjacent :math:`= d \sin(\theta)`.  This is just a distance though, we need to convert this to a time, using the speed of light: time elapsed :math:`= d \sin(\theta) / c` [seconds].  This equation applies between any adjacent elements of our array, although we can multiply the whole thing by an integer to calculate between non-adjacent elements since they are uniformly spaced (we'll do this later).  

Now to connect this trig and speed of light math to the signal processing world.  Let's denote our transmit signal at baseband :math:`x(t)` and it's being transmitting at some carrier, :math:`f_c` , so the transmit signal is :math:`x(t) e^{2j \pi f_c t}`.  Lets say this signal hits the first element at time :math:`t = 0`, which means it hits the next element after :math:`d \sin(\theta) / c` [seconds] like we calculated above.  This means the 2nd element receives:

.. math::
 x(t - \Delta t) e^{2j \pi f_c (t - \Delta t)}

.. math::
 \mathrm{where} \quad \Delta t = d \sin(\theta) / c

recall that when you have a time shift, it is subtracted from the time argument.

When the receiver or SDR does the downconversion process to receive the signal, its essentially multiplying it by the carrier but in the reverse direction, so after doing downconversion the receiver sees:

.. math::
 x(t - \Delta t) e^{2j \pi f_c (t - \Delta t)} e^{-2j \pi f_c t}

.. math::
 = x(t - \Delta t) e^{-2j \pi f_c \Delta t}

Now we can do a little trick to simplify this even further; consider how when we sample a signal it can be modeled by substituting :math:`t` for :math:`nT` where :math:`T` is sample period and :math:`n` is just 0, 1, 2, 3...  Substituting this in we get :math:`x(nT - \Delta t) e^{-2j \pi f_c \Delta t}`. Well, :math:`nT` is so much greater than :math:`\Delta t` that we can get rid of the first :math:`\Delta t` term and we are left with :math:`x(nT) e^{-2j \pi f_c \Delta t}`.  If the sample rate ever gets fast enough to approach the speed of light over a tiny distance, we can revisit this, but remember that our sample rate only needs to be a bit larger than the signal of interest's bandwidth.

Let's keep going with this math but we'll start representing things in discrete terms so that it will better resemble our Python code.  The last equation can be represented as the following, let's plug back in :math:`\Delta t`:

.. math::
 x[n] e^{-2j \pi f_c \Delta t}

.. math::
 = x[n] e^{-2j \pi f_c d \sin(\theta) / c}

We're almost done, but luckily there's one more simplification we can make.  Recall the relationship between center frequency and wavelength: :math:`\lambda = \frac{c}{f_c}` or the form we'll use: :math:`f_c = \frac{c}{\lambda}`.  Plugging this in we get:

.. math::
 x[n] e^{-2j \pi \frac{c}{\lambda} d \sin(\theta) / c}

.. math::
 = x[n] e^{-2j \pi d \sin(\theta) / \lambda}


In applied beamforming/DOA what we like to do is represent :math:`d`, the distance between adjacent elements, as a fraction of wavelength (instead of meters), the most common value chosen for :math:`d` during the array design process is to use one half the wavelength. Regardless of what :math:`d` is, from this point on we're going to represent :math:`d` as a fraction of wavelength instead of meters, making the equation and all our code simpler:

.. math::
 x[n] e^{-2j \pi d \sin(\theta)}

This is for adjacent elements, for the :math:`k`'th element we just need to multiply :math:`d` times :math:`k`:

.. math::
 x[n] e^{-2j \pi d k \sin(\theta)}

And we're done! This equation above is what you'll see in DOA papers and implementations everywhere! We typically call that exponential term the "steering vector" (often denoted as :math:`s` and in code :code:`s`) and represent it as an array, a 1D array for a 1D antenna array, etc.  In python :math:`s` is:

.. code-block:: python

 s = [np.exp(-2j*np.pi*d*0*np.sin(theta)), np.exp(-2j*np.pi*d*1*np.sin(theta)), np.exp(-2j*np.pi*d*2*np.sin(theta)), ...] # note the increasing k
 # or
 s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # where Nr is the number of receive antenna elements

Note how element 0 results in a 1+0j (because :math:`e^{0}=1`); this makes sense because everything above was relative to that first element, so it's receiving the signal as-is without any relative phase shifts.  This is purely how the math works out, in reality any element could be thought of as the reference, but as you'll see in our math/code later on, what matters is the difference in phase/amplitude received between elements.  It's all relative.

*******************
Receiving a Signal
*******************

Let's use the steering vector concept to simulate a signal arriving at an array.  For a transmit signal we'll just use a tone for now:

.. code-block:: python

 import numpy as np
 import matplotlib.pyplot as plt
 
 sample_rate = 1e6
 N = 10000 # number of samples to simulate
 
 # Create a tone to act as the transmitter signal
 t = np.arange(N)/sample_rate # time vector
 f_tone = 0.02e6
 tx = np.exp(2j * np.pi * f_tone * t)

Now let's simulate an array consisting of three omnidirectional antennas in a line, with 1/2 wavelength between adjacent ones (a.k.a. "half-wavelength spacing").  We will simulate the transmitter's signal arriving at this array at a certain angle, theta.  Understanding the steering vector :code:`s` below is why we went through all that math above.

.. code-block:: python

 d = 0.5 # half wavelength spacing
 Nr = 3
 theta_degrees = 20 # direction of arrival (feel free to change this, it's arbitrary)
 theta = theta_degrees / 180 * np.pi # convert to radians
 s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # Steering Vector
 print(s) # note that it's 3 elements long, it's complex, and the first element is 1+0j

To apply the steering vector we have to do a matrix multiplication of :code:`s` and :code:`tx`, so first let's convert both to 2D, using the approach we discussed earlier when we reviewed doing matrix math in Python.  We'll start off by making both into row vectors using :code:`ourarray.reshape(-1,1)`.  We then perform the matrix multiply, indicated by the :code:`@` symbol.  We also have to convert :code:`tx` from a row vector to a column vector using a transpose operation (picture it rotating 90 degrees) so that the matrix multiply inner dimensions match.

.. code-block:: python

 s = s.reshape(-1,1)
 print(s.shape) # 3x1
 tx = tx.reshape(-1,1)
 print(tx.shape) # 10000x1
 
 # matrix multiply
 r = s @ tx.T  # dont get too caught up by the transpose, the important thing is we're multiplying the steering vector by the tx signal
 print(r.shape) # 3x10000.  r is now going to be a 2D array, 1D is time and 1D is the spatial dimension

At this point :code:`r` is a 2D array, size 3 x 10000 because we have three array elements and 10000 samples simulated.  We can pull out each individual signal and plot the first 200 samples, below we'll plot the real part only, but there's also an imaginary part, like any baseband signal.  One annoying part of matrix math in Python is needing to add the :code:`.squeeze()`, which removes all dimensions with length 1, to get it back to a normal 1D NumPy array that plotting and other operations expects.

.. code-block:: python

 plt.plot(np.asarray(r[0,:]).squeeze().real[0:200]) # the asarray and squeeze are just annoyances we have to do because we came from a matrix
 plt.plot(np.asarray(r[1,:]).squeeze().real[0:200])
 plt.plot(np.asarray(r[2,:]).squeeze().real[0:200])
 plt.show()

.. image:: ../_images/doa_time_domain.svg
   :align: center 
   :target: ../_images/doa_time_domain.svg

Note the phase shifts between elements like we expect to happen (unless the signal arrives at boresight in which case it will reach all elements at the same time and there won't be a shift, set theta to 0 to see).  Element 0 appears to arrive first, with the others slightly delayed.  Try adjusting the angle and see what happens.

As one final step, let's add noise to this received signal, as every signal we will deal with has some amount of noise. We want to apply the noise after the steering vector is applied, because each element experiences an independent noise signal (we can do this because AWGN with a phase shift applied is still AWGN):

.. code-block:: python

 n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
 r = r + 0.5*n # r and n are both 3x10000

.. image:: ../_images/doa_time_domain_with_noise.svg
   :align: center 
   :target: ../_images/doa_time_domain_with_noise.svg

******************************
Conventional Beamforming & DOA
******************************

We will now process these samples :code:`r`, pretending we don't know the angle of arrival, and perform DOA, which involves estimating the angle of arrival(s) with DSP and some Python code!  As discussed earlier in this chapter, the act of beamforming and performing DOA are very similar and are often built off the same techniques.  Throughout the rest of this chapter we will investigate different "beamformers", and for each one we will start with the beamformer math/code that calculates the weights, :math:`w`.  These weights can be "applied" to the incoming signal :code:`r` through the simple equation :math:`w^H r`, or in Python :code:`w.conj().T @ r`.  In the example above, :code:`r` is a :code:`3x10000` matrix, but after we apply the weights we are left with :code:`1x10000`, as if our receiver only had one antenna, and we can use normal RF DSP to process the signal.  After developing the beamformer, we will apply that beamformer to the DOA problem.

We'll start with the "conventional" beamforming approach, a.k.a. delay-and-sum beamforming.  Our weights vector :code:`w` needs to be a 1D array for a uniform linear array, in our example of three elements, :code:`w` is a :code:`3x1` array of complex weights.  With conventional beamforming we leave the magnitude of the weights at 1, and adjust the phases so that the signal constructively adds up in the direction of our desired signal, which we will refer to as :math:`\theta`.  It turns out that this is the exact same math we did above, i.e., our weights are our steering vector!

.. math::
 w_{conv} = e^{-2j \pi d k \sin(\theta)}

or in Python:

.. code-block:: python

 w = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # Conventional, aka delay-and-sum, beamformer
 r = w.conj().T @ r # example of applying the weights to the received signal (i.e., perform the beamforming)

where :code:`Nr` is the number of elements in our uniform linear array with spacing of :code:`d` fractions of wavelength (most often ~0.5).  As you can see, the weights don't depend on anything other than the array geometry and the angle of interest.  If our array involved calibrating the phase, we would include those calibration values too.  You may have been able to notice by the equation for :code:`w` that the weights are complex valued and the magnitudes are all equal to one (unity).

But how do we know the angle of interest :code:`theta`?  We must start by performing DOA, which involves scanning through (sampling) all directions of arrival from -π to +π (-180 to +180 degrees), e.g., in 1 degree increments.  At each direction we calculate the weights using a beamformer; we will start by using the conventional beamformer.  Applying the weights to our signal :code:`r` will give us a 1D array of samples, as if we received it with 1 directional antenna.  We can then calculate the power in the signal by taking the variance with :code:`np.var()`, and repeat for every angle in our scan.  We will plot the results and look at it with our human eyes/brain, but what most RF DSP does is find the angle of maximum power (with a peak-finding algorithm) and call it the DOA estimate.

.. code-block:: python

 theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # 1000 different thetas between -180 and +180 degrees
 results = []
 for theta_i in theta_scan:
    w = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_i)) # Conventional, aka delay-and-sum, beamformer
    r_weighted = w.conj().T @ r # apply our weights. remember r is 3x10000
    results.append(10*np.log10(np.var(r_weighted))) # power in signal, in dB so its easier to see small and large lobes at the same time
 results -= np.max(results) # normalize
 
 # print angle that gave us the max value
 print(theta_scan[np.argmax(results)] * 180 / np.pi) # 19.99999999999998
 
 plt.plot(theta_scan*180/np.pi, results) # lets plot angle in degrees
 plt.xlabel("Theta [Degrees]")
 plt.ylabel("DOA Metric")
 plt.grid()
 plt.show()

.. image:: ../_images/doa_conventional_beamformer.svg
   :align: center 
   :target: ../_images/doa_conventional_beamformer.svg

We found our signal!  You're probably starting to realize where the term electrically steered array comes in. Try increasing the amount of noise to push it to its limit, you might need to simulate more samples being received for low SNRs.  Also try changing the direction of arrival. 

If you prefer viewing angle on a polar plot, use the following code:

.. code-block:: python

 fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
 ax.plot(theta_scan, results) # MAKE SURE TO USE RADIAN FOR POLAR
 ax.set_theta_zero_location('N') # make 0 degrees point up
 ax.set_theta_direction(-1) # increase clockwise
 ax.set_rlabel_position(55)  # Move grid labels away from other labels
 plt.show()

.. image:: ../_images/doa_conventional_beamformer_polar.svg
   :align: center 
   :target: ../_images/doa_conventional_beamformer_polar.svg
   :alt: Example polar plot of performing direction of arrival (DOA) showing the beam pattern and 180-degree ambiguity

We will keep seeing this pattern of looping over angles, and having some method of calculating the beamforming weights, then applying them to the received signal.  In the next beamforming method (MVDR) we will use our received signal :code:`r` as part of the weight calculations, making it an adaptive technique.  But first we will investigate some interesting things that happen with phased arrays, including why we have that second peak at 160 degrees.

********************
180-Degree Ambiguity
********************

Let's talk about why is there a second peak at 160 degrees; the DOA we simulated was 20 degrees, but it is not a coincidence that 180 - 20 = 160.  Picture three omnidirectional antennas in a line placed on a table.  The array's boresight is 90 degrees to the axis of the array, as labeled in the first diagram in this chapter.  Now imagine the transmitter in front of the antennas, also on the (very large) table, such that its signal arrives at a +20 degree angle from boresight.  Well the array sees the same effect whether the signal is arriving with respect to its front or back, the phase delay is the same, as depicted below with the array elements in red and the two possible transmitter DOA's in green.  Therefore, when we perform the DOA algorithm, there will always be a 180-degree ambiguity like this, the only way around it is to have a 2D array, or a second 1D array positioned at any other angle w.r.t the first array.  You may be wondering if this means we might as well only calculate -90 to +90 degrees to save compute cycles, and you would be correct!

.. image:: ../_images/doa_from_behind.svg
   :align: center 
   :target: ../_images/doa_from_behind.svg

Let's try sweeping the angle of arrival (AoA) from -90 to +90 degrees instead of keeping it constant at 20:

.. image:: ../_images/doa_sweeping_angle_animation.gif
   :scale: 100 %
   :align: center
   :alt: Animation of direction of arrival (DOA) showing the broadside of the array

As we approach the broadside of the array (a.k.a. endfire), which is when the signal arrives at or near the axis of the array, performance drops.  We see two main degradations: 1) the main lobe gets wider and 2) we get ambiguity and don't know whether the signal is coming from the left or the right.  This ambiguity adds to the 180-degree ambiguity discussed earlier, where we get an extra lobe at 180 - theta, causing certain AoA to lead to three lobes of roughly equal size.  This broadside ambiguity makes sense though, the phase shifts that occur between elements are identical whether the signal arrives from the left or right side w.r.t. the array axis.  Just like with the 180-degree ambiguity, the solution is to use a 2D array or two 1D arrays at different angles.  In general, beamforming works best when the angle is closer to the boresight.

From this point on, we will only be displaying -90 to +90 degrees in our polar plots, as the pattern will always be mirrored over the axis of the array, at least for 1D linear arrays (which is all we cover in this chapter).

********************
Beam Pattern
********************

The plots we have shown so far are DOA results; they correspond to the received power at each angle after applying the beamformer.  They were specific to a scenario that involved transmitters arriving from certain angles.  But we can also take a look at the beam pattern itself, before receiving any signal, this is sometimes referred to as the "quiescent antenna pattern" or "array response".

Recall that our steering vector we keep seeing,

.. code-block:: python

 np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta))

encapsulates the ULA geometry, and its only other parameter is the direction you want to steer towards.  We can calculate and plot the quiescent antenna pattern (array response) when steered towards a certain direction, which will tell us the arrays natural response if we don't do any additional beamforming.  This can be done by taking the FFT of the complex conjugated weights, no for loop needed!  The tricky part is padding to increase resolution, and mapping the bins of the FFT output to angle in radians or degrees, which involves an arcsine as you can see in the full example below:

.. code-block:: python

    N_fft = 512
    theta_degrees = 20 # there is no SOI, we arent processing samples, this is just the direction we want to point at
    theta = theta_degrees / 180 * np.pi
    w = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # conventional beamformer
    w = np.conj(w) # or else our answer will be negative/inverted
    w_padded = np.concatenate((w, np.zeros(N_fft - Nr))) # zero pad to N_fft elements to get more resolution in the FFT
    w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) # magnitude of fft in dB
    w_fft_dB -= np.max(w_fft_dB) # normalize to 0 dB at peak
    
    # Map the FFT bins to angles in radians
    theta_bins = np.arcsin(np.linspace(-1, 1, N_fft)) # in radians
    
    # find max so we can add it to plot
    theta_max = theta_bins[np.argmax(w_fft_dB)]
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta_bins, w_fft_dB) # MAKE SURE TO USE RADIAN FOR POLAR
    ax.plot([theta_max], [np.max(w_fft_dB)],'ro')
    ax.text(theta_max - 0.1, np.max(w_fft_dB) - 4, np.round(theta_max * 180 / np.pi))
    ax.set_theta_zero_location('N') # make 0 degrees point up
    ax.set_theta_direction(-1) # increase clockwise
    ax.set_rlabel_position(55)  # Move grid labels away from other labels
    ax.set_thetamin(-90) # only show top half
    ax.set_thetamax(90)
    ax.set_ylim([-30, 1]) # because there's no noise, only go down 30 dB
    plt.show()

.. image:: ../_images/doa_quiescent.svg
   :align: center 
   :target: ../_images/doa_quiescent.svg

It turns out that this pattern is going to almost exactly match the pattern you get when performing DOA with the conventional beamformer (delay-and-sum), when there is a single tone present at `theta_degrees` and little-to-no noise.  The plot may look different because of how low the y-axis gets in dB, or due to the size of the FFT used to create this quiescent response pattern.  Try tweaking :code:`theta_degrees` or the number of elements :code:`Nr` to see how the response changes.

Just for fun, the following animation shows the beam pattern of the conventional beamformer, for an 8-element array being steered between -90 and +90 degrees.  Also shown are the eight weights plotted in the complex plane (real and imaginary axis).

.. image:: ../_images/delay_and_sum.gif
   :scale: 90 %
   :align: center
   :alt: Beam pattern of delay and sum while viewing each weight on the complex plane

Note how all weights have unity magnitude (they stay on the unit circle), and how the higher numbered elements "spin" faster.  If you watch closely you'll notice at 0 degrees they all line up; they are all equal to 0 phase shift (1+0j).

*******************
When d is not λ/2
*******************

So far we have been using a distance between elements, d, equal to one half wavelength.  So for example, an array designed for 2.4 GHz WiFi with λ/2 spacing would have a spacing of 3e8/2.4e9/2 = 12.5cm or about 5 inches, meaning a 4x4 element array would be about 15" x 15" x the height of the antennas.  There are times when an array may not be able to achieve exactly λ/2 spacing, such as when space is restricted, or when the same array has to work on a variety of carrier frequencies.

Let's examine when the spacing is greater than λ/2, i.e., too much spacing, by varying d between λ/2 and 4λ.  We will remove the bottom half of the polar plot since it's a mirror of the top anyway.

.. image:: ../_images/doa_d_is_large_animation.gif
   :scale: 100 %
   :align: center
   :alt: Animation of direction of arrival (DOA) showing what happens when distance d is much more than half-wavelength

As you can see, in addition to the 180-degree ambiguity we discussed earlier, we now have additional ambiguity, and it gets worse as d gets higher (extra/incorrect lobes form).  These extra lobes are known as grating lobes, and they are a result of "spatial aliasing".  As we learned in the :ref:`sampling-chapter` chapter, when we don't sample fast enough we get aliasing.  The same thing happens in the spatial domain; if our elements are not spaced close enough together w.r.t. the carrier frequency of the signal being observed, we get garbage results in our analysis.  You can think of spacing out antennas as sampling space!  In this example we can see that the grating lobes don't get too problematic until d > λ, but they will occur as soon as you go above λ/2 spacing.

Now what happens when d is less than λ/2, such as when we need to fit the array in a small space?  Let's repeat the same simulation:

.. image:: ../_images/doa_d_is_small_animation.gif
   :scale: 100 %
   :align: center
   :alt: Animation of direction of arrival (DOA) showing what happens when distance d is much less than half-wavelength

While the main lobe gets wider as d gets lower, it still has a maximum at 20 degrees, and there are no grating lobes, so in theory this would still work (at least at high SNR).  To better understand what breaks as d gets too small, let's repeat the experiment but with an additional signal arriving from -40 degrees:

.. image:: ../_images/doa_d_is_small_animation2.gif
   :scale: 100 %
   :align: center
   :alt: Animation of direction of arrival (DOA) showing what happens when distance d is much less than half-wavelength and there are two signals present

Once we get lower than λ/4 there is no distinguishing between the two different paths, and the array performs poorly.  As we will see later in this chapter, there are beamforming techniques that provide more precise beams than conventional beamforming, but keeping d as close to λ/2 as possible will continue to be a theme.


*******************
Number of Elements
*******************

Coming soon!


..
   COMMENTED OUT BECAUSE IT"S NOT CLEAR WHAT THIS SECTION IS PROVIDING TO THE READER BESIDES AN ALTERNATIVE EQUATION AND TERM WHICH COULD BE PRESENTED A LOT MORE CONCISE
   **********************
   Bartlett Beamformer
   **********************

   Now that we've covered the basics, we will take a quick detour into some notational and algebraic details of what we just did, to gain knowledge on how to mathematically represent sweeping beams across space in a condensed and elegant manner.  The following algebriac notations renders itself well to vectorization, making it suitable for real-time processing.

   The process of sweeping beams across space to get an estimate of DOA actually has a technical name; it goes by "Bartlett Beamforming" (a.k.a. Fourier beamforming to some, but note that Fourier beamforming can also mean a different technique altogether).  Let's do a quick recap of what we did earlier in order to calculate our DOA, using what we now know is called Bartlett beamforming:

   #. We picked a bunch of directions to point at (e.g., -90 to +90 degrees at some interval)
   #. We calculated the beamforming weights at each direction, to point our beam in that direction
   #. The outputs of the array elements were multiplied with their corresponding wieght, and all results were summed
   #. We calculated the signal power at each direction, then plotted the results
   #. Peaks were found, each one inferring that a signal was likely received from that direction

   We are now going to write the series of steps we just reiterated mathematically.  Let the signal received by the array be represented by the steering vector :math:`\mathbf{s}`. This received signal is a function of the direction of arrival (DOA) of the signal, which we will denote as :math:`\theta`. Let the weight applied to the steering vector be represented by :math:`\mathbf{w}`. The output of the array is the dot product of the steering vector and the weight, which we will denote as :math:`\mathbf{w}^{H} \mathbf{s}`.  Now, the power of the received signal can be obtained by squaring the magnitude of the output of the array. This is represented as :math:`\left| \mathbf{w}^{H} \mathbf{s} \right|^{2} = \mathbf{w}^{H} \mathbf{s} \mathbf{s}^{H} \mathbf{w} = \mathbf{w} \mathbf{R_{ss}} \mathbf{w}`, where :math:`\mathbf{R}` is the spatial covariance matrix estimate. The spatial covariance matrix measures the similarity between the samples received from the different elements of the array. We repeat for each direction we want to scan, but note that the only thing that changes between direction is \mathbf{w}.  We are also free to pick the list of directions, it doesn't have to be a -90 to +90 degree sweep, and we can process them all in parallel if we wish, using the same value of :math:`\mathbf{R}` for all.  This is the essence of Bartlett beamforming, i.e the beam sweep that we described using the earlier generated python code.

   .. math::
      P = \left\| \mathbf{w} \mathbf{s}\right\|^2 
      
      = (\mathbf{w}^H\mathbf{s})(\mathbf{w}^H\mathbf{s})^* 
      
      = \mathbf{s}^H\mathbf{w}\mathbf{w}^H\mathbf{s}
      
      = \mathbf{s}^H\mathbf{R}\mathbf{s}

   This mathematical representation extends to other DOA techniques as well.

**********************
Spatial Tapering
**********************

Spatial tapering is a technique used alongside the conventional beamformer, where the magnitude of the weights are adjusted to achieve certain features.  Although even if you aren't using the conventional beamformer, the concept of tapering is still important to understand.  Recall that when we calculated the conventional beamformer weights, it was a series of complex numbers which all had magnitudes of one (unity).  With spatial tapering we will multiply the weights by scalars to scale their magnitude.  Let's start by seeing what happens if we multiply the weights by random values between 0 and 1, i.e.:

.. code-block:: python

    tapering = np.random.uniform(0, 1, Nr) # random tapering
    w *= tapering

We will simulate a signal being received at boresight (0 degrees) at high SNR to see what happens.  Note that this process is equivalent and will have the same results as simulated the quiescent antenna pattern for the given weights, as we discuss at the end of this chapter.

.. image:: ../_images/spatial_tapering_animation.gif
   :scale: 80 %
   :align: center
   :alt: Spatial tapering using random values to adjust the magnitude of the weights

Try to observe the width of the main lobe, and the position of nulls.

It turns out that tapering can reduce the sidelobes, which is often desired, by reducing the magnitude of the weights at the **edges** of the array.  This time we will transition between using a rectangular window (no window) and a Hamming window, as our tapering function.

.. code-block:: python

    tapering = np.hamming(Nr) # Hamming window function
    w *= tapering

.. image:: ../_images/spatial_tapering_animation2.gif
   :scale: 80 %
   :align: center
   :alt: Spatial tapering using a hamming window to adjust the magnitude of the weights

The main lobe width is also affected by tapering, and it can be made wider or narrower depending on the tapering function used (less sidelobes usually leads to a wider mainlobe).

*********************
Adaptive Beamforming
*********************

The conventional beamformer we discussed earlier is a simple and effective way to perform beamforming, but it has some limitations.  For example, it doesn't work well when there are multiple signals arriving from different directions, or when the noise level is high.  In these cases, we need to use more advanced beamforming techniques, which are often referred to as "adaptive" beamforming.  The idea behind adaptive beamforming is to use the received signal to calculate the weights, instead of using a fixed set of weights like we did with the conventional beamformer.  This allows the beamformer to adapt to the environment and provide better performance, because the weights are now based on the statistics of the received data.

Adaptive beamforming techniques can be further broken down into regular and subspace-based.  Subspace methods such as MUSIC and ESPRIT are very powerful, but they require guessing how many signals are present, and they require at least three elements to function (although it is recommended to have at least four).  

The first adaptive beamforming technique we will investigate is MVDR, which tends to be the de facto algorithm when people talk about adaptive beamforming.

**********************
MVDR/Capon Beamformer
**********************

We will now look at a beamformer that is slightly more complicated than the conventional/delay-and-sum technique, but tends to perform much better, called the Minimum Variance Distortionless Response (MVDR) or Capon Beamformer.  Recall that variance of a signal corresponds to how much power is in the signal.  The idea behind MVDR is to keep the signal at the angle of interest at a fixed gain of 1 (0 dB), while minimizing the total variance/power of the resulting beamformed signal.  If our signal of interest is kept fixed then minimizing the total power means minimizing interferers and noise as much as possible.  It is often referred to as a "statistically optimal" beamformer.

The MVDR/Capon beamformer can be summarized in the following equation:

.. math::

 w_{mvdr} = \frac{R^{-1} s}{s^H R^{-1} s}

The vector :math:`s` is the steering vector corresponding to the desired direction and was discussed at the beginning of this chapter.  :math:`R` is the spatial covariance matrix estimate based on our received samples, found using :code:`R = np.cov(r)` or calculated manually by multiplying :code:`r` with the complex conjugate transpose of itself, i.e., :math:`R = r r^H`,  The spatial covariance matrix is a :code:`Nr` x :code:`Nr` size matrix (3x3 in the examples we have seen so far) that tells us how similar the samples received from the three elements are.




.. raw:: html

   <details>
   <summary>For those interested in the MVDR derivation, expand this</summary>


**Beamforming Output** - The output of the beamformer using a weight vector :math:`\mathbf{w}` is given by:

.. math::

 y(t) = \mathbf{w}^H \mathbf{x}(t)


**Optimization Problem** - The goal is to determine the beamforming weights that minimize the output power subject to a distortion-less response towards a desired direction :math:`\theta_0`. Formally, the problem can be expressed as:

.. math::

 \min_{\mathbf{w}} \, \mathbf{w}^H \mathbf{R} \mathbf{w} \quad \text{subject to} \quad \mathbf{w}^H \mathbf{s} = 1

where:

* :math:`\mathbf{R} = E[\mathbf{X}\mathbf{X}^H]` is the covariance matrix of the received signals
* :math:`\mathbf{s}` is the steering vector towards the desired signal direction :math:`\theta_0`

**Lagrangian Method** - Introduce a Lagrange multiplier :math:`\lambda` and form the Lagrangian:

.. math::

 L(\mathbf{w}, \lambda) = \mathbf{w}^H \mathbf{R} \mathbf{w} - \lambda (\mathbf{w}^H \mathbf{s} - 1)

**Solving the Optimization** - Differentiating the Lagrangian with respect to the :math:`\mathbf{w^H}` and setting the derivative to zero, we obtain:

.. math::

 \frac{\partial L}{\partial \mathbf{w}^*} = 2\mathbf{R}\mathbf{w} - \lambda \mathbf{s} = 0

 \mathbf{w} = \lambda \mathbf{s} \mathbf{{R^{-1}}}


To solve for :math:`\lambda`, apply the constraint :math:`\mathbf{w}^H \mathbf{s} = 1`:

.. math::

 \implies (\lambda \mathbf{s^{H}}\mathbf{{R^{-1}}})s = 1

 \implies \lambda = \frac{1}{\mathbf{s}^{H}\mathbf{R}^{-1}\mathbf{s}}
 
 \mathbf{R}\mathbf{w} = \lambda \mathbf{s}
 
 \mathbf{w_{mvdr}} = \frac{\mathbf{R}^{-1} \mathbf{s}}{\mathbf{s}^H \mathbf{R}^{-1} \mathbf{s}}

.. raw:: html

   </details>

If we already know the direction of the signal of interest, and that direction does not change, we only have to calculate the weights once and simply use them to receive our signal of interest.  Although even if the direction doesn't change, we benefit from recalculating these weights periodically, to account for changes in the interference/noise, which is why we refer to these non-conventional digital beamformers as "adaptive" beamforming; they use information in the signal we receive to calculate the best weights.  Just as a reminder, we can *perform* beamforming using MVDR by calculating these weights and applying them to the signal with :code:`w.conj().T @ r`, just like we did in the conventional method, the only difference is how the weights are calculated.

To perform DOA using the MVDR beamformer, we simply repeat the MVDR calculation while scanning through all angles of interest.  I.e., we act like our signal is coming from angle :math:`\theta`, even if it isn't.  At each angle we calculate the MVDR weights, then apply them to the received signal, then calculate the power in the signal.  The angle that gives us the highest power is our DOA estimate, or even better we can plot power as a function of angle to see the beam pattern, as we did above with the conventional beamformer, that way we don't need to assume how many signals are present.

In Python we can implement the MVDR/Capon beamformer as follows, which will be done as a function so that it's easy to use later on:

.. code-block:: python

 # theta is the direction of interest, in radians, and r is our received signal
 def w_mvdr(theta, r):
    s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # steering vector in the desired direction theta
    s = s.reshape(-1,1) # make into a column vector (size 3x1)
    R = np.cov(r) # gives a Nr x Nr covariance matrix of the samples
    #R = (r @ r.conj().T)/r.shape[1] # or calc the covariance matrix manually
    Rinv = np.linalg.pinv(R) # 3x3. pseudo-inverse tends to work better/faster than a true inverse
    w = (Rinv @ s)/(s.conj().T @ Rinv @ s) # MVDR/Capon equation! numerator is 3x3 * 3x1, denominator is 1x3 * 3x3 * 3x1, resulting in a 3x1 weights vector
    return w

Using this MVDR beamformer in the context of DOA, we get the following Python example:

.. code-block:: python

 theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # 1000 different thetas between -180 and +180 degrees
 results = []
 for theta_i in theta_scan:
    w = w_mvdr(theta_i, r) # 3x1
    r_weighted = w.conj().T @ r # apply weights
    power_dB = 10*np.log10(np.var(r_weighted)) # power in signal, in dB so its easier to see small and large lobes at the same time
    results.append(power_dB)
 results -= np.max(results) # normalize

When applied to the previous DOA example simulation, we get the following:

.. image:: ../_images/doa_capons.svg
   :align: center 
   :target: ../_images/doa_capons.svg

It appears to work fine, but to really compare this to other techniques we'll have to create a more interesting problem.  Let's set up a simulation with an 8-element array receiving three signals from different angles: 20, 25, and 40 degrees, with the 40 degree one received at a much lower power than the other two, as a way to spice things up.  Our goal will be to detect all three signals, meaning we want to be able to see noticeable peaks (high enough for a peak-finder algorithm to extract).  The code to generate this new scenario is as follows:

.. code-block:: python

 Nr = 8 # 8 elements
 theta1 = 20 / 180 * np.pi # convert to radians
 theta2 = 25 / 180 * np.pi
 theta3 = -40 / 180 * np.pi
 s1 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta1)).reshape(-1,1) # 8x1
 s2 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta2)).reshape(-1,1)
 s3 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta3)).reshape(-1,1)
 # we'll use 3 different frequencies.  1xN
 tone1 = np.exp(2j*np.pi*0.01e6*t).reshape(1,-1)
 tone2 = np.exp(2j*np.pi*0.02e6*t).reshape(1,-1)
 tone3 = np.exp(2j*np.pi*0.03e6*t).reshape(1,-1)
 r = s1 @ tone1 + s2 @ tone2 + 0.1 * s3 @ tone3
 n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
 r = r + 0.05*n # 8xN

You can put this code at the top of your script, since we are generating a different signal than the original example. If we run our MVDR beamformer on this new scenario we get the following results:

.. image:: ../_images/doa_capons2.svg
   :align: center 
   :target: ../_images/doa_capons2.svg

It works pretty well, we can see the two signals received only 5 degrees apart, and we can also see the 3rd signal (at -40 or 320 degrees) that was received at one tenth the power of the others.   Now let's run the conventional beamformer on this new scenario:

.. image:: ../_images/doa_complex_scenario.svg
   :align: center 
   :target: ../_images/doa_complex_scenario.svg

While it might be a pretty shape, it's not finding all three signals at all...  By comparing these two results we can see the benefit from using a more complex and "adaptive" beamformer.  

As a quick aside for the interested reader, there is actually an optimization that can be made when performing DOA with MVDR, using a trick.  Recall that we calculate the power in a signal by taking the variance, which is the mean of the magnitude squared (assuming our signals average value is zero which is almost always the case for baseband RF).  We can represent taking the power in our signal after applying our weights as:

.. math::

 P_{mvdr} = \frac{1}{N} \sum_{n=0}^{N-1} \left| w^H_{mvdr} r_n \right|^2

If we switch from using a summation to the expectation operator, and plug in the equation for the MVDR weights, we get:

.. math::

   P_{mvdr} = E \left( \left| w^H_{mvdr} r_n \right| ^2 \right)

   = w^H_{mvdr} E \left( r r^H \right) w_{mvdr}

   = w^H_{mvdr} R w_{mvdr}

   = \frac{s^H R^{-1} s}{s^H R^{-1} s} \cdot R \cdot \frac{R^{-1} s}{s^H R^{-1} s}

   = \frac{s^H R^{-1} s}{(s^H R^{-1} s)(s^H R^{-1} s)}

   = \frac{1}{s^H R^{-1} s}

Meaning we don't have to apply the weights at all, this final equation above for power can be used directly in our DOA scan, saving us some computations:

.. code-block:: python

    def power_mvdr(theta, r):
        s = np.exp(-2j * np.pi * d * np.arange(r.shape[0]) * np.sin(theta)) # steering vector in the desired direction theta_i
        s = s.reshape(-1,1) # make into a column vector (size 3x1)
        R = np.cov(r) # Calc covariance matrix. gives a Nr x Nr covariance matrix of the samples
        Rinv = np.linalg.pinv(R) # 3x3. pseudo-inverse tends to work better than a true inverse
        return 1/(s.conj().T @ Rinv @ s).squeeze()

To use this in the previous simulation, within the for loop, the only thing left to do is take the :code:`10*np.log10()` and you're done, there are no weights to apply; we skipped calculating the weights!

There are many more beamformers out there, but next we are going to take a moment to discuss how the number of elements impacts our ability to perform beamforming and DOA.

**********************
Covariance Matrix
**********************

Let's take a brief moment to discuss the spatial covariance matrix, which is a key concept in *adaptive* beamforming.  A covariance matrix is a mathematical representation of the similarity between pairs of elements in a random vector (in our case, it's the elements in our array, so we call it the *spatial* covariance matrix).  A covariance matrix is always square, and the values along the diagonal correspond to the covariance of each element with itself.  We calculate the spatial covariance matrix *estimate*; it is only an estimate because we have a limited number of samples. 

In general, the covariance matrix is defined as:

:math:`\mathrm{cov}(X) = E \left[ (X - E[X])(X - E[X])^H \right]`

for wireless signals at baseband, :math:`E[X]` is typically zero or very close to zero, so this simplifies to:

:math:`\mathrm{cov}(X) = E[X X^H]`

Given a limited number of IQ samples, :math:`\bm{r}`, we can estimate this covariance, which we will denote as :math:`\hat{R}`:

.. math::

 \hat{R} = \frac{\bm{r} \bm{r}^H}{N}

         = \frac{1}{N} \sum^N_{n=1} r_n r_n^H

where :math:`N` is the number of samples (not the number of elements).  In Python this looks like:

:code:`R = (r @ r.conj().T)/r.shape[1]`

Alternatively, we can use the built-in NumPy function:

:code:`R = np.cov(r)`
    
As an example, we will look at the spatial covariance matrix for the scenario where we only had one transmitter and three elements:

.. code-block:: python

   [[ 1.494+0.j    0.486+0.881j -0.543+0.839j]
    [ 0.486-0.881j 1.517 +0.j    0.483+0.886j]
    [-0.543-0.839j 0.483-0.886j  1.499+0.j   ]]

Note how the diagonal elements are real and roughly the same, this is because they are really only telling us the received signal power at each element, which will be roughly the same between elements since they are all set to the same gain.  The off-diagonal elements are really where the important values are, although looking at the raw values doesn't tell us much other than there is a significant amount of correlation between elements.

As part of adaptive beamforming you will see a pattern where we take the inverse of the spatial correlation matrix. This inverse tells us how two elements are related to each other after removing the influence of other elements. It is referred to as the "precision matrix" in statistics and "whitening matrix" in radar.

**********************
LCMV Beamformer
**********************

While MVDR is powerful, what if we have more than one SOI?  Thankfully, with just a small tweak to MVDR, we can implement a scheme that handles multiple SOIs, called the Linearly Constrained Minimum Variance (LCMV) beamformer.  It is a generalization of MVDR, where we specify the desired response for multiple directions, kind of like a spatial version of SciPy's :code:`firwin2()` for those familiar with it.  The optimum weight vector for the LCMV beamformer can be summarized in the following equation: 

.. math::

   w_{lcmv} = R^{-1} C [C^H R^{-1} C]^{-1} f

where :math:`C` is a matrix comprising of the steering vectors of the corresponding SOIs and interferers, and :math:`f` is the desired response vector. The vector :math:`f` for a particular row takes the value of 0 when the corresponding steering vector is to be nulled, and takes a value of 1 when we want a beam pointed at it. For example, if we have two sources of interest and two sources of interference, we can set :code:`f = [1,1,0,0]`. The LCMV beamformer is a powerful tool that can be used to suppress interference and noise from multiple directions while simultaneously enhancing the signal of interest from multiple directions.  The catch is that the total number of nulls and beams you can form simultaneously is limited by the size of the array (the number of elements). Furthermore, you need to craft the steering vector for each of the SOIs and interferers, which isn't always readily available in practical applications. When estimates are used instead, the performance of the LCMV beamformer can degrade.  It is for this reason that we prefer to steer nulls using the spatial covariance matrix :math:`R` (based on statistics of the received signal), instead of "hardcoding" nulls by estimating the AoA of the interferer (which could have error) and crafting the steering vector in that direction, with a 0 added to :math:`f`.  

As far as performing LCMV in Python, it is very similar to MVDR, but we have to specify :code:`C` which is made up of potentially multiple steering vectors, and :code:`f` which is a 1D array of 1's and 0's as previously mentioned.  The following code snippet demonstrates how to implement the LCMV beamformer for two SOIs (15 and 60 degrees); recall that MVDR only supports 1 SOI at a time.  Therefore, our :code:`f = [1; 1]` with no zeros, as we will not be including any "hardcoded" nulls.  We will simulate a scenario with four interferers, arriving from angles -60, -30, 0, and 30 degrees.

.. code-block:: python

    # Let's point at the SOI at 15 deg, and another potential SOI that we didn't actually simulate at 60 deg
    soi1_theta = 15 / 180 * np.pi # convert to radians
    soi2_theta = 60 / 180 * np.pi

    # LCMV weights
    R_inv = np.linalg.pinv(np.cov(r)) # 8x8
    s1 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(soi1_theta)).reshape(-1,1) # 8x1
    s2 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(soi2_theta)).reshape(-1,1) # 8x1
    C = np.concatenate((s1, s2), axis=1) # 8x2
    f = np.ones(2).reshape(-1,1) # 2x1

    # LCMV equation
    #    8x8   8x2                    2x8        8x8   8x2  2x1
    w = R_inv @ C @ np.linalg.pinv(C.conj().T @ R_inv @ C) @ f # output is 8x1

We can plot the beam pattern of :code:`w` using the FFT method we showed earlier:

.. image:: ../_images/lcmv_beam_pattern.svg
   :align: center 
   :target: ../_images/lcmv_beam_pattern.svg
   :alt: Example beam pattern when using the LCMV beamformer

As you can see, we have beams pointed at the two directions of interest, and nulls at the locations of the interferers (like MVDR, we don't have to tell it where the emitters are, it figures it out based on the received signal).  Green and red dots are added to the plot to show AoAs of the SOIs and interferers, respectively.

.. raw:: html

   <details>
   <summary>For the full code expand this section</summary>

.. code-block:: python

    # Simulate received signal
    Nr = 8 # 8 elements
    theta1 = -60 / 180 * np.pi # convert to radians
    theta2 = -30 / 180 * np.pi
    theta3 = 0 / 180 * np.pi
    theta4 = 30 / 180 * np.pi
    s1 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta1)).reshape(-1,1) # 8x1
    s2 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta2)).reshape(-1,1)
    s3 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta3)).reshape(-1,1)
    s4 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta4)).reshape(-1,1)
    # we'll use 3 different frequencies.  1xN
    tone1 = np.exp(2j*np.pi*0.01e6*t).reshape(1,-1)
    tone2 = np.exp(2j*np.pi*0.02e6*t).reshape(1,-1)
    tone3 = np.exp(2j*np.pi*0.03e6*t).reshape(1,-1)
    tone4 = np.exp(2j*np.pi*0.04e6*t).reshape(1,-1)
    r = s1 @ tone1 + s2 @ tone2 + s3 @ tone3 + s4 @ tone4
    n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
    r = r + 0.5*n # 8xN

    # Let's point at the SOI at 15 deg, and another potential SOI that we didn't actually simulate at 60 deg
    soi1_theta = 15 / 180 * np.pi # convert to radians
    soi2_theta = 60 / 180 * np.pi

    # LCMV weights
    R_inv = np.linalg.pinv(np.cov(r)) # 8x8
    s1 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(soi1_theta)).reshape(-1,1) # 8x1
    s2 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(soi2_theta)).reshape(-1,1) # 8x1
    C = np.concatenate((s1, s2), axis=1) # 8x2
    f = np.ones(2).reshape(-1,1) # 2x1

    # LCMV equation
    #    8x8   8x2                    2x8        8x8   8x2  2x1
    w = R_inv @ C @ np.linalg.pinv(C.conj().T @ R_inv @ C) @ f # output is 8x1

    # Plot beam pattern
    w = w.squeeze() # reduce to a 1D array
    N_fft = 1024
    w = np.conj(w) # or else our answer will be negative/inverted
    w_padded = np.concatenate((w, np.zeros(N_fft - Nr))) # zero pad to N_fft elements to get more resolution in the FFT
    w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) # magnitude of fft in dB
    w_fft_dB -= np.max(w_fft_dB) # normalize to 0 dB at peak
    theta_bins = np.arcsin(np.linspace(-1, 1, N_fft)) # Map the FFT bins to angles in radians
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta_bins, w_fft_dB) # MAKE SURE TO USE RADIAN FOR POLAR
    # Add dots where interferers and SOIs are
    ax.plot([theta1], [0], 'or')
    ax.plot([theta2], [0], 'or')
    ax.plot([theta3], [0], 'or')
    ax.plot([theta4], [0], 'or')
    ax.plot([soi1_theta], [0], 'og')
    ax.plot([soi2_theta], [0], 'og')
    ax.set_theta_zero_location('N') # make 0 degrees point up
    ax.set_theta_direction(-1) # increase clockwise
    ax.set_thetagrids(np.arange(-90, 105, 15)) # it's in degrees
    ax.set_rlabel_position(55)  # Move grid labels away from other labels
    ax.set_thetamin(-90) # only show top half
    ax.set_thetamax(90)
    ax.set_ylim([-30, 1]) # because there's no noise, only go down 30 dB
    plt.show()

.. raw:: html

   </details>

*******************
MUSIC
*******************

We will now change gears and talk about a different kind of beamformer. All of the previous ones have fallen in the "delay-and-sum" category, but now we will dive into "sub-space" methods.  These involve dividing the signal subspace and noise subspace, which means we must estimate how many signals are being received by the array, to get a good result.  MUltiple SIgnal Classification (MUSIC) is a very popular sub-space method that involves calculating the eigenvectors of the covariance matrix (which is a computationally intensive operation by the way).  We split the eigenvectors into two groups: signal sub-space and noise-subspace, then project steering vectors into the noise sub-space and steer for nulls.  That might seem confusing at first, which is part of why MUSIC seems like black magic!

The core MUSIC equation is the following:

.. math::
 \hat{\theta} = \mathrm{argmax}\left(\frac{1}{s^H V_n V^H_n s}\right)

where :math:`V_n` is that list of noise sub-space eigenvectors we mentioned (a 2D matrix).  It is found by first calculating the eigenvectors of :math:`R`, which is done simply by :code:`w, v = np.linalg.eig(R)` in Python, and then splitting up the vectors (:code:`w`) based on how many signals we think the array is receiving.  There is a trick for estimating the number of signals that we'll talk about later, but it must be between 1 and :code:`Nr - 1`.  I.e., if you are designing an array, when you are choosing the number of elements you must have one more than the number of anticipated signals.  One thing to note about the equation above is :math:`V_n` does not depend on the steering vector :math:`s`, so we can precalculate it before we start looping through theta.  The full MUSIC code is as follows:

.. code-block:: python

 num_expected_signals = 3 # Try changing this!
 
 # part that doesn't change with theta_i
 R = np.cov(r) # Calc covariance matrix. gives a Nr x Nr covariance matrix
 w, v = np.linalg.eig(R) # eigenvalue decomposition, v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
 eig_val_order = np.argsort(np.abs(w)) # find order of magnitude of eigenvalues
 v = v[:, eig_val_order] # sort eigenvectors using this order
 # We make a new eigenvector matrix representing the "noise subspace", it's just the rest of the eigenvalues
 V = np.zeros((Nr, Nr - num_expected_signals), dtype=np.complex64)
 for i in range(Nr - num_expected_signals):
    V[:, i] = v[:, i]
 
 theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # -180 to +180 degrees
 results = []
 for theta_i in theta_scan:
     s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_i)) # Steering Vector
     s = s.reshape(-1,1)
     metric = 1 / (s.conj().T @ V @ V.conj().T @ s) # The main MUSIC equation
     metric = np.abs(metric.squeeze()) # take magnitude
     metric = 10*np.log10(metric) # convert to dB
     results.append(metric) 
 
 results /= np.max(results) # normalize

Running this algorithm on the complex scenario we have been using, we get the following very precise results, showing the power of MUSIC:

.. image:: ../_images/doa_music.svg
   :align: center 
   :target: ../_images/doa_music.svg
   :alt: Example of direction of arrival (DOA) using MUSIC algorithm beamforming

Now what if we had no idea how many signals were present?  Well there is a trick; you sort the eigenvalue magnitudes from highest to lowest, and plot them (it may help to plot them in dB):

.. code-block:: python

 plot(10*np.log10(np.abs(w)),'.-')

.. image:: ../_images/doa_eigenvalues.svg
   :align: center 
   :target: ../_images/doa_eigenvalues.svg

The eigenvalues associated with the noise-subspace are going to be the smallest, and they will all tend around the same value, so we can treat these low values like a "noise floor", and any eigenvalue above the noise floor represents a signal.  Here we can clearly see there are three signals being received, and adjust our MUSIC algorithm accordingly.  If you don't have a lot of IQ samples to process or the signals are at low SNR, the number of signals might not be as obvious.  Feel free to play around by adjusting :code:`num_expected_signals` between 1 and 7, you'll find that underestimating the number will lead to missing signal(s) while overestimating will only slightly hurt performance.

Another experiment worth trying with MUSIC is to see how close two signals can arrive at (in angle) while still distinguishing between them; sub-space techniques are especially good at that.  The animation below shows an example, with one signal at 18 degrees and another slowly sweeping angle of arrival.

.. image:: ../_images/doa_music_animation.gif
   :scale: 100 %
   :align: center

*******************
ESPRIT
*******************

Coming soon!

*******************
Circular Arrays
*******************

We will briefly talk about the Uniform Circular Array (UCA), which is a popular array geometry for DOA because it gets around the 180-degree ambiguity issue ULAs have.  The KrakenSDR, for example, is a 5-element array, and it is common to place those five elements in a circle with equal spacing between elements.  In theory, only three elements is needed to form a UCA, just like how we can make a ULA with only two elements.

All of the code we have studied so far applies to UCAs, we just have to replace the steering vector equation with one specific to the UCA:

.. code-block:: python

   radius = 0.05 # normalized by wavelength!
   d = np.sqrt(2 * radius**2 * (1 - np.cos(2*np.pi/Nr)))
   sf = 1.0 / (np.sqrt(2.0) * np.sqrt(1.0 - np.cos(2*np.pi/Nr))) # scaling factor based on geometry, eg for a hexagon it is 1.0
   x = d * sf * np.cos(2 * np.pi / Nr * np.arange(Nr))
   y = -1 * d * sf * np.sin(2 * np.pi / Nr * np.arange(Nr))
   s = np.exp(1j * 2 * np.pi * (x * np.cos(theta) + y * np.sin(theta)))
   s = s.reshape(-1, 1) # Nrx1

Lastly, you will want to scan from 0 to 360 degrees, instead of just -90 to +90 degrees like with a ULA.

*************************
Conclusion and References
*************************

All Python code, including code used to generate the figures/animations, can be found `on the textbook's GitHub page <https://github.com/777arc/PySDR/blob/master/figure-generating-scripts/doa.py>`_.

* DOA implementation in GNU Radio - https://github.com/EttusResearch/gr-doa
* DOA implementation used by KrakenSDR - https://github.com/krakenrf/krakensdr_doa/blob/main/_signal_processing/krakenSDR_signal_processor.py

.. |br| raw:: html

      <br>
