.. _intro-chapter:

#############
Introduction
#############

***************************
Purpose and Target Audience
***************************

First and foremost, a couple important terms:

**Software-Defined Radio (SDR):**
    As a *concept* it refers to using software to perform signal processing tasks that were traditionally performed by hardware, specific to radio/RF applications.  This software can be run on a general-purpose computer (CPU), FPGA, or even GPU, and it can be used for real-time applications or offline processing of recorded signals.  Analogous terms include "software radio" and "RF digital signal processing".

    As a *thing* (e.g., "an SDR") it typically refers to a device that you can plug an antenna into and receive RF signals, with the digitized RF samples being sent to a computer for processing or recording (e.g., over USB, Ethernet, PCI).  Many SDRs also have transmit capabilities, allowing the computer to send samples to the SDR which then transmits the signal at a specified RF frequency.  Some embedded-style SDRs include an onboard computer.

**Digital Signal Processing (DSP):**
    The digital processing of signals; in our case, RF signals.

This textbook acts as a hands-on introduction to the areas of DSP, SDR, and wireless communications.  It is designed for someone who is:

#. Interested in *using* SDRs to do cool stuff
#. Good with Python
#. Relatively new to DSP, wireless communications, and SDR
#. A visual learner, preferring animations over equations
#. Better at understanding equations *after* learning the concepts
#. Looking for concise explanations, not a 1,000 page textbook

An example is a Computer Science student interested in a job involving wireless communications after graduation, although it can be used by anyone itching to learn about SDR who has programming experience.  As such, it covers the necessary theory to understand DSP techniques without the intense math that is usually included in DSP courses.  Instead of burying ourselves in equations, an abundance of images and animations are used to help convey the concepts, such as the Fourier series complex plane animation below.  I believe that equations are best understood *after* learning the concepts through visuals and practical exercises.  The heavy use of animations is why PySDR will never have a hard copy version being sold on Amazon.  

.. image:: ../_images/fft_logo_wide.gif
   :scale: 70 %   
   :align: center
   :alt: The PySDR logo created using a Fourier transform
   
This textbook is meant to introduce concepts quickly and smoothly, enabling the reader to perform DSP and use SDRs intelligently.  It is not meant to be a reference textbook for all DSP/SDR topics; there are plenty of great textbooks already out there, such as `Analog Devices' SDR textbook
<https://www.analog.com/en/education/education-library/software-defined-radio-for-engineers.html>`_ and `dspguide.com <http://www.dspguide.com/>`_.  You can always use Google to recall trig identities or the Shannon limit.  Think of this textbook like a gateway into the world of DSP and SDR: it's lighter and less of a time and monetary commitment, when compared to more traditional courses and textbooks.

To cover foundational DSP theory, an entire semester of "Signals and Systems", a typical course within electrical engineering, is condensed into a few chapters.  Once the DSP fundamentals are covered, we launch into SDRs, although DSP and wireless communications concepts continue to come up throughout the textbook.

***********
Why Python?
***********

Given the name PySDR, you may think Python is a critical part of this resource, but in reality the choice of programming language is not a big deal.  In the era of AI, converting code between languages is trivial.  In this textbook **we use Python almost as a form of pseudocode**, with the bonus that we can actually run it, see results, plot signals, sweep parameters, etc.  Python was chosen as the language simply because it's free, easy to run on all platforms, low boilerplate, lightweight syntax, easily readable, and has a massive ecosystem of libraries and example code in the wild.  It also helps that most SDRs have a Python API.

PySDR purposefully does not include a custom Python library or any wrapper functions, all code is in straight Python, using the standard libraries such as NumPy (standard library for arrays and high-level math), SciPy (more DSP-specific functions such as filter design), and Matplotlib (plotting, allows us to visualize signals).

Note that while Python is "slower" than C/C++ in general, most functions within Python/NumPy are actually implemented in C/C++ under the hood and heavily optimized, so you might be surprised how fast CPU-based DSP can run in Python.  Likewise, the SDR APIs we use (e.g., UHD) are simply a set of Python bindings for C/C++ functions/classes.  For fielded RF systems, high-rate signal processing is typically implemented in the FPGA anyway!

Some PySDR chapters contain example code that can be opened as a web-based Jupyter notebook (using JupyterLite), allowing you to play with and run the Python examples entirely from your browser without installing anything.  To check if it works on your browser, try opening `this example <../jupyterlite/notebooks/index.html?path=example.ipynb>`_.

************
Contributing
************

If you got value from PySDR, please share it with colleagues, students, and other lifelong learners who may be interested in the material.  You can also donate through the `PySDR Patreon <https://www.patreon.com/PySDR>`_ as a way to say thanks and get your name on the left of every page below the chapter list. There is also an option to `make a one-time donation <https://www.paypal.com/donate/?hosted_button_id=FH3LQCJRUVPWL>`_.

If you get through any amount of this textbook and email me at marc@pysdr.org with questions/comments/suggestions, then congratulations, you will have contributed to this textbook!  You can also edit the source material directly on the `textbook's GitHub page <https://github.com/777arc/PySDR/tree/master/content>`_ (your change will start a new pull request).  Feel free to submit an issue or even a Pull Request (PR) with fixes or improvements.  Those who submit valuable feedback/fixes will be permanently added to the acknowledgments section below.  Not good at Git but have changes to suggest?  Feel free to email me at marc@pysdr.org.

*****************
Acknowledgements
*****************

Thank you to anyone who has read any portion of this textbook and provided feedback, and especially to:

- `Barry Duggan <http://github.com/duggabe>`_
- Matthew Hannon
- James Hayek
- Deidre Stuffer
- Tarik Benaddi for `translating PySDR to French <https://pysdr.org/fr/index-fr.html>`_
- `Daniel Versluis <https://versd.bitbucket.io/content/about.html>`_ for `translating PySDR to Dutch <https://pysdr.org/nl/index-nl.html>`_
- `mrbloom <https://github.com/mrbloom>`_ for `translating PySDR to Ukrainian <https://pysdr.org/ukraine/index-ukraine.html>`_
- `Yimin Zhao <https://github.com/doctormin>`_ for `translating PySDR to Simplified Chinese <https://pysdr.org/zh/index-zh.html>`_
- `Eduardo Chancay <https://github.com/edulchan>`_ for `translating PySDR to Spanish <https://pysdr.org/es/index-es.html>`_
- John Marcovici
- `Vishwaksen Reddy Dhareddy <https://www.linkedin.com/in/vishwaksen-/>`_ for contributing the Detection Chapter section on real-time packet detection

As well as all `PySDR Patreon <https://www.patreon.com/PySDR>`_ supporters!
