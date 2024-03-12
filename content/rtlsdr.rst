.. _rtlsdr-chapter:

##################
RTL-SDR in Python
##################

The RTL-SDR is by far the cheapest SDR, at around $30, and a great SDR to start with.  While it is receive-only and can only tune up to ~1.75 GHz, there are numerous applications it can be used for.  In this chapter, we learn how to set up the RTL-SDR software and use its Python API.

.. image:: ../_images/rtlsdrs.svg
   :align: center 
   :target: ../_images/rtlsdrs.svg
   :alt: Example RTL-SDRs

********************************
RTL-SDR Background
********************************

The RTL-SDR came into existence around 2010 when folks discovered they could hack low-cost DVB-T dongles that contained the Realtek RTL2832U chip.  DVB-T is a digital television standard primarily used in Europe, but what was interesting about the RTL2832U was that the raw IQ samples could be directly accessed, allowing the chip to be used to build a general purpose receive-only SDR.  

The RTL2832U chip includes the analog-to-digital converter (ADC) and USB controller, but it must be paired with an RF tuner.  Popular tuner chips include the Rafael Micro R820T, R828D, and Elonics E4000.  The tunable frequency range is based on the tuner chip and is usually around 50 - 1700 MHz.  The maximum sample rate, on the other hand, is determined by the RTL2832U and your computer's USB bus, and is usually around 2.4 MHz without dropping too many samples.  Keep in mind that these tuners are extremely low-cost and have very poor RF sensitivity, so adding a low-noise amplifier (LNA) and bandpass filter is often necessary to receive weak signals.

The RTL2832U always uses 8-bit samples, so the host machine will receive two bytes per IQ sample.  Premium RTL-SDRs usually come with a temperature-controlled oscillator (a.k.a. TCXO) in place of the cheaper crystal oscillator, which provides better frequency stability.  Another optional feature is a bias tee (a.k.a. bias-T), which is an onboard circuit that provides ~4.5V DC on the SMA connector, used to conveniently power an external LNA or other RF components.  This extra DC offset is on the RF side of the SDR so it does not interfere with the basic receiving operation.

For those interested in direction of arrival (DOA) or other beamforming applications, the `KrakenSDR <https://www.crowdsupply.com/krakenrf/krakensdr>`_ is a phase-coherent SDR made from five RTL-SDRs that share an oscillator and sample clock.

********************************
Software Setup
********************************

Ubuntu (or Ubuntu within WSL)
#############################

On Ubuntu 20, 22, and other Debian-based systems, you can install the RTL-SDR software with the following command.  

.. code-block:: bash

 sudo apt install rtl-sdr

This will install the librtlsdr library, and command line tools such as rtl_sdr, rtl_tcp, rtl_fm, and rtl_test.

Next, install the Python wrapper for librtlsdr using:

.. code-block:: bash

 sudo pip install pyrtlsdr

If you are using Ubuntu through WSL, on the Windows side download the latest `Zadig <https://zadig.akeo.ie/>`_ and run it to install the "WinUSB" driver for the RTL-SDR (there may be two Bulk-In Interfaces, in which case install "WinUSB" on both).  Unplug and replug the RTL-SDR once Zadig finishes.  

Next, you will need to forward the RTL-SDR USB device to WSL, first by installing the latest `usbipd utility msi <https://github.com/dorssel/usbipd-win/releases>`_ (this guide assumes you have usbipd-win 4.0.0 or higher), then opening PowerShell in administrator mode and running:

.. code-block:: bash

    # (unplug RTL-SDR)
    usbipd list
    # (plug in RTL-SDR)
    usbipd list
    # (find the new device and substitute its index in the command below)
    usbipd bind --busid 1-5
    usbipd attach --wsl --busid 1-5

On the WSL side, you should be able to run :code:`lsusb` and see a new item called RTL2838 DVB-T or something similar.

If you run into permissions issues (e.g., the test below only works when using :code:`sudo`), you will need to setup udev rules.  First run :code:`lsusb` to find the ID of the RTL-SDR, then create the file :code:`/etc/udev/rules.d/10-rtl-sdr.rules` with the following content, substituting the idVendor and idProduct of your RTL-SDR if yours is different:

.. code-block::

 SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2838", MODE="0666"

To refresh udev, run:

.. code-block:: bash

    sudo udevadm control --reload-rules
    sudo udevadm trigger

You may also need to unplug and replug the RTL-SDR (for WSL you will have to :code:`usbipd attach` again). 

Windows
###################

For Windows users, see https://www.rtl-sdr.com/rtl-sdr-quick-start-guide/.  

********************************
Testing the RTL-SDR
********************************

If the software setup worked, you should be able to run the following test, which will tune the RTL-SDR to the FM radio band and record 1 million samples to a file called recording.iq in /tmp.

.. code-block:: bash

    rtl_sdr /tmp/recording.iq -s 2e6 -f 100e6 -n 1e6

If you get :code:`No supported devices found`, even when adding a :code:`sudo` to the beginning, then linux is unable to see the RTL-SDR at all.  If it works with :code:`sudo`, then it's a udev rules problem, try restarting the computer after going through the udev setup instructions above.  Alternatively, you can just use :code:`sudo` for everything, including running Python.

You can test out Python's ability to see the RTL-SDR using the following script:

.. code-block:: python

 from rtlsdr import RtlSdr

 sdr = RtlSdr()
 sdr.sample_rate = 2.048e6 # Hz
 sdr.center_freq = 100e6   # Hz
 sdr.freq_correction = 60  # PPM
 sdr.gain = 'auto'
 
 print(len(sdr.read_samples(1024)))
 sdr.close()

which should output:

.. code-block:: bash

 Found Rafael Micro R820T tuner
 [R82XX] PLL not locked!
 1024

********************************
RTL-SDR Python Code
********************************

The code above can be considered a basic usage example of the RTL-SDR in Python.  The following sections will go into more detail on the various settings and usage tricks.

Avoiding RTL-SDR Glitching
###############################

At the end of our script, or whenever we are done grabbing samples off the RTL-SDR, we will call :code:`sdr.close()`, which will help prevent the RTL-SDR from going into a glitched out state where it needs to be unplugged/replugged.  Even using close() it can still happen, you will know it if the RTL-SDR stalls during the read_samples() call.  If this happens, you will need to unplug and replug the RTL-SDR, and possibly restart your computer.  If you are using WSL, you will need to reattach the RTL-SDR using usbipd.

Gain Setting
#############

By setting :code:`sdr.gain = 'auto'` we are enabling automatic gain control (AGC), which will cause the RTL-SDR to adjust the receive gain based on the signals it receives, attempting to fill out the 8-bit ADC without saturating it.  For a lot of situations, such as making a spectrum analyzer, it is useful to keep the gain at a constant value, meaning we have to set a manual gain.  The RTL-SDR does not have an infinitely adjustable gain; you can see the list of valid gain values using :code:`print(sdr.valid_gains_db)`.  That being said, if you set it to a gain not on this list, it will autmoatically pick the closest allowable value.  You can always check what the current gain is set to with :code:`print(sdr.gain)`.  In the example below, we set the gain to a 49.6 dB and receive 4096 samples, then plot them in the time domain:

.. code-block:: python

 from rtlsdr import RtlSdr
 import numpy as np
 import matplotlib.pyplot as plt
 
 sdr = RtlSdr()
 sdr.sample_rate = 2.048e6 # Hz
 sdr.center_freq = 100e6   # Hz
 sdr.freq_correction = 60  # PPM
 print(sdr.valid_gains_db)
 sdr.gain = 49.6
 print(sdr.gain)
 
 x = sdr.read_samples(4096)
 sdr.close()
 
 plt.plot(x.real)
 plt.plot(x.imag)
 plt.legend(["I", "Q"])
 plt.savefig("../_images/rtlsdr-gain.svg", bbox_inches='tight')
 plt.show()

.. image:: ../_images/rtlsdr-gain.svg
   :align: center 
   :target: ../_images/rtlsdr-gain.svg
   :alt: RTL-SDR manual gain example

There are a couple things to note here.  The first ~2k samples do not seem to have much signal power in them, because they represent transients.  It is recommended to throw away the first 2k samples each script, e.g., using :code:`sdr.read_samples(2048)` and not doing anything with the output.  The other thing we notice is that pyrtlsdr is returning the samples to us as floats, in between -1 and +1.  Even though it uses an 8-bit ADC and produces integer values, pyrtlsdr is dividing by 127.0 for our convinience.

Allowed Sample Rates
#####################

Most RTL-SDRs require the sample rate to be set either between 230-300 kHz, or between 900-3.2 MHz.  Note that the higher rates, especially above 2.4 MHz, may not get 100% of samples through the USB connection.  If you give it an unsupported sample rate, it will simply return with the error :code:`rtlsdr.rtlsdr.LibUSBError: Error code -22: Could not set sample rate to 899000 Hz`.  When setting an allowable sample rate, you will notice the console message showing the exact sample rate; this exact value can also be retrieved by calling :code:`sdr.sample_rate`.  Some applications may benefit from having a more exact value used in calculations.

As an exercise, we will set the sample rate to 2.4 MHz and create a spectrogram of the FM radio band:

.. code-block:: python

 # ...
 sdr.sample_rate = 2.4e6 # Hz
 # ...
 
 fft_size = 512
 num_rows = 500
 x = sdr.read_samples(2048) # get rid of initial empty samples
 x = sdr.read_samples(fft_size*num_rows) # get all the samples we need for the spectrogram
 spectrogram = np.zeros((num_rows, fft_size))
 for i in range(num_rows):
     spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)
 extent = [(sdr.center_freq + sdr.sample_rate/-2)/1e6,
             (sdr.center_freq + sdr.sample_rate/2)/1e6,
             len(x)/sdr.sample_rate, 0]
 plt.imshow(spectrogram, aspect='auto', extent=extent)
 plt.xlabel("Frequency [MHz]")
 plt.ylabel("Time [s]")
 plt.show()

.. image:: ../_images/rtlsdr-waterfall.svg
   :align: center 
   :target: ../_images/rtlsdr-waterfall.svg
   :alt: RTL-SDR waterfall (aka spectrogram) example

PPM Setting
############

For those curious about the ppm setting, every RTL-SDR has a small frequency offset/error, due to the low-cost nature of the tuner chips and lack of calibration.  The frequency offset should be relatively linear (not a constant frequency shift) across the spectrum, so we can correct for it by entering a PPM value in parts per million.  For example, if you tune to 100 MHz and set the PPM to 25, it will shift the received signal up by 100e6/1e6*25=2500 Hz.  Narrower signals will have a greater impact from frequency error.  That being said, many modern signals involve a frequency synchronization step that will correct for any frequency offset on the transmitter, receiver, or due to Doppler shift.

********************************
Further Reading
********************************

#. `RTL-SDR.com's About Page <https://www.rtl-sdr.com/about-rtl-sdr/>`_
#. https://hackaday.com/2019/07/31/rtl-sdr-seven-years-later/
#. https://osmocom.org/projects/rtl-sdr/wiki/Rtl-sdr
