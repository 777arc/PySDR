.. _hackrf-chapter:

####################
HackRF One in Python
####################

The `HackRF One <https://greatscottgadgets.com/hackrf/one/>`_ from Great Scott Gadgets is a USB 2.0 SDR of transmission or reception from 1 MHz to 6 GHz and a sample rate from 2 to 20 MHz.  It is one of the only low-cost transmit-capable SDRs that goes down to 1 MHz, making it great for HF application (e.g., ham radio) in addition to higher frequency fun.  The max transmit power of 15 dBm is also higher than most other SDRs, see `this page <https://hackrf.readthedocs.io/en/latest/faq.html#what-is-the-transmit-power-of-hackrf>`_ for full transmit power specs.  It uses half-duplex operation, meaning it is either in transmit or receive mode at any given time, and it uses 8-bit ADC/DAC.

.. image:: ../_images/hackrf1.jpeg
   :scale: 60 %
   :align: center 
   :alt: HackRF One

********************************
HackRF Architecture
********************************

asdasdsad

.. image:: ../_images/hackrf_block_diagram.webp
   :align: center 
   :alt: HackRF One Block Diagram
   :target: ../_images/hackrf_block_diagram.webp

The HackRF One is highly expandable and hackable.  Inside the plastic case are four headers (P9, P20, P22, and P28), specifics can be `found here <https://hackrf.readthedocs.io/en/latest/expansion_interface.html>`_, but note that 8 GPIO pins and 4 ADC inputs are on the P20 header, while SPI, I2C, and UART are on the P22 header.  The P28 header can be used to trigger/synchronize transmit/receive operations with another device (e.g., TR-switch, external amp, or another HackRF), through the trigger input and output, with delay of less than one sample period.

.. image:: ../_images/hackrf2.jpeg
   :scale: 50 %
   :align: center 
   :alt: HackRF One PCB

HackRF One produces a 10 MHz clock signal on CLKOUT; a standard 3.3V 10 MHz square wave intended for a high impedance load.  The CLKIN port is designed to take a similar 10 MHz 3.3V square wave, and the HackRF One will use the input clock instead of the internal crystal when a clock signal is detected (note, the transition to or from CLKIN only happens when a transmit or receive operation begins).  

********************************
Software and Hardware Setup
********************************

The following was tested to work on Ubuntu 22.04 (using hackrf hash 17f3943 in Sept '24):

.. code-block:: bash

    git clone https://github.com/greatscottgadgets/hackrf.git
    cd hackrf/host
    mkdir build
    cd build
    cmake ..
    make
    sudo make install
    sudo ldconfig
    sudo cp /usr/local/bin/hackrf* /usr/bin/.

After installing hackrf you will be able to run the following utilities:

* :code:`hackrf_info` - Read device information from HackRF such as serial number and firmware version.
* :code:`hackrf_transfer` - Send and receive signals using HackRF. Input/output files are 8-bit signed quadrature samples.
* :code:`hackrf_sweep` - a command-line spectrum analyzer.
* :code:`hackrf_clock` - Read and write clock input and output configuration.
* :code:`hackrf_operacake` - Configure Opera Cake antenna switch connected to HackRF.
* :code:`hackrf_spiflash` - A tool to write new firmware to HackRF. See: Updating Firmware.
* :code:`hackrf_debug` - Read and write registers and other low-level configuration for debugging.

If you are using Ubuntu through WSL, on the Windows side you will need to forward the bladeRF USB device to WSL, first by installing the latest `usbipd utility msi <https://github.com/dorssel/usbipd-win/releases>`_ (this guide assumes you have usbipd-win 4.0.0 or higher), then opening PowerShell in administrator mode and running:

.. code-block:: bash

    usbipd list
    <find the BUSID labeled HackRF One and substitute it in the two commands below>
    usbipd bind --busid 1-10
    usbipd attach --wsl --busid 1-10

On the WSL side, you should be able to run :code:`lsusb` and see a new item called :code:`Great Scott Gadgets HackRF One`.  Note that you can add the :code:`--auto-attach` flag to the :code:`usbipd attach` command if you want it to auto reconnect.  Lastly, you have to add the udev rules using the following command:

.. code-block:: bash

    echo 'ATTR{idVendor}=="1d50", ATTR{idProduct}=="6089", SYMLINK+="hackrf-one-%k", MODE="660", TAG+="uaccess"' | sudo tee /etc/udev/rules.d/53-hackrf.rules
    sudo udevadm trigger

Then unplug and replug your HackRF One (and redo the :code:`usbipd attach` part).  Note, I had permissions issues with the step below until I switched to using `WSL USB Manager <https://gitlab.com/alelec/wsl-usb-gui/-/releases>`_ on the Windows side, to manage forwarding to WSL, which apparently also deals with the udev rules.

Whether you're on native Linux or WSL, at this point you should be able to run :code:`hackrf_info` and see something like:

.. code-block:: bash

    hackrf_info version: git-17f39433
    libhackrf version: git-17f39433 (0.9)
    Found HackRF
    Index: 0
    Serial number: 00000000000000007687865765a765
    Board ID Number: 2 (HackRF One)
    Firmware Version: 2024.02.1 (API:1.08)
    Part ID Number: 0xa000cb3c 0x004f4762
    Hardware Revision: r10
    Hardware appears to have been manufactured by Great Scott Gadgets.
    Hardware supported by installed firmware: HackRF One

Let's also make an IQ recording of the FM band, 10 MHz wide centered at 100 MHz, and we'll grab 1 million samples:

.. code-block:: bash

    hackrf_transfer -r out.iq -f 100000000 -s 10000000 -n 1000000 -a 0 -l 30 -g 50

This utility produces a binary IQ file of int8 samples (2 bytes per IQ sample), which in our case should be 2MB.  If you're curious, the signal recording can be read in Python using the following code:

.. code-block:: python

    import numpy as np
    samples = np.fromfile('out.iq', dtype=np.int8)
    samples = samples[::2] + 1j * samples[1::2]
    print(len(samples))
    print(samples[0:10])
    print(np.max(samples))

If your max is 127 (which means you saturated the ADC) then lower the two gain values at the end of the command.

Lastly, we must install the HackRF One `Python bindings <https://github.com/GvozdevLeonid/python_hackrf>`_, maintained by `GvozdevLeonid <https://github.com/GvozdevLeonid>`_, using:

.. code-block:: bash

    sudo apt install libusb-1.0-0-dev
    cd ~
    git clone https://github.com/GvozdevLeonid/python_hackrf.git
    cd python_hackrf
    export LDFLAGS="-L/usr/lib/x86_64-linux-gnu -L/usr/local/lib"
    export CFLAGS="-I/usr/include/libusb-1.0 -I/usr/local/include/libhackrf"
    python setup.py build_ext --inplace
    pip install -e .

We can test the above install using:

.. code-block:: python




********************************
Tx and Rx Gain
********************************

The HackRF One on the receive side has three different gain stages:

* RF ("amp", either 0 or 11 dB)
* IF ("lna", 0 to 40 dB in 8 dB steps)
* baseband ("vga", 0 to 62 dB in 2 dB steps)

For most signals it is recommended to leave the RF amplifier off (0 dB), unless you are dealing with an extremely weak signal and there are definitely no strong signals nearby.  The IF gain is the most important gain stage to adjust, to maximize your SNR while avoiding saturation of the ADC.

On the transmit side, there are two gain stages:

* RF [either 0 or 11 dB]
* IF [0 to 47 dB in 1 dB steps]


