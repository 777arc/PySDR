.. _hackrf-chapter:

####################
HackRF One in Python
####################

Das `HackRF One <https://greatscottgadgets.com/hackrf/one/>`_ von Great Scott Gadgets ist ein USB 2.0 SDR, das von 1 MHz bis 6 GHz senden oder empfangen kann und eine Abtastrate von 2 bis 20 MHz hat. Es wurde 2014 veröffentlicht und hat im Laufe der Jahre einige kleinere Verbesserungen erfahren. Es ist eines der wenigen kostengünstigen senderfähigen SDRs, das bis auf 1 MHz hinuntergeht, was es ideal für HF-Anwendungen (z.B. Amateurfunk) zusätzlich zu Spaß bei höheren Frequenzen macht. Die maximale Sendeleistung von 15 dBm ist ebenfalls höher als bei den meisten anderen SDRs; vollständige Sendeleistungsspezifikationen findest du `auf dieser Seite <https://hackrf.readthedocs.io/en/latest/faq.html#what-is-the-transmit-power-of-hackrf>`_. Es verwendet Halbduplexbetrieb, d.h. es befindet sich zu einem bestimmten Zeitpunkt entweder im Sende- oder Empfangsmodus, und es verwendet einen 8-Bit-ADC/DAC.

.. image:: ../_images/hackrf1.jpeg
   :scale: 60 %
   :align: center
   :alt: HackRF One

********************************
HackRF-Architektur
********************************

Das HackRF basiert auf dem Analog Devices MAX2839-Chip, einem 2,3-GHz-bis-2,7-GHz-Transceiver, der ursprünglich für WiMAX entwickelt wurde, in Kombination mit einem MAX5864-HF-Frontend-Chip (im Wesentlichen nur der ADC und DAC) und einem RFFC5072-Breitband-Synthesizer/VCO (zur Auf- und Abwärtskonvertierung des Signals in der Frequenz). Dies steht im Gegensatz zu den meisten anderen kostengünstigen SDRs, die einen einzelnen Chip namens RFIC verwenden. Abgesehen von der Einstellung der im RFFC5072 erzeugten Frequenz werden alle anderen Parameter, die wir anpassen werden, wie Dämpfung und analoge Filterung, im MAX2839 vorgenommen. Anstatt ein FPGA oder System-on-Chip (SoC) wie viele SDRs zu verwenden, verwendet das HackRF ein Complex Programmable Logic Device (CPLD), das als einfache Verbindungslogik fungiert, und einen Mikrocontroller, den ARM-basierten LPC4320, der alle eingebetteten DSP-Aufgaben und die USB-Schnittstelle mit dem Host übernimmt (sowohl die Übertragung von IQ-Samples in beide Richtungen als auch die Steuerung der SDR-Einstellungen). Das folgende schöne Blockdiagramm von Great Scott Gadgets zeigt die Architektur der neuesten Revision des HackRF One:

.. image:: ../_images/hackrf_block_diagram.webp
   :align: center
   :alt: HackRF One Block Diagram
   :target: ../_images/hackrf_block_diagram.webp

Das HackRF One ist sehr erweiterbar und hackbar. Im Inneren des Gehäuses befinden sich vier Header (P9, P20, P22 und P28); Details dazu findest du `hier <https://hackrf.readthedocs.io/en/latest/expansion_interface.html>`_. Beachte, dass sich 8 GPIO-Pins und 4 ADC-Eingänge auf dem P20-Header befinden, während SPI, I2C und UART auf dem P22-Header sind. Der P28-Header kann verwendet werden, um Sende-/Empfangsoperationen mit einem anderen Gerät (z.B. TR-Schalter, externer Verstärker oder ein anderes HackRF) über den Trigger-Eingang und -Ausgang zu triggern/synchronisieren, mit einer Verzögerung von weniger als einer Abtastperiode.

.. image:: ../_images/hackrf2.jpeg
   :scale: 50 %
   :align: center
   :alt: HackRF One PCB

Der Takt, der sowohl für den LO als auch für den ADC/DAC verwendet wird, wird entweder vom internen 25-MHz-Oszillator oder von einer externen 10-MHz-Referenz abgeleitet, die über SMA eingespeist wird. Unabhängig davon, welcher Takt verwendet wird, erzeugt das HackRF ein 10-MHz-Taktsignal auf CLKOUT; ein Standard-3,3-V-10-MHz-Rechtecksignal für eine hochohmige Last. Der CLKIN-Anschluss ist dafür ausgelegt, ein ähnliches 10-MHz-3,3-V-Rechtecksignal aufzunehmen, und das HackRF One verwendet den Eingangsclk anstelle des internen Quarzes, wenn ein Taktsignal erkannt wird (die Umschaltung zu oder von CLKIN erfolgt nur, wenn ein Sende- oder Empfangsvorgang beginnt).

********************************
Software- und Hardware-Setup
********************************

Der Software-Installationsprozess umfasst zwei Schritte: Zuerst installieren wir die HackRF-Hauptbibliothek von Great Scott Gadgets und dann installieren wir die Python-API.

HackRF-Bibliothek installieren
###############################

Folgendes wurde auf Ubuntu 22.04 getestet (unter Verwendung von Commit-Hash 17f3943 im März '25):

.. code-block:: bash

    git clone https://github.com/greatscottgadgets/hackrf.git
    cd hackrf
    git checkout 17f3943
    cd host
    mkdir build
    cd build
    cmake ..
    make
    sudo make install
    sudo ldconfig
    sudo cp /usr/local/bin/hackrf* /usr/bin/.

Nach der Installation von :code:`hackrf` kannst du folgende Hilfsprogramme ausführen:

* :code:`hackrf_info` - Geräteinformationen vom HackRF lesen, wie Seriennummer und Firmware-Version.
* :code:`hackrf_transfer` - Signale mit dem HackRF senden und empfangen. Eingabe-/Ausgabedateien sind vorzeichenbehaftete 8-Bit-Quadratur-Samples.
* :code:`hackrf_sweep` - Ein Kommandozeilen-Spektrumanalysator.
* :code:`hackrf_clock` - Takteingangs- und -ausgangskonfiguration lesen und schreiben.
* :code:`hackrf_operacake` - Opera Cake Antennen-Switch, der an HackRF angeschlossen ist, konfigurieren.
* :code:`hackrf_spiflash` - Ein Werkzeug zum Schreiben neuer Firmware auf das HackRF. Siehe: Firmware aktualisieren.
* :code:`hackrf_debug` - Register und andere Low-Level-Konfiguration für das Debugging lesen und schreiben.

Wenn du Ubuntu über WSL verwendest, musst du auf der Windows-Seite das HackRF USB-Gerät an WSL weiterleiten, zuerst durch die Installation des neuesten `usbipd-Utility-msi <https://github.com/dorssel/usbipd-win/releases>`_ (diese Anleitung setzt voraus, dass du usbipd-win 4.0.0 oder höher hast), dann PowerShell im Administratormodus öffnen und folgendes ausführen:

.. code-block:: bash

    usbipd list
    <find the BUSID labeled HackRF One and substitute it in the two commands below>
    usbipd bind --busid 1-10
    usbipd attach --wsl --busid 1-10

Auf der WSL-Seite solltest du :code:`lsusb` ausführen und ein neues Element namens :code:`Great Scott Gadgets HackRF One` sehen können. Beachte, dass du das Flag :code:`--auto-attach` zum :code:`usbipd attach`-Befehl hinzufügen kannst, wenn es automatisch neu verbinden soll. Zuletzt musst du die udev-Regeln mit folgendem Befehl hinzufügen:

.. code-block:: bash

    echo 'ATTR{idVendor}=="1d50", ATTR{idProduct}=="6089", SYMLINK+="hackrf-one-%k", MODE="660", TAG+="uaccess"' | sudo tee /etc/udev/rules.d/53-hackrf.rules
    sudo udevadm trigger

Dann das HackRF One ausstecken und wieder einstecken (und den :code:`usbipd attach`-Teil wiederholen). Ich hatte Berechtigungsprobleme mit dem folgenden Schritt, bis ich auf der Windows-Seite `WSL USB Manager <https://gitlab.com/alelec/wsl-usb-gui/-/releases>`_ für die Weiterleitung zu WSL verwendete, der offenbar auch die udev-Regeln behandelt.

Ob du auf nativem Linux oder WSL bist, an diesem Punkt solltest du :code:`hackrf_info` ausführen und etwas wie folgendes sehen können:

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

Lass uns auch eine IQ-Aufzeichnung des UKW-Bands machen, 10 MHz breit zentriert bei 100 MHz, und wir nehmen 1 Million Samples:

.. code-block:: bash

    hackrf_transfer -r out.iq -f 100000000 -s 10000000 -n 1000000 -a 0 -l 30 -g 50

Dieses Hilfsprogramm erzeugt eine binäre IQ-Datei mit int8-Samples (2 Bytes pro IQ-Sample), die in unserem Fall 2 MB groß sein sollte. Falls du neugierig bist, kann die Signalaufzeichnung in Python mit folgendem Code gelesen werden:

.. code-block:: python

    import numpy as np
    samples = np.fromfile('out.iq', dtype=np.int8)
    samples = samples[::2] + 1j * samples[1::2]
    print(len(samples))
    print(samples[0:10])
    print(np.max(samples))

Wenn dein Maximum 127 ist (was bedeutet, dass du den ADC gesättigt hast), dann senke die beiden Verstärkungswerte am Ende des Befehls.

Python-API installieren
########################

Zuletzt müssen wir die HackRF One `Python-Bindings <https://github.com/GvozdevLeonid/python_hackrf>`_ installieren, die von `GvozdevLeonid <https://github.com/GvozdevLeonid>`_ gepflegt werden. Dies wurde auf Ubuntu 22.04 am 11.04.2024 mit dem neuesten Main-Branch getestet.

.. code-block:: bash

    sudo apt install libusb-1.0-0-dev
    pip install python_hackrf==1.2.7

Wir können die obige Installation testen, indem wir folgenden Code ausführen. Wenn es keine Fehler gibt (es wird auch keine Ausgabe geben), sollte alles einsatzbereit sein!

.. code-block:: python

    from python_hackrf import pyhackrf  # type: ignore
    pyhackrf.pyhackrf_init()
    sdr = pyhackrf.pyhackrf_open()
    sdr.pyhackrf_set_sample_rate(10e6)
    sdr.pyhackrf_set_antenna_enable(False)
    sdr.pyhackrf_set_freq(100e6)
    sdr.pyhackrf_set_amp_enable(False)
    sdr.pyhackrf_set_lna_gain(30) # LNA-Verstärkung - 0 bis 40 dB in 8-dB-Schritten
    sdr.pyhackrf_set_vga_gain(50) # VGA-Verstärkung - 0 bis 62 dB in 2-dB-Schritten
    sdr.pyhackrf_close()

Für einen tatsächlichen Test des Empfangens von Samples siehe den Beispielcode unten.

********************************
Sende- und Empfangsverstärkung
********************************

Empfangsseite
#############

Das HackRF One hat auf der Empfangsseite drei verschiedene Verstärkungsstufen:

* HF (:code:`amp`, entweder 0 oder 11 dB)
* ZF (:code:`lna`, 0 bis 40 dB in 8-dB-Schritten)
* Basisband (:code:`vga`, 0 bis 62 dB in 2-dB-Schritten)

Für den Empfang der meisten Signale wird empfohlen, den HF-Verstärker ausgeschaltet (0 dB) zu lassen, es sei denn, du hast es mit einem extrem schwachen Signal zu tun und es gibt definitiv keine starken Signale in der Nähe. Die ZF-(LNA-)Verstärkung ist die wichtigste Verstärkungsstufe, die angepasst werden sollte, um deinen SNR zu maximieren und gleichzeitig eine Sättigung des ADC zu vermeiden – das ist der erste Regler, den du anpassen solltest. Die Basisbandverstärkung kann auf einem relativ hohen Wert belassen werden, z.B. lassen wir sie einfach bei 50 dB.

Sendeseite
##########

Auf der Sendeseite gibt es zwei Verstärkungsstufen:

* HF [entweder 0 oder 11 dB]
* ZF [0 bis 47 dB in 1-dB-Schritten]

Du wirst wahrscheinlich den HF-Verstärker aktivieren wollen, und dann kannst du die ZF-Verstärkung nach deinen Bedürfnissen anpassen.

**************************************************
IQ-Samples mit dem HackRF in Python empfangen
**************************************************

Derzeit enthält das :code:`python_hackrf`-Python-Paket keine Komfortfunktionen zum Empfangen von Samples; es ist lediglich eine Reihe von Python-Bindings, die auf die C++-API des HackRF abgebildet werden. Das bedeutet, dass wir für den Empfang von IQ eine beträchtliche Menge Code verwenden müssen. Das Python-Paket ist so eingerichtet, dass es eine Callback-Funktion zum Empfang weiterer Samples verwendet – das ist eine Funktion, die wir einrichten müssen, die aber automatisch aufgerufen wird, wenn mehr Samples vom HackRF bereit sind. Diese Callback-Funktion muss immer drei spezifische Argumente haben und :code:`0` zurückgeben, wenn wir einen weiteren Satz von Samples wollen. Im folgenden Code konvertieren wir innerhalb jedes Aufrufs unserer Callback-Funktion die Samples in NumPys komplexen Typ, skalieren sie von -1 bis +1 und speichern sie dann in einem größeren :code:`samples`-Array.

Wenn in deinem Zeitdiagramm die Samples die ADC-Grenzen von -1 und +1 erreichen, reduziere :code:`lna_gain` um 3 dB, bis es klar nicht mehr die Grenzen erreicht.

.. code-block:: python

    from python_hackrf import pyhackrf  # type: ignore
    import matplotlib.pyplot as plt
    import numpy as np
    import time

    # Diese Einstellungen sollten mit dem hackrf_transfer-Beispiel im Lehrbuch übereinstimmen, und der resultierende Wasserfall sollte ungefähr gleich aussehen
    recording_time = 1  # Sekunden
    center_freq = 100e6  # Hz
    sample_rate = 10e6
    baseband_filter = 7.5e6
    lna_gain = 30 # 0 bis 40 dB in 8-dB-Schritten
    vga_gain = 50 # 0 bis 62 dB in 2-dB-Schritten

    pyhackrf.pyhackrf_init()
    sdr = pyhackrf.pyhackrf_open()

    allowed_baseband_filter = pyhackrf.pyhackrf_compute_baseband_filter_bw_round_down_lt(baseband_filter) # unterstützte Bandbreite relativ zur gewünschten berechnen

    sdr.pyhackrf_set_sample_rate(sample_rate)
    sdr.pyhackrf_set_baseband_filter_bandwidth(allowed_baseband_filter)
    sdr.pyhackrf_set_antenna_enable(False)  # Scheint die Stromversorgung des Antennenanschlusses zu aktivieren/deaktivieren. Standardmäßig False. Die Firmware deaktiviert dies automatisch nach Rückkehr in den IDLE-Modus

    sdr.pyhackrf_set_freq(center_freq)
    sdr.pyhackrf_set_amp_enable(False)  # Standardmäßig False
    sdr.pyhackrf_set_lna_gain(lna_gain)  # LNA-Verstärkung - 0 bis 40 dB in 8-dB-Schritten
    sdr.pyhackrf_set_vga_gain(vga_gain)  # VGA-Verstärkung - 0 bis 62 dB in 2-dB-Schritten

    print(f'center_freq: {center_freq} sample_rate: {sample_rate} baseband_filter: {allowed_baseband_filter}')

    num_samples = int(recording_time * sample_rate)
    samples = np.zeros(num_samples, dtype=np.complex64)
    last_idx = 0

    def rx_callback(device, buffer, buffer_length, valid_length):  # Diese Callback-Funktion muss immer diese vier Argumente haben
        global samples, last_idx

        accepted = valid_length // 2
        accepted_samples = buffer[:valid_length].astype(np.int8) # -128 bis 127
        accepted_samples = accepted_samples[0::2] + 1j * accepted_samples[1::2]  # In komplexen Typ konvertieren (IQ entflechten)
        accepted_samples /= 128 # -1 bis +1
        samples[last_idx: last_idx + accepted] = accepted_samples

        last_idx += accepted

        return 0

    sdr.set_rx_callback(rx_callback)
    sdr.pyhackrf_start_rx()
    print('is_streaming', sdr.pyhackrf_is_streaming())

    time.sleep(recording_time)

    sdr.pyhackrf_stop_rx()
    sdr.pyhackrf_close()
    pyhackrf.pyhackrf_exit()

    samples = samples[100000:] # Die ersten 100k Samples verwerfen, nur zur Sicherheit wegen Transienten

    fft_size = 2048
    num_rows = len(samples) // fft_size
    spectrogram = np.zeros((num_rows, fft_size))
    for i in range(num_rows):
        spectrogram[i, :] = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples[i * fft_size:(i+1) * fft_size]))) ** 2)
    extent = [(center_freq + sample_rate / -2) / 1e6, (center_freq + sample_rate / 2) / 1e6, len(samples) / sample_rate, 0]

    plt.figure(0)
    plt.imshow(spectrogram, aspect='auto', extent=extent) # type: ignore
    plt.xlabel("Frequenz [MHz]")
    plt.ylabel("Zeit [s]")

    plt.figure(1)
    plt.plot(np.real(samples[0:10000]))
    plt.plot(np.imag(samples[0:10000]))
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend(["Real", "Imaginär"])

    plt.show()

Wenn du eine Antenne verwendest, die das UKW-Band empfangen kann, solltest du etwas wie folgendes erhalten, mit mehreren UKW-Sendern, die im Wasserfalldiagramm sichtbar sind:

.. image:: ../_images/hackrf_time_screenshot.png
   :align: center
   :scale: 50 %
   :alt: Time plot of the samples grabbed from HackRF

.. image:: ../_images/hackrf_freq_screenshot.png
   :align: center
   :scale: 50 %
   :alt: Spectrogram (frequency over time) plot of the samples grabbed from HackRF
