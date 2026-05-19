.. _bladerf-chapter:

##################
BladeRF in Python
##################

Das bladeRF 2.0 (auch bekannt als bladeRF 2.0 micro) des Unternehmens `Nuand <https://www.nuand.com>`_ ist ein USB-3.0-basiertes SDR mit zwei Empfangskanälen, zwei Sendekanälen, einem abstimmbaren Bereich von 47 MHz bis 6 GHz und der Möglichkeit, mit bis zu 61 MHz oder sogar 122 MHz abzutasten, wenn man es entsprechend modifiziert. Es verwendet denselben AD9361 HF-integrierten Schaltkreis (RFIC) wie das USRP B210 und das PlutoSDR, sodass die HF-Leistung vergleichbar ist. Das bladeRF 2.0 wurde 2021 veröffentlicht, hat einen kompakten Formfaktor von 2,5" x 4,5" und ist in zwei verschiedenen FPGA-Größen erhältlich (xA4 und xA9). Obwohl sich dieses Kapitel auf das bladeRF 2.0 konzentriert, gilt ein Großteil des Codes auch für das ursprüngliche bladeRF, das `2013 erschien <https://www.kickstarter.com/projects/1085541682/bladerf-usb-30-software-defined-radio>`_.

.. image:: ../_images/bladeRF_micro.png
   :scale: 35 %
   :align: center
   :alt: bladeRF 2.0 Produktfoto

********************************
bladeRF-Architektur
********************************

Das bladeRF 2.0 basiert auf dem AD9361 RFIC, kombiniert mit einem Cyclone-V-FPGA (entweder dem 49 kLE :code:`5CEA4` oder dem 301 kLE :code:`5CEA9`), sowie einem Cypress FX3 USB-3.0-Controller mit einem 200 MHz ARM9-Kern, der mit einer angepassten Firmware bespielt ist. Das Blockdiagramm des bladeRF 2.0 ist unten dargestellt:

.. image:: ../_images/bladeRF-2.0-micro-Block-Diagram-4.png
   :scale: 80 %
   :align: center
   :alt: bladeRF 2.0 Blockdiagramm

Das FPGA steuert den RFIC, führt digitale Filterung durch und verpackt Pakete für die Übertragung über USB (unter anderem). Der `Quellcode <https://github.com/Nuand/bladeRF/tree/master/hdl>`_ für das FPGA-Image ist in VHDL geschrieben und erfordert die kostenlose Quartus Prime Lite Design-Software, um benutzerdefinierte Images zu kompilieren. Vorkompilierte Images sind `hier <https://www.nuand.com/fpga_images/>`_ verfügbar.

Der `Quellcode <https://github.com/Nuand/bladeRF/tree/master/fx3_firmware>`_ für die Cypress FX3 Firmware ist Open-Source und enthält Code zum:

1. Laden des FPGA-Images
2. Übertragen von IQ-Samples zwischen dem FPGA und dem Host über USB 3.0
3. Steuern der GPIO des FPGAs über UART

Aus der Perspektive des Signalflusses gibt es zwei Empfangskanäle und zwei Sendekanäle, wobei jeder Kanal je nach verwendetem Frequenzband einen Nieder- und Hochfrequenzeingang/-ausgang zum RFIC hat. Aus diesem Grund wird zwischen dem RFIC und den SMA-Anschlüssen ein elektronischer HF-Schalter mit einem Pol und zwei Ausgängen (SPDT) benötigt. Das Bias-T ist eine integrierte Schaltung auf der Platine, die etwa 4,5 V Gleichstrom am SMA-Anschluss bereitstellt und dazu dient, einen externen Verstärker oder andere HF-Komponenten bequem zu versorgen. Dieser zusätzliche Gleichstromoffset befindet sich auf der HF-Seite des SDR und stört daher den grundlegenden Sende-/Empfangsbetrieb nicht.

JTAG ist eine Art Debug-Schnittstelle, die das Testen und Verifizieren von Designs während des Entwicklungsprozesses ermöglicht.

Am Ende dieses Kapitels besprechen wir den VCTCXO-Oszillator, den PLL und den Erweiterungsport.

********************************
Software- und Hardware-Einrichtung
********************************

Ubuntu (oder Ubuntu innerhalb von WSL)
#######################################

Unter Ubuntu und anderen Debian-basierten Systemen kannst du die bladeRF-Software mit folgenden Befehlen installieren:

.. code-block:: bash

 sudo apt update
 sudo apt install cmake python3-pip libusb-1.0-0
 cd ~
 git clone --depth 1 https://github.com/Nuand/bladeRF.git
 cd bladeRF/host
 mkdir build && cd build
 cmake ..
 make -j8
 sudo make install
 sudo ldconfig
 cd ../libraries/libbladeRF_bindings/python
 sudo python3 setup.py install

Damit werden die libbladerf-Bibliothek, Python-Bindungen, bladeRF-Kommandozeilenwerkzeuge, der Firmware-Downloader und der FPGA-Bitstream-Downloader installiert. Um zu prüfen, welche Version der Bibliothek du installiert hast, verwende :code:`bladerf-tool version` (dieser Leitfaden wurde mit libbladeRF Version v2.5.0 geschrieben).

Wenn du Ubuntu über WSL verwendest, musst du auf der Windows-Seite das bladeRF-USB-Gerät an WSL weiterleiten. Installiere dazu zunächst das neueste `usbipd-Dienstprogramm als MSI <https://github.com/dorssel/usbipd-win/releases>`_ (dieser Leitfaden setzt usbipd-win 4.0.0 oder höher voraus), öffne dann PowerShell im Administratormodus und führe folgendes aus:

.. code-block:: bash

    usbipd list
    # (finde die BUSID mit der Bezeichnung bladeRF 2.0 und setze sie im folgenden Befehl ein)
    usbipd bind --busid 1-23
    usbipd attach --wsl --busid 1-23

Auf der WSL-Seite solltest du :code:`lsusb` ausführen und einen neuen Eintrag namens :code:`Nuand LLC bladeRF 2.0 micro` sehen können. Beachte, dass du das Flag :code:`--auto-attach` zum Befehl :code:`usbipd attach` hinzufügen kannst, wenn es sich automatisch neu verbinden soll.

(Möglicherweise nicht erforderlich) Sowohl für natives Linux als auch für WSL müssen wir die udev-Regeln installieren, damit wir keine Berechtigungsfehler erhalten:

.. code-block::

 sudo nano /etc/udev/rules.d/88-nuand.rules

und folgende Zeile einfügen:

.. code-block::

 ATTRS{idVendor}=="2cf0", ATTRS{idProduct}=="5250", MODE="0666"

Zum Speichern und Beenden von nano: Strg+O, dann Enter, dann Strg+X. Um udev neu zu laden, führe aus:

.. code-block:: bash

    sudo udevadm control --reload-rules && sudo udevadm trigger

Wenn du WSL verwendest und die Meldung :code:`Failed to send reload request: No such file or directory` erscheint, bedeutet das, dass der udev-Dienst nicht läuft. Du musst dann :code:`sudo nano /etc/wsl.conf` öffnen und folgende Zeilen hinzufügen:

.. code-block:: bash

 [boot]
 command="service udev start"

Starte dann WSL neu mit folgendem Befehl in PowerShell als Administrator: :code:`wsl.exe --shutdown`.

Trenne und verbinde dein bladeRF erneut (WSL-Nutzer müssen es erneut anhängen) und teste die Berechtigungen mit:

.. code-block:: bash

 bladerf-tool probe
 bladerf-tool info

Es hat funktioniert, wenn du dein bladeRF 2.0 aufgelistet siehst und **nicht** die Meldung :code:`Found a bladeRF via VID/PID, but could not open it due to insufficient permissions` erscheint. Wenn es geklappt hat, notiere die angezeigte FPGA-Version und Firmware-Version.

(Optional) Installiere die neueste Firmware und FPGA-Images (zum Zeitpunkt der Erstellung dieses Leitfadens v2.4.0 bzw. v0.15.0) mit:

.. code-block:: bash

 cd ~/Downloads
 wget https://www.nuand.com/fx3/bladeRF_fw_latest.img
 bladerf-tool flash_fw bladeRF_fw_latest.img

 # für xA4 verwende:
 wget https://www.nuand.com/fpga/hostedxA4-latest.rbf
 bladerf-tool flash_fpga hostedxA4-latest.rbf

 # für xA9 verwende:
 wget https://www.nuand.com/fpga/hostedxA9-latest.rbf
 bladerf-tool flash_fpga hostedxA9-latest.rbf

Trenne und verbinde dein bladeRF erneut, um es neu zu starten.

Nun testen wir die Funktionalität, indem wir 1 Million Samples im FM-Radioband bei 10 MHz Abtastrate in die Datei /tmp/samples.sc16 aufnehmen:

.. code-block:: bash

 bladerf-tool rx --num-samples 1000000 /tmp/samples.sc16 100e6 10e6

Ein paar :code:`Hit stall for buffer`-Meldungen sind zu erwarten, aber es hat funktioniert, wenn du eine 4 MB große Datei /tmp/samples.sc16 siehst.

Abschließend testen wir die Python-API mit:

.. code-block:: bash

 python3
 import bladerf
 bladerf.BladeRF()
 exit()

Es hat funktioniert, wenn du etwas wie :code:`<BladeRF(<DevInfo(...)>)>` und keine Warnungen oder Fehler siehst.

Windows und macOS
##################

Für Windows-Nutzer (die WSL nicht bevorzugen) siehe https://github.com/Nuand/bladeRF/wiki/Getting-Started%3A-Windows, und für macOS-Nutzer siehe https://github.com/Nuand/bladeRF/wiki/Getting-started:-Mac-OSX.

********************************
bladeRF Python API Grundlagen
********************************

Zunächst fragen wir das bladeRF nach einigen nützlichen Informationen ab, mit folgendem Skript. **Benenne dein Skript nicht bladerf.py**, da es sonst mit dem bladeRF Python-Modul selbst in Konflikt gerät!

.. code-block:: python

 from bladerf import _bladerf
 import numpy as np
 import matplotlib.pyplot as plt

 sdr = _bladerf.BladeRF()

 print("Device info:", _bladerf.get_device_list()[0])
 print("libbladeRF version:", _bladerf.version()) # v2.5.0
 print("Firmware version:", sdr.get_fw_version()) # v2.4.0
 print("FPGA version:", sdr.get_fpga_version())   # v0.15.0

 rx_ch = sdr.Channel(_bladerf.CHANNEL_RX(0)) # 0 oder 1 übergeben
 print("sample_rate_range:", rx_ch.sample_rate_range)
 print("bandwidth_range:", rx_ch.bandwidth_range)
 print("frequency_range:", rx_ch.frequency_range)
 print("gain_modes:", rx_ch.gain_modes)
 print("manual gain range:", sdr.get_gain_range(_bladerf.CHANNEL_RX(0))) # Kanal 0 oder 1

Für das bladeRF 2.0 xA9 sollte die Ausgabe in etwa so aussehen:

.. code-block:: python

    Device info: Device Information
        backend  libusb
        serial   f80a27b1010448dfb7a003ef7fa98a59
        usb_bus  2
        usb_addr 5
        instance 0
    libbladeRF version: v2.5.0 ("2.5.0-git-624994d")
    Firmware version: v2.4.0 ("2.4.0-git-a3d5c55f")
    FPGA version: v0.15.0 ("0.15.0")
    sample_rate_range: Range
        min   520834
        max   61440000
        step  2
        scale 1.0

    bandwidth_range: Range
        min   200000
        max   56000000
        step  1
        scale 1.0

    frequency_range: Range
        min   70000000
        max   6000000000
        step  2
        scale 1.0

    gain_modes: [<GainMode.Default: 0>, <GainMode.Manual: 1>, <GainMode.FastAttack_AGC: 2>, <GainMode.SlowAttack_AGC: 3>, <GainMode.Hybrid_AGC: 4>]

    manual gain range: Range
        min   -15
        max   60
        step  1
        scale 1.0

Der Bandwidth-Parameter legt den Filter fest, den das SDR beim Empfang verwendet. Daher setzen wir ihn typischerweise gleich oder leicht unterhalb von sample_rate/2. Die Gain-Modi sind wichtig zu verstehen: Das SDR verwendet entweder einen manuellen Verstärkungsmodus, bei dem du die Verstärkung in dB vorgibst, oder eine automatische Verstärkungsregelung (AGC), die drei verschiedene Einstellungen hat (schnell, langsam, hybrid). Für Anwendungen wie die Spektrumüberwachung wird ein manueller Gain empfohlen, damit du erkennen kannst, wann Signale auftauchen und verschwinden. Für Anwendungen, bei denen du ein bestimmtes Signal empfangen möchtest, das du erwartest, ist AGC nützlicher, da es den Gain automatisch anpasst, damit das Signal den Analog-Digital-Wandler (ADC) optimal ausnutzt.

Um die wichtigsten Parameter des SDR einzustellen, können wir folgenden Code hinzufügen:

.. code-block:: python

 sample_rate = 10e6
 center_freq = 100e6
 gain = 50 # -15 bis 60 dB
 num_samples = int(1e6)

 rx_ch.frequency = center_freq
 rx_ch.sample_rate = sample_rate
 rx_ch.bandwidth = sample_rate/2
 rx_ch.gain_mode = _bladerf.GainMode.Manual
 rx_ch.gain = gain

********************************
Samples empfangen in Python
********************************

Als nächstes bauen wir auf dem vorherigen Codeblock auf und empfangen 1 Million Samples im FM-Radioband bei 10 MHz Abtastrate – genauso wie zuvor. Jede Antenne am RX1-Port sollte FM empfangen können, da die Signale sehr stark sind. Der folgende Code zeigt, wie die synchrone Stream-API des bladeRF funktioniert: Sie muss konfiguriert und ein Empfangspuffer muss erstellt werden, bevor der Empfang beginnt. Die :code:`while True:`-Schleife empfängt so lange Samples, bis die angeforderte Anzahl erreicht ist. Die empfangenen Samples werden in einem separaten NumPy-Array gespeichert, damit wir sie nach der Schleife verarbeiten können.

.. code-block:: python

 # Synchronen Stream konfigurieren
 sdr.sync_config(layout = _bladerf.ChannelLayout.RX_X1, # oder RX_X2
                 fmt = _bladerf.Format.SC16_Q11, # int16s
                 num_buffers    = 16,
                 buffer_size    = 8192,
                 num_transfers  = 8,
                 stream_timeout = 3500)

 # Empfangspuffer erstellen
 bytes_per_sample = 4 # nicht ändern, es werden immer int16s verwendet
 buf = bytearray(1024 * bytes_per_sample)

 # Modul aktivieren
 print("Starte Empfang")
 rx_ch.enable = True

 # Empfangsschleife
 x = np.zeros(num_samples, dtype=np.complex64) # Speicher für IQ-Samples
 num_samples_read = 0
 while True:
     if num_samples > 0 and num_samples_read == num_samples:
         break
     elif num_samples > 0:
         num = min(len(buf) // bytes_per_sample, num_samples - num_samples_read)
     else:
         num = len(buf) // bytes_per_sample
     sdr.sync_rx(buf, num) # In Puffer einlesen
     samples = np.frombuffer(buf, dtype=np.int16)
     samples = samples[0::2] + 1j * samples[1::2] # In komplexen Typ umwandeln
     samples /= 2048.0 # Auf -1 bis 1 skalieren (12-Bit-ADC)
     x[num_samples_read:num_samples_read+num] = samples[0:num] # Puffer im Samples-Array speichern
     num_samples_read += num

 print("Stoppe")
 rx_ch.enable = False
 print(x[0:10]) # erste 10 IQ-Samples ansehen
 print(np.max(x)) # wenn dieser Wert nahe 1 ist, überlädst du den ADC und solltest den Gain reduzieren

Ein paar :code:`Hit stall for buffer`-Meldungen am Ende sind zu erwarten. Die letzte ausgegebene Zahl zeigt den maximalen empfangenen Samplewert. Du solltest deinen Gain so einstellen, dass dieser Wert ungefähr zwischen 0,5 und 0,8 liegt. Wenn er 0,999 beträgt, ist dein Empfänger überlastet/gesättigt und das Signal wird verzerrt (es erscheint im Frequenzbereich verschmiert).

Um das empfangene Signal zu visualisieren, zeigen wir die IQ-Samples als Spektrogramm an (siehe :ref:`spectrogram-section` für weitere Details zur Funktionsweise von Spektrogrammen). Füge folgendes am Ende des vorherigen Codeblocks hinzu:

.. code-block:: python

 # Spektrogramm erstellen
 fft_size = 2048
 num_rows = len(x) // fft_size # // ist eine ganzzahlige Division, die abrundet
 spectrogram = np.zeros((num_rows, fft_size))
 for i in range(num_rows):
     spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)
 extent = [(center_freq + sample_rate/-2)/1e6, (center_freq + sample_rate/2)/1e6, len(x)/sample_rate, 0]
 plt.imshow(spectrogram, aspect='auto', extent=extent)
 plt.xlabel("Frequenz [MHz]")
 plt.ylabel("Zeit [s]")
 plt.show()

.. image:: ../_images/bladerf-waterfall.svg
   :align: center
   :target: ../_images/bladerf-waterfall.svg
   :alt: bladeRF Spektrogramm-Beispiel

Jede vertikale gewellte Linie ist ein FM-Radiosignal. Was das pulsierende Signal auf der rechten Seite verursacht, ist unklar – eine Reduzierung des Gains ließ es nicht verschwinden.


********************************
Samples senden in Python
********************************

Das Senden von Samples mit dem bladeRF ist dem Empfangen sehr ähnlich. Der wesentliche Unterschied besteht darin, dass wir die zu sendenden Samples generieren und sie dann mit der Methode :code:`sync_tx` an das bladeRF schreiben müssen, die unseren gesamten Batch an Samples auf einmal verarbeiten kann (bis zu ca. 4 Milliarden Samples). Der folgende Code zeigt, wie man einen einfachen Ton sendet und ihn 30 Mal wiederholt. Der Ton wird mit NumPy generiert und dann auf den Bereich -2048 bis 2048 skaliert, um in den 12-Bit-Digital-Analog-Wandler (DAC) zu passen. Anschließend wird der Ton in Bytes umgewandelt, die int16-Werte repräsentieren, und als Sendepuffer verwendet. Die synchrone Stream-API wird zum Senden der Samples verwendet, und die :code:`while True:`-Schleife sendet so lange, bis die gewünschte Anzahl an Wiederholungen erreicht ist. Wenn du stattdessen Samples aus einer Datei senden möchtest, verwende :code:`samples = np.fromfile('deinedatei.iq', dtype=np.int16)` (oder welchen Datentyp sie auch haben), um die Samples einzulesen, und konvertiere sie dann mit :code:`samples.tobytes()` in Bytes. Beachte dabei den Wertebereich des DAC von -2048 bis 2048.

.. code-block:: python

 from bladerf import _bladerf
 import numpy as np

 sdr = _bladerf.BladeRF()
 tx_ch = sdr.Channel(_bladerf.CHANNEL_TX(0)) # 0 oder 1 übergeben

 sample_rate = 10e6
 center_freq = 100e6
 gain = 0 # -15 bis 60 dB. Beim Senden klein anfangen und langsam erhöhen; Antenne anschließen!
 num_samples = int(1e6)
 repeat = 30 # Anzahl der Wiederholungen des Signals
 print('Sendedauer:', num_samples/sample_rate*repeat, 'Sekunden')

 # IQ-Samples zum Senden generieren (hier ein einfacher Ton)
 t = np.arange(num_samples) / sample_rate
 f_tone = 1e6
 samples = np.exp(1j * 2 * np.pi * f_tone * t) # liegt zwischen -1 und +1
 samples = samples.astype(np.complex64)
 samples *= 2048.0 # Auf -2048 bis 2048 skalieren (12-Bit-DAC)
 samples = samples.view(np.int16)
 buf = samples.tobytes() # Samples in Bytes umwandeln und als Sendepuffer verwenden

 tx_ch.frequency = center_freq
 tx_ch.sample_rate = sample_rate
 tx_ch.bandwidth = sample_rate/2
 tx_ch.gain = gain

 # Synchronen Stream konfigurieren
 sdr.sync_config(layout=_bladerf.ChannelLayout.TX_X1, # oder TX_X2
                 fmt=_bladerf.Format.SC16_Q11, # int16s
                 num_buffers=16,
                 buffer_size=8192,
                 num_transfers=8,
                 stream_timeout=3500)

 print("Starte Senden!")
 repeats_remaining = repeat - 1
 tx_ch.enable = True
 while True:
     sdr.sync_tx(buf, num_samples) # an bladeRF schreiben
     print(repeats_remaining)
     if repeats_remaining > 0:
         repeats_remaining -= 1
     else:
         break

 print("Stoppe Senden")
 tx_ch.enable = False

Ein paar :code:`Hit stall for buffer`-Meldungen am Ende sind zu erwarten.

Um gleichzeitig zu senden und zu empfangen, müssen Threads verwendet werden. Am besten nutzt du dafür Nuands Beispiel `txrx.py <https://github.com/Nuand/bladeRF/blob/624994d65c02ad414a01b29c84154260912f4e4f/host/examples/python/txrx/txrx.py>`_, das genau das tut.

***********************************
Oszillatoren, PLLs und Kalibrierung
***********************************

Alle Direktkonversions-SDRs (einschließlich aller AD9361-basierten SDRs wie dem USRP B2X0, Analog Devices Pluto und bladeRF) sind auf einen einzelnen Oszillator angewiesen, der einen stabilen Takt für den HF-Transceiver bereitstellt. Jeder Versatz oder Jitter in der von diesem Oszillator erzeugten Frequenz überträgt sich als Frequenzversatz und Frequenzjitter auf das empfangene oder gesendete Signal. Dieser Oszillator befindet sich an Bord, kann aber optional durch ein separates Rechteck- oder Sinussignal „diszipliniert" werden, das über einen U.FL-Steckverbinder auf der Platine in das bladeRF eingespeist wird.

An Bord des bladeRF befindet sich ein `Abracon VCTCXO <https://abracon.com/Oscillators/ASTX12_ASVTX12.pdf>`_ (spannungsgesteuerter, temperaturkompensierter Oszillator) mit einer Frequenz von 38,4 MHz. Der „temperaturkompensierte" Aspekt bedeutet, dass er so ausgelegt ist, dass er über einen weiten Temperaturbereich stabil bleibt. Der spannungsgesteuerte Aspekt bedeutet, dass ein Spannungspegel verwendet wird, um leichte Anpassungen an der Oszillatorfrequenz vorzunehmen. Beim bladeRF wird diese Spannung von einem separaten 10-Bit-Digital-Analog-Wandler (DAC) bereitgestellt, wie im Blockdiagramm unten in Grün dargestellt. Das bedeutet, dass wir über Software feine Anpassungen an der Frequenz des Oszillators vornehmen können – so wird das VCTCXO des bladeRF kalibriert (auch als „Trimmen" bezeichnet). Glücklicherweise werden die bladeRFs bereits im Werk kalibriert, wie wir später in diesem Abschnitt besprechen. Wenn du jedoch entsprechende Messgeräte zur Verfügung hast, kannst du diesen Wert jederzeit feinabstimmen, besonders wenn die Frequenz des Oszillators im Laufe der Jahre driftet.

.. image:: ../_images/bladeRF-2.0-micro-Block-Diagram-4-oscillator.png
   :scale: 80 %
   :align: center
   :alt: bladeRF 2.0 Blockdiagramm mit Oszillator

Bei Verwendung einer externen Frequenzreferenz (die nahezu jede Frequenz bis 300 MHz haben kann) wird das Referenzsignal direkt in den `Analog Devices ADF4002 <http://www.analog.com/en/adf4002>`_ PLL auf dem bladeRF eingespeist. Dieser PLL synchronisiert sich mit dem Referenzsignal und sendet ein Signal an den VCTCXO (wie oben in Blau dargestellt), das proportional zur Frequenz- und Phasendifferenz zwischen dem (skalierten) Referenzeingang und dem VCTCXO-Ausgang ist. Sobald der PLL eingerastet ist, ist dieses Signal zwischen PLL und VCTCXO eine stationäre Gleichspannung, die den VCTCXO-Ausgang bei „genau" 38,4 MHz hält (vorausgesetzt, die Referenz war korrekt) und phasensynchron mit dem Referenzeingang ist. Bei Verwendung einer externen Referenz muss :code:`clock_ref` aktiviert werden (entweder über Python oder die CLI) und die Eingangsreferenzfrequenz (auch als :code:`refin_freq` bezeichnet, standardmäßig 10 MHz) eingestellt werden. Gründe für die Verwendung einer externen Referenz sind eine bessere Frequenzgenauigkeit und die Möglichkeit, mehrere SDRs mit derselben Referenz zu synchronisieren.

Jeder VCTCXO-DAC-Trimwert des bladeRF wird im Werk auf innerhalb von 1 Hz bei 38,4 MHz bei Raumtemperatur kalibriert. Du kannst deine Seriennummer auf `dieser Seite <https://www.nuand.com/calibration/>`_ eingeben, um den werksseitig kalibrierten Wert abzufragen (die Seriennummer findest du auf der Platine oder mit :code:`bladerf-tool probe`). Laut Nuand sollte eine neue Platine gut innerhalb von 0,5 ppm und wahrscheinlich näher an 0,1 ppm liegen. Wenn du Messgeräte zur Überprüfung der Frequenzgenauigkeit hast oder den Wert auf den Werkswert setzen möchtest, kannst du folgende Befehle verwenden:

.. code-block:: bash

 $ bladeRF-cli -i
 bladeRF> flash_init_cal 301 0x2049

Ersetze dabei :code:`301` durch deine bladeRF-Größe und :code:`0x2049` durch den Hex-Wert deines VCTCXO-DAC-Trimwerts. Ein Neustart ist erforderlich, damit die Änderung wirksam wird.

***********************************
Abtastung bei 122 MHz
***********************************

Kommt bald!

***********************************
Erweiterungsports
***********************************

Das bladeRF 2.0 verfügt über einen Erweiterungsport mit einem BSH-030-Steckverbinder. Weitere Informationen zur Verwendung dieses Ports folgen bald!

********************************
Weiterführende Literatur
********************************

#. `bladeRF Wiki <https://github.com/Nuand/bladeRF/wiki>`_
#. `Nuands txrx.py Beispiel <https://github.com/Nuand/bladeRF/blob/master/host/examples/python/txrx/txrx.py>`_
