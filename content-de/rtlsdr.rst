.. _rtlsdr-chapter:

##################
RTL-SDR in Python
##################

Das RTL-SDR ist mit einem Preis von etwa 30 USD das bei weitem günstigste SDR und ein hervorragender Einstieg in die Welt der Software-Defined Radios.
Obwohl es nur zum Empfangen geeignet ist und nur bis zu etwa 1,75 GHz abstimmen kann, gibt es zahlreiche Anwendungsmöglichkeiten. 
In diesem Kapitel lernen wir, wie man die RTL-SDR-Software einrichtet und wie man dessen Python-API verwendet.

.. image:: ../_images/rtlsdrs.svg
   :align: center
   :target: ../_images/rtlsdrs.svg
   :alt: Beispiele für RTL-SDRs

********************************
Hintergrund zum RTL-SDR
********************************

Das RTL-SDR entstand um das Jahr 2010, als Enthusiasten herausfanden, dass sie günstige DVB-T-Dongles hacken konnten, die den Realtek RTL2832U-Chip enthielten. DVB-T ist ein digitaler Fernsehstandard, der hauptsächlich in Europa verwendet wird. 
Das Besondere am RTL2832U war, dass die rohen IQ-Samples direkt abgerufen werden konnten, was es ermöglichte, den Chip als universelles Empfangs-SDR zu nutzen.

Der RTL2832U-Chip enthält den Analog-Digital-Wandler (ADC) und den USB-Controller, muss jedoch mit einem HF-Tuner kombiniert werden. Verbreitete Tuner-Chips sind der Rafael Micro R820T, R828D und Elonics E4000. Der abstimmbare Frequenzbereich hängt vom Tuner-Chip ab und liegt typischerweise bei etwa 50 – 1700 MHz. Die maximale Abtastrate hingegen wird vom RTL2832U und dem USB-Bus des Computers bestimmt und beträgt ohne zu viele fehlende Samples üblicherweise etwa 2,4 MHz. Diese Tuner sind extrem günstig und weisen eine sehr schlechte HF-Empfindlichkeit auf; daher ist es oft notwendig, einen rauscharmen Verstärker (LNA) und einen Bandpassfilter hinzuzufügen, um schwache Signale zu empfangen.

Der RTL2832U verwendet immer 8-Bit-Samples, sodass der Host-Rechner zwei Bytes pro IQ-Sample empfängt. Premium-RTL-SDRs sind in der Regel mit einem temperaturgesteuerten Oszillator (TCXO) anstelle des günstigeren Quarzoszillators ausgestattet, was eine bessere Frequenzstabilität bietet. Ein weiteres optionales Merkmal ist ein Bias-Tee (auch Bias-T genannt), eine integrierte Schaltung, die am SMA-Anschluss etwa 4,5 V Gleichspannung bereitstellt, um bequem einen externen LNA oder andere HF-Komponenten zu versorgen. Diese zusätzliche Gleichspannung liegt auf der HF-Seite des SDRs und stört den normalen Empfangsbetrieb nicht.

Für diejenigen, die sich für Richtungsbestimmung (Direction of Arrival, DOA) oder andere Beamforming-Anwendungen interessieren, ist das `KrakenSDR <https://www.crowdsupply.com/krakenrf/krakensdr>`_ ein phasenkohärentes SDR, das aus fünf RTL-SDRs besteht, die sich einen Oszillator und einen Sample-Takt teilen.

********************************
Software-Einrichtung
********************************

Ubuntu (oder Ubuntu unter WSL)
###############################

Auf Ubuntu 20, 22 und anderen Debian-basierten Systemen kann die RTL-SDR-Software mit folgendem Befehl installiert werden:

.. code-block:: bash

 sudo apt install rtl-sdr

Dadurch werden die librtlsdr-Bibliothek sowie Kommandozeilenwerkzeuge wie :code:`rtl_sdr`, :code:`rtl_tcp`, :code:`rtl_fm` und :code:`rtl_test` installiert.

Anschließend wird der Python-Wrapper für librtlsdr installiert:

.. code-block:: bash

 sudo pip install pyrtlsdr

Falls Ubuntu über WSL verwendet wird, muss auf der Windows-Seite das neueste `Zadig <https://zadig.akeo.ie/>`_ heruntergeladen und ausgeführt werden, um den „WinUSB"-Treiber für das RTL-SDR zu installieren (es kann zwei Bulk-In-Schnittstellen geben – in diesem Fall ist „WinUSB" auf beiden zu installieren). Danach das RTL-SDR einmal aus- und wieder einstecken.

Als nächstes muss das RTL-SDR-USB-Gerät an WSL weitergeleitet werden. Dazu wird zunächst das neueste `usbipd utility msi <https://github.com/dorssel/usbipd-win/releases>`_ installiert (diese Anleitung geht von usbipd-win 4.0.0 oder höher aus). Dann wird PowerShell im Administratormodus geöffnet und folgendes ausgeführt:

.. code-block:: bash

    # (RTL-SDR ausstecken)
    usbipd list
    # (RTL-SDR einstecken)
    usbipd list
    # (das neue Gerät finden und den Index im folgenden Befehl einsetzen)
    usbipd bind --busid 1-5
    usbipd attach --wsl --busid 1-5

Auf der WSL-Seite sollte :code:`lsusb` ausgeführt werden können, und ein neues Gerät namens RTL2838 DVB-T oder ähnliches sollte erscheinen.

Bei Berechtigungsproblemen (z. B. wenn der unten beschriebene Test nur mit :code:`sudo` funktioniert) müssen udev-Regeln eingerichtet werden. Zunächst :code:`lsusb` ausführen, um die ID des RTL-SDR zu finden, dann die Datei :code:`/etc/udev/rules.d/10-rtl-sdr.rules` mit folgendem Inhalt erstellen (idVendor und idProduct entsprechend anpassen, falls abweichend):

.. code-block::

 SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2838", MODE="0666"

Um udev neu zu laden, folgenden Befehl ausführen:

.. code-block:: bash

    sudo udevadm control --reload-rules
    sudo udevadm trigger

Falls unter WSL die Meldung :code:`Failed to send reload request: No such file or directory` erscheint, bedeutet dies, dass der udev-Dienst nicht läuft. In diesem Fall :code:`sudo nano /etc/wsl.conf` öffnen und folgende Zeilen hinzufügen:

.. code-block:: bash

 [boot]
 command="service udev start"

Anschließend WSL mit folgendem Befehl in PowerShell als Administrator neu starten: :code:`wsl.exe --shutdown`.

Möglicherweise muss das RTL-SDR auch aus- und wieder eingesteckt werden (unter WSL muss :code:`usbipd attach` erneut ausgeführt werden).

Windows
###################

Für Windows-Nutzer siehe https://www.rtl-sdr.com/rtl-sdr-quick-start-guide/.

********************************
RTL-SDR testen
********************************

Wenn die Software-Einrichtung erfolgreich war, sollte folgender Test ausführbar sein, der das RTL-SDR auf das UKW-Radioband abstimmt und 1 Million Samples in eine Datei namens :code:`recording.iq` unter :code:`/tmp` aufzeichnet:

.. code-block:: bash

    rtl_sdr /tmp/recording.iq -s 2e6 -f 100e6 -n 1e6

Falls die Meldung :code:`No supported devices found` erscheint (auch beim Hinzufügen von :code:`sudo`) dann kann Linux das RTL-SDR überhaupt nicht erkennen. Falls es mit :code:`sudo` funktioniert, liegt ein Problem mit den udev-Regeln vor. In diesem Fall den Computer nach der oben beschriebenen udev-Einrichtung neu starten. Alternativ kann :code:`sudo` für alles verwendet werden, einschließlich der Ausführung von Python.

Die Erkennung des RTL-SDR durch Python kann mit folgendem Skript getestet werden:

.. code-block:: python

 from rtlsdr import RtlSdr

 sdr = RtlSdr()
 sdr.sample_rate = 2.048e6 # Hz
 sdr.center_freq = 100e6   # Hz
 sdr.freq_correction = 60  # PPM
 sdr.gain = 'auto'

 print(len(sdr.read_samples(1024)))
 sdr.close()

Die erwartete Ausgabe lautet:

.. code-block:: bash

 Found Rafael Micro R820T tuner
 [R82XX] PLL not locked!
 1024

********************************
RTL-SDR Python-Code
********************************

Der obige Code kann als grundlegendes Verwendungsbeispiel des RTL-SDR in Python betrachtet werden. Die folgenden Abschnitte gehen näher auf die verschiedenen Einstellungen und Nutzungstipps ein.

RTL-SDR-Abstürze vermeiden
###############################

Am Ende unseres Skripts oder wann immer wir mit dem Abrufen von Samples vom RTL-SDR fertig sind, rufen wir :code:`sdr.close()` auf. Dies verhindert, dass das RTL-SDR in einen fehlerhaften Zustand gerät, in dem es aus- und wieder eingesteckt werden müsste. Auch mit close() kann dies noch passieren – man erkennt es daran, dass das RTL-SDR beim Aufruf von read_samples() einfriert. In diesem Fall muss das RTL-SDR aus- und wieder eingesteckt werden, möglicherweise ist auch ein Neustart des Computers notwendig. Unter WSL muss das RTL-SDR mit usbipd erneut verbunden werden.

Verstärkungseinstellung
#########################

Durch das Setzen von :code:`sdr.gain = 'auto'` wird die automatische Verstärkungsregelung (AGC) aktiviert, die dazu führt, dass das RTL-SDR die Empfangsverstärkung basierend auf den empfangenen Signalen anpasst und versucht, den 8-Bit-ADC ohne Übersteuerung auszulasten. In vielen Situationen, wie etwa beim Erstellen eines Spektrumanalysators, ist es sinnvoll, die Verstärkung auf einem konstanten Wert zu halten, sodass eine manuelle Verstärkung eingestellt werden muss. Das RTL-SDR hat keine stufenlos einstellbare Verstärkung; die Liste der gültigen Verstärkungswerte kann mit :code:`print(sdr.valid_gains_db)` angezeigt werden. Wird ein nicht auf dieser Liste befindlicher Wert gesetzt, wählt das Gerät automatisch den nächstliegenden zulässigen Wert. Die aktuelle Verstärkungseinstellung kann jederzeit mit :code:`print(sdr.gain)` abgerufen werden. Im folgenden Beispiel wird die Verstärkung auf 49,6 dB gesetzt, 4096 Samples empfangen und diese dann im Zeitbereich dargestellt:

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
   :alt: RTL-SDR Beispiel mit manueller Verstärkung

Einige Dinge sind hier zu beachten. Die ersten ca. 2000 Samples scheinen kaum Signalleistung zu enthalten, da sie Transienten darstellen. Es wird empfohlen, die ersten 2000 Samples in jedem Skript zu verwerfen, z. B. mit :code:`sdr.read_samples(2048)`, ohne die Ausgabe weiterzuverwenden. Außerdem ist zu beachten, dass pyrtlsdr die Samples als Float-Werte zwischen -1 und +1 zurückgibt. Obwohl ein 8-Bit-ADC verwendet wird, der ganzzahlige Werte liefert, teilt pyrtlsdr diese aus Bequemlichkeitsgründen durch 127,0.

Erlaubte Abtastraten
#####################

Die meisten RTL-SDRs erfordern, dass die Abtastrate entweder zwischen 230–300 kHz oder zwischen 900 kHz und 3,2 MHz liegt. Beachte, dass bei höheren Raten, insbesondere über 2,4 MHz, möglicherweise nicht alle Samples vollständig über die USB-Verbindung übertragen werden. Wird eine nicht unterstützte Abtastrate angegeben, gibt das Gerät den Fehler :code:`rtlsdr.rtlsdr.LibUSBError: Error code -22: Could not set sample rate to 899000 Hz` zurück. Bei einer zulässigen Abtastrate wird die genaue Abtastrate in der Konsolenausgabe angezeigt; dieser genaue Wert kann auch durch Aufruf von :code:`sdr.sample_rate` abgerufen werden. In manchen Anwendungen kann die Verwendung des exakten Wertes in Berechnungen von Vorteil sein.

Als Übung setzen wir die Abtastrate auf 2,4 MHz und erstellen ein Spektrogramm des UKW-Radiobands:

.. code-block:: python

 # ...
 sdr.sample_rate = 2.4e6 # Hz
 # ...

 fft_size = 512
 num_rows = 500
 x = sdr.read_samples(2048) # anfängliche leere Samples verwerfen
 x = sdr.read_samples(fft_size*num_rows) # alle Samples für das Spektrogramm abrufen
 spectrogram = np.zeros((num_rows, fft_size))
 for i in range(num_rows):
     spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)
 extent = [(sdr.center_freq + sdr.sample_rate/-2)/1e6,
             (sdr.center_freq + sdr.sample_rate/2)/1e6,
             len(x)/sdr.sample_rate, 0]
 plt.imshow(spectrogram, aspect='auto', extent=extent)
 plt.xlabel("Frequenz [MHz]")
 plt.ylabel("Zeit [s]")
 plt.show()

.. image:: ../_images/rtlsdr-waterfall.svg
   :align: center
   :target: ../_images/rtlsdr-waterfall.svg
   :alt: RTL-SDR Wasserfall (auch Spektrogramm) Beispiel

PPM-Einstellung
################

Für diejenigen, die neugierig auf die PPM-Einstellung sind: Jedes RTL-SDR weist aufgrund der günstigen Tuner-Chips und mangelnder Kalibrierung einen kleinen Frequenzversatz/Fehler auf. Der Frequenzversatz sollte über das Spektrum hinweg relativ linear sein (kein konstanter Frequenzversatz), sodass wir ihn durch Eingabe eines PPM-Werts in Teilen pro Million korrigieren können. Wird beispielsweise auf 100 MHz abgestimmt und der PPM-Wert auf 25 gesetzt, verschiebt sich das empfangene Signal um 100e6/1e6*25 = 2500 Hz nach oben. Schmalere Signale sind stärker von Frequenzfehlern betroffen. Viele moderne Signale enthalten jedoch einen Frequenzsynchronisierungsschritt, der eventuelle Frequenzversätze beim Sender, Empfänger oder durch den Doppler-Effekt korrigiert.

********************************
Weiterführende Links
********************************

#. `RTL-SDR.com's About Page <https://www.rtl-sdr.com/about-rtl-sdr/>`_
#. https://hackaday.com/2019/07/31/rtl-sdr-seven-years-later/
#. https://osmocom.org/projects/rtl-sdr/wiki/Rtl-sdr
