.. _phaser-chapter:

####################
Praktisch mit dem Phaser
####################

In diesem Kapitel verwenden wir das `Analog Devices Phaser <https://wiki.analog.com/resources/eval/user-guides/circuits-from-the-lab/cn0566>`_ (auch CN0566 oder ADALM-PHASER), ein kostengünstiges 8-Kanal-Phased-Array-SDR, das einen PlutoSDR, Raspberry Pi und ADAR1000-Beamformer kombiniert und für einen Betrieb um 10,25 GHz konzipiert ist. Wir behandeln die Einrichtungs- und Kalibrierungsschritte und gehen dann durch einige Beamforming-Beispiele in Python. Für diejenigen, die keinen Phaser haben, sind Screenshots und Animationen enthalten.

.. image:: ../_images/phaser_on_tripod.png
   :scale: 60 %
   :align: center
   :alt: Der Phaser (CN0566) von Analog Devices

************************
Hardware-Übersicht
************************

.. image:: ../_images/phaser_front_and_back.png
   :scale: 40 %
   :align: center
   :alt: Vorder- und Rückseite des Phaser-Geräts

Der Phaser ist eine einzelne Platine, die das Phased Array und viele andere Komponenten enthält, mit einem Raspberry Pi auf einer Seite und einem Pluto auf der anderen Seite. Das übergeordnete Blockdiagramm ist unten dargestellt. Einige wichtige Punkte:

1. Obwohl es wie ein 32-Element-2D-Array aussieht, ist es eigentlich ein 8-Element-1D-Array
2. Beide Empfangskanäle des Pluto werden verwendet (der zweite Kanal nutzt einen u.FL-Stecker auf der Platine selbst)
3. Der LO auf der Platine wird verwendet, um das empfangene Signal von etwa 10,25 GHz auf etwa 2 GHz abwärts zu mischen, damit der Pluto es empfangen kann
4. Jeder ADAR1000 hat vier Phasenschieber mit einstellbarer Verstärkung, und alle vier Kanäle werden summiert, bevor sie an den Pluto gesendet werden
5. Der Phaser enthält im Wesentlichen zwei „Subarrays", wobei jedes Subarray vier Kanäle enthält
6. Nicht gezeigt sind GPIO- und Seriellsignale vom Raspberry Pi zur Steuerung verschiedener Komponenten auf dem Phaser

.. image:: ../_images/phaser_components.png
   :scale: 40 %
   :align: center
   :alt: Die Komponenten des Phaser (CN0566) einschließlich ADF4159, LTC5548, ADAR1000

Ignorieren wir für jetzt die Sendeseite des Phasers; in diesem Kapitel verwenden wir nur das HB100-Gerät als Testsender. Der ADF4159 ist ein Frequenzsynthesizer, der einen Ton bis zu 13 GHz erzeugt – was wir den Lokaloszillator (LO) nennen. Dieser LO wird in einen Mischer, den LTC5548, eingespeist, der für Auf- oder Abwärtskonversion verwendet werden kann. Für die Abwärtskonversion nimmt er den LO sowie ein Signal von 2 bis 14 GHz und multipliziert die beiden, was eine Frequenzverschiebung durchführt. Das resultierende abwärtsgemischte Signal kann sich irgendwo von DC bis 6 GHz befinden, obwohl wir etwa 2 GHz anstreben. Der ADAR1000 ist ein 4-Kanal-Analog-Beamformer; der Phaser nutzt zwei davon. Auf dem Phaser gibt jeder ADAR1000 ein Signal aus, das abwärtsgemischt und dann vom Pluto empfangen wird. Mit dem Raspberry Pi können wir die Phase und Verstärkung aller acht Kanäle in Echtzeit steuern.

Für Interessierte ist unten ein etwas detaillierteres Blockdiagramm angegeben.

.. image:: ../_images/phaser_detailed_block_diagram.png
   :scale: 80 %
   :align: center
   :alt: Detailliertes Blockdiagramm des Phaser (CN0566)


************************
SD-Karten-Vorbereitung
************************

Wir gehen davon aus, dass du den Raspberry Pi auf dem Phaser direkt (mit Monitor/Tastatur/Maus) verwendest. Dies vereinfacht die Einrichtung, da Analog Devices ein vorgefertigtes SD-Karten-Image mit allen notwendigen Treibern und Software veröffentlicht. Du kannst das SD-Karten-Image herunterladen und Anweisungen zum Abbilden der SD-Karte `hier <https://wiki.analog.com/resources/tools-software/linux-software/kuiper-linux>`_ finden. Das Image basiert auf Raspberry Pi OS und enthält alle erforderliche Software bereits installiert.

************************
Hardware-Vorbereitung
************************

1. Verbinde Plutos mittleren Micro-USB-Port mit dem Raspberry Pi
2. Optional: Schraube das Stativ vorsichtig in die Stativhalterung
3. Wir gehen davon aus, dass du ein HDMI-Display, eine USB-Tastatur und eine USB-Maus am Raspberry Pi verwendest
4. Versorge Pi und Phaser-Platine über den Type-C-Port des Phaser (CN0566), d. h. schließe KEIN Netzteil am USB-C des Raspberry Pi an

************************
Software-Installation
************************

Nachdem du mit dem vorgefertigten Image im Raspberry Pi gebootet hast (Standard-Benutzer/Passwort: analog/analog), wird empfohlen, die folgenden Schritte auszuführen:

.. code-block:: bash

 wget https://github.com/mthoren-adi/rpi_setup_stuff/raw/main/phaser/phaser_sdcard_setup.sh
 sudo chmod +x phaser_sdcard_setup.sh
 ./phaser_sdcard_setup.sh
 sudo reboot

 sudo raspi-config

Weitere Hilfe bei der Einrichtung des Phasers findest du auf der `Phaser-Wiki-Quickstart-Seite <https://wiki.analog.com/resources/eval/user-guides/circuits-from-the-lab/cn0566/quickstart>`_.

************************
HB100-Einrichtung
************************

.. image:: ../_images/phaser_hb100.png
   :scale: 50 %
   :align: center
   :alt: HB100, das mit dem Phaser geliefert wird

Das HB100, das mit dem Phaser geliefert wird, ist ein kostengünstiges Doppler-Radarmodul, das wir als Testsender verwenden; es sendet einen kontinuierlichen Ton um 10 GHz. Es wird mit 2 AA-Batterien oder einer 3-V-Tischversorgung betrieben und hat eine rote LED, wenn es eingeschaltet ist.

Da das HB100 kostengünstig ist und günstige HF-Komponenten verwendet, variiert seine Sendefrequenz von Einheit zu Einheit um Hunderte von MHz – ein Bereich, der größer ist als die höchste Bandbreite, die wir mit dem Pluto empfangen können (56 MHz). Um sicherzustellen, dass wir unseren Pluto so abstimmen, dass das HB100-Signal immer empfangen wird, müssen wir die Sendefrequenz des HB100 bestimmen. Dies geschieht mit einer Beispiel-App von Analog Devices, die einen Frequenzsweep durchführt und FFTs berechnet, während sie nach einem Spike sucht. Stelle sicher, dass dein HB100 eingeschaltet und in der Nähe des Phasers ist, und führe dann das Dienstprogramm aus:

.. code-block:: bash

 cd ~/pyadi-iio/examples/phaser
 python phaser_find_hb100.py

Es sollte eine Datei namens :code:`hb100_freq_val.pkl` im selben Verzeichnis erstellen. Diese Datei enthält die HB100-Sendefrequenz in Hz (gepickelt, daher nicht im Klartext sichtbar), die wir im nächsten Schritt verwenden.

************************
Kalibrierung
************************

Schließlich müssen wir das Phased Array kalibrieren. Dazu muss das HB100 in Hauptstrahlrichtung (0 Grad) des Arrays gehalten werden. Die Seite des HB100 mit dem Barcode ist die Seite, die das Signal sendet; diese Seite sollte einige Fuß vom Phaser entfernt, direkt davor und mittig ausgerichtet, auf den Phaser gezeigt werden. Dann führe das Kalibrierungsdienstprogramm aus:

.. code-block:: bash

 python phaser_examples.py cal

Dies erstellt zwei weitere Pickle-Dateien: phase_cal_val.pkl und gain_cal_val.pkl im selben Verzeichnis. Jede enthält ein Array mit 8 Zahlen, die den Phasen- und Verstärkungs-Korrekturen entsprechen, die zur Kalibrierung jedes Kanals benötigt werden. Diese Werte sind für jeden Phaser einzigartig, da sie während der Herstellung variieren können.

************************
Vorgefertigte Beispiel-App
************************

Nachdem wir unseren Phaser kalibriert und die HB100-Frequenz gefunden haben, können wir die Beispiel-App ausführen, die Analog Devices bereitstellt:

.. code-block:: bash

 python phaser_gui.py

Wenn du das Kontrollkästchen „Auto Refresh Data" unten links aktivierst, sollte es zu laufen beginnen. Du solltest etwas Ähnliches wie das Folgende sehen, wenn du das HB100 in der Hauptstrahlrichtung des Phasers hältst.

.. image:: ../_images/phaser_gui.png
   :scale: 50 %
   :align: center
   :alt: Phaser-Beispiel-GUI-Tool von Analog Devices

************************
Phaser in Python
************************

Jetzt tauchen wir in den praktischen Python-Teil ein. Für diejenigen, die keinen Phaser haben, werden Screenshots und Animationen bereitgestellt.

Phaser und Pluto initialisieren
##############################

Der folgende Python-Code richtet unseren Phaser und Pluto ein. Zu diesem Zeitpunkt solltest du bereits die Kalibrierungsschritte ausgeführt haben, die drei Pickle-Dateien erzeugen. Stelle sicher, dass du das Python-Skript aus demselben Verzeichnis heraus ausführst, in dem sich diese Pickle-Dateien befinden.

.. code-block:: python

 import time
 import sys
 import matplotlib.pyplot as plt
 import numpy as np
 import pickle
 from adi import ad9361
 from adi.cn0566 import CN0566

 phase_cal = pickle.load(open("phase_cal_val.pkl", "rb"))
 gain_cal = pickle.load(open("gain_cal_val.pkl", "rb"))
 signal_freq = pickle.load(open("hb100_freq_val.pkl", "rb"))
 d = 0.014  # Abstand zwischen benachbarten Antennenelementen

 phaser = CN0566(uri="ip:localhost")
 sdr = ad9361(uri="ip:192.168.2.1")
 phaser.sdr = sdr
 print("PlutoSDR und CN0566 verbunden!")

 time.sleep(0.5) # von Analog Devices empfohlen

 phaser.configure(device_mode="rx")

 # Alle Antennenelemente auf halbe Skalierung setzen
 gain = 64 # 64 ist etwa halbe Skalierung
 for i in range(8):
     phaser.set_chan_gain(i, gain, apply_cal=False)

 # Strahl in Hauptstrahlrichtung (null Grad) ausrichten
 phaser.set_beam_phase_diff(0.0)

 # Verschiedene SDR-Einstellungen
 sdr._ctrl.debug_attrs["adi,frequency-division-duplex-mode-enable"].value = "1"
 sdr._ctrl.debug_attrs["adi,ensm-enable-txnrx-control-enable"].value = "0" # Pin-Steuerung deaktivieren
 sdr._ctrl.debug_attrs["initialize"].value = "1"
 sdr.rx_enabled_channels = [0, 1] # Rx1 und Rx2 aktivieren
 sdr._rxadc.set_kernel_buffers_count(1) # Keine veralteten Puffer
 sdr.tx_hardwaregain_chan0 = int(-80) # Sicherstellen, dass die Tx-Kanäle gedämpft sind
 sdr.tx_hardwaregain_chan1 = int(-80)

 # Grundlegende PlutoSDR-Einstellungen
 sample_rate = 30e6
 sdr.sample_rate = int(sample_rate)
 sdr.rx_buffer_size = int(1024)  # Samples pro Puffer
 sdr.rx_rf_bandwidth = int(10e6)  # Analogfilter-Bandbreite

 # Manuelle Verstärkung (keine automatische Verstärkungsregelung)
 sdr.gain_control_mode_chan0 = "manual"
 sdr.gain_control_mode_chan1 = "manual"
 sdr.rx_hardwaregain_chan0 = 10 # dB, 0 ist die niedrigste Verstärkung
 sdr.rx_hardwaregain_chan1 = 10 # dB

 sdr.rx_lo = int(2.2e9) # Der Pluto stimmt auf diese Frequenz ab

 # PLL des Phasers (ADF4159 auf der Platine) einstellen
 offset = 1000000 # kleiner Versatz, damit wir nicht bei 0 Hz mit DC-Spike sind
 phaser.lo = int(signal_freq + sdr.rx_lo - offset)


Samples vom Pluto empfangen
################################

Zu diesem Zeitpunkt sind Phaser und Pluto konfiguriert. Wir können nun Daten vom Pluto empfangen. Holen wir uns einen einzelnen Batch von 1024 Samples und nehmen dann die FFT jedes der beiden Kanäle.

.. code-block:: python

 # Samples empfangen (wie viele wir für rx_buffer_size gesetzt haben)
 data = sdr.rx()

 # FFT berechnen
 PSD0 = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(data[0])))**2)
 PSD1 = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(data[1])))**2)
 f = np.linspace(-sample_rate/2, sample_rate/2, len(data[0]))

 # Zeitbereichsplot zum Überprüfen des HB100-Empfangs und der Sättigung
 plt.subplot(2, 1, 1)
 plt.plot(data[0].real) # nur Realteil plotten
 plt.plot(data[1].real)
 plt.xlabel("Datenpunkt")
 plt.ylabel("ADC-Ausgang")

 # PSDs zeigen, wo das HB100-Signal ist
 plt.subplot(2, 1, 2)
 plt.plot(f/1e6, PSD0)
 plt.plot(f/1e6, PSD1)
 plt.xlabel("Frequenz [MHz]")
 plt.ylabel("Signalstärke [dB]")
 plt.tight_layout()
 plt.show()

Was du zu diesem Zeitpunkt siehst, hängt davon ab, ob dein HB100 eingeschaltet ist und wohin es zeigt. Wenn du es einige Fuß vom Phaser entfernt und direkt darauf gerichtet hältst, solltest du etwas wie dieses sehen:

.. image:: ../_images/phaser_rx_psd.png
   :scale: 100 %
   :align: center
   :alt: Phaser-Anfangsbeispiel

Beachte den starken Spike nahe 0 Hz; der zweite kürzere Spike ist lediglich ein Artefakt, das ignoriert werden kann, da es etwa 40 dB niedriger ist.

Beamforming durchführen
##############################

Nun schwenken wir die Phase! Im folgenden Code schwenken wir die Phase von -180 bis +180 Grad in 2-Grad-Schritten. Beachte, dass dies nicht der Winkel ist, auf den der Beamformer zeigt; es ist die Phasendifferenz zwischen benachbarten Kanälen. Wir müssen den Ankunftswinkel (AoA) berechnen, der jedem Phasenschritt entspricht:

.. math::

 \phi = \frac{2 \pi d}{\lambda} \sin(\theta_{AOA})

wobei :math:`\theta_{AOA}` der Ankunftswinkel bezüglich der Hauptstrahlrichtung, :math:`d` der Antennenabstand in Metern und :math:`\lambda` die Wellenlänge des Signals ist. Umstellen nach :math:`\theta_{AOA}`:

.. math::

 \theta_{AOA} = \sin^{-1}\left(\frac{c \phi}{2 \pi f d}\right)

.. code-block:: python

 powers = [] # Haupt-DOA-Ergebnis
 angle_of_arrivals = []
 for phase in np.arange(-180, 180, 2): # Winkel schwenken
     print(phase)
     # Phasendifferenz zwischen benachbarten Kanälen setzen
     for i in range(8):
         channel_phase = (phase * i + phase_cal[i]) % 360.0
         phaser.elements.get(i + 1).rx_phase = channel_phase
     phaser.latch_rx_settings() # Einstellungen anwenden

     steer_angle = np.degrees(np.arcsin(max(min(1, (3e8 * np.radians(phase)) / (2 * np.pi * signal_freq * phaser.element_spacing)), -1)))
     angle_of_arrivals.append(steer_angle)
     data = phaser.sdr.rx() # Batch von Samples empfangen
     data_sum = data[0] + data[1] # zwei Subarrays summieren
     power_dB = 10*np.log10(np.sum(np.abs(data_sum)**2))
     powers.append(power_dB)

 powers -= np.max(powers) # normalisieren, sodass Maximum bei 0 dB ist

 plt.plot(angle_of_arrivals, powers, '.-')
 plt.xlabel("Ankunftswinkel")
 plt.ylabel("Magnitude [dB]")
 plt.show()

Für jeden :code:`phase`-Wert setzen wir die Phasenschieber, fügen die Phasenkalibrierungswerte hinzu und berechnen dann die Signalleistung. Das Ergebnis sollte ungefähr so aussehen:

.. image:: ../_images/phaser_sweep.png
   :scale: 100 %
   :align: center
   :alt: Phaser-Einzelsweep

In diesem Beispiel wurde das HB100 leicht seitlich der Hauptstrahlrichtung gehalten.

Für einen Polarplot:

.. code-block:: python

 # Polarplot
 fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
 ax.plot(np.deg2rad(angle_of_arrivals), powers) # x-Achse in Radiant
 ax.set_rticks([-40, -30, -20, -10, 0])
 ax.set_thetamin(np.min(angle_of_arrivals))
 ax.set_thetamax(np.max(angle_of_arrivals))
 ax.set_theta_direction(-1) # im Uhrzeigersinn zunehmen
 ax.set_theta_zero_location('N') # 0 Grad nach oben
 ax.grid(True)
 plt.show()

.. image:: ../_images/phaser_sweep_polar.png
   :scale: 100 %
   :align: center
   :alt: Phaser-Einzelsweep als Polarplot

Durch das Maximum können wir die Ankunftsrichtung des Signals schätzen!

Echtzeit und räumliche Fensterung
#####################################

Bisher haben wir die Verstärkungsanpassungen jedes Kanals auf gleiche Werte gelassen, sodass alle acht Kanäle gleichmäßig summiert werden. Genau wie wir ein Fenster vor der FFT anwenden, können wir ein Fenster im räumlichen Bereich anwenden, indem wir Gewichte auf diese acht Kanäle anwenden. Wir verwenden dieselben Fensterfunktionen wie Hanning, Hamming usw. Passen wir auch den Code an, damit er in Echtzeit läuft:

.. code-block:: python

 plt.ion() # für Echtzeitansicht benötigt
 print("Gestartet, mit Strg+C stoppen")
 try:
     while True:
         powers = [] # Haupt-DOA-Ergebnis
         angle_of_arrivals = []
         for phase in np.arange(-180, 180, 6): # Winkel schwenken
             # Phasendifferenz zwischen benachbarten Kanälen setzen
             for i in range(8):
                 channel_phase = (phase * i + phase_cal[i]) % 360.0
                 phaser.elements.get(i + 1).rx_phase = channel_phase

             # Verstärkungen setzen, incl. gain_cal (kann für Fensterung verwendet werden)
             gain_list = [127] * 8 # rechteckiges Fenster          [127, 127, 127, 127, 127, 127, 127, 127]
             #gain_list = np.rint(np.hamming(8) * 127)         # [ 10,  32,  82, 121, 121,  82,  32,  10]
             #gain_list = np.rint(np.hanning(10)[1:-1] * 127)  # [ 15,  52,  95, 123, 123,  95,  52,  15]
             #gain_list = np.rint(np.blackman(10)[1:-1] * 127) # [  6,  33,  80, 121, 121,  80,  33,   6]
             #gain_list = np.rint(np.bartlett(10)[1:-1] * 127) # [ 28,  56,  85, 113, 113,  85,  56,  28]
             for i in range(8):
                 channel_gain = int(gain_list[i] * gain_cal[i])
                 phaser.elements.get(i + 1).rx_gain = channel_gain

             phaser.latch_rx_settings() # Einstellungen anwenden

             steer_angle = np.degrees(np.arcsin(max(min(1, (3e8 * np.radians(phase)) / (2 * np.pi * signal_freq * phaser.element_spacing)), -1)))
             angle_of_arrivals.append(steer_angle)
             data = phaser.sdr.rx() # Samples empfangen
             data_sum = data[0] + data[1] # Subarrays summieren
             power_dB = 10*np.log10(np.sum(np.abs(data_sum)**2))
             powers.append(power_dB)

         powers -= np.max(powers) # normalisieren

         # Echtzeitansicht
         plt.plot(angle_of_arrivals, powers, '.-')
         plt.xlabel("Ankunftswinkel")
         plt.ylabel("Magnitude [dB]")
         plt.draw()
         plt.pause(0.001)
         plt.clf()

 except KeyboardInterrupt:
     sys.exit() # Python beenden

Du solltest eine Echtzeitversion der vorherigen Übung sehen. Versuche, die :code:`gain_list` zu wechseln, um verschiedene Fenster auszuprobieren. Hier ist ein Beispiel mit dem rechteckigen Fenster (d. h. keine Fensterfunktion):

.. image:: ../_images/phaser_animation_rect.gif
   :scale: 100 %
   :align: center
   :alt: Beamforming-Animation mit dem Phaser und einem rechteckigen Fenster

und hier ein Beispiel mit dem Hamming-Fenster:

.. image:: ../_images/phaser_animation_hamming.gif
   :scale: 100 %
   :align: center
   :alt: Beamforming-Animation mit dem Phaser und einem Hamming-Fenster

Beachte das Fehlen von Nebenkeulen beim Hamming-Fenster. Jedes Fenster außer dem rechteckigen reduziert die Nebenkeulen erheblich, aber dafür wird die Hauptkeule etwas breiter.

************************
Monopuls-Tracking
************************

Bisher haben wir einzelne Sweeps durchgeführt, um den Ankunftswinkel eines Testsenders (des HB100) zu finden. Angenommen, wir möchten kontinuierlich ein Kommunikations- oder Radarsignal empfangen, das sich bewegt und dessen Ankunftswinkel sich im Laufe der Zeit ändert. Diesen Prozess nennen wir Tracking, und er setzt voraus, dass wir bereits eine grobe Schätzung des Ankunftswinkels haben. Wir verwenden Monopuls-Tracking, um die Gewichte adaptiv zu aktualisieren, damit die Hauptkeule mit der Zeit auf das Signal gerichtet bleibt.

Das 1943 von Robert Page am Naval Research Laboratory (NRL) erfundene Grundkonzept des Monopuls-Trackings besteht darin, zwei Strahlen zu verwenden, die beide leicht vom aktuellen Ankunftswinkel (oder zumindest unserer Schätzung davon) versetzt sind, aber auf verschiedenen Seiten, wie im Diagramm unten gezeigt.

.. image:: ../_images/monopulse.svg
   :align: center
   :target: ../_images/monopulse.svg
   :alt: Monopuls-Strahldiagramm mit zwei Strahlen und dem Summenstrahl

Wir nehmen dann sowohl die Summe als auch die Differenz (auch Delta genannt) dieser beiden Strahlen digital, was bedeutet, dass wir zwei digitale Kanäle des Phasers verwenden müssen. Der Summenstrahl entspricht einem Strahl, der auf den aktuellen Ankunftswinkel zentriert ist und für Demodulation/Dekodierung verwendet werden kann. Der Delta-Strahl hat eine Nullstelle beim Ankunftswinkel. Wir können das Verhältnis zwischen Summen- und Delta-Strahl (als Fehler bezeichnet) für unser Tracking verwenden. Im Code unten ist :code:`data[0]` der erste Kanal des Pluto (erste vier Phaser-Elemente) und :code:`data[1]` der zweite Kanal (zweite vier Elemente):

.. code-block:: python

   data = phaser.sdr.rx()
   sum_beam = data[0] + data[1]
   delta_beam = data[0] - data[1]
   error = np.mean(np.real(delta_beam / sum_beam))

Das Vorzeichen des Fehlers sagt uns, aus welcher Richtung das Signal tatsächlich kommt, und die Magnitude sagt uns, wie weit wir vom Signal entfernt sind. Durch Wiederholen dieses Prozesses in Echtzeit können wir das Signal verfolgen.

Zuerst kopieren wir den Code, den wir früher verwendet haben, um einen 180-Grad-Sweep durchzuführen, und extrahieren die Phase, bei der die empfangene Leistung maximal war:

.. code-block:: python

   # Phase einmal sweepen für erste AoA-Schätzung (Code von oben)
   # ...
   current_phase = phase_angles[np.argmax(powers)]
   print("max_phase:", current_phase)

Als nächstes erstellen wir zwei Strahlen, wir beginnen damit, 5 Grad niedriger und 5 Grad höher als die aktuelle Schätzung zu versuchen (in Phaseneinheiten). Die folgenden Code-Zeilen steuern die ersten 4 Elemente für den unteren Strahl und die letzten 4 Elemente für den oberen Strahl:

.. code-block:: python

   # Zwei Strahlen auf beiden Seiten unserer aktuellen Schätzung erstellen
   phase_offset = np.radians(5) # KANN ANGEPASST WERDEN
   phase_lower = current_phase - phase_offset
   phase_upper = current_phase + phase_offset
   # erste 4 Elemente für unteren Strahl
   for i in range(0, 4):
      channel_phase = (phase_lower * i + phase_cal[i]) % 360.0
      phaser.elements.get(i + 1).rx_phase = channel_phase
   # letzte 4 Elemente für oberen Strahl
   for i in range(4, 8):
      channel_phase = (phase_upper * i + phase_cal[i]) % 360.0
      phaser.elements.get(i + 1).rx_phase = channel_phase
   phaser.latch_rx_settings() # Einstellungen anwenden

Bevor wir das eigentliche Tracking durchführen, testen wir das oben Genannte, indem wir die Strahlgewichte konstant lassen und das HB100 links und rechts bewegen:

.. code-block:: python

   print("HB100 ETWAS NACH LINKS UND RECHTS BEWEGEN")
   error_log = []
   for i in range(1000):
      data = phaser.sdr.rx() # Batch von Samples empfangen
      sum_beam = data[0] + data[1]
      delta_beam = data[0] - data[1]
      error = np.mean(np.real(delta_beam / sum_beam))
      error_log.append(error)
      print(error)
      time.sleep(0.01)

   plt.plot(error_log)
   plt.plot([0,len(error_log)], [0,0], 'r--')
   plt.xlabel("Zeit")
   plt.ylabel("Fehler")
   plt.show()

.. image:: ../_images/monopulse_waving.svg
   :align: center
   :target: ../_images/monopulse_waving.svg
   :alt: Fehlerfunktion für Monopuls-Tracking ohne Gewichtsaktualisierung

Das Take-away ist: Je weiter sich das HB100 vom Startwinkel entfernt, desto höher der Fehler, und das Vorzeichen des Fehlers sagt uns, auf welcher Seite das HB100 relativ zum Startwinkel ist.

Nun nutzen wir den Fehlerwert, um die Gewichte zu aktualisieren. Wir ersetzen die vorherige Schleife durch eine neue, die den gesamten Prozess umfasst:

.. code-block:: python

   # Phase einmal sweepen für erste AoA-Schätzung
   # ...
   current_phase = phase_angles[np.argmax(powers)]
   print("max_phase:", current_phase)

   # current_phase basierend auf Fehler aktualisieren
   print("HB100 ETWAS NACH LINKS UND RECHTS BEWEGEN")
   phase_log = []
   error_log = []
   for ii in range(500):
      # Zwei Strahlen auf beiden Seiten unserer aktuellen Schätzung erstellen
      phase_offset = np.radians(5)
      phase_lower = current_phase - phase_offset
      phase_upper = current_phase + phase_offset
      # erste 4 Elemente für unteren Strahl
      for i in range(0, 4):
            channel_phase = (phase_lower * i + phase_cal[i]) % 360.0
            phaser.elements.get(i + 1).rx_phase = channel_phase
      # letzte 4 Elemente für oberen Strahl
      for i in range(4, 8):
            channel_phase = (phase_upper * i + phase_cal[i]) % 360.0
            phaser.elements.get(i + 1).rx_phase = channel_phase
      phaser.latch_rx_settings() # Einstellungen anwenden

      data = phaser.sdr.rx() # Batch von Samples empfangen
      sum_beam = data[0] + data[1]
      delta_beam = data[0] - data[1]
      error = np.mean(np.real(delta_beam / sum_beam))
      error_log.append(error)
      print(error)

      # Geschätzte Ankunftsrichtung basierend auf Fehler aktualisieren
      current_phase += -10 * error # manuell angepasst für angemessene Tracking-Geschwindigkeit
      steer_angle = np.degrees(np.arcsin(max(min(1, (3e8 * np.radians(current_phase)) / (2 * np.pi * signal_freq * phaser.element_spacing)), -1)))
      phase_log.append(steer_angle) # Steuerwinkel statt Phase plotten sieht besser aus

      time.sleep(0.01)

   fig, [ax0, ax1] = plt.subplots(2, 1, figsize=(8, 10))

   ax0.plot(phase_log)
   ax0.plot([0,len(phase_log)], [0,0], 'r--')
   ax0.set_xlabel("Zeit")
   ax0.set_ylabel("Phasenschätzung [Grad]")

   ax1.plot(error_log)
   ax1.plot([0,len(error_log)], [0,0], 'r--')
   ax1.set_xlabel("Zeit")
   ax1.set_ylabel("Fehler")

   plt.show()

.. image:: ../_images/monopulse_tracking.svg
   :align: center
   :target: ../_images/monopulse_tracking.svg
   :alt: Monopuls-Tracking-Demo mit Phaser und HB100

Man sieht, dass der Fehler im Wesentlichen die Ableitung der Phasenschätzung ist; da wir erfolgreich tracken, entspricht die Phasenschätzung mehr oder weniger dem tatsächlichen Ankunftswinkel. Das Ziel ist, dass die Änderung des Ankunftswinkels nie so schnell ist, dass das Signal über die Hauptkeulen der zwei Strahlen hinausgeht.

Praktische Anwendungsfälle des Monopuls-Trackings sind fast immer 2D (mit einem 2D/planaren Array statt einem linearen Array wie dem Phaser). Im 2D-Fall werden vier Strahlen erstellt, und es gibt einen Summenstrahl und vier Delta-Strahlen für das Lenken in beiden Dimensionen.

************************
Radar mit dem Phaser
************************

Demnächst verfügbar!

************************
Schlussfolgerung
************************

Der gesamte Code zur Erzeugung der Abbildungen in diesem Kapitel ist auf der GitHub-Seite des Lehrbuchs verfügbar.
