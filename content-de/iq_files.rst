.. _iq-files-chapter:

##################
IQ-Dateien und SigMF
##################

In all unseren bisherigen Python-Beispielen haben wir Signale als eindimensionale NumPy-Arrays vom Typ "complex float" gespeichert. In diesem Kapitel lernen wir, wie Signale in einer Datei gespeichert und anschließend wieder in Python eingelesen werden können, und wir stellen den SigMF-Standard vor. Das Speichern von Signaldaten in einer Datei ist äußerst nützlich: Vielleicht möchtest du ein Signal aufzeichnen, um es später manuell offline zu analysieren, es mit einem Kollegen zu teilen oder einen ganzen Datensatz aufzubauen.

*************************
Binärdateien
*************************

Zur Erinnerung: Ein digitales Signal im Basisband ist eine Folge komplexer Zahlen.

Beispiel: [0.123 + j0.512,    0.0312 + j0.4123,    0.1423 + j0.06512, ...]

Diese Zahlen entsprechen [I+jQ, I+jQ, I+jQ, I+jQ, I+jQ, I+jQ, I+jQ, ...]

Wenn wir komplexe Zahlen in einer Datei speichern möchten, speichern wir sie im Format IQIQIQIQIQIQIQIQ. Das heißt, wir speichern eine Reihe von Ganzzahlen oder Gleitkommazahlen hintereinander, und beim Einlesen müssen wir sie wieder in [I+jQ, I+jQ, ...] aufteilen.

Obwohl es möglich ist, die komplexen Zahlen in einer Textdatei oder CSV-Datei zu speichern, bevorzugen wir das Speichern in einer sogenannten „Binärdatei", um Speicherplatz zu sparen. Bei hohen Abtastraten können deine Signalaufzeichnungen leicht mehrere GB groß sein, und wir wollen so speichereffizient wie möglich sein. Wenn du jemals eine Datei in einem Texteditor geöffnet hast und sie unverständlich aussah (wie der Screenshot unten), war es wahrscheinlich eine Binärdatei. Binärdateien enthalten eine Folge von Bytes, und du musst das Format selbst im Blick behalten. Binärdateien sind die effizienteste Methode zur Datenspeicherung, vorausgesetzt, alle möglichen Komprimierungen wurden durchgeführt. Da unsere Signale meist wie eine zufällige Folge von Ganzzahlen oder Gleitkommazahlen aussehen, versuchen wir in der Regel nicht, die Daten zu komprimieren. Binärdateien werden auch für viele andere Dinge verwendet, z.B. für kompilierte Programme (auch „Binaries" genannt). Wenn sie zum Speichern von Signalen verwendet werden, nennen wir sie binäre „IQ-Dateien" mit der Dateiendung .iq.

.. image:: ../_images/binary_file.png
   :scale: 70 %
   :align: center

In Python ist der Standard-Komplextyp np.complex128, der zwei 64-Bit-Gleitkommazahlen pro Sample verwendet. In der DSP/SDR-Welt verwenden wir jedoch tendenziell 16-Bit-Ganzzahlen oder 32-Bit-Gleitkommazahlen, da die ADCs unserer SDRs nicht **so** viel Präzision haben, dass 64-Bit-Gleitkommazahlen gerechtfertigt wären. Tatsächlich haben die meisten SDRs 12-Bit-ADCs, sodass wir den Speicherbedarf minimieren können, indem wir als 16-Bit-Ganzzahlen speichern (np.int16 in Python). Jedes IQ-Sample benötigt dann 4 Bytes, und unsere HF-Aufzeichnung erzeugt eine Datei, die 4-mal die Abtastrate in Bytes groß ist – bekannt als „Seans 4x-Regel". In den Python-Beispielen unten verwenden wir **np.complex64**, das zwei 32-Bit-Gleitkommazahlen verwendet, da Python keinen nativen komplexen Ganzzahltyp hat (das hindert uns nicht daran, IQ als Ganzzahlen in eine Datei zu speichern, wie du gleich sehen wirst). Wenn du ein Signal in Python verarbeitest, spielt das keine große Rolle. Aber wenn du das eindimensionale Array in einer Datei speichern möchtest, solltest du sicherstellen, dass es zuerst ein Array vom Typ np.complex64 ist (oder np.int16 mit verschachtelten IQ-Werten).

*************************
Python-Beispiele
*************************

In Python und speziell in NumPy verwenden wir die Funktion :code:`tofile()`, um ein NumPy-Array in eine Datei zu speichern. Hier ist ein kurzes Beispiel, das ein einfaches QPSK-Signal mit Rauschen erstellt und es in eine Datei im selben Verzeichnis speichert, aus dem wir unser Skript ausführen:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    num_symbols = 10000

    # x_symbols enthält komplexe Zahlen, die die QPSK-Symbole darstellen. Jedes Symbol hat den Betrag 1 und einen Phasenwinkel entsprechend einem der vier QPSK-Konstellationspunkte (45, 135, 225 oder 315 Grad)
    x_int = np.random.randint(0, 4, num_symbols) # 0 bis 3
    x_degrees = x_int*360/4.0 + 45 # 45, 135, 225, 315 Grad
    x_radians = x_degrees*np.pi/180.0 # sin() und cos() nehmen Bogenmaß
    x_symbols = np.cos(x_radians) + 1j*np.sin(x_radians) # erzeugt unsere komplexen QPSK-Symbole
    n = (np.random.randn(num_symbols) + 1j*np.random.randn(num_symbols))/np.sqrt(2) # AWGN mit Einheitsleistung
    r = x_symbols + n * np.sqrt(0.01) # Rauschleistung von 0.01
    print(r)
    plt.plot(np.real(r), np.imag(r), '.')
    plt.grid(True)
    plt.show()

    # Jetzt in eine IQ-Datei speichern
    print(type(r[0])) # Datentyp prüfen. Ups, es ist 128 statt 64!
    r = r.astype(np.complex64) # In 64 konvertieren
    print(type(r[0])) # Verifizieren, dass es 64 ist
    r.tofile('qpsk_in_noise.iq') # In Datei speichern


Prüfe nun die Details der erzeugten Datei und schau, wie viele Bytes sie hat. Es sollten num_symbols * 8 sein, da wir np.complex64 verwendet haben, was 8 Bytes pro Sample entspricht (4 Bytes pro Gleitkommazahl, 2 Gleitkommazahlen pro Sample).

Mit einem neuen Python-Skript können wir diese Datei mit :code:`np.fromfile()` einlesen, wie folgt:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    samples = np.fromfile('qpsk_in_noise.iq', np.complex64) # Datei einlesen. Wir müssen das Format angeben
    print(samples)

    # Konstellation plotten, um sicherzustellen, dass sie korrekt aussieht
    plt.plot(np.real(samples), np.imag(samples), '.')
    plt.grid(True)
    plt.show()

Ein häufiger Fehler ist, :code:`np.fromfile()` das Dateiformat nicht mitzuteilen. Binärdateien enthalten keine Informationen über ihr Format. Standardmäßig geht :code:`np.fromfile()` davon aus, dass es ein Array von float64-Werten liest.

Die meisten anderen Programmiersprachen haben Methoden zum Einlesen von Binärdateien, z.B. in MATLAB kann man fread() verwenden. Zum visuellen Analysieren einer RF-Datei siehe den Abschnitt weiter unten.

Wenn du es jemals mit int16-Werten (auch bekannt als Short-Ints) oder einem anderen Datentyp zu tun hast, für den NumPy kein komplexes Äquivalent hat, musst du die Samples als reelle Werte einlesen, auch wenn sie tatsächlich komplex sind. Der Trick besteht darin, sie als reell einzulesen und sie dann selbst wieder in das IQIQIQ...-Format zu verschachteln. Zwei verschiedene Möglichkeiten dafür sind unten gezeigt:

.. code-block:: python

 samples = np.fromfile('iq_samples_as_int16.iq', np.int16).astype(np.float32).view(np.complex64)

oder

.. code-block:: python

 samples = np.fromfile('iq_samples_as_int16.iq', np.int16)
 samples /= 32768 # in -1 bis +1 umwandeln (optional)
 samples = samples[::2] + 1j*samples[1::2] # in IQIQIQ... konvertieren

*****************************
Umstieg von MATLAB
*****************************

Wenn du von MATLAB zu Python wechselst, fragst du dich vielleicht, wie du deine MATLAB-Variablen und .mat-Dateien als binäre IQ-Dateien speichern kannst. Zunächst müssen wir einen Formattyp wählen. Wenn unsere Samples beispielsweise Ganzzahlen zwischen -127 und +127 sind, können wir 8-Bit-Ints verwenden. In diesem Fall können wir folgenden MATLAB-Code verwenden, um die Samples in eine binäre IQ-Datei zu speichern:

.. code-block:: MATLAB

 % Angenommen, unsere IQ-Samples befinden sich in der Variable samples
 disp(samples(1:20))
 filename = 'samples.iq'
 fwrite(fopen(filename,'w'), reshape([real(samples);imag(samples)],[],1), 'int8')

Alle erlaubten Formattypen für fwrite() findest du in der `MATLAB-Dokumentation <https://www.mathworks.com/help/matlab/ref/fwrite.html#buakf91-1-precision>`_. Am besten bleibst du bei :code:`'int8'`, :code:`'int16'` oder :code:`'float32'`.

Auf der Python-Seite kannst du diese Datei wie folgt einlesen:

.. code-block:: python

 samples = np.fromfile('samples.iq', np.int8)
 samples = samples[::2] + 1j*samples[1::2]
 print(samples[0:20]) # sicherstellen, dass die ersten 20 Samples mit MATLAB übereinstimmen

Für mit :code:`'float32'` aus MATLAB gespeicherte Daten kannst du auf der Python-Seite :code:`np.complex64` verwenden, was verschachtelte float32-Werte sind. Den Teil :code:`samples[::2] + 1j*samples[1::2]` kannst du dann weglassen, da NumPy die verschachtelten Gleitkommazahlen automatisch als komplexe Zahlen interpretiert.

*****************************
Visuelle Analyse einer RF-Datei
*****************************

Obwohl wir im Kapitel :ref:`freq-domain-chapter` gelernt haben, unsere eigenen Spektrogrammplots zu erstellen, kommt nichts an bereits fertige Software heran. Wenn es darum geht, RF-Aufzeichnungen ohne Installation zu analysieren, ist die empfohlene Website `IQEngine <https://iqengine.org>`_ – ein komplettes Toolkit zum Analysieren, Verarbeiten und Teilen von RF-Aufzeichnungen.

Für diejenigen, die eine Desktop-Anwendung bevorzugen, gibt es auch `inspectrum <https://github.com/miek/inspectrum>`_. Inspectrum ist ein recht einfaches, aber leistungsstarkes grafisches Werkzeug zum visuellen Durchsuchen einer RF-Datei mit feiner Kontrolle über den Farbkartenbereich und die FFT-Größe (Zoomstufe). Du kannst Alt gedrückt halten und das Scrollrad verwenden, um durch die Zeit zu scrollen. Es hat optionale Cursors zum Messen der Zeit zwischen zwei Energieausbrüchen und die Möglichkeit, einen Ausschnitt der RF-Datei in eine neue Datei zu exportieren. Für die Installation auf Debian-basierten Plattformen wie Ubuntu verwende folgende Befehle:

.. code-block:: bash

 sudo apt-get install qt5-default libfftw3-dev cmake pkg-config libliquid-dev
 git clone https://github.com/miek/inspectrum.git
 cd inspectrum
 mkdir build
 cd build
 cmake ..
 make
 sudo make install
 inspectrum

.. image:: ../_images/inspectrum.jpg
   :scale: 30 %
   :align: center

*************************
Maximalwerte und Sättigung
*************************

Beim Empfangen von Samples von einem SDR ist es wichtig, den maximalen Samplewert zu kennen. Viele SDRs geben die Samples als Gleitkommazahlen mit einem Maximalwert von 1,0 und einem Minimalwert von -1,0 aus. Andere SDRs liefern Samples als Ganzzahlen, meist 16-Bit, wobei die Maximal- und Minimalwerte +32767 bzw. -32768 sind (sofern nicht anders angegeben). Du kannst durch 32.768 teilen, um sie in Gleitkommazahlen von -1,0 bis 1,0 umzuwandeln. Der Grund, den Maximalwert deines SDR zu kennen, liegt in der Sättigung: Wenn ein sehr starkes Signal empfangen wird (oder wenn der Gain zu hoch eingestellt ist), „sättigt" der Empfänger und schneidet die hohen Werte auf den maximalen Samplewert ab. Die ADCs unserer SDRs haben eine begrenzte Anzahl von Bits. Beim Entwickeln einer SDR-Anwendung ist es ratsam, immer auf Sättigung zu prüfen und sie bei Auftreten irgendwie anzuzeigen.

Ein gesättigtes Signal sieht im Zeitbereich abgehackt aus, wie folgt:

.. image:: ../_images/saturated_time.png
   :scale: 30 %
   :align: center
   :alt: Beispiel eines gesättigten Empfängers, bei dem das Signal abgeschnitten ist

Aufgrund der plötzlichen Änderungen im Zeitbereich durch das Abschneiden kann der Frequenzbereich verschmiert aussehen. Mit anderen Worten: Der Frequenzbereich enthält falsche Merkmale, die durch die Sättigung entstanden sind und nicht wirklich Teil des Signals sind. Dies kann zu Verwirrung bei der Signalanalyse führen.

*****************************
SigMF und IQ-Dateien annotieren
*****************************

Da die IQ-Datei selbst keine zugehörigen Metadaten enthält, ist es üblich, eine zweite Datei mit Informationen über das Signal zu haben, mit demselben Dateinamen, aber einer .txt- oder anderen Dateiendung. Diese sollte mindestens die verwendete Abtastrate und die Frequenz, auf die das SDR abgestimmt war, enthalten. Nach der Analyse des Signals könnte die Metadatendatei Informationen über Samplebereiche interessanter Merkmale enthalten, wie z.B. Energieausbrüche. Der Sample-Index ist einfach eine Ganzzahl, die bei 0 beginnt und bei jedem komplexen Sample um 1 erhöht wird. Wenn du wüsstest, dass Energie von Sample 492342 bis 528492 vorhanden ist, könntest du die Datei einlesen und diesen Teil des Arrays herausziehen: :code:`samples[492342:528493]`.

Glücklicherweise gibt es jetzt einen offenen Standard, der ein Metadatenformat zur Beschreibung von Signalaufzeichnungen festlegt, bekannt als `SigMF <https://github.com/sigmf/SigMF>`_. Durch die Verwendung eines offenen Standards wie SigMF können mehrere Parteien RF-Aufzeichnungen einfacher teilen und verschiedene Werkzeuge auf denselben Datensätzen verwenden, wie z.B. `IQEngine <https://iqengine.org/sigmf>`_. Es verhindert auch den „Bitrot" von RF-Datensätzen, bei dem Details der Aufnahme im Laufe der Zeit verloren gehen, weil die Aufzeichnungsdetails nicht zusammen mit der Aufzeichnung selbst gespeichert werden.

Die einfachste (und minimalste) Methode, den SigMF-Standard zur Beschreibung einer erstellten binären IQ-Datei zu verwenden, besteht darin, die .iq-Datei in .sigmf-data umzubenennen und eine neue Datei mit demselben Namen, aber der Endung .sigmf-meta zu erstellen. Dabei muss sichergestellt werden, dass das Datentypfeld in der Meta-Datei dem Binärformat deiner Datendatei entspricht. Diese Meta-Datei ist eine Klartextdatei mit JSON-Inhalt, die du einfach mit einem Texteditor öffnen und manuell ausfüllen kannst (später besprechen wir das programmatische Vorgehen). Hier ist eine Beispiel-.sigmf-meta-Datei, die du als Vorlage verwenden kannst:

.. code-block::

 {
     "global": {
         "core:datatype": "cf32_le",
         "core:sample_rate": 1000000,
         "core:hw": "PlutoSDR with 915 MHz whip antenna",
         "core:author": "Art Vandelay",
         "core:version": "1.0.0"
     },
     "captures": [
         {
             "core:sample_start": 0,
             "core:frequency": 915000000
         }
     ],
     "annotations": []
 }

Beachte: :code:`core:cf32_le` gibt an, dass deine .sigmf-data vom Typ IQIQIQIQ... mit 32-Bit-Gleitkommazahlen ist, d.h. np.complex64, wie wir es zuvor verwendet haben. In der Spezifikation findest du weitere verfügbare Datentypen, z.B. wenn du reelle statt komplexe Daten hast oder 16-Bit-Ganzzahlen statt Gleitkommazahlen verwendest, um Speicherplatz zu sparen.

Neben dem Datentyp sind die wichtigsten auszufüllenden Zeilen :code:`core:sample_rate` und :code:`core:frequency`. Es ist gute Praxis, auch Informationen über die verwendete Hardware (:code:`core:hw`) einzugeben, wie z.B. SDR-Typ und Antenne, sowie eine Beschreibung des Bekannten über das/die Signal(e) in der Aufzeichnung in :code:`core:description`. :code:`core:version` ist einfach die Version des SigMF-Standards, der zum Zeitpunkt der Erstellung der Metadatendatei verwendet wurde.

Wenn du deine RF-Aufzeichnung innerhalb von Python aufzeichnest, z.B. mit der Python-API für dein SDR, kannst du das manuelle Erstellen dieser Metadatendateien vermeiden, indem du das SigMF Python-Paket verwendest. Dieses kann auf einem Ubuntu/Debian-basierten Betriebssystem wie folgt installiert werden:

.. code-block:: bash

 pip install sigmf

Der Python-Code zum Schreiben der .sigmf-meta-Datei für das Beispiel am Anfang dieses Kapitels, in dem wir :code:`qpsk_in_noise.iq` gespeichert haben, ist unten dargestellt:

.. code-block:: python

 import datetime as dt

 import numpy as np
 import sigmf
 from sigmf import SigMFFile

 # <Code aus dem Beispiel>

 # r.tofile('qpsk_in_noise.iq')
 r.tofile('qpsk_in_noise.sigmf-data') # obige Zeile durch diese ersetzen

 # Metadaten erstellen
 meta = SigMFFile(
     data_file='qpsk_in_noise.sigmf-data', # Endung ist optional
     global_info = {
         SigMFFile.DATATYPE_KEY: 'cf32_le',
         SigMFFile.SAMPLE_RATE_KEY: 8000000,
         SigMFFile.AUTHOR_KEY: 'Dein Name und/oder E-Mail',
         SigMFFile.DESCRIPTION_KEY: 'Simulation von QPSK mit Rauschen',
         SigMFFile.VERSION_KEY: sigmf.__version__,
     }
 )

 # Capture-Eintrag bei Zeitindex 0 erstellen
 meta.add_capture(0, metadata={
     SigMFFile.FREQUENCY_KEY: 915000000,
     SigMFFile.DATETIME_KEY: dt.datetime.now(dt.timezone.utc).isoformat(),
 })

 # Auf Fehler prüfen und auf Disk schreiben
 meta.validate()
 meta.tofile('qpsk_in_noise.sigmf-meta') # Endung ist optional

Ersetze einfach :code:`8000000` und :code:`915000000` durch die Variablen, die du für Abtastrate und Mittenfrequenz verwendet hast.

Um eine SigMF-Aufzeichnung in Python einzulesen, verwende folgenden Code. In diesem Beispiel sollten die beiden SigMF-Dateien :code:`qpsk_in_noise.sigmf-meta` und :code:`qpsk_in_noise.sigmf-data` heißen.

.. code-block:: python

 from sigmf import SigMFFile, sigmffile

 # Datensatz laden
 filename = 'qpsk_in_noise'
 signal = sigmffile.fromfile(filename)
 samples = signal.read_samples().view(np.complex64).flatten()
 print(samples[0:10]) # erste 10 Samples ansehen

 # Einige Metadaten und alle Annotierungen abrufen
 sample_rate = signal.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
 sample_count = signal.sample_count
 signal_duration = sample_count / sample_rate

Weitere Details findest du in der `SigMF Python-Dokumentation <https://github.com/sigmf/sigmf-python>`_.

Ein kleiner Bonus für alle, die bis hier gelesen haben: Das SigMF-Logo ist selbst als SigMF-Aufzeichnung gespeichert, und wenn das Signal als Konstellation (IQ-Plot) über die Zeit geplottet wird, ergibt es folgende Animation:

.. image:: ../_images/sigmf_logo.gif
   :scale: 100 %
   :align: center
   :alt: Die SigMF-Logo-Animation

Der Python-Code zum Einlesen der Logo-Datei (abrufbar `hier <https://github.com/sigmf/SigMF/tree/main/logo>`_) und zur Erstellung des animierten GIFs ist unten dargestellt:

.. code-block:: python

 from pathlib import Path
 from tempfile import TemporaryDirectory

 import numpy as np
 import matplotlib.pyplot as plt
 import imageio.v3 as iio
 from sigmf import SigMFFile, sigmffile

 # Datensatz laden
 filename = 'sigmf_logo' # angenommen, es liegt im selben Verzeichnis wie dieses Skript
 signal = sigmffile.fromfile(filename)
 samples = signal.read_samples().view(np.complex64).flatten()

 # Nullen am Ende hinzufügen, damit der Übergang bei der Wiederholung der Animation erkennbar ist
 samples = np.concatenate((samples, np.zeros(50000)))

 sample_count = len(samples)
 samples_per_frame = 5000
 num_frames = int(sample_count/samples_per_frame)

 with TemporaryDirectory() as temp_dir:
    filenames = []
    output_dir = Path(temp_dir)
    for i in range(num_frames):
        print(f"Frame {i} von {num_frames}")
        # Frame plotten
        fig, ax = plt.subplots(figsize=(5, 5))
        samples_frame = samples[i*samples_per_frame:(i+1)*samples_per_frame]
        ax.plot(np.real(samples_frame), np.imag(samples_frame), color="cyan", marker=".", linestyle="None", markersize=1)
        ax.axis([-0.35,0.35,-0.35,0.35])  # Achsen konstant halten
        ax.set_facecolor('black')  # Hintergrundfarbe

        # Plot in Datei speichern
        filename = output_dir.joinpath(f"sigmf_logo_{i}.png")
        fig.savefig(filename, bbox_inches='tight')
        plt.close()
        filenames.append(filename)

    # Animiertes GIF erstellen
    images = [iio.imread(f) for f in filenames]
    iio.imwrite('sigmf_logo.gif', images, fps=20)

**************************************
SigMF Collection für Array-Aufzeichnungen
**************************************

Wenn du ein Phased Array, MIMO-Digitalarray, TDOA-Sensoren oder eine andere Situation hast, in der du mehrere Kanäle synchronisierter RF-Daten aufzeichnest, fragst du dich wahrscheinlich, wie du das rohe IQ mehrerer Streams mit SigMF in Dateien speicherst. Das SigMF **Collection**-System wurde genau für diese Anwendungen entwickelt. Eine Collection ist einfach eine Gruppe von SigMF-Recordings (jedes bestehend aus einer Meta- und einer Datendatei), die über eine übergeordnete :code:`.sigmf-collection`-JSON-Datei zusammengefasst werden. Diese JSON-Datei ist recht einfach aufgebaut: Sie benötigt die SigMF-Version, eine optionale Beschreibung und dann eine Liste von „Streams", die eigentlich nur der Basisname jeder SigMF-Recording in der Collection ist. Hier ist ein Beispiel einer :code:`.sigmf-collection`-Datei:

.. code-block:: json

    {
        "collection": {
            "core:version": "1.2.0",
            "core:description": "eine 4-Elemente-Phased-Array-Aufzeichnung",
            "core:streams": [
                {
                    "name": "channel-0"
                },
                {
                    "name": "channel-1"
                },
                {
                    "name": "channel-2"
                },
                {
                    "name": "channel-3"
                }
            ]
        }
    }

Die Namen der Recordings müssen nicht :code:`channel-0`, :code:`channel-1`, ... sein – sie können beliebig sein, solange sie eindeutig sind und jeder einem Daten- und einer Meta-Datei entspricht. Im obigen Beispiel muss diese .sigmf-collection-Datei, die wir z.B. :code:`4_element_recording.sigmf-collection` nennen könnten, im selben Verzeichnis wie die Meta- und Datendateien liegen:

* :code:`4_element_recording.sigmf-collection`
* :code:`channel-0.sigmf-meta`
* :code:`channel-0.sigmf-data`
* :code:`channel-1.sigmf-meta`
* :code:`channel-1.sigmf-data`
* :code:`channel-2.sigmf-meta`
* :code:`channel-2.sigmf-data`
* :code:`channel-3.sigmf-meta`
* :code:`channel-3.sigmf-data`

Du denkst vielleicht, dass das zu sehr vielen Dateien führt – ein 16-Elemente-Array würde z.B. 33 Dateien ergeben! Aus diesem Grund führte SigMF das **Archive**-System ein, was schlicht SigMFs Begriff für das Tarball-Archivieren einer Dateisammlung ist. Eine SigMF-Archive-Datei verwendet die Endung :code:`.sigmf`, nicht :code:`.tar`! Viele Leute denken, dass .tar-Dateien komprimiert sind, aber das stimmt nicht. Sie sind lediglich eine Methode zum Zusammenfassen von Dateien (es ist im Wesentlichen eine Dateiverkettung ohne Komprimierung). Vielleicht hast du schon eine :code:`.tar.gz`-Datei gesehen – das ist ein Tarball, der mit gzip komprimiert wurde. Für unsere SigMF-Archive werden wir keine Komprimierung verwenden, da die Datendateien bereits binär sind und sich nicht stark komprimieren lassen, vor allem wenn automatische Verstärkungsregelung verwendet wurde. Wenn du ein SigMF-Archiv in Python erstellen möchtest, kannst du alle Dateien in einem Verzeichnis wie folgt in einen Tarball packen:

.. code-block:: python

    import tarfile
    import os

    target_dir = '/mnt/c/Users/marclichtman/Downloads/exampletar/' # SigMF-Dateien befinden sich hier
    with tarfile.open(os.path.join(target_dir, '4_element_recording.sigmf'), 'x') as tar: # x bedeutet erstellen, aber fehlschlagen, falls bereits vorhanden
        for file in os.listdir(target_dir):
            tar.add(os.path.join(target_dir, file), arcname=file) # arcname verhindert, dass der vollständige Pfad im Tar enthalten ist

Das war's! Versuche (vorübergehend), .sigmf in .tar umzubenennen, und sieh dir die Dateien in deinem Datei-Browser an. Um eine der Dateien direkt (ohne manuelles Entpacken des Tarballs) in Python zu öffnen, verwende:

.. code-block:: python

    import tarfile
    import json

    collection_file = '/mnt/c/Users/marclichtman/Downloads/exampletar/4_element_recording.sigmf'
    tar_obj = tarfile.open(collection_file)
    print(tar_obj.getnames()) # Liste aller Dateinamen im Tar als Strings
    channel_0_meta = tar_obj.extractfile('channel-0.sigmf-meta').read() # eine der Meta-Dateien einlesen, als Beispiel
    channel_0_dict = json.loads(channel_0_meta) # in Python-Dictionary konvertieren
    print(channel_0_dict)

Zum Einlesen von IQ-Samples aus dem Tar verwenden wir statt :code:`np.fromfile()` die Funktion :code:`np.frombuffer()`:

.. code-block:: python

    import tarfile
    import numpy as np

    collection_file = '/mnt/c/Users/marclichtman/Downloads/exampletar/4_element_recording.sigmf'
    tar_obj = tarfile.open(collection_file)
    channel_0_data_f = tar_obj.extractfile('channel-0.sigmf-data').read() # Typ: bytes
    samples = np.frombuffer(channel_0_data_f, dtype=np.int16)
    samples = samples[::2] + 1j*samples[1::2] # in IQIQIQ... konvertieren
    samples /= 32768 # in -1 bis +1 konvertieren
    print(samples[0:10])

Wenn du zu einem anderen Teil der Datei springen möchtest, verwende :code:`tar_obj.extractfile('channel-0.sigmf-data').seek(offset)`. Um eine bestimmte Anzahl von Bytes zu lesen, verwende :code:`.read(num_bytes)`. Stelle sicher, dass die Anzahl der Bytes ein Vielfaches deines Datentyps ist!

Zusammenfassend sollten beim Erstellen eines neuen SigMF Collection Archives folgende Schritte durchgeführt werden:

1. Erstelle die .sigmf-meta- und .sigmf-data-Datei für jeden Kanal
2. Erstelle die .sigmf-collection-Datei
3. Packe alle Dateien in einen .sigmf-Tarball
4. (Optional) Teile die .sigmf-Datei mit anderen!

Zum Einlesen der Aufzeichnung musst du den Tarball nicht entpacken – du kannst die Dateien direkt lesen.

**********************
Midas Blue File Format
**********************

Blue Files, auch bekannt als BLUEFILES oder Midas Files, sind ein Dateiformat, das eine Vielzahl von Datenstrukturen darstellen kann, einschließlich ein- und zweidimensionaler Daten, und wird in bestimmten Organisationen zur Aufzeichnung von rohen HF-Signalen in Dateien verwendet. Im Kontext von RF/SDR können Blue Files als IQ-Dateiformat betrachtet werden. Blue Files werden im X-Midas Signalverarbeitungsframework sowie in dessen Ablegern Midas 2k (C++), NeXtMidas (Java) und XMPy (Python) verwendet. Wer REDHAWK kennt: Ein Teil von NeXtMidas ist darin eingebettet. Einige Anwendungen erzeugen Blue Files mit der Dateiendung :code:`.blue`, andere verwenden :code:`.cdif` – beides ist dasselbe zugrunde liegende Format.

Blue Files sind Binärdateien mit drei Komponenten in folgender Reihenfolge:

1. 512-Byte-Header mit Datei-Metadaten
2. Daten, in unserem Fall binäres IQ (Ints oder Floats im Format IQIQIQ...)
3. Optionaler „Extended Header" (auch bekannt als tailing bytes) mit Hilfs-Metadaten in Form beliebiger Schlüssel/Wert-Paare

Die im Header enthaltenen Felder sind auf `dieser Seite <https://sigplot.lgsinnovations.com/html/doc/bluefile.html>`_ beschrieben. Wichtige für uns sind:

- Byte 52: Datenfomatcode, zwei Zeichen. Das erste Zeichen gibt an, ob es sich um reelle (S) oder komplexe (C) Daten handelt. Das zweite Zeichen bezeichnet den Datentyp: :code:`B` = 8-Bit-Ganzzahl mit Vorzeichen, :code:`I` = 16-Bit-Ganzzahl mit Vorzeichen, :code:`L` = 32-Bit-Ganzzahl mit Vorzeichen, :code:`F` = 32-Bit-Gleitkommazahl, :code:`D` = 64-Bit-Gleitkommazahl.
- Byte 8: Datendarstellung, vier Zeichen: :code:`IEEE` bedeutet Big-Endian, :code:`EEEI` bedeutet Little-Endian (am häufigsten)
- Byte 24: Extended-Header-Start, ein int32, in 512-Byte-Blöcken
- Byte 28: Extended-Header-Größe, ein int32, in Bytes
- Byte 264: Zeitintervall zwischen Samples, d.h. 1/Abtastrate, als float64 in Sekunden

So entspricht beispielsweise :code:`CI` dem SigMF-Typ :code:`ci16_le`, und :code:`CF` entspricht SigMF's :code:`cf32_le`. Auch wenn der Extended Header (d.h. die tailing bytes) seine Länge und Startposition angegeben hat, ist der einfache Ansatz, einfach die letzten paar tausend IQ-Samples der Datei zu ignorieren – damit vermeidest du den Extended Header fast mit Sicherheit und liest keine ungültigen IQ-Werte ein.

Der Python-Code zum Einlesen der oben besprochenen Felder sowie der IQ-Samples ist wie folgt:

.. code-block:: python

    import numpy as np
    import os
    import matplotlib.pyplot as plt

    filename = 'deinedatei.blue' # oder cdif

    filesize = os.path.getsize(filename)
    print('Dateigröße', filesize, 'Bytes')
    with open(filename, 'rb') as f:
        header = f.read(512)

    # Header dekodieren
    dtype = header[52:54].decode('utf-8') # z.B. 'CI'
    endianness = header[8:12].decode('utf-8') # sollte 'EEEI' sein! ab hier nehmen wir das an
    extended_header_start = int.from_bytes(header[24:28], byteorder='little') * 512 # in Bytes
    extended_header_size = int.from_bytes(header[28:32], byteorder='little')
    if extended_header_size != filesize - extended_header_start:
        print('Warnung: Extended-Header-Größe scheint falsch')
    time_interval = np.frombuffer(header[264:272], dtype=np.float64)[0]
    sample_rate = 1/time_interval
    print('Abtastrate', sample_rate/1e6, 'MHz')

    # IQ-Samples einlesen
    if dtype == 'CI':
        samples = np.fromfile(filename, dtype=np.int16, offset=512, count=(filesize-extended_header_size))
        samples = samples[::2] + 1j*samples[1::2] # in IQIQIQ... konvertieren

    # Jeden 1000sten Sample plotten, um auf Fehler zu prüfen
    print(len(samples))
    plt.plot(samples.real[::1000])
    plt.show()

Der „Extended Header" (d.h. die tailing bytes) mit seinen beliebigen Schlüssel/Wert-Paaren wird in einem Format beschrieben, das in Abschnitt 3.3 der `Blue File Format Spezifikation <https://web.archive.org/web/20150413061156/http://nextmidas.techma.com/nm/nxm/sys/docs/MidasBlueFileFormat.pdf>`_ festgelegt ist. Er enthält oft Informationen wie die HF-Frequenz, den Gain und den verwendeten Empfänger/SDR. Der Python-Code zum Dekodieren dieser Schlüssel/Wert-Paare ist unten dargestellt, angepasst von `diesem Code <https://github.com/tkzilla/rsa_api_sandbox/blob/master/cdif_reader.py>`_:

.. code-block:: python

    ...

    # Extended Header am Ende der Datei einlesen
    with open(filename, 'rb') as f:
        f.seek(filesize-extended_header_size)
        ext_header = f.read(extended_header_size)
        print("Länge des Extended Headers", len(ext_header), '\n')

    def parse_extended_header(idx):
        next_offset = np.frombuffer(ext_header[idx:idx+4], dtype=np.int32)[0]
        non_data_length = np.frombuffer(ext_header[idx+4:idx+6], dtype=np.int16)[0]
        name_length = ext_header[idx+6]
        dataStart = idx + 8
        dataLength = dataStart + next_offset - non_data_length
        midas_to_np = {'O' : np.uint8, 'B' : np.int8, 'I' : np.int16, 'L' : np.int32, 'X' : np.int64, 'F' : np.float32, 'D' : np.float64}
        format_code = chr(ext_header[idx+7])
        if format_code == 'A':
            val = ext_header[dataStart:dataLength].decode('latin_1')
        else:
            val = np.frombuffer(ext_header[dataStart:dataLength], dtype=midas_to_np[format_code])[0]
        key = ext_header[dataLength:dataLength+name_length].decode('latin_1')
        print(key, '  ', val)
        return idx + next_offset

    next_idx = 0
    while next_idx < extended_header_size:
        next_idx = parse_extended_header(next_idx)

Als Randnotiz: Blue Files und andere binäre IQ-Formate mit Metadaten und Daten in derselben Datei sind der Grund, warum SigMF eine Variante namens Non-Conforming Datasets (NCDs) enthält. Diese erlauben es, binäre IQ-Dateien mit zusätzlichen Bytes am Anfang und/oder Ende (für Metadaten) in ein SigMF-ähnliches Format zu bringen. Weitere Informationen findest du in den SigMF-Metadatenfeldern: dataset, header_bytes, trailing_bytes. Rein aus der Perspektive des Datenlesens können wir eine Blue File wie eine normale binäre IQ-Datei behandeln, solange wir die ersten 512 Bytes und alle Extended-Header-Bytes am Ende ignorieren.

Externe Ressourcen zu Blue Files:

#.  https://web.archive.org/web/20150413061156/http://nextmidas.techma.com/nm/nxm/sys/docs/MidasBlueFileFormat.pdf
#.  https://sigplot.lgsinnovations.com/html/doc/bluefile.html
#.  https://lgsinnovations.github.io/sigfile/bluefile.js.html
#.  http://nextmidas.com.s3-website-us-gov-west-1.amazonaws.com/
#.  https://web.archive.org/web/20181020012349/http://nextmidas.techma.com/nm/htdocs/usersguide/BlueFiles.html
#.  https://github.com/Geontech/XMidasBlueReader
