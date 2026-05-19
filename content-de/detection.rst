.. _detection-chapter:

#####################################################
Detektion mittels Korrelation
#####################################################

.. raw:: html

 <span style="display: table; margin: 0 auto; font-size: 20px;">Co-authored by <a href="https://www.linkedin.com/in/samuel-brown-vt">Sam Brown</a></span>

In diesem Kapitel lernen wir, wie wir das Vorhandensein von Signalen erkennen und ihre Zeitlage durch Kreuzkorrelation der empfangenen Samples mit einem bekannten Teil des Signals – beispielsweise einer Paket-Präambel – wiederherstellen können. Das führt natürlich zu einer einfachen Form der Klassifikation mithilfe einer Bank von Korrelatoren. Wir stellen die Grundideen der Signaldetektion vor und konzentrieren uns darauf, wie man entscheidet, ob ein bestimmtes Signal in einem verrauschten Umfeld vorhanden oder nicht vorhanden ist. Dabei behandeln wir Theorie und praktische Verfahren, um unter Unsicherheit gute Entscheidungen zu treffen.

****************************************************
Signaldetektion und Korrelator-Grundlagen
****************************************************

Signaldetektion ist die Aufgabe zu entscheiden, ob ein beobachteter Energiepeak ein bedeutungsvolles Signal oder nur Hintergrundrauschen ist.

Die Herausforderung: In Systemen wie Radar oder Sonar ist Rauschen allgegenwärtig. Ist der Detektor zu empfindlich, entstehen Fehlalarme. Ist er nicht empfindlich genug, werden echte Ziele übersehen.

Die Lösung beginnt mit dem Neyman-Pearson-Detektor, der einen mathematischen Mittelweg bietet, indem er die Wahrscheinlichkeit, ein Signal zu finden, maximiert und gleichzeitig Fehlalarme unter einem definierten Grenzwert hält. CFAR-Detektoren bauen auf dieser Idee auf, indem sie sich an Änderungen des Rauschpegels anpassen. Sie sind besonders nützlich, wenn die Rauschstatistik nicht stationär ist – d. h. wenn Rauschboden und Rauschverteilung durch Interferenz oder sich verändernde Kanalbedingungen variieren. Ziel ist es, die Detektionsschwelle automatisch anzupassen, wenn das Hintergrundrauschen schwankt, während eine gewählte Fehlalarmrate aufrechterhalten wird. Dazu muss der Rauschboden über die Zeit geschätzt werden.

Sobald ein System weiß, dass etwas vorhanden ist, muss es noch herausfinden, wo die Daten genau beginnen. Digitale Pakete in LTE, 5G oder WLAN beginnen mit einer Präambel – einem bekannten, sich wiederholenden digitalen Muster. Ein Präambel-Korrelator funktioniert wie ein Schloss-Schlüssel-Mechanismus: Der Schlüssel ist eine dem Empfänger bekannte Symbolfolge, die eindeutig für das empfangene Signal ist. Indem eine Kopie der Präambel über das eingehende Signal geschoben und bei jeder Verzögerung ein Skalarprodukt gebildet wird, misst der Empfänger, wie ähnlich die Vorlage den empfangenen Samples an jeder Position ist. Wenn beide gut übereinstimmen, erscheint ein scharfer Spike und teilt dem Empfänger mit, wo er die Daten zu lesen beginnen soll. Fortgeschrittenere Versionen berücksichtigen auch Frequenzversätze, die durch kleine Abstimmunterschiede zwischen Handy und Basisstation oder durch Doppler-Verschiebungen entstehen.

Wenn ein bekanntes Signal – oder eine Präambel – über einen nur durch additives weißes Gaußsches Rauschen (AWGN) gestörten Kanal übertragen wird, besteht die Aufgabe darin zu entscheiden, ob das Signal vorhanden ist. Dies ist das einfachste und grundlegendste Detektionsproblem.

Die Kreuzkorrelationsfunktion
###############################

Ein Korrelator ist in seiner einfachsten Form eine Kreuzkorrelation zwischen einem empfangenen Signal und einer Vorlage. Kreuzkorrelation ist lediglich ein Skalarprodukt zwischen zwei Vektoren, während einer der Vektoren über den anderen gleitet. Falls du Faltung kennst: Es ist fast dieselbe Operation, außer dass der zweite Vektor nicht gespiegelt wird – also etwas einfacher. Bei komplexen Signalen, mit denen wir es zu tun haben werden, muss einer der Eingänge zusätzlich konjugiert werden. In Python lässt sich das wie folgt implementieren:

.. code-block:: python

    def correlate(a, v):
        n = len(a)
        m = len(v)
        result = []
        for i in range(n - m + 1):
            s = 0
            for j in range(m):
                s += a[i + j] * v[j].conjugate()
            result.append(s)
        return result

    # Beispielverwendung:
    a = [1+2j, 2+1j, 3+0j, 4-1j, 5-2j]
    v = [0+1j, 1+0j, 0.5-0.5j]
    correlate(a, v)

Beachte, wie wir :code:`a` verschieben und dabei :code:`v` konjugieren, und wie die Schleife mit :code:`j` und :code:`s` im Grunde nur ein Vektor-Skalarprodukt ist. Glücklicherweise müssen wir Kreuzkorrelation nicht von Grund auf implementieren – in Python können wir NumPys :code:`correlate`-Funktion verwenden. Es gibt auch eine SciPy-Version, mit der man experimentieren kann.

Python-Beispiel einer Kreuzkorrelation
########################################################

Um ein grundlegendes Python-Beispiel eines Korrelatorszu erstellen, benötigen wir zunächst ein Beispielsignal mit einer bekannten Präambel, die in Rauschen eingebettet ist. Wir verwenden eine Zadoff-Chu-Sequenz als Präambel, wegen ihrer hervorragenden Auto-Korrelationseigenschaften und ihrer verbreiteten Nutzung in Kommunikationssystemen. Wir verwenden keinen weiteren Datenteil des Signals, obwohl in den meisten Systemen nach der bekannten Präambel unbekannte Daten folgen würden. Eine Zadoff-Chu-Sequenz kann wie folgt erzeugt werden:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    N = 839  # Länge der Zadoff-Chu-Sequenz
    u = 25  # Wurzel der ZC-Sequenz
    t = np.arange(N)
    zadoff_chu = np.exp(-1j * np.pi * u * t * (t + 1) / N)

Die resultierende Sequenz ist selbst ein Signal. Die IQ-Samples in :code:`zadoff_chu` repräsentieren ein Basisband-Komplexsignal, ähnlich vielen Signalen, die wir bereits in diesem Lehrbuch gesehen haben, aber es repräsentiert keine Bits. Wir können ein realistisches Szenario nachbilden, indem wir das Zadoff-Chu-Signal an einem zufälligen Offset in einen längeren AWGN-Strom einbetten:

.. code-block:: python

    signal_length = 10 * N # Gesamtlänge des simulierten Signals
    offset = np.random.randint(N, signal_length - N)
    print(f"True offset: {offset}")
    snr_db = -15
    noise_power = 1 / (2 * (10**(snr_db / 10)))
    signal = np.sqrt(noise_power/2) * (np.random.randn(signal_length) + 1j * np.random.randn(signal_length))
    signal[offset:offset+N] += zadoff_chu # ZC-Signal an zufälligem Offset einfügen

Beachte, dass wir ein sehr niedriges SNR verwenden. Es ist so niedrig, dass du die Zadoff-Chu-Sequenz im Zeitbereichssignal überhaupt nicht erkennen kannst. Unsere Sequenz ist 839 Samples lang, von insgesamt etwa 8.000 simulierten Samples, und ist so tief im Rauschen vergraben, dass man nicht einmal einen leichten Anstieg der Signalamplitude erkennen kann.

.. image:: ../_images/detection_basic_1.svg
   :align: center
   :target: ../_images/detection_basic_1.svg
   :alt: Time Domain Signal with Zadoff-Chu Sequence

Jetzt können wir den Korrelator implementieren, indem wir das empfangene Signal gegen unsere bekannte Zadoff-Chu-Sequenz mit :code:`np.correlate()` kreuzkorrelieren. Dies setzt voraus, dass der Empfänger die genaue verwendete Präambel kennt. Im obigen Code wurde :code:`zadoff_chu` ursprünglich zur Simulation des Signals erstellt, repräsentiert nun aber auch die vom Empfänger verwendete Präambel-Vorlage. Der Korrelator lässt sich in einer einzigen Python-Zeile implementieren:

.. code-block:: python

 correlation = np.correlate(signal, zadoff_chu, mode='valid')

Der :code:`valid`-Modus wird gleich behandelt. Wir normalisieren die Ausgabe auch durch die Länge der Sequenz und berechnen das Betragsquadrat, um die Leistung zu erhalten; auch der bloße Betrag würde funktionieren. Das Wichtigste ist die :code:`np.correlate()`-Operation selbst.

.. code-block:: python

 correlation = np.abs(correlation / N)**2 # durch N normalisieren und Betragsquadrat nehmen

Unten plotten wir das Betragsquadrat und markieren die tatsächliche Startposition der Sequenz, um zu sehen, ob der Korrelator sie finden konnte:

.. image:: ../_images/detection_basic_2.svg
   :align: center
   :target: ../_images/detection_basic_2.svg
   :alt: Correlator Output

Obwohl das SNR sehr niedrig ist, zeigt die Korrelatorausgabe einen deutlichen Spike genau dort, wo die Zadoff-Chu-Sequenz platziert wurde. Dieser Spike markiert den Beginn der Sequenz, sodass die 839 Samples ab dort die Präambel enthalten. Das ist die Kraft der korrelationsbasierten Detektion kombiniert mit einer langen Präambel. Bisher haben wir noch keine Schwelle festgelegt, um zu entscheiden, ob der Spike unser interessierendes Signal oder nur Rauschen ist – wir betrachten die Ausgabe nur visuell. Der Rest des Kapitels geht darum, diese Entscheidung zu automatisieren, insbesondere wenn Rauschboden und Hintergrundinterferenz sich ändern.

Gültig, Gleich, Voll-Modi
#######################################

Du hast vielleicht bemerkt, dass sowohl :code:`np.correlate()` als auch :code:`np.convolve()` drei Modi unterstützen: :code:`valid`, :code:`same` und :code:`full`. Diese Modi bestimmen die Länge des Ausgabe-Arrays relativ zu den Eingabe-Arrays. In unserem Fall haben wir :code:`valid` verwendet, was bedeutet, dass die Ausgabe nur Punkte enthält, bei denen die beiden Eingabe-Arrays vollständig überlappen. Dies ergibt eine Ausgabelänge von :code:`len(signal) - len(zadoff_chu) + 1`. Hätten wir :code:`same` verwendet, wäre die Ausgabe gleich lang wie das längere Eingabesignal. Bei :code:`full` erhält man die vollständige diskrete lineare Faltung mit einer etwas längeren Array-Länge von :code:`max(M, N) - min(M, N) + 1`, wobei :code:`M` und :code:`N` die Eingabelängen sind. In der HF-Signalverarbeitung wird Faltung oft zur Anwendung eines FIR-Filters verwendet, wobei Ein- und Ausgabe gleicher Länge praktisch ist, daher ist :code:`same` dort häufig. Für die korrelationsbasierte Detektion wollen wir jedoch üblicherweise :code:`valid`, da wir uns nur für Punkte interessieren, bei denen die Präambel vollständig mit dem empfangenen Signal überlappt, insbesondere wenn wir annehmen, dass das Signal nach dem Beginn des Empfangs startet.

Der Neyman-Pearson-Detektor
############################

Der Goldstandard für die Wahl einer Schwelle für unsere Korrelatorausgabe ist der Neyman-Pearson-Detektor. Diese Theorie hilft uns, unter einer bestimmten Einschränkung eine optimale Entscheidung zu treffen: Sie findet die Schwelle, die die Detektionswahrscheinlichkeit :math:`P_{D}` für eine feste und akzeptable Fehlalarmwahrscheinlichkeit :math:`P_{FA}` maximiert. Einfach gesagt: Du entscheidest, wie viele Fehldetektionen du tolerieren kannst – z. B. ein Fehlalarm pro Stunde –, und der Neyman-Pearson-Detektor gibt dir die beste Schwelle, um möglichst viele echte Signale zu finden. Zur Erkennung einer bekannten Präambel in AWGN verwendet er einen unkomplizierten Ansatz: Er berechnet die Korrelation zwischen dem empfangenen Signal und dem bekannten Präambelmuster. Überschreitet dieser Wert eine vorher festgelegte Schwelle :math:`\tau`, wird das Signal als vorhanden deklariert; andernfalls wird angenommen, dass nur Rauschen vorhanden ist.

Die Leistung dieses Detektors – gemessen durch :math:`P_{D}` und :math:`P_{FA}` – hängt von der Schwelle :math:`\tau`, dem SNR und der Präambellänge :math:`L` ab. Die Fehlalarmwahrscheinlichkeit hängt von der Schwelle und der Rauschvarianz :math:`\sigma_n^2` ab:

:math:`P_{FA} = Q\left(\frac{\tau}{\sigma_n}\right)`

Die Detektionswahrscheinlichkeit ist eine Funktion von Schwelle, Rauschvarianz und der Energie der Präambel (:math:`E_s = L \cdot S`, wobei :math:`S` die mittlere Symbolleistung ist):

:math:`P_{D} = Q\left(\frac{\tau - \sqrt{E_s}}{\sigma_n}\right) = Q\left(\frac{\tau - \sqrt{L \cdot S}}{\sigma_n}\right)`

Dabei ist :math:`Q(x)` die Q-Funktion (die Schwanzwahrscheinlichkeit der Standardnormalverteilung), die die Wahrscheinlichkeit angibt, dass eine standardnormalverteilte Zufallsvariable den Wert :math:`x` überschreitet.

Leistungsanalyse: ROC-Kurven und Pd-vs.-SNR-Kurven
#################################################################

Um die Leistung eines Korrelatordetektors im Rauschen zu quantifizieren, verwenden Ingenieure zwei hauptsächliche Visualisierungen: die Receiver Operating Characteristic (ROC)-Kurve und die Detektionswahrscheinlichkeit (:math:`P_{d}`) vs. SNR-Kurve.

Die ROC-Kurve trägt die Detektionswahrscheinlichkeit (:math:`P_{D}`) gegen die Fehlalarmwahrscheinlichkeit (:math:`P_{FA}`) bei festem SNR auf. Durch Anpassung der Detektionsschwelle am Korrelatorausgang wählt man einen Punkt auf dieser Kurve – es ist also ein grundlegender Kompromiss. Eine niedrigere Schwelle erhöht :math:`P_{D}`, indem mehr echte Signale gefunden werden, erhöht aber auch :math:`P_{FA}`, indem häufiger auf Rauschen angeschlagen wird. Die Wölbung der Kurve in Richtung der oberen linken Ecke zeigt einen besseren Detektor an. Ein perfekter Detektor erreicht die obere linke Ecke mit 100% :math:`P_{D}` und 0% :math:`P_{FA}`; eine Diagonale repräsentiert zufälliges Raten.

.. image:: ../_images/detection_pd_vs_snr.svg
   :align: center
   :target: ../_images/detection_pd_vs_snr.svg
   :alt: Pd vs SNR Curve and ROC curve

Zusammen zeigen die Gleichungen und die Intuition, dass die Präambellänge :math:`L` ein kritischer Entwurfsparameter ist, weil sie direkt den Verarbeitungsgewinn und damit die Detektionsleistung steuert. Bei fester Schwelle und festem SNR steigt :math:`P_{D}` mit :math:`L`. Eine längere Präambel erlaubt es, mehr Signalenergie zu sammeln, was es einfacher macht, das Signal vom Hintergrundrauschen zu unterscheiden. Diese Verbesserung wird Verarbeitungsgewinn genannt und üblicherweise in dB als :math:`10\log_{10}(L)` gemessen. Er ist entscheidend für die Erkennung schwacher Signale, die sonst übersehen würden. Durch Integration der Energie über mehr Samples können wir Signale aus dem Rauschen herausziehen, selbst wenn sie unterhalb des Rauschbodens liegen. GPS ist ein gutes Beispiel aus der Praxis, da der Empfänger sehr schwache Signale mit einer bekannten Codestruktur empfangen muss.

****************************************************
Beispiel: GPS-Signale unterhalb des Rauschbodens detektieren
****************************************************

Kurzeinführung in GPS-Signale
###############################

Stand März 2026 gibt es 31 operationelle Satelliten in der U.S.-GPS-Konstellation, die auf mittlerer Erdumlaufbahn (MEO) fliegen und die Erde zweimal täglich umkreisen. Alle Satelliten senden ein Signal bei 1575,42 MHz, genannt L1, und verwenden alle dieselbe Trägerfrequenz. Wenn das Signal die Erdoberfläche erreicht, ist es extrem schwach und liegt deutlich unterhalb des Rauschbodens. Orthogonalität zwischen den Satelliten wird dadurch erreicht, dass jedem Satelliten ein eindeutiger 1023-Chip-Pseudozufallsrausch-(PRN-)Code, der sogenannte C/A-Code, zugewiesen wird – deshalb wird das Signal manchmal als L1 C/A bezeichnet. Diese C/A-Codes verwenden Gold-Codes und sind so ausgelegt, dass je zwei davon nahezu orthogonal sind: Wenn man die Codes zweier Satelliten miteinander korreliert, erhält man nahezu null. Der C/A-Code läuft mit 1,023 Millionen Chips pro Sekunde und ist nur 1023 Chips lang, wiederholt sich also alle 1 ms. Zusätzlich moduliert jeder Satellit langsam Navigationsdaten wie Orbitposition und Uhrenkorrekturen mit nur 50 bit/s, sodass ein Datenbit 20 vollständige Codewiederholungen überspannt. Diese Verwendung eines verschiedenen Codes pro Sender ist als CDMA (Code Division Multiple Access) bekannt – dasselbe Prinzip wie bei 3G-Mobiltelefonen.

Auf der Empfängerseite bedeutet das Finden eines der 31 Satelliten, eine lokale Kopie der PRN-Sequenz dieses Satelliten zu erzeugen und einen Korrelator zu verwenden, um den Beginn der Codeperiode zu finden. Im GPS kann dieser Beginn wie der Anfang eines Pakets oder Rahmens behandelt werden, obwohl das System kontinuierlich sendet. Der genaue Korrelationspeak wird auch verwendet, um zu schätzen, wie weit das Signal gereist ist, bevor es den Empfänger erreicht hat; sobald das für vier oder mehr Satelliten bekannt ist, kann der Empfänger seinen Standort auf der Erde trilaterieren. Da sich die Satelliten mit etwa 4 km/s relativ zu dir bewegen, muss der Empfänger auch über ein Raster möglicher Frequenzversätze suchen, um den besten Korrelationspeak zu finden – ein 2D-Suchproblem. Der maximale Doppler beträgt etwa ±20 kHz (:code:`4e3 / 3e8 * 1.575e9`). Dieser Prozess wiederholt sich alle 1 ms, obwohl der Empfänger Verzögerung und Doppler verfolgt und keine vollständige Suche jedes Mal durchführen muss. Die anfängliche Suche nach jedem Satelliten wird als Akquisition bezeichnet, der anschließende Prozess als Tracking. Akquisition ist der rechenintensivere Teil und kann Minuten dauern, wenn der Empfänger ohne Vorinformationen über sichtbare Satelliten, Doppler-Verschiebungen oder seinen eigenen Standort startet.

Korrelationsansatz
###############################

Wir kreuzkorrelieren das eingehende Signal – in diesem Fall eine L1-Aufzeichnung – gegen eine lokal erzeugte Kopie des Codes jedes Satelliten. Ein großer Korrelationspeak bedeutet, dass der Satellit sichtbar ist und gibt uns den Beginn der 1-ms-Codeperiode. Um auch über Frequenz zu suchen, verwenden wir eine FFT-basierte Korrelation im Frequenzbereich, die es uns erlaubt, mehrere Frequenzversätze effizient zu testen, indem wir die FFT-Bins der lokalen Codereplika verschieben. Schließlich akkumulieren wir das Betragsquadrat der Korrelation über mehrere 1-ms-Blöcke, um das SNR zu verbessern. Das nennt man nicht-kohärente Integration, und sie hilft, GPS-Signale zu detektieren, die unterhalb des Rauschbodens empfangen werden. Wir schwellen das Ergebnis gegen die Korrelationsausgabe dividiert durch die mittlere Korrelationsleistung über alle Verzögerungen, was das Ergebnis normiert.

Beispielaufzeichnung
###############################

Wir verwenden eine Beispiel-GPS-Aufzeichnung von Daniel Estévez, die du `hier herunterladen <https://raw.githubusercontent.com/777arc/PySDR/refs/heads/master/figure-generating-scripts/GPS_L1_recording_10ms_4MHz_cf32.iq>`_ kannst. Es handelt sich um eine Complex-Float32-Datei, abgetastet mit 4 MHz und zentriert bei 1575,42 MHz.

Unten ist das Spektrogramm der Aufzeichnung. Es gibt nicht viel zu sehen, und die vertikale Linie ist nicht das eigentliche GPS-Signal – sie ist wahrscheinlich schmalbandige Interferenz. Die eigentlichen GPS-L1-Signale verwenden eine Chiprate von 1,023 MHz mit einem sehr langsamen Datensignal oben drauf, sodass das Signal etwa 2 MHz breit wird, was wir im Spektrogramm einfach nicht sehen. Das ist ein gutes Beispiel dafür, wie GPS-Signale weit unterhalb des Rauschbodens empfangen werden, und warum wir korrelationsbasierte Detektion benötigen, um sie zu finden.

.. image:: ../_images/detection_gps_spectrogram.svg
   :align: center
   :target: ../_images/detection_gps_spectrogram.svg
   :alt: Spectrogram of GPS L1 Recording

Für Interessierte: Diese Aufzeichnung ist ein kleiner Ausschnitt einer viel größeren Datei auf `IQEngine <https://iqengine.org/>`_ unter :code:`estevez/GPS and other GNSS`; suche nach der Aufzeichnung :code:`GPS-L1-2022-03-27`. Auf IQEngine ist es eine Int16-Datei im SigMF-Format.

Python-Beispiel
#####################

Passe :code:`filename` an den Speicherort an, wohin du die IQ-Datei heruntergeladen hast. Beachte, dass :code:`num_integrations` bestimmt, wie viel von der Aufzeichnung eingelesen und verarbeitet wird; die Gesamtdauer ist einfach diese Zahl mal 1 ms, maximal 10 für die kürzere Aufzeichnung.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    filename = "GPS_L1_recording_10ms_4MHz_cf32.iq"
    sample_rate = 4e6
    chip_rate = 1023000 # Chips/s (Teil der GPS-Spezifikation)
    num_chips = 1023 # Chips pro C/A-Codeperiode
    samples_per_code = int(round(sample_rate / chip_rate * num_chips))  # Genaue Sample-Anzahl in einer 1-ms-Codeperiode bei 4 MHz
    doppler_min_hz = -5e3 # GPS-Doppler ≈ ±4 kHz für stationären Empfänger
    doppler_max_hz = 5e3
    doppler_step_hz = 500 # gut genug für eine grobe Suche
    num_integrations = 10 # nicht-kohärente Leistungsintegrationen (also 10 ms gesamt), bestimmt wie viel der IQ-Aufzeichnung eingelesen wird!
    detection_thresh_dB =  14.0 # Peak-to-Mean-Ratio (PMR)-Schwelle in dB für eine Detektion, GPS C/A-Signale haben typischerweise 14-20 dB PMR bei 10 ms Integration
    gps_svs = list(range(1, 33)) # 1–32

    ##### C/A-Code-Erzeugung #####
    # Der GPS C/A-Code ist ein Gold-Code, gebildet durch XOR zweier 10-stufiger
    # maximallängen Schieberegister (G1 und G2). G2 wird effektiv um eine
    # satellitenspezifische Anzahl von Chips vor dem XOR verzögert.
    # Referenz: IS-GPS-200, Tabelle 3-Ia
    G2_DELAY = [ # G2-Phasenverzögerung (Chips) für gps_svs 1–32
        5,   6,   7,   8,  17,  18, 139, 140,   #  1– 8
        141, 251, 252, 254, 255, 256, 257, 258,   #  9–16
        469, 470, 471, 472, 473, 474, 509, 512,   # 17–24
        513, 514, 515, 516, 859, 860, 861, 862,   # 25–32
    ]

    """G1 LFSR: Polynom x^10 + x^3 + 1, Alles-Einsen-Init, Ausgabe bei Stufe 10."""
    reg = np.ones(10, dtype=np.int8)
    G1 = np.empty(num_chips, dtype=np.int8)
    for i in range(num_chips):
        G1[i] = reg[9]
        fb = reg[2] ^ reg[9] # Stufen 3 und 10 (0-indiziert: 2 und 9)
        reg = np.roll(reg, 1)
        reg[0] = fb

    """G2 LFSR: Polynom x^10+x^9+x^8+x^6+x^3+x^2+1, Alles-Einsen-Init."""
    reg = np.ones(10, dtype=np.int8)
    G2 = np.empty(num_chips, dtype=np.int8)
    for i in range(num_chips):
        G2[i] = reg[9]
        fb = reg[1]^reg[2]^reg[5]^reg[7]^reg[8]^reg[9]  # Abgriffe 2,3,6,8,9,10
        reg = np.roll(reg, 1)
        reg[0] = fb

    # 1023-Chip C/A PRN-Code für SV sv (1-32) als float32, 1 und -1, also BPSK
    def make_prn(sv: int) -> np.ndarray:
        g2_delayed = np.roll(G2, G2_DELAY[sv - 1])
        bits = G1 ^ g2_delayed           # {0, 1}
        return (1 - 2 * bits).astype(np.float32)   # BPSK: {+1, −1}

    def upsample_prn(sv: int) -> np.ndarray:
        """Nächster-Nachbar-Upsampling des 1023-Chip C/A-Codes → samples_per_code Samples."""
        code = make_prn(sv)
        idx = (np.arange(samples_per_code) * num_chips / samples_per_code).astype(int)
        return code[idx]

    # Vorberechnete Template-Signale – konjugierte FFTs aller upgesampleten PRN-Codes
    template_signals = {sv: np.conj(np.fft.fft(upsample_prn(sv))) for sv in gps_svs}

    # IQ-Datei einlesen
    n_needed = samples_per_code * num_integrations
    iq = np.fromfile(filename, dtype=np.complex64, count=n_needed)
    # Für die vollständige Version von IQEngine stattdessen folgendes verwenden:
    #iq = np.fromfile(filename, dtype=np.int16, count=n_needed * 2)
    #iq = (iq[0::2] + 1j * iq[1::2]).astype(np.complex64)

    # Jeden Satelliten über Doppler und Codephase durchsuchen
    results = []
    detected = []
    print(f"  {'SV':>3}  {'Doppler (Hz)':>13}  {'Phase (chips)':>14}"
            f"  {'Phase (samp)':>13}  {'Delay (µs)':>11}  {'PMR (dB)':>9}")
    doppler_bins = np.arange(doppler_min_hz, doppler_max_hz + doppler_step_hz, doppler_step_hz)
    for sv in gps_svs:
        corr_map = np.zeros((len(doppler_bins), samples_per_code))
        n_total = samples_per_code * num_integrations
        for di, f_d in enumerate(doppler_bins):
            t = np.arange(n_total) / sample_rate # Zeitvektor
            mixed = iq[:n_total] * np.exp(-2j*np.pi*float(f_d)*t) # Frequenzverschiebung anwenden

            # Betragsquadrat der Korrelation nicht-kohärent akkumulieren
            for k in range(num_integrations):
                blk = mixed[k * samples_per_code:(k + 1) * samples_per_code]
                sig_fft = np.fft.fft(blk)
                corr = np.fft.ifft(sig_fft * template_signals[sv]) # Korrelation im Frequenzbereich
                corr_map[di] += np.abs(corr)**2

        # Durch Mittelwert normalisieren und in dB umrechnen
        peak_val = float(np.max(corr_map))
        mean_val = float(np.mean(corr_map))
        pmr_db = 10.0 * np.log10(peak_val / mean_val)

        peak_idx = np.unravel_index(np.argmax(corr_map), corr_map.shape)
        best_doppler_hz   = float(doppler_bins[peak_idx[0]])
        best_phase_samp   = int(peak_idx[1])
        best_phase_chips  = best_phase_samp * num_chips / samples_per_code

        r = {
            "sv": sv,
            "detected": pmr_db >= detection_thresh_dB,
            "doppler_hz": best_doppler_hz,
            "code_phase_samp": best_phase_samp, # Sample-Offset = "Paketbeginn"
            "code_phase_chip": best_phase_chips,
            "pmr_db": pmr_db,
            "corr_map": corr_map,
            "doppler_bins": doppler_bins,
        }
        results.append(r)

        # Ergebniszeile ausgeben
        delay_us = r['code_phase_samp'] / sample_rate * 1e6
        flag = "  ← DETECTED" if r['detected'] else ""
        print(f"  {sv:>3}  {r['doppler_hz']:>+13.0f}  {r['code_phase_chip']:>14.2f}"
            f"  {r['code_phase_samp']:>13d}  {delay_us:>11.3f}  {r['pmr_db']:>9.1f}{flag}")

Dies sollte folgende Ausgabe erzeugen:

.. code-block::

   SV   Doppler (Hz)   Phase (chips)   Phase (samp)   Delay (µs)   PMR (dB)
    1          -3000          757.79           2963      740.750        5.6
    2          +1500          264.19           1033      258.250        9.1
    3          -2000          316.62           1238      309.500        5.8
    4          +5000          577.48           2258      564.500        5.0
    5          +1000           64.96            254       63.500        5.3
    6          +1500          511.76           2001      500.250        5.0
    7          -4000          763.41           2985      746.250        5.0
    8          +3500          961.62           3760      940.000        5.4
    9          +3500          118.67            464      116.000        4.9
   10             +0          890.52           3482      870.500        5.4
   11          +2500          837.33           3274      818.500       14.6  ← DETECTED
   12           -500          871.60           3408      852.000       16.4  ← DETECTED
   13          +1000          137.85            539      134.750        5.9
   14          +2500          287.72           1125      281.250        5.0
   15          -5000          908.68           3553      888.250        5.3
   16          +1500          292.58           1144      286.000        5.9
   17           +500          994.61           3889      972.250        5.3
   18          +4500         1005.61           3932      983.000        5.4
   19          +5000          588.48           2301      575.250        5.0
   20             +0          768.53           3005      751.250        5.4
   21          -3000          749.60           2931      732.750        5.0
   22          +2500          558.05           2182      545.500       14.4  ← DETECTED
   23          -5000          390.02           1525      381.250        5.3
   24          +2500          955.48           3736      934.000        5.9
   25          +1500          597.94           2338      584.500       15.5  ← DETECTED
   26          -1500          239.89            938      234.500        6.2
   27          -2500          488.74           1911      477.750        4.7
   28          +3000          858.81           3358      839.500        5.2
   29          -4000          998.70           3905      976.250        5.2
   30          -2000          937.58           3666      916.500        5.2
   31          +5000          463.42           1812      453.000       15.9  ← DETECTED
   32          +1000          342.45           1339      334.750       16.2  ← DETECTED

Wie man sieht, wurden 6 Satelliten detektiert. Obwohl unsere Schwelle 14,0 war, kann man aus dieser Liste leicht erkennen, dass die meisten anderen Satelliten nicht sichtbar waren, mit Ausnahme von SV-2, der wahrscheinlich sichtbar war, aber die Schwelle knapp nicht erreicht hat. Wer das verifizieren möchte: Die Aufzeichnung wurde am 27.03.2022T11:32:04 irgendwo in Spanien aufgenommen.

Darstellung
###########

Versuchen wir, die Ergebnisse für Satellit 11 – den ersten detektierten – darzustellen. Das erste Diagramm ist die 2D-Korrelationskarte über Doppler und Zeit/Verzögerung, das zweite ist ein Schnitt der Korrelationskarte beim besten Doppler-Bin, der die Korrelationsleistung über die Zeit zeigt, wie wir es im vorherigen Abschnitt gesehen haben.

.. code-block:: python

    # Darstellung
    sv = 11 # wir haben 11, 12, 22, 25, 31, 32 detektiert – versuche auch einen nicht detektierten!
    r = results[sv - 1] # Dict mit Ergebnissen für diesen SV ausgeben
    cmap = r['corr_map'] # 2D-Array der Korrelationsleistung vs. Doppler und Codephase
    d_bins = r['doppler_bins'] # zugehörige Doppler-Bins
    chips_axis = np.arange(samples_per_code) * num_chips / samples_per_code

    # 2D Doppler × Codephase-Karte
    plt.figure(0, figsize=(10, 6))
    im = plt.pcolormesh(chips_axis, d_bins, cmap, shading='auto', cmap='viridis')
    plt.xlabel("Code Phase (chips)")
    plt.ylabel("Doppler (Hz)")
    plt.title(f"SV {sv}  —  2-D Acquisition Map  (PMR = {r['pmr_db']:.1f} dB)")
    plt.legend(fontsize=8, loc='upper right')
    plt.colorbar(im, label="Correlation Power")

    # Codephase-Schnitt beim besten Doppler
    best_di = int(np.argmin(np.abs(d_bins - r['doppler_hz'])))
    plt.figure(1, figsize=(10, 6))
    plt.plot(chips_axis, cmap[best_di], lw=1, color='steelblue')
    plt.xlabel("Code Phase (chips)")
    plt.ylabel("Correlation Power")
    plt.title(f"SV {sv}  —  Code-Phase Slice  (Doppler = {r['doppler_hz']:+.0f} Hz)")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    plt.show()

.. image:: ../_images/detection_gps_2d_map.png
   :align: center
   :width: 700px
   :alt: 2-D Acquisition Map

.. image:: ../_images/detection_gps_code_phase_slice.svg
   :align: center
   :target: ../_images/detection_gps_code_phase_slice.svg
   :alt: Code-Phase Slice

Auf die Trilateration gehen wir hier nicht ein, aber die genaue Position dieses Spikes ist letztendlich das, was dem GPS-Empfänger ermöglicht, die Entfernung zum Satelliten zu bestimmen. Kombiniert mit denselben Informationen von vier oder mehr Satelliten kann er seinen Standort auf der Erde ermitteln.

****************************************************
CFAR-Detektoren: In dynamischen Umgebungen bestehen
****************************************************

Während der Neyman-Pearson-Detektor bei festem Rauschpegel optimal ist, sind reale Bedingungen selten so stabil. In einer dynamischen Umgebung – wie einem Radar, das ein Flugzeug durch Regen verfolgt, oder einem Funkempfänger in einer belebten Stadt – schwanken Hintergrundrauschen und Interferenzpegel ständig. Hier wird der CFAR-Detektor (Constant False Alarm Rate) unverzichtbar.

CFAR-Detektoren sind die Arbeitspferde von Systemen, in denen ein unvorhersehbarer Hintergrund eine feste Schwelle unmöglich macht:

- Radar und Sonar werden zur Erkennung von Zielen (Flugzeuge, U-Boote) gegen „Clutter" verwendet – Reflexionen von Wellen, Regen oder Land, die sich ändern, wenn sich der Sensor bewegt.
- Drahtlose Kommunikation, z. B. Cognitive Radio und LTE/5G-Systeme, nutzen CFAR, um verfügbares Spektrum zu identifizieren oder eingehende Pakete zu detektieren, wenn Interferenz von anderen Geräten stoßweise und unvorhersehbar ist.
- Medizinische Bildgebung setzt CFAR in automatisierter Ultraschall- oder MRT-Analyse ein, um echte Gewebemerkmale von variierenden elektronischen Rauschpegeln zu unterscheiden.

Das „C" in CFAR steht für Constant, weil das Ziel darin besteht, die Fehlalarmwahrscheinlichkeit (:math:`P_{FA}`) auf einem stabilen, vorhersehbaren Niveau zu halten.

Um eine Schwelle zu setzen, muss man ein statistisches Modell für das Rauschen annehmen – die Rauschverteilung. Bei einfachem AWGN folgt Rauschen einer Gauß-Verteilung. Bei Radar-Clutter könnte es jedoch einer Rayleigh- oder Weibull-Verteilung folgen. Wenn dein Modell falsch ist, „driftet" dein :math:`P_{FA}` – das System wird entweder blind oder von Fehlauslösungen überwältigt.

Statt eines fest codierten Wertes schätzt ein CFAR-Detektor die Rauschleistung in der lokalen „Nachbarschaft" des Signals und multipliziert diese Schätzung mit einem Skalierungsfaktor (:math:`T`), der von deinem gewünschten :math:`P_{FA}` abgeleitet wird. Dadurch wird sichergestellt, dass die Schwelle mit dem Rauschboden steigt, wenn dieser ansteigt.

Pro-Lag- vs. systemweite Fehlalarmraten
####################################################

Dies ist eine wichtige Unterscheidung, die Anfänger häufig übersehen. Wenn du nach einer Präambel suchst, führst du üblicherweise eine gleitende Korrelation durch und prüfst die Schwelle bei tausenden verschiedenen Zeitversätzen (oder „Lags") pro Sekunde.

Pro-Lag-:math:`P_{FA}`: Das ist die Wahrscheinlichkeit, dass eine einzelne spezifische Korrelationsprüfung einen Fehlalarm ergibt. Wenn du deine Mathematik für ein :math:`P_{FA}` von 0,001 einstellst, hat jeder einzelne Lag eine 1-zu-1.000-Chance, ein „Geistersignal" zu sein.

Systemweites (globales) :math:`P_{FA}`: Das ist die Wahrscheinlichkeit, dass das System mindestens einen Fehlalarm während eines gesamten Suchfensters auslöst (z. B. über 2.048 Lags).

Mathematisch gesehen gilt: Wenn dein Pro-Lag-:math:`P_{FA}` :math:`p` ist, beträgt die Wahrscheinlichkeit mindestens eines Fehlalarms über :math:`N` Lags ungefähr :math:`1-(1-p)^{N}`.

Folglich: Wenn du 1.000 Lags und ein Pro-Lag-:math:`P_{FA}` von 0,001 hast, wird dein System fast 63% der Zeit einen Fehlalarm melden! Um die systemweite Fehlalarmrate niedrig zu halten, muss das Pro-Lag-:math:`P_{FA}` auf einen extrem kleinen Wert gesetzt werden.

Python-Beispiel
###############

Um mit unserem eigenen CFAR-Detektor zu spielen, simulieren wir zunächst ein Szenario mit sich wiederholenden QPSK-Paketen mit einer bekannten Präambel über einen Kanal mit zeitlich variierendem Rauschboden. Dann implementieren wir einen einfachen Cell-Averaging CFAR (CA-CFAR)-Algorithmus zur Detektion der Präambeln im empfangenen Signal. Der folgende Python-Code erzeugt das empfangene Signal:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import correlate

    def generate_qpsk_packets(num_packets, sps, preamble):
        """Erzeugt sich wiederholende QPSK-Pakete mit Lücken und variablem Rauschen."""
        qpsk_map = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        data_len = 200
        gap_len = 100
        full_signal = []

    # Upgesampelte Präambel für Korrelation vorberechnen
        upsampled_preamble = np.repeat(preamble, sps)

        for _ in range(num_packets):
            data = qpsk_map[np.random.randint(0, 4, data_len)]
            packet = np.concatenate([preamble, data])
            full_signal.extend(np.repeat(packet, sps))
            full_signal.extend(np.zeros(gap_len * sps))

        return np.array(full_signal), upsampled_preamble

    # Simulationsparameter
    sps = 4
    preamble_syms = np.array([1+1j, 1+1j, -1-1j, -1-1j, 1-1j, -1+1j]) / np.sqrt(2)
    tx_signal, ref_preamble = generate_qpsk_packets(5, sps, preamble_syms)

    # Zeitlich variierender Rauschboden
    t = np.arange(len(tx_signal))
    noise_env = 0.05 + 0.3 * np.sin(2 * np.pi * 0.0003 * t)**2
    noise = (np.random.randn(len(tx_signal)) + 1j*np.random.randn(len(tx_signal))) * noise_env
    rx_signal = tx_signal + noise

Der erste Schritt ist eine einzelne Korrelation des empfangenen Signals gegen die bekannte Präambel (in der Praxis wird das meist in Batches von Samples durchgeführt, aber wir machen es in einem Batch):

.. code-block:: python

    # Korrelationsspike erscheint, wenn die Referenz mit dem empfangenen Segment übereinstimmt
    corr_out = correlate(rx_signal, ref_preamble, mode='same')
    corr_power = np.abs(corr_out)**2

TODO: nur die rohe Ausgabe dieses Schrittes betrachten

Jetzt implementieren wir den CFAR-Detektor, wenden ihn auf die Korrelatorausgabe an und visualisieren die Ergebnisse:

.. code-block:: python

    # CFAR-Detektion auf der Korrelatorausgabe
    def ca_cfar_adaptive(data, num_train, num_guard, pfa):
        num_cells = len(data)
        thresholds = np.zeros(num_cells)
        alpha = num_train * (pfa**(-1/num_train) - 1)  # Skalierungsfaktor
        half_window = (num_train + num_guard) // 2
        guard_half = num_guard // 2
        for i in range(half_window, num_cells - half_window):
            # Trainingsbereich um die Testzelle aufbauen
            lagging_win = data[i - half_window : i - guard_half]
            leading_win = data[i + guard_half + 1 : i + half_window + 1]
            noise_floor_est = np.mean(np.concatenate([lagging_win, leading_win]))
            thresholds[i] = alpha * noise_floor_est
        return thresholds

    # Peaks in Korrelationsleistung detektieren
    cfar_thresholds = ca_cfar_adaptive(corr_power, num_train=60, num_guard=20, pfa=1e-5)
    detections = np.where(corr_power > cfar_thresholds)[0]
    # Randdetektionen entfernen, wo die Schwelle undefiniert ist
    detections = detections[cfar_thresholds[detections] > 0]

    # Teilgrafik 1: empfangenes Signal und rohe Leistung
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.plot(np.abs(rx_signal)**2, color='gray', alpha=0.4, label='Rx Signal Power ($|r(t)|^2$)')
    plt.title("Time-Domain Received Signal")
    plt.ylabel("Power")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Teilgrafik 2: Korrelatorausgabe vs. adaptive Schwelle
    plt.subplot(2, 1, 2)
    plt.plot(corr_power, label='Correlator Output $|r(t) * p^*(-t)|^2$', color='blue')
    plt.plot(cfar_thresholds, label='CFAR Adaptive Threshold', color='red', linestyle='--', linewidth=1.5)
    if len(detections) > 0: # Detektionen einblenden
        plt.scatter(detections, corr_power[detections], color='lime', edgecolors='black', label='Detections (Preamble Found)', zorder=5)
    plt.title("Preamble Correlator Output with Adaptive CFAR Threshold")
    plt.xlabel("Sample Index")
    plt.ylabel("Correlation Power")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

.. image:: ../_images/detection_cfar.svg
   :align: center
   :target: ../_images/detection_cfar.svg
   :alt: CFAR Detector Output Example



Frequenzversatz-robuste Präambel-Korrelatoren
####################################################

Die Detektion einer Präambel wird zu einem mehrdimensionalen Suchproblem, wenn die Mittenfrequenz unbekannt ist. In einem perfekt synchronisierten System wirkt ein kohärenter Korrelator als Matched Filter und maximiert das SNR. Frequenzversätze führen jedoch zu einer zeitlich variierenden Phasenrotation, die das Signal von der lokalen Vorlage dekorreliert und zu einer erheblichen Verschlechterung der Detektionsempfindlichkeit führt.

Der Einfluss eines Frequenzversatzes :math:`\Delta f` hängt von seiner Größe relativ zur Präambeldauer (:math:`T_{p}`) ab:

Leicht versetzte Signale, wie sie durch Doppler oder Frequenzdrift verursacht werden, entstehen typischerweise durch LO-ppm-Ungenauigkeiten oder langsame Bewegung. In diesem Fall gilt :math:`\Delta f \cdot T_{p} \ll 1`. Der Korrelationspeak wird leicht abgeschwächt, aber die Zeitlage kann noch wiederhergestellt werden.

Wenn der Frequenzversatz völlig unbekannt ist – z. B. bei der Kaltstart-Satellitenakquisition oder Hochdynamik-UAV-Links –, kann die kohärente Summe auf null fallen, wenn die Phase sich um mehr als :math:`180^{\circ}` über die Präambel dreht (:math:`\Delta f > 1/(2T_{p})`). In diesem Fall wird Detektion unabhängig vom SNR unmöglich.

Der Verlust in der Korrelationsamplitude durch einen Frequenzversatz wird durch den Dirichlet-Kernel (oder die periodische sinc-Funktion) beschrieben. Mit zunehmendem Frequenzversatz folgt die kohärente Summe rotierter Vektoren diesem sinc-artigen Abfall.

Der Verlust in dB durch den Frequenzversatz lässt sich durch folgende Formel approximieren:

:math:`L_{dB}(\Delta f) = 20 \log_{10} \left| \frac{\sin(\pi \Delta f N T_{s})}{N \sin(\pi \Delta f T_{s})} \right|`

Dabei gilt:

   - :math:`N`: Anzahl der Symbole in der Präambel.
   - :math:`T_{s}`: Symboldauer.
   - :math:`\Delta f`: Frequenzversatz in Hz.

Mit zunehmendem :math:`\Delta f` oszilliert der Zähler, während der Nenner wächst – das erzeugt Nullstellen in der Detektorempfindlichkeit. Für einen Standard-Korrelator liegt die erste Nullstelle bei :math:`\Delta f = 1/(N T_{s})`. Bei einem halben Bin-Versatz verlierst du ungefähr 3,9 dB, was dein effektives SNR und :math:`P_{d}` erheblich verschlechtert.

Methoden zur Robustheit gegenüber Frequenzversätzen
###########################################

A. Kohärenter segmentierter Korrelator

Die Präambel der Länge :math:`N` wird in :math:`M` Segmente der Länge :math:`L = N/M` aufgeteilt. Jedes Segment wird kohärent korreliert, und die Ergebnisse werden durch Kompensation der Phasendrift zwischen den Segmenten kombiniert.

:math:`Y_{coh} = \sum_{m=0}^{M-1} \left( \sum_{k=0}^{L-1} r[k+mL] \cdot p^{*}[k] \right) e^{-j \hat{\phi}_m}`

Dabei ist :math:`\hat{\phi}_m` eine Schätzung der Phasenrotation für dieses Segment. Das bewahrt den SNR-Gewinn einer vollständigen Präambel, erfordert aber eine genaue Frequenzschätzung zur Phasenausrichtung.

B. Nicht-kohärenter segmentierter Korrelator

Segmente werden kohärent korreliert, aber ihre Beträge werden summiert – Phaseninformation wird verworfen.

:math:`Y_{non-coh} = \sum_{m=0}^{M-1} \left| \sum_{k=0}^{L-1} r[k+mL] \cdot p^{*}[k] \right|^{2}`

Dieser Ansatz ist extrem robust gegenüber Frequenzversätzen (bis zu :math:`1/(L T_{s})`). Er leidet jedoch unter dem nicht-kohärenten Integrationsverlust: Das Summieren von Beträgen statt komplexer Werte lässt Rauschen schneller akkumulieren als das Signal und reduziert effektiv das „Post-Detektions"-SNR.

C. Brute-Force-Frequenzsuche

Der Empfänger betreibt mehrere parallele Korrelatoren, jeweils um eine diskrete Frequenz :math:`\Delta f_{i}` verschoben.

Diese Methode liefert die beste SNR-Leistung (voller kohärenter Gewinn), ist aber am rechenintensivsten. Der „Bin-Abstand" muss eng genug sein (basierend auf der Dirichlet-Formel), um sicherzustellen, dass der schlechteste Verlust zwischen den Bins akzeptabel ist (z. B. < 1 dB).

Bei Zeitbereichs-Abtastung werden Samples mit einem festen Satz von Gewichten gefaltet. Bei einer Frequenzsuche erfordert das eine separate FIR-Bank für jedes Frequenzbin – effizient für kurze Präambeln auf FPGAs mit Xilinx-DSP48-Slices.
Frequenzbereichsverarbeitung (FFT): Zur Durchführung einer Suche nimmt man die FFT des eingehenden Signals und der Präambel. Multiplikation im Frequenzbereich entspricht Korrelation.
Der „Frequenzverschiebungs-Trick": Um verschiedene Frequenzversätze zu testen, benötigt man nicht mehrere FFTs – man kann einfach die FFT-Bins der Präambel relativ zum Signal zirkulär verschieben, bevor man die punktweise Multiplikation und IFFT durchführt.
Für kontinuierliche Datenströme werden Chunking-Methoden wie Overlap-Save oder Overlap-Add verwendet, um Daten in Blöcken zu verarbeiten, ohne Korrelationspeaks an den Rändern der FFT-Fenster zu verlieren.

Robustheit gegenüber Frequenzversätzen ist ein Kompromiss zwischen Verarbeitungsgewinn und Rechenkomplexität. Nicht-kohärente segmentierte Korrelation ist die robusteste Wahl in Umgebungen mit hoher Unsicherheit, erfordert aber höhere Linkmargen. Kohärente segmentierte und Brute-Force-FFT-Suchen bieten bessere Empfindlichkeit, benötigen aber erheblich mehr Hardware-Ressourcen. Das Verständnis des Dirichlet-getriebenen Verlusts ist entscheidend bei der Wahl der Bin-Dichte für jeden frequenzsuchenden Empfänger.

TODO: Diesen Plot erklären und einen Teil des Python-Codes dem Abschnitt hinzufügen

.. image:: ../_images/detection_freq_offset.svg
   :align: center
   :target: ../_images/detection_freq_offset.svg
   :alt: Frequency Offset Impact on Correlation

*****************************************************************
Detektion von Direct Sequence Spread Spectrum (DSSS)-Signalen
*****************************************************************

In einem DSSS-System (Direct Sequence Spread Spectrum) ist der Korrelationsdetektor das Bindeglied, das ein bedeutungsvolles Signal aus dem herausarbeitet, was zunächst wie zufälliges Rauschen aussieht. Durch Verwendung einer Chip-Sequenz mit hoher Rate – oder einem Chipping-Code – verteilt das System die Signalenergie über eine viel größere Bandbreite als das ursprüngliche Datensignal benötigt. Da die Gesamtleistung konstant bleibt, sinkt durch die Verteilung über einen breiteren Frequenzbereich die Leistungsspektraldichte (PSD). Dieser Spektralverdünnungseffekt kann den Signalpegel unter den thermischen Rauschboden treiben und ihn für konventionelle Schmalband-Empfänger nahezu unsichtbar machen. Für den vorgesehenen Empfänger kann jedoch dieselbe Chip-Sequenz angewendet werden, um das Signal zu de-spreizen – die Energie wird in die ursprüngliche schmale Bandbreite zurückkonzentriert, während gleichzeitig Schmalband-Interferenz gespreizt wird. Das ermöglicht eine zuverlässige Detektion selbst in sehr verrauschten Umgebungen. Der nächste Unterabschnitt befasst sich mit dem Timing-Aspekt dieses Problems.

Die Rolle von Auto-Korrelationseigenschaften
########################################

Die Wahl der richtigen Sequenz ist entscheidend für Synchronisation und Mehrwegausbreitung-Unterdrückung. Idealerweise sollte eine Sequenz perfekte Auto-Korrelation haben: einen hohen Peak bei perfekter Ausrichtung und nahezu null Werte bei jedem anderen Zeitversatz. Scharfe Auto-Korrelationspeaks erlauben dem Empfänger, das Signal mit Sub-Chip-Timing-Genauigkeit zu erfassen. Wenn ein Signal von einem Gebäude reflektiert wird und verzögert ankommt, sorgen gute Auto-Korrelationseigenschaften dafür, dass der Empfänger die verzögerte Version als unkorreliertes Rauschen behandelt statt als destruktive Interferenz, was Mehrwege-Ausbreitung abmildert.


Häufige Spreizsequenzen
##########################

Verschiedene Anwendungen erfordern unterschiedliche mathematische Eigenschaften in ihren Spreizsequenzen. Einige Beispiele:

- Barker-Codes sind bekannt für die bestmöglichen Auto-Korrelationseigenschaften für kurze Längen (bis 13) und werden berühmtlich in 802.11b WLAN verwendet.
- M-Sequenzen (maximale Länge), erzeugt mit linearen Rückkopplungs-Schieberegistern (LFSRs), bieten ausgezeichnete Zufälligkeit und Auto-Korrelation über sehr lange Perioden.
- Gold-Codes, abgeleitet aus Paaren von M-Sequenzen, bieten eine große Menge von Sequenzen mit kontrollierter Kreuzkorrelation und sind der Standard für GPS und CDMA, wo mehrere Signale koexistieren müssen.
- Zadoff-Chu (ZC)-Sequenzen sind komplexwertige Sequenzen mit konstanter Amplitude und null Auto-Korrelation für alle Nicht-null-Verschiebungen und sind inzwischen ein Standard in LTE und 5G für Synchronisation.
- Kasami-Codes ähneln Gold-Codes, haben aber noch niedrigere Kreuzkorrelation für eine gegebene Sequenzlänge, was sie in dichten Umgebungen nützlich macht.

Chip-Timing-Synchronisation in DSSS
####################################################

In einem DSSS-System hängt die Fähigkeit des Empfängers, Daten wiederherzustellen, vollständig von der Synchronisation mit der eingehenden Chip-Sequenz ab. Da Chips viel kürzer als Datenbits sind, kann selbst ein kleiner fraktionaler Timing-Fehler – bei dem der Empfänger zwischen Chips abtastet – den Korrelationspeak erheblich reduzieren. Wir können den Einfluss eines fraktionalen Timing-Versatzes untersuchen, indem wir ein einfaches DSSS-System simulieren und die Korrelationsausgabe darstellen, während der Timing-Versatz von 0 auf 1 Chip variiert. Beachte, dass wir hier keine vollständige Korrelation durchführen – wir nehmen nur ein Skalarprodukt bei Lag 0, weil wir bereits wissen, dass dort der Peak liegt.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    # Barker-11-Sequenz
    barker11 = np.array([1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1])
    samples_per_chip = 100

    # Sequenz upsampeln, um kontinuierliche Zeit zu simulieren
    sig = np.repeat(barker11, samples_per_chip)

    offsets = np.linspace(-1.5, 1.5, 500) # Fraktionale Chip-Versätze
    peaks = []

    for offset in offsets:
        # Signal um fraktionale Anzahl von Chips verschieben, umgerechnet in Samples
        shift_samples = int(offset * samples_per_chip)
        if shift_samples > 0:
            shifted_sig = np.pad(sig, (shift_samples, 0))[:len(sig)]
        elif shift_samples < 0:
            shifted_sig = np.pad(sig, (0, abs(shift_samples)))[abs(shift_samples):]
        else:
            shifted_sig = sig

        # Normierte Korrelation bei Lag 0 für diesen Versatz berechnen
        correlation = np.vdot(sig, shifted_sig) / np.vdot(sig, sig)
        peaks.append(np.abs(correlation))

    plt.figure(figsize=(10, 5))
    plt.plot(offsets, peaks, label='Normalized Correlation', color='blue', linewidth=2)
    plt.axvline(0, color='red', linestyle='--', alpha=0.5, label='Perfect Alignment')
    plt.title('DSSS Correlation Peak vs. Fractional Chip Timing Offset')
    plt.xlabel('Offset (Fraction of a Chip)')
    plt.ylabel('Normalized Correlation Peak Magnitude')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('../_images/detection_dsss.svg', bbox_inches='tight')
    plt.show()

.. image:: ../_images/detection_dsss.svg
   :align: center
   :target: ../_images/detection_dsss.svg
   :alt: DSSS

Der Peak tritt erwartungsgemäß bei null Versatz auf und fällt linear ab, bis er bei einem halben Chip Versatz die Hälfte des Peakwertes erreicht. Nach mehr als einem Chip Versatz kann die Korrelation wieder anzusteigen scheinen, aber der eigentliche Peak ist niedrig, weil das Signal nicht mehr mit der Sequenz ausgerichtet ist.

****************************************************
Echtzeit-Paketdetektion in kontinuierlichen IQ-Strömen
****************************************************

Bisher haben wir die theoretischen Grundlagen der Signaldetektion erforscht – von Korrelatoren über CFAR-Detektoren bis zu Spread-Spectrum-Systemen. Jetzt kombinieren wir sie, um ein häufiges praktisches Problem zu lösen: **die Detektion intermittierender Pakete in einem kontinuierlichen Strom von IQ-Samples von einem SDR**. Betrachte folgendes Szenario: Ein Modem oder IoT-Gerät sendet einmal pro Sekunde oder in unregelmäßigen Abständen ein Datenpaket. Dein SDR empfängt kontinuierlich Samples bei z. B. 1 MHz. Die Pakete kommen zu unvorhersehbaren Zeiten an, eingebettet in Rauschen und Interferenz. Du musst:

1. Erkennen, wann ein Paket eintrifft
2. Den genauen Sample-Index bestimmen, wo es beginnt
3. Das Paket zur weiteren Verarbeitung (Demodulation, Dekodierung usw.) extrahieren
4. Das alles in Echtzeit tun, ohne Pakete zu verpassen

Das unterscheidet sich grundlegend von der Verarbeitung einer voraufgezeichneten IQ-Datei, wo du das gesamte Signal auf einmal analysieren kannst. Hier kommen Samples kontinuierlich an, und du musst in Echtzeit mit begrenzten Rechenressourcen Entscheidungen treffen. Wir kombinieren mehrere in diesem Kapitel behandelte Techniken:

1. **Kreuzkorrelation**: Um das bekannte Präambelmuster zu finden
2. **CFAR-Detektion**: Um Schwellen trotz variablem Rauschen adaptiv zu setzen
3. **Pufferverwaltung**: Um kontinuierliche Streaming-Daten zu verwalten
4. **Peak-Detektion**: Um genaues Paket-Timing zu extrahieren

Um in Echtzeit zu arbeiten, akkumulieren wir Samples in **Puffern** von z. B. 100.000 Samples, führen den Detektor auf jedem Puffer aus und halten Zustand über Puffergrenzen hinweg aufrecht, damit Pakete, die zwei Puffer überspannen, nicht übersehen werden.

Implementierung
##############

Unser Detektor folgt diesem Ablauf:

.. mermaid::

 flowchart TD
    A("Kontinuierlicher IQ-Strom vom SDR<br/>(1 MHz Abtastrate)")
    B("Puffer-Akkumulation<br/>(100k Samples = 0,1 s)")
    C("Kreuzkorrelation mit bekannter Präambel")
    D("CFAR-Schwellenberechnung")
    E("Peak-Detektion<br/>(Korrelation > Schwelle)")
    F("Paketextraktion & Validierung")
    A --> B --> C --> D --> E --> F

Um Pakete zu vermeiden, die Puffergrenzen überqueren, verwenden wir einen **Overlap-Save**-Ansatz, bei dem jeder Puffer die letzten ``N_preamble`` Samples aus dem vorherigen Puffer enthält. Das stellt sicher, dass Pakete, die nahe dem Ende von Puffer ``i`` beginnen, vollständig in Puffer ``i+1`` enthalten sind. Es fügt einen kleinen rechnerischen Overhead hinzu, aber das ist dem Verpassen von Paketen an der Puffergrenze vorzuziehen.

Bauen wir Schritt für Schritt einen vollständigen Paketdetektor in Python. Wir verwenden eine kürzere Zadoff-Chu-Präambel als zuvor und implementieren einen adaptiven CFAR-Detektor.

Schritt 1: Präambel und Parameter definieren
*******************************************

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import correlate

    # Präambel: Zadoff-Chu-Sequenz (hervorragende Korrelationseigenschaften)
    N_zc = 63  # ZC-Sequenzlänge (typischerweise prim oder 2er-Potenz minus 1)
    u = 5      # ZC-Wurzel
    t = np.arange(N_zc)
    preamble = np.exp(-1j * np.pi * u * t * (t + 1) / N_zc)

    # Systemparameter
    sample_rate = 1e6
    buffer_size = 100000
    overlap_size = len(preamble)  # Überlappung für Grenzpakete

    # CFAR-Parameter
    cfar_guard = 10
    cfar_train = 50
    pfa_target = 1e-6

    # Paketparameter (für Simulation)
    packet_length = 500  # Gesamtpaketlänge in Samples (Präambel + Daten)
    snr_db = -5

Schritt 2: CFAR-Detektorfunktion
*******************************

Wir verwenden den Cell-Averaging CFAR (CA-CFAR) von früher, leicht optimiert:

.. code-block:: python

    def ca_cfar_1d(signal, num_train, num_guard, pfa):
        """
        1D Cell-Averaging CFAR-Detektor.

        Args:
            signal: Eingangssignal (typischerweise Korrelationsbetrag)
            num_train: Anzahl der Trainingszellen (auf jeder Seite)
            num_guard: Anzahl der Schutzzellen (auf jeder Seite)
            pfa: Ziel-Fehlalarmwahrscheinlichkeit

        Returns:
            threshold: Adaptives Schwellen-Array
        """
        n = len(signal)
        threshold = np.zeros(n)
        alpha = num_train * (pfa**(-1/num_train) - 1)

        for i in range(n):
            # Trainingsbereich-Indizes definieren
            train_start_left = max(0, i - num_guard - num_train)
            train_end_left = max(0, i - num_guard)
            train_start_right = min(n, i + num_guard + 1)
            train_end_right = min(n, i + num_guard + num_train + 1)

            # Trainingszellen sammeln (Schutzzellen und CUT ausschließen)
            train_cells = np.concatenate([
                signal[train_start_left:train_end_left],
                signal[train_start_right:train_end_right]
            ])

            if len(train_cells) > 0:
                noise_est = np.mean(train_cells)
                threshold[i] = alpha * noise_est

        return threshold

Schritt 3: Paketdetektionsfunktion
**********************************

.. code-block:: python

    def detect_packets(buffer, preamble, cfar_guard, cfar_train, pfa,
                      min_spacing=None):
        """
        Pakete in einem Puffer von IQ-Samples detektieren.

        Args:
            buffer: Komplexe IQ-Samples
            preamble: Bekannte Präambelsequenz
            cfar_guard: CFAR-Schutzzellen
            cfar_train: CFAR-Trainingszellen
            pfa: Ziel-Fehlalarmwahrscheinlichkeit
            min_spacing: Minimale Samples zwischen Detektionen (verhindert Duplikate)

        Returns:
            detections: Liste von Sample-Indizes, wo Pakete beginnen
        """
        # Puffer mit Präambel korrelieren
        corr = correlate(buffer, preamble, mode='same')
        corr_power = np.abs(corr)**2

        # Adaptive Schwelle berechnen
        threshold = ca_cfar_1d(corr_power, cfar_train, cfar_guard, pfa)

        # Peaks oberhalb der Schwelle finden
        detections_raw = np.where(corr_power > threshold)[0]

        # Korrelationsversatz kompensieren (Peak tritt len(preamble)//2 nach dem echten Start auf)
        half_preamble = len(preamble) // 2
        detections_raw = detections_raw - half_preamble

        # Randdetektionen entfernen (unzuverlässig)
        half_preamble = len(preamble) // 2
        detections_raw = detections_raw[
            (detections_raw > half_preamble) &
            (detections_raw < len(buffer) - half_preamble)
        ]

        # Doppelte Detektionen entfernen (Peaks nah beieinander)
        if min_spacing is None:
            min_spacing = len(preamble)

        detections = []
        if len(detections_raw) > 0:
            detections.append(detections_raw[0])
            for det in detections_raw[1:]:
                if det - detections[-1] > min_spacing:
                    detections.append(det)

        return detections, corr_power, threshold

Schritt 4: Simulation – Testsignal erzeugen
******************************************

.. code-block:: python

    def generate_packet_stream(preamble, packet_length, num_packets,
                               sample_rate, snr_db):
        """
        Simulierten IQ-Strom mit intermittierenden Paketen erzeugen.

        Returns:
            signal: Komplexe IQ-Samples
            true_starts: Echte Paketstart-Indizes
        """
        # Rauschleistung aus SNR berechnen
        signal_power = 1.0  # Normierte Präambelleistung
        noise_power = signal_power / (10**(snr_db/10))
        noise_std = np.sqrt(noise_power / 2)  # Komplexes Rauschen

        # QPSK-Daten erzeugen (zufällige Nutzlast nach Präambel)
        qpsk_map = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)

        # Zeit zwischen Paketen (1 Sekunde ±20% Jitter)
        packets_per_sec = 1
        avg_gap = int(sample_rate / packets_per_sec)

        signal = []
        true_starts = []

        for i in range(num_packets):
            # Lücke hinzufügen (nur Rauschen)
            if i == 0:
                gap_length = np.random.randint(avg_gap//2, avg_gap)
            else:
                gap_length = np.random.randint(int(avg_gap*0.8), int(avg_gap*1.2))

            noise = noise_std * (np.random.randn(gap_length) +
                                1j*np.random.randn(gap_length))
            signal.extend(noise)

            # Echten Paketstart aufzeichnen
            true_starts.append(len(signal))

            # Paket hinzufügen (Präambel + Daten)
            data_length = packet_length - len(preamble)
            data = qpsk_map[np.random.randint(0, 4, data_length)]
            packet = np.concatenate([preamble, data])

            # Rauschen zum Paket hinzufügen
            packet_noisy = packet + noise_std * (np.random.randn(len(packet)) +
                                                 1j*np.random.randn(len(packet)))
            signal.extend(packet_noisy)

        # Abschließende Lücke hinzufügen
        gap_length = np.random.randint(avg_gap//2, avg_gap)
        noise = noise_std * (np.random.randn(gap_length) +
                            1j*np.random.randn(gap_length))
        signal.extend(noise)

        return np.array(signal), true_starts

    # 5 Sekunden Signal mit ~5 Paketen erzeugen
    signal, true_starts = generate_packet_stream(
        preamble, packet_length, num_packets=5,
        sample_rate=sample_rate, snr_db=snr_db
    )

    print(f"Generated {len(signal)} samples ({len(signal)/sample_rate:.1f} sec)")
    print(f"True packet starts: {true_starts}")

Schritt 5: Detektion im Streaming-Modus ausführen
****************************************

Jetzt verarbeiten wir das Signal in Blöcken und simulieren Echtzeit-Streaming:

.. code-block:: python

    def process_stream(signal, preamble, buffer_size, overlap_size,
                      cfar_guard, cfar_train, pfa):
        """
        Kontinuierlichen IQ-Strom in Puffern verarbeiten (simuliert Echtzeit).

        Returns:
            all_detections: Liste detektierter Paketstarts (globale Indizes)
        """
        all_detections = []
        n_samples = len(signal)
        current_pos = 0

        while current_pos < n_samples:
            # Puffer mit Überlappung definieren
            buffer_start = max(0, current_pos - overlap_size)
            buffer_end = min(n_samples, current_pos + buffer_size)
            buffer = signal[buffer_start:buffer_end]

            # Pakete in diesem Puffer detektieren
            detections, corr_power, threshold = detect_packets(
                buffer, preamble, cfar_guard, cfar_train, pfa
            )

            # Puffer-relative Indizes in globale Indizes umrechnen
            for det in detections:
                global_idx = buffer_start + det

                # Doppelte Detektionen aus Überlappungsbereich vermeiden
                if len(all_detections) == 0 or \
                   global_idx - all_detections[-1] > len(preamble):
                    all_detections.append(global_idx)

            current_pos += buffer_size

        return all_detections


    detected_starts = process_stream(
        signal, preamble, buffer_size, overlap_size,
        cfar_guard, cfar_train, pfa_target
    )

    print(f"\nDetection Results:")
    print(f"True packets:     {len(true_starts)}")
    print(f"Detected packets: {len(detected_starts)}")
    print(f"Detected starts:  {detected_starts}")

Schritt 6: Leistung bewerten
*****************************

.. code-block:: python

    # Detektionsstatistiken berechnen
    tolerance = len(preamble)

    matched_detections = []
    false_alarms = []

    for det in detected_starts:
        # Prüfen, ob Detektion mit einem echten Paket übereinstimmt
        matched = False
        for true_start in true_starts:
            if abs(det - true_start) <= tolerance:
                matched_detections.append(det)
                matched = True
                break
        if not matched:
            false_alarms.append(det)

    missed_packets = len(true_starts) - len(matched_detections)

    print(f"\nPerformance Metrics:")
    print(f"  Correct detections: {len(matched_detections)}/{len(true_starts)}")
    print(f"  Missed packets:     {missed_packets}")
    print(f"  False alarms:       {len(false_alarms)}")

    # Timing-Fehler berechnen
    timing_errors = []
    for det in matched_detections:
        errors = [abs(det - ts) for ts in true_starts]
        timing_errors.append(min(errors))

    if len(timing_errors) > 0:
        print(f"  Timing error (avg): {np.mean(timing_errors):.1f} samples")
        print(f"  Timing error (max): {np.max(timing_errors):.1f} samples")

Schritt 7: Ergebnisse visualisieren
**************************

.. code-block:: python

    # Einen Puffer für detaillierte Visualisierung verarbeiten
    buffer_start = max(0, true_starts[0] - 5000)
    buffer_end = min(len(signal), true_starts[0] + 20000)
    viz_buffer = signal[buffer_start:buffer_end]

    detections_viz, corr_viz, thresh_viz = detect_packets(
        viz_buffer, preamble, cfar_guard, cfar_train, pfa_target
    )

    # In globale Indizes für Darstellung umrechnen
    detections_viz_global = [d + buffer_start for d in detections_viz]

    # Visualisierung erstellen
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    time_axis = (np.arange(len(viz_buffer)) + buffer_start) / sample_rate * 1000  # ms

    # Teilgrafik 1: Leistung des empfangenen Signals
    axes[0].plot(time_axis, np.abs(viz_buffer)**2, 'gray', alpha=0.6, linewidth=0.5)
    axes[0].set_ylabel('Power')
    axes[0].set_title('Received IQ Signal Power')
    axes[0].grid(True, alpha=0.3)

    # Echte Paketpositionen markieren
    for ts in true_starts:
        if buffer_start <= ts <= buffer_end:
            t_ms = ts / sample_rate * 1000
            axes[0].axvline(t_ms, color='green', linestyle='--', alpha=0.7,
                          label='True Packet' if ts == true_starts[0] else '')
    axes[0].legend()

    # Teilgrafik 2: Korrelationsausgabe
    axes[1].plot(time_axis, corr_viz, 'blue', linewidth=1, label='Correlation')
    axes[1].plot(time_axis, thresh_viz, 'red', linestyle='--', linewidth=1.5,
                label='CFAR Threshold')
    axes[1].set_ylabel('Correlation Power')
    axes[1].set_title('Preamble Correlation with Adaptive CFAR Threshold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Teilgrafik 3: Detektionen
    detection_mask = np.zeros(len(viz_buffer))
    for det in detections_viz:
        detection_mask[det] = corr_viz[det]

    axes[2].plot(time_axis, corr_viz, 'blue', alpha=0.4, linewidth=0.8)
    axes[2].scatter(time_axis[detection_mask > 0], detection_mask[detection_mask > 0],
                   color='lime', edgecolors='black', s=100, zorder=5,
                   label='Detected Packets')
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_ylabel('Correlation Power')
    axes[2].set_title('Detected Packet Locations')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.show()

Die Visualisierung sollte zeigen:

1. **Obere Grafik**: Rohe Signalleistung mit markierten echten Paketpositionen
2. **Mittlere Grafik**: Korrelationsausgabe mit adaptiver CFAR-Schwelle, die den Rauschboden verfolgt
3. **Untere Grafik**: Detektierte Pakete als hervorgehobene Peaks oberhalb der Schwelle

.. image:: ../_images/detection_realtime.png
   :align: center
   :scale: 50 %
   :alt: Real-time packet detection results

Praktische Überlegungen und Parametereinstellung
####################################

Puffergröße: Kompromisse
***********************

**Größere Puffer**, z. B. 1M Samples:

- ✅ Bessere CFAR-Rauschschätzung (mehr Trainingszellen)
- ✅ Geringerer Rechenoverhead (weniger Verarbeitungsaufrufe)
- ❌ Höhere Latenz (Puffer muss sich erst füllen)
- ❌ Mehr Speicher benötigt

**Kleinere Puffer**, z. B. 10k Samples:

- ✅ Niedrigere Latenz (schnellere Reaktion)
- ✅ Weniger Speicherverbrauch
- ❌ CFAR-Leistung verschlechtert sich (weniger Trainingszellen)
- ❌ Höhere CPU-Auslastung (häufigere Verarbeitung)

**Empfehlung**: Starte mit einer Puffergröße von 10× bis 100× deiner Präambellänge. Für eine 63-Sample-Präambel bei 1 Msps probiere 10k bis 100k Samples.

CFAR-Parametereinstellung
**********************

Die drei CFAR-Parameter steuern das Detektorverhalten:

**num_guard**: Schutzzellen

- Verhindert Signalleckage in die Rauschschätzung
- Zu klein: Signal leckt in den Trainingsbereich, erhöht die Schwelle und verursacht verpasste Detektionen
- Zu groß: weniger Trainingszellen und schlechtere Rauschschätzung
- Faustregel: auf etwa 0,5 bis 1,0× die Präambellänge setzen

**num_train**: Trainingszellen

- Schätzt den lokalen Rauschboden
- Zu klein: verrauschte Schwelle und mehr Fehlalarme oder verpasste Detektionen
- Zu groß: Schwelle passt sich nicht schnell genug an Rauschänderungen an
- Faustregel: auf etwa 3 bis 5× die Präambellänge setzen

**pfa**: Fehlalarmwahrscheinlichkeit

- Steuert die Detektionsempfindlichkeit
- Zu hoch, z. B. 1e-2: viele Fehlalarme
- Zu niedrig, z. B. 1e-10: verpasste schwache Pakete
- Faustregel: mit 1e-5 für Pro-Lag-PFA beginnen, dann anhand der systemweiten Fehlalarmrate anpassen

Erinnere dich an die Beziehung zwischen Pro-Lag- und systemweiter Fehlalarmrate aus dem früheren Abschnitt dieses Kapitels.
