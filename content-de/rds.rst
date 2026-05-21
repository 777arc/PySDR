.. _rds-chapter:

##################
End-to-End-Beispiel
##################

In diesem Kapitel bringen wir viele der zuvor gelernten Konzepte zusammen und gehen ein vollständiges Beispiel für den Empfang und die Dekodierung eines realen digitalen Signals durch. Wir werden uns das Radio Data System (RDS) ansehen, ein Kommunikationsprotokoll zum Einbetten kleiner Informationsmengen in UKW-Rundfunksendungen, wie Sender- und Songnamen. Wir werden FM demodulieren, frequenzverschieben, filtern, dezimieren, resamplen, synchronisieren, dekodieren und die Bytes parsen. Eine Beispiel-IQ-Datei wird zu Testzwecken bereitgestellt, falls du kein SDR zur Hand hast.

********************************
Einführung in UKW-Radio und RDS
********************************

Um RDS zu verstehen, müssen wir zunächst UKW-Rundfunksendungen und die Struktur ihrer Signale wiederholen. Du bist wahrscheinlich mit dem Audioanteil von UKW-Signalen vertraut, die einfach Audiosignale sind, die frequenzmoduliert und auf Mittenfrequenzen entsprechend dem Sendernamen übertragen werden, z.B. ist "WPGC 95.5 FM" genau auf 95,5 MHz zentriert. Zusätzlich zum Audioanteil enthält jede UKW-Sendung einige andere Komponenten, die zusammen mit dem Audio frequenzmoduliert werden. Anstatt einfach nach der Signalstruktur zu googeln, werfen wir einen Blick auf die Leistungsspektraldichte (PSD) eines Beispiel-UKW-Signals, *nach* der FM-Demodulation. Wir sehen nur den positiven Anteil, da die Ausgabe der FM-Demodulation ein reales Signal ist, obwohl der Eingang komplex war (den Code zur Durchführung dieser Demodulation sehen wir in Kürze).

.. image:: ../_images/fm_psd.svg
   :align: center
   :target: ../_images/fm_psd.svg
   :alt: Leistungsspektraldichte (PSD) eines UKW-Radiosignals nach der FM-Demodulation, zeigt RDS

Wenn wir das Signal im Frequenzbereich betrachten, bemerken wir folgende einzelne Signale:

#. Ein leistungsstarkes Signal zwischen 0 - 17 kHz
#. Ein Ton bei 19 kHz
#. Zentriert bei 38 kHz und ungefähr 30 kHz breit sehen wir ein interessant aussehendes symmetrisches Signal
#. Doppelkeulenförmiges Signal zentriert bei 57 kHz
#. Einzelkeulenförmiges Signal zentriert bei 67 kHz

Das ist im Wesentlichen alles, was wir nur durch Betrachtung der PSD bestimmen können, und denke daran, dass dies *nach* der FM-Demodulation ist. Die PSD vor der FM-Demodulation sieht wie folgt aus, was uns nicht viel verrät.

.. image:: ../_images/fm_before_demod.svg
   :align: center
   :target: ../_images/fm_before_demod.svg
   :alt: Leistungsspektraldichte (PSD) eines UKW-Radiosignals vor jeder Demodulation

Es ist jedoch wichtig zu verstehen, dass beim FM-Modulieren eines Signals eine höhere Frequenz im Datensignal zu einer höheren Frequenz im resultierenden FM-Signal führt. Das bei 67 kHz zentrierte Signal erhöht also die gesamte vom übertragenen FM-Signal belegte Bandbreite, da die maximale Frequenzkomponente jetzt bei etwa 75 kHz liegt, wie in der ersten PSD oben gezeigt. `Carsons Bandbreitenregel <https://en.wikipedia.org/wiki/Carson_bandwidth_rule>`_ angewendet auf FM sagt uns, dass UKW-Sender etwa 250 kHz Spektrum belegen, weshalb wir normalerweise mit 250 kHz abtasten (zur Erinnerung: bei der Quadratur/IQ-Abtastung entspricht die empfangene Bandbreite der Abtastrate).

Als kurze Nebenbemerkung: Manche Leser kennen vielleicht den Anblick des UKW-Bandes mit einem SDR oder Spektrumanalysator und sehen das folgende Spektrogramm, wobei sie glauben, dass die blockartige Signale neben einigen UKW-Sendern RDS sind.

.. image:: ../_images/fm_band_psd.png
   :scale: 80 %
   :align: center
   :alt: Spektrogramm des UKW-Bands

Es stellt sich heraus, dass diese blockartigen Signale tatsächlich HD Radio sind, eine digitale Version desselben UKW-Radiosignals (gleicher Audioinhalt). Diese digitale Version führt zu einem Audio-Signal höherer Qualität am Empfänger, da analoges UKW nach der Demodulation immer etwas Rauschen enthält (als analoges Verfahren), während das digitale Signal ohne Rauschen demoduliert/dekodiert werden kann, vorausgesetzt es gibt null Bitfehler.

Zurück zu den fünf Signalen, die wir in unserer PSD entdeckt haben; das folgende Diagramm zeigt, wofür jedes Signal verwendet wird.

.. image:: ../_images/fm_psd_labeled.svg
   :align: center
   :target: ../_images/fm_psd_labeled.svg
   :alt: Komponenten innerhalb eines UKW-Radiosignals, einschließlich Mono- und Stereoaudio, RDS und DirectBand-Signale

Die einzelnen Signale in beliebiger Reihenfolge:

Die Mono- und Stereoaudiosignale tragen einfach das Audiosignal, wobei Addition und Subtraktion den linken und rechten Kanal ergibt.

Der 19-kHz-Pilotton wird zur Demodulation des Stereoaudios verwendet. Wenn man ihn verdoppelt, dient er als Frequenz- und Phasenreferenz, da das Stereoaudiosignal bei 38 kHz zentriert ist. Das Verdoppeln des Tons kann einfach durch Quadrieren der Abtastwerte erfolgen – erinnere dich an die Frequenzverschiebungs-Fourier-Eigenschaft aus dem Kapitel :ref:`freq-domain-chapter`.

DirectBand war ein nordamerikanisches drahtloses Datennetzwerk im Besitz von Microsoft, das auch als "MSN Direct" auf dem Verbrauchermarkt bekannt war. DirectBand übermittelte Informationen an Geräte wie tragbare GPS-Empfänger, Armbanduhren und Heim-Wetterstationen. Es ermöglichte sogar Benutzern, kurze Nachrichten über Windows Live Messenger zu empfangen. Eine der erfolgreichsten Anwendungen von DirectBand waren Echtzeit-Staudaten auf Garmin-GPS-Empfängern, die von Millionen von Menschen genutzt wurden, bevor Smartphones allgegenwärtig wurden. Der DirectBand-Dienst wurde im Januar 2012 eingestellt, was die Frage aufwirft, warum wir ihn in unserem UKW-Signal sehen, das nach 2012 aufgezeichnet wurde. Meine einzige Vermutung ist, dass die meisten UKW-Sender lange vor 2012 entwickelt und gebaut wurden und ohne aktiven DirectBand-Feed immer noch etwas senden, vielleicht Pilotsymbole.

Schließlich kommen wir zu RDS, dem Fokus des restlichen Kapitels. Wie wir in unserer ersten PSD sehen können, hat RDS etwa 4 kHz Bandbreite (bevor es FM-moduliert wird) und liegt zwischen dem Stereoaudio und dem DirectBand-Signal. Es ist ein Niedrigdatenraten-Digitalkommunikationsprotokoll, das UKW-Sendern ermöglicht, Sender-Identifikation, Programminformationen, Zeit und andere verschiedene Informationen zusammen mit dem Audio zu übermitteln. Der RDS-Standard ist als IEC-Standard 62106 veröffentlicht und `kann hier gefunden werden <http://www.interactive-radio-system.com/docs/EN50067_RDS_Standard.pdf>`_.

********************************
Das RDS-Signal
********************************

In diesem Kapitel werden wir Python zum Empfangen von RDS verwenden, aber um den Empfang am besten zu verstehen, müssen wir zunächst lernen, wie das Signal geformt und übertragen wird.

Sendeseite
#############

Die vom UKW-Sender zu übertragenden RDS-Informationen (z.B. Titelname usw.) werden in Sätze von 8 Bytes codiert. Jeder Satz von 8 Bytes, der 64 Bits entspricht, wird mit 40 "Prüfbits" kombiniert, um eine einzelne "Gruppe" zu bilden. Diese 104 Bits werden zusammen übertragen, obwohl es keine Zeitlücke zwischen Gruppen gibt, sodass der Empfänger aus seiner Perspektive diese Bits ununterbrochen empfängt und die Grenze zwischen den 104-Bit-Gruppen bestimmen muss. Wir werden mehr Details zur Codierung und Nachrichtenstruktur sehen, sobald wir uns mit der Empfangsseite beschäftigen.

Zur drahtlosen Übertragung dieser Bits verwendet RDS BPSK, das wie wir im Kapitel :ref:`modulation-chapter` gelernt haben, ein einfaches digitales Modulationsverfahren ist, das 1er und 0er der Phase eines Trägers zuordnet. Wie viele BPSK-basierte Protokolle verwendet RDS differentielle Codierung, was einfach bedeutet, dass die 1er und 0er der Daten in Änderungen von 1ern und 0ern codiert werden, was es ermöglicht, nicht mehr auf eine 180-Grad-Phasendrehung zu achten (mehr dazu später). Die BPSK-Symbole werden mit 1187,5 Symbolen pro Sekunde übertragen, und da BPSK ein Bit pro Symbol trägt, bedeutet das, dass RDS eine rohe Datenrate von etwa 1,2 kbps hat (einschließlich Overhead). RDS enthält keine Kanalcodierung (a.k.a. Vorwärtsfehlerkorrektur), obwohl die Datenpakete eine zyklische Redundanzprüfung (CRC) enthalten, um zu wissen, wann ein Fehler aufgetreten ist.

Das endgültige BPSK-Signal wird dann auf 57 kHz hochverschoben und zu allen anderen Komponenten des FM-Signals hinzugefügt, bevor es auf der Stationsfrequenz FM-moduliert und über die Luft übertragen wird. UKW-Radiosignale werden mit extrem hoher Leistung im Vergleich zu den meisten anderen drahtlosen Kommunikationen übertragen, bis zu 80 kW! Deshalb haben viele SDR-Benutzer einen FM-Ablehnfilter (d.h. einen Bandsperrfilter) in Reihe mit ihrer Antenne, damit FM keine Interferenz zu dem hinzufügt, was sie empfangen möchten.

Obwohl dies nur ein kurzer Überblick über die Sendeseite war, werden wir mehr Details besprechen, wenn wir den RDS-Empfang behandeln.

Empfangsseite
############

Um RDS zu demodulieren und zu dekodieren, führen wir die folgenden Schritte durch, von denen viele den senderseitigen Schritten in umgekehrter Reihenfolge entsprechen (du musst diese Liste nicht auswendig lernen, wir gehen jeden Schritt unten einzeln durch):

#. Ein UKW-Radiosignal zentriert auf die Stationsfrequenz empfangen (oder eine IQ-Aufzeichnung einlesen), üblicherweise bei einer Abtastrate von 250 kHz
#. Das UKW mithilfe der sogenannten "Quadratur-Demodulation" demodulieren
#. Um 57 kHz frequenzverschieben, sodass das RDS-Signal bei 0 Hz zentriert ist
#. Tiefpassfilter, um alles außer RDS herauszufiltern (wirkt auch als Matched Filter)
#. Um 10 dezimieren, damit wir bei einer niedrigeren Abtastrate arbeiten können, da wir die höheren Frequenzen ohnehin herausgefiltert haben
#. Auf 19 kHz resamplen, was uns eine ganzzahlige Anzahl von Abtastwerten pro Symbol gibt
#. Zeitsynchronisation auf Symbolebene, in diesem Beispiel mit Mueller und Müller
#. Feine Frequenzsynchronisation mit einem Costas-Regelkreis
#. Das BPSK in 1er und 0er demodulieren
#. Differenzielle Dekodierung, um die angewendete differentielle Codierung rückgängig zu machen
#. Dekodierung der 1er und 0er in Gruppen von Bytes
#. Parsen der Gruppen von Bytes in unsere endgültige Ausgabe

Auch wenn das viele Schritte zu sein scheinen, ist RDS tatsächlich eines der einfachsten drahtlosen digitalen Kommunikationsprotokolle. Ein modernes drahtloses Protokoll wie WiFi oder 5G erfordert ein ganzes Lehrbuch, nur um die übergeordneten PHY/MAC-Schichtinformationen abzudecken.

Wir werden nun in den Python-Code eintauchen, der zum Empfangen von RDS verwendet wird. Dieser Code wurde mit einer `UKW-Radioaufzeichnung, die du hier findest <https://github.com/777arc/498x/blob/master/fm_rds_250k_1Msamples.iq?raw=true>`_, getestet. Du solltest auch dein eigenes Signal einspeisen können, solange es mit ausreichend hohem SNR empfangen wird – einfach auf die Mittenfrequenz des Senders abstimmen und bei 250 kHz abtasten. Um die empfangene Signalleistung zu maximieren (z.B. wenn du drinnen bist), hilft es, eine Halbwellen-Dipolantenne der richtigen Länge (~1,5 Meter) zu verwenden, nicht die 2,4-GHz-Antennen, die mit dem Pluto geliefert werden. Davon abgesehen ist FM ein sehr lautes Signal, und wenn du in der Nähe eines Fensters oder draußen bist, werden die 2,4-GHz-Antennen wahrscheinlich ausreichen, um die stärkeren Radiosender zu empfangen.

In diesem Abschnitt präsentieren wir kleine Teile des Codes einzeln mit Erläuterungen, aber derselbe Code wird am Ende dieses Kapitels in einem großen Block bereitgestellt. Jeder Abschnitt präsentiert einen Codeblock und erklärt dann, was er tut.

********************************
Ein Signal erfassen
********************************

.. code-block:: python

 import numpy as np
 from scipy.signal import resample_poly, firwin, bilinear, lfilter
 import matplotlib.pyplot as plt

 # Signal einlesen
 x = np.fromfile('/home/marc/Downloads/fm_rds_250k_1Msamples.iq', dtype=np.complex64)
 sample_rate = 250e3
 center_freq = 99.5e6

Wir lesen unsere Testaufzeichnung ein, die mit 250 kHz abgetastet und auf einem UKW-Sender mit hohem SNR empfangen wurde. Stelle sicher, dass du den Dateipfad entsprechend deinem System und dem Speicherort der Aufzeichnung aktualisierst. Wenn du ein SDR bereits in Python eingerichtet und zum Laufen gebracht hast, kannst du gerne ein Live-Signal empfangen, obwohl es hilfreich ist, zuerst den gesamten Code mit einer `bekannt funktionierenden IQ-Aufzeichnung <https://github.com/777arc/498x/blob/master/fm_rds_250k_1Msamples.iq?raw=true>`_ getestet zu haben. Wir verwenden :code:`x` zur Speicherung des aktuell manipulierten Signals.

********************************
FM-Demodulation
********************************

.. code-block:: python

 # Quadratur-Demodulation
 x = 0.5 * np.angle(x[0:-1] * np.conj(x[1:])) # siehe https://wiki.gnuradio.org/index.php/Quadrature_Demod

Wie am Anfang dieses Kapitels besprochen, werden mehrere einzelne Signale in der Frequenz kombiniert und FM-moduliert, um das tatsächlich über die Luft übertragene Signal zu erstellen. Der erste Schritt ist also, diese FM-Modulation rückgängig zu machen. Eine andere Art, darüber nachzudenken: Die Information ist in der Frequenzvariation des empfangenen Signals gespeichert, und wir möchten es demodulieren, sodass die Information jetzt in der Amplitude und nicht in der Frequenz ist. Beachte, dass die Ausgabe dieser Demodulation ein reales Signal ist, obwohl wir ein komplexes Signal eingegeben haben.

Was diese einzelne Python-Zeile tut, ist zunächst das Produkt unseres Signals mit einer verzögerten und konjugierten Version unseres Signals zu berechnen. Dann findet sie die Phase jedes Abtastwerts in diesem Ergebnis, was der Moment ist, an dem es von komplex zu real wechselt. Um uns zu beweisen, dass dies uns die in den Frequenzvariation enthaltene Information liefert, betrachte einen Ton bei Frequenz :math:`f` mit einer beliebigen Phase :math:`\phi`, den wir als :math:`e^{j2 \pi (f t + \phi)}` darstellen können. Bei der diskreten Zeit, die ein ganzzahliges :math:`n` anstelle von :math:`t` verwendet, wird dies zu :math:`e^{j2 \pi (f n + \phi)}`. Die konjugierte und verzögerte Version ist :math:`e^{-j2 \pi (f (n-1) + \phi)}`. Das Multiplizieren dieser beiden ergibt :math:`e^{j2 \pi f}`, was großartig ist, weil :math:`\phi` verschwunden ist, und wenn wir die Phase dieses Ausdrucks berechnen, bleibt nur :math:`f` übrig.

Ein praktischer Nebeneffekt der FM-Modulation ist, dass Amplitudenvariationen des empfangenen Signals die Lautstärke des Audios tatsächlich nicht verändern, im Gegensatz zum AM-Radio.

********************************
Frequenzverschiebung
********************************

.. code-block:: python

 # Frequenzverschiebung
 N = len(x)
 f_o = -57e3 # Betrag der Verschiebung
 t = np.arange(N)/sample_rate # Zeitvektor
 x = x * np.exp(2j*np.pi*f_o*t) # Abwärtsverschiebung

Als nächstes verschieben wir die Frequenz um 57 kHz nach unten, mit dem :math:`e^{j2 \pi f_ot}`-Trick aus dem Kapitel :ref:`sync-chapter`, wobei :code:`f_o` die Frequenzverschiebung in Hz und :code:`t` einfach ein Zeitvektor ist. Da es ein reales Signal ist, das eingegeben wird, ist es eigentlich egal, ob du -57 oder +57 kHz verwendest, da die negativen Frequenzen den positiven entsprechen, sodass wir in jedem Fall unser RDS auf 0 Hz verschoben bekommen.

********************************
Filter zum Isolieren von RDS
********************************

.. code-block:: python

 # Tiefpassfilter
 taps = firwin(numtaps=101, cutoff=7.5e3, fs=sample_rate)
 x = np.convolve(x, taps, 'valid')

Nun müssen wir alles außer RDS herausfiltern. Da wir RDS bei 0 Hz zentriert haben, ist ein Tiefpassfilter genau das, was wir möchten. Wir verwenden :code:`firwin()`, um einen FIR-Filter zu entwerfen (d.h. die Taps zu finden), der nur wissen muss, wie viele Taps wir möchten und die Grenzfrequenz. Die Abtastrate muss ebenfalls angegeben werden, sonst ergibt die Grenzfrequenz für firwin keinen Sinn. Das Ergebnis ist ein symmetrischer Tiefpassfilter, sodass wir wissen, dass die Taps reelle Zahlen sein werden, und wir können den Filter auf unser Signal mittels Faltung anwenden. Wir wählen :code:`'valid'`, um die Randeffekte der Faltung loszuwerden, obwohl es in diesem Fall keine Rolle spielt, da wir ein so langes Signal einspeisen, dass ein paar merkwürdige Abtastwerte an beiden Rändern nichts durcheinanderbringen werden.

Randnotiz: Irgendwann werde ich den obigen Filter aktualisieren, um ein richtiges Matched Filter (Root-Raised-Cosine, glaube ich, was RDS verwendet) zu verwenden. Ich erhielt jedoch dieselben Fehlerraten mit dem firwin()-Ansatz wie mit dem korrekten Matched Filter in GNU Radio, also ist es offensichtlich keine strenge Anforderung.

********************************
Um 10 dezimieren
********************************

.. code-block:: python

 # Um 10 dezimieren, jetzt wo wir gefiltert haben und kein Aliasing auftreten wird
 x = x[::10]
 sample_rate = 25e3

Immer wenn du auf einen kleinen Bruchteil deiner Bandbreite herausfiltert (z.B. haben wir mit 125 kHz *echter* Bandbreite begonnen und nur 7,5 kHz davon behalten), macht es Sinn zu dezimieren. Erinnere dich an den Anfang des Kapitels :ref:`sampling-chapter`, wo wir über die Nyquist-Rate und die Fähigkeit lernten, bandbegrenzte Informationen vollständig zu speichern, solange wir mit der doppelten Höchstfrequenz abtasten. Jetzt, da wir unseren Tiefpassfilter verwendet haben, liegt unsere Höchstfrequenz bei etwa 7,5 kHz, sodass wir nur eine Abtastrate von 15 kHz benötigen. Sicherheitshalber fügen wir etwas Spielraum hinzu und verwenden eine neue Abtastrate von 25 kHz (das funktioniert später mathematisch gut).

Wir führen die Dezimierung durch, indem wir einfach 9 von je 10 Abtastwerten verwerfen, da wir vorher bei 250 kHz waren und jetzt 25 kHz haben möchten. Das mag zunächst verwirrend erscheinen, da das Verwerfen von 90% der Abtastwerte sich anfühlt, als würde man Informationen wegwerfen. Wenn du jedoch das Kapitel :ref:`sampling-chapter` durchgehst, wirst du sehen, warum wir nichts verlieren, weil wir ordnungsgemäß gefiltert haben (was als Anti-Aliasing-Filter fungierte) und unsere Maximalfrequenz und damit die Signalbandbreite reduziert haben.

Aus Code-Perspektive ist dies wahrscheinlich der einfachste Schritt von allen, aber vergiss nicht, deine Variable :code:`sample_rate` auf die neue Abtastrate zu aktualisieren.

********************************
Auf 19 kHz resamplen
********************************

.. code-block:: python

 # Auf 19 kHz resamplen
 x = resample_poly(x, 19, 25) # hoch, runter
 sample_rate = 19e3

Im Kapitel :ref:`pulse-shaping-chapter` haben wir das Konzept der "Abtastwerte pro Symbol" gefestigt und die Bequemlichkeit einer ganzzahligen Anzahl von Abtastwerten pro Symbol gelernt (ein Bruchwert ist gültig, aber nicht praktisch). Wie zuvor erwähnt, verwendet RDS BPSK und überträgt 1187,5 Symbole pro Sekunde. Wenn wir unser Signal wie bisher bei 25 kHz belassen, haben wir 21,052631579 Abtastwerte pro Symbol (denke kurz über die Mathematik nach, wenn das nicht stimmt). Was wir wirklich wollen, ist eine Abtastrate, die ein ganzzahliges Vielfaches von 1187,5 Hz ist, aber wir können nicht zu niedrig gehen, sonst können wir nicht die volle Bandbreite des Signals "speichern". Im vorherigen Unterabschnitt haben wir über die Notwendigkeit einer Abtastrate von mindestens 15 kHz gesprochen, und wir haben 25 kHz gewählt, um etwas Spielraum zu haben.

Die beste Abtastrate zum Resamplen hängt davon ab, wie viele Abtastwerte pro Symbol wir möchten. Angenommen, wir zielen auf 10 Abtastwerte pro Symbol ab. Die RDS-Symbolrate von 1187,5 multipliziert mit 10 würde uns eine Abtastrate von 11,875 kHz geben, was leider nicht hoch genug für Nyquist ist. Was ist mit 13 Abtastwerten pro Symbol? 1187,5 multipliziert mit 13 ergibt 15437,5 Hz, was über 15 kHz liegt, aber eine ziemlich ungerade Zahl ist. Wie wäre es mit der nächsten Zweierpotenz, also 16 Abtastwerte pro Symbol? 1187,5 multipliziert mit 16 ist genau 19 kHz! Die gerade Zahl ist weniger Zufall als eine bewusste Protokolldesign-Entscheidung.

Um von 25 kHz auf 19 kHz zu resamplen, verwenden wir :code:`resample_poly()`, das um einen ganzzahligen Wert aufwärts sampelt, filtert und dann um einen ganzzahligen Wert abwärts sampelt. Das ist praktisch, weil wir anstelle von 25000 und 19000 einfach 25 und 19 eingeben können. Wenn wir 13 Abtastwerte pro Symbol mit einer Abtastrate von 15437,5 Hz verwendet hätten, könnten wir :code:`resample_poly()` nicht verwenden und der Resampling-Prozess wäre viel komplizierter.

Denke immer daran, deine Variable :code:`sample_rate` zu aktualisieren, wenn du eine Operation durchführst, die sie ändert.

***********************************
Zeitsynchronisation (Symbolebene)
***********************************

.. code-block:: python

 # Symbolsynchronisation, wie im Sync-Kapitel gemacht
 samples = x # um mit dem Sync-Kapitel übereinzustimmen
 samples_interpolated = resample_poly(samples, 32, 1) # wir verwenden 32 als Interpolationsfaktor, beliebig gewählt
 sps = 16
 mu = 0.01 # Anfangsschätzung der Phase des Abtastwerts
 out = np.zeros(len(samples) + 10, dtype=np.complex64)
 out_rail = np.zeros(len(samples) + 10, dtype=np.complex64) # speichert Werte; jede Iteration braucht die vorherigen 2 Werte plus aktuellen
 i_in = 0 # Eingangs-Abtastwert-Index
 i_out = 2 # Ausgangsindex (erste zwei Ausgaben sind 0)
 while i_out < len(samples) and i_in+32 < len(samples):
     out[i_out] = samples_interpolated[i_in*32 + int(mu*32)] # den vermeintlich "besten" Abtastwert nehmen
     out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j*int(np.imag(out[i_out]) > 0)
     x = (out_rail[i_out] - out_rail[i_out-2]) * np.conj(out[i_out-1])
     y = (out[i_out] - out[i_out-2]) * np.conj(out_rail[i_out-1])
     mm_val = np.real(y - x)
     mu += sps + 0.01*mm_val
     i_in += int(np.floor(mu)) # auf nächste ganze Zahl abrunden, da als Index verwendet
     mu = mu - np.floor(mu) # ganzzahligen Teil von mu entfernen
     i_out += 1 # Ausgangsindex erhöhen
 x = out[2:i_out] # erste zwei entfernen und alles nach i_out (das nie befüllt wurde)

Wir sind endlich bereit für unsere Symbol/Zeitsynchronisation. Hier verwenden wir denselben Mueller-und-Müller-Taktwiederherstellungscode aus dem Kapitel :ref:`sync-chapter`. Verweise darauf, wenn du mehr darüber erfahren möchtest, wie es funktioniert. Wir setzen die Abtastwerte pro Symbol (:code:`sps`) auf 16, wie zuvor besprochen. Ein mu-Verstärkungswert von 0,01 wurde durch Experimentieren als gut funktionierend gefunden. Die Ausgabe sollte nun ein Abtastwert pro Symbol sein, d.h. unsere Ausgabe sind unsere "weichen Symbole", möglicherweise mit eingeschlossenem Frequenzoffset. Das folgende Konstellationsplot-Animation wird verwendet, um zu überprüfen, dass wir BPSK-Symbole erhalten (mit einem Frequenzoffset, der eine Rotation verursacht):

.. image:: ../_images/constellation-animated.gif
   :scale: 80 %
   :align: center
   :alt: Animation von BPSK, das sich dreht, weil noch keine feine Frequenzsynchronisation durchgeführt wurde

Wenn du dein eigenes FM-Signal verwendest und an diesem Punkt keine zwei deutlichen Cluster von komplexen Abtastwerten erhältst, bedeutet das entweder, dass die Symbolsynchronisation oben fehlgeschlagen ist, oder dass etwas mit einem der vorherigen Schritte nicht stimmt. Du musst die Konstellation nicht animieren, aber wenn du sie plottest, vermeide es, alle Abtastwerte zu plotten, da es nur wie ein Kreis aussieht. Wenn du nur 100 oder 200 Abtastwerte auf einmal plottest, bekommst du ein besseres Gefühl dafür, ob sie sich in zwei Clustern befinden oder nicht, auch wenn sie sich drehen.

********************************
Feine Frequenzsynchronisation
********************************

.. code-block:: python

 # Feine Frequenzsynchronisation
 samples = x # um mit dem Sync-Kapitel übereinzustimmen
 N = len(samples)
 phase = 0
 freq = 0
 # Diese zwei Parameter sind anzupassen, um den Feedback-Regelkreis schneller oder langsamer zu machen
 alpha = 8.0
 beta = 0.02
 out = np.zeros(N, dtype=np.complex64)
 freq_log = []
 for i in range(N):
     out[i] = samples[i] * np.exp(-1j*phase) # Eingabe um das Inverse des geschätzten Phasenoffsets anpassen
     error = np.real(out[i]) * np.imag(out[i]) # Fehlerformel für Costas-Regelkreis 2. Ordnung (z.B. für BPSK)

     # Regelkreis fortschreiben (Phase und Frequenzoffset neu berechnen)
     freq += (beta * error)
     freq_log.append(freq * sample_rate / (2*np.pi)) # von Winkelgeschwindigkeit in Hz umrechnen zum Protokollieren
     phase += freq + (alpha * error)

     # Optional: Phase so anpassen, dass sie immer zwischen 0 und 2pi liegt
     while phase >= 2*np.pi:
         phase -= 2*np.pi
     while phase < 0:
         phase += 2*np.pi
 x = out

Wir kopieren auch den Python-Code für die feine Frequenzsynchronisation aus dem Kapitel :ref:`sync-chapter`, der einen Costas-Regelkreis verwendet, um den verbleibenden Frequenzoffset zu entfernen und unser BPSK auf die reale (I)-Achse auszurichten, indem Q so nahe wie möglich an null gebracht wird. Alles, was in Q verbleibt, ist wahrscheinlich auf das Rauschen im Signal zurückzuführen, vorausgesetzt, der Costas-Regelkreis wurde ordnungsgemäß eingestellt. Aus Spaß sehen wir dieselbe Animation wie oben, aber nach der Frequenzsynchronisation (kein Drehen mehr!):

.. image:: ../_images/constellation-animated-postcostas.gif
   :scale: 80 %
   :align: center
   :alt: Animation des Frequenzsynchronisationsprozesses mit einem Costas-Regelkreis

Zusätzlich können wir den geschätzten Frequenzfehler über die Zeit betrachten, um den Costas-Regelkreis bei der Arbeit zu sehen – beachte, wie wir ihn im obigen Code protokolliert haben. Es scheint, dass etwa 13 Hz Frequenzoffset vorhanden waren, entweder aufgrund des Oszillators/LO des Senders oder des LO des Empfängers (höchstwahrscheinlich des Empfängers). Wenn du dein eigenes FM-Signal verwendest, musst du möglicherweise :code:`alpha` und :code:`beta` optimieren, bis die Kurve ähnlich aussieht; sie sollte die Synchronisation recht schnell erreichen (z.B. ein paar hundert Symbole) und sie mit minimaler Schwingung aufrechterhalten. Das Muster, das du unten nach Erreichen des Gleichgewichtszustands siehst, ist Frequenz-Jitter, keine Schwingung.

.. image:: ../_images/freq_error.png
   :scale: 40 %
   :align: center
   :alt: Der Frequenzsynchronisationsprozess mit einem Costas-Regelkreis zeigt den geschätzten Frequenzoffset über die Zeit

********************************
BPSK demodulieren
********************************

.. code-block:: python

 # BPSK demodulieren
 bits = (np.real(x) > 0).astype(int) # 1er und 0er

Das Demodulieren des BPSK ist an diesem Punkt sehr einfach: Jeder Abtastwert repräsentiert ein weiches Symbol, also müssen wir nur prüfen, ob jeder Abtastwert über oder unter 0 liegt. Das :code:`.astype(int)` sorgt dafür, dass wir mit einem Array von Integer-Werten statt einem Array von Booleans arbeiten. Du magst dich fragen, ob über/unter null 1 oder 0 repräsentiert. Wie du im nächsten Schritt sehen wirst, ist das egal!

********************************
Differentielle Dekodierung
********************************

.. code-block:: python

 # Differentielle Dekodierung, damit es keine Rolle spielt, ob unser BPSK um 180 Grad gedreht war
 bits = (bits[1:] - bits[0:-1]) % 2
 bits = bits.astype(np.uint8) # für den Decoder

Das BPSK-Signal verwendete bei seiner Erstellung differentielle Codierung, was bedeutet, dass jede 1 und 0 der ursprünglichen Daten so transformiert wurde, dass ein Wechsel von 1 zu 0 oder 0 zu 1 auf 1 gemappt wurde und kein Wechsel auf 0. Der nette Vorteil der differenziellen Codierung ist, dass man sich keine Sorgen über 180-Grad-Rotationen beim BPSK-Empfang machen muss, da es keine Rolle mehr spielt, ob wir 1 als größer oder kleiner als null betrachten. Was zählt, ist die Änderung zwischen 1 und 0. Dieses Konzept ist vielleicht leichter zu verstehen, indem man Beispieldaten betrachtet; unten werden die ersten 10 Symbole vor und nach der differenziellen Dekodierung gezeigt:

.. code-block:: python

 [1 1 1 1 0 1 0 0 1 1] # vor der differenziellen Dekodierung
 [- 0 0 0 1 1 1 0 1 0] # nach der differenziellen Dekodierung

********************************
RDS-Dekodierung
********************************

Wir haben endlich unsere Informationsbits und sind bereit, zu dekodieren, was sie bedeuten! Der unten bereitgestellte massive Codeblock ist das, was wir verwenden, um die 1er und 0er in Gruppen von Bytes zu dekodieren. Dieser Teil würde viel mehr Sinn ergeben, wenn wir zuerst den Senderteil von RDS erstellt hätten, aber wisse vorerst, dass in RDS Bytes in Gruppen von 12 Bytes gruppiert sind, wobei die ersten 8 die Daten darstellen und die letzten 4 als Synchronisationswort (sogenannte "Offset-Wörter") fungieren. Die letzten 4 Bytes werden vom nächsten Schritt (dem Parser) nicht benötigt, daher fügen wir sie nicht in die Ausgabe ein. Dieser Codeblock nimmt die oben erstellten 1er und 0er (in Form eines 1D-Arrays von uint8) auf und gibt eine Liste von Listen von Bytes aus (eine Liste von 8 Bytes, wobei diese 8 Bytes in einer Liste sind). Das macht es für den nächsten Schritt praktisch, der durch die Liste von 8 Bytes iteriert, eine Gruppe von 8 nach der anderen.

Der größte Teil des eigentlichen Dekodierungscodes dreht sich um Synchronisation (auf Byte-Ebene, nicht Symbol) und Fehlerprüfung. Er arbeitet in Blöcken von 104 Bits; jeder Block wird entweder korrekt oder fehlerhaft empfangen (mit CRC zur Prüfung), und alle 50 Blöcke wird geprüft, ob mehr als 35 davon mit Fehler empfangen wurden. In diesem Fall wird alles zurückgesetzt und versucht, erneut zu synchronisieren. Die CRC wird mit einer 10-Bit-Prüfung und Polynom :math:`x^{10}+x^8+x^7+x^5+x^4+x^3+1` durchgeführt; dies geschieht, wenn :code:`reg` mit 0x5B9 XOR-verknüpft wird, was dem binären Äquivalent dieses Polynoms entspricht. In Python sind die bitweisen Operatoren für [und, oder, nicht, xor] :code:`& | ~ ^`, genau wie in C++. Ein Linksbitshift ist :code:`x << y` (entspricht der Multiplikation von x mit 2**y), und ein Rechtsbitshift ist :code:`x >> y` (entspricht der Division von x durch 2**y), ebenfalls wie in C++.

Beachte: Du **musst** diesen Code nicht durchgehen, besonders wenn du dich auf die Physical Layer (PHY) Seite von DSP und SDR konzentrierst, da dies *keine* Signalverarbeitung darstellt. Dieser Code ist lediglich eine Implementierung eines RDS-Decoders, und im Wesentlichen kann nichts davon für andere Protokolle wiederverwendet werden, da er so spezifisch für die Funktionsweise von RDS ist.

.. code-block:: python

 # Konstanten
 syndrome = [383, 14, 303, 663, 748]
 offset_pos = [0, 1, 2, 3, 2]
 offset_word = [252, 408, 360, 436, 848]

 # siehe Anhang B, Seite 64 des Standards
 def calc_syndrome(x, mlen):
     reg = 0
     plen = 10
     for ii in range(mlen, 0, -1):
         reg = (reg << 1) | ((x >> (ii-1)) & 0x01)
         if (reg & (1 << plen)):
             reg = reg ^ 0x5B9
     for ii in range(plen, 0, -1):
         reg = reg << 1
         if (reg & (1 << plen)):
             reg = reg ^ 0x5B9
     return reg & ((1 << plen) - 1) # untere plen Bits von reg auswählen

 # Alle Arbeitsvariablen initialisieren, die wir während der Schleife benötigen
 synced = False
 presync = False

 wrong_blocks_counter = 0
 blocks_counter = 0
 group_good_blocks_counter = 0

 reg = np.uint32(0) # war unsigned long in C++ (64 Bit), aber numpy unterstützt keine bitweisen Ops von uint64
 lastseen_offset_counter = 0
 lastseen_offset = 0

 # der Synchronisationsprozess ist in Anhang C, Seite 66 des Standards beschrieben
 bytes_out = []
 for i in range(len(bits)):
     # in C++ wird reg nicht initialisiert, also ist es anfangs zufällig; bei uns sind es 0er
     # bits sind entweder 0 oder 1
     reg = np.bitwise_or(np.left_shift(reg, 1), bits[i]) # reg enthält die letzten 26 RDS-Bits
     if not synced:
         reg_syndrome = calc_syndrome(reg, 26)
         for j in range(5):
             if reg_syndrome == syndrome[j]:
                 if not presync:
                     lastseen_offset = j
                     lastseen_offset_counter = i
                     presync = True
                 else:
                     if offset_pos[lastseen_offset] >= offset_pos[j]:
                         block_distance = offset_pos[j] + 4 - offset_pos[lastseen_offset]
                     else:
                         block_distance = offset_pos[j] - offset_pos[lastseen_offset]
                     if (block_distance*26) != (i - lastseen_offset_counter):
                         presync = False
                     else:
                         print('Sync State Detected')
                         wrong_blocks_counter = 0
                         blocks_counter = 0
                         block_bit_counter = 0
                         block_number = (j + 1) % 4
                         group_assembly_started = False
                         synced = True
             break # Syndrom gefunden, keine weiteren Zyklen

     else: # SYNCHRONISIERT
         # warten bis 26 Bits in den Puffer kommen
         if block_bit_counter < 25:
             block_bit_counter += 1
         else:
             good_block = False
             dataword = (reg >> 10) & 0xffff
             block_calculated_crc = calc_syndrome(dataword, 16)
             checkword = reg & 0x3ff
             if block_number == 2: # Sonderfall von C oder C' Offset-Wort verwalten
                 block_received_crc = checkword ^ offset_word[block_number]
                 if (block_received_crc == block_calculated_crc):
                     good_block = True
                 else:
                     block_received_crc = checkword ^ offset_word[4]
                     if (block_received_crc == block_calculated_crc):
                         good_block = True
                     else:
                         wrong_blocks_counter += 1
                         good_block = False
             else:
                 block_received_crc = checkword ^ offset_word[block_number] # bitweises XOR
                 if block_received_crc == block_calculated_crc:
                     good_block = True
                 else:
                     wrong_blocks_counter += 1
                     good_block = False

             # CRC-Prüfung abgeschlossen
             if block_number == 0 and good_block:
                 group_assembly_started = True
                 group_good_blocks_counter = 1
                 group = bytearray(8) # 8 Bytes mit 0er befüllt
             if group_assembly_started:
                 if not good_block:
                     group_assembly_started = False
                 else:
                     # rohe Datenbytes, wie von RDS empfangen
                     group[block_number*2] = (dataword >> 8) & 255
                     group[block_number*2+1] = dataword & 255
                     group_good_blocks_counter += 1
                 if group_good_blocks_counter == 5:
                     bytes_out.append(group) # Liste von Länge-8-Listen von Bytes
             block_bit_counter = 0
             block_number = (block_number + 1) % 4
             blocks_counter += 1
             if blocks_counter == 50:
                 if wrong_blocks_counter > 35: # So viele falsche Blöcke bedeuten Sync-Verlust
                     print("Lost Sync (Got ", wrong_blocks_counter, " bad blocks on ", blocks_counter, " total)")
                     synced = False
                     presync = False
                 else:
                     print("Still Sync-ed (Got ", wrong_blocks_counter, " bad blocks on ", blocks_counter, " total)")
                 blocks_counter = 0
                 wrong_blocks_counter = 0

Unten ist eine Beispielausgabe dieses Dekodierungsschritts. Beachte, wie in diesem Beispiel die Synchronisation recht schnell erfolgte, aber dann aus irgendeinem Grund ein paar Mal verloren geht, obwohl alle Daten noch geparst werden können, wie wir sehen werden. Wenn du die herunterladbare 1M-Abtastwerte-Datei verwendest, siehst du nur die ersten paar Zeilen unten. Der tatsächliche Inhalt dieser Bytes sieht je nach Darstellung wie Zufallszahlen/-zeichen aus, aber im nächsten Schritt werden wir sie in menschenlesbare Informationen parsen!

.. code-block:: console

 Sync State Detected
 Still Sync-ed (Got  0  bad blocks on  50  total)
 Still Sync-ed (Got  0  bad blocks on  50  total)
 Still Sync-ed (Got  0  bad blocks on  50  total)
 Still Sync-ed (Got  0  bad blocks on  50  total)
 Still Sync-ed (Got  1  bad blocks on  50  total)
 Still Sync-ed (Got  5  bad blocks on  50  total)
 Still Sync-ed (Got  26  bad blocks on  50  total)
 Lost Sync (Got  50  bad blocks on  50  total)
 Sync State Detected
 Still Sync-ed (Got  3  bad blocks on  50  total)
 ...

********************************
RDS-Parsen
********************************

Da wir jetzt Bytes in Gruppen von 8 haben, können wir die endgültigen Daten extrahieren, d.h. die endgültige Ausgabe, die für Menschen verständlich ist. Dies ist als Parsen der Bytes bekannt, und wie der Decoder im vorherigen Abschnitt ist es einfach eine Implementierung des RDS-Protokolls und ist wirklich nicht so wichtig zu verstehen. Glücklicherweise ist es nicht viel Code, wenn man die zwei am Anfang definierten Tabellen nicht einrechnet, die einfach die Lookup-Tabellen für den Typ des UKW-Kanals und das Versorgungsgebiet sind.

Für diejenigen, die lernen möchten, wie dieser Code funktioniert, gebe ich einige zusätzliche Informationen. Das Protokoll verwendet das Konzept eines A/B-Flags, was bedeutet, dass einige Nachrichten als A und andere als B markiert sind, und das Parsen ändert sich je nachdem, ob es A oder B ist (ob es A oder B ist, wird im dritten Bit des zweiten Bytes gespeichert). Es verwendet auch verschiedene "Gruppen"-Typen, die Nachrichtentypen ähneln. In diesem Code parsen wir nur Nachrichtentyp 2, das ist der Nachrichtentyp, der den Radiotext enthält – das ist der interessante Teil: der Text, der auf dem Display im Auto scrollt. Wir werden immer noch in der Lage sein, den Kanaltyp und die Region zu parsen, da sie in jeder Nachricht gespeichert sind. Beachte, dass :code:`radiotext` eine Zeichenkette ist, die auf lauter Leerzeichen initialisiert wird, langsam befüllt wird während Bytes geparst werden, und dann zu lauter Leerzeichen zurückgesetzt wird, wenn eine bestimmte Gruppe von Bytes empfangen wird.

.. code-block:: python

 # Anhang F des RBDS-Standards Tabelle F.1 (Nordamerika) und Tabelle F.2 (Europa)
 #              Europa                   Nordamerika
 pty_table = [["Undefined",             "Undefined"],
              ["News",                  "News"],
              ["Current Affairs",       "Information"],
              ["Information",           "Sports"],
              ["Sport",                 "Talk"],
              ["Education",             "Rock"],
              ["Drama",                 "Classic Rock"],
              ["Culture",               "Adult Hits"],
              ["Science",               "Soft Rock"],
              ["Varied",                "Top 40"],
              ["Pop Music",             "Country"],
              ["Rock Music",            "Oldies"],
              ["Easy Listening",        "Soft"],
              ["Light Classical",       "Nostalgia"],
              ["Serious Classical",     "Jazz"],
              ["Other Music",           "Classical"],
              ["Weather",               "Rhythm & Blues"],
              ["Finance",               "Soft Rhythm & Blues"],
              ["Children's Programmes", "Language"],
              ["Social Affairs",        "Religious Music"],
              ["Religion",              "Religious Talk"],
              ["Phone-In",              "Personality"],
              ["Travel",                "Public"],
              ["Leisure",               "College"],
              ["Jazz Music",            "Spanish Talk"],
              ["Country Music",         "Spanish Music"],
              ["National Music",        "Hip Hop"],
              ["Oldies Music",          "Unassigned"],
              ["Folk Music",            "Unassigned"],
              ["Documentary",           "Weather"],
              ["Alarm Test",            "Emergency Test"],
              ["Alarm",                 "Emergency"]]
 pty_locale = 1 # auf 0 für Europa setzen, um die erste Spalte zu verwenden

 # Seite 72, Anhang D, Tabelle D.2 im Standard
 coverage_area_codes = ["Local",
                        "International",
                        "National",
                        "Supra-regional",
                        "Regional 1",
                        "Regional 2",
                        "Regional 3",
                        "Regional 4",
                        "Regional 5",
                        "Regional 6",
                        "Regional 7",
                        "Regional 8",
                        "Regional 9",
                        "Regional 10",
                        "Regional 11",
                        "Regional 12"]

 radiotext_AB_flag = 0
 radiotext = [' ']*65
 first_time = True
 for group in bytes_out:
     group_0 = group[1] | (group[0] << 8)
     group_1 = group[3] | (group[2] << 8)
     group_2 = group[5] | (group[4] << 8)
     group_3 = group[7] | (group[6] << 8)

     group_type = (group_1 >> 12) & 0xf # Bedeutung: ["BASIC", "PIN/SL", "RT", "AID", "CT", "TDC", "IH", "RP", "TMC", "EWS", "___", "___", "___", "___", "EON", "___"]
     AB = (group_1 >> 11 ) & 0x1 # b wenn 1, a wenn 0

     program_identification = group_0     # "PI"

     program_type = (group_1 >> 5) & 0x1f # "PTY"
     pty = pty_table[program_type][pty_locale]

     pi_area_coverage = (program_identification >> 8) & 0xf
     coverage_area = coverage_area_codes[pi_area_coverage]

     pi_program_reference_number = program_identification & 0xff # einfach ein Int

     if first_time:
         print("PTY:", pty)
         print("program:", pi_program_reference_number)
         print("coverage_area:", coverage_area)
         first_time = False

     if group_type == 2:
         # wenn das A/B-Flag umgeschaltet wird, aktuellen Radiotext leeren
         if radiotext_AB_flag != ((group_1 >> 4) & 0x01):
             radiotext = [' ']*65
         radiotext_AB_flag = (group_1 >> 4) & 0x01
         text_segment_address_code = group_1 & 0x0f
         if AB:
             radiotext[text_segment_address_code * 2    ] = chr((group_3 >> 8) & 0xff)
             radiotext[text_segment_address_code * 2 + 1] = chr(group_3        & 0xff)
         else:
             radiotext[text_segment_address_code *4     ] = chr((group_2 >> 8) & 0xff)
             radiotext[text_segment_address_code * 4 + 1] = chr(group_2        & 0xff)
             radiotext[text_segment_address_code * 4 + 2] = chr((group_3 >> 8) & 0xff)
             radiotext[text_segment_address_code * 4 + 3] = chr(group_3        & 0xff)
         print(''.join(radiotext))
     else:
         pass

Unten ist die Ausgabe des Parse-Schritts für ein Beispiel-UKW-Sender. Beachte, wie er die Radiotext-Zeichenkette über mehrere Nachrichten aufbauen muss und dann periodisch die Zeichenkette löscht und von vorne beginnt. Wenn du die 1M-Abtastwerte-Datei verwendest, siehst du nur die ersten paar Zeilen unten.

.. code-block:: console

 PTY: Top 40
 program: 29
 coverage_area: Regional 4
             ing.
             ing. Upb
             ing. Upbeat.
             ing. Upbeat. Rea

 WAY-
 WAY-FM U
 WAY-FM Uplif
 WAY-FM Uplifting
 WAY-FM Uplifting. Up
 WAY-FM Uplifting. Upbeat
 WAY-FM Uplifting. Upbeat. Re

 WayF
 WayFM Up
 WayFM Uplift
 WayFM Uplifting.
 WayFM Uplifting. Upb
 WayFM Uplifting. Upbeat.
 WayFM Uplifting. Upbeat. Rea



********************************
Abschluss und endgültiger Code
********************************

Geschafft! Unten ist der gesamte obige Code zusammengefügt. Er sollte mit der `Test-UKW-Radioaufzeichnung, die du hier findest <https://github.com/777arc/498x/blob/master/fm_rds_250k_1Msamples.iq?raw=true>`_, funktionieren. Du solltest auch dein eigenes Signal einspeisen können, solange es mit ausreichend hohem SNR empfangen wird. Wenn du Anpassungen vornehmen musstest, um es mit deiner eigenen Aufzeichnung oder einem Live-SDR zum Laufen zu bringen, teile uns mit, was du getan hast; du kannst es als GitHub-PR auf `der GitHub-Seite des Lehrbuchs <https://github.com/777arc/PySDR>`_ einreichen. Eine Version dieses Codes mit Dutzenden von Debug-Plots/Prints `findest du hier <https://github.com/777arc/PySDR/blob/master/figure-generating-scripts/rds_demo.py>`_.

.. raw:: html

   <details>
   <summary>Endgültiger Code</summary>

.. code-block:: python

 import numpy as np
 from scipy.signal import resample_poly, firwin, bilinear, lfilter
 import matplotlib.pyplot as plt

 # Signal einlesen
 x = np.fromfile('/your/path/fm_rds_250k_1Msamples.iq', dtype=np.complex64)
 sample_rate = 250e3
 center_freq = 99.5e6

 # Quadratur-Demodulation
 x = 0.5 * np.angle(x[0:-1] * np.conj(x[1:]))

 # Frequenzverschiebung
 N = len(x)
 f_o = -57e3
 t = np.arange(N)/sample_rate
 x = x * np.exp(2j*np.pi*f_o*t)

 # Tiefpassfilter
 taps = firwin(numtaps=101, cutoff=7.5e3, fs=sample_rate)
 x = np.convolve(x, taps, 'valid')

 # Um 10 dezimieren
 x = x[::10]
 sample_rate = 25e3

 # Auf 19 kHz resamplen
 x = resample_poly(x, 19, 25)
 sample_rate = 19e3

 # Symbolsynchronisation
 samples = x
 samples_interpolated = resample_poly(samples, 32, 1)
 sps = 16
 mu = 0.01
 out = np.zeros(len(samples) + 10, dtype=np.complex64)
 out_rail = np.zeros(len(samples) + 10, dtype=np.complex64)
 i_in = 0
 i_out = 2
 while i_out < len(samples) and i_in+32 < len(samples):
     out[i_out] = samples_interpolated[i_in*32 + int(mu*32)]
     out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j*int(np.imag(out[i_out]) > 0)
     x = (out_rail[i_out] - out_rail[i_out-2]) * np.conj(out[i_out-1])
     y = (out[i_out] - out[i_out-2]) * np.conj(out_rail[i_out-1])
     mm_val = np.real(y - x)
     mu += sps + 0.01*mm_val
     i_in += int(np.floor(mu))
     mu = mu - np.floor(mu)
     i_out += 1
 x = out[2:i_out]

 sample_rate /= 16

 # Feine Frequenzsynchronisation
 samples = x
 N = len(samples)
 phase = 0
 freq = 0
 alpha = 8.0
 beta = 0.02
 out = np.zeros(N, dtype=np.complex64)
 freq_log = []
 for i in range(N):
     out[i] = samples[i] * np.exp(-1j*phase)
     error = np.real(out[i]) * np.imag(out[i])
     freq += (beta * error)
     freq_log.append(freq * sample_rate / (2*np.pi))
     phase += freq + (alpha * error)
     while phase >= 2*np.pi:
         phase -= 2*np.pi
     while phase < 0:
         phase += 2*np.pi
 x = out

 # BPSK demodulieren
 bits = (np.real(x) > 0).astype(int)

 # Differentielle Dekodierung
 bits = (bits[1:] - bits[0:-1]) % 2
 bits = bits.astype(np.uint8)

 ###########
 # DECODER #
 ###########

 syndrome = [383, 14, 303, 663, 748]
 offset_pos = [0, 1, 2, 3, 2]
 offset_word = [252, 408, 360, 436, 848]

 def calc_syndrome(x, mlen):
     reg = 0
     plen = 10
     for ii in range(mlen, 0, -1):
         reg = (reg << 1) | ((x >> (ii-1)) & 0x01)
         if (reg & (1 << plen)):
             reg = reg ^ 0x5B9
     for ii in range(plen, 0, -1):
         reg = reg << 1
         if (reg & (1 << plen)):
             reg = reg ^ 0x5B9
     return reg & ((1 << plen) - 1)

 synced = False
 presync = False
 wrong_blocks_counter = 0
 blocks_counter = 0
 group_good_blocks_counter = 0
 reg = np.uint32(0)
 lastseen_offset_counter = 0
 lastseen_offset = 0
 bytes_out = []
 for i in range(len(bits)):
     reg = np.bitwise_or(np.left_shift(reg, 1), bits[i])
     if not synced:
         reg_syndrome = calc_syndrome(reg, 26)
         for j in range(5):
             if reg_syndrome == syndrome[j]:
                 if not presync:
                     lastseen_offset = j
                     lastseen_offset_counter = i
                     presync = True
                 else:
                     if offset_pos[lastseen_offset] >= offset_pos[j]:
                         block_distance = offset_pos[j] + 4 - offset_pos[lastseen_offset]
                     else:
                         block_distance = offset_pos[j] - offset_pos[lastseen_offset]
                     if (block_distance*26) != (i - lastseen_offset_counter):
                         presync = False
                     else:
                         print('Sync State Detected')
                         wrong_blocks_counter = 0
                         blocks_counter = 0
                         block_bit_counter = 0
                         block_number = (j + 1) % 4
                         group_assembly_started = False
                         synced = True
             break
     else:
         if block_bit_counter < 25:
             block_bit_counter += 1
         else:
             good_block = False
             dataword = (reg >> 10) & 0xffff
             block_calculated_crc = calc_syndrome(dataword, 16)
             checkword = reg & 0x3ff
             if block_number == 2:
                 block_received_crc = checkword ^ offset_word[block_number]
                 if (block_received_crc == block_calculated_crc):
                     good_block = True
                 else:
                     block_received_crc = checkword ^ offset_word[4]
                     if (block_received_crc == block_calculated_crc):
                         good_block = True
                     else:
                         wrong_blocks_counter += 1
                         good_block = False
             else:
                 block_received_crc = checkword ^ offset_word[block_number]
                 if block_received_crc == block_calculated_crc:
                     good_block = True
                 else:
                     wrong_blocks_counter += 1
                     good_block = False
             if block_number == 0 and good_block:
                 group_assembly_started = True
                 group_good_blocks_counter = 1
                 group = bytearray(8)
             if group_assembly_started:
                 if not good_block:
                     group_assembly_started = False
                 else:
                     group[block_number*2] = (dataword >> 8) & 255
                     group[block_number*2+1] = dataword & 255
                     group_good_blocks_counter += 1
                 if group_good_blocks_counter == 5:
                     bytes_out.append(group)
             block_bit_counter = 0
             block_number = (block_number + 1) % 4
             blocks_counter += 1
             if blocks_counter == 50:
                 if wrong_blocks_counter > 35:
                     print("Lost Sync (Got ", wrong_blocks_counter, " bad blocks on ", blocks_counter, " total)")
                     synced = False
                     presync = False
                 else:
                     print("Still Sync-ed (Got ", wrong_blocks_counter, " bad blocks on ", blocks_counter, " total)")
                 blocks_counter = 0
                 wrong_blocks_counter = 0

 ###########
 # PARSER  #
 ###########

 pty_table = [["Undefined",             "Undefined"],
              ["News",                  "News"],
              ["Current Affairs",       "Information"],
              ["Information",           "Sports"],
              ["Sport",                 "Talk"],
              ["Education",             "Rock"],
              ["Drama",                 "Classic Rock"],
              ["Culture",               "Adult Hits"],
              ["Science",               "Soft Rock"],
              ["Varied",                "Top 40"],
              ["Pop Music",             "Country"],
              ["Rock Music",            "Oldies"],
              ["Easy Listening",        "Soft"],
              ["Light Classical",       "Nostalgia"],
              ["Serious Classical",     "Jazz"],
              ["Other Music",           "Classical"],
              ["Weather",               "Rhythm & Blues"],
              ["Finance",               "Soft Rhythm & Blues"],
              ["Children's Programmes", "Language"],
              ["Social Affairs",        "Religious Music"],
              ["Religion",              "Religious Talk"],
              ["Phone-In",              "Personality"],
              ["Travel",                "Public"],
              ["Leisure",               "College"],
              ["Jazz Music",            "Spanish Talk"],
              ["Country Music",         "Spanish Music"],
              ["National Music",        "Hip Hop"],
              ["Oldies Music",          "Unassigned"],
              ["Folk Music",            "Unassigned"],
              ["Documentary",           "Weather"],
              ["Alarm Test",            "Emergency Test"],
              ["Alarm",                 "Emergency"]]
 pty_locale = 1

 coverage_area_codes = ["Local",
                        "International",
                        "National",
                        "Supra-regional",
                        "Regional 1",
                        "Regional 2",
                        "Regional 3",
                        "Regional 4",
                        "Regional 5",
                        "Regional 6",
                        "Regional 7",
                        "Regional 8",
                        "Regional 9",
                        "Regional 10",
                        "Regional 11",
                        "Regional 12"]

 radiotext_AB_flag = 0
 radiotext = [' ']*65
 first_time = True
 for group in bytes_out:
     group_0 = group[1] | (group[0] << 8)
     group_1 = group[3] | (group[2] << 8)
     group_2 = group[5] | (group[4] << 8)
     group_3 = group[7] | (group[6] << 8)
     group_type = (group_1 >> 12) & 0xf
     AB = (group_1 >> 11 ) & 0x1
     program_identification = group_0
     program_type = (group_1 >> 5) & 0x1f
     pty = pty_table[program_type][pty_locale]
     pi_area_coverage = (program_identification >> 8) & 0xf
     coverage_area = coverage_area_codes[pi_area_coverage]
     pi_program_reference_number = program_identification & 0xff
     if first_time:
         print("PTY:", pty)
         print("program:", pi_program_reference_number)
         print("coverage_area:", coverage_area)
         first_time = False
     if group_type == 2:
         if radiotext_AB_flag != ((group_1 >> 4) & 0x01):
             radiotext = [' ']*65
         radiotext_AB_flag = (group_1 >> 4) & 0x01
         text_segment_address_code = group_1 & 0x0f
         if AB:
             radiotext[text_segment_address_code * 2    ] = chr((group_3 >> 8) & 0xff)
             radiotext[text_segment_address_code * 2 + 1] = chr(group_3        & 0xff)
         else:
             radiotext[text_segment_address_code *4     ] = chr((group_2 >> 8) & 0xff)
             radiotext[text_segment_address_code * 4 + 1] = chr(group_2        & 0xff)
             radiotext[text_segment_address_code * 4 + 2] = chr((group_3 >> 8) & 0xff)
             radiotext[text_segment_address_code * 4 + 3] = chr(group_3        & 0xff)
         print(''.join(radiotext))
     else:
         pass

.. raw:: html

   </details>

Die Beispiel-UKW-Aufzeichnung, die mit diesem Code funktioniert, `findest du hier <https://github.com/777arc/498x/blob/master/fm_rds_250k_1Msamples.iq?raw=true>`_.

Für diejenigen, die das eigentliche Audiosignal demodulieren möchten, füge einfach die folgenden Zeilen direkt nach dem Abschnitt "Ein Signal erfassen" ein (besonderer Dank an `Joel Cordeiro <http://github.com/joeugenio>`_ für den Code):

.. code-block:: python

 # Folgenden Code direkt nach dem Abschnitt "Ein Signal erfassen" einfügen

 from scipy.io import wavfile

 # Demodulation
 x = np.diff(np.unwrap(np.angle(x)))

 # De-Emphasis-Filter, H(s) = 1/(RC*s + 1), implementiert als IIR via bilineare Transformation
 bz, az = bilinear(1, [75e-6, 1], fs=sample_rate)
 x = lfilter(bz, az, x)

 # um 6 dezimieren, um Mono-Audio zu erhalten
 x = x[::6]
 sample_rate_audio = sample_rate/6

 # Lautstärke normalisieren, damit sie zwischen -1 und +1 liegt
 x /= np.max(np.abs(x))

 # manche Maschinen wollen int16
 x *= 32767
 x = x.astype(np.int16)

 # Als WAV-Datei speichern (z.B. in Audacity öffnen)
 wavfile.write('fm.wav', int(sample_rate_audio), x)

Der komplizierteste Teil ist der De-Emphasis-Filter, `über den du hier mehr erfahren kannst <https://wiki.gnuradio.org/index.php/FM_Preemphasis>`_, obwohl er eigentlich ein optionaler Schritt ist, wenn du mit Audio mit unausgewogener Bass-/Höhenbalance einverstanden bist. Für Neugierige sieht die Frequenzantwort des `IIR <https://en.wikipedia.org/wiki/Infinite_impulse_response>`_-De-Emphasis-Filters so aus. Er filtert keine Frequenzen vollständig heraus, sondern ist eher ein "Formgebungs"-Filter.

.. image:: ../_images/fm_demph_filter_freq_response.svg
   :align: center
   :target: ../_images/fm_demph_filter_freq_response.svg

********************************
Danksagungen
********************************

Die meisten der oben beschriebenen Schritte zum Empfangen von RDS wurden aus der GNU Radio-Implementierung von RDS adaptiert, die im GNU Radio Out-of-Tree-Modul namens `gr-rds <https://github.com/bastibl/gr-rds>`_ lebt, ursprünglich von Dimitrios Symeonidis erstellt und von Bastian Bloessl gepflegt. Ich möchte die Arbeit dieser Autoren würdigen. Um dieses Kapitel zu erstellen, begann ich damit, gr-rds in GNU Radio mit einer funktionierenden UKW-Aufzeichnung zu verwenden, und konvertierte jeden der Blöcke (einschließlich vieler eingebauter Blöcke) langsam zu Python. Es brauchte viel Zeit; es gibt einige Nuancen zu den eingebauten Blöcken, die leicht zu übersehen sind, und der Wechsel von stream-artigem Signalprozessieren (d.h. die Verwendung einer Work-Funktion, die einige tausend Abtastwerte auf zustandsbehaftete Weise verarbeitet) zu einem Python-Block ist nicht immer unkompliziert. GNU Radio ist ein erstaunliches Werkzeug für diese Art von Prototyping, und ich hätte nie all diesen funktionierenden Python-Code ohne es erstellen können.

********************************
Weiterführende Lektüre
********************************

#. https://en.wikipedia.org/wiki/Radio_Data_System
#. `https://www.sigidwiki.com/wiki/Radio_Data_System_(RDS) <https://www.sigidwiki.com/wiki/Radio_Data_System_(RDS)>`_
#. https://github.com/bastibl/gr-rds
#. https://www.gnuradio.org/
