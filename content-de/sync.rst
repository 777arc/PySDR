.. _sync-chapter:

################
Synchronisation
################

Dieses Kapitel behandelt die Synchronisation drahtloser Signale in Zeit und Frequenz, um Trägerfrequenzoffsets zu korrigieren und eine Timing-Ausrichtung auf Symbol- und Rahmenebene durchzuführen. Wir werden die Mueller-und-Müller-Taktwiederherstellungstechnik und den Costas-Regelkreis in Python nutzen.

***************************
Einführung
***************************

Wir haben besprochen, wie man digital über die Luft sendet, indem man ein digitales Modulationsverfahren wie QPSK verwendet und Impulsformung anwendet, um die Signalbandbreite zu begrenzen. Kanalcodierung kann verwendet werden, um mit verrauschten Kanälen umzugehen, z.B. wenn das SNR am Empfänger niedrig ist. Es hilft immer, so viel wie möglich herauszufiltern, bevor man das Signal digital verarbeitet. In diesem Kapitel untersuchen wir, wie Synchronisation auf der Empfangsseite durchgeführt wird. Synchronisation ist eine Reihe von Verarbeitungsschritten, die *vor* der Demodulation und Kanaldecodierung stattfinden. Die gesamte Tx-Kanal-Rx-Kette ist unten dargestellt, wobei die in diesem Kapitel behandelten Blöcke gelb hervorgehoben sind. (Dieses Diagramm ist nicht vollständig – die meisten Systeme enthalten auch Entzerrung und Multiplexing.)

.. image:: ../_images/sync-diagram.svg
   :align: center
   :target: ../_images/sync-diagram.svg
   :alt: Die Sende-Empfangs-Kette, mit den in diesem Kapitel besprochenen Blöcken gelb hervorgehoben, inklusive Zeit- und Frequenzsynchronisation

***************************
Drahtlosen Kanal simulieren
***************************

Bevor wir lernen, wie man Zeit- und Frequenzsynchronisation implementiert, müssen wir unsere simulierten Signale realistischer machen. Ohne das Hinzufügen einer zufälligen Zeitverzögerung ist die Zeitdomain-Synchronisation trivial. Tatsächlich muss man nur die Abtastverzögerung der verwendeten Filter berücksichtigen. Wir möchten auch einen Frequenzoffset simulieren, denn wie wir besprechen werden, sind Oszillatoren nicht perfekt; es wird immer einen gewissen Offset zwischen den Mittenfrequenzen von Sender und Empfänger geben.

Untersuchen wir Python-Code zur Simulation einer nicht-ganzzahligen Verzögerung und eines Frequenzoffsets. Der Python-Code in diesem Kapitel baut auf dem Code auf, den wir während der Python-Übung zur Impulsformung geschrieben haben (klicke unten, falls du ihn benötigst); du kannst ihn als Ausgangspunkt des Codes in diesem Kapitel betrachten, und aller neue Code kommt danach.

.. raw:: html

   <details>
   <summary>Python-Code aus der Impulsformung</summary>

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal
    import math

    # Dieser Teil stammt aus der Impulsformungs-Übung
    num_symbols = 100
    sps = 8
    bits = np.random.randint(0, 2, num_symbols) # Zu übertragende Daten, 1er und 0er
    pulse_train = np.array([])
    for bit in bits:
        pulse = np.zeros(sps)
        pulse[0] = bit*2-1 # ersten Wert auf 1 oder -1 setzen
        pulse_train = np.concatenate((pulse_train, pulse)) # die 8 Abtastwerte zum Signal hinzufügen

    # Raised-Cosine-Filter erstellen
    num_taps = 101
    beta = 0.35
    Ts = sps # Abtastrate als 1 Hz angenommen, Abtastperiode ist 1, Symbolperiode ist 8
    t = np.arange(-51, 52) # letzte Zahl nicht enthalten
    h = np.sinc(t/Ts) * np.cos(np.pi*beta*t/Ts) / (1 - (2*beta*t/Ts)**2)

    # Signal filtern, um Impulsformung anzuwenden
    samples = np.convolve(pulse_train, h, "same")

.. raw:: html

   </details>

Wir lassen den Code für Plots weg, da du mittlerweile gelernt hast, wie man beliebige Signale plottet. Plots wie in diesem Lehrbuch hübsch zu gestalten erfordert viel Zusatzcode, der zum Verständnis nicht notwendig ist.


Verzögerung hinzufügen
##############

Wir können eine Verzögerung einfach durch Verschieben von Abtastwerten simulieren, aber das simuliert nur eine Verzögerung, die ein ganzzahliges Vielfaches unserer Abtastperiode ist. In der realen Welt wird die Verzögerung ein Bruchteil einer Abtastperiode sein. Wir können die Verzögerung um einen Bruchteil einer Abtastperiode simulieren, indem wir einen "fraktionalen Verzögerungsfilter" erstellen, der alle Frequenzen durchlässt, aber die Abtastwerte um einen Betrag verzögert, der nicht auf das Abtastintervall beschränkt ist. Du kannst es dir als Allpass-Filter vorstellen, das auf alle Frequenzen dieselbe Phasenverschiebung anwendet. (Zur Erinnerung: Zeitverzögerung und Phasenverschiebung sind äquivalent.) Der Python-Code zum Erstellen dieses Filters wird unten gezeigt:

.. code-block:: python

    # Fraktionalen Verzögerungsfilter erstellen und anwenden
    delay = 0.4 # fraktionale Verzögerung, in Abtastwerten
    N = 21 # Anzahl der Taps, ungerade halten
    n = np.arange(-(N-1)//2, N//2+1) # -10,-9,...,0,...,9,10
    h = np.sinc(n - delay) # Filtertaps berechnen
    h *= np.hamming(N) # Filter fenstern, damit er auf beiden Seiten auf 0 abklingt
    h /= np.sum(h) # normalisieren für Einheitsverstärkung, Amplitude/Leistung nicht verändern
    samples = np.convolve(samples, h) # Filter anwenden

Wie du sehen kannst, berechnen wir die Filtertaps mithilfe einer sinc()-Funktion. Ein Sinc im Zeitbereich ist ein Rechteck im Frequenzbereich, und unser Rechteck für diesen Filter umfasst den gesamten Frequenzbereich unseres Signals. Dieser Filter verändert das Signal nicht, er verzögert es lediglich in der Zeit. In unserem Beispiel verzögern wir um 0,4 Abtastwerte. Beachte, dass das Anwenden *jedes* Filters ein Signal um die Hälfte der Filtertaps minus eins verzögert, durch den Vorgang der Faltung des Signals durch den Filter.

Wenn wir den "Vorher"- und "Nachher"-Zustand der Filterung eines Signals darstellen, können wir die fraktionale Verzögerung beobachten. In unserem Plot zoomen wir auf nur ein paar Symbole. Sonst ist die fraktionale Verzögerung nicht erkennbar.

.. image:: ../_images/fractional-delay-filter.svg
   :align: center
   :target: ../_images/fractional-delay-filter.svg



Frequenzoffset hinzufügen
##########################

Um unser simuliertes Signal realistischer zu machen, wenden wir einen Frequenzoffset an. Sagen wir, unsere Abtastrate in dieser Simulation ist 1 MHz (es spielt keine Rolle, wie groß sie tatsächlich ist, aber du wirst sehen, warum es einfacher ist, eine Zahl zu wählen). Wenn wir einen Frequenzoffset von 13 kHz (eine beliebige Zahl) simulieren möchten, können wir dies mit folgendem Code tun:

.. code-block:: python

    # Frequenzoffset anwenden
    fs = 1e6 # Abtastrate als 1 MHz angenommen
    fo = 13000 # Frequenzoffset simulieren
    Ts = 1/fs # Abtastperiode berechnen
    t = np.arange(0, Ts*len(samples), Ts) # Zeitvektor erstellen
    samples = samples * np.exp(1j*2*np.pi*fo*t) # Frequenzverschiebung durchführen

Unten wird das Signal vor und nach dem Frequenzoffset demonstriert.

.. image:: ../_images/sync-freq-offset.svg
   :align: center
   :target: ../_images/sync-freq-offset.svg
   :alt: Python-Simulation eines Signals vor und nach dem Anwenden eines Frequenzoffsets

Wir haben den Q-Anteil nicht geplottet, da wir BPSK übertragen haben, wobei Q immer null ist. Da wir jetzt eine Frequenzverschiebung hinzufügen, um drahtlose Kanäle zu simulieren, verteilt sich die Energie auf I und Q. Ab jetzt sollten wir sowohl I als auch Q plotten. Probiere gerne einen anderen Frequenzoffset in deinem Code aus. Wenn du den Offset auf etwa 1 kHz absenkst, kannst du die Sinuswelle in der Hüllkurve des Signals sehen, da sie langsam genug schwingt, um mehrere Symbole zu umspannen.

Was die Wahl einer beliebigen Abtastrate betrifft: Wenn du den Code genauer betrachtest, wirst du bemerken, dass es auf das Verhältnis von :code:`fo` zu :code:`fs` ankommt.

Du kannst dir vorstellen, dass die beiden oben gezeigten Codeblöcke den drahtlosen Kanal simulieren. Der Code sollte nach dem sendeseitigen Code (was wir im Kapitel zur Impulsformung gemacht haben) und vor dem empfangsseitigen Code kommen, den wir im Rest dieses Kapitels erkunden werden.

***************************
Zeitsynchronisation
***************************

Wenn wir ein Signal drahtlos übertragen, kommt es beim Empfänger mit einer zufälligen Phasenverschiebung an, die durch die zurückgelegte Zeit verursacht wird. Wir können nicht einfach mit unserer Symbolrate Symbole abtasten, da wir das Signal wahrscheinlich nicht am richtigen Punkt des Impulses abtasten, wie am Ende des Kapitels :ref:`pulse-shaping-chapter` besprochen. Schau dir die drei Abbildungen am Ende dieses Kapitels an, falls du es nicht mehr im Kopf hast.

Die meisten Timing-Synchronisationstechniken haben die Form eines Phasenregelkreises (PLL); wir werden PLLs hier nicht im Detail studieren, aber es ist wichtig, den Begriff zu kennen, und du kannst bei Interesse selbst darüber lesen. PLLs sind geschlossene Regelkreise, die Feedback verwenden, um kontinuierlich etwas anzupassen; in unserem Fall ermöglicht eine Zeitverschiebung die Abtastung am Peak der digitalen Symbole.

Du kannst dir die Timing-Wiederherstellung als einen Block im Empfänger vorstellen, der einen Strom von Abtastwerten akzeptiert und einen anderen Strom ausgibt (ähnlich wie ein Filter). Wir programmieren diesen Timing-Wiederherstellungsblock mit Informationen über unser Signal, wobei die Anzahl der Abtastwerte pro Symbol am wichtigsten ist (oder unsere beste Schätzung davon, wenn wir nicht 100% sicher sind, was übertragen wurde). Dieser Block wirkt als "Dezimator", d.h. unsere Abtastausgabe ist ein Bruchteil der Anzahl der eingehenden Abtastwerte. Wir möchten einen Abtastwert pro digitalem Symbol, also ist die Dezimationsrate einfach die Anzahl der Abtastwerte pro Symbol. Wenn der Sender mit 1M Symbolen pro Sekunde sendet und wir mit 16 Msps abtasten, erhalten wir 16 Abtastwerte pro Symbol. Das ist die Abtastrate, die in den Timing-Sync-Block eingeht. Die Abtastrate, die aus dem Block herauskommt, beträgt 1 Msps, da wir einen Abtastwert pro digitalem Symbol möchten.

Die meisten Timing-Wiederherstellungsmethoden nutzen die Tatsache, dass unsere digitalen Symbole ansteigen und dann abfallen, und der Scheitelpunkt ist der Punkt, an dem wir das Symbol abtasten möchten. Anders ausgedrückt: Wir tasten den maximalen Punkt nach dem Betrag ab:

.. image:: ../_images/symbol_sync2.png
   :scale: 40 %
   :align: center

Es gibt viele Timing-Wiederherstellungsmethoden, die meisten ähneln einem PLL. Im Allgemeinen ist der Unterschied zwischen ihnen die Gleichung, die zur "Korrektur" des Timing-Offsets verwendet wird, den wir als :math:`\mu` oder :code:`mu` im Code bezeichnen. Der Wert von :code:`mu` wird bei jeder Schleifeniteration aktualisiert. Er ist in Abtastwert-Einheiten angegeben, und du kannst ihn als den Betrag betrachten, um den wir verschieben müssen, um zum "perfekten" Zeitpunkt abtasten zu können. Wenn also :code:`mu = 3.61` ist, bedeutet das, dass wir den Eingang um 3,61 Abtastwerte verschieben müssen, um am richtigen Punkt zu tasten. Da wir 8 Abtastwerte pro Symbol haben, wird :code:`mu`, wenn es über 8 geht, einfach wieder auf null zurückgesetzt.

Der folgende Python-Code implementiert die Mueller-und-Müller-Taktwiederherstellungstechnik.

.. code-block:: python

    mu = 0 # Anfangsschätzung der Phase des Abtastwerts
    out = np.zeros(len(samples) + 10, dtype=np.complex64)
    out_rail = np.zeros(len(samples) + 10, dtype=np.complex64) # speichert Werte; jede Iteration benötigt die vorherigen 2 Werte plus den aktuellen
    i_in = 0 # Eingangs-Abtastwert-Index
    i_out = 2 # Ausgangsindex (erste zwei Ausgaben sind 0)
    while i_out < len(samples) and i_in+16 < len(samples):
        out[i_out] = samples[i_in] # den vermeintlich "besten" Abtastwert nehmen
        out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j*int(np.imag(out[i_out]) > 0)
        x = (out_rail[i_out] - out_rail[i_out-2]) * np.conj(out[i_out-1])
        y = (out[i_out] - out[i_out-2]) * np.conj(out_rail[i_out-1])
        mm_val = np.real(y - x)
        mu += sps + 0.3*mm_val
        i_in += int(np.floor(mu)) # auf nächste ganze Zahl abrunden, da als Index verwendet
        mu = mu - np.floor(mu) # ganzzahligen Teil von mu entfernen
        i_out += 1 # Ausgangsindex erhöhen
    out = out[2:i_out] # erste zwei entfernen und alles nach i_out (das nie befüllt wurde)
    samples = out # diese Zeile nur einschließen, wenn du diesen Code-Ausschnitt mit dem Costas-Regelkreis verbinden möchtest

Der Timing-Wiederherstellungsblock erhält "empfangene" Abtastwerte und produziert einen Ausgangs-Abtastwert nach dem anderen (beachte, dass :code:`i_out` bei jeder Schleifeniteration um 1 erhöht wird). Der Wiederherstellungsblock nimmt die "empfangenen" Abtastwerte nicht einfach nacheinander, weil die Schleife :code:`i_in` anpasst. Sie überspringt einige Abtastwerte in dem Versuch, den "korrekten" Abtastwert zu ziehen, der der am Peak des Impulses wäre. Während die Schleife Abtastwerte verarbeitet, synchronisiert sie sich langsam auf das Symbol, oder versucht es zumindest, indem sie :code:`mu` anpasst. Aufgrund der Struktur des Codes wird der ganzzahlige Teil von :code:`mu` zu :code:`i_in` addiert und dann von :code:`mu` entfernt (beachte, dass :code:`mm_val` in jeder Schleife negativ oder positiv sein kann). Sobald sie vollständig synchronisiert ist, sollte die Schleife nur den mittleren Abtastwert aus jedem Symbol/Impuls ziehen. Du kannst die Konstante 0,3 anpassen, was beeinflusst, wie schnell der Feedback-Regelkreis reagiert; ein höherer Wert lässt ihn schneller reagieren, erhöht aber das Stabilitätsrisiko.

Der nächste Plot zeigt eine Beispielausgabe, bei der wir die fraktionale Zeitverzögerung sowie den Frequenzoffset *deaktiviert* haben. Wir zeigen nur I, da Q mit deaktiviertem Frequenzoffset lauter Nullen ist. Die drei Plots sind übereinander gestapelt, um zu zeigen, wie die Bits vertikal ausgerichtet sind.

**Oberer Plot**
    Ursprüngliche BPSK-Symbole, d.h. 1er und -1er. Zur Erinnerung: Es gibt Nullen dazwischen, weil wir 8 Abtastwerte pro Symbol möchten.
**Mittlerer Plot**
    Abtastwerte nach der Impulsformung, aber vor dem Synchronisierer.
**Unterer Plot**
    Ausgabe des Symbolsynchronisierers, der nur 1 Abtastwert pro Symbol liefert. Diese Abtastwerte können direkt in einen Demodulator eingespeist werden, der für BPSK prüft, ob der Wert größer oder kleiner als 0 ist.

.. image:: ../_images/time-sync-output.svg
   :align: center
   :target: ../_images/time-sync-output.svg

Konzentrieren wir uns auf den unteren Plot, der die Ausgabe des Synchronisierers ist. Es dauerte fast 30 Symbole, bis die Synchronisation auf die richtige Verzögerung eingerastet ist. Da es unvermeidlich Zeit braucht, bis Synchronisierer einrasten, verwenden viele Kommunikationsprotokolle eine Präambel, die eine Synchronisierungssequenz enthält: Sie dient als Ankündigung, dass ein neues Paket angekommen ist, und gibt dem Empfänger Zeit, sich darauf zu synchronisieren. Aber nach diesen ~30 Abtastwerten funktioniert der Synchronisierer perfekt. Wir haben perfekte 1er und -1er, die mit den Eingangsdaten übereinstimmen. Es hilft, dass diesem Beispiel kein Rauschen hinzugefügt wurde. Füge gerne Rauschen oder Zeitverschiebungen hinzu und beobachte, wie sich der Synchronisierer verhält. Wenn wir QPSK verwendet hätten, würden wir mit komplexen Zahlen arbeiten, aber der Ansatz wäre derselbe.

****************************************
Zeitsynchronisation mit Interpolation
****************************************

Symbolsynchronisierer tendieren dazu, die Eingangs-Abtastwerte um einen bestimmten Faktor zu interpolieren, z.B. 16, sodass sie um einen *Bruchteil* eines Abtastwerts verschieben können. Die zufällige Verzögerung durch den drahtlosen Kanal wird wahrscheinlich kein genaues Vielfaches eines Abtastwerts sein, sodass der Peak des Symbols möglicherweise nicht auf einem Abtastwert liegt. Dies gilt insbesondere, wenn es nur 2 oder 4 Abtastwerte pro Symbol geben könnte. Durch die Interpolation der Abtastwerte können wir "zwischen" tatsächlichen Abtastwerten abtasten, um den genauen Peak jedes Symbols zu treffen. Die Ausgabe des Synchronisierers ist immer noch nur 1 Abtastwert pro Symbol. Die Eingangs-Abtastwerte selbst werden interpoliert.

Unser oben implementierter Zeitssynchronisations-Python-Code enthielt keine Interpolation. Um unseren Code zu erweitern, aktiviere die fraktionale Zeitverzögerung, die wir am Anfang dieses Kapitels implementiert haben, damit unser empfangenes Signal eine realistischere Verzögerung aufweist. Lasse den Frequenzoffset zunächst deaktiviert. Wenn du die Simulation erneut ausführst, wirst du feststellen, dass der Synchronisierer nicht vollständig auf das Signal synchronisiert. Das liegt daran, dass wir nicht interpolieren und der Code keine Möglichkeit hat, "zwischen Abtastwerten zu abtasten", um die fraktionale Verzögerung auszugleichen. Fügen wir die Interpolation hinzu.

Eine schnelle Möglichkeit, ein Signal in Python zu interpolieren, ist die Verwendung von SciPy's :code:`signal.resample` oder :code:`signal.resample_poly`. Diese Funktionen tun dasselbe, arbeiten aber intern unterschiedlich. Wir verwenden die letztere Funktion, da sie tendenziell schneller ist. Interpolieren wir um den Faktor 16 (beliebig gewählt, andere Werte können ausprobiert werden), d.h. wir fügen 15 zusätzliche Abtastwerte zwischen jeden Abtastwert ein. Es kann in einer Codezeile erledigt werden, und es sollte *vor* der Zeitsynchronisation passieren (vor dem großen Code-Ausschnitt oben). Plotten wir auch den Vorher- und Nachher-Zustand, um den Unterschied zu sehen:

.. code-block:: python

 samples_interpolated = signal.resample_poly(samples, 16, 1)

 # Alt vs. neu plotten
 plt.figure('before interp')
 plt.plot(samples,'.-')
 plt.figure('after interp')
 plt.plot(samples_interpolated,'.-')
 plt.show()

Wenn wir *sehr* weit hineinzoomen, sehen wir, dass es dasselbe Signal ist, nur mit 16x mehr Punkten:

.. image:: ../_images/time-sync-interpolated-samples.svg
   :align: center
   :target: ../_images/time-sync-interpolated-samples.svg
   :alt: Beispiel für die Interpolation eines Signals in Python

Hoffentlich wird deutlich, warum wir innerhalb des Timing-Sync-Blocks interpolieren müssen. Diese zusätzlichen Abtastwerte ermöglichen die Berücksichtigung einer Bruchteilverzögerung. Zusätzlich zur Berechnung von :code:`samples_interpolated` müssen wir auch eine Codezeile in unserem Zeitssynchronisierer ändern. Wir ändern die erste Zeile innerhalb der while-Schleife zu:

.. code-block:: python

 out[i_out] = samples_interpolated[i_in*16 + int(mu*16)]

Wir haben hier ein paar Dinge geändert. Erstens können wir :code:`i_in` nicht mehr direkt als Eingangs-Abtastwert-Index verwenden. Wir müssen ihn mit 16 multiplizieren, weil wir unsere Eingangs-Abtastwerte um 16 interpoliert haben. Zur Erinnerung: Der Feedback-Regelkreis passt die Variable :code:`mu` an. Sie repräsentiert die Verzögerung, die dazu führt, dass wir im richtigen Moment abtasten. Erinnere dich auch daran, dass wir nach der Berechnung des neuen Werts von :code:`mu` den ganzzahligen Teil zu :code:`i_in` addiert haben. Jetzt verwenden wir den Restteil, der ein Float von 0 bis 1 ist und den Bruchteil eines Abtastwerts repräsentiert, um den wir verzögern müssen. Zuvor konnten wir nicht um einen Bruchteil eines Abtastwerts verzögern, jetzt schon – zumindest in Schritten von 1/16 eines Abtastwerts. Was wir tun, ist :code:`mu` mit 16 zu multiplizieren, um herauszufinden, um wie viele Abtastwerte unseres interpolierten Signals wir verzögern müssen. Und dann runden wir diese Zahl, da der Wert in den Klammern letztendlich ein Index ist und eine ganze Zahl sein muss.

Die tatsächliche Plot-Ausgabe dieses neuen Codes sollte in etwa gleich aussehen wie zuvor. Wir haben unsere Simulation nur realistischer gemacht, indem wir eine Bruchteilverzögerung hinzugefügt haben, und dann haben wir den Interpolator zum Synchronisierer hinzugefügt, um diese Bruchteilverzögerung auszugleichen.

Spielé gerne mit verschiedenen Interpolationsfaktoren herum, d.h. ändere alle 16er auf einen anderen Wert. Du kannst auch den Frequenzoffset aktivieren oder dem Signal weißes Gaußsches Rauschen vor dem Empfang hinzufügen, um zu sehen, wie sich das auf die Synchronisationsleistung auswirkt (Hinweis: Du musst möglicherweise den Multiplikator 0,3 anpassen).

Wenn wir nur den Frequenzoffset mit einer Frequenz von 1 kHz aktivieren, ergibt sich folgende Zeitsync-Performance. Wir müssen jetzt sowohl I als auch Q zeigen, da wir einen Frequenzoffset hinzugefügt haben:

.. image:: ../_images/time-sync-output2.svg
   :align: center
   :target: ../_images/time-sync-output2.svg
   :alt: Ein Python-simuliertes Signal mit einem leichten Frequenzoffset

Es mag schwer zu erkennen sein, aber die Zeitsynchronisation funktioniert noch einwandfrei. Es dauert etwa 20 bis 30 Symbole, bis sie eingerastet ist. Es gibt jedoch ein Sinusmuster, da wir noch einen Frequenzoffset haben, und wir werden im nächsten Abschnitt lernen, wie wir damit umgehen.

Unten ist der IQ-Plot (a.k.a. Konstellationsplot) des Signals vor und nach der Synchronisation. Du kannst Abtastwerte auf einem IQ-Plot mit einem Streudiagramm darstellen: :code:`plt.plot(np.real(samples), np.imag(samples), '.')`. In der Animation unten haben wir die ersten 30 Symbole ausgelassen. Sie traten auf, bevor die Zeitsynchronisation abgeschlossen war. Die verbleibenden Symbole liegen alle grob auf dem Einheitskreis, aufgrund des Frequenzoffsets.

.. image:: ../_images/time-sync-constellation.svg
   :align: center
   :target: ../_images/time-sync-constellation.svg
   :alt: Ein IQ-Plot eines Signals vor und nach der Zeitsynchronisation

Um noch mehr Einblick zu gewinnen, können wir die Konstellation über die Zeit betrachten, um zu erkennen, was tatsächlich mit den Symbolen passiert. Ganz am Anfang, für kurze Zeit, sind die Symbole nicht 0 oder auf dem Einheitskreis. Das ist der Zeitraum, in dem die Zeitsynchronisation die richtige Verzögerung findet. Es geht sehr schnell, schau genau hin! Das Drehen ist nur der Frequenzoffset. Frequenz ist eine konstante Phasenänderung, sodass ein Frequenzoffset eine Rotation des BPSK verursacht (was im statischen/persistenten Plot oben einen Kreis erzeugt).

.. image:: ../_images/time-sync-constellation-animated.gif
   :align: center
   :target: ../_images/time-sync-constellation-animated.gif
   :alt: Animation eines IQ-Plots von BPSK mit einem Frequenzoffset, der rotierende Cluster zeigt

Hoffentlich hast du durch das Sehen eines Beispiels der tatsächlich stattfindenden Zeitsynchronisation ein Gefühl dafür, was sie tut und eine allgemeine Vorstellung, wie sie funktioniert. In der Praxis würde die von uns erstellte while-Schleife nur auf einer kleinen Anzahl von Abtastwerten gleichzeitig arbeiten (z.B. 1000). Du musst den Wert von :code:`mu` zwischen den Aufrufen der Sync-Funktion speichern, sowie die letzten paar Werte von :code:`out` und :code:`out_rail`.

Als nächstes untersuchen wir die Frequenzsynchronisation, die wir in grobe und feine Frequenzsynchronisation aufteilen. Die grobe kommt üblicherweise vor der Zeitsynchronisation, die feine danach.



**********************************
Grobe Frequenzsynchronisation
**********************************

Auch wenn wir dem Sender und Empfänger sagen, auf derselben Mittenfrequenz zu arbeiten, wird es aufgrund von Hardware-Unvollkommenheiten (z.B. des Oszillators) oder eines Doppler-Shifts durch Bewegung einen leichten Frequenzoffset geben. Dieser Frequenzoffset wird im Verhältnis zur Trägerfrequenz winzig sein, aber selbst ein kleiner Offset kann ein digitales Signal durcheinanderbringen. Der Offset wird sich wahrscheinlich über die Zeit ändern, was einen ständig laufenden Feedback-Regelkreis erfordert, um den Offset zu korrigieren. Als Beispiel hat der Oszillator im Pluto eine maximale Offset-Spezifikation von 25 PPM. Das sind 25 Teile pro Million relativ zur Mittenfrequenz. Wenn du auf 2,4 GHz abgestimmt bist, wäre das ein maximaler Offset von +/- 60 kHz. Die Abtastwerte, die unser SDR liefert, liegen im Basisband, was dazu führt, dass sich jeder Frequenzoffset in diesem Basisbandsignal manifestiert. Ein BPSK-Signal mit einem kleinen Trägeroffset sieht ungefähr wie der unten stehende Zeitplot aus, was für die Demodulation von Bits offensichtlich nicht ideal ist. Wir müssen alle Frequenzoffsets vor der Demodulation entfernen.

.. image:: ../_images/carrier-offset.png
   :scale: 60 %
   :align: center

Die Frequenzsynchronisation ist üblicherweise in grobe und feine Synchronisation unterteilt, wobei die grobe große Offsets in der Größenordnung von kHz oder mehr korrigiert, während die feine das verbleibende korrigiert. Die grobe findet vor der Zeitsynchronisation statt, die feine danach.

Mathematisch gilt: Wenn wir ein Basisbandsignal :math:`s(t)` haben und es einen Frequenz-(a.k.a. Träger-)Offset von :math:`f_o` Hz erfährt, können wir das Empfangene darstellen als:

.. math::

 r(t) = s(t) e^{j2\pi f_o t} + n(t)

wobei :math:`n(t)` das Rauschen ist.

Der erste Trick, den wir kennenlernen, um eine grobe Frequenzoffset-Schätzung durchzuführen (wenn wir die Offsetfrequenz schätzen können, können wir sie rückgängig machen), ist, unser Signal zu quadrieren. Ignorieren wir zunächst das Rauschen, um die Mathematik einfacher zu halten:

.. math::

 r^2(t) = s^2(t) e^{j4\pi f_o t}

Sehen wir uns an, was passiert, wenn wir unser Signal :math:`s(t)` quadrieren, indem wir betrachten, was QPSK täte. Das Quadrieren komplexer Zahlen führt zu interessantem Verhalten, besonders wenn wir über Konstellationen wie BPSK und QPSK sprechen. Die folgende Animation zeigt, was passiert, wenn du QPSK quadrierst und dann noch einmal quadrierst. Ich habe speziell QPSK anstelle von BPSK verwendet, weil du sehen kannst, dass du beim einmaligen Quadrieren von QPSK im Wesentlichen BPSK erhältst. Und nach einem weiteren Quadrieren wird es ein einzelner Cluster. (Danke an http://ventrella.com/ComplexSquaring/ für diese nette Web-App.)

.. image:: ../_images/squaring-qpsk.gif
   :scale: 80 %
   :align: center

Sehen wir uns an, was passiert, wenn unser QPSK-Signal eine kleine Phasenrotation und Amplitudenskalierung erfährt, was realistischer ist:

.. image:: ../_images/squaring-qpsk2.gif
   :scale: 80 %
   :align: center

Es wird immer noch ein Cluster, nur mit einer Phasenverschiebung. Die wichtigste Erkenntnis hier ist, dass wenn du QPSK zweimal (und BPSK einmal) quadrierst, alle vier Cluster von Punkten zu einem Cluster zusammengeführt werden. Warum ist das nützlich? Nun, durch das Zusammenführen der Cluster entfernen wir im Wesentlichen die Modulation! Wenn alle Punkte jetzt im selben Cluster sind, ist das wie eine Reihe von Konstanten. Es ist, als ob keine Modulation mehr vorhanden wäre, und das Einzige, was übrig bleibt, ist die Sinuswelle, die durch den Frequenzoffset verursacht wird (wir haben auch Rauschen, aber lass uns das vorerst weiterhin ignorieren). Es stellt sich heraus, dass du das Signal N-mal quadrieren musst, wobei N die Ordnung des verwendeten Modulationsverfahrens ist. Das bedeutet, dass dieser Trick nur funktioniert, wenn du das Modulationsverfahren im Voraus kennst. Die Gleichung lautet eigentlich:

.. math::

 r^N(t) = s^N(t) e^{j2N\pi f_o t}

Für unseren BPSK-Fall mit Modulationsordnung 2 verwenden wir folgende Gleichung für unsere grobe Frequenzsynchronisation:

.. math::

 r^2(t) = s^2(t) e^{j4\pi f_o t}

Wir haben entdeckt, was mit dem :math:`s(t)`-Teil der Gleichung passiert, aber was ist mit dem Sinusoid-Teil (a.k.a. komplexe Exponentialfunktion)? Wie wir sehen können, fügt er den :math:`N`-Term hinzu, was ihn einem Sinusoid bei einer Frequenz von :math:`Nf_o` statt nur :math:`f_o` entspricht. Eine einfache Methode, um :math:`f_o` herauszufinden, ist, die FFT des Signals nach N-maligem Quadrieren zu nehmen und zu sehen, wo die Spitze auftritt. Simulieren wir es in Python. Wir kehren zur Generierung unseres BPSK-Signals zurück, und anstatt eine Bruchteilverzögerung darauf anzuwenden, wenden wir einen Frequenzoffset an, indem wir das Signal mit :math:`e^{j2\pi f_o t}` multiplizieren, genau wie wir es im Kapitel :ref:`filters-chapter` getan haben, um einen Tiefpassfilter in einen Hochpassfilter umzuwandeln.

Verwende den Code vom Anfang dieses Kapitels und wende einen +13-kHz-Frequenzoffset auf dein digitales Signal an. Dies kann direkt vor oder nach dem Hinzufügen der Bruchteilverzögerung passieren; es spielt keine Rolle, welche Reihenfolge. Es muss jedoch *nach* der Impulsformung, aber vor empfangsseitigen Funktionen wie Zeitsync erfolgen.

Da wir nun ein Signal mit einem 13-kHz-Frequenzoffset haben, plotten wir die FFT vor und nach dem Quadrieren, um zu sehen, was passiert. Du solltest mittlerweile wissen, wie man eine FFT macht, einschließlich der abs()- und fftshift()-Operation. Für diese Übung spielt es keine Rolle, ob du den Logarithmus nimmst oder ob du nach dem abs() quadrierst.

Zunächst das Signal vor dem Quadrieren (einfache FFT):

.. code-block:: python

    psd = np.fft.fftshift(np.abs(np.fft.fft(samples)))
    f = np.linspace(-fs/2.0, fs/2.0, len(psd))
    plt.plot(f, psd)
    plt.show()

.. image:: ../_images/coarse-freq-sync-before.svg
   :align: center
   :target: ../_images/coarse-freq-sync-before.svg

Wir sehen keine Spitze, die mit dem Trägeroffset zusammenhängt. Sie wird von unserem Signal verdeckt.

Jetzt mit hinzugefügtem Quadrieren (nur Potenz 2, da es BPSK ist):

.. code-block:: python

    # Vor der FFT-Zeile hinzufügen
    samples = samples**2

Wir müssen sehr weit hineinzoomen, um zu sehen, bei welcher Frequenz die Spitze liegt:

.. image:: ../_images/coarse-freq-sync.svg
   :align: center
   :target: ../_images/coarse-freq-sync.svg

Du kannst versuchen, die Anzahl der simulierten Symbole zu erhöhen (z.B. 1000 Symbole), damit wir genug Abtastwerte haben. Je mehr Abtastwerte in unsere FFT eingehen, desto genauer wird unsere Schätzung des Frequenzoffsets sein. Zur Erinnerung: Der obige Code sollte *vor* dem Timing-Synchronisierer kommen.

Die Offset-Frequenzspitze erscheint bei :math:`Nf_o`. Wir müssen diesen Bin (26,6 kHz) durch 2 teilen, um unsere endgültige Antwort zu finden, die sehr nahe an den 13 kHz Frequenzoffset kommt, den wir am Anfang des Kapitels angewendet haben! Wenn du mit dieser Zahl gespielt hast und es nicht mehr 13 kHz sind, ist das in Ordnung. Stelle nur sicher, dass du weißt, was du eingestellt hast.

Da unsere Abtastrate 1 MHz beträgt, betragen die maximalen Frequenzen, die wir sehen können, -500 kHz bis 500 kHz. Wenn wir unser Signal auf die Potenz N bringen, können wir Frequenzoffsets nur bis zu :math:`500e3/N` "sehen", oder im Fall von BPSK +/- 250 kHz. Wenn wir ein QPSK-Signal empfingen, wären es nur +/- 125 kHz, und Trägeroffsets über oder unter diesem Bereich wären mit dieser Technik außerhalb unseres Bereichs. Um ein Gefühl für den Doppler-Shift zu bekommen: Wenn du im 2,4-GHz-Band sendest und entweder Sender oder Empfänger mit 60 mph fährt (die Relativgeschwindigkeit ist entscheidend), würde es eine Frequenzverschiebung von 214 Hz verursachen. Der Offset durch einen minderwertigen Oszillator wird in dieser Situation wahrscheinlich der Hauptverursacher sein.

Die eigentliche Korrektur dieses Frequenzoffsets erfolgt genau so, wie wir den Offset ursprünglich simuliert haben: Multiplikation mit einer komplexen Exponentialfunktion, diesmal jedoch mit negativem Vorzeichen, da wir den Offset entfernen möchten.

.. code-block:: python

    max_freq = f[np.argmax(psd)]
    Ts = 1/fs # Abtastperiode berechnen
    t = np.arange(0, Ts*len(samples), Ts) # Zeitvektor erstellen
    samples = samples * np.exp(-1j*2*np.pi*max_freq*t/2.0)

Es liegt an dir, ob du ihn korrigieren oder den anfänglichen Frequenzoffset auf eine kleinere Zahl (wie 500 Hz) ändern möchtest, um die feine Frequenzsynchronisation zu testen, die wir jetzt lernen werden.

**********************************
Feine Frequenzsynchronisation
**********************************

Als nächstes wechseln wir zur feinen Frequenzsynchronisation. Der vorherige Trick ist eher für die grobe Synchronisation, und es ist kein geschlossener Regelkreis (Feedback-Typ). Für die feine Frequenzsynchronisation möchten wir einen Feedback-Regelkreis, durch den wir Abtastwerte streamen, was wiederum eine Form von PLL sein wird. Unser Ziel ist es, den Frequenzoffset auf null zu bringen und dort zu halten, auch wenn sich der Offset über die Zeit ändert. Wir müssen den Offset kontinuierlich verfolgen. Feine Frequenzsynchronisationstechniken funktionieren am besten mit einem Signal, das bereits zeitlich auf Symbolebene synchronisiert wurde. Der Code, den wir in diesem Abschnitt besprechen, kommt daher *nach* der Timing-Synchronisation.

Wir verwenden eine Technik namens Costas-Regelkreis (Costas Loop). Es ist eine Form von PLL, die speziell für die Trägerfrequenzoffset-Korrektur für digitale Signale wie BPSK und QPSK entwickelt wurde. Sie wurde von John P. Costas bei General Electric in den 1950er Jahren erfunden und hatte einen großen Einfluss auf die moderne digitale Kommunikation. Der Costas-Regelkreis beseitigt den Frequenzoffset und korrigiert auch jeden Phasenoffset. Die Energie wird auf die I-Achse ausgerichtet. Frequenz ist lediglich eine Phasenänderung, sodass sie als eins verfolgt werden können. Der Costas-Regelkreis wird mit folgendem Diagramm zusammengefasst (beachte, dass die 1/2 in den Gleichungen weggelassen wurden, da sie funktional keine Rolle spielen).

.. image:: ../_images/costas-loop.svg
   :align: center
   :target: ../_images/costas-loop.svg
   :alt: Costas-Regelkreis-Diagramm mit mathematischen Ausdrücken, eine Form von PLL für die RF-Signalverarbeitung

Der spannungsgesteuerte Oszillator (VCO) ist einfach ein Sin/Cos-Wellengenerator, der eine Frequenz basierend auf der Eingabe verwendet. In unserem Fall, da wir einen drahtlosen Kanal simulieren, ist es keine Spannung, sondern ein durch eine Variable dargestellter Pegel. Er bestimmt die Frequenz und Phase der generierten Sinus- und Kosinuswellen. Was er tut, ist das empfangene Signal mit einem intern generierten Sinusoid zu multiplizieren, in dem Versuch, den Frequenz- und Phasenoffset rückgängig zu machen. Dieses Verhalten ähnelt dem, wie ein SDR heruntermischt und die I- und Q-Zweige erstellt.


Unten ist der Python-Code für unseren Costas-Regelkreis:

.. code-block:: python

    N = len(samples)
    phase = 0
    freq = 0
    # Diese zwei Parameter sind anzupassen, um den Feedback-Regelkreis schneller oder langsamer zu machen (was die Stabilität beeinflusst)
    alpha = 0.132
    beta = 0.00932
    out = np.zeros(N, dtype=np.complex64)
    freq_log = []
    for i in range(N):
        out[i] = samples[i] * np.exp(-1j*phase) # Eingabe-Abtastwert um das Inverse des geschätzten Phasenoffsets anpassen
        error = np.real(out[i]) * np.imag(out[i]) # Fehlerformel für Costas-Regelkreis 2. Ordnung (z.B. für BPSK)

        # Regelkreis fortschreiben (Phase und Frequenzoffset neu berechnen)
        freq += (beta * error)
        freq_log.append(freq * fs / (2*np.pi)) # von Winkelgeschwindigkeit in Hz umrechnen zum Protokollieren
        phase += freq + (alpha * error)

        # Optional: Phase so anpassen, dass sie immer zwischen 0 und 2pi liegt; Phase läuft alle 2pi durch
        while phase >= 2*np.pi:
            phase -= 2*np.pi
        while phase < 0:
            phase += 2*np.pi

    # Frequenz über Zeit plotten, um zu sehen wie lange es dauert, den richtigen Offset zu finden
    plt.plot(freq_log,'.-')
    plt.show()

Hier ist viel los, also gehen wir es durch. Einige Zeilen sind einfach, andere sehr komplex. :code:`samples` ist unser Eingang und :code:`out` sind die Ausgangs-Abtastwerte. :code:`phase` und :code:`frequency` sind wie das :code:`mu` aus dem Zeitsync-Code. Sie enthalten die aktuellen Offset-Schätzungen, und bei jeder Schleifeniteration erstellen wir die Ausgangs-Abtastwerte, indem wir die Eingangs-Abtastwerte mit :code:`np.exp(-1j*phase)` multiplizieren. Die Variable :code:`error` enthält die "Fehler"-Metrik, und für einen Costas-Regelkreis 2. Ordnung ist es eine sehr einfache Gleichung. Wir multiplizieren den reellen Teil des Abtastwerts (I) mit dem imaginären Teil (Q), und da Q für BPSK gleich null sein sollte, wird die Fehlerfunktion minimiert, wenn kein Phasen- oder Frequenzoffset Energie von I nach Q verschiebt. Für einen Costas-Regelkreis 4. Ordnung ist es etwas komplexer, aber nicht viel länger, da sowohl I als auch Q Energie haben, selbst wenn kein Phasen- oder Frequenzoffset für QPSK vorhanden ist. Wenn du neugierig bist, wie es aussieht, klicke unten, aber wir verwenden es in unserem Code vorerst nicht. Der Grund, warum es für QPSK funktioniert, ist, dass wenn du den Absolutwert von I und Q nimmst, du +1+1j erhältst, und wenn kein Phasen- oder Frequenzoffset vorhanden ist, sollte die Differenz zwischen den Absolutwerten von I und Q nahe null sein.

.. raw:: html

   <details>
   <summary>Costas-Regelkreis 4. Ordnung Fehlergleichung (für Neugierige)</summary>

.. code-block:: python

    # Für QPSK
    def phase_detector_4(sample):
        if sample.real > 0:
            a = 1.0
        else:
            a = -1.0
        if sample.imag > 0:
            b = 1.0
        else:
            b = -1.0
        return a * sample.imag - b * sample.real




.. raw:: html

   </details>

Die Variablen :code:`alpha` und :code:`beta` bestimmen, wie schnell Phase bzw. Frequenz aktualisiert werden. Es gibt eine Theorie dahinter, warum ich diese zwei Werte gewählt habe; wir werden sie hier jedoch nicht besprechen. Wenn du neugierig bist, kannst du versuchen, :code:`alpha` und/oder :code:`beta` zu optimieren, um zu sehen, was passiert.

Wir protokollieren den Wert von :code:`freq` bei jeder Iteration, um ihn am Ende zu plotten und zu sehen, wie der Costas-Regelkreis auf den richtigen Frequenzoffset konvergiert. Wir müssen :code:`freq` mit der Abtastrate multiplizieren und von der Winkelfrequenz in Hz umrechnen, indem wir durch :math:`2\pi` dividieren. Beachte: Wenn du vor dem Costas-Regelkreis eine Zeitsynchronisation durchgeführt hast, musst du auch durch deinen :code:`sps`-Wert (z.B. 8) dividieren, da die Abtastwerte aus der Zeitsynchronisation mit einer Rate ausgegeben werden, die gleich deiner ursprünglichen Abtastrate geteilt durch :code:`sps` ist.

Abschließend addieren oder subtrahieren wir nach der Neuberechnung der Phase genug :math:`2 \pi`, um die Phase zwischen 0 und :math:`2 \pi` zu halten, was die Phase umläuft.

Unser Signal vor und nach dem Costas-Regelkreis sieht so aus:

.. image:: ../_images/costas-loop-output.svg
   :align: center
   :target: ../_images/costas-loop-output.svg
   :alt: Python-Simulation eines Signals vor und nach dem Costas-Regelkreis

Und die Frequenzoffset-Schätzung über die Zeit, die auf den richtigen Offset konvergiert (in diesem Beispielsignal wurde ein -300-Hz-Offset verwendet):

.. image:: ../_images/costas-loop-freq-tracking.svg
   :align: center
   :target: ../_images/costas-loop-freq-tracking.svg

Es dauert fast 70 Abtastwerte, bis der Algorithmus vollständig auf den Frequenzoffset eingerastet ist. Du kannst sehen, dass in meinem simulierten Beispiel nach der groben Frequenzsynchronisation noch etwa -300 Hz übrig waren. Deines kann variieren. Wie ich bereits erwähnt habe, kannst du die grobe Frequenzsynchronisation deaktivieren und den anfänglichen Frequenzoffset auf einen beliebigen Wert setzen und sehen, ob der Costas-Regelkreis es herausfindet.

Der Costas-Regelkreis hat neben der Beseitigung des Frequenzoffsets unser BPSK-Signal auf den I-Anteil ausgerichtet, sodass Q wieder null ist. Es ist ein nützlicher Nebeneffekt des Costas-Regelkreises, und er lässt den Costas-Regelkreis im Wesentlichen als unseren Demodulator fungieren. Jetzt müssen wir nur noch I nehmen und prüfen, ob es größer oder kleiner als null ist. Wir werden nicht wissen, wie negativ und positiv zu 0 und 1 werden, da möglicherweise eine Invertierung vorhanden ist oder nicht; der Costas-Regelkreis (oder unsere Zeitsynchronisation) hat keine Möglichkeit, das zu wissen. Hier kommt die differentielle Codierung ins Spiel. Sie beseitigt die Mehrdeutigkeit, da 1er und 0er darauf basieren, ob sich das Symbol geändert hat, nicht ob es +1 oder -1 war. Wenn wir differentielle Codierung hinzufügten, würden wir immer noch BPSK verwenden. Wir würden einen Differenzcodierungsblock direkt vor der Modulation auf der Tx-Seite und direkt nach der Demodulation auf der Rx-Seite hinzufügen.

Unten ist eine Animation der gleichzeitig laufenden Zeit- und Frequenzsynchronisation. Die Zeitsynchronisation erfolgt fast sofort, aber die Frequenzsynchronisation dauert fast die gesamte Animation, bis sie sich vollständig eingestellt hat. Das lag daran, dass :code:`alpha` und :code:`beta` mit 0,005 bzw. 0,001 zu niedrig eingestellt waren. Den Code zur Erzeugung dieser Animation findest du `hier <https://github.com/777arc/PySDR/blob/master/figure-generating-scripts/costas_loop_animation.py>`_.

.. image:: ../_images/costas_animation.gif
   :align: center
   :target: ../_images/costas_animation.gif
   :alt: Costas-Regelkreis-Animation

Der folgende (ausgeklappte) Codeblock enthält das vollständige Python-Beispiel des bisherigen Kapitels. Dieses wurde mit Python 3.12.3 und NumPy 1.26.4 getestet. Es enthält auch eine Bitfehlerprüfung am Ende. AWGN wurde weggelassen, um zu sehen, wie eng das BPSK allein durch Synchronisation werden kann. Du kannst AWGN hinzufügen, z.B. direkt nach dem Hinzufügen der Bruchteilverzögerung. Beachte, dass der Plot von IQ über die Zeit vor der Frequenzsynchronisation liegt, sodass du sehen kannst, wie die BPSK-Energie langsam zwischen I und Q wechselt.

.. raw:: html

   <details>
   <summary>Vollständiges Python-Beispiel</summary>

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from scipy import signal

   # BPSK-Signal erstellen
   num_symbols = 100
   sps = 8
   bits = np.random.randint(0, 2, num_symbols) # Zu übertragende Daten, 1er und 0er
   pulse_train = np.array([])
   for bit in bits:
      pulse = np.zeros(sps)
      pulse[0] = bit*2-1 # ersten Wert auf 1 oder -1 setzen
      pulse_train = np.concatenate((pulse_train, pulse)) # die 8 Abtastwerte zum Signal hinzufügen

   # Impulsformung auf BPSK anwenden
   num_taps = 101
   beta = 0.35
   Ts = sps # Abtastrate als 1 Hz angenommen, Symbolperiode ist 8
   t = np.arange(-51, 52) # letzte Zahl nicht enthalten
   h = np.sinc(t/Ts) * np.cos(np.pi*beta*t/Ts) / (1 - (2*beta*t/Ts)**2)
   samples = np.convolve(pulse_train, h, 'same')

   # Fraktionalen Verzögerungsfilter erstellen und anwenden, um zufälligen Timing-Offset zu emulieren
   delay = 0.456 # fraktionale Verzögerung, in Abtastwerten
   N = 21 # Anzahl der Taps, ungerade halten
   n = np.arange(-(N-1)//2, N//2+1) # -10,-9,...,0,...,9,10
   h = np.sinc(n - delay) # Filtertaps berechnen
   h *= np.hamming(N) # Filter fenstern
   h /= np.sum(h) # für Einheitsverstärkung normalisieren
   samples = np.convolve(samples, h) # Filter anwenden

   # Erheblichen Frequenzoffset anwenden
   fs = 1e6 # Abtastrate als 1 MHz angenommen
   fo = 13000 # Frequenzoffset simulieren – GROBER OFFSET!
   Ts = 1/fs # Abtastperiode berechnen
   t = np.arange(0, Ts*len(samples), Ts) # Zeitvektor erstellen
   samples = samples * np.exp(1j*2*np.pi*fo*t) # Frequenzverschiebung durchführen

   # Groben Frequenzoffset schätzen und korrigieren
   samples_sq = samples**2
   psd = np.fft.fftshift(np.abs(np.fft.fft(samples_sq, 2048)))
   f = np.linspace(-fs/2.0, fs/2.0, len(psd))
   max_freq = f[np.argmax(psd)] / 2.0
   print(f"Geschätzter Frequenzoffset: {max_freq:.2f} Hz")
   Ts = 1/fs # Abtastperiode berechnen
   t = np.arange(0, Ts*len(samples), Ts) # Zeitvektor erstellen
   samples = samples * np.exp(-1j*2*np.pi*max_freq*t)

   # An diesem Punkt sollte weniger als 1 kHz Frequenzoffset im Signal vorhanden sein

   # Symbol-/Timing-Synchronisation
   mu = 0 # Anfangsschätzung der Phase des Abtastwerts
   out = np.zeros(len(samples) // sps + 2, dtype=np.complex64)
   out_rail = np.zeros(len(samples) // sps + 2, dtype=np.complex64)
   i_in = 0 # Eingangs-Abtastwert-Index
   i_out = 2 # Ausgangsindex
   interpolation_factor = 16
   samples_interpolated = signal.resample_poly(samples, interpolation_factor, 1)
   while i_out < len(samples) and i_in+16 < len(samples):
      out[i_out] = samples_interpolated[i_in*interpolation_factor + int(mu*interpolation_factor)]
      out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j*int(np.imag(out[i_out]) > 0)
      x = (out_rail[i_out] - out_rail[i_out-2]) * np.conj(out[i_out-1])
      y = (out[i_out] - out[i_out-2]) * np.conj(out_rail[i_out-1])
      mm_val = np.real(y - x)
      mu += sps + 0.3*mm_val
      i_in += int(np.floor(mu))
      mu = mu - np.floor(mu)
      i_out += 1
   out = out[3:i_out]
   samples = out

   plt.figure(2)
   plt.plot(np.real(samples))
   plt.plot(np.imag(samples))
   plt.xlabel('Abtastwert-Index')
   plt.ylabel('Abtastwert')
   plt.legend(['I', 'Q'])
   plt.grid()

   N = len(samples)
   phase = 0
   freq = 0
   alpha = 0.132
   beta = 0.00932
   out = np.zeros(N, dtype=np.complex64)
   freq_log = []
   for i in range(N):
      out[i] = samples[i] * np.exp(-1j*phase)
      error = np.real(out[i]) * np.imag(out[i])

      freq += (beta * error)
      freq_log.append(freq * fs / (2*np.pi))
      phase += freq + (alpha * error)

      while phase >= 2*np.pi:
         phase -= 2*np.pi
      while phase < 0:
         phase += 2*np.pi

   # Bitfehlerrate berechnen
   rx_bits = (np.real(out) > 0).astype(int)
   num_bit_errors = np.sum(rx_bits != bits[:len(rx_bits)])
   print(f"Anzahl der Bitfehler: {num_bit_errors} von {len(rx_bits)} Bits, BER: {num_bit_errors/len(rx_bits):.4f}")

   # Frequenz über Zeit plotten
   plt.figure(0)
   plt.plot(freq_log,'.-')
   plt.xlabel('Abtastwert-Index')
   plt.ylabel('Frequenzoffset-Schätzung (Hz)')

   # Nach ~80 Abtastwerten synchronisiert – Konstellation der restlichen 20 plotten
   plt.figure(1)
   plt.plot(np.real(out[80:]), np.imag(out[80:]), '.')
   plt.xlabel('I')
   plt.ylabel('Q')
   plt.xlim(-1.5, 1.5)
   plt.ylim(-1.5, 1.5)
   plt.grid()
   plt.show()

.. raw:: html

   </details>

***************************
Rahmensynchronisation
***************************

Wir haben besprochen, wie man Zeit-, Frequenz- und Phasenoffsets in unserem empfangenen Signal korrigiert. Aber die meisten modernen Kommunikationsprotokolle senden nicht einfach mit 100% Tastverhältnis Bits in einem Strom. Stattdessen verwenden sie Pakete/Rahmen. Am Empfänger müssen wir in der Lage sein zu erkennen, wann ein neuer Rahmen beginnt. Üblicherweise enthält der Rahmen-Header (auf der MAC-Schicht) die Anzahl der Bytes im Rahmen. Wir können diese Information nutzen, um die Länge des Rahmens z.B. in Abtastwerten oder Symbolen zu kennen. Dennoch ist die Erkennung des Rahmenbeginns eine völlig separate Aufgabe. Unten ist ein Beispiel einer WiFi-Rahmenstruktur. Beachte, dass das Allererste, was übertragen wird, ein PHY-Schicht-Header ist, und die erste Hälfte dieses Headers ist eine "Präambel". Diese Präambel enthält eine Synchronisierungssequenz, die der Empfänger verwendet, um den Rahmenbeginn zu erkennen, und sie ist eine Sequenz, die dem Empfänger im Voraus bekannt ist.

.. image:: ../_images/wifi-frame.png
   :scale: 60 %
   :align: center

Eine gängige und unkomplizierte Methode zur Erkennung dieser Sequenzen am Empfänger ist die Kreuzkorrelation der empfangenen Abtastwerte mit der bekannten Sequenz. Wenn die Sequenz vorkommt, ähnelt diese Kreuzkorrelation einer Autokorrelation (mit hinzugefügtem Rauschen). Typischerweise werden für Präambeln Sequenzen gewählt, die gute Autokorrelationseigenschaften haben, z.B. erzeugt die Autokorrelation der Sequenz eine einzelne starke Spitze bei 0 und keine anderen Spitzen. Ein Beispiel sind Barker-Codes; in 802.11/WiFi wird eine Barker-Sequenz der Länge 11 für die 1- und 2-Mbit/sec-Raten verwendet:

.. code-block::

    +1 +1 +1 −1 −1 −1 +1 −1 −1 +1 −1

Du kannst sie als 11 BPSK-Symbole betrachten. Wir können die Autokorrelation dieser Sequenz sehr einfach in Python betrachten:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    x = [1,1,1,-1,-1,-1,1,-1,-1,1,-1]
    plt.plot(np.correlate(x,x,'same'),'.-')
    plt.grid()
    plt.show()

.. image:: ../_images/barker-code.svg
   :align: center
   :target: ../_images/barker-code.svg

Du kannst sehen, dass es in der Mitte 11 (Länge der Sequenz) ist und für alle anderen Verzögerungen -1 oder 0. Es eignet sich gut zum Auffinden des Rahmenbeginns, da es im Wesentlichen die Energie von 11 Symbolen integriert, um eine 1-Bit-Spitze in der Ausgabe der Kreuzkorrelation zu erzeugen. Tatsächlich ist der schwierigste Teil der Rahmenbeginnserkennung die Bestimmung eines guten Schwellenwerts. Du möchtest nicht, dass Rahmen, die eigentlich nicht Teil deines Protokolls sind, ausgelöst werden. Das bedeutet, dass du zusätzlich zur Kreuzkorrelation auch eine Art Leistungsnormalisierung durchführen musst, was wir hier nicht berücksichtigen. Bei der Wahl eines Schwellenwerts musst du einen Kompromiss zwischen Erkennungswahrscheinlichkeit und Fehlalarmwahrscheinlichkeit machen. Denke daran, dass der Rahmen-Header selbst Informationen enthält, sodass einige Fehlalarme in Ordnung sind; du wirst schnell feststellen, dass es kein tatsächlicher Rahmen ist, wenn du versuchst, den Header zu decodieren und die CRC zwangsläufig fehlschlägt (weil es kein tatsächlicher Rahmen war). Während einige Fehlalarme in Ordnung sind, ist das Verpassen einer Rahmenerkennung schlecht.

Eine weitere Sequenz mit hervorragenden Autokorrelationseigenschaften sind Zadoff-Chu-Sequenzen, die in LTE verwendet werden. Sie haben den Vorteil, in Mengen zu existieren; du kannst mehrere verschiedene Sequenzen haben, die alle gute Autokorrelationseigenschaften haben, aber sich gegenseitig nicht auslösen (d.h. auch gute Kreuzkorrelationseigenschaften, wenn du verschiedene Sequenzen in der Menge kreuzkorrelierst). Dank dieser Eigenschaft werden verschiedenen Zelltürmen unterschiedliche Sequenzen zugewiesen, sodass ein Mobiltelefon nicht nur den Rahmenbeginn finden kann, sondern auch weiß, von welchem Turm es empfängt.
