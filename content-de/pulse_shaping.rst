.. _pulse-shaping-chapter:

#######################
Impulsformung
#######################

Dieses Kapitel behandelt Impulsformung, Intersymbolinterferenz, Matched Filter und Raised-Cosine-Filter. Am Ende verwenden wir Python, um BPSK-Symbolen Impulsformung hinzuzufügen. Du kannst diesen Abschnitt als Teil II des Filterkapitels betrachten, in dem wir tiefer in die Impulsformung eintauchen.

**********************************
Intersymbolinterferenz (ISI)
**********************************

Im Kapitel :ref:`filters-chapter` haben wir gelernt, dass blockförmige Symbole/Impulse übermäßig viel Spektrum verwenden, und wir können die verwendete Spektrummenge durch das "Formen" unserer Impulse erheblich reduzieren. Du kannst jedoch nicht einfach irgendeinen Tiefpassfilter verwenden, sonst kann Intersymbolinterferenz (ISI) entstehen, bei der Symbole ineinander übergehen und sich gegenseitig stören.

Wenn wir digitale Symbole übertragen, senden wir sie nacheinander (im Gegensatz dazu, eine Zeit dazwischen zu warten). Wenn du einen Impulsformfilter anwendest, verlängert er den Impuls im Zeitbereich (um ihn im Frequenzbereich zu komprimieren), wodurch sich benachbarte Symbole überlappen. Die Überlappung ist in Ordnung, solange dein Impulsformfilter dieses eine Kriterium erfüllt: Alle Impulse müssen an jedem Vielfachen unserer Symbolperiode :math:`T` zu null summieren, außer bei einem der Impulse. Die Idee wird am besten durch folgende Visualisierung verständlich:

.. image:: ../_images/pulse_train.svg
   :align: center
   :target: ../_images/pulse_train.svg
   :alt: Ein Impulszug aus Sinc-Impulsen

Wie du sehen kannst, gibt es in jedem Intervall von :math:`T` genau einen Peak eines Impulses, während die restlichen Impulse bei 0 liegen (sie schneiden die x-Achse). Wenn der Empfänger das Signal abtastet, tut er dies zum richtigen Zeitpunkt (am Peak der Impulse), was bedeutet, dass dies der einzige relevante Zeitpunkt ist. Normalerweise gibt es am Empfänger einen Symbolsynchronisierungsblock, der sicherstellt, dass die Symbole an den Peaks abgetastet werden.

**********************************
Matched Filter
**********************************

Ein Trick, den wir in der drahtlosen Kommunikation verwenden, nennt sich Matched Filtering. Um Matched Filtering zu verstehen, musst du zunächst diese zwei Punkte verstehen:

1. Die oben besprochenen Impulse müssen nur *am Empfänger* vor der Abtastung perfekt ausgerichtet sein. Bis dahin ist es nicht wichtig, ob ISI vorhanden ist, d.h. die Signale können mit ISI durch die Luft fliegen und das ist in Ordnung.

2. Wir möchten einen Tiefpassfilter in unserem Sender einsetzen, um die von unserem Signal verwendete Spektrummenge zu reduzieren. Aber der Empfänger benötigt auch einen Tiefpassfilter, um so viel Rauschen/Interferenz neben dem Signal wie möglich zu eliminieren. Daher haben wir am Sender (Tx) und am Empfänger (Rx) jeweils einen Tiefpassfilter, und die Abtastung erfolgt nach beiden Filtern (und den Auswirkungen des drahtlosen Kanals).

Was wir in der modernen Kommunikation tun, ist den Impulsformfilter gleichmäßig zwischen Tx und Rx aufzuteilen. Sie müssen *nicht* identische Filter sein, aber theoretisch ist der optimale lineare Filter zur Maximierung des SNR bei AWGN, *denselben* Filter bei Tx und Rx zu verwenden. Diese Strategie nennt sich das "Matched Filter"-Konzept.

Eine andere Art, über Matched Filter nachzudenken, ist, dass der Empfänger das empfangene Signal mit dem bekannten Vorlagensignal korreliert. Das Vorlagensignal sind im Wesentlichen die Impulse, die der Sender sendet, unabhängig von den auf sie angewendeten Phasen-/Amplitudenverschiebungen. Zur Erinnerung: Filtern erfolgt durch Faltung, die im Grunde Korrelation ist (tatsächlich sind sie mathematisch identisch, wenn die Vorlage symmetrisch ist). Dieser Prozess der Korrelation des empfangenen Signals mit der Vorlage gibt uns die beste Chance, das Gesendete wiederherzustellen, und deshalb ist es theoretisch optimal. Als Analogie denke an ein Bilderkennungssystem, das mithilfe einer Gesichtsvorlage und einer 2D-Korrelation nach Gesichtern sucht:

.. image:: ../_images/face_template.png
   :scale: 70 %
   :align: center

**********************************
Einen Filter halbieren
**********************************

Wie teilen wir einen Filter tatsächlich in zwei Hälften auf? Faltung ist assoziativ, was bedeutet:

.. math::
 (f * g) * h = f * (g * h)

Stellen wir uns vor, :math:`f` ist unser Eingangssignal, und :math:`g` und :math:`h` sind Filter. :math:`f` mit :math:`g` und dann :math:`h` zu filtern ist dasselbe wie mit einem einzigen Filter :math:`g * h` zu filtern.

Beachte auch, dass Faltung im Zeitbereich Multiplikation im Frequenzbereich entspricht:

.. math::
 g(t) * h(t) \leftrightarrow G(f)H(f)

Um einen Filter zu halbieren, kann man die Quadratwurzel der Frequenzantwort nehmen.

.. math::
 X(f) = X_H(f) X_H(f) \quad \mathrm{wobei} \quad X_H(f) = \sqrt{X(f)}

Unten ist ein vereinfachtes Diagramm einer Sende- und Empfangskette, bei der ein Raised-Cosine-Filter (RC) in zwei Root-Raised-Cosine-Filter (RRC) aufgeteilt wird; der auf der Sendeseite ist der Impulsformfilter, und der auf der Empfangsseite ist das Matched Filter. Zusammen bewirken sie, dass die Impulse am Demodulator so aussehen, als wären sie mit einem einzigen RRC-Filter impulsgeformt worden.

.. image:: ../_images/splitting_rc_filter.svg
   :align: center
   :target: ../_images/splitting_rc_filter.svg
   :alt: Diagramm einer Sende- und Empfangskette mit einem Raised-Cosine-Filter (RC), der in zwei Root-Raised-Cosine-Filter (RRC) aufgeteilt wird

**********************************
Spezifische Impulsformfilter
**********************************

Wir wissen, dass wir Folgendes möchten:

1. Einen Filter entwerfen, der die Bandbreite unseres Signals reduziert (um weniger Spektrum zu verwenden), und alle Impulse außer einem sollten in jedem Symbolintervall zu null summieren.

2. Den Filter halbieren, eine Hälfte in Tx und die andere in Rx platzieren.

Schauen wir uns einige spezifische Filter an, die häufig zur Impulsformung verwendet werden.

Raised-Cosine-Filter
#########################

Der beliebteste Impulsformfilter scheint der "Raised-Cosine"-Filter zu sein. Es ist ein guter Tiefpassfilter zur Begrenzung der Bandbreite unseres Signals, und er hat die Eigenschaft, in Intervallen von :math:`T` zu null zu summieren:

.. image:: ../_images/raised_cosine.svg
   :align: center
   :target: ../_images/raised_cosine.svg
   :alt: Der Raised-Cosine-Filter im Zeitbereich mit verschiedenen Roll-Off-Werten

Beachte, dass obiger Plot im Zeitbereich ist. Er zeigt die Impulsantwort des Filters. Der Parameter :math:`\beta` ist der einzige Parameter des Raised-Cosine-Filters und bestimmt, wie schnell der Filter im Zeitbereich abfällt, was umgekehrt proportional dazu ist, wie schnell er in der Frequenz abfällt:

.. image:: ../_images/raised_cosine_freq.svg
   :align: center
   :target: ../_images/raised_cosine_freq.svg
   :alt: Der Raised-Cosine-Filter im Frequenzbereich mit verschiedenen Roll-Off-Werten

Der Grund, warum er Raised-Cosine-Filter heißt, ist, dass der Frequenzbereich bei :math:`\beta = 1` eine halbe Periode einer Kosinuswelle ist, die auf die x-Achse angehoben wird.

Die Gleichung, die die Impulsantwort des Raised-Cosine-Filters definiert, lautet:

.. math::
 h(t) = \mathrm{sinc}\left( \frac{t}{T} \right) \frac{\cos\left(\frac{\pi\beta t}{T}\right)}{1 - \left( \frac{2 \beta t}{T}   \right)^2}

Weitere Informationen zur :math:`\mathrm{sinc}()`-Funktion findest du `hier <https://en.wikipedia.org/wiki/Sinc_function>`_. Du findest möglicherweise anderswo Gleichungen, die einen zusätzlichen Skalierungsfaktor :math:`\frac{1}{T}` enthalten; dieser bewirkt, dass der Filter eine Einheitsverstärkung hat, sodass das Ausgangssignal dieselbe Leistung wie das Eingangssignal hat (eine gängige Praxis beim Filterentwurf im Allgemeinen). Wir wenden ihn jedoch auf einen Impulszug von Symbolen (z.B. 1er und -1er) an und möchten nicht, dass die Amplitude dieser Symbole nach der Impulsformung geändert wird, daher lassen wir den Skalierungsfaktor weg. Dies wird klarer, sobald wir uns in das Python-Beispiel vertiefen und die Ausgabe plotten.

Denke daran: Wir teilen diesen Filter gleichmäßig zwischen Tx und Rx auf. Das führt uns zum Root-Raised-Cosine-Filter (RRC)!

Root-Raised-Cosine-Filter
#########################

Der Root-Raised-Cosine-Filter (RRC) ist das, was wir tatsächlich in unserem Tx und Rx implementieren. Zusammen bilden sie einen normalen Raised-Cosine-Filter, wie besprochen. Da das Halbieren eines Filters die Quadratwurzel im Frequenzbereich erfordert, wird die Impulsantwort etwas unübersichtlich:

.. image:: ../_images/rrc_filter.png
   :scale: 70 %
   :align: center

Glücklicherweise ist es ein häufig verwendeter Filter, und es gibt viele Implementierungen, darunter `in Python <https://commpy.readthedocs.io/en/latest/generated/commpy.filters.rrcosfilter.html>`_.

Andere Impulsformfilter
###########################

Andere Filter umfassen den Gaußschen Filter, dessen Impulsantwort einer Gaußschen Funktion ähnelt. Es gibt auch einen Sinc-Filter, der dem Raised-Cosine-Filter bei :math:`\beta = 0` entspricht. Der Sinc-Filter ist eher ein idealer Filter, was bedeutet, dass er die notwendigen Frequenzen ohne viel Übergangsbereich eliminiert.

**********************************
Roll-Off-Faktor
**********************************

Untersuchen wir den Parameter :math:`\beta`. Es ist eine Zahl zwischen 0 und 1 und wird als "Roll-Off"-Faktor oder manchmal als "Überschussbandbreite" bezeichnet. Er bestimmt, wie schnell der Filter im Zeitbereich auf null abfällt. Zur Erinnerung: Um als Filter verwendet zu werden, sollte die Impulsantwort auf beiden Seiten auf null abklingen:

.. image:: ../_images/rrc_rolloff.svg
   :align: center
   :target: ../_images/rrc_rolloff.svg
   :alt: Plot des Raised-Cosine Roll-Off-Parameters

Je kleiner :math:`\beta` wird, desto mehr Filtertaps sind erforderlich. Wenn :math:`\beta = 0` ist, erreicht die Impulsantwort nie vollständig null, also versuchen wir, :math:`\beta` so klein wie möglich zu halten, ohne andere Probleme zu verursachen. Je kleiner der Roll-Off, desto kompakter können wir unser Signal für eine gegebene Symbolrate im Frequenzbereich erzeugen, was immer wichtig ist.

Eine häufig verwendete Gleichung zur näherungsweisen Berechnung der Bandbreite in Hz für eine gegebene Symbolrate und einen Roll-Off-Faktor lautet:

.. math::
    \mathrm{BW} = R_S(\beta + 1)

:math:`R_S` ist die Symbolrate in Hz. In der drahtlosen Kommunikation bevorzugen wir üblicherweise einen Roll-Off zwischen 0,2 und 0,5. Als Faustregel gilt: Ein digitales Signal, das die Symbolrate :math:`R_S` verwendet, belegt etwas mehr als :math:`R_S` an Spektrum, einschließlich positiver und negativer Spektrumanteile. Sobald wir unser Signal hochkonvertieren und senden, sind beide Seiten relevant. Wenn wir QPSK mit 1 Million Symbolen pro Sekunde (MSps) senden, belegt es etwa 1,3 MHz. Die Datenrate beträgt 2 Mbps (QPSK verwendet 2 Bits pro Symbol), einschließlich Overhead wie Kanalcodierung und Frame-Header.

**********************************
Python-Übung
**********************************

Als Python-Übung filtern und formen wir einige Impulse. Wir verwenden BPSK-Symbole, damit es einfacher zu visualisieren ist – vor der Impulsformung sendet BPSK 1er oder -1er mit dem "Q"-Anteil gleich null. Mit Q gleich null können wir nur den I-Anteil plotten, was einfacher zu betrachten ist.

In dieser Simulation verwenden wir 8 Abtastwerte pro Symbol, und anstelle eines rechteckwellenähnlichen Signals aus 1ern und -1ern verwenden wir einen Impulszug aus Dirac-Impulsen. Wenn du einen Impuls durch einen Filter schickst, ist die Ausgabe die Impulsantwort (daher der Name). Wenn du also eine Reihe von Impulsen möchtest, verwende Dirac-Impulse mit Nullen dazwischen, um rechteckige Impulse zu vermeiden.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal

    num_symbols = 10
    sps = 8

    bits = np.random.randint(0, 2, num_symbols) # Zu übertragende Daten, 1er und 0er

    x = np.array([])
    for bit in bits:
        pulse = np.zeros(sps)
        pulse[0] = bit*2-1 # ersten Wert auf 1 oder -1 setzen
        x = np.concatenate((x, pulse)) # die 8 Abtastwerte zum Signal hinzufügen
    plt.figure(0)
    plt.plot(x, '.-')
    plt.grid(True)
    plt.show()

.. image:: ../_images/pulse_shaping_python1.png
   :scale: 80 %
   :align: center
   :alt: Ein Impulszug aus Dirac-Impulsen im Zeitbereich, simuliert in Python

Zu diesem Zeitpunkt sind unsere Symbole noch 1er und -1er. Lass dich nicht von der Verwendung der Dirac-Impulse verwirren. Tatsächlich könnte es einfacher sein, die Impulsantwort *nicht* zu visualisieren, sondern es als Array zu betrachten:

.. code-block:: python

 bits: [0, 1, 1, 1, 1, 0, 0, 0, 1, 1]
 BPSK-Symbole: [-1, 1, 1, 1, 1, -1, -1, -1, 1, 1]
 Mit 8 Abtastwerten pro Symbol: [-1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, ...]

Wir erstellen einen Raised-Cosine-Filter mit einem :math:`\beta` von 0,35, und wir machen ihn 101 Taps lang, damit das Signal genug Zeit hat, auf null abzuklingen. Obwohl die Raised-Cosine-Gleichung nach unserer Symbolperiode und einem Zeitvektor :math:`t` fragt, können wir eine **Abtast**periode von 1 Sekunde annehmen, um unsere Simulation zu "normalisieren". Das bedeutet, unsere Symbolperiode :math:`Ts` ist 8, weil wir 8 Abtastwerte pro Symbol haben. Unser Zeitvektor ist dann eine Liste von ganzen Zahlen. Mit der Art, wie die Raised-Cosine-Gleichung funktioniert, möchten wir, dass :math:`t=0` in der Mitte liegt. Wir erzeugen den 101-langen Zeitvektor, der bei -51 beginnt und bei +51 endet.

.. code-block:: python

    # Raised-Cosine-Filter erstellen
    num_taps = 101
    beta = 0.35
    Ts = sps # Abtastrate als 1 Hz angenommen, Abtastperiode ist 1, also Symbolperiode ist 8
    t = np.arange(num_taps) - (num_taps-1)//2
    h = np.sinc(t/Ts) * np.cos(np.pi*beta*t/Ts) / (1 - (2*beta*t/Ts)**2)
    plt.figure(1)
    plt.plot(t, h, '.')
    plt.grid(True)
    plt.show()


.. image:: ../_images/pulse_shaping_python2.png
   :scale: 80 %
   :align: center

Beachte, dass die Ausgabe definitiv auf null abklingt. Die Tatsache, dass wir 8 Abtastwerte pro Symbol verwenden, bestimmt, wie schmal dieser Filter erscheint und wie schnell er auf null abfällt. Die obige Impulsantwort sieht wie ein typischer Tiefpassfilter aus, und es gibt keine Möglichkeit für uns zu erkennen, dass es sich um einen impulsformspezifischen Filter im Vergleich zu einem anderen Tiefpassfilter handelt.

Schließlich können wir unser Signal :math:`x` filtern und das Ergebnis untersuchen. Konzentriere dich nicht zu sehr auf die Einführung einer for-Schleife im bereitgestellten Code. Wir werden nach dem Codeblock erklären, warum sie da ist.

.. code-block:: python

    # Signal filtern, um die Impulsformung anzuwenden
    x_shaped = np.convolve(x, h)
    plt.figure(2)
    plt.plot(x_shaped, '.-')
    for i in range(num_symbols):
        plt.plot([i*sps+num_taps//2,i*sps+num_taps//2], [0, x_shaped[i*sps+num_taps//2]])
    plt.grid(True)
    plt.show()

.. image:: ../_images/pulse_shaping_python3.svg
   :align: center
   :target: ../_images/pulse_shaping_python3.svg

Das resultierende Signal ist aus vielen unserer Impulsantworten summiert, wobei ungefähr die Hälfte zunächst mit -1 multipliziert wurde. Es mag kompliziert aussehen, aber wir werden es gemeinsam durchgehen.

Zunächst gibt es transiente Abtastwerte vor und nach den Daten, die durch den Filter und die Funktionsweise der Faltung entstehen. Diese zusätzlichen Abtastwerte werden in unsere Übertragung aufgenommen, enthalten aber keine "Peaks" von Impulsen.

Zweitens wurden die vertikalen Linien in der for-Schleife zur Visualisierung erstellt. Sie sollen zeigen, wo Intervalle von :math:`Ts` auftreten. Diese Intervalle repräsentieren, wo dieses Signal vom Empfänger abgetastet wird. Beachte, dass die Kurve bei Intervallen von :math:`Ts` genau den Wert 1,0 oder -1,0 hat, was sie zu den idealen Zeitpunkten zum Abtasten macht.

Wenn wir dieses Signal hochkonvertieren und senden würden, müsste der Empfänger bestimmen, wo die Grenzen von :math:`Ts` liegen – z.B. mit einem Symbolsynchronisierungsalgorithmus. Auf diese Weise weiß der Empfänger *genau*, wann er abtasten soll, um die richtigen Daten zu erhalten. Wenn der Empfänger etwas zu früh oder zu spät abtastet, sieht er Werte, die aufgrund von ISI leicht verzerrt sind, und wenn er weit daneben liegt, erhält er eine Menge merkwürdiger Zahlen.

Hier ist ein Beispiel, das mit GNU Radio erstellt wurde und zeigt, wie der IQ-Plot (a.k.a. Konstellation) aussieht, wenn wir zum richtigen und falschen Zeitpunkt abtasten. Die Bitwerte der ursprünglichen Impulse sind annotiert.

.. image:: ../_images/symbol_sync1.png
   :scale: 50 %
   :align: center

Das folgende Diagramm zeigt die ideale Abtastposition in der Zeit sowie den IQ-Plot:

.. image:: ../_images/symbol_sync2.png
   :scale: 40 %
   :align: center
   :alt: GNU Radio-Simulation mit perfektem Timing beim Abtasten

Vergleiche das mit dem schlechtesten Abtastzeitpunkt. Beachte die drei Cluster in der Konstellation. Wir tasten direkt zwischen jedem Symbol ab; unsere Abtastwerte werden völlig daneben liegen.

.. image:: ../_images/symbol_sync3.png
   :scale: 40 %
   :align: center
   :alt: GNU Radio-Simulation mit unvollkommenem Timing beim Abtasten

Hier ist ein weiteres Beispiel für eine schlechte Abtastzeit, irgendwo zwischen unserem idealen und schlechtesten Fall. Beachte die vier Cluster. Bei hohem SNR könnten wir mit diesem Abtastzeitintervall durchkommen, obwohl es nicht empfehlenswert ist.

.. image:: ../_images/symbol_sync4.png
   :scale: 40 %
   :align: center

Denke daran, dass unsere Q-Werte nicht im Zeitbereichsplot gezeigt werden, weil sie ungefähr null sind, sodass sich die IQ-Plots nur horizontal ausbreiten.
