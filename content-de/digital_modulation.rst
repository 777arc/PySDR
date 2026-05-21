.. _modulation-chapter:

###################
Digitale Modulation
###################

In diesem Kapitel werden wir das *tatsächliche Übertragen von Daten* mittels digitaler Modulation und drahtlosen Symbolen besprechen! Wir werden Signale entwerfen, die „Informationen" übermitteln, z.B. Einsen und Nullen, unter Verwendung von Modulationsverfahren wie ASK, PSK, QAM und FSK. Wir werden auch IQ-Diagramme und Konstellationen besprechen und das Kapitel mit einigen Python-Beispielen abschließen.

Das Hauptziel der Modulation ist es, so viele Daten wie möglich in so wenig Spektrum wie möglich zu pressen. Technisch gesehen wollen wir die „spektrale Effizienz" in Einheiten von Bit/s/Hz maximieren. Das schnellere Übertragen von Einsen und Nullen erhöht die Bandbreite unseres Signals (erinnere dich an Fourier-Eigenschaften), was bedeutet, dass mehr Spektrum genutzt wird. Wir werden auch andere Techniken als nur das schnellere Senden untersuchen. Es wird viele Kompromisse geben, wenn wir entscheiden, wie wir modulieren wollen, aber es wird auch Raum für Kreativität geben.

*******************
Symbole
*******************
Neuer Begriff! Unser Sendesignal wird aus „Symbolen" bestehen. Jedes Symbol trägt eine bestimmte Anzahl von Informationsbits, und wir werden Symbole nacheinander übertragen, Tausende oder sogar Millionen hintereinander.

Als vereinfachtes Beispiel, sagen wir, wir haben ein Kabel und senden Einsen und Nullen mit hohen und niedrigen Spannungspegeln. Ein Symbol ist eine dieser Einsen oder Nullen:

.. image:: ../_images/symbols.png
   :scale: 60 %
   :align: center
   :alt: Pulse train of ones and zeros depicting the concept of a digital symbol that carries information

Im obigen Beispiel repräsentiert jedes Symbol ein Bit. Wie können wir mehr als ein Bit pro Symbol übermitteln? Lass uns die Signale untersuchen, die durch Ethernet-Kabel laufen, die in einem IEEE-Standard namens IEEE 802.3 1000BASE-T definiert sind. Der gängige Betriebsmodus von Ethernet verwendet eine 4-stufige Amplitudenmodulation (2 Bit pro Symbol) mit 8-ns-Symbolen.

.. image:: ../_images/ethernet.svg
   :align: center
   :target: ../_images/ethernet.svg
   :alt: Plot of IEEE 802.3 1000BASE-T Ethernet voltage signal showing 4-level amplitude shift keying (ASK)

Nimm dir einen Moment, um diese Fragen zu beantworten:

1. Wie viele Bits pro Sekunde werden im oben gezeigten Beispiel übertragen?
2. Wie viele Paare dieser Datenkabel wären nötig, um 1 Gigabit/s zu übertragen?
3. Wenn ein Modulationsverfahren 16 verschiedene Stufen hat, wie viele Bits pro Symbol sind das?
4. Mit 16 verschiedenen Stufen und 8-ns-Symbolen, wie viele Bits pro Sekunde sind das?

.. raw:: html

   <details>
   <summary>Antworten</summary>

1. 250 Mbps - (1/8e-9)*2
2. Vier (was Ethernet-Kabel haben)
3. 4 Bits pro Symbol - log_2(16)
4. 0,5 Gbps - (1/8e-9)*4

.. raw:: html

   </details>

*******************
Drahtlose Symbole
*******************
Frage: Warum können wir das oben in der Abbildung gezeigte Ethernet-Signal nicht direkt übertragen? Es gibt viele Gründe, die zwei Größten sind:

1. Niedrige Frequenzen erfordern *riesige* Antennen, und das obige Signal enthält Frequenzen bis hinunter zu DC (0 Hz). Wir können DC nicht übertragen.
2. Rechteckwellen belegen für die Bits pro Sekunde übermäßig viel Spektrum – erinnere dich aus dem Kapitel :ref:`freq-domain-chapter`, dass schnelle Änderungen im Zeitbereich eine große Menge an Bandbreite/Spektrum verbrauchen:

.. image:: ../_images/square-wave.svg
   :align: center
   :target: ../_images/square-wave.svg
   :alt: A square wave in time and frequency domain showing the large amount of bandwidth that a square wave uses

Was wir bei drahtlosen Signalen tun, ist mit einem Träger zu beginnen, der nur ein Sinusoid ist. Z.B. verwendet UKW-Radio einen Träger wie 101,1 MHz oder 100,3 MHz. Wir modulieren diesen Träger auf irgendeine Weise (es gibt viele). Bei UKW-Radio ist es eine analoge Modulation, nicht digital, aber es ist dasselbe Konzept wie bei der digitalen Modulation.

Auf welche Weisen können wir den Träger modulieren? Eine andere Weise, dieselbe Frage zu stellen: Was sind die verschiedenen Eigenschaften eines Sinusoids?

1. Amplitude
2. Phase
3. Frequenz

Wir können unsere Daten auf einen Träger modulieren, indem wir eine (oder mehrere) dieser drei Eigenschaften ändern.

****************************
Amplitudenumtastung (ASK)
****************************

Amplitudenumtastung (ASK, Amplitude Shift Keying) ist das erste digitale Modulationsverfahren, das wir besprechen werden, weil Amplitudenmodulation am einfachsten unter den drei Sinusoid-Eigenschaften zu visualisieren ist. Wir modulieren buchstäblich die **Amplitude** des Trägers. Hier ist ein Beispiel für 2-stufiges ASK, genannt 2-ASK:

.. image:: ../_images/ASK.svg
   :align: center
   :target: ../_images/ASK.svg
   :alt: Example of amplitude shift keying (ASK) in the time domain, specifically 2-ASK

Beachte, dass der Durchschnittswert null ist; das bevorzugen wir immer, wenn möglich.

Wir können mehr als zwei Stufen verwenden, was mehr Bits pro Symbol ermöglicht. Unten ist ein Beispiel für 4-ASK (Aus ist eine der vier Stufen). In diesem Fall trägt jedes Symbol 2 Bit Informationen.

.. image:: ../_images/ask2.svg
   :align: center
   :target: ../_images/ask2.svg
   :alt: Example of amplitude shift keying (ASK) in the time domain, specifically 4-ASK

Frage: Wie viele Symbole sind im obigen Signalausschnitt gezeigt? Wie viele Bits werden insgesamt dargestellt?

.. raw:: html

   <details>
   <summary>Antworten</summary>

20 Symbole, also 40 Bits Informationen

.. raw:: html

   </details>

Wie erstellen wir dieses Signal eigentlich digital, durch Code? Alles, was wir tun müssen, ist einen Vektor mit N Samples pro Symbol zu erstellen und diesen Vektor dann mit einem Sinusoid zu multiplizieren. Dies moduliert das Signal auf einen Träger (das Sinusoid fungiert als dieser Träger). Das folgende Beispiel zeigt 2-ASK mit 10 Samples pro Symbol.

.. image:: ../_images/ask3.svg
   :align: center
   :target: ../_images/ask3.svg
   :alt: Samples per symbol depiction using 2-ASK in the time domain, with 10 samples per symbol (sps)

Das obere Diagramm zeigt die diskreten Samples als rote Punkte, d.h. unser digitales Signal. Das untere Diagramm zeigt, wie das resultierende modulierte Signal aussieht, das über die Luft übertragen werden könnte. In echten Systemen ist die Frequenz des Trägers normalerweise viel höher als die Rate, mit der sich die Symbole ändern. In diesem Beispiel gibt es nur 2,5 Zyklen des Sinusoids in jedem Symbol, aber in der Praxis könnte es Tausende geben, je nachdem, wie hoch im Spektrum das Signal übertragen wird.

************************
Phasenumtastung (PSK)
************************

Lass uns nun die Phase auf ähnliche Weise modulieren wie wir es mit der Amplitude getan haben. Die einfachste Form ist Binäre PSK, auch BPSK genannt, bei der es zwei Phasenstufen gibt:

1. Keine Phasenänderung
2. 180-Grad-Phasenänderung

Beispiel für BPSK (beachte die Phasenänderungen):

.. image:: ../_images/bpsk.svg
   :align: center
   :target: ../_images/bpsk.svg
   :alt: Simple example of binary phase shift keying (BPSK) in the time domain, showing a modulated carrier

Es ist nicht sehr spaßig, Diagramme wie dieses anzusehen:

.. image:: ../_images/bpsk2.svg
   :align: center
   :target: ../_images/bpsk2.svg
   :alt: Phase shift keying like BPSK in the time domain is difficult to read, so we tend to use a constellation plot or complex plane

Stattdessen stellen wir die Phase normalerweise in der komplexen Ebene dar.

***********************
IQ-Diagramme/Konstellationen
***********************

Du hast IQ-Diagramme bereits im Unterabschnitt komplexe Zahlen des Kapitels :ref:`sampling-chapter` gesehen, aber jetzt werden wir sie auf eine neue und interessante Weise verwenden.
Für ein gegebenes Symbol können wir die Amplitude und Phase in einem IQ-Diagramm zeigen. Für das BPSK-Beispiel sagten wir, dass wir Phasen von 0 und 180 Grad haben. Lass uns diese zwei Punkte im IQ-Diagramm darstellen. Wir nehmen eine Magnitude von 1 an. In der Praxis spielt es keine Rolle, welche Magnitude du verwendest; ein höherer Wert bedeutet ein stärkeres Signal, aber du kannst auch einfach den Verstärkungsgrad des Verstärkers erhöhen.

.. image:: ../_images/bpsk_iq.png
   :scale: 80 %
   :align: center
   :alt: IQ plot or constellation plot of BPSK

Das obige IQ-Diagramm zeigt, was wir übertragen werden, oder genauer die Menge der Symbole, aus denen wir übertragen werden. Es zeigt nicht den Träger, also kannst du es dir als die Symbole im Basisband vorstellen. Wenn wir die Menge möglicher Symbole für ein gegebenes Modulationsverfahren zeigen, nennen wir das die „Konstellation". Viele Modulationsverfahren können durch ihre Konstellation definiert werden.

Um BPSK zu empfangen und zu dekodieren, können wir IQ-Abtastung verwenden, wie wir im letzten Kapitel gelernt haben, und prüfen, wo die Punkte im IQ-Diagramm landen. Es wird jedoch eine zufällige Phasendrehung durch den drahtlosen Kanal geben, weil das Signal eine zufällige Verzögerung haben wird, wenn es durch die Luft zwischen Antennen geht. Die zufällige Phasendrehung kann mit verschiedenen Methoden, die wir später lernen werden, rückgängig gemacht werden. Hier ist ein Beispiel für einige verschiedene Arten, wie ein BPSK-Signal am Empfänger erscheinen könnte (ohne Rauschen):

.. image:: ../_images/bpsk3.png
   :scale: 60 %
   :align: center
   :alt: A random phase rotation of BPSK occurs as the wireless signal travels through the air

Zurück zu PSK. Was wäre, wenn wir vier verschiedene Phasenstufen hätten? D.h. 0, 90, 180 und 270 Grad. In diesem Fall würde es im IQ-Diagramm so dargestellt werden, und es bildet ein Modulationsverfahren, das wir Quadraturphasenumtastung (QPSK) nennen:

.. image:: ../_images/qpsk.png
   :scale: 60 %
   :align: center
   :alt: Example of Quadrature Phase Shift Keying (QPSK) in the IQ plot or constellation plot

Bei PSK haben wir immer N verschiedene Phasen, gleichmäßig um 360 Grad verteilt, für beste Ergebnisse. Wir zeigen oft den Einheitskreis, um zu betonen, dass alle Punkte dieselbe Magnitude haben:

.. image:: ../_images/psk_set.png
   :scale: 60 %
   :align: center
   :alt: Phase shift keying uses equally spaced constellation points on the IQ plot

Frage: Was ist falsch an einem PSK-Verfahren wie dem im folgenden Bild? Ist es ein gültiges PSK-Modulationsverfahren?

.. image:: ../_images/weird_psk.png
   :scale: 60 %
   :align: center
   :alt: Example of non-uniformly spaced PSK constellation plot

.. raw:: html

   <details>
   <summary>Antwort</summary>

Es gibt nichts Ungültiges an diesem PSK-Verfahren. Du kannst es sicherlich verwenden, aber da die Symbole nicht gleichmäßig verteilt sind, ist dieses Verfahren nicht so effektiv wie es sein könnte. Die Verfahrenseffizienz wird klarer, sobald wir besprechen, wie Rauschen unsere Symbole beeinflusst. Die kurze Antwort ist, dass wir zwischen den Symbolen so viel Raum wie möglich lassen wollen, falls es Rauschen gibt, damit ein Symbol nicht vom Empfänger als eines der anderen (falschen) Symbole interpretiert wird. Wir wollen nicht, dass eine 0 als 1 empfangen wird.

.. raw:: html

   </details>

Lass uns kurz zu ASK zurückkehren. Beachte, dass wir ASK genauso wie PSK im IQ-Diagramm darstellen können. Hier ist das IQ-Diagramm von 2-ASK, 4-ASK und 8-ASK in der bipolaren Konfiguration, sowie 2-ASK und 4-ASK in der unipolaren Konfiguration. In diesem Kontext bedeutet bipolar, dass das modulierte Signal sowohl positive als auch negative Amplitudenwerte annehmen kann, während unipolares ASK nur positive Amplituden verwendet.

.. image:: ../_images/ask_set.png
   :scale: 50 %
   :align: center
   :alt: Bipolar and unipolar amplitude shift keying (ASK) constellation or IQ plots

Wie du vielleicht bemerkt hast, sind bipolares 2-ASK und BPSK dasselbe. Eine 180-Grad-Phasenverschiebung ist dasselbe wie das Multiplizieren des Sinusoids mit -1. Wir nennen es BPSK, wahrscheinlich weil PSK viel häufiger verwendet wird als ASK.

**************************************
Quadraturamplitudenmodulation (QAM)
**************************************
Was, wenn wir ASK und PSK kombinieren? Wir nennen dieses Modulationsverfahren Quadraturamplitudenmodulation (QAM). QAM sieht normalerweise ungefähr so aus:

.. image:: ../_images/64qam.png
   :scale: 90 %
   :align: center
   :alt: Example of Quadrature Amplitude Modulation (QAM) on the IQ or constellation plot

Hier sind einige weitere Beispiele für QAM:

.. image:: ../_images/qam.png
   :scale: 50 %
   :align: center
   :alt: Example of 16QAM, 32QAM, 64QAM, and 256QAM on the IQ or constellation plot

Bei einem QAM-Modulationsverfahren können wir technisch gesehen Punkte überall im IQ-Diagramm platzieren, da sowohl Phase *als auch* Amplitude moduliert werden. Die „Parameter" eines gegebenen QAM-Verfahrens sind am besten durch die QAM-Konstellation definiert. Alternativ kannst du die I- und Q-Werte für jeden Punkt auflisten, wie unten für QPSK:

.. image:: ../_images/qpsk_list.png
   :scale: 80 %
   :align: center
   :alt: Constellation or IQ plots can also be represented using a table of symbols

Beachte, dass die meisten Modulationsverfahren, außer den verschiedenen ASKs und BPSK, im Zeitbereich ziemlich schwer zu „sehen" sind. Um meinen Punkt zu beweisen, hier ist ein Beispiel von QAM im Zeitbereich. Kannst du die Phase jedes Symbols im folgenden Bild unterscheiden? Es ist schwierig.

.. image:: ../_images/qam_time_domain.png
   :scale: 50 %
   :align: center
   :alt: Looking at QAM in the time domain is difficult which is why we use constellation or IQ plots

Angesichts der Schwierigkeit, Modulationsverfahren im Zeitbereich zu unterscheiden, bevorzugen wir IQ-Diagramme gegenüber der Anzeige des Zeitbereichssignals. Wir könnten dennoch das Zeitbereichssignal zeigen, wenn es eine bestimmte Paketstruktur gibt oder die Abfolge der Symbole wichtig ist.

****************************
Frequenzumtastung (FSK)
****************************

Das Letzte auf der Liste ist Frequenzumtastung (FSK, Frequency Shift Keying). FSK ist ziemlich einfach zu verstehen – wir wechseln einfach zwischen N Frequenzen, wobei jede Frequenz ein mögliches Symbol ist. Da wir jedoch einen Träger modulieren, sind es wirklich unsere Trägerfrequenz +/- diese N Frequenzen. Z.B. könnten wir bei einem Träger von 1,2 GHz zwischen diesen vier Frequenzen wechseln:

1. 1,2001 GHz
2. 1,2003 GHz
3. 1,1999 GHz
4. 1,1997 GHz

Das obige Beispiel wäre 4-FSK, also gäbe es zwei Bit pro Symbol. Der Frequenzabstand beträgt 200 kHz, und das Gesamtsignal würde etwas mehr als 600 kHz umspannen. Dieses 4-FSK-Signal im Frequenzbereich beim Basisband könnte beim Durchführen einer FFT über viele Symbole ungefähr so aussehen:

.. image:: ../_images/fsk.svg
   :align: center
   :target: ../_images/fsk.svg
   :alt: Example of Frequency Shift Keying (FSK), specifically 4FSK

Wenn du FSK verwendest, musst du eine kritische Frage stellen: Wie groß soll der Abstand zwischen den Frequenzen sein? Wir bezeichnen diesen Abstand oft als :math:`\Delta f` in Hz. Wir wollen Überlappungen im Frequenzbereich vermeiden, damit der Empfänger weiß, welche Frequenz ein bestimmtes Symbol verwendet hat, daher muss :math:`\Delta f` groß genug sein. Die Breite jedes Trägers in der Frequenz ist eine Funktion unserer Symbolrate und eines angewendeten Pulsformungsfilters. Mehr Symbole pro Sekunde bedeutet kürzere Symbole, was breitere Bandbreite bedeutet (erinnere dich an die inverse Beziehung zwischen Zeit- und Frequenzskalierung). Je schneller wir Symbole übertragen, desto breiter wird jeder Träger, und folglich müssen wir :math:`\Delta f` größer machen, um überlappende Träger zu vermeiden.

IQ-Diagramme können nicht verwendet werden, um verschiedene Frequenzen zu zeigen. Sie zeigen Magnitude und Phase. 
Obwohl es möglich ist, FSK im Zeitbereich zu zeigen, macht es mehr als 2 Frequenzen schwierig, zwischen Symbolen zu unterscheiden:

.. image:: ../_images/fsk2.svg
   :align: center
   :target: ../_images/fsk2.svg
   :alt: Frequency Shift Keying (FSK) or 2FSK in the time domain

Nebenbei bemerkt, beachte, dass UKW-Radio Frequenzmodulation (FM) verwendet, die wie eine analoge Version von FSK ist. Anstatt zwischen diskreten Frequenzen zu wechseln, verwendet UKW-Radio ein kontinuierliches Audiosignal, um die Frequenz des Trägers zu modulieren. Unten ist ein Beispiel für FM- und AM-Modulation, bei dem das „Signal" oben das Audiosignal ist, das auf den Träger moduliert wird.

.. image:: ../_images/am_fm_animation.gif
   :align: center
   :scale: 75 %
   :target: ../_images/am_fm_animation.gif
   :alt: Animation of a carrier, amplitude modulation (AM), and frequency modulation (FM) in the time domain

In diesem Lehrbuch beschäftigen wir uns hauptsächlich mit digitalen Formen der Modulation.

*******************
Differenzialkodierung
*******************

In vielen drahtlosen (und kabelgebundenen) Kommunikationsprotokollen auf Basis von PSK oder QAM wirst du wahrscheinlich auf einen Schritt stoßen, der direkt vor der Bitmodulation (oder direkt nach der Demodulation) stattfindet, genannt Differenzialkodierung. Um ihren Nutzen zu demonstrieren, betrachte den Empfang eines BPSK-Signals. Wenn das Signal durch die Luft fliegt, erfährt es eine gewisse zufällige Verzögerung zwischen Sender und Empfänger, was eine zufällige Rotation in der Konstellation verursacht, wie wir zuvor erwähnt haben. Wenn der Empfänger sich darauf synchronisiert und das BPSK auf die „I"-Achse (real) ausrichtet, hat er keine Möglichkeit zu wissen, ob es 180 Grad außer Phase ist oder nicht, weil die Konstellation symmetrisch ist. Eine Option ist, Symbole zu übertragen, deren Wert dem Empfänger im Voraus bekannt ist, die mit den Informationen gemischt werden, bekannt als Pilotsymbole. Der Empfänger kann diese bekannten Symbole verwenden, um zu bestimmen, welcher Cluster eine 1 oder 0 ist, im Fall von BPSK. Pilotsymbole müssen in einem bestimmten Zeitraum gesendet werden, der damit zusammenhängt, wie schnell sich der drahtlose Kanal ändert, was letztendlich die Datenrate reduziert. Anstatt Pilotsymbole in die übertragene Wellenform mischen zu müssen, können wir wählen, Differenzialkodierung zu verwenden.

Der einfachste Fall der Differenzialkodierung ist bei der Verwendung neben BPSK, was ein Bit pro Symbol umfasst. Anstatt einfach eine 1 für binär 1 und eine -1 für binär 0 zu übertragen, beinhaltet BPSK-Differenzialkodierung das Übertragen einer 0, wenn das Eingangsbit dasselbe ist wie die **Kodierung** des vorherigen Bits (nicht das vorherige Eingangsbit selbst), und das Übertragen einer 1, wenn es sich unterscheidet. Wir übertragen immer noch dieselbe Anzahl von Bits, abgesehen von einem zusätzlichen Bit, das am Anfang benötigt wird, um die Ausgangssequenz zu starten, aber jetzt müssen wir uns keine Sorgen über die 180-Grad-Phasenmehrdeutigkeit machen. Dieses Kodierungsschema kann mit der folgenden Gleichung beschrieben werden, wobei :math:`x` die Eingangsbits und :math:`y` die Ausgangsbits sind, die mit BPSK moduliert werden:

.. math::
  y_i = y_{i-1} \oplus x_i

Da der Ausgang auf dem Ausgang des vorherigen Schritts basiert, müssen wir den Ausgang mit einer willkürlichen 1 oder 0 beginnen, und wie wir während des Dekodierungsprozesses zeigen werden, spielt es keine Rolle, welche wir wählen (wir müssen dieses Startsymbol trotzdem übertragen!).

Für visuelle Lernende kann der Differenzialkodierungsprozess als Diagramm dargestellt werden, wobei der Verzögerungsblock eine Verzögerung-um-1-Operation ist:

.. image:: ../_images/differential_coding2.svg
   :align: center
   :target: ../_images/differential_coding2.svg
   :alt: Differential coding block diagram

Als Kodierungsbeispiel betrachte das Übertragen der 10 Bits [1, 1, 0, 0, 1, 1, 1, 1, 1, 0] mit BPSK. Nehmen wir an, wir starten die Ausgangssequenz mit 1; es spielt tatsächlich keine Rolle, ob du 1 oder 0 verwendest. Es hilft, die Bits übereinander zu zeigen und dabei darauf zu achten, den Eingang zu verschieben, um Platz für das Startausgangsbit zu machen:

.. code-block::

 Eingang:     1 1 0 0 1 1 1 1 1 0
 Ausgang:  1

Als nächstes baust du den Ausgang auf, indem du das Eingangsbit mit dem vorherigen **Ausgangs**-Bit vergleichst und die XOR-Operation aus der obigen Tabelle anwendest. Das nächste Ausgangsbit ist daher eine 0, weil 1 und 1 übereinstimmen:

.. code-block::

 Eingang:    1 1 0 0 1 1 1 1 1 0
 Ausgang:  1 0

Wiederhole für den Rest und du erhältst:

.. code-block::

 Eingang:    1 1 0 0 1 1 1 1 1 0
 Ausgang:  1 0 1 1 1 0 1 0 1 0 0

Nach der Anwendung der Differenzialkodierung würden wir letztendlich [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0] übertragen. Die Einsen und Nullen werden immer noch auf die positiven und negativen Symbole abgebildet, die wir zuvor besprochen haben.

Der Dekodierungsprozess, der am Empfänger stattfindet, vergleicht das empfangene Bit mit dem vorherigen **empfangenen** Bit, was viel einfacher zu verstehen ist:

.. math::
  x_i = y_i \oplus y_{i-1}

Wenn du die BPSK-Symbole [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0] empfangen würdest, würdest du von links beginnen und prüfen, ob die ersten beiden übereinstimmen; in diesem Fall tun sie es nicht, also ist das erste Bit eine 1. Wiederhole und du erhältst die Sequenz, mit der wir begonnen haben, [1, 1, 0, 0, 1, 1, 1, 1, 1, 0]. Es mag nicht offensichtlich sein, aber das Startbit, das wir hinzugefügt haben, hätte eine 1 oder eine 0 sein können und wir würden dasselbe Ergebnis erhalten.

Der Kodierungs- und Dekodierungsprozess ist in der folgenden Grafik zusammengefasst:

.. image:: ../_images/differential_coding.svg
   :align: center
   :target: ../_images/differential_coding.svg
   :alt: Demonstration of differential coding using sequence of encoded and decoded bits


Der große Nachteil der Verwendung von Differenzialkodierung ist, dass ein Bitfehler zu zwei Bitfehlern führt. Die Alternative zur Verwendung von Differenzialkodierung für BPSK ist das periodische Hinzufügen von Pilotsymbolen, wie zuvor besprochen, die auch verwendet werden können, um Mehrwegausbreitung durch den Kanal umzukehren/zu invertieren. Aber ein Problem mit Pilotsymbolen ist, dass sich der drahtlose Kanal sehr schnell ändern kann, in der Größenordnung von Zehnten oder Hunderten von Symbolen, wenn es ein beweglicher Empfänger und/oder Sender ist, sodass du Pilotsymbole oft genug benötigen würdest, um den sich ändernden Kanal widerzuspiegeln. Wenn also ein drahtloses Protokoll großen Wert auf die Reduzierung der Komplexität des Empfängers legt, wie RDS, das wir im Kapitel :ref:`rds-chapter` studieren, kann es sich für die Verwendung von Differenzialkodierung entscheiden.

Denk daran, dass das obige Differenzialkodierungsbeispiel spezifisch für BPSK war. Differenzialkodierung wird auf Symbolebene angewendet, also arbeitest du bei der Anwendung auf QPSK mit Paaren von Bits gleichzeitig, und so weiter für höhere QAM-Schemata. Differenzielles QPSK wird oft als DQPSK bezeichnet.

*******************
Python-Beispiel
*******************

Als kurzes Python-Beispiel lass uns QPSK im Basisband erzeugen und die Konstellation darstellen.

Obwohl wir die komplexen Symbole direkt erzeugen könnten, fangen wir mit dem Wissen an, dass QPSK vier Symbole in 90-Grad-Abständen um den Einheitskreis hat. Wir werden 45, 135, 225 und 315 Grad für unsere Punkte verwenden. Zuerst erzeugen wir zufällige Zahlen zwischen 0 und 3 und führen Mathematik durch, um die gewünschten Grad zu erhalten, bevor wir in Radiant umrechnen.

.. code-block:: python

 import numpy as np
 import matplotlib.pyplot as plt

 num_symbols = 1000

 x_int = np.random.randint(0, 4, num_symbols) # 0 bis 3
 x_degrees = x_int*360/4.0 + 45 # 45, 135, 225, 315 Grad
 x_radians = x_degrees*np.pi/180.0 # sin() und cos() nehmen Radiant
 x_symbols = np.cos(x_radians) + 1j*np.sin(x_radians) # erzeugt unsere QPSK-Komplexsymbole
 plt.plot(np.real(x_symbols), np.imag(x_symbols), '.')
 plt.grid(True)
 plt.show()

.. image:: ../_images/qpsk_python.svg
   :align: center
   :target: ../_images/qpsk_python.svg
   :alt: QPSK generated or simulated in Python

Beachte, wie alle von uns erzeugten Symbole überlappen. Es gibt kein Rauschen, sodass alle Symbole denselben Wert haben. Lass uns etwas Rauschen hinzufügen:

.. code-block:: python

 n = (np.random.randn(num_symbols) + 1j*np.random.randn(num_symbols))/np.sqrt(2) # AWGN mit Einheitsleistung
 noise_power = 0.01
 r = x_symbols + n * np.sqrt(noise_power)
 plt.plot(np.real(r), np.imag(r), '.')
 plt.grid(True)
 plt.show()

.. image:: ../_images/qpsk_python2.svg
   :align: center
   :target: ../_images/qpsk_python2.svg
   :alt: QPSK with AWGN noise generated or simulated in Python

Beachte, wie additives weißes Gauß'sches Rauschen (AWGN) eine gleichmäßige Streuung um jeden Punkt in der Konstellation erzeugt. Wenn es zu viel Rauschen gibt, beginnen Symbole, die Grenze (die vier Quadranten) zu überschreiten, und werden vom Empfänger als falsches Symbol interpretiert. Versuche, :code:`noise_power` zu erhöhen, bis das passiert.

Für diejenigen, die daran interessiert sind, Phasenrauschen zu simulieren, das aus Phasenjitter innerhalb des lokalen Oszillators (LO) resultieren könnte, ersetze :code:`r` durch:

.. code-block:: python

 phase_noise = np.random.randn(len(x_symbols)) * 0.1 # Multiplikator für „Stärke" des Phasenrauschens anpassen
 r = x_symbols * np.exp(1j*phase_noise)

.. image:: ../_images/phase_jitter.svg
   :align: center
   :target: ../_images/phase_jitter.svg
   :alt: QPSK with phase jitter generated or simulated in Python

Du könntest sogar Phasenrauschen mit AWGN kombinieren, um die volle Erfahrung zu machen:

.. image:: ../_images/phase_jitter_awgn.svg
   :align: center
   :target: ../_images/phase_jitter_awgn.svg
   :alt: QPSK with AWGN noise and phase jitter generated or simulated in Python

Wir hören an diesem Punkt auf. Wenn wir sehen wollten, wie das QPSK-Signal im Zeitbereich aussieht, müssten wir mehrere Samples pro Symbol erzeugen (in dieser Übung haben wir nur 1 Sample pro Symbol gemacht). Du wirst lernen, warum du mehrere Samples pro Symbol erzeugen musst, sobald wir die Pulsformung besprechen. Das Python-Beispiel im Kapitel :ref:`pulse-shaping-chapter` wird dort weitermachen, wo wir hier aufgehört haben.

*******************
Weiterführende Literatur
*******************

#. https://en.wikipedia.org/wiki/Differential_coding
