.. _multipath-chapter:

#######################
Mehrwegeausbreitung
#######################

In diesem Kapitel stellen wir die Mehrwegeausbreitung vor – ein Ausbreitungsphänomen, bei dem Signale auf zwei oder mehr Wegen den Empfänger erreichen. Dieses Phänomen tritt in realen drahtlosen Systemen auf. Bislang haben wir nur den "AWGN-Kanal" besprochen, d.h. ein Modell für einen drahtlosen Kanal, bei dem dem Signal einfach Rauschen hinzugefügt wird – was nur für Signale über Kabel und einige Satellitenkommunikationssysteme gilt.

*************************
Mehrwegeausbreitung
*************************

Alle realistischen drahtlosen Kanäle enthalten viele "Reflektoren", da RF-Signale reflektiert werden. Jedes Objekt zwischen oder in der Nähe des Senders (Tx) oder Empfängers (Rx) kann zusätzliche Ausbreitungswege verursachen. Jeder Weg erfährt eine unterschiedliche Phasenverschiebung (Verzögerung) und Dämpfung (Amplitudenskalierung). Am Empfänger addieren sich alle Wege. Sie können sich konstruktiv, destruktiv oder gemischt addieren. Wir bezeichnen dieses Konzept mehrerer Signalwege als "Mehrwegeausbreitung". Es gibt den Sichtverbindungsweg (LOS, Line-of-Sight) und dann alle anderen Wege. Im folgenden Beispiel zeigen wir den LOS-Weg und einen einzelnen Nicht-LOS-Weg:

.. image:: ../_images/multipath.svg
   :align: center
   :target: ../_images/multipath.svg
   :alt: Einfache Darstellung der Mehrwegeausbreitung mit dem LOS-Weg und einem einzelnen Mehrweg

Destruktive Interferenz kann auftreten, wenn die Überlagerung der Wege ungünstig ausfällt. Betrachte das obige Beispiel mit nur zwei Wegen. Abhängig von der Frequenz und dem genauen Abstand der Wege können die beiden Wege mit ungefähr gleicher Amplitude und 180 Grad Phasenverschiebung empfangen werden, wodurch sie sich gegenseitig auslöschen (wie unten dargestellt). Du hast vielleicht im Physikunterricht von konstruktiver und destruktiver Interferenz gehört. In drahtlosen Systemen bezeichnen wir diese destruktive Überlagerung als "tiefes Fading" (Deep Fade), weil unser Signal kurzzeitig verschwindet.

.. image:: ../_images/destructive_interference.svg
   :align: center
   :target: ../_images/destructive_interference.svg

Wege können sich auch konstruktiv addieren und ein starkes Signal erzeugen. Jeder Weg hat eine unterschiedliche Phasenverschiebung und Amplitude, die wir in einem Zeitbereichsdiagramm, dem sogenannten "Leistungsverzögerungsprofil" (Power Delay Profile), visualisieren können:

.. image:: ../_images/multipath2.svg
   :align: center
   :target: ../_images/multipath2.svg
   :alt: Mehrwegeausbreitung mit dem Leistungsverzögerungsprofil über die Zeit

Der erste Weg, der der y-Achse am nächsten liegt, ist immer der LOS-Weg (sofern vorhanden), da kein anderer Weg den Empfänger schneller erreichen kann als der LOS-Weg. Typischerweise nimmt die Amplitude mit zunehmender Verzögerung ab, da ein Weg, der später am Empfänger ankommt, eine größere Distanz zurückgelegt hat.

*************************
Fading
*************************

Was in der Regel passiert, ist eine Mischung aus konstruktiver und destruktiver Interferenz, die sich mit der Zeit ändert, wenn sich Rx, Tx oder die Umgebung bewegen/verändern. Wir verwenden den Begriff "Fading", wenn wir uns auf die Auswirkungen eines Mehrwegekanals beziehen, der sich **über die Zeit** verändert. Deshalb bezeichnen wir es oft als "Mehrwege-Fading"; es ist wirklich die Kombination aus konstruktiver/destruktiver Interferenz und einer sich verändernden Umgebung. Das Ergebnis ist ein SNR, das über die Zeit variiert; Änderungen liegen je nach Bewegungsgeschwindigkeit von Tx/Rx typischerweise im Millisekunden- bis Mikrosekundenbereich. Unten ist ein SNR-Plot über die Zeit in Millisekunden, der Mehrwege-Fading demonstriert.

.. image:: ../_images/multipath_fading.png
   :scale: 100 %
   :align: center
   :alt: Mehrwege-Fading verursacht periodische tiefe Fades oder Nullstellen, wo der SNR extrem niedrig wird

Es gibt zwei Arten von Fading aus **Zeit**-Domain-Perspektive:

- **Langsames Fading:** Der Kanal ändert sich innerhalb eines Pakets nicht. D.h. ein tiefes Null bei langsamem Fading löscht das gesamte Paket aus.
- **Schnelles Fading:** Der Kanal ändert sich sehr schnell im Vergleich zur Länge eines Pakets. Vorwärtsfehlerkorrektur (Forward Error Correction), kombiniert mit Interleaving, kann schnelles Fading bekämpfen.

Es gibt auch zwei Arten von Fading aus **Frequenz**-Domain-Perspektive:

**Frequenzselektives Fading**: Die konstruktive/destruktive Interferenz ändert sich innerhalb des Frequenzbereichs des Signals. Bei einem Breitbandsignal überdecken wir einen großen Frequenzbereich. Denke daran, dass die Wellenlänge bestimmt, ob es konstruktiv oder destruktiv ist. Wenn unser Signal einen weiten Frequenzbereich umfasst, umfasst es auch einen weiten Wellenlängenbereich (da Wellenlänge die inverse Frequenz ist). Folglich können wir unterschiedliche Kanalqualitäten in verschiedenen Teilen unseres Signals (im Frequenzbereich) erhalten. Daher der Name frequenzselektives Fading.

**Flaches Fading**: Tritt auf, wenn die Signalbandbreite schmal genug ist, sodass alle Frequenzen ungefähr denselben Kanal erfahren. Wenn ein tiefes Fading auftritt, verschwindet das gesamte Signal (für die Dauer des tiefen Fadings).

In der folgenden Abbildung zeigt die :red:`rote` Form unser Signal im Frequenzbereich, und die schwarze geschwungene Linie zeigt den aktuellen Kanalzustand über die Frequenz. Da das schmalere Signal die gleichen Kanalbedingungen im gesamten Signal erfährt, erlebt es flaches Fading. Das breitere Signal erfährt deutlich frequenzselektives Fading.

.. image:: ../_images/flat_vs_freq_selective.png
   :scale: 70 %
   :align: center
   :alt: Flaches Fading vs. frequenzselektives Fading

Hier ist ein Beispiel eines 16 MHz breiten Signals, das kontinuierlich sendet. Es gibt mehrere Momente in der Mitte, wo kurzzeitig ein Teil des Signals fehlt. Dieses Beispiel zeigt frequenzselektives Fading, das Lücken im Signal erzeugt, die einige Frequenzen auslöschen, andere jedoch nicht.

.. image:: ../_images/fading_example.jpg
   :scale: 60 %
   :align: center
   :alt: Beispiel für frequenzselektives Fading in einem Spektrogramm (auch Wasserfallplot), das Verschmierung und ein Loch im Spektrogramm bei einem tiefen Null zeigt

**************************
Rayleigh-Fading simulieren
**************************

Rayleigh-Fading wird verwendet, um Fading über die Zeit zu modellieren, wenn es keinen signifikanten LOS-Weg gibt. Wenn ein dominanter LOS-Weg vorhanden ist, ist das Rician-Fading-Modell besser geeignet, aber wir konzentrieren uns auf Rayleigh. Beachte, dass die Rayleigh- und Rician-Modelle weder den primären Pfadverlust zwischen Sender und Empfänger (wie den als Teil eines Linkbudgets berechneten Pfadverlust) noch Abschattung durch große Objekte beinhalten. Ihre Rolle ist es, das Mehrwege-Fading zu modellieren, das über die Zeit als Ergebnis von Bewegung und Streuobjekten in der Umgebung auftritt.

Aus dem Rayleigh-Fading-Modell ergeben sich viele Theorien, wie z.B. Ausdrücke für die Pegelüberschreitungsrate und die durchschnittliche Fadedauer. Das Rayleigh-Fading-Modell sagt uns jedoch nicht direkt, wie wir einen Kanal mit dem Modell tatsächlich simulieren sollen. Um Rayleigh-Fading in der Simulation zu erzeugen, müssen wir eine von vielen veröffentlichten Methoden verwenden, und im folgenden Python-Beispiel verwenden wir Clarkes "Summe von Sinusoidalsignalen"-Methode.

Um einen Rayleigh-Fading-Kanal in Python zu generieren, müssen wir zunächst die maximale Doppler-Verschiebung in Hz angeben, die davon abhängt, wie schnell sich Sender und/oder Empfänger bewegen, bezeichnet mit :math:`\Delta v`. Wenn die Geschwindigkeit klein im Vergleich zur Lichtgeschwindigkeit ist – was in der drahtlosen Kommunikation immer der Fall ist – kann die Doppler-Verschiebung berechnet werden als:

.. math::

  f_D = \frac{\Delta v f_c}{c}

wobei :math:`c` die Lichtgeschwindigkeit ist, ungefähr 3e8 m/s, und :math:`f_c` die Trägerfrequenz der Übertragung ist.

Wir wählen auch die Anzahl der zu simulierenden Sinusoide, und es gibt keine richtige Antwort, da sie auf der Anzahl der Streuer in der Umgebung basiert, die wir nie wirklich kennen. Als Teil der Berechnungen nehmen wir an, dass die Phase des empfangenen Signals von jedem Weg gleichmäßig zufällig zwischen 0 und :math:`2\pi` ist. Der folgende Code simuliert einen Rayleigh-Fading-Kanal mit Clarkes Methode:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    # Simulationsparameter, diese können frei angepasst werden
    v_mph = 60 # Geschwindigkeit von TX oder RX, in Meilen pro Stunde
    center_freq = 200e6 # RF-Trägerfrequenz in Hz
    Fs = 1e5 # Abtastrate der Simulation
    N = 100 # Anzahl der zu summierenden Sinusoide

    v = v_mph * 0.44704 # Umrechnung in m/s
    fd = v*center_freq/3e8 # maximale Doppler-Verschiebung
    print("Maximale Doppler-Verschiebung:", fd)
    t = np.arange(0, 1, 1/Fs) # Zeitvektor. (Start, Stop, Schritt)
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    for i in range(N):
        alpha = (np.random.rand() - 0.5) * 2 * np.pi
        phi = (np.random.rand() - 0.5) * 2 * np.pi
        x = x + np.random.randn() * np.cos(2 * np.pi * fd * t * np.cos(alpha) + phi)
        y = y + np.random.randn() * np.sin(2 * np.pi * fd * t * np.cos(alpha) + phi)

    # z ist der komplexe Koeffizient, der den Kanal darstellt – als Phasenverschiebung und Amplitudenskalierung zu verstehen
    z = (1/np.sqrt(N)) * (x + 1j*y) # das ist, was du bei der Kanalsimulation tatsächlich verwenden würdest
    z_mag = np.abs(z) # Betrag für die Darstellung nehmen
    z_mag_dB = 10*np.log10(z_mag) # in dB umrechnen

    # Fading über die Zeit darstellen
    plt.plot(t, z_mag_dB)
    plt.plot([0, 1], [0, 0], ':r') # 0 dB
    plt.legend(['Rayleigh-Fading', 'Kein Fading'])
    plt.axis([0, 1, -15, 5])
    plt.show()

Wenn du dieses Kanalmodell als Teil einer größeren Simulation verwenden möchtest, multiplizierst du das empfangene Signal einfach mit der komplexen Zahl :code:`z`, die flaches Fading darstellt. Der Wert :code:`z` würde dann bei jedem Zeitschritt aktualisiert. Das bedeutet, dass alle Frequenzkomponenten des Signals zu einem gegebenen Zeitpunkt denselben Kanal erfahren, sodass du **kein** frequenzselektives Fading simulierst – dafür ist eine Mehrtap-Kanalimpulsantwort erforderlich, auf die wir in diesem Kapitel nicht eingehen. Wenn wir den Betrag von :code:`z` betrachten, können wir das Rayleigh-Fading über die Zeit sehen:

.. image:: ../_images/rayleigh.svg
   :align: center
   :target: ../_images/rayleigh.svg
   :alt: Simulation von Rayleigh-Fading

Beachte die tiefen Fades, die kurz auftreten, sowie den kleinen Zeitanteil, in dem der Kanal tatsächlich besser abschneidet als ohne Fading.


****************************
Mehrwege-Fading bekämpfen
****************************

In der modernen Kommunikation haben wir Methoden entwickelt, um Mehrwege-Fading zu bekämpfen.

CDMA
#####

3G-Mobilfunk verwendet eine Technologie namens Code Division Multiple Access (CDMA). Mit CDMA nimmst du ein Schmalbandssignal und spreizst es vor der Übertragung über eine breite Bandbreite (mit einer Spreizspektrumtechnik namens DSSS). Bei frequenzselektivem Fading ist es unwahrscheinlich, dass alle Frequenzen gleichzeitig ein tiefes Null haben. Am Empfänger wird die Spreizung rückgängig gemacht, und dieser De-Spreizungsprozess bekämpft ein tiefes Null erheblich.

.. image:: ../_images/cdma.png
   :scale: 100 %
   :align: center

OFDM
#####

4G-Mobilfunk, WiFi und viele andere Technologien verwenden ein Schema namens Orthogonal Frequency-Division Multiplexing (OFDM). OFDM verwendet sogenannte Subträger, bei denen wir das Signal im Frequenzbereich in viele schmale, eng beieinander liegende Signale aufteilen. Um Mehrwege-Fading zu bekämpfen, können wir die Zuweisung von Daten an Subträger vermeiden, die sich in einem tiefen Fading befinden, obwohl dies erfordert, dass die Empfangsseite schnell genug Kanalinformationen zurück an den Sender schickt. Wir können auch höherwertige Modulationsverfahren Subträgern mit guter Kanalqualität zuweisen, um unsere Datenrate zu maximieren.
