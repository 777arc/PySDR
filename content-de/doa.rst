.. _doa-chapter:

#################
Beamforming & DOA
#################

In diesem Kapitel behandeln wir die Konzepte Beamforming, Ankunftsrichtungsschätzung (DOA) und Phased Arrays im Allgemeinen. Wir vergleichen verschiedene Array-Typen und -Geometrien und zeigen, welche entscheidende Rolle der Elementabstand spielt. Techniken wie MVDR/Capon und MUSIC werden eingeführt und anhand von Python-Simulationsbeispielen demonstriert.

*********************
Übersicht: Beamforming
*********************

Ein Phased Array, auch als elektronisch geschwenktes Array bezeichnet, ist ein Array bzw. eine Sammlung von Antennen, das auf der Sende- oder Empfangsseite in Kommunikations- und Radarsystemen eingesetzt werden kann. Phased Arrays findet man in bodengestützten Systemen, in der Luft und auf Satelliten. Die einzelnen Antennen eines Arrays bezeichnen wir üblicherweise als Elemente, und manchmal wird das Array auch als „Sensor" bezeichnet. Diese Array-Elemente sind meist omnidirektionale Antennen, die gleichmäßig in einer Linie oder über zwei Dimensionen verteilt sind.

Beamforming ist eine Signalverarbeitungsoperation, die mit Antennenarrays eingesetzt wird, um einen *räumlichen* Filter zu erzeugen – er filtert Signale aus allen Richtungen außer der gewünschten Richtung(en) heraus. Beamforming kann eingesetzt werden, um den SNR von Nutzsignalen zu erhöhen, Störsender zu unterdrücken, Strahlmuster zu formen oder sogar mehrere Datenströme gleichzeitig auf derselben Frequenz zu senden/empfangen. Als Teil des Beamformings verwenden wir Gewichte (auch Koeffizienten genannt), die auf jedes Element des Arrays angewendet werden, entweder digital oder in analoger Schaltung. Wir manipulieren die Gewichte, um die Strahlen des Arrays zu formen – daher der Name Beamforming! Wir können diese Strahlen (und Nullstellen) extrem schnell schwenken – viel schneller als mechanisch geschwenkte Antennen, die als Alternative zu Phased Arrays angesehen werden können. Beamforming wird typischerweise im Kontext einer Kommunikationsverbindung diskutiert, bei der der Empfänger versucht, ein oder mehrere Signale mit dem bestmöglichen SNR zu empfangen. Arrays spielen auch eine wichtige Rolle im Radar, wo das Ziel die Erkennung und Verfolgung von Zielen ist.

.. image:: ../_images/doa_complex_scenario.svg
   :align: center
   :target: ../_images/doa_complex_scenario.svg
   :alt: Diagramm eines komplexen Szenarios mit mehreren Signalen, die auf ein Array treffen

Beamforming-Ansätze lassen sich in drei Kategorien unterteilen: konventionell, adaptiv und blind. Konventionelles Beamforming ist am nützlichsten, wenn die Ankunftsrichtung des Nutzsignals bereits bekannt ist; dabei werden Gewichte gewählt, die den Array-Gewinn in dieser Richtung maximieren. Dies kann sowohl auf der Empfangs- als auch auf der Sendeseite eines Kommunikationssystems eingesetzt werden. Adaptives Beamforming hingegen passt die Gewichte typischerweise anhand des Eingangssignals des Beamformers an, um ein bestimmtes Kriterium zu optimieren (z. B. einen Störsender zu unterdrücken, mehrere Hauptkeulen zu erzeugen usw.). Aufgrund seiner geschlossenen Regelschleife wird adaptives Beamforming typischerweise nur auf der Empfangsseite eingesetzt; das „Eingangssignal des Beamformers" ist dabei einfach das empfangene Signal, und adaptives Beamforming passt die Gewichte anhand der Statistiken dieser empfangenen Daten an.

Die folgende Taxonomie versucht, die vielen Bereiche des Beamformings zu kategorisieren und zeigt Beispieltechniken:

.. image:: ../_images/beamforming_taxonomy.svg
   :align: center
   :target: ../_images/beamforming_taxonomy.svg
   :alt: Eine Beamforming-Taxonomie, die Beamforming in konventionell, adaptiv und blind kategorisiert sowie zeigt, wie DOA-Schätzung einzuordnen ist

******************************
Übersicht: Ankunftsrichtung (DOA)
******************************

Direction-of-Arrival (DOA) bezeichnet im DSP/SDR-Bereich den Prozess, ein Antennenarray zu verwenden, um die Ankunftsrichtungen eines oder mehrerer vom Array empfangener Signale zu erkennen und zu schätzen (im Gegensatz zum Beamforming, das sich auf den Empfang eines Signals bei gleichzeitiger Unterdrückung von Rauschen und Interferenz konzentriert). Obwohl DOA eindeutig unter das Thema Beamforming fällt, können die beiden Begriffe verwechselt werden. Einige Techniken wie konventionelles Beamforming und MVDR können sowohl für DOA als auch für Beamforming verwendet werden, da dieselbe Technik, die für Beamforming verwendet wird, auch für DOA eingesetzt wird: Man schwenkt den Winkel über alle Interessenwinkel und führt die Beamforming-Operation bei jedem Winkel durch, sucht dann nach Peaks im Ergebnis (jeder Peak ist ein Signal, aber wir wissen nicht, ob es das Nutzsignal, ein Störsender oder sogar eine Mehrwegreflexion des Nutzsignals ist). Man kann diese DOA-Techniken als einen Wrapper um einen bestimmten Beamformer betrachten. Andere Beamformer können nicht einfach in eine DOA-Routine eingebettet werden, z. B. aufgrund zusätzlicher Eingaben, die im DOA-Kontext nicht verfügbar sind. Es gibt auch DOA-Techniken wie MUSIC und ESPRIT, die ausschließlich für DOA vorgesehen sind und keine Beamformer sind. Da die meisten Beamforming-Techniken voraussetzen, dass der Ankunftswinkel des Nutzsignals bekannt ist, muss DOA kontinuierlich als Zwischenschritt durchgeführt werden, wenn sich das Ziel oder das Array bewegt – selbst wenn das primäre Ziel das Empfangen und Demodulieren des Nutzsignals ist.

Phased Arrays und Beamforming/DOA finden Anwendung in verschiedensten Bereichen, am häufigsten jedoch in verschiedenen Formen des Radars, neueren WLAN-Standards, mmWave-Kommunikation in 5G, Satellitenkommunikation und Störsendung (Jamming). Im Allgemeinen eignen sich alle Anwendungen, die eine Antenne mit hohem Gewinn oder eine schnell bewegliche Hochgewinn-Antenne erfordern, gut für Phased Arrays.

******************
Array-Typen
******************

Phased Arrays lassen sich in drei Typen unterteilen:

1. **Analog**, auch als passiv elektronisch geschwenktes Array (PESA) oder traditionelles Phased Array bezeichnet, wobei analoge Phasenschieber zur Strahlsteuerung verwendet werden. Auf der Empfangsseite werden alle Elemente nach der Phasenverschiebung (und optional einer einstellbaren Verstärkung) summiert und in ein einzelnes Signal umgewandelt, das abwärtsgemischt und empfangen wird. Auf der Sendeseite läuft der Prozess umgekehrt ab: Ein einzelnes digitales Signal wird ausgegeben, und auf der Analogseite werden Phasenschieber und Verstärkungsstufen verwendet, um das Signal für jede Antenne zu erzeugen. Diese digitalen Phasenschieber haben eine begrenzte Bit-Auflösung und eine Steuerlatenz.
2. **Digital**, auch als aktiv elektronisch geschwenktes Array (AESA) bezeichnet, bei dem jedes einzelne Element sein eigenes HF-Frontend hat und das Beamforming vollständig im digitalen Bereich erfolgt. Dies ist der teuerste Ansatz, da HF-Komponenten teuer sind, bietet jedoch viel mehr Flexibilität und Geschwindigkeit als PESAs. Digitale Arrays sind bei SDRs beliebt, obwohl die Anzahl der Empfangs- oder Sendekanäle des SDR die Anzahl der Elemente im Array begrenzt.
3. **Hybrid**, bei dem das Array aus vielen Subarrays besteht, die einzeln einem analogen Array ähneln, wobei jedes Subarray ein eigenes HF-Frontend hat, genau wie bei digitalen Arrays. Dies ist der häufigste Ansatz für moderne Phased Arrays, da er das Beste aus beiden Welten bietet.

Beachte, dass die Begriffe PESA und AESA hauptsächlich im Radarkontext verwendet werden und es gewisse Unklarheiten darüber gibt, was genau ein PESA oder AESA ausmacht. Daher ist die Verwendung der Begriffe analog/digital/hybrid-Array klarer und kann auf jede Art von Anwendung angewendet werden.

Ein reales Beispiel für jeden Typ ist unten dargestellt:

.. image:: ../_images/beamforming_examples.svg
   :align: center
   :target: ../_images/beamforming_examples.svg
   :alt: Beispiele für Phased Arrays, einschließlich PESA, AESA und Hybrid-Array

Neben diesen drei Typen gibt es auch die Geometrie des Arrays. Die einfachste Geometrie ist das uniforme lineare Array (ULA), bei dem die Antennen in einer geraden Linie mit gleichem Abstand angeordnet sind (d. h. in 1 Dimension). ULAs haben eine 180-Grad-Mehrdeutigkeit, auf die wir später eingehen, und eine Lösung besteht darin, die Antennen in einem Kreis anzuordnen, was wir als uniformes kreisförmiges Array (UCA) bezeichnen. Für 2D-Strahlen verwenden wir üblicherweise ein uniformes rechteckiges Array (URA), bei dem die Antennen in einem Rastermuster angeordnet sind.

In diesem Kapitel konzentrieren wir uns auf digitale Arrays, da sie besser für Simulation und DSP geeignet sind, aber die Konzepte übertragen sich auf analoge und hybride Arrays. Im nächsten Kapitel arbeiten wir praktisch mit dem „Phaser"-SDR von Analog Devices, das ein 10-GHz-8-Element-Analog-Array mit Phasen- und Verstärkungsstellern hat, das mit einem Pluto und einem Raspberry Pi verbunden ist. Wir konzentrieren uns auch auf die ULA-Geometrie, da sie die einfachste Mathematik und den einfachsten Code bietet, aber alle Konzepte übertragen sich auf andere Geometrien, und am Ende des Kapitels berühren wir kurz das UCA.

*******************
SDR-Anforderungen
*******************

Analoge Phased Arrays beinhalten einen Phasenschieber (und oft eine einstellbare Verstärkungsstufe) pro Kanal/Element, der in analoger HF-Schaltung implementiert ist. Das bedeutet, dass ein analoges Phased Array ein dediziertes Hardwarestück ist, das neben einem SDR verwendet werden muss oder für eine bestimmte Anwendung speziell gebaut wurde. Andererseits kann jedes SDR, das mehr als einen Kanal enthält, ohne zusätzliche Hardware als digitales Array verwendet werden, solange die Kanäle phasenkohärent sind und mit demselben Takt abgetastet werden, was typischerweise bei SDRs mit mehreren Empfangskanälen der Fall ist. Es gibt viele SDRs mit **zwei** Empfangskanälen, wie das Ettus USRP B210 und Analog Devices Pluto (der 2. Kanal ist über einen uFL-Stecker auf der Platine zugänglich). Leider erfordert das Überschreiten von zwei Kanälen den Einstieg in das SDR-Segment über 10.000 USD, zumindest Stand 2023, wie das Ettus USRP N310 oder das Analog Devices QuadMXFE (16 Kanäle). Die Hauptherausforderung besteht darin, dass kostengünstige SDRs typischerweise nicht „zusammengekettet" werden können, um die Anzahl der Kanäle zu skalieren. Ausnahmen sind das KerberosSDR (4 Kanäle) und das KrakenSDR (5 Kanäle), die mehrere RTL-SDRs mit einem gemeinsamen LO verwenden, um ein kostengünstiges digitales Array zu bilden; der Nachteil ist die sehr begrenzte Abtastrate (bis zu 2,56 MHz) und der Abstimmbereich (bis zu 1766 MHz). Die KrakenSDR-Platine und eine Beispiel-Antennenkonfiguration sind unten dargestellt.

.. image:: ../_images/krakensdr.jpg
   :align: center
   :alt: Das KrakenSDR
   :target: ../_images/krakensdr.jpg

In diesem Kapitel verwenden wir keine spezifischen SDRs; stattdessen simulieren wir den Empfang von Signalen mit Python und gehen dann durch die DSP-Verarbeitung für Beamforming/DOA bei digitalen Arrays.

**************************************
Einführung in Matrizenrechnung mit Python/NumPy
**************************************

Python hat viele Vorteile gegenüber MATLAB, wie z. B. kostenlos und Open-Source zu sein, Vielfalt der Anwendungen, lebendige Community, Indizes beginnen bei 0 wie jede andere Sprache, Verwendung in KI/ML, und es scheint eine Bibliothek für alles zu geben, was man sich vorstellen kann. Wo es jedoch schwächer ist, ist die Art und Weise, wie Matrizenmanipulation kodiert/dargestellt wird (rechnerisch/geschwindigkeitsmäßig ist es durchaus schnell, wobei Funktionen intern effizient in C/C++ implementiert sind). Es hilft nicht, dass es mehrere Möglichkeiten gibt, Matrizen in Python darzustellen, wobei die :code:`np.matrix`-Methode zugunsten von :code:`np.ndarray` veraltet ist. In diesem Abschnitt geben wir eine kurze Einführung in Matrizenrechnung in Python mit NumPy, damit du bei den DOA-Beispielen komfortabler bist.

Beginnen wir mit dem lästigsten Teil der Matrizenrechnung in NumPy: Vektoren werden als 1D-Arrays behandelt, sodass es keine Möglichkeit gibt, zwischen einem Zeilen- und einem Spaltenvektor zu unterscheiden (er wird standardmäßig als Zeilenvektor behandelt), während ein Vektor in MATLAB ein 2D-Objekt ist. In Python kannst du einen neuen Vektor mit :code:`a = np.array([2,3,4,5])` erstellen oder eine Liste in einen Vektor umwandeln mit :code:`mylist = [2, 3, 4, 5]` dann :code:`a = np.asarray(mylist)`, aber sobald du Matrizenrechnung durchführen möchtest, spielt die Ausrichtung eine Rolle, und diese werden als Zeilenvektoren interpretiert. Ein Transponieren dieses Vektors, z. B. mit :code:`a.T`, ändert ihn **nicht** in einen Spaltenvektor! Die Möglichkeit, aus einem normalen Vektor :code:`a` einen Spaltenvektor zu machen, ist :code:`a = a.reshape(-1,1)`. Die :code:`-1` weist NumPy an, die Größe dieser Dimension automatisch zu bestimmen, während die zweite Dimension die Länge 1 behält. Was dadurch entsteht, ist technisch gesehen ein 2D-Array, aber die zweite Dimension hat die Länge 1, sodass es aus mathematischer Sicht immer noch im Wesentlichen 1D ist. Es ist nur eine extra Zeile, aber sie kann den Fluss von Matrizenrechencode wirklich stören.

Nun ein kurzes Beispiel für Matrizenrechnung in Python; wir multiplizieren eine :code:`3x10`-Matrix mit einer :code:`10x1`-Matrix. Erinnere daran, dass :code:`10x1` 10 Zeilen und 1 Spalte bedeutet, bekannt als Spaltenvektor, weil es nur eine Spalte ist. Aus unserer frühen Schulzeit wissen wir, dass dies eine gültige Matrizenmultiplikation ist, da die inneren Dimensionen übereinstimmen, und die resultierende Matrixgröße die äußeren Dimensionen sind, also :code:`3x1`. Wir verwenden :code:`np.random.randn()` zur Erstellung der :code:`3x10`-Matrix und :code:`np.arange()` zur Erstellung der :code:`10x1`-Matrix:

.. code-block:: python

 A = np.random.randn(3,10) # 3x10
 B = np.arange(10) # 1D-Array der Länge 10
 B = B.reshape(-1,1) # 10x1
 C = A @ B # Matrizenmultiplikation
 print(C.shape) # 3x1
 C = C.squeeze() # siehe nächsten Abschnitt
 print(C.shape) # 1D-Array der Länge 3, einfacher zum Plotten und anderem Nicht-Matrix-Python-Code

Nach der Matrizenrechnung kann das Ergebnis so aussehen: :code:`[[ 0.  0.125  0.251  -0.376  -0.251 ...]]`, was eindeutig nur eine Datendimension hat, aber wenn du es plotten möchtest, erhältst du entweder einen Fehler oder einen Plot, der nichts anzeigt. Das liegt daran, dass das Ergebnis technisch gesehen ein 2D-Array ist und du es mit :code:`a.squeeze()` in ein 1D-Array umwandeln musst. Die :code:`squeeze()`-Funktion entfernt alle Dimensionen der Länge 1 und ist praktisch bei der Matrizenrechnung in Python. Im oben genannten Beispiel wäre das Ergebnis :code:`[ 0.  0.125  0.251  -0.376  -0.251 ...]` (beachte die fehlenden zweiten Klammern), was geplottet oder in anderem Python-Code verwendet werden kann, der etwas 1D erwartet.

Beim Kodieren von Matrizenrechnung ist die beste Plausibilitätsprüfung, die du durchführen kannst, die Ausgabe der Dimensionen (mit :code:`A.shape`), um zu überprüfen, ob sie deinen Erwartungen entsprechen. Erwäge, die Form in die Kommentare nach jeder Zeile einzufügen, als zukünftige Referenz und um sicherzustellen, dass Dimensionen bei Matrix- oder elementweiser Multiplikation übereinstimmen.

Hier sind einige häufige Operationen in MATLAB und Python als eine Art Spickzettel:

.. list-table::
   :widths: 35 25 40
   :header-rows: 1

   * - Operation
     - MATLAB
     - Python/NumPy
   * - Zeilenvektor erstellen, Größe :code:`1 x 4`
     - :code:`a = [2 3 4 5];`
     - :code:`a = np.array([2,3,4,5])`
   * - Spaltenvektor erstellen, Größe :code:`4 x 1`
     - :code:`a = [2; 3; 4; 5];` oder :code:`a = [2 3 4 5].'`
     - :code:`a = np.array([[2],[3],[4],[5]])` oder |br| :code:`a = np.array([2,3,4,5])` dann |br| :code:`a = a.reshape(-1,1)`
   * - 2D-Matrix erstellen
     - :code:`A = [1 2; 3 4; 5 6];`
     - :code:`A = np.array([[1,2],[3,4],[5,6]])`
   * - Größe bestimmen
     - :code:`size(A)`
     - :code:`A.shape`
   * - Transponieren, auch :math:`A^T`
     - :code:`A.'`
     - :code:`A.T`
   * - Konjugiert komplexes Transponieren |br| auch Hermitesches Transponieren |br| auch :math:`A^H`
     - :code:`A'`
     - :code:`A.conj().T` |br| |br| (leider gibt es kein :code:`A.H` für ndarrays)
   * - Elementweise Multiplikation
     - :code:`A .* B`
     - :code:`A * B` oder :code:`np.multiply(a,b)`
   * - Matrizenmultiplikation
     - :code:`A * B`
     - :code:`A @ B` oder :code:`np.matmul(A,B)`
   * - Skalarprodukt zweier Vektoren (1D)
     - :code:`dot(a,b)`
     - :code:`np.dot(a,b)` (np.dot nie für 2D verwenden)
   * - Verketten
     - :code:`[A A]`
     - :code:`np.concatenate((A,A))`

*********************
Steuervektor
*********************

Um zum interessanten Teil zu kommen, müssen wir etwas Mathematik durcharbeiten, aber der folgende Abschnitt wurde so geschrieben, dass die Mathematik relativ unkompliziert ist und Diagramme dazu gibt; es werden nur grundlegendste Trigonometrie- und Exponentialeigenschaften verwendet. Es ist wichtig, die grundlegende Mathematik hinter dem zu verstehen, was wir in Python zur DOA-Berechnung tun werden.

Betrachte ein eindimensionales, gleichmäßig verteiltes Array mit drei Elementen:

.. image:: ../_images/doa.svg
   :align: center
   :target: ../_images/doa.svg
   :alt: Diagramm zur Ankunftsrichtung (DOA) eines Signals auf einem gleichmäßig verteilten Antennenarray, mit Boresight-Winkel und Abstand zwischen Elementen

In diesem Beispiel kommt ein Signal von der rechten Seite, trifft also zuerst das rechteste Element. Berechnen wir die Verzögerung zwischen dem Zeitpunkt, an dem das Signal dieses erste Element trifft, und wann es das nächste Element erreicht. Dazu formulieren wir das folgende trigonometrische Problem. Das rot markierte Segment stellt die Strecke dar, die das Signal zurücklegen muss, *nachdem* es das erste Element erreicht hat, bevor es das nächste trifft.

.. image:: ../_images/doa_trig.svg
   :align: center
   :target: ../_images/doa_trig.svg
   :alt: Trigonometrie zur Ankunftsrichtung (DOA) eines gleichmäßig verteilten Arrays

Wenn wir SOH CAH TOA anwenden, interessiert uns hier die „Ankathete" und wir haben die Länge der Hypotenuse (:math:`d`), daher benötigen wir einen Kosinus:

.. math::
  \cos(90 - \theta) = \frac{\mathrm{Ankathete}}{\mathrm{Hypotenuse}}

Wir müssen die Ankathete bestimmen, da diese angibt, wie weit das Signal zwischen dem ersten und zweiten Element zurücklegen muss: Ankathete :math:`= d \cos(90 - \theta)`. Eine trigonometrische Identität erlaubt es uns, dies in Ankathete :math:`= d \sin(\theta)` umzuwandeln. Dies ist jedoch nur eine Strecke; wir müssen sie in eine Zeit umrechnen, indem wir die Lichtgeschwindigkeit verwenden: vergangene Zeit :math:`= d \sin(\theta) / c` Sekunden. Diese Gleichung gilt zwischen zwei beliebigen benachbarten Elementen unseres Arrays, obwohl wir das Ganze mit einer ganzen Zahl multiplizieren können, um die Zeit zwischen nicht benachbarten Elementen zu berechnen, da sie gleichmäßig verteilt sind (das machen wir später).

Verbinden wir jetzt diese Trigonometrie und Lichtgeschwindigkeitsmathematik mit der Signalverarbeitungswelt. Sei unser Sendesignal im Basisband :math:`x(t)`, das auf einem Träger :math:`f_c` übertragen wird, sodass das Sendesignal :math:`x(t) e^{2j \pi f_c t}` ist. Wir verwenden :math:`d_m` für den Antennenabstand in Metern. Angenommen, dieses Signal trifft das erste Element zum Zeitpunkt :math:`t = 0`, was bedeutet, dass es das nächste Element nach :math:`d_m \sin(\theta) / c` Sekunden trifft. Damit empfängt das 2. Element:

.. math::
 x(t - \Delta t) e^{2j \pi f_c (t - \Delta t)}

.. math::
 \mathrm{wobei} \quad \Delta t = d_m \sin(\theta) / c

Zur Erinnerung: Eine Zeitverschiebung wird vom Zeitargument subtrahiert.

Wenn der Empfänger oder SDR die Abwärtskonversion durchführt, um das Signal zu empfangen, multipliziert er es im Wesentlichen mit dem Träger in umgekehrter Richtung; nach der Abwärtskonversion sieht der Empfänger:

.. math::
 x(t - \Delta t) e^{2j \pi f_c (t - \Delta t)} e^{-2j \pi f_c t}

.. math::
 = x(t - \Delta t) e^{-2j \pi f_c \Delta t}

Jetzt können wir einen kleinen Trick anwenden, um dies weiter zu vereinfachen: Wenn wir ein Signal abtasten, kann es modelliert werden, indem wir :math:`t` durch :math:`nT` ersetzen, wobei :math:`T` die Abtastperiode und :math:`n` einfach 0, 1, 2, 3... ist. Eingesetzt ergibt das :math:`x(nT - \Delta t) e^{-2j \pi f_c \Delta t}`. Da :math:`nT` so viel größer als :math:`\Delta t` ist, können wir den ersten :math:`\Delta t`-Term weglassen und erhalten :math:`x(nT) e^{-2j \pi f_c \Delta t}`. Falls die Abtastrate jemals schnell genug wird, um sich der Lichtgeschwindigkeit über eine winzige Distanz anzunähern, können wir dies überdenken, aber erinnere daran, dass die Abtastrate nur etwas größer als die Bandbreite des Signals von Interesse sein muss.

Fahren wir mit dieser Mathematik fort, aber beginnen wir nun, Dinge in diskreten Begriffen darzustellen, damit sie unserem Python-Code besser ähneln. Die letzte Gleichung kann wie folgt dargestellt werden; setzen wir :math:`\Delta t` wieder ein:

.. math::
 x[n] e^{-2j \pi f_c \Delta t}

.. math::
 = x[n] e^{-2j \pi f_c d_m \sin(\theta) / c}

Fast fertig, aber glücklicherweise gibt es noch eine weitere Vereinfachung. Erinnere dich an die Beziehung zwischen Mittenfrequenz und Wellenlänge: :math:`\lambda = \frac{c}{f_c}`, oder umgekehrt :math:`f_c = \frac{c}{\lambda}`. Eingesetzt ergibt das:

.. math::
 x[n] e^{-2j \pi d_m \sin(\theta) / \lambda}

Im angewandten Beamforming und DOA stellen wir :math:`d`, den Abstand zwischen benachbarten Elementen, als Bruchteil der Wellenlänge dar (statt in Metern). Der gängigste Wert für :math:`d` beim Array-Design ist die halbe Wellenlänge. Unabhängig vom Wert von :math:`d` stellen wir :math:`d` von nun an als Bruchteil der Wellenlänge dar, was die Gleichungen und unseren Code vereinfacht. D. h. :math:`d` (ohne den Index :math:`m`) steht für den normierten Abstand und ist gleich :math:`d = d_m / \lambda`. Damit können wir die obige Gleichung vereinfachen zu:

.. math::
 x[n] e^{-2j \pi d \sin(\theta)}

Die obige Gleichung gilt für benachbarte Elemente; für das Signal, das das :math:`k`-te Element empfängt, multiplizieren wir :math:`d` einfach mit :math:`k`:

.. math::
 x[n] e^{-2j \pi d k \sin(\theta)}

Betrachten wir nun die Koordinatenkonvention, die wir verwenden wollen. In diesem Lehrbuch stellt 0 Grad tangential zur Array-Platzierung dar (d. h. die Linie, auf der die Elemente liegen), wie im obigen Diagramm gezeigt, und Theta nimmt im Uhrzeigersinn zu. Wir betrachten auch das erste/Referenzelement als das linkste, und jedes weitere Element ist um eine Distanz :math:`d_m` weiter rechts. Dies ist umgekehrt zu unserem obigen Diagramm, daher müssen wir die Richtung der Phasenverschiebung umkehren, was bedeutet, das negative Vorzeichen zu entfernen:

.. math::
 x[n] e^{2j \pi d k \sin(\theta)}

Wir können dies in Matrixform darstellen, indem wir einfach die obige Gleichung für alle :code:`Nr` Elemente im Array von :math:`k = 0, 1, ... , N-1` anordnen:

.. math::

   x
   \begin{bmatrix}
           e^{2j \pi d (0) \sin(\theta)} \\
           e^{2j \pi d (1) \sin(\theta)} \\
           e^{2j \pi d (2) \sin(\theta)} \\
           \vdots \\
           e^{2j \pi d (N_r - 1) \sin(\theta)} \\
    \end{bmatrix}

wobei :math:`x` der 1D-Zeilenvektor mit dem Sendesignal ist und der ausgeschriebene Spaltenvektor der sogenannte „Steuervektor" (oft als :math:`s` bezeichnet und im Code :code:`s`) ist. Weil :math:`e^{0} = 1`, ist das erste Element des Steuervektors immer 1, und der Rest sind Phasenverschiebungen relativ zum ersten Element:

.. math::

   s =
   \begin{bmatrix}
           1 \\
           e^{2j \pi d (1) \sin(\theta)} \\
           e^{2j \pi d (2) \sin(\theta)} \\
           \vdots \\
           e^{2j \pi d (N_r - 1) \sin(\theta)} \\
    \end{bmatrix}

Fertig! Dieser Vektor ist das, was du in DOA-Papieren und ULA-Implementierungen überall sehen wirst! Du wirst ihn auch mit :math:`2\pi\sin(\theta)` ausgedrückt als Symbol wie :math:`\psi` sehen; in diesem Fall wäre der Steuervektor einfach :math:`e^{jd\psi}`, was die allgemeinere Form ist (wir werden diese Form jedoch nicht verwenden). In Python ist :code:`s`:

.. code-block:: python

 s = [np.exp(2j*np.pi*d*0*np.sin(theta)), np.exp(2j*np.pi*d*1*np.sin(theta)), np.exp(2j*np.pi*d*2*np.sin(theta)), ...] # beachte das steigende k
 # oder
 s = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # wobei Nr die Anzahl der Empfangsantennenelemente ist

Beachte, dass Element 0 zu 1+0j führt (weil :math:`e^{0}=1`); das ergibt Sinn, da alles oben relativ zu diesem ersten Element war, sodass es das Signal ohne relative Phasenverschiebungen empfängt. :code:`d` hat die Einheit Wellenlängen, nicht Meter!

*******************
Ein Signal empfangen
*******************

Verwenden wir das Steuervektor-Konzept, um ein auf einem Array eintreffendes Signal zu simulieren. Als Sendesignal verwenden wir zunächst einen Ton:

.. code-block:: python

 import numpy as np
 import matplotlib.pyplot as plt

 sample_rate = 1e6
 N = 10000 # Anzahl der zu simulierenden Samples

 # Erstelle einen Ton als Sendersignal
 t = np.arange(N)/sample_rate # Zeitvektor
 f_tone = 0.02e6
 tx = np.exp(2j * np.pi * f_tone * t)

Simulieren wir nun ein Array aus drei omnidirektionalen Antennen in einer Linie mit Halbwellenlängenabstand. Wir simulieren das Signal des Senders, das bei einem bestimmten Winkel theta an diesem Array ankommt:

.. code-block:: python

 d = 0.5 # halber Wellenlängenabstand
 Nr = 3
 theta_degrees = 20 # Ankunftswinkel (kann frei verändert werden)
 theta = theta_degrees / 180 * np.pi # in Radiant umrechnen
 s = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # Steuervektor
 print(s) # beachte: 3 Elemente lang, komplex, erstes Element ist 1+0j

Um den Steuervektor anzuwenden, müssen wir eine Matrizenmultiplikation von :code:`s` und :code:`tx` durchführen. Zuerst konvertieren wir beide in 2D, indem wir :code:`ourarray.reshape(-1,1)` verwenden:

.. code-block:: python

 s = s.reshape(-1,1) # s als Spaltenvektor
 print(s.shape) # 3x1
 tx = tx.reshape(1,-1) # tx als Zeilenvektor
 print(tx.shape) # 1x10000

 X = s @ tx # Empfangenes Signal X durch Matrizenmultiplikation simulieren
 print(X.shape) # 3x10000. X ist jetzt ein 2D-Array, 1D für Zeit und 1D für die räumliche Dimension

Zu diesem Zeitpunkt ist :code:`X` ein 2D-Array der Größe 3 x 10000, da wir drei Array-Elemente und 10000 simulierte Samples haben. Wir plotten die ersten 200 Samples (nur den Realteil):

.. code-block:: python

 plt.plot(np.asarray(X[0,:]).squeeze().real[0:200])
 plt.plot(np.asarray(X[1,:]).squeeze().real[0:200])
 plt.plot(np.asarray(X[2,:]).squeeze().real[0:200])
 plt.show()

.. image:: ../_images/doa_time_domain.svg
   :align: center
   :target: ../_images/doa_time_domain.svg

Beachte die Phasenverschiebungen zwischen den Elementen, wie wir es erwartet haben (außer wenn das Signal in Hauptstrahlrichtung (Boresight) ankommt; setze theta auf 0, um dies zu sehen). Versuche, den Winkel zu ändern und schau, was passiert.

Als letzten Schritt fügen wir dem empfangenen Signal Rauschen hinzu. Das Rauschen wird nach der Steuervektor-Anwendung hinzugefügt, da jedes Element ein unabhängiges Rauschsignal erfährt:

.. code-block:: python

 n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
 X = X + 0.1*n # X und n sind beide 3x10000

.. image:: ../_images/doa_time_domain_with_noise.svg
   :align: center
   :target: ../_images/doa_time_domain_with_noise.svg

******************************
Konventionelles Beamforming & DOA
******************************

Wir werden nun diese Samples :code:`X` verarbeiten und so tun, als ob wir den Ankunftswinkel nicht kennen, und DOA durchführen, d. h. den/die Ankunftswinkel mit DSP und Python schätzen! Wie zu Beginn dieses Kapitels besprochen, sind Beamforming und DOA sehr ähnlich und bauen oft auf denselben Techniken auf. Im Rest dieses Kapitels untersuchen wir verschiedene „Beamformer" und beginnen jeweils mit der Beamformer-Mathematik/dem -Code zur Berechnung der Gewichte :math:`w`. Diese Gewichte können über die einfache Gleichung :math:`w^H X` oder in Python :code:`w.conj().T @ X` auf das eingehende Signal :code:`X` angewendet werden. Im obigen Beispiel ist :code:`X` eine :code:`3x10000`-Matrix, aber nach Anwenden der Gewichte verbleiben wir mit :code:`1x10000`, als hätte unser Empfänger nur eine Antenne.

Wir beginnen mit dem „konventionellen" Beamforming-Ansatz, auch Delay-and-Sum-Beamforming genannt. Unser Gewichtsvektor :code:`w` muss ein 1D-Array für ein ULA sein; in unserem Beispiel mit drei Elementen ist :code:`w` ein :code:`3x1`-Array komplexer Gewichte. Beim konventionellen Beamforming lassen wir die Magnitude der Gewichte bei 1 und passen die Phasen so an, dass sich das Signal in der Richtung unseres gewünschten Signals konstruktiv addiert – das ist genau dieselbe Mathematik wie oben, d. h. unsere Gewichte sind unser Steuervektor!

.. math::
 w_{conv} = e^{2j \pi d k \sin(\theta)}

oder in Python:

.. code-block:: python

 w = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # konventioneller Beamformer (Delay-and-Sum)
 X_weighted = w.conj().T @ X # Gewichte auf empfangenes Signal anwenden
 print(X_weighted.shape) # 1x10000

Aber wie kennen wir den Interessenwinkel :code:`theta`? Wir müssen zunächst DOA durchführen, was das Abtasten aller Ankunftsrichtungen von -π bis +π (-180 bis +180 Grad) beinhaltet, z. B. in 1-Grad-Schritten. Bei jeder Richtung berechnen wir die Gewichte und berechnen dann die Signalleistung. Wir plotten die Ergebnisse und suchen nach Peaks.

.. code-block:: python

 theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # 1000 verschiedene Thetas zwischen -180 und +180 Grad
 results = []
 for theta_i in theta_scan:
    w = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta_i)) # konventioneller Beamformer
    X_weighted = w.conj().T @ X # Gewichte anwenden
    results.append(10*np.log10(np.var(X_weighted))) # Signalleistung in dB
 results -= np.max(results) # normalisieren (optional)

 # Winkel mit maximalem Wert ausgeben
 print(theta_scan[np.argmax(results)] * 180 / np.pi) # 19.99999999999998

 plt.plot(theta_scan*180/np.pi, results) # Winkel in Grad plotten
 plt.xlabel("Theta [Grad]")
 plt.ylabel("DOA-Metrik")
 plt.grid()
 plt.show()

.. image:: ../_images/doa_conventional_beamformer.svg
   :align: center
   :target: ../_images/doa_conventional_beamformer.svg

Wir haben unser Signal gefunden! Für einen Polarplot der DOA-Ergebnisse:

.. code-block:: python

 fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
 ax.plot(theta_scan, results) # UNBEDINGT RADIANT FÜR POLAR VERWENDEN
 ax.set_theta_zero_location('N') # 0 Grad nach oben
 ax.set_theta_direction(-1) # im Uhrzeigersinn zunehmen
 ax.set_rlabel_position(55)
 plt.show()

.. image:: ../_images/doa_conventional_beamformer_polar.svg
   :align: center
   :target: ../_images/doa_conventional_beamformer_polar.svg
   :alt: Beispiel-Polarplot der DOA mit Strahlmuster und 180-Grad-Mehrdeutigkeit

********************
180-Grad-Mehrdeutigkeit
********************

Erklären wir, warum es einen zweiten Peak bei 160 Grad gibt; die simulierte DOA war 20 Grad, aber es ist kein Zufall, dass 180 - 20 = 160. Stell dir drei omnidirektionale Antennen in einer Linie vor. Das Array sieht denselben Effekt, unabhängig davon, ob das Signal von vorne oder von hinten ankommt – die Phasenverzögerung ist gleich. Deshalb gibt es bei der DOA-Berechnung immer eine solche 180-Grad-Mehrdeutigkeit; der einzige Ausweg ist ein 2D-Array oder ein zweites 1D-Array in einem anderen Winkel.

.. image:: ../_images/doa_from_behind.svg
   :align: center
   :target: ../_images/doa_from_behind.svg

Wenn der Ankunftswinkel (AoA) sich dem „Endfire" des Arrays nähert (d. h. wenn das Signal entlang der Array-Achse ankommt), nimmt die Leistung ab: Die Hauptkeule wird breiter und es entsteht Mehrdeutigkeit zwischen links und rechts. Von diesem Punkt an zeigen wir in unseren Polarplots nur noch -90 bis +90 Grad, da das Muster für 1D-lineare Arrays immer entlang der Array-Achse gespiegelt wird.

********************
Strahlmuster
********************

Die bisherigen Plots sind DOA-Ergebnisse – sie entsprechen der empfangenen Leistung bei jedem Winkel nach Anwenden des Beamformers. Wir können aber auch das Strahlmuster selbst betrachten, bevor wir ein Signal empfangen; dies wird manchmal als „quiescent antenna pattern" oder „Array-Antwort" bezeichnet.

Unser Steuervektor

.. code-block:: python

 np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta))

kapselt die ULA-Geometrie und hat als einzigen weiteren Parameter die gewünschte Steuerrichtung. Wir können das quieszente Antennendiagramm berechnen und plotten, wenn in eine bestimmte Richtung gelenkt wird. Dies kann durch die FFT der komplex-konjugierten Gewichte erledigt werden:

.. code-block:: python

    Nr = 3
    d = 0.5
    N_fft = 512
    theta_degrees = 20 # nur die Richtung, auf die wir zeigen wollen
    theta = theta_degrees / 180 * np.pi
    w = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # konventioneller Beamformer
    w_padded = np.concatenate((w, np.zeros(N_fft - Nr))) # mit Nullen auf N_fft Elemente auffüllen
    w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) # Betrag der FFT in dB
    w_fft_dB -= np.max(w_fft_dB) # auf 0 dB am Peak normalisieren

    # FFT-Bins auf Winkel in Radiant abbilden
    theta_bins = np.arcsin(np.linspace(-1, 1, N_fft))

    theta_max = theta_bins[np.argmax(w_fft_dB)]

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta_bins, w_fft_dB)
    ax.plot([theta_max], [np.max(w_fft_dB)],'ro')
    ax.text(theta_max - 0.1, np.max(w_fft_dB) - 4, np.round(theta_max * 180 / np.pi))
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(55)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_ylim([-30, 1])
    plt.show()

.. image:: ../_images/doa_quiescent.svg
   :align: center
   :target: ../_images/doa_quiescent.svg

Die folgende Animation zeigt das Strahlmuster des konventionellen Beamformers für ein 8-Element-Array, das zwischen -90 und +90 Grad gesteuert wird:

.. image:: ../_images/delay_and_sum.gif
   :scale: 90 %
   :align: center
   :alt: Strahlmuster von Delay-and-Sum mit Gewichten in der komplexen Ebene

Beachte, wie alle Gewichte Einheitsamplitude haben (sie bleiben auf dem Einheitskreis) und wie die höher nummerierten Elemente schneller „rotieren".

********************
Array-Strahlbreite
********************

Für Interessierte gibt es Gleichungen zur Näherung der Hauptkeulenbreite in Abhängigkeit von der Elementanzahl (sie funktionieren gut bei vielen Elementen, z. B. 8 oder mehr). Die Halbwertbreite (HPBW) ist 3 dB unterhalb des Hauptkeulengipfels und beträgt ungefähr :math:`\frac{0.9 \lambda}{N_rd\cos(\theta)}` [1], was beim Halbwellenlängenabstand vereinfacht zu:

.. math::

 \text{HPBW} \approx \frac{1.8}{N_r\cos(\theta)} \text{ [Radiant]} \qquad \text{wenn } d = \lambda/2

Die Breite der ersten Nullstellen (FNBW), die Breite der Hauptkeule von Null zu Null, beträgt ungefähr :math:`\frac{2\lambda}{N_rd}` [1], was beim Halbwellenlängenabstand vereinfacht zu:

.. math::

 \text{FNBW} \approx \frac{4}{N_r} \text{ [Radiant]} \qquad \text{wenn } d = \lambda/2

.. image:: ../_images/doa_quiescent_beamwidth.svg
   :align: center
   :target: ../_images/doa_quiescent_beamwidth.svg

*******************
Wenn d nicht λ/2 ist
*******************

Bisher haben wir einen Elementabstand d gleich einer halben Wellenlänge verwendet. Wenn d größer als λ/2 ist, entstehen sogenannte Gitterkeulen (Grating Lobes) – ein Ergebnis von „räumlichem Aliasing". Wie wir im Kapitel :ref:`sampling-chapter` gelernt haben, entsteht Aliasing, wenn wir nicht schnell genug abtasten. Dasselbe gilt im räumlichen Bereich: Wenn die Elemente nicht nahe genug beieinander liegen, erhalten wir fehlerhafte Ergebnisse.

.. image:: ../_images/doa_d_is_large_animation.gif
   :scale: 100 %
   :align: center
   :alt: Animation der DOA bei d >> λ/2

Das Nyquist-Kriterium gilt auch für den räumlichen Bereich: :math:`d \leq \lambda/2`. Solange dieser Abstand eingehalten wird, gibt es keine Gitterkeulen!

Was passiert, wenn d kleiner als λ/2 ist?

.. image:: ../_images/doa_d_is_small_animation.gif
   :scale: 100 %
   :align: center
   :alt: Animation der DOA bei d << λ/2

Die Hauptkeule wird breiter, aber es gibt keine Gitterkeulen. Mit einem zweiten Signal von -40 Grad:

.. image:: ../_images/doa_d_is_small_animation2.gif
   :scale: 100 %
   :align: center
   :alt: Animation der DOA bei d << λ/2 mit zwei Signalen

Unterhalb von λ/4 lassen sich die beiden Wege nicht mehr unterscheiden. d so nah wie möglich bei λ/2 zu halten bleibt ein wichtiges Thema.

**********************
Räumliche Fensterung
**********************

Räumliche Fensterung (Spatial Tapering) ist eine Technik, die zusammen mit dem konventionellen Beamformer verwendet wird, wobei die Magnitude der Gewichte angepasst wird, um bestimmte Eigenschaften zu erzielen. Beim konventionellen Beamformer haben alle Gewichte eine Magnitude von 1 (Einheitsamplitude). Mit räumlicher Fensterung multiplizieren wir die Gewichte mit Skalaren, um ihre Magnitude zu skalieren.

.. code-block:: python

    tapering = np.random.uniform(0, 1, Nr) # zufällige Fensterung
    w *= tapering

.. image:: ../_images/spatial_tapering_animation.gif
   :scale: 80 %
   :align: center
   :alt: Räumliche Fensterung mit zufälligen Werten

Fensterung kann die Nebenkeulen reduzieren, indem die Magnitude der Gewichte an den **Rändern** des Arrays reduziert wird. Zum Beispiel kann eine Hamming-Fensterfunktion als Fensterungswerte verwendet werden:

.. code-block:: python

    tapering = np.hamming(Nr) # Hamming-Fensterfunktion
    w *= tapering

.. image:: ../_images/spatial_tapering_animation2.gif
   :scale: 80 %
   :align: center
   :alt: Räumliche Fensterung mit Hamming-Fenster

Wir bemerken zwei Änderungen: Erstens wird die Hauptkeulenbreite von der Fensterfunktion beeinflusst (weniger Nebenkeulen führen typischerweise zu einer breiteren Hauptkeule). Ein rechteckiges Fenster (keine Fensterung) führt zur schmalsten Hauptkeule, aber zu den höchsten Nebenkeulen. Zweitens nimmt der Gewinn der Hauptkeule ab, wenn wir ein Fenster anwenden, da wir letztendlich weniger Signalenergie empfangen.

******************************
Gewichte manuell anpassen
******************************

Der konventionelle Beamformer liefert eine Gleichung zur Berechnung der Gewichte, aber für einen Moment wollen wir so tun, als hätten wir keine Methode zur Gewichtsberechnung, und stattdessen mit den Gewichten (sowohl Magnitude als auch Phase) manuell spielen, um zu sehen, was passiert. Unten ist eine kleine App in JavaScript, die das Strahlmuster eines 8-Element-Arrays simuliert, mit Schiebereglern zur Steuerung von Verstärkung und Phase jedes Elements:

.. raw:: html

    <div id="rectPlot"><!-- Plotly chart will be drawn inside this DIV --></div>
    <br />
    Element &nbsp;&nbsp;&nbsp; Magnitude (Gain) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Phase
    <div id="sliders"></div>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <script>
    beamforming_slider_app()
    </script>

*********************
Adaptives Beamforming
*********************

Der konventionelle Beamformer ist eine einfache und effektive Methode für Beamforming, hat aber einige Einschränkungen. Zum Beispiel funktioniert er nicht gut, wenn mehrere Signale aus verschiedenen Richtungen ankommen oder wenn der Rauschpegel hoch ist. In diesen Fällen müssen wir fortschrittlichere Beamforming-Techniken verwenden, die oft als „adaptives" Beamforming bezeichnet werden. Die Idee hinter adaptivem Beamforming ist es, das empfangene Signal zur Berechnung der Gewichte zu verwenden, anstatt einen festen Satz von Gewichten wie beim konventionellen Beamformer zu verwenden. Dies ermöglicht es dem Beamformer, sich an die Umgebung anzupassen und eine bessere Leistung zu erbringen.

Adaptive Beamforming-Techniken können weiter in reguläre und unterraumbasierte Methoden unterteilt werden. Unterraummethoden wie MUSIC und ESPRIT sind sehr leistungsfähig, erfordern aber eine Schätzung der Anzahl vorhandener Signale, und sie benötigen mindestens drei Elemente (mindestens vier werden empfohlen).

Die erste adaptive Beamforming-Technik, die wir untersuchen werden, ist MVDR.

**********************
MVDR/Capon-Beamformer
**********************

Wir betrachten jetzt einen Beamformer, der etwas komplizierter als die konventionelle Delay-and-Sum-Technik ist, aber tendenziell viel besser abschneidet: den Minimum Variance Distortionless Response (MVDR)- oder Capon-Beamformer. Die Idee hinter MVDR ist es, das Signal aus dem Interessenwinkel bei einer festen Verstärkung von 1 (0 dB) zu halten, während die Gesamtleistung des resultierenden Beamformer-Signals minimiert wird. Er wird oft als „statistisch optimaler" Beamformer bezeichnet.

Der MVDR/Capon-Beamformer lässt sich in folgender Gleichung zusammenfassen:

.. math::

 w_{mvdr} = \frac{R^{-1} s}{s^H R^{-1} s}

Der Vektor :math:`s` ist der Steuervektor entsprechend der gewünschten Richtung. :math:`R` ist die räumliche Kovarianzmatrixschätzung auf Basis unserer empfangenen Samples, berechnet mit :code:`R = np.cov(X)` oder manuell durch :math:`R = X X^H`. Die räumliche Kovarianzmatrix ist eine :code:`Nr` x :code:`Nr`-Matrix (3x3 in den bisherigen Beispielen), die angibt, wie ähnlich die von den drei Elementen empfangenen Samples sind. Der Nenner dient hauptsächlich der Skalierung; der Zähler ist der wichtige Teil – die invertierte Kovarianzmatrix multipliziert mit dem Steuervektor.

.. raw:: html

   <details>
   <summary>Für Interessierte: MVDR-Herleitung hier aufklappen</summary>


**Beamformer-Ausgang** – Der Ausgang des Beamformers mit dem Gewichtsvektor :math:`\mathbf{w}` ist:

.. math::

 y(t) = \mathbf{w}^H \mathbf{x}(t)


**Optimierungsproblem** – Das Ziel ist es, die Beamforming-Gewichte zu bestimmen, die die Ausgangsleistung unter der Nebenbedingung einer verzerrungsfreien Antwort in die gewünschte Richtung :math:`\theta_0` minimieren:

.. math::

 \min_{\mathbf{w}} \, \mathbf{w}^H \mathbf{R} \mathbf{w} \quad \text{s. t.} \quad \mathbf{w}^H \mathbf{s} = 1

wobei:

* :math:`\mathbf{R} = E[\mathbf{X}\mathbf{X}^H]` die Kovarianzmatrix der empfangenen Signale ist
* :math:`\mathbf{s}` der Steuervektor in Richtung des gewünschten Signals :math:`\theta_0` ist

**Lagrange-Methode** – Einführung eines Lagrange-Multiplikators :math:`\lambda`:

.. math::

 L(\mathbf{w}, \lambda) = \mathbf{w}^H \mathbf{R} \mathbf{w} - \lambda (\mathbf{w}^H \mathbf{s} - 1)

**Lösung der Optimierung** – Durch Differenzierung nach :math:`\mathbf{w^H}` und Nullsetzen ergibt sich:

.. math::

 \frac{\partial L}{\partial \mathbf{w}^*} = 2\mathbf{R}\mathbf{w} - \lambda \mathbf{s} = 0

 \mathbf{w} = \lambda \mathbf{s} \mathbf{{R^{-1}}}


Zur Lösung für :math:`\lambda` mit der Nebenbedingung :math:`\mathbf{w}^H \mathbf{s} = 1`:

.. math::

 \implies (\lambda \mathbf{s^{H}}\mathbf{{R^{-1}}})s = 1

 \implies \lambda = \frac{1}{\mathbf{s}^{H}\mathbf{R}^{-1}\mathbf{s}}

 \mathbf{R}\mathbf{w} = \lambda \mathbf{s}

 \mathbf{w_{mvdr}} = \frac{\mathbf{R}^{-1} \mathbf{s}}{\mathbf{s}^H \mathbf{R}^{-1} \mathbf{s}}

.. raw:: html

   </details>

In Python implementieren wir den MVDR/Capon-Beamformer wie folgt:

.. code-block:: python

 # theta ist die Interessenrichtung in Radiant, X ist unser empfangenes Signal
 def w_mvdr(theta, X):
    s = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # Steuervektor
    s = s.reshape(-1,1) # als Spaltenvektor (3x1)
    R = (X @ X.conj().T)/X.shape[1] # Kovarianzmatrix berechnen (Nr x Nr)
    Rinv = np.linalg.pinv(R) # Pseudo-Inverse ist stabiler als echte Inverse
    w = (Rinv @ s)/(s.conj().T @ Rinv @ s) # MVDR/Capon-Gleichung
    return w

Verwendung dieses MVDR-Beamformers für DOA:

.. code-block:: python

 theta_scan = np.linspace(-1*np.pi, np.pi, 1000)
 results = []
 for theta_i in theta_scan:
    w = w_mvdr(theta_i, X) # 3x1
    X_weighted = w.conj().T @ X # Gewichte anwenden
    power_dB = 10*np.log10(np.var(X_weighted))
    results.append(power_dB)
 results -= np.max(results) # normalisieren

Für ein komplexeres Szenario mit einem 8-Element-Array, das drei Signale aus verschiedenen Winkeln empfängt (20, 25 und 40 Grad, wobei das letzte viel schwächer ist):

.. code-block:: python

 Nr = 8 # 8 Elemente
 theta1 = 20 / 180 * np.pi
 theta2 = 25 / 180 * np.pi
 theta3 = -40 / 180 * np.pi
 s1 = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta1)).reshape(-1,1) # 8x1
 s2 = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta2)).reshape(-1,1)
 s3 = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta3)).reshape(-1,1)
 # verschiedene Frequenzen verwenden. 1xN
 tone1 = np.exp(2j*np.pi*0.01e6*t).reshape(1,-1)
 tone2 = np.exp(2j*np.pi*0.02e6*t).reshape(1,-1)
 tone3 = np.exp(2j*np.pi*0.03e6*t).reshape(1,-1)
 X = s1 @ tone1 + s2 @ tone2 + 0.1 * s3 @ tone3 # letztes Signal hat 1/10 der Leistung
 n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
 X = X + 0.05*n # 8xN

MVDR-Ergebnis für dieses Szenario:

.. image:: ../_images/doa_capons2.svg
   :align: center
   :target: ../_images/doa_capons2.svg

Alle drei Signale sind erkennbar – auch die beiden nur 5 Grad voneinander entfernten und das schwächere bei -40 Grad. Zum Vergleich das konventionelle Beamforming-Ergebnis:

.. image:: ../_images/doa_complex_scenario.svg
   :align: center
   :target: ../_images/doa_complex_scenario.svg

Das konventionelle Beamforming findet nicht alle drei Signale – das zeigt den Vorteil eines adaptiven Beamformers.

Als Optimierung kann für die DOA-Anwendung von MVDR die Leistungsberechnung ohne explizite Gewichtsanwendung erfolgen:

.. math::

   P_{mvdr} = \frac{1}{s^H R^{-1} s}

.. code-block:: python

    def power_mvdr(theta, X):
        s = np.exp(2j * np.pi * d * np.arange(X.shape[0]) * np.sin(theta))
        s = s.reshape(-1,1)
        R = (X @ X.conj().T)/X.shape[1]
        Rinv = np.linalg.pinv(R)
        return 1/(s.conj().T @ Rinv @ s).squeeze()

**********************
Kovarianzmatrix
**********************

Nehmen wir uns kurz Zeit, die räumliche Kovarianzmatrix zu besprechen, ein Schlüsselkonzept im adaptiven Beamforming. Eine Kovarianzmatrix ist eine mathematische Darstellung der Ähnlichkeit zwischen Paaren von Elementen in einem zufälligen Vektor (in unserem Fall die Elemente unseres Arrays). Eine Kovarianzmatrix ist immer quadratisch, und die Werte entlang der Diagonale entsprechen der Kovarianz jedes Elements mit sich selbst.

Im Allgemeinen ist die Kovarianzmatrix definiert als:

:math:`\mathrm{cov}(X) = E \left[ (X - E[X])(X - E[X])^H \right]`

Für drahtlose Signale im Basisband ist :math:`E[X]` typischerweise null oder sehr nahe null, sodass sich dies vereinfacht zu:

:math:`\mathrm{cov}(X) = E[X X^H]`

Mit einer begrenzten Anzahl von IQ-Samples können wir diese Kovarianz schätzen als :math:`\hat{R}`:

.. math::

 \hat{R} = \frac{\boldsymbol{X} \boldsymbol{X}^H}{N}

         = \frac{1}{N} \sum^N_{n=1} X_n X_n^H

In Python: :code:`R = (X @ X.conj().T)/X.shape[1]` oder alternativ :code:`R = np.cov(X)`.

Die Diagonalelemente sind reell und ungefähr gleich; sie geben im Wesentlichen die empfangene Signalleistung an jedem Element an. Die Nicht-Diagonalelemente sind die wichtigen Werte. Die Inverse der räumlichen Kovarianzmatrix (auch „Precision Matrix" oder in der Radartechnik „Whitening Matrix" genannt) gibt an, wie zwei Elemente miteinander verwandt sind, nachdem der Einfluss anderer Elemente entfernt wurde.

**********************
LCMV-Beamformer
**********************

Was, wenn wir mehr als ein Nutzsignal (SOI) haben? Mit einer kleinen Anpassung an MVDR implementieren wir den Linearly Constrained Minimum Variance (LCMV)-Beamformer, der mehrere SOIs verarbeiten kann. Der optimale Gewichtsvektor für den LCMV-Beamformer lautet:

.. math::

   w_{lcmv} = R^{-1} C [C^H R^{-1} C]^{-1} f

wobei :math:`C` eine Matrix aus den Steuervektoren der entsprechenden SOIs und Störsender ist, und :math:`f` der gewünschte Antwortvektor ist. :math:`f` nimmt den Wert 0 an, wenn der entsprechende Steuervektor unterdrückt werden soll, und 1, wenn ein Strahl darauf gerichtet werden soll. Für zwei SOIs und zwei Störsender z. B. :code:`f = [1,1,0,0]`. Die Gesamtzahl der gleichzeitig formbaren Nullstellen und Strahlen ist durch die Arraygröße (Elementanzahl) begrenzt.

Implementierung in Python für zwei SOIs (15 und 60 Grad) ohne explizit hardcodierte Nullstellen (MVDR übernimmt das automatisch anhand der Statistik):

.. code-block:: python

    # SOI bei 15 Grad, weiteres mögliches SOI bei 60 Grad
    soi1_theta = 15 / 180 * np.pi
    soi2_theta = 60 / 180 * np.pi

    # LCMV-Gewichte
    R_inv = np.linalg.pinv(np.cov(X)) # 8x8
    s1 = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(soi1_theta)).reshape(-1,1) # 8x1
    s2 = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(soi2_theta)).reshape(-1,1) # 8x1
    C = np.concatenate((s1, s2), axis=1) # 8x2
    f = np.ones(2).reshape(-1,1) # 2x1

    # LCMV-Gleichung
    #    8x8   8x2                    2x8        8x8   8x2  2x1
    w = R_inv @ C @ np.linalg.pinv(C.conj().T @ R_inv @ C) @ f # Ausgabe ist 8x1

.. image:: ../_images/lcmv_beam_pattern.svg
   :align: center
   :target: ../_images/lcmv_beam_pattern.svg
   :alt: Beispiel-Strahlmuster mit LCMV-Beamformer

Wir haben Strahlen in die zwei Interessenrichtungen und Nullstellen bei den Störsendern (grüne und rote Punkte zeigen SOI- und Störsender-AoAs).

.. raw:: html

   <details>
   <summary>Vollständigen Code aufklappen</summary>

.. code-block:: python

    # Empfangenes Signal simulieren
    Nr = 8 # 8 Elemente
    theta1 = -60 / 180 * np.pi
    theta2 = -30 / 180 * np.pi
    theta3 = 0 / 180 * np.pi
    theta4 = 30 / 180 * np.pi
    s1 = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta1)).reshape(-1,1) # 8x1
    s2 = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta2)).reshape(-1,1)
    s3 = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta3)).reshape(-1,1)
    s4 = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta4)).reshape(-1,1)
    # verschiedene Frequenzen. 1xN
    tone1 = np.exp(2j*np.pi*0.01e6*t).reshape(1,-1)
    tone2 = np.exp(2j*np.pi*0.02e6*t).reshape(1,-1)
    tone3 = np.exp(2j*np.pi*0.03e6*t).reshape(1,-1)
    tone4 = np.exp(2j*np.pi*0.04e6*t).reshape(1,-1)
    X = s1 @ tone1 + s2 @ tone2 + s3 @ tone3 + s4 @ tone4
    n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
    X = X + 0.5*n # 8xN

    # SOI bei 15 Grad, weiteres mögliches SOI bei 60 Grad
    soi1_theta = 15 / 180 * np.pi
    soi2_theta = 60 / 180 * np.pi

    # LCMV-Gewichte
    R_inv = np.linalg.pinv(np.cov(X)) # 8x8
    s1 = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(soi1_theta)).reshape(-1,1) # 8x1
    s2 = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(soi2_theta)).reshape(-1,1) # 8x1
    C = np.concatenate((s1, s2), axis=1) # 8x2
    f = np.ones(2).reshape(-1,1) # 2x1

    # LCMV-Gleichung
    w = R_inv @ C @ np.linalg.pinv(C.conj().T @ R_inv @ C) @ f # Ausgabe ist 8x1

    # Strahlmuster plotten
    w = w.squeeze()
    N_fft = 1024
    w_padded = np.concatenate((w, np.zeros(N_fft - Nr)))
    w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2)
    w_fft_dB -= np.max(w_fft_dB)
    theta_bins = np.arcsin(np.linspace(-1, 1, N_fft))

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta_bins, w_fft_dB)
    ax.plot([theta1], [0], 'or')
    ax.plot([theta2], [0], 'or')
    ax.plot([theta3], [0], 'or')
    ax.plot([theta4], [0], 'or')
    ax.plot([soi1_theta], [0], 'og')
    ax.plot([soi2_theta], [0], 'og')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(-90, 105, 15))
    ax.set_rlabel_position(55)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_ylim([-30, 1])
    plt.show()

.. raw:: html

   </details>

Ein besonderer Anwendungsfall von LCMV ist die Erstellung breiterer Strahlen oder Nullstellen, indem :code:`f` für einen Bereich von Winkeln auf 1 oder 0 gesetzt wird. Beispiel mit einem 18-Element-Array:

.. code-block:: python

    Nr = 18
    X = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N) # nur Rauschen

    # SOI von 15 bis 30 Grad mit 4 Winkeln
    soi_thetas = np.linspace(15, 30, 4) / 180 * np.pi

    # Nullstelle von 45 bis 60 Grad mit 4 Winkeln
    null_thetas = np.linspace(45, 60, 4) / 180 * np.pi

    # LCMV-Gewichte
    R_inv = np.linalg.pinv(np.cov(X))
    s = []
    for soi_theta in soi_thetas:
        s.append(np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(soi_theta)).reshape(-1,1))
    for null_theta in null_thetas:
        s.append(np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(null_theta)).reshape(-1,1))
    C = np.concatenate(s, axis=1)
    f = np.asarray([1]*len(soi_thetas) + [0]*len(null_thetas)).reshape(-1,1)
    w = R_inv @ C @ np.linalg.pinv(C.conj().T @ R_inv @ C) @ f # LCMV-Gleichung

.. image:: ../_images/lcmv_beam_pattern_spread.svg
   :align: center
   :target: ../_images/lcmv_beam_pattern_spread.svg
   :alt: Beispiel-Strahlmuster mit LCMV und gespreiztem Strahl und Nullstelle

*******************
Nullsteuerung
*******************

Jetzt ist es sinnvoll, eine einfachere Technik zu untersuchen, die in analogen und digitalen Arrays eingesetzt werden kann: die Nullsteuerung (Null Steering). Sie ist wie eine Erweiterung des konventionellen Beamformers; zusätzlich zum Zeigen eines Strahls in die Interessenrichtung können wir Nullstellen bei bestimmten Winkeln platzieren. Diese Technik verwendet keine Gewichte basierend auf dem empfangenen Signal und wird daher nicht als adaptiv betrachtet.

Die Gewichte für die Nullsteuerung werden berechnet, indem mit dem konventionellen Beamformer begonnen wird und dann die Sidelobe-Canceler-Gleichung angewendet wird, um Nullstellen hinzuzufügen:

.. math::

 w_{\text{neu}} = w_{\text{orig}} - \frac{w_{\text{null}}^H w_{\text{orig}}}{w_{\text{null}}^H w_{\text{null}}} w_{\text{null}}

wobei :math:`w_{\text{null}}` der Steuervektor in Richtung der zu erzeugenden Nullstelle ist. Der vollständige Prozess:

.. math::

 \text{1:} \qquad w_{\text{orig}} = e^{2j \pi d k \sin(\theta_{SOI})} \qquad

 \text{2:} \qquad w_{\text{null}} = e^{2j \pi d k \sin(\theta_{null})} \qquad

 \text{3:} \qquad w_{\text{neu}} = w_{\text{orig}} - \frac{w_{\text{null}}^H w_{\text{orig}}}{w_{\text{null}}^H w_{\text{null}}} w_{\text{null}}

 \text{4:} \qquad w_{\text{orig}} = w_{\text{neu}} \qquad \qquad \qquad

 \text{5:} \qquad \text{GOTO 2 für nächste Nullstelle}

Simulieren wir ein 8-Element-Array mit vier Nullstellen:

.. code-block:: python

    d = 0.5
    Nr = 8

    theta_soi = 30 / 180 * np.pi
    nulls_deg = [-60, -30, 0, 60] # Grad
    nulls_rad = np.asarray(nulls_deg) / 180 * np.pi

    # Mit konventionellem Beamformer auf theta_soi beginnen
    w = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta_soi)).reshape(-1,1)

    # Nullstellen durchlaufen
    for null_rad in nulls_rad:
        # Gewichte gleich Steuervektor in Nullstellenrichtung
        w_null = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(null_rad)).reshape(-1,1)

        # Skalierungsfaktor für w in der Nullstellenrichtung
        scaling_factor = w_null.conj().T @ w / (w_null.conj().T @ w_null)
        print("scaling_factor:", scaling_factor, scaling_factor.shape)

        # Gewichte aktualisieren
        w = w - w_null @ scaling_factor # Sidelobe-Canceler-Gleichung

    # Strahlmuster plotten
    N_fft = 1024
    w_padded = np.concatenate((w.squeeze(), np.zeros(N_fft - Nr)))
    w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2)
    w_fft_dB -= np.max(w_fft_dB)
    theta_bins = np.arcsin(np.linspace(-1, 1, N_fft))

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta_bins, w_fft_dB)
    for null_rad in nulls_rad:
        ax.plot([null_rad], [0], 'or')
    ax.plot([theta_soi], [0], 'og')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(-90, 105, 15))
    ax.set_rlabel_position(55)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_ylim([-40, 1])
    plt.show()

.. image:: ../_images/null_steering.svg
   :align: center
   :target: ../_images/null_steering.svg
   :alt: Beispiel für Nullsteuerung beim Beamforming

*******************
MUSIC
*******************

Wir wechseln nun zu einer anderen Art von Beamformer. Alle bisherigen fielen in die Kategorie „Delay-and-Sum", aber jetzt tauchen wir in „Unterraum"-Methoden ein. Diese beinhalten die Aufteilung in Signal- und Rauschunterraum, was bedeutet, dass wir die Anzahl der empfangenen Signale schätzen müssen. MUltiple SIgnal Classification (MUSIC) ist eine sehr beliebte Unterraummethode, die die Berechnung der Eigenvektoren der Kovarianzmatrix beinhaltet. Wir teilen die Eigenvektoren in zwei Gruppen auf: Signal-Unterraum und Rausch-Unterraum, projizieren dann Steuervektoren in den Rausch-Unterraum und suchen nach Nullstellen.

Die Kern-MUSIC-Gleichung lautet:

.. math::
 \hat{\theta} = \mathrm{argmax}\left(\frac{1}{s^H V_n V^H_n s}\right)

wobei :math:`V_n` die Liste der Rausch-Unterraum-Eigenvektoren ist. Sie wird durch Berechnung der Eigenvektoren von :math:`R` gefunden (in Python: :code:`w, v = np.linalg.eig(R)`), dann werden die Vektoren basierend auf der geschätzten Anzahl der Signale aufgeteilt. :math:`V_n` hängt nicht vom Steuervektor :math:`s` ab, daher kann es vorberechnet werden. Der vollständige MUSIC-Code:

.. code-block:: python

 num_expected_signals = 3 # Anzahl erwarteter Signale (kann verändert werden!)

 # Teil, der sich mit theta_i nicht ändert
 R = np.cov(X) # Kovarianzmatrix (Nr x Nr)
 w, v = np.linalg.eig(R) # Eigenzerlegung
 eig_val_order = np.argsort(np.abs(w)) # Reihenfolge der Eigenwertbeträge
 v = v[:, eig_val_order] # Eigenvektoren nach Magnitude sortieren
 # Neue Eigenvektor-Matrix für den Rauschunterraum
 V = np.zeros((Nr, Nr - num_expected_signals), dtype=np.complex64)
 for i in range(Nr - num_expected_signals):
    V[:, i] = v[:, i]

 theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # -180 bis +180 Grad
 results = []
 for theta_i in theta_scan:
     s = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta_i)) # Steuervektor
     s = s.reshape(-1,1)
     metric = 1 / (s.conj().T @ V @ V.conj().T @ s) # MUSIC-Gleichung
     metric = np.abs(metric.squeeze())
     metric = 10*np.log10(metric)
     results.append(metric)

 results /= np.max(results) # normalisieren

.. image:: ../_images/doa_music.svg
   :align: center
   :target: ../_images/doa_music.svg
   :alt: Beispiel DOA mit MUSIC-Algorithmus

Um die Anzahl der Signale zu schätzen, sortieren wir die Eigenwertbeträge und plotten sie:

.. code-block:: python

 plot(10*np.log10(np.abs(w)),'.-')

.. image:: ../_images/doa_eigenvalues.svg
   :align: center
   :target: ../_images/doa_eigenvalues.svg

Die mit dem Rausch-Unterraum assoziierten Eigenwerte sind die kleinsten und tendieren zu einem ähnlichen Wert. Hier sehen wir klar drei Signale. Die folgende Animation zeigt, wie gut MUSIC zwei eng benachbarte Signale trennt:

.. image:: ../_images/doa_music_animation.gif
   :scale: 100 %
   :align: center

***
LMS
***

Der Least Mean Squares (LMS)-Beamformer ist ein rechenarmer Beamformer, der von Bernard Widrow eingeführt wurde. Er unterscheidet sich von allen bisherigen Beamformern in zwei Punkten: 1) Er erfordert die Kenntnis des SOI (oder zumindest eines Teils davon, z. B. einer Synchronisationssequenz, Piloten usw.) und 2) er ist iterativ, d. h. die Gewichte werden über eine Reihe von Iterationen verfeinert. Der LMS-Algorithmus minimiert den mittleren quadratischen Fehler zwischen dem gewünschten Signal (SOI) und dem Ausgang des Beamformers:

.. math::

 w_{n+1} = w_n + \mu \underbrace{\left(y_n -  w_{n}^H x_n\right)^*}_{Fehler} x_n

wobei :math:`w_n` der Gewichtsvektor bei Iteration/Sample :math:`n`, :math:`\mu` die Schrittgröße, :math:`x_n` das empfangene Sample bei :math:`n`, :math:`y_n` der erwartete Wert (d. h. das bekannte SOI) und :math:`*` eine komplexe Konjugation ist. Der Term :math:`w_{n}^H x_n` ist einfach das Ergebnis der Anwendung der aktuellen Gewichte auf das Eingangssignal. Die Schrittgröße :math:`\mu` steuert, wie schnell die Gewichte gegen ihre optimalen Werte konvergieren.

Im folgenden Python-Beispiel simulieren wir ein 8-Element-Array mit einem SOI aus einem wiederholten Gold-Code (als BPSK gesendet) und zwei Ton-Störsendern bei 60 und -50 Grad:

.. image:: ../_images/doa_lms_animation.gif
   :scale: 100 %
   :align: center

.. code-block:: python

 # Szenario
 sample_rate = 1e6
 d = 0.5 # halber Wellenlängenabstand
 N = 100000 # Anzahl der Samples
 Nr = 8 # Elemente
 theta_soi = 20 / 180 * np.pi
 theta2    = 60 / 180 * np.pi
 theta3   = -50 / 180 * np.pi
 t = np.arange(N)/sample_rate # Zeitvektor
 s1 = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta_soi)).reshape(-1,1) # 8x1
 s2 = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta2)).reshape(-1,1)
 s3 = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta3)).reshape(-1,1)

 # SOI ist ein Gold-Code, wiederholt, Länge 127
 gold_code = np.array([-1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1])
 soi_samples_per_symbol = 8
 soi = np.repeat(gold_code, soi_samples_per_symbol)
 num_sequence_repeats = int(N / soi.shape[0]) + 1 # Anzahl der Wiederholungen
 soi = np.tile(soi, num_sequence_repeats)[:N] # Sequenz wiederholen und kürzen
 soi = soi.reshape(1, -1) # 1xN

 # Störsender (Ton-Jammer) aus verschiedenen Richtungen
 tone2 = np.exp(2j*np.pi*0.02e6*t).reshape(1,-1)
 tone3 = np.exp(2j*np.pi*0.03e6*t).reshape(1,-1)

 # Empfangenes Signal simulieren
 r = s1 @ soi + s2 @ tone2 + s3 @ tone3
 n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
 r = r + 0.5*n # 8xN

 # LMS – Richtung des SOI unbekannt, aber SOI-Signal selbst bekannt
 mu = 0.5e-5 # LMS-Schrittgröße
 w_lms = np.zeros((Nr, 1), dtype=np.complex128) # mit Nullen beginnen

 # Empfangene Samples durchlaufen
 error_log = []
 for i in range(N):
    r_sample = r[:, i].reshape(-1, 1) # 8x1
    soi_sample = soi[0, i] # Skalar
    y = w_lms.conj().T @ r_sample # Gewichte anwenden
    y = y.squeeze() # als Skalar
    error = soi_sample - y
    error_log.append(np.abs(error)**2)
    w_lms += mu * np.conj(error) * r_sample # Gewichte sind noch 8x1

 w_lms /= np.linalg.norm(w_lms) # Gewichte normalisieren

 plt.plot(error_log)
 plt.xlabel('Iteration')
 plt.ylabel('Mittlerer quadratischer Fehler')
 plt.show()

 # Strahlmuster wie zuvor plotten

*******************
Trainingsdaten
*******************

Im Kontext der Array-Verarbeitung gibt es das Konzept des „Trainings", bei dem die Kovarianzmatrix :code:`R` berechnet wird, bevor das potenzielle SOI vorhanden ist. Dies wird besonders in der Radartechnik verwendet, wo die meiste Zeit kein SOI vorhanden ist und der gesamte Erkennungsprozess darin besteht, eine Reihe von Winkeln zu testen, ob ein SOI vorhanden ist. Wenn wir :code:`R` vor dem SOI berechnen, können wir Gewichte mit Methoden wie MVDR berechnen, die nur die Störer und Rauschbedingungen in der Kovarianzmatrix enthalten. So besteht keine Gefahr, dass MVDR eine Nullstelle in oder nahe der Richtung des SOI platziert.

Zur Demonstration des Wertes von Trainingsdaten verwenden wir Aufzeichnungen eines echten 16-Element-Arrays (QUAD-MxFE-Plattform von Analog Devices). Zunächst führen wir MVDR wie üblich durch, wobei das gesamte empfangene Signal zur Berechnung von :code:`R` verwendet wird. Dann verwenden wir eine separate Aufzeichnung (ohne SOI) für :code:`R`.

Die Aufzeichnungen wurden bei einer HF-Frequenz von 3,3 GHz mit einem Array mit 0,045 m Abstand (d = 0,495) und einer Abtastrate von 30 MHz aufgenommen. Wir bezeichnen die drei Signale als A, B und C. Signal C ist das SOI, A und B sind Störsender. Daher benötigen wir eine Aufzeichnung nur mit A und B für die Trainingsdaten.

Aufzeichnungsdateien:

https://github.com/777arc/777arc.github.io/raw/master/3p3G_A_B.npy

https://github.com/777arc/777arc.github.io/raw/master/3p3G_A_B_C.npy

Normale MVDR mit der A_B_C-Aufzeichnung:

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   # Array-Parameter
   center_freq = 3.3e9
   sample_rate = 30e6
   d = 0.045 * center_freq / 3e8
   print("d:", d)

   # Enthält alle drei Signale; C ist unser SOI
   filename = '3p3G_A_B_C.npy'
   X = np.load(filename)
   Nr = X.shape[0]

DOA mit MVDR zur Identifizierung der Ankunftswinkel:

.. code-block:: python

   # DOA durchführen, um Ankunftswinkel von C zu finden
   theta_scan = np.linspace(-1*np.pi/2, np.pi/2, 10000) # zwischen -90 und +90 Grad
   results = []
   R = X @ X.conj().T # Kovarianzmatrix berechnen
   Rinv = np.linalg.pinv(R)
   for theta_i in theta_scan:
      a = np.exp(2j * np.pi * d * np.arange(X.shape[0]) * np.sin(theta_i)) # Steuervektor
      a = a.reshape(-1,1)
      power = 1/(a.conj().T @ Rinv @ a).squeeze() # MVDR-Leistungsgleichung
      power_dB = 10*np.log10(np.abs(power))
      results.append(power_dB)
   results -= np.max(results) # auf 0 dB am Peak normalisieren

.. image:: ../_images/DOA_without_training.svg
   :align: center
   :target: ../_images/DOA_without_training.svg
   :alt: DOA ohne Trainingsdaten

Ankunftswinkel von C extrahieren:

.. code-block:: python

   # Winkel von C extrahieren, nach Nullen bei den Störsenderwinkeln
   results_temp = np.array(results)
   results_temp[int(len(results)*0.4):] = -9999*np.ones(int(len(results)*0.6))
   max_angle = theta_scan[np.argmax(results_temp)] # Radiant
   print("max_angle:", max_angle)

MVDR-Gewichte berechnen:

.. code-block:: python

   # MVDR-Gewichte berechnen
   s = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(max_angle)) # Steuervektor
   s = s.reshape(-1,1)
   w = (Rinv @ s)/(s.conj().T @ Rinv @ s) # MVDR/Capon-Gleichung

.. raw:: html

   <details>
   <summary>Plot-Code aufklappen (nichts Neues)</summary>

.. code-block:: python

   # Strahlmuster berechnen
   w = w.squeeze()
   N_fft = 2048
   w_padded = np.concatenate((w, np.zeros(N_fft - Nr)))
   w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2)
   w_fft_dB -= np.max(w_fft_dB)
   theta_bins = np.arcsin(np.linspace(-1, 1, N_fft))

   # Strahlmuster und DOA-Ergebnisse plotten
   plt.plot(theta_bins * 180 / np.pi, w_fft_dB)
   plt.plot(theta_scan * 180 / np.pi, results, 'r')
   plt.vlines(ymax=np.max(results), ymin=np.min(results) , x=max_angle*180/np.pi, color='g', linestyle='--')
   plt.xlabel("Winkel [Grad]")
   plt.ylabel("Magnitude [dB]")
   plt.title("Strahlmuster und DOA-Ergebnisse, ohne Training")
   plt.grid()
   plt.show()

.. raw:: html

   </details>

.. image:: ../_images/DOA_without_training_pattern.svg
   :align: center
   :target: ../_images/DOA_without_training_pattern.svg
   :alt: DOA ohne Trainingsdaten: DOA und MVDR-Strahlmuster

Wir erzeugen erfolgreich Nullstellen bei A und B. Das Lobe bei C ist jedoch nicht so stark, da der Hauptstrahl gegen die Nullstellen kämpft. Um einen starken Hauptstrahl bei unserem :code:`max_angle` zu erhalten, verwenden wir **Trainingsdaten**.

Laden der A-B-Aufzeichnung als Trainingsdaten:

.. code-block:: python

   # Trainingsdaten laden (nur A und B), dann Rinv berechnen
   filename = '3p3G_A_B.npy'
   X_A_B = np.load(filename)
   R_training = X_A_B @ X_A_B.conj().T # Kovarianzmatrix berechnen
   Rinv_training = np.linalg.pinv(R_training)

MVDR-Gewichte mit Trainings-Rinv berechnen:

.. code-block:: python

   # MVDR-Gewichte mit Trainings-Rinv berechnen
   s = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(max_angle)) # Steuervektor
   s = s.reshape(-1,1)
   w = (Rinv_training @ s)/(s.conj().T @ Rinv_training @ s) # MVDR/Capon-Gleichung

.. image:: ../_images/DOA_with_training.svg
   :align: center
   :target: ../_images/DOA_with_training.svg
   :alt: DOA mit Trainingsdaten: DOA und MVDR-Strahlmuster

Jetzt gibt es Nullstellen bei A und B, aber dieses Mal einen massiven Hauptstrahl in Richtung unseres Interessenwinkels C. Das ist die Stärke von Trainingsdaten.

*******************************
Breitbandstörer simulieren
*******************************

Die Methode, die wir bisher verwendet haben, um Signale zu simulieren, die aus einem bestimmten Ankunftswinkel auf unser Array treffen, verwendet eine Schmalbandannahme – d. h. das Signal wird als eine einzelne Frequenz angenommen. Dies funktioniert nicht gut für Breitbandsignale (z. B. mit einer Bandbreite größer als etwa 5% der Mittenfrequenz). Wir beschreiben kurz einen Trick zur Simulation von Breitband-**Rauschen** aus einer bestimmten Richtung (z. B. Barrage-Jamming).

Die Methode baut eine Kovarianzmatrix :code:`R`, indem die Beiträge jeder Breitbandrauschquelle summiert werden. Die Quadratwurzelmatrix :code:`A` wird berechnet und die Samples :code:`X` werden durch „Einfärben" von normalem komplexem Gaußschen Rauschen mit :code:`A` erzeugt. Ein wichtiger Parameter ist :code:`fractional_bw` (Bandbreite des Rauschsignals geteilt durch Mittenfrequenz):

.. code-block:: python

 N = 10 # Anzahl der Elemente im ULA
 num_samples = 10000
 d = 0.5

 num_jammers = 3
 jammer_pow_dB = np.array([30, 30, 30]) # Jammer-Leistungen in dB
 jammer_aoa_deg = np.array([-70, -20, 40])  # Jammer-Winkel in Grad
 jammer_aoa = np.sin(np.deg2rad(jammer_aoa_deg)) * np.pi
 element_gain_dB = np.zeros(N) # Gewinne der Array-Elemente in dB
 element_gain_linear = 10.0 ** (element_gain_dB / 10) # in lineare Werte umwandeln
 fractional_bw = 0.1 # wenn 0, entspricht die Methode dem traditionellen Ansatz

 # NxN Jammer-Kovarianzmatrix R aufbauen
 R = np.zeros((N, N), dtype=complex)
 for m in range(N):
     for n in range(N):
         for j in range(num_jammers):
             total_element_gain = np.sqrt(element_gain_linear[m] * element_gain_linear[n])
             sinc_term = np.sinc(0.5 * fractional_bw * (m - n) * jammer_aoa[j] / np.pi)
             exp_term = np.exp(1j * (m - n) * jammer_aoa[j])
             R[m, n] += 10.0 ** (jammer_pow_dB[j] / 10) * total_element_gain * sinc_term * exp_term
 R = np.eye(N, dtype=complex) + R

 # Empfangene Samples erzeugen
 A = fractional_matrix_power(R, 0.5) # Matrix-Quadratwurzel berechnen
 A = A / np.sqrt(2)
 X = np.zeros((N, num_samples), dtype=complex)
 for k in range(num_samples):
     noise_vec = np.random.randn(N) + 1j * np.random.randn(N) # komplexes Rauschen
     X[:, k] = A.conj().T @ noise_vec

Mit :code:`fractional_bw=0` (Schmalband-Annahme):

.. image:: ../_images/doa_covariance_method_1.svg
   :align: center
   :target: ../_images/doa_covariance_method_1.svg
   :alt: DOA-Kovarianzmethod mit Bruchbandbreite 0

Mit :code:`fractional_bw=0.1` (Breitbandrauschen), wodurch MVDR viel breitere Nullstellen erzeugt:

.. image:: ../_images/doa_covariance_method_2.svg
   :align: center
   :target: ../_images/doa_covariance_method_2.svg
   :alt: DOA-Kovarianzmethod mit Bruchbandbreite 0.1

*******************
Kreisförmige Arrays
*******************

Wir sprechen kurz über das Uniforme Kreisförmige Array (UCA), das für DOA beliebt ist, da es die 180-Grad-Mehrdeutigkeit von ULAs umgeht. Das KrakenSDR ist z. B. ein 5-Element-Array, und es ist üblich, diese fünf Elemente in einem Kreis mit gleichem Abstand anzuordnen. Theoretisch reichen nur drei Elemente für ein UCA.

Der gesamte Code, den wir bisher untersucht haben, gilt für UCAs; wir müssen nur die Steuervektor-Gleichung durch eine UCA-spezifische ersetzen:

.. code-block:: python

   radius = 0.05 # normiert durch Wellenlänge!
   d = np.sqrt(2 * radius**2 * (1 - np.cos(2*np.pi/Nr)))
   sf = 1.0 / (np.sqrt(2.0) * np.sqrt(1.0 - np.cos(2*np.pi/Nr))) # Skalierungsfaktor
   x = d * sf * np.cos(2 * np.pi / Nr * np.arange(Nr))
   y = -1 * d * sf * np.sin(2 * np.pi / Nr * np.arange(Nr))
   s = np.exp(1j * 2 * np.pi * (x * np.cos(theta) + y * np.sin(theta)))
   s = s.reshape(-1, 1) # Nrx1

Außerdem solltest du von 0 bis 360 Grad scannen, anstatt nur von -90 bis +90 Grad wie bei einem ULA.

Für 2D-Arrays (z. B. rechteckig) siehe :ref:`2d-beamforming-chapter`.

*************************
Schlussfolgerung und Referenzen
*************************

Den gesamten Python-Code, einschließlich des Codes zur Erzeugung der Abbildungen/Animationen, findest du `auf der GitHub-Seite des Lehrbuchs <https://github.com/777arc/PySDR/blob/master/figure-generating-scripts/doa.py>`_.

* DOA-Implementierung in GNU Radio – https://github.com/EttusResearch/gr-doa
* DOA-Implementierung für KrakenSDR – https://github.com/krakenrf/krakensdr_doa/blob/main/_signal_processing/krakenSDR_signal_processor.py

[1] Mailloux, Robert J. Phased Array Antenna Handbook. Second edition, Artech House, 2005

[2] Van Trees, Harry L. Optimum Array Processing: Part IV of Detection, Estimation, and Modulation Theory. Wiley, 2002.

.. |br| raw:: html

      <br>
