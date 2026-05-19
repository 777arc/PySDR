.. _freq-domain-chapter:

##########################
Zyklostationäre Verarbeitung
##########################

.. raw:: html

 <span style="display: table; margin: 0 auto; font-size: 20px;">Mitautorin: <a href="https://www.linkedin.com/in/samuel-brown-vt">Sam Brown</a></span>

In diesem Kapitel entmystifizieren wir die zyklostationäre Signalverarbeitung (auch CSP genannt), ein relativ spezialisiertes Gebiet der HF-Signalverarbeitung, das zur Analyse oder Erkennung (oft bei sehr niedrigem SNR!) von Signalen mit zyklostationären Eigenschaften eingesetzt wird, wie z. B. die meisten modernen digitalen Modulationsverfahren. Wir behandeln die zyklische Autokorrelationsfunktion (CAF), die Spektrale Korrelationsfunktion (SCF), die Spektrale Kohärenzfunktion (COH), deren konjugierte Versionen und wie sie angewendet werden können. Dieses Kapitel enthält mehrere vollständige Python-Implementierungen mit Beispielen für BPSK, QPSK, OFDM und mehrere kombinierte Signale.

****************
Einführung
****************

Zyklostationäre Signalverarbeitung (CSP) ist eine Sammlung von Techniken zur Ausnutzung der zyklostationären Eigenschaft, die in vielen realen Kommunikationssignalen vorkommt. Dazu gehören modulierte Signale wie AM/FM/TV-Rundfunk, Mobilfunk und WLAN sowie Radarsignale und andere Signale, die Periodizität in ihrer Statistik aufweisen. Ein großer Teil der traditionellen Signalverarbeitungstechniken basiert auf der Annahme, dass das Signal stationär ist, d. h. die Statistiken des Signals wie Mittelwert, Varianz und Momente höherer Ordnung ändern sich nicht mit der Zeit. Die meisten realen HF-Signale sind jedoch zyklostationär, d. h. die Statistiken des Signals ändern sich *periodisch* mit der Zeit. CSP-Techniken nutzen diese zyklostationäre Eigenschaft und können zur Erkennung von Signalen im Rauschen, zur Modulationserkennung und zur Trennung von Signalen verwendet werden, die sich sowohl in Zeit als auch in Frequenz überlappen.

Wenn du nach dem Durchlesen dieses Kapitels und dem Spielen in Python tiefer in CSP eintauchen möchtest, schau dir William Gardners Lehrbuch von 1994 `Cyclostationarity in Communications and Signal Processing <https://faculty.engineering.ucdavis.edu/gardner/wp-content/uploads/sites/146/2014/05/Cyclostationarity.pdf>`_ an, sein Lehrbuch von 1987 `Statistical Spectral Analysis <https://faculty.engineering.ucdavis.edu/gardner/wp-content/uploads/sites/146/2013/02/Statistical_Spectral_Analysis_A_Nonprobabilistic_Theory.pdf>`_ oder Chad Spooners `Sammlung von Blog-Beiträgen <https://cyclostationary.blog/>`_.

Am Ende des SCF-Abschnitts findest du eine interaktive JavaScript-App, mit der du die SCF eines Beispielsignals untersuchen kannst, um zu sehen, wie sich die SCF mit verschiedenen Signal- und SCF-Parametern ändert – alles in deinem Browser!

*************************
Wiederholung der Autokorrelation
*************************

Auch wenn du mit der Autokorrelationsfunktion vertraut bist, lohnt es sich, sie kurz zu wiederholen, da sie die Grundlage von CSP ist. Die Autokorrelationsfunktion ist ein Maß für die Ähnlichkeit (auch Korrelation genannt) zwischen einem Signal und seiner zeitverschobenen Version. Sie gibt an, in welchem Maße ein Signal repetitives Verhalten aufweist. Die Autokorrelation des Signals :math:`x(t)` ist definiert als:

.. math::
    R_x(\tau) = E[x(t)x^*(t-\tau)]

wobei :math:`E` der Erwartungswertoperator, :math:`\tau` die Zeitverzögerung und :math:`*` das Symbol für die komplexe Konjugation ist. In diskreter Zeit mit einer begrenzten Anzahl von Samples wird dies zu:

.. math::
    R_x(\tau) = \frac{1}{N} \sum_{n=-N/2}^{N/2} x\left[ n+\frac{\tau}{2} \right] x^*\left[ n-\frac{\tau}{2} \right]

wobei :math:`N` die Anzahl der Samples im Signal ist.

Wenn das Signal in irgendeiner Weise periodisch ist, wie z. B. die sich wiederholende Symbolform eines QPSK-Signals, dann wird auch die Autokorrelation über einen Bereich von Tau periodisch sein. Wenn ein QPSK-Signal beispielsweise 8 Samples pro Symbol hat, gibt es bei Tau als ganzzahligem Vielfachen von 8 ein viel stärkeres „Ähnlichkeitsmaß" als bei anderen Tau-Werten.

************************************************
Die Zyklische Autokorrelationsfunktion (CAF)
************************************************

Wir wollen herausfinden, wann in unserer Autokorrelation Periodizität vorhanden ist. Erinnere dich an die Fourier-Transformationsgleichung: Wenn wir testen wollen, wie stark eine bestimmte Frequenz :math:`f` in einem beliebigen Signal :math:`x(t)` vorhanden ist, können wir dies mit:

.. math::
    X(f) = \int x(t) e^{-j2\pi ft} dt

Um also Periodizität in unserer Autokorrelation zu finden, berechnen wir:

.. math::
    R_x(\tau, \alpha) = \lim_{T\rightarrow\infty} \frac{1}{T} \int_{-T/2}^{T/2} x(t + \tau/2)x^*(t - \tau/2)e^{-j2\pi \alpha t}dt.

oder in diskreter Zeit:

.. math::
    R_x(\tau, \alpha) = \frac{1}{N} \sum_{n=-N/2}^{N/2} x\left[ n+\frac{\tau}{2} \right] x^*\left[ n-\frac{\tau}{2} \right] e^{-j2\pi \alpha n}

Dies testet, wie stark die Frequenz :math:`\alpha` ist. Wir nennen die obige Gleichung die Zyklische Autokorrelationsfunktion (CAF). Eine andere Möglichkeit, die CAF zu betrachten, ist als Satz von Fourier-Reihenkoeffizienten, die diese Periodizität beschreiben. Wir verwenden den Begriff „zyklostationär" für Signale, die eine periodische oder fast periodische Autokorrelation besitzen.

In Python kann die CAF des Basisbands :code:`samples` bei einem gegebenen :code:`alpha`- und :code:`tau`-Wert mit dem folgenden Code-Schnipsel berechnet werden:

.. code-block:: python

 CAF = (np.exp(1j * np.pi * alpha * tau) *
        np.sum(samples * np.conj(np.roll(samples, tau)) *
               np.exp(-2j * np.pi * alpha * np.arange(N))))

Wir verwenden :code:`np.roll()`, um einen der Sample-Sätze um tau zu verschieben, da die Verschiebung um eine ganzzahlige Anzahl von Samples erfolgen muss.

Um mit der CAF in Python zu spielen, simulieren wir zunächst ein Beispielsignal: ein rechteckiges BPSK-Signal (d. h. BPSK ohne Impulsformung) mit 20 Samples pro Symbol, weißem Gaußschen Rauschen (AWGN) und einem Frequenzversatz von 0,2 Hz:

.. code-block:: python

 N = 100000 # Anzahl der zu simulierenden Samples
 f_offset = 0.2 # Hz normiert
 sps = 20 # zyklische Freq (alpha) wird 1/sps oder 0,05 Hz normiert sein

 symbols = np.random.randint(0, 2, int(np.ceil(N/sps))) * 2 - 1 # zufällige 1en und -1en
 bpsk = np.repeat(symbols, sps)  # jedes Symbol sps-mal wiederholen
 bpsk = bpsk[:N]  # auf N Samples kürzen
 bpsk = bpsk * np.exp(2j * np.pi * f_offset * np.arange(N)) # BPSK nach oben verschieben (macht es auch komplex)
 noise = np.random.randn(N) + 1j*np.random.randn(N) # komplexes weißes Gaußsches Rauschen
 samples = bpsk + 0.1*noise  # Rauschen zum Signal hinzufügen

Da die absolute Abtastrate und Symbolrate in diesem Kapitel keine Rolle spielen, verwenden wir normierte Frequenz (entspricht Abtastrate = 1 Hz). Das bedeutet, das Signal muss zwischen -0,5 und +0,5 Hz liegen.

Zur Veranschaulichung zeigen wir die Leistungsspektraldichte (d. h. FFT) des Signals *vor* jeder CSP-Verarbeitung:

.. image:: ../_images/psd_of_bpsk_used_for_caf.svg
   :align: center
   :target: ../_images/psd_of_bpsk_used_for_caf.svg
   :alt: PSD des für CAF verwendeten BPSK

Wir berechnen nun die CAF bei dem richtigen Alpha (1/20 = 0,05 Hz) über einen Bereich von Tau-Werten:

.. code-block:: python

    # CAF nur beim richtigen Alpha
    alpha_of_interest = 1/sps # entspricht 0,05 Hz
    taus = np.arange(-50, 51)
    CAF = np.zeros(len(taus), dtype=complex)
    for i in range(len(taus)):
        CAF[i] = (np.exp(1j * np.pi * alpha_of_interest * taus[i]) *
                  np.sum(samples * np.conj(np.roll(samples, taus[i])) *
                         np.exp(-2j * np.pi * alpha_of_interest * np.arange(N))))

.. image:: ../_images/caf_at_correct_alpha.svg
   :align: center
   :target: ../_images/caf_at_correct_alpha.svg
   :alt: CAF beim richtigen Alpha

Zum Vergleich die CAF bei einem falschen Alpha (0,08 Hz):

.. image:: ../_images/caf_at_incorrect_alpha.svg
   :align: center
   :target: ../_images/caf_at_incorrect_alpha.svg
   :alt: CAF beim falschen Alpha

Beachte die y-Achse – viel weniger Energie in der CAF diesmal. Wir können die CAF über einen Bereich von Alphas berechnen und bei jedem Alpha die Leistung in der CAF bestimmen:

.. code-block:: python

    alphas = np.arange(0, 0.5, 0.005)
    CAF = np.zeros((len(alphas), len(taus)), dtype=complex)
    for j in range(len(alphas)):
        for i in range(len(taus)):
            CAF[j, i] = (np.exp(1j * np.pi * alphas[j] * taus[i]) *
                         np.sum(samples * np.conj(np.roll(samples, taus[i])) *
                                np.exp(-2j * np.pi * alphas[j] * np.arange(N))))
    CAF_magnitudes = np.average(np.abs(CAF), axis=1) # bei jedem Alpha Leistung berechnen
    plt.plot(alphas, CAF_magnitudes)
    plt.xlabel('Alpha')
    plt.ylabel('CAF-Leistung')

.. image:: ../_images/caf_avg_over_alpha.svg
   :align: center
   :target: ../_images/caf_avg_over_alpha.svg
   :alt: CAF-Durchschnitt über Alpha

Wir sehen nicht nur den erwarteten Spike bei 0,05 Hz, sondern auch Spikes bei ganzzahligen Vielfachen davon. Dies liegt daran, dass die CAF eine Fourier-Reihe ist und die Harmonischen der Grundfrequenz in der CAF vorhanden sind.

************************************************
Die Spektrale Korrelationsfunktion (SCF)
************************************************

So wie die CAF die Periodizität in der Autokorrelation eines Signals zeigt, zeigt die SCF die Periodizität in der PSD eines Signals. Autokorrelation und PSD sind ein Fourier-Transformationspaar, und daher sollte es keine Überraschung sein, dass CAF und SCF ebenfalls ein Fourier-Transformationspaar sind. Diese Beziehung ist als *Zyklische Wiener-Beziehung* bekannt.

Man kann einfach die Fourier-Transformierte der CAF nehmen, um die SCF zu erhalten. Zurück zu unserem BPSK-Signal mit 20 Samples pro Symbol – schauen wir uns die SCF beim richtigen Alpha (0,05 Hz) an. Alles was wir tun müssen, ist die FFT der CAF zu nehmen und den Betrag zu plotten:

.. code-block:: python

 f = np.linspace(-0.5, 0.5, len(taus))
 SCF = np.fft.fftshift(np.fft.fft(CAF))
 plt.plot(f, np.abs(SCF))
 plt.xlabel('Frequenz')
 plt.ylabel('SCF')

.. image:: ../_images/fft_of_caf.svg
   :align: center
   :target: ../_images/fft_of_caf.svg
   :alt: FFT der CAF

Wir sehen den 0,2-Hz-Frequenzversatz, den wir bei der Simulation des BPSK-Signals angewendet haben. Unten ist eine interaktive JavaScript-App, die eine SCF implementiert, damit du mit verschiedenen Signal- und SCF-Parametern spielen kannst:

.. raw:: html

    <form id="mainform" name="mainform">
        <label>Samples to Simulate </label>
        <select id="N">
            <option value="1024">1024</option>
            <option value="2048">2048</option>
            <option value="4096">4096</option>
            <option value="8192" selected="selected">8192</option>
            <option value="16384">16384</option>
            <option value="32768">32768</option>
            <option value="65536">65536</option>
            <option value="131072">131072</option>
            <option value="262144">262144</option>
        </select>
        <br />
        <label>Frequency [normalized Hz] </label>
        <input type="range" id="freq" value="0.2" min="-0.5" max="0.5" step="0.05">
        <span id="freq_display">0.2</span>
        <br />
        <label>Samples per Symbol [int] </label>
        <input type="range" id="sps" value="20" min="4" max="30" step="1">
        <span id="sps_display">20</span>
        <br />
        <label>RC Rolloff [0 to 1] </label>
        <input type="number" id="rolloff" value="0.5" min="0" max="1" step="0.0001">
        <label>Rectangular Pulses </label>
        <input type="checkbox" id="rect" checked>
        <br />
        <label>Alpha Start </label>
        <input type="number" id="alpha_start" value="0" min="0" max="100" step="0.0001">
        <br />
        <label>Alpha Stop </label>
        <input type="number" id="alpha_stop" value="0.3" min="0" max="1" step="0.0001">
        <br />
        <label>Alpha Step </label>
        <input type="number" id="alpha_step" value="0.001" min="0.0001" max="0.1" step="0.0001">
        <br />
        <label>Noise Level </label>
        <input type="number" id="noise" value="0.001" min="0" max="10" step="0.0001">
        <br />
        <button type="submit" id="submit_button">Submit</button>
    </form>
    <form id="resetform" name="resetform">
        <button type="submit" id="submit_button">Reset</button>
    </form>
    <canvas id="scf_canvas"></canvas>
    <script>cyclostationary_app()</script>
    </body>


********************************
Frequenzglättungsmethode (FSM)
********************************

Nun betrachten wir, wie wir die SCF effizient berechnen können. Betrachte zunächst das Periodogramm, das einfach der quadratische Betrag der Fourier-Transformierten eines Signals ist:

.. math::

 I(u,f) = \frac{1}{N}\left|X(u,f)\right|^2

Wir können das zyklische Periodogramm durch das Produkt zweier frequenzverschobener Fourier-Transformierter erhalten:

.. math::

 I(u,f,\alpha) = \frac{1}{N}X(u,f + \alpha/2) X^*(u,f - \alpha/2)

Beide stellen Schätzungen der PSD und der SCF dar; zur Mittelung über die Frequenz verwenden wir die Frequenzglättungsmethode (FSM):

.. math::
    S_X(f, \alpha) = \lim_{\Delta\rightarrow 0} \lim_{T\rightarrow \infty} \frac{1}{T} g_{\Delta}(f) \otimes \left[X(t,f + \alpha/2) X^*(t,f - \alpha/2)\right]

Unten ist eine minimale Python-Implementierung der FSM. Zuerst berechnet sie das zyklische Periodogramm durch Multiplikation zweier verschobener FFT-Versionen, dann wird jede Scheibe mit einer Fensterfunktion gefiltert, deren Länge die Auflösung der resultierenden SCF-Schätzung bestimmt:

.. code-block:: python

    alphas = np.arange(0, 0.3, 0.001)
    Nw = 256 # Fensterlänge
    N = len(samples) # Signallänge
    window = np.hanning(Nw)

    X = np.fft.fftshift(np.fft.fft(samples)) # FFT des gesamten Signals

    num_freqs = int(np.ceil(N/Nw)) # Frequenzauflösung nach Dezimierung
    SCF = np.zeros((len(alphas), num_freqs), dtype=complex)
    for i in range(len(alphas)):
        shift = int(alphas[i] * N/2)
        SCF_slice = np.roll(X, -shift) * np.conj(np.roll(X, shift))
        SCF[i, :] = np.convolve(SCF_slice, window, mode='same')[::Nw] # Fenster anwenden und um Nw dezimieren
    SCF = np.abs(SCF)
    SCF[0, :] = 0 # alpha=0 auf Null setzen, da es nur die PSD des Signals ist

    extent = (-0.5, 0.5, float(np.max(alphas)), float(np.min(alphas)))
    plt.imshow(SCF, aspect='auto', extent=extent, vmax=np.max(SCF)/2)
    plt.xlabel('Frequenz [normiert Hz]')
    plt.ylabel('Zyklische Frequenz [normiert Hz]')
    plt.show()

Beachte, dass aufgrund der Art und Weise, wie die Verschiebung berechnet und auf eine ganzzahlige Anzahl von Samples gerundet wird, es hilft, mindestens :code:`2 / alpha_resolution` Samples auf einmal zu verarbeiten.

SCF für das rechteckige BPSK-Signal mit 20 Samples pro Symbol:

.. image:: ../_images/scf_freq_smoothing.svg
   :align: center
   :target: ../_images/scf_freq_smoothing.svg
   :alt: SCF mit der Frequenzglättungsmethode (FSM)

***************************
Zeitglättungsmethode (TSM)
***************************

Nun betrachten wir eine Implementierung der TSM in Python. Der folgende Code-Schnipsel teilt das Signal in *num_windows* Blöcke auf, jeder der Länge *Nw* mit einer Überlappung von *Noverlap*:

.. code-block:: python

    alphas = np.arange(0, 0.3, 0.001)
    Nw = 256 # Fensterlänge
    N = len(samples) # Signallänge
    Noverlap = int(2/3*Nw) # Blocküberlappung
    num_windows = int((N - Noverlap) / (Nw - Noverlap)) # Anzahl der Fenster
    window = np.hanning(Nw)

    SCF = np.zeros((len(alphas), Nw), dtype=complex)
    for ii in range(len(alphas)): # Schleife über zyklische Frequenzen
        neg = samples * np.exp(-1j*np.pi*alphas[ii]*np.arange(N))
        pos = samples * np.exp( 1j*np.pi*alphas[ii]*np.arange(N))
        for i in range(num_windows):
            pos_slice = window * pos[i*(Nw-Noverlap):i*(Nw-Noverlap)+Nw]
            neg_slice = window * neg[i*(Nw-Noverlap):i*(Nw-Noverlap)+Nw]
            SCF[ii, :] += np.fft.fft(neg_slice) * np.conj(np.fft.fft(pos_slice)) # Kreuz-Zyklisches Leistungsspektrum
    SCF = np.fft.fftshift(SCF, axes=1) # HF-Frequenzachse verschieben
    SCF = np.abs(SCF)
    SCF[0, :] = 0 # alpha=0 auf Null setzen

    extent = (-0.5, 0.5, float(np.max(alphas)), float(np.min(alphas)))
    plt.imshow(SCF, aspect='auto', extent=extent, vmax=np.max(SCF)/2)
    plt.xlabel('Frequenz [normiert Hz]')
    plt.ylabel('Zyklische Frequenz [normiert Hz]')
    plt.show()

.. image:: ../_images/scf_time_smoothing.svg
   :align: center
   :target: ../_images/scf_time_smoothing.svg
   :alt: SCF mit der Zeitglättungsmethode (TSM)

Sieht ungefähr genauso aus wie die FSM!

*****************
Impulsgeformtes BPSK
*****************

Bisher haben wir nur CSP eines *rechteckigen* BPSK-Signals untersucht. In realen HF-Systemen sehen wir jedoch fast nie rechteckige Pulse. Schauen wir uns jetzt ein BPSK-Signal mit einer Raised-Cosine (RC)-Impulsform an, die eine gängige Impulsform in der digitalen Kommunikation ist:

.. math::
 h(t) = \mathrm{sinc}\left( \frac{t}{T} \right) \frac{\cos\left(\frac{\pi\beta t}{T}\right)}{1 - \left( \frac{2 \beta t}{T}   \right)^2}

Der Parameter :math:`\beta` bestimmt, wie schnell der Filter im Zeitbereich abklingt:

.. image:: ../_images/raised_cosine_freq.svg
   :align: center
   :target: ../_images/raised_cosine_freq.svg
   :alt: Der Raised-Cosine-Filter im Frequenzbereich mit verschiedenen Roll-Off-Werten

Simulation eines BPSK-Signals mit Raised-Cosine-Impulsformung:

.. code-block:: python

    N = 100000 # Anzahl der zu simulierenden Samples
    f_offset = 0.2 # Hz normiert
    sps = 20
    num_symbols = int(np.ceil(N/sps))
    symbols = np.random.randint(0, 2, num_symbols) * 2 - 1 # zufällige 1en und -1en

    pulse_train = np.zeros(num_symbols * sps)
    pulse_train[::sps] = symbols

    # Raised-Cosine-Filter für Impulsformung
    beta = 0.3 # Roll-Off-Parameter
    num_taps = 101
    t = np.arange(num_taps) - (num_taps-1)//2
    h = np.sinc(t/sps) * np.cos(np.pi*beta*t/sps) / (1 - (2*beta*t/sps)**2) # RC-Gleichung
    bpsk = np.convolve(pulse_train, h, 'same') # Impulsformung anwenden

    bpsk = bpsk[:N]
    bpsk = bpsk * np.exp(2j * np.pi * f_offset * np.arange(N))
    noise = np.random.randn(N) + 1j*np.random.randn(N)
    samples = bpsk + 0.1*noise

.. image:: ../_images/pulse_shaped_BSPK.svg
   :align: center
   :target: ../_images/pulse_shaped_BSPK.svg
   :alt: Impulsgeformtes BPSK-Signal mit Raised-Cosine-Impulsform

SCF des impulsgeformten BPSK mit verschiedenen Roll-Off-Werten:

:code:`beta = 0.3`:

.. image:: ../_images/scf_freq_smoothing_pulse_shaped_bpsk.svg
   :align: center
   :target: ../_images/scf_freq_smoothing_pulse_shaped_bpsk.svg
   :alt: SCF des impulsgeformten BPSK (FSM) beta 0,3

:code:`beta = 0.6`:

.. image:: ../_images/scf_freq_smoothing_pulse_shaped_bpsk2.svg
   :align: center
   :target: ../_images/scf_freq_smoothing_pulse_shaped_bpsk2.svg
   :alt: SCF des impulsgeformten BPSK (FSM) beta 0,6

:code:`beta = 0.9`:

.. image:: ../_images/scf_freq_smoothing_pulse_shaped_bpsk3.svg
   :align: center
   :target: ../_images/scf_freq_smoothing_pulse_shaped_bpsk3.svg
   :alt: SCF des impulsgeformten BPSK (FSM) beta 0,9

In allen drei erhalten wir keine Nebenkeulen in der Frequenzachse mehr, und in der zyklischen Frequenzachse bekommen wir nicht mehr die gleichen starken Harmonischen. Impulsgeformte Signale haben tendenziell eine viel „sauberere" SCF als rechteckige Signale, die einem einzelnen Spike mit einer Verschmierung darüber ähnelt.

********************************
SNR und Anzahl der Symbole
********************************

Demnächst! Wir werden behandeln, wie ab einem bestimmten Punkt ein höheres SNR nicht hilft und stattdessen mehr Symbole benötigt werden, und wie paketbasierte Wellenformen zu einer begrenzten Anzahl von Symbolen pro Übertragung führen.

********************************
QPSK und Modulation höherer Ordnung
********************************

Demnächst! Es werden QPSK, PSK höherer Ordnung, QAM und eine kurze Einführung in zyklische Momente und Kumulanten höherer Ordnung enthalten sein.

********************************
Mehrere überlappende Signale
********************************

Bisher haben wir uns immer nur ein Signal auf einmal angeschaut, aber was wenn unser empfangenes Signal mehrere einzelne Signale enthält, die sich in Frequenz, Zeit und sogar zyklischer Frequenz überlappen? In CSP sind wir oft damit beschäftigt, das Vorhandensein von Signalen bei verschiedenen zyklischen Frequenzen zu erkennen, die sich in Zeit und Frequenz überlappen.

Wir simulieren drei Signale mit unterschiedlichen Eigenschaften:

* Signal 1: Rechteckiges BPSK mit 20 Samples pro Symbol und 0,2 Hz Frequenzversatz
* Signal 2: Impulsgeformtes BPSK mit 20 Samples pro Symbol, -0,1 Hz Frequenzversatz und Roll-Off 0,35
* Signal 3: Impulsgeformtes QPSK mit 4 Samples pro Symbol, 0,2 Hz Frequenzversatz und Roll-Off 0,21

Zwei Signale haben dieselbe zyklische Frequenz, und zwei haben dieselbe HF-Frequenz, was uns verschiedene Überlappungsgrade zum Experimentieren ermöglicht.

.. raw:: html

   <details>
   <summary>Python-Code zur Simulation der drei Signale aufklappen</summary>

.. code-block:: python

    N = 1000000 # Anzahl der zu simulierenden Samples

    def fractional_delay(x, delay):
        N = 21 # Anzahl der Koeffizienten
        n = np.arange(-N//2, N//2) # ...-3,-2,-1,0,1,2,3...
        h = np.sinc(n - delay) # Filterkoeffizienten berechnen
        h *= np.hamming(N) # Filter fensterieren
        h /= np.sum(h) # normalisieren
        return np.convolve(x, h, 'same') # Filter anwenden

    # Signal 1, Rechteck-BPSK
    sps = 20
    f_offset = 0.2
    signal1 = np.repeat(np.random.randint(0, 2, int(np.ceil(N/sps))) * 2 - 1, sps)
    signal1 = signal1[:N] * np.exp(2j * np.pi * f_offset * np.arange(N))
    signal1 = fractional_delay(signal1, 0.12345)

    # Signal 2, impulsgeformtes BPSK
    sps = 20
    f_offset = -0.1
    beta = 0.35
    symbols = np.random.randint(0, 2, int(np.ceil(N/sps))) * 2 - 1
    pulse_train = np.zeros(int(np.ceil(N/sps)) * sps)
    pulse_train[::sps] = symbols
    t = np.arange(101) - (101-1)//2
    h = np.sinc(t/sps) * np.cos(np.pi*beta*t/sps) / (1 - (2*beta*t/sps)**2)
    signal2 = np.convolve(pulse_train, h, 'same')
    signal2 = signal2[:N] * np.exp(2j * np.pi * f_offset * np.arange(N))
    signal2 = fractional_delay(signal2, 0.52634)

    # Signal 3, impulsgeformtes QPSK
    sps = 4
    f_offset = 0.2
    beta = 0.21
    data = x_int = np.random.randint(0, 4, int(np.ceil(N/sps))) # 0 bis 3
    data_degrees = data*360/4.0 + 45 # 45, 135, 225, 315 Grad
    symbols = np.cos(data_degrees*np.pi/180.0) + 1j*np.sin(data_degrees*np.pi/180.0)
    pulse_train = np.zeros(int(np.ceil(N/sps)) * sps, dtype=complex)
    pulse_train[::sps] = symbols
    t = np.arange(101) - (101-1)//2
    h = np.sinc(t/sps) * np.cos(np.pi*beta*t/sps) / (1 - (2*beta*t/sps)**2)
    signal3 = np.convolve(pulse_train, h, 'same')
    signal3 = signal3[:N] * np.exp(2j * np.pi * f_offset * np.arange(N))
    signal3 = fractional_delay(signal3, 0.3526)

    # Rauschen hinzufügen
    noise = np.random.randn(N) + 1j*np.random.randn(N)
    samples = 0.5*signal1 + signal2 + 1.5*signal3 + 0.1*noise

.. raw:: html

   </details>

PSD dieser kombinierten Signale:

.. image:: ../_images/psd_of_multiple_signals.svg
   :align: center
   :target: ../_images/psd_of_multiple_signals.svg
   :alt: PSD von drei verschiedenen Signalen

SCF dieser kombinierten Signale mit der FSM:

.. image:: ../_images/scf_freq_smoothing_pulse_multiple_signals.svg
   :align: center
   :target: ../_images/scf_freq_smoothing_pulse_multiple_signals.svg
   :alt: SCF von drei verschiedenen Signalen (FSM)

Beachte, dass Signal 1 trotz rechteckiger Impulsform seine Harmonischen im Kegel über Signal 3 verborgen hat. Mittels CSP können wir erkennen, dass Signal 1 vorhanden ist, und eine gute Näherung seiner zyklischen Frequenz erhalten, die dann zur Synchronisation genutzt werden kann. Das ist die Stärke der zyklostationären Signalverarbeitung!

************************
Alternative CSP-Merkmale
************************

Die SCF ist nicht die einzige Möglichkeit, Zyklostationarität in einem Signal zu erkennen. Eine einfache Methode (sowohl konzeptionell als auch rechnerisch) beinhaltet das **Nehmen der FFT des Betrags** des Signals und die Suche nach Spikes. In Python ist das extrem einfach:

.. code-block:: python

    samples_mag = np.abs(samples)
    magnitude_metric = np.abs(np.fft.fft(samples_mag))

Bevor wir das Ergebnis plotten, nullen wir die DC-Komponente aus und nehmen nur die Hälfte der FFT-Ausgabe (da der Eingang reell ist):

.. code-block:: python

    magnitude_metric = magnitude_metric[:len(magnitude_metric)//2] # nur Hälfte benötigt, da Eingang reell ist
    magnitude_metric[0] = 0 # DC-Komponente auf Null setzen
    f = np.linspace(-0.5, 0.5, len(samples))
    plt.plot(f, magnitude_metric)

.. image:: ../_images/non_csp_metric.svg
   :align: center
   :target: ../_images/non_csp_metric.svg
   :alt: Metrik zur Erkennung von Zyklostationarität ohne CAF oder SCF

Für BPSK-Signale kann auch die FFT des quadrierten Signals berechnet werden; es zeigt einen Spike beim Trägerfrequenzversatz multipliziert mit zwei. Für QPSK die FFT des Signals in der 4. Potenz:

.. code-block:: python

    samples_squared = samples**2
    squared_metric = np.abs(np.fft.fftshift(np.fft.fft(samples_squared)))/len(samples)
    squared_metric[len(squared_metric)//2] = 0 # DC-Komponente auf Null setzen

    samples_quartic = samples**4
    quartic_metric = np.abs(np.fft.fftshift(np.fft.fft(samples_quartic)))/len(samples)
    quartic_metric[len(quartic_metric)//2] = 0 # DC-Komponente auf Null setzen

*********************************
Spektrale Kohärenzfunktion (COH)
*********************************

*Kurzfassung: Die spektrale Kohärenzfunktion ist eine normierte Version der SCF, die in manchen Situationen anstelle der regulären SCF verwendet werden sollte.*

Ein weiteres Maß für Zyklostationarität, das in vielen Fällen aufschlussreicher als die rohe SCF sein kann, ist die Spektrale Kohärenzfunktion (COH). Die COH nimmt die SCF und normiert sie so, dass das Ergebnis zwischen -1 und 1 liegt (wobei wir auf den Betrag schauen, der zwischen 0 und 1 liegt). Dies ist nützlich, da es die Informationen über die Zyklostationarität des Signals von Informationen über das Leistungsspektrum des Signals isoliert.

Um die COH zu berechnen, berechnen wir zunächst die SCF und normieren dann durch das Produkt zweier verschobener PSD-Terme, analog zur Normierung durch das Produkt der Standardabweichungen:

.. math::
    \rho = C_x(f, \alpha) = \frac{S_X(f,\alpha)}{\sqrt{C_x^0(f + \alpha/2) C_x^0(f - \alpha/2)}}

Wir wenden das auf unseren Python-Code an (spezifisch die SCF mit der FSM). Die COH-Scheibe wird innerhalb der for-Schleife berechnet:

.. code-block:: python

    COH_slice = SCF_slice / np.sqrt(np.roll(X, -shift) * np.roll(X, shift))

Dann wird dieselbe Faltung und Dezimierung wie für die SCF angewendet:

.. code-block:: python

    COH[i, :] = np.convolve(COH_slice, window, mode='same')[::Nw]

.. raw:: html

   <details>
   <summary>Vollständigen Code zur Erzeugung und Darstellung von SCF und COH aufklappen</summary>

.. code-block:: python

    alphas = np.arange(0, 0.3, 0.001)
    Nw = 256 # Fensterlänge
    N = len(samples) # Signallänge
    window = np.hanning(Nw)

    X = np.fft.fftshift(np.fft.fft(samples)) # FFT des gesamten Signals

    num_freqs = int(np.ceil(N/Nw))
    SCF = np.zeros((len(alphas), num_freqs), dtype=complex)
    COH = np.zeros((len(alphas), num_freqs), dtype=complex)
    for i in range(len(alphas)):
        shift = int(alphas[i] * N/2)
        SCF_slice = np.roll(X, -shift) * np.conj(np.roll(X, shift))
        SCF[i, :] = np.convolve(SCF_slice, window, mode='same')[::Nw]
        COH_slice = SCF_slice / np.sqrt(np.roll(X, -shift) * np.roll(X, shift))
        COH[i, :] = np.convolve(COH_slice, window, mode='same')[::Nw]
    SCF = np.abs(SCF)
    COH = np.abs(COH)

    # alpha=0 für beide auf Null setzen
    SCF[np.argmin(np.abs(alphas)), :] = 0
    COH[np.argmin(np.abs(alphas)), :] = 0

    extent = (-0.5, 0.5, float(np.max(alphas)), float(np.min(alphas)))
    fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(10, 5))
    ax0.imshow(SCF, aspect='auto', extent=extent, vmax=np.max(SCF)/2)
    ax0.set_xlabel('Frequenz [normiert Hz]')
    ax0.set_ylabel('Zyklische Frequenz [normiert Hz]')
    ax0.set_title('Reguläre SCF')
    ax1.imshow(COH, aspect='auto', extent=extent, vmax=np.max(COH)/2)
    ax1.set_xlabel('Frequenz [normiert Hz]')
    ax1.set_title('Spektrale Kohärenzfunktion (COH)')
    plt.show()

.. raw:: html

   </details>

SCF und COH für ein rechteckiges BPSK-Signal mit 20 Samples pro Symbol und 0,2 Hz Frequenzversatz:

.. image:: ../_images/scf_coherence.svg
   :align: center
   :target: ../_images/scf_coherence.svg
   :alt: SCF und COH eines rechteckigen BPSK-Signals

Wie zu sehen, sind die höheren Alphas in der COH viel ausgeprägter als in der SCF. Für das impulsgeformte BPSK-Signal:

.. image:: ../_images/scf_coherence_pulse_shaped.svg
   :align: center
   :target: ../_images/scf_coherence_pulse_shaped.svg
   :alt: SCF und COH eines impulsgeformten BPSK-Signals

Versuche, sowohl SCF als auch COH für deine Anwendung zu generieren, um zu sehen, welche besser funktioniert!

**********
Konjugierte
**********

Bisher haben wir die folgenden Formeln für die CAF und die SCF verwendet, bei denen die komplexe Konjugation (:math:`*` Symbol) des Signals im zweiten Term verwendet wird:

.. math::
    R_x(\tau,\alpha) = \lim_{T\rightarrow\infty} \frac{1}{T} \int_{-T/2}^{T/2} x(t + \tau/2)x^*(t - \tau/2)e^{-j2\pi \alpha t}dt \\
    S_X(f,\alpha) = \lim_{T\rightarrow\infty} \frac{1}{T} \lim_{U\rightarrow\infty} \frac{1}{U} \int_{-U/2}^{U/2} X(t,f + \alpha/2) X^*(t,f - \alpha/2) dt

Es gibt jedoch eine alternative Form für CAF und SCF, bei der keine Konjugation enthalten ist. Diese Formen werden als *konjugierte CAF* und *konjugierte SCF* bezeichnet. Die konjugierten CAF und SCF sind wie folgt definiert:

.. math::
    R_{x^*}(\tau,\alpha) = \lim_{T\rightarrow\infty} \frac{1}{T} \int_{-T/2}^{T/2} x(t + \tau/2)x(t - \tau/2)e^{-j2\pi \alpha t}dt \\
    S_{x^*}(f,\alpha) = \lim_{T\rightarrow\infty} \frac{1}{T} \lim_{U\rightarrow\infty} \frac{1}{U} \int_{-U/2}^{U/2} X(t,f + \alpha/2) X(t,f - \alpha/2) dt

Zur Implementierung der konjugierten SCF mit der FSM gibt es neben dem Entfernen des :code:`conj()` noch einen zusätzlichen Schritt. Es gibt eine Eigenschaft der Fourier-Transformierten, die besagt, dass eine komplexe Konjugation im Zeitbereich einer Spiegelung und Konjugation im Frequenzbereich entspricht:

.. math::
    x^*(t) \leftrightarrow X^*(-f)

Daher ist der Code wie folgt:

.. code-block:: python

    SCF_slice = np.roll(X, -shift) * np.flip(np.roll(X, -shift - 1))

Beachte das hinzugefügte :code:`np.flip()`, und :code:`roll()` muss in die umgekehrte Richtung erfolgen. Die vollständige FSM-Implementierung der konjugierten SCF:

.. code-block:: python

    alphas = np.arange(-1, 1, 0.01) # Konj. SCF sollte von -1 bis +1 berechnet werden
    Nw = 256 # Fensterlänge
    N = len(samples) # Signallänge
    window = np.hanning(Nw)

    X = np.fft.fftshift(np.fft.fft(samples)) # FFT des gesamten Signals

    num_freqs = int(np.ceil(N/Nw))
    SCF = np.zeros((len(alphas), num_freqs), dtype=complex)
    for i in range(len(alphas)):
        shift = int(np.round(alphas[i] * N/2))
        SCF_slice = np.roll(X, -shift) * np.flip(np.roll(X, -shift - 1)) # DIESER UNTERSCHIED
        SCF[i, :] = np.convolve(SCF_slice, window, mode='same')[::Nw]
    SCF = np.abs(SCF)

    extent = (-0.5, 0.5, float(np.min(alphas)), float(np.max(alphas)))
    plt.imshow(SCF, aspect='auto', extent=extent, vmax=np.max(SCF)/2, origin='lower')
    plt.xlabel('Frequenz [normiert Hz]')
    plt.ylabel('Zyklische Frequenz [normiert Hz]')
    plt.show()

Ein weiterer wichtiger Unterschied bei der konjugierten SCF ist, dass wir Alphas zwischen -1 und +1 berechnen wollen, während wir bei der normalen SCF aufgrund der Symmetrie nur 0,0 bis 0,5 gemacht haben.

Konjugierte SCF des rechteckigen BPSK-Signals mit 20 Samples pro Symbol und 0,2 Hz Frequenzversatz:

.. image:: ../_images/scf_conj_rect_bpsk.svg
   :align: center
   :target: ../_images/scf_conj_rect_bpsk.svg
   :alt: Konjugierte SCF des rechteckigen BPSK (FSM)

Das Wichtigste aus diesem Abschnitt: Was du in der konjugierten SCF erhältst, sind Spikes bei der zyklischen Frequenz +/- **zweimal** dem Trägerfrequenzversatz, den wir als :math:`f_c` bezeichnen. In der Frequenzachse liegt er bei 0 Hz statt bei :math:`f_c`. Unser Frequenzversatz war 0,2 Hz, also erhalten wir Spikes bei 0,4 Hz +/- der zyklischen Frequenz 0,05 Hz. Merke dir:

.. math::
    2f_c \pm \alpha

Konjugierte SCF des impulsgeformten BPSK:

.. image:: ../_images/scf_conj_pulseshaped_bpsk.svg
   :align: center
   :target: ../_images/scf_conj_pulseshaped_bpsk.svg
   :alt: Konjugierte SCF des impulsgeformten BPSK (FSM)

Konjugierte SCF des rechteckigen QPSK:

.. image:: ../_images/scf_conj_rect_qpsk.svg
   :align: center
   :target: ../_images/scf_conj_rect_qpsk.svg
   :alt: Konjugierte SCF des rechteckigen QPSK (FSM)

Beachte die Farbskala – die gesamte Ausgabe ist relativ niedrig, denn es gibt *keine Spikes in der konjugierten SCF bei QPSK*. Hier ist dieselbe Ausgabe mit Skalierung entsprechend unseren BPSK-Beispielen:

.. image:: ../_images/scf_conj_rect_qpsk_scaled.svg
   :align: center
   :target: ../_images/scf_conj_rect_qpsk_scaled.svg
   :alt: Konjugierte SCF des rechteckigen QPSK (FSM) mit Skalierung

Die konjugierte SCF für QPSK, sowie für PSK und QAM höherer Ordnung, ist im Wesentlichen null/Rauschen. Das bedeutet, wir können die konjugierte SCF verwenden, um das Vorhandensein von BPSK zu erkennen, selbst wenn viele QPSK/QAM-Signale damit überlappen!

Konjugierte SCF des Drei-Signal-Szenarios:

.. image:: ../_images/scf_conj_multiple_signals.svg
   :align: center
   :target: ../_images/scf_conj_multiple_signals.svg
   :alt: Konjugierte SCF von drei verschiedenen Signalen (FSM)

Wir können die beiden BPSK-Signale sehen, aber das QPSK-Signal taucht nicht auf.

********************************
FFT-Akkumulationsmethode (FAM)
********************************

FSM und TSM funktionieren gut, besonders wenn du einen bestimmten Satz von zyklischen Frequenzen berechnen möchtest. Es gibt jedoch eine noch effizientere SCF-Implementierung, die als FFT-Akkumulationsmethode (FAM) bekannt ist, die inhärent den vollständigen Satz von zyklischen Frequenzen berechnet. Es gibt auch eine ähnliche Technik namens `Strip Spectral Correlation Analyzer (SSCA) <https://cyclostationary.blog/2016/03/22/csp-estimators-the-strip-spectral-correlation-analyzer/>`_.

.. mermaid::

   flowchart TD
      A[Eingangssamples] --> B[In überlappende Fenster aufteilen]
      B --> C[Hanning-Fenster anwenden]
      C --> D[Erste FFT über jedes Fenster]
      D --> E[Frequenzverschiebung]
      E --> F[Zweite FFT]
      F --> G[Betragsquadrat]
      G --> H[SCF-Schätzung]

.. code-block:: python

    N = 2**14
    x = samples[0:N]
    Np = 512 # Anzahl der Eingangskanäle, sollte Potenz von 2 sein
    L = Np//4 # Versatz zwischen Punkten in derselben Spalte
    num_windows = (len(x) - Np) // L + 1
    Pe = int(np.floor(int(np.log(num_windows)/np.log(2))))
    P = 2**Pe
    N = L*P

    # Kanalbildung
    xs = np.zeros((num_windows, Np), dtype=complex)
    for i in range(num_windows):
        xs[i,:] = x[i*L:i*L+Np]
    xs2 = xs[0:P,:]

    # Fensterung
    xw = xs2 * np.tile(np.hanning(Np), (P,1))

    # erste FFT
    XF1 = np.fft.fftshift(np.fft.fft(xw))

    # Frequenzverschiebung nach unten
    f = np.arange(Np)/float(Np) - 0.5
    f = np.tile(f, (P, 1))
    t = np.arange(P)*L
    t = t.reshape(-1,1) # als Spaltenvektor
    t = np.tile(t, (1, Np))
    XD = XF1 * np.exp(-2j*np.pi*f*t)

    # Hauptberechnungen
    SCF = np.zeros((2*N, Np))
    Mp = N//Np//2
    for k in range(Np):
        for l in range(Np):
            XF2 = np.fft.fftshift(np.fft.fft(XD[:,k]*np.conj(XD[:,l]))) # zweite FFT
            i = (k + l) // 2
            a = int(((k - l) / Np + 1) * N)
            SCF[a-Mp:a+Mp, i] = np.abs(XF2[(P//2-Mp):(P//2+Mp)])**2

.. image:: ../_images/scf_fam.svg
   :align: center
   :target: ../_images/scf_fam.svg
   :alt: SCF mit der FFT-Akkumulationsmethode (FAM)

Zoom in den interessanten Bereich um 0,2 Hz:

.. image:: ../_images/scf_fam_zoomedin.svg
   :align: center
   :target: ../_images/scf_fam_zoomedin.svg
   :alt: Vergrößerte SCF mit der FAM

1D-Darstellung der SCF:

.. image:: ../_images/scf_fam_1d.svg
   :align: center
   :target: ../_images/scf_fam_1d.svg
   :alt: Zyklische Frequenz mit der FAM

Bei der FAM wird eine enorme Anzahl von Pixeln erzeugt. Um die Anzahl der Pixel in der zyklischen Frequenzachse zu reduzieren, kann Max-Pooling verwendet werden (erfordert ggf. :code:`pip install scikit-image`):

.. code-block:: python

    # Max-Pooling im zyklischen Bereich
    import skimage.measure
    print("Alte Form der SCF:", SCF.shape)
    SCF = skimage.measure.block_reduce(SCF, block_size=(16, 1), func=np.max)
    print("Neue Form der SCF:", SCF.shape)

Externe Ressourcen zur FAM:

* R.S. Roberts, W. A. Brown, and H. H. Loomis, Jr., "Computationally Efficient Algorithms for Cyclic Spectral Analysis," IEEE Signal Processing Magazine, April 1991, pp. 38-49. `Verfügbar hier <https://www.researchgate.net/profile/Faxin-Zhang-2/publication/353071530_Computationally_efficient_algorithms_for_cyclic_spectral_analysis/links/60e69d2d30e8e50c01eb9484/Computationally-efficient-algorithms-for-cyclic-spectral-analysis.pdf>`_
* Da Costa, Evandro Luiz. Detection and identification of cyclostationary signals. Diss. Naval Postgraduate School, 1996. `Verfügbar hier <https://apps.dtic.mil/sti/pdfs/ADA311555.pdf>`_
* Chads Blog-Beitrag zur FAM: https://cyclostationary.blog/2018/06/01/csp-estimators-the-fft-accumulation-method/

********************************
OFDM
********************************

Zyklostationarität ist bei OFDM-Signalen besonders stark aufgrund des zyklischen Präfix (CP) von OFDM, bei dem die letzten Samples jedes OFDM-Symbols kopiert und an den Anfang des OFDM-Symbols angehängt werden. Dies führt zu einer starken zyklischen Frequenz, die der OFDM-Symbollänge entspricht.

Simulation eines OFDM-Signals mit CP, 64 Subträgern, 25% CP und QPSK-Modulation auf jedem Subträger. Wir interpolieren um den Faktor 2, sodass die OFDM-Symbollänge in Samples (64 + (64*0,25)) * 2 = 160 Samples beträgt. Das bedeutet, wir sollten Spikes bei ganzzahligen Vielfachen von 1/160 erhalten:

.. code-block:: python

    from scipy.signal import resample
    N = 200000 # Anzahl der zu simulierenden Samples
    num_subcarriers = 64
    cp_len = num_subcarriers // 4 # Länge des zyklischen Präfix in Symbolen (25%)
    print("CP-Länge in Samples", cp_len*2) # 2x Interpolation
    print("OFDM-Symbollänge in Samples", (num_subcarriers+cp_len)*2)
    num_symbols = int(np.floor(N/(num_subcarriers+cp_len))) // 2
    print("Anzahl der OFDM-Symbole:", num_symbols)

    qpsk_mapping = {
        (0,0) : 1+1j,
        (0,1) : 1-1j,
        (1,0) : -1+1j,
        (1,1) : -1-1j,
    }
    bits_per_symbol = 2

    samples = np.empty(0, dtype=np.complex64)
    for _ in range(num_symbols):
        data = np.random.binomial(1, 0.5, num_subcarriers*bits_per_symbol) # 1en und 0en
        data = data.reshape((num_subcarriers, bits_per_symbol)) # nach Subträgern gruppieren
        symbol_freq = np.array([qpsk_mapping[tuple(b)] for b in data]) # OFDM beginnt im Frequenzbereich
        symbol_time = np.fft.ifft(symbol_freq)
        symbol_time = np.hstack([symbol_time[-cp_len:], symbol_time]) # CP anhängen
        samples = np.concatenate((samples, symbol_time))

    samples = resample(samples, len(samples)*2) # 2x interpolieren
    samples = samples[:N]

    # Rauschen hinzufügen
    SNR_dB = 5
    n = np.sqrt(np.var(samples) * 10**(-SNR_dB/10) / 2) * (np.random.randn(N) + 1j*np.random.randn(N))
    samples = samples + n

Ergebnisse mit :code:`alphas = np.arange(0, 0.02, 1e-5)`:

.. image:: ../_images/scf_freq_smoothing_ofdm_zoomed_in.svg
   :align: center
   :target: ../_images/scf_freq_smoothing_ofdm_zoomed_in.svg
   :alt: SCF von OFDM (FSM)

Drei Spikes sind klar erkennbar.

Externe Ressourcen zu OFDM im CSP-Kontext:

#. Sutton, Paul D., Keith E. Nolan, and Linda E. Doyle. "Cyclostationary signatures in practical cognitive radio applications." IEEE Journal on selected areas in Communications 26.1 (2008): 13-24. `Verfügbar hier <https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4413137&casa_token=81U1yMeRKMsAAAAA:6sQr9-VngNa2p_OW4zVyeQsRdUrZPkx3L-6ZPsH9LCo-pnTxs_AhjfAx27MFBbo4kl3YlgdkQJk&tag=1>`_

********************************************
Signalerkennung mit bekannter zyklischer Frequenz
********************************************

In manchen Anwendungen möchtest du CSP verwenden, um ein bereits bekanntes Signal/Wellenform zu erkennen, wie z. B. Varianten von 802.11, LTE, 5G usw. Wenn du die zyklische Frequenz des Signals kennst und deine Abtastrate weißt, musst du wirklich nur ein einziges Alpha und ein einziges Tau berechnen. Demnächst wird ein Beispiel für diese Art von Problem mit einer HF-Aufzeichnung von WLAN folgen.

******************
Externe Ressourcen
******************

#. Antonio Napolitanos Lehrbuch `Cyclostationary Processes and Time Series: Theory, Applications, and Generalizations <https://www.sciencedirect.com/book/monograph/9780081027080/cyclostationary-processes-and-time-series>`_
#. R.S. Roberts, W. A. Brown, and H. H. Loomis, Jr., "Computationally Efficient Algorithms for Cyclic Spectral Analysis," IEEE Signal Processing Magazine, April 1991, pp. 38-49. `Verfügbar hier <https://www.researchgate.net/profile/Faxin-Zhang-2/publication/353071530_Computationally_efficient_algorithms_for_cyclic_spectral_analysis/links/60e69d2d30e8e50c01eb9484/Computationally-efficient-algorithms-for-cyclic-spectral-analysis.pdf>`_
#. Da Costa, Evandro Luiz. Detection and identification of cyclostationary signals. Diss. Naval Postgraduate School, 1996. `Verfügbar hier <https://apps.dtic.mil/sti/pdfs/ADA311555.pdf>`_
#. `Chad Spooners zyklostationäres Blog/Website <https://cyclostationary.blog>`_
#. Sutton, Paul D., Keith E. Nolan, and Linda E. Doyle. "Cyclostationary signatures in practical cognitive radio applications." IEEE Journal on selected areas in Communications 26.1 (2008): 13-24. `Verfügbar hier <https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4413137&casa_token=81U1yMeRKMsAAAAA:6sQr9-VngNa2p_OW4zVyeQsRdUrZPkx3L-6ZPsH9LCo-pnTxs_AhjfAx27MFBbo4kl3YlgdkQJk&tag=1>`_
