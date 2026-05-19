.. _2d-beamforming-chapter:

####################
2D-Beamforming
####################

Dieses Kapitel erweitert das 1D-Beamforming/DOA-Kapitel auf 2D-Arrays. Wir beginnen mit einem einfachen Rechteck-Array und entwickeln die Steuervektorgleichung und den MVDR-Beamformer, dann arbeiten wir mit echten Daten von einem 3×5-Array. Abschließend verwenden wir das interaktive Werkzeug, um die Auswirkungen verschiedener Array-Geometrien und Elementabstände zu erkunden.

*************************************
Rechteck-Arrays und 2D-Beamforming
*************************************

Rechteck-Arrays (a.k.a. planare Arrays) bestehen aus einem 2D-Array von Elementen. Mit einer zusätzlichen Dimension erhalten wir etwas mehr Komplexität, aber dieselben Grundprinzipien gelten, und der schwierigste Teil ist die Visualisierung der Ergebnisse (z.B. keine einfachen Polarplots mehr, jetzt brauchen wir 3D-Oberflächenplots). Obwohl unser Array jetzt 2D ist, müssen wir nicht jede Datenstruktur um eine Dimension erweitern. Zum Beispiel behalten wir unsere Gewichte als 1D-Array komplexer Zahlen bei. Wir müssen jedoch die Positionen unserer Elemente in 2D darstellen. Wir verwenden weiterhin :code:`theta` für den Azimutwinkel, aber jetzt führen wir einen neuen Winkel :code:`phi` als Elevationswinkel ein. Es gibt viele sphärische Koordinatenkonventionen, aber wir verwenden folgende:

.. image:: ../_images/Spherical_Coordinates.svg
   :align: center
   :target: ../_images/Spherical_Coordinates.svg
   :alt: Sphärisches Koordinatensystem mit theta und phi

Was entspricht:

.. math::

 x = \sin(\theta) \cos(\phi)

 y = \cos(\theta) \cos(\phi)

 z = \sin(\phi)

Wir wechseln auch zu einer verallgemeinerten Steuervektorgleichung, die nicht spezifisch für eine Array-Geometrie ist:

.. math::

   s = e^{2j \pi \boldsymbol{p} u / \lambda}

wobei :math:`\boldsymbol{p}` die Menge der x/y/z-Elementpositionen in Metern ist (Größe :code:`Nr` x 3) und :math:`u` der Richtungseinheitsvektor in x/y/z ist (Größe 3x1). In Python sieht das so aus:

.. code-block:: python

 def steering_vector(pos, dir):
     #                           Nrx3  3x1
     return np.exp(2j * np.pi * pos @ dir / wavelength) # gibt Nr x 1 aus (Spaltenvektor)

Versuchen wir, diese verallgemeinerte Steuervektorgleichung mit einem einfachen ULA mit 4 Elementen zu verwenden, um die Verbindung zu dem herzustellen, was wir bisher gelernt haben. Wir stellen :code:`d` jetzt in Metern statt relativ zur Wellenlänge dar. Wir platzieren die Elemente entlang der y-Achse:

.. code-block:: python

 Nr = 4
 fc = 5e9
 wavelength = 3e8 / fc
 d = 0.5 * wavelength # in Metern

 # Elementpositionen als Liste von (x,y,z)-Koordinaten, auch wenn es nur ein ULA entlang der y-Achse ist
 pos = np.zeros((Nr, 3)) # Elementpositionen als Liste von x,y,z-Koordinaten in Metern
 for i in range(Nr):
     pos[i,0] = 0     # x-Position
     pos[i,1] = d * i # y-Position
     pos[i,2] = 0     # z-Position

Die folgende Grafik zeigt eine Draufsicht des ULA mit einem Beispiel-Theta von 20 Grad.

.. image:: ../_images/2d_beamforming_ula.svg
   :align: center
   :target: ../_images/2d_beamforming_ula.svg
   :alt: ULA mit Theta von 20 Grad

Das Einzige, was noch fehlt, ist die Verbindung unseres alten :code:`theta` mit diesem neuen Einheitsvektoransatz. Wir können :code:`dir` basierend auf :code:`theta` leicht berechnen. Wir wissen, dass die x- und z-Komponente unseres Einheitsvektors 0 sein wird, da wir uns noch im 1D-Raum befinden, und basierend auf unserer sphärischen Koordinatenkonvention ist die y-Komponente :code:`np.cos(theta)`, also lautet der vollständige Code :code:`dir = np.asmatrix([0, np.cos(theta_i), 0]).T`. An diesem Punkt solltest du die Verbindung zwischen unserer verallgemeinerten Steuervektorgleichung und der ULA-Steuervektorgleichung herstellen können. Probiere diesen neuen Code aus, wähle ein :code:`theta` zwischen 0 und 360 Grad (vergiss die Umrechnung in Bogenmaß!), und der Steuervektor sollte ein 4x1-Array sein.

Gehen wir jetzt zum 2D-Fall über. Wir platzieren unser Array in der X-Z-Ebene, mit Boresight horizontal in Richtung der positiven y-Achse zeigend (:math:`\theta = 0`, :math:`\phi = 0`). Wir verwenden denselben Elementabstand wie zuvor, haben jetzt aber insgesamt 16 Elemente:

.. code-block:: python

 # Jetzt auf 2D umsteigen, mit einem 4x4-Array mit halbem Wellenlängenabstand, also 16 Elemente insgesamt
 Nr = 16

 # Elementpositionen als Liste von x,y,z-Koordinaten in Metern, Array in der X-Z-Ebene
 pos = np.zeros((Nr,3))
 for i in range(Nr):
     pos[i,0] = d * (i % 4)  # x-Position
     pos[i,1] = 0            # y-Position
     pos[i,2] = d * (i // 4) # z-Position

Die Draufsicht unseres rechteckigen 4×4-Arrays:

.. image:: ../_images/2d_beamforming_element_pos.svg
   :align: center
   :target: ../_images/2d_beamforming_element_pos.svg
   :alt: Rechteckige Array-Elementpositionen

Um auf ein bestimmtes Theta und Phi zu zeigen, müssen wir diese Winkel in einen Einheitsvektor umrechnen. Wir können dieselbe verallgemeinerte Steuervektorgleichung wie zuvor verwenden, müssen aber jetzt den Einheitsvektor basierend auf Theta und Phi berechnen, mithilfe der Gleichungen am Anfang dieses Kapitels:

.. code-block:: python

 # In eine beliebige Richtung zeigen
 theta = np.deg2rad(60) # Azimutwinkel
 phi = np.deg2rad(30) # Elevationswinkel

 # Mit unserer sphärischen Koordinatenkonvention den Einheitsvektor berechnen:
 def get_unit_vector(theta, phi):  # Winkel in Bogenmaß
     return np.asmatrix([np.sin(theta) * np.cos(phi), # x-Komponente
                         np.cos(theta) * np.cos(phi), # y-Komponente
                         np.sin(phi)]).T              # z-Komponente

 dir = get_unit_vector(theta, phi)
 # dir ist 3x1
 # [[0.75     ]
 #  [0.4330127]
 #  [0.5      ]]

Jetzt verwenden wir unsere verallgemeinerte Steuervektorfunktion, um den Steuervektor zu berechnen:

.. code-block:: python

 s = steering_vector(pos, dir)

 # Konventionellen Beamformer verwenden (Gewichte gleich dem Steuervektor), Strahlmuster plotten
 w = s # 16x1 Gewichtsvektor

An dieser Stelle ist es erwähnenswert, dass wir beim Übergang von 1D zu 2D die Dimensionen von nichts geändert haben – wir haben nur nicht-null x/y/z-Komponenten, die Steuervektorgleichung ist dieselbe und die Gewichte sind immer noch ein 1D-Array. Es mag verlockend sein, die Gewichte als 2D-Array zusammenzustellen, damit sie visuell der Array-Geometrie entsprechen, aber das ist nicht notwendig und am besten als 1D zu belassen. Für jedes Element gibt es ein entsprechendes Gewicht, und die Liste der Gewichte ist in derselben Reihenfolge wie die Liste der Elementpositionen.

Das Strahlmuster dieser Gewichte zu visualisieren ist etwas komplizierter, da wir einen 3D-Plot oder eine 2D-Heatmap benötigen. Wir scannen :code:`theta` und :code:`phi`, um ein 2D-Array von Leistungspegeln zu erhalten, und plotten das dann mit :code:`imshow()`. Der folgende Code macht genau das, und das Ergebnis wird in der Abbildung unten gezeigt, zusammen mit einem Punkt an dem zuvor eingegebenen Winkel:

.. code-block:: python

    resolution = 100 # Anzahl der Punkte in jeder Richtung
    theta_scan = np.linspace(-np.pi/2, np.pi/2, resolution) # Azimutwinkel
    phi_scan = np.linspace(-np.pi/4, np.pi/4, resolution) # Elevationswinkel
    results = np.zeros((resolution, resolution)) # 2D-Array zum Speichern der Ergebnisse
    for i, theta_i in enumerate(theta_scan):
        for j, phi_i in enumerate(phi_scan):
            a = steering_vector(pos, get_unit_vector(theta_i, phi_i)) # Arrayfaktor
            results[i, j] = np.abs(w.conj().T @ a)[0,0] # Leistung im Signal, linear sieht besser aus
    plt.imshow(results.T, extent=(theta_scan[0]*180/np.pi, theta_scan[-1]*180/np.pi, phi_scan[0]*180/np.pi, phi_scan[-1]*180/np.pi), origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Leistung [linear]')
    plt.scatter(theta*180/np.pi, phi*180/np.pi, color='red', s=50) # Punkt an richtigem Theta/Phi hinzufügen
    plt.xlabel('Azimutwinkel [Grad]')
    plt.ylabel('Elevationswinkel [Grad]')
    plt.show()

.. image:: ../_images/2d_beamforming_2dplot.svg
   :align: center
   :target: ../_images/2d_beamforming_2dplot.svg
   :alt: 3D-Plot des Strahlmusters

Simulieren wir jetzt einige echte Abtastwerte; wir fügen zwei Ton-Jammer hinzu, die aus verschiedenen Richtungen ankommen:

.. code-block:: python

 N = 10000 # Anzahl der zu simulierenden Abtastwerte

 jammer1_theta = np.deg2rad(-30)
 jammer1_phi = np.deg2rad(10)
 jammer1_dir = get_unit_vector(jammer1_theta, jammer1_phi)
 jammer1_s = steering_vector(pos, jammer1_dir) # Nr x 1
 jammer1_tone = np.exp(2j*np.pi*0.1*np.arange(N)).reshape(1,-1) # als Zeilenvektor

 jammer2_theta = np.deg2rad(10)
 jammer2_phi = np.deg2rad(50)
 jammer2_dir = get_unit_vector(jammer2_theta, jammer2_phi)
 jammer2_s = steering_vector(pos, jammer2_dir)
 jammer2_tone = np.exp(2j*np.pi*0.2*np.arange(N)).reshape(1,-1) # als Zeilenvektor

 noise = np.random.normal(0, 1, (Nr, N)) + 1j * np.random.normal(0, 1, (Nr, N)) # komplexes Gaußsches Rauschen
 r = jammer1_s @ jammer1_tone + jammer2_s @ jammer2_tone + noise # erzeugt 16 x 10000 Matrix

Spaßeshalber berechnen wir die MVDR-Beamformer-Gewichte in Richtung des zuvor verwendeten Theta und Phi (ein Einheitsvektor in diese Richtung ist noch als :code:`dir` gespeichert):

.. code-block:: python

 s = steering_vector(pos, dir) # 16 x 1
 R = np.cov(r) # Kovarianzmatrix, 16 x 16
 Rinv = np.linalg.pinv(R)
 w = (Rinv @ s)/(s.conj().T @ Rinv @ s) # MVDR/Capon-Gleichung

Anstatt das Strahlmuster im wenig übersichtlichen 3D-Plot anzusehen, verwenden wir eine alternative Methode, um zu prüfen, ob diese Gewichte sinnvoll sind: Wir bewerten die Antwort der Gewichte in verschiedene Richtungen und berechnen die Leistung in dB. Beginnen wir mit der Richtung, in die wir zeigen:

.. code-block:: python

 # Leistung in der Richtung, in die wir zeigen (theta=60, phi=30, noch als dir gespeichert):
 a = steering_vector(pos, dir) # Arrayfaktor
 resp = w.conj().T @ a # Skalar
 print("Leistung in Zeige-Richtung:", 10*np.log10(np.abs(resp)[0,0]), 'dB')

Dies gibt 0 dB aus, was wir erwarten, da MVDRs Ziel ist, in der gewünschten Richtung Einheitsleistung zu erzielen. Jetzt prüfen wir die Leistung in Richtung der zwei Jammer sowie in einer zufälligen Richtung und einer Richtung, die einen Grad von unserer gewünschten Richtung abweicht (derselbe Code, nur :code:`dir` aktualisieren). Die Ergebnisse sind in der folgenden Tabelle dargestellt:

.. list-table::
   :widths: 70 30
   :header-rows: 1

   * - Zeige-Richtung
     - Gewinn
   * - :code:`dir` (für MVDR-Gewichtsberechnung verwendete Richtung)
     - 0 dB
   * - Jammer 1
     - -17,488 dB
   * - Jammer 2
     - -18,551 dB
   * - 1 Grad weg von :code:`dir` in :math:`\theta` und :math:`\phi`
     - -0,00683 dB
   * - Eine zufällige Richtung
     - -10,591 dB

Deine Ergebnisse können aufgrund des zufälligen Rauschens variieren. Die wichtigste Erkenntnis ist: die Jammer befinden sich in einem Null und haben sehr niedrige Leistung, die 1-Grad-Abweichung von :code:`dir` liegt leicht unter 0 dB, befindet sich aber noch in der Hauptkeule, und eine zufällige Richtung liegt unter 0 dB, aber über den Jammern. Beachte, dass du mit MVDR einen Gewinn von 0 dB für die Hauptkeule erhältst, aber mit dem konventionellen Beamformer würdest du :math:`10 \log_{10}(Nr)` erhalten, also etwa 12 dB für unser 16-Element-Array – das zeigt einen der Kompromisse von MVDR.

Den Code für diesen Abschnitt findest du `hier <https://github.com/777arc/PySDR/blob/master/figure-generating-scripts/doa_2d.py>`_.

**********************************************
Signale von einem echten 2D-Array verarbeiten
**********************************************

In diesem Abschnitt arbeiten wir mit echten Daten, die von einem 3×5-Array aufgezeichnet wurden, das mit einer `QUAD-MxFE <https://www.analog.com/en/resources/evaluation-hardware-and-software/evaluation-boards-kits/quad-mxfe.html#eb-overview>`_-Plattform von Analog Devices erstellt wurde, die bis zu 16 Sende- und Empfangskanäle unterstützt (wir haben nur 15 verwendet und nur im Empfangsmodus). Es werden zwei Aufzeichnungen bereitgestellt: Die erste enthält einen Sender am Boresight des Arrays, den wir zur Kalibrierung verwenden. Die zweite Aufzeichnung enthält zwei Sender in verschiedenen Richtungen, die wir für Beamforming- und DOA-Tests verwenden.

- `IQ-Aufzeichnung von nur C <https://github.com/777arc/RADAR-2025-Beamforming-Labs/raw/refs/heads/main/Lab%207%20-%202D%20Rectangular%20Array/C_only_capture1.npy>`_ (zur Kalibrierung verwendet, da C am Boresight ist)
- `IQ-Aufzeichnung von B und D <https://github.com/777arc/RADAR-2025-Beamforming-Labs/raw/refs/heads/main/Lab%207%20-%202D%20Rectangular%20Array/DandB_capture1.npy>`_ (für Beamforming/DOA-Tests verwendet)

Das QUAD-MxFE wurde auf 2,8 GHz abgestimmt und alle Sender verwendeten einen einfachen Ton innerhalb der Beobachtungsbandbreite. Interessant an dieser DSP-Verarbeitung ist, dass die Abtastrate tatsächlich keine Rolle spielt; keine der verwendeten Array-Verarbeitungstechniken hängt von der Abtastrate ab, sie setzen nur voraus, dass das Signal irgendwo im Basisbandsignal liegt. Die DSP hängt von der Mittenfrequenz ab, weil die Phasenverschiebung zwischen Elementen von der Frequenz und dem Ankunftswinkel abhängt. Das ist das Gegenteil der meisten anderen Signalverarbeitung, wo die Abtastrate wichtig ist, aber die Mittenfrequenz nicht.

Wir können diese Aufzeichnungen mit folgendem Code in Python laden:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    r = np.load("DandB_capture1.npy")[0:15] # 16. Element nicht angeschlossen, aber trotzdem aufgezeichnet
    r_cal = np.load("C_only_capture1.npy")[0:15] # nur das Kalibrierungssignal (am Boresight)

Der Abstand zwischen Antennen betrug 0,051 Meter. Wir können die Elementpositionen als Liste von x,y,z-Koordinaten in Metern darstellen. Wir platzieren das Array in der X-Z-Ebene, da das Array vertikal montiert war (mit Boresight horizontal zeigend).

.. code-block:: python

	fc = 2.8e9 # Mittenfrequenz in Hz
	d = 0.051 # Abstand zwischen Antennen in Metern
	wavelength = 3e8 / fc
	Nr = 15
	rows = 3
	cols = 5

	# Elementpositionen als Liste von x,y,z-Koordinaten in Metern
	pos = np.zeros((Nr, 3))
	for i in range(Nr):
		pos[i,0] = d * (i % cols)  # x-Position
		pos[i,1] = 0 # y-Position
		pos[i,2] = d * (i // cols) # z-Position

	# Elementpositionen plotten und beschriften
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	ax.scatter(pos[:,0], pos[:,1], pos[:,2], 'o')
	# Indizes beschriften
	for i in range(Nr):
		ax.text(pos[i,0], pos[i,1], pos[i,2], str(i), fontsize=10)
	plt.xlabel("X-Position [m]")
	plt.ylabel("Y-Position [m]")
	ax.set_zlabel("Z-Position [m]")
	plt.grid()
	plt.show()

Der Plot beschriftet jedes Element mit seinem Index, der der Reihenfolge der Elemente in den aufgezeichneten :code:`r`- und :code:`r_cal`-IQ-Abtastwerten entspricht.

.. image:: ../_images/2d_array_element_positions.svg
   :align: center
   :target: ../_images/2d_array_element_positions.svg
   :alt: 2D-Array-Elementpositionen

Die Kalibrierung wird nur mit den :code:`r_cal`-Abtastwerten durchgeführt, die nur mit dem Sender am Boresight aufgezeichnet wurden. Das Ziel ist, die Phasen- und Amplitudenoffsets für jedes Element zu finden. Bei perfekter Kalibrierung und unter der Annahme, dass der Sender genau am Boresight war, sollten alle einzelnen Empfangselemente dasselbe Signal empfangen – alle in Phase miteinander und mit der gleichen Amplitude. Aufgrund von Unvollkommenheiten im Array/Kabeln/Antennen hat jedes Element jedoch einen anderen Phasen- und Amplitudenoffset. Der Kalibrierungsprozess besteht darin, diese Offsets zu finden, die wir später auf die :code:`r`-Abtastwerte anwenden, bevor wir versuchen, Array-Verarbeitung darauf durchzuführen.

Es gibt viele Möglichkeiten, eine Kalibrierung durchzuführen, aber wir verwenden eine Methode, die die Eigenwertzersetzung der Kovarianzmatrix beinhaltet. Der Eigenvektor entsprechend dem größten Eigenwert ist derjenige, der hoffentlich das empfangene Signal repräsentiert, und wir verwenden ihn, um die Phasenoffsets für jedes Element zu finden, indem wir einfach die Phase jedes Elements des Eigenvektors nehmen und auf das erste Element normieren, das wir als Referenzelement behandeln. Die Amplitudenkalibrierung verwendet nicht den Eigenvektor, sondern die mittlere Amplitude des empfangenen Signals für jedes Element.

.. code-block:: python

	# Kovarianzmatrix berechnen, Nr x Nr
	R_cal = r_cal @ r_cal.conj().T

    # Eigenwertzersetzung, v[:,i] ist der Eigenvektor entsprechend dem Eigenwert w[i]
	w, v = np.linalg.eig(R_cal)

	# Eigenwerte plotten, um sicherzustellen, dass es nur einen großen gibt
	w_dB = 10*np.log10(np.abs(w))
	w_dB -= np.max(w_dB) # normalisieren
	fig, (ax1) = plt.subplots(1, 1, figsize=(7, 3))
	ax1.plot(w_dB, '.-')
	ax1.set_xlabel('Index')
	ax1.set_ylabel('Eigenwert [dB]')
	plt.show()

	# Maximalen Eigenvektor zur Kalibrierung verwenden
	v_max = v[:, np.argmax(np.abs(w))]
	mags = np.mean(np.abs(r_cal), axis=1)
	mags = mags[0] / mags # auf erstes Element normieren
	phases = np.angle(v_max)
	phases = phases[0] - phases # auf erstes Element normieren
	cal_table = mags * np.exp(1j * phases)
	print("cal_table", cal_table)

Unten ist der Plot der Eigenwertverteilung. Wir möchten sicherstellen, dass es nur einen großen Wert gibt und die anderen klein sind, was ein empfangenes Signal repräsentiert. Störer oder Mehrwege können den Kalibrierungsprozess beeinträchtigen.

.. image:: ../_images/2d_array_eigenvalues.svg
   :align: center
   :target: ../_images/2d_array_eigenvalues.svg
   :alt: 2D-Array-Eigenwertverteilung

Die Kalibrierungstabelle ist eine Liste komplexer Zahlen, eine für jedes Element, die die Phasen- und Amplitudenoffsets darstellen. Das erste Element ist das Referenzelement und ist immer 1,0 + 0j. Die übrigen Elemente sind die Offsets für jedes Element in derselben Reihenfolge wie :code:`pos`.

.. code-block:: python

	[1.        +0.j          0.99526771+0.76149029j -0.91754588-0.66825262j
	-0.96840297+0.37251012j  0.87866849+0.40446665j  0.56040169+1.50499875j
	-0.80109196-1.29299264j -1.28464742-0.31133052j  1.26622038+0.46047599j
	 2.01855809+9.77121302j -0.29249322-1.09413205j -1.0372309 -0.17983522j
	-0.70614339+0.78682873j -0.75612972+5.67234809j  1.00032754-0.60824109j]


Wir können diese Offsets auf alle von dem Array aufgezeichneten Abtastwerte anwenden, indem wir einfach jedes Element der Abtastwerte mit dem entsprechenden Element der Kalibrierungstabelle multiplizieren:

.. code-block:: python

	# Kalibrierungsoffsets auf r anwenden
	for i in range(Nr):
		r[i, :] *= cal_table[i]

Als Randnotiz: Deshalb haben wir die Offsets mit :code:`mags[0] / mags` und :code:`phases[0] - phases` berechnet. Hätten wir die Reihenfolge umgekehrt, müssten wir eine Division zum Anwenden der Offsets durchführen, aber wir bevorzugen die Multiplikation.

Als nächstes führen wir die DOA-Schätzung mit dem MUSIC-Algorithmus durch. Wir verwenden die zuvor definierten Funktionen :code:`steering_vector()` und :code:`get_unit_vector()`, um den Steuervektor für jedes Element des Arrays zu berechnen, und dann den MUSIC-Algorithmus, um die DOA der zwei Sender in den :code:`r`-Abtastwerten zu schätzen. Der MUSIC-Algorithmus wurde im vorherigen Kapitel besprochen.

.. code-block:: python

	# DOA mit MUSIC
	resolution = 400 # Anzahl der Punkte in jeder Richtung
	theta_scan = np.linspace(-np.pi/2, np.pi/2, resolution) # Azimutwinkel
	phi_scan = np.linspace(-np.pi/4, np.pi/4, resolution) # Elevationswinkel
	results = np.zeros((resolution, resolution)) # 2D-Array für Ergebnisse
	R = np.cov(r) # Kovarianzmatrix, 15 x 15
	Rinv = np.linalg.pinv(R)
	expected_num_signals = 4
	w, v = np.linalg.eig(R) # Eigenwertzersetzung
	eig_val_order = np.argsort(np.abs(w))
	v = v[:, eig_val_order] # Eigenvektoren sortieren
	V = np.zeros((Nr, Nr - expected_num_signals), dtype=np.complex64) # Rauschunterraum
	for i in range(Nr - expected_num_signals):
		V[:, i] = v[:, i]
	for i, theta_i in enumerate(theta_scan):
		for j, phi_i in enumerate(phi_scan):
			dir_i = get_unit_vector(-1*theta_i, phi_i) # TODO: -1* war nötig, um der Realität zu entsprechen
			s = steering_vector(pos, dir_i) # 15 x 1
			music_metric = 1 / (s.conj().T @ V @ V.conj().T @ s)
			music_metric = np.abs(music_metric).squeeze()
			music_metric = np.clip(music_metric, 0, 2) # Nützlich für ABCD
			results[i, j] = music_metric

Unsere Ergebnisse sind 2D, da das Array 2D ist. Wir können entweder einen 3D-Plot oder eine 2D-Heatmap verwenden. Zuerst ein 3D-Plot mit Elevation auf einer Achse und Azimut auf der anderen:

.. code-block:: python

	# 3D-Az-El-DOA-Ergebnisse
	results = 10*np.log10(results) # in dB umrechnen
	results[results < -20] = -20 # z-Achse auf bestimmten dB-Pegel beschneiden
	fig, ax = plt.subplots(subplot_kw={"projection": "3d", "computed_zorder": False})
	surf = ax.plot_surface(np.rad2deg(theta_scan[:,None]),
							np.rad2deg(phi_scan[None,:]),
							results,
							cmap='viridis')
	ax.set_xlabel('Azimut (theta)')
	ax.set_ylabel('Elevation (phi)')
	ax.set_zlabel('Leistung [dB]')
	fig.savefig('../_images/2d_array_3d_doa_plot.svg', bbox_inches='tight')
	plt.show()

.. image:: ../_images/2d_array_3d_doa_plot.png
   :align: center
   :scale: 30%
   :target: ../_images/2d_array_3d_doa_plot.png
   :alt: 3D-DOA-Plot

Je nach Situation kann es schwierig sein, Zahlen aus einem 3D-Plot abzulesen, daher können wir auch eine 2D-Heatmap mit :code:`imshow()` erstellen:

.. code-block:: python

	# 2D-Az-El-Heatmap
	extent=(np.min(theta_scan)*180/np.pi,
			np.max(theta_scan)*180/np.pi,
			np.min(phi_scan)*180/np.pi,
			np.max(phi_scan)*180/np.pi)
	plt.imshow(results.T, extent=extent, origin='lower', aspect='auto', cmap='viridis')
	plt.colorbar(label='Leistung [linear]')
	plt.xlabel('Theta (Azimut, Grad)')
	plt.ylabel('Phi (Elevation, Grad)')
	plt.savefig('../_images/2d_array_2d_doa_plot.svg', bbox_inches='tight')
	plt.show()

.. image:: ../_images/2d_array_2d_doa_plot.svg
   :align: center
   :target: ../_images/2d_array_2d_doa_plot.svg
   :alt: 2D-DOA-Plot

Anhand dieses 2D-Plots können wir den geschätzten Azimut und die Elevation der beiden Sender leicht ablesen (und sehen, dass es nur zwei waren). Basierend auf dem Testaufbau, der zur Erstellung dieser Aufzeichnung verwendet wurde, stimmen diese Ergebnisse mit der Realität überein. Der genaue Azimut und die Elevation der Sender wurden nie tatsächlich gemessen, da dafür spezialisierte Ausrüstung erforderlich wäre.

Als Übung versuche den konventionellen Beamformer sowie MVDR zu verwenden und vergleiche die Ergebnisse mit MUSIC.

Den vollständigen Code für diesen Abschnitt findest du `hier <https://github.com/777arc/PySDR/blob/master/figure-generating-scripts/2d_array_recording.py>`_.

***********************
Interaktives Entwurfswerkzeug
***********************

Das folgende interaktive Werkzeug wurde von `Jason Durbin <https://www.linkedin.com/in/jasondurbin/>`_ erstellt, einem freiberuflichen Phased-Array-Ingenieur, der freundlicherweise seine Einbettung in PySDR erlaubt hat. Besuche gerne das `vollständige Projekt <https://jasondurbin.github.io/PhasedArrayVisualizer>`_ oder sein `Beratungsunternehmen <https://neonphysics.com/>`_. Dieses Werkzeug ermöglicht das Ändern der Geometrie eines Phased-Arrays, des Elementabstands, der Steuerposition, das Hinzufügen von Nebenkeulen-Tapering und weitere Funktionen.

Einige Details zu diesem Werkzeug: Antennenelemente werden als isotrop angenommen. Die Richtwirkungsberechnung nimmt jedoch Halbhemisphären-Strahlung an (z.B. keine Rückzipfel). Daher wird die berechnete Richtwirkung 3 dBi höher sein als bei reiner isotroper Verwendung. Das Gitter kann durch Erhöhung von Theta/Phi, U/V oder Azimut/Elevation-Punkten feiner gemacht werden. Klicken (oder langes Drücken) auf Elemente in den Phasen-/Dämpfungsplots ermöglicht das manuelle Einstellen von Phase/Dämpfung. Außerdem ermöglicht das Dämpfungs-Popup das Deaktivieren von Elementen.

.. raw:: html

	<input type="text" id="pa-atten-manual" hidden />
	<input type="text" id="pa-phase-manual" hidden />
	<div class="text-group">
		<div class="pa-settings">
			<div id="pa-geometry-controls">
				<h3>Geometry</h3>
			</div>
			<div>
				<h3>Steering</h3>
				<select id="pa-steering-domain" style="width:100%;"></select>
				<div class="form-group" id="pa-theta-div">
					<label for="pa-theta">Theta (deg)</label>
					<input type="number" min="-90" max="90" value="0" id="pa-theta" name="pa-theta" />
				</div>
				<div class="form-group" id="pa-phi-div">
					<label for="pa-phi">Phi (deg)</label>
					<input type="number" min="-90" max="90" value="0" id="pa-phi" name="pa-phi" />
				</div>
			</div>
			<div>
				<h3>Taper(s)</h3>
				<div class="form-group" id="pa-taper-sampling-div">
					<label for="pa-taper-sampling">Sampling</label>
					<select id="pa-taper-sampling"><option>X & Y</option><option>Radial</option></select>
				</div>
				<div id="pa-taper-x-group" style="margin: 5px 0px;"></div>
				<div id="pa-taper-y-group" style="margin: 5px 0px;"></div>
			</div>
			<div>
				<h3>Quantization</h3>
				<div class="form-group" id="pa-phase-bits-div">
					<label for="pa-phase-bits">Phase Bits</label>
					<input type="number" min="0" max="10" value="0" step="1" id="pa-phase-bits" name="pa-phase-bits" />
				</div>
				<div class="form-group" id="pa-atten-bits-div">
					<label for="pa-atten-bits">Atten. Bits</label>
					<input type="number" min="0" max="10" value="0" step="1" id="pa-atten-bits" name="pa-atten-bits" />
				</div>
				<div class="form-group" id="pa-atten-lsb-div">
					<label for="pa-atten-lsb">Atten. LSB (dB)</label>
					<input type="number" min="0" max="5" value="0.5" step="0.25" id="pa-atten-lsb" name="pa-atten-lsb" />
				</div>
				<div class="form-group" style="font-size:0.7em;font-style: italic;">
					0 bits would be no quantization.
				</div>
			</div>
		</div>
		<div class="pa-update-div">
			<div style="display:flex; gap: 4px; justify-content: center;"><button id="pa-refresh">Update</button><button id="pa-reset">Reset</button></div>
			<progress id="pa-progress" max="100" value="70"></progress>
			<div id="pa-status">Loading...</div>
		</div>
	</div>
	<div class="canvas-grid">
		<div class="canvas-container">
			<div class="canvas-header"><h2>Element<br>Phase</h2><span>&nbsp;</span></div>
			<div class="canvas-wrapper">
				<canvas id="pa-geometry-phase-canvas" class="canvas-grid"></canvas>
			</div>
			<div class="canvas-footer footer-group">
				<div>
					<label for="pa-geometry-phase-colormap">Colormap</label>
					<select id="pa-geometry-phase-colormap" name="pa-geometry-phase-colormap"></select>
				</div>
			</div>
		</div>
		<div class="canvas-container">
			<div class="canvas-header"><h2>Element Attenuation</h2><span>&nbsp;</span></div>
			<div class="canvas-wrapper">
				<canvas id="pa-geometry-magnitude-canvas" class="canvas-grid"></canvas>
			</div>
			<div class="canvas-footer footer-group">
				<div>
					<label for="pa-atten-scale">Scale</label>
					<input type="number" max="200" min="5" value="40" id="pa-atten-scale" name="pa-atten-scale">
				</div>
				<div>
					<label for="pa-geometry-magnitude-colormap">Colormap</label>
					<select id="pa-geometry-magnitude-colormap" name="pa-geometry-magnitude-colormap"></select>
				</div>
			</div>
		</div>
		<div class="canvas-container">
			<div class="canvas-header"><h2>2-D Radiation Pattern</h2><span id="pa-directivity-max">&nbsp;</span></div>
			<div class="canvas-wrapper">
				<canvas id="pa-farfield-canvas-2d" class="canvas-grid"></canvas>
			</div>
			<div class="canvas-footer">
				<div class="footer-group">
					<div>
						<label for="pa-farfield-domain">Domain</label>
						<select id="pa-farfield-domain"></select>
					</div>
					<div>
						<label for="pa-farfield-2d-scale">Scale</label>
						<input type="number" max="200" min="5" value="40" id="pa-farfield-2d-scale" name="pa-farfield-2d-scale">
					</div>
					<div>
						<label for="pa-farfield-2d-colormap">Colormap</label>
						<select id="pa-farfield-2d-colormap" name="pa-farfield-2d-colormap"></select>
					</div>
					<div>
						<label for="pa-farfield-ax1-points">Theta Points</label>
						<input type="number" min="11" max="513" value="257" size="6" id="pa-farfield-ax1-points" name="pa-farfield-ax1-points">
					</div>
					<div>
						<label for="pa-farfield-ax2-points">Phi Points</label>
						<input type="number" min="11" max="513" value="257" size="6" id="pa-farfield-ax2-points" name="pa-farfield-ax2-points">
					</div>
				</div>
			</div>
		</div>
	</div>
	<div class="canvas-full">
		<div class="canvas-container">
			<div class="canvas-header"><h2>1-D Pattern Cuts</h2></div>
			<div class="canvas-wrapper">
				<canvas id="pa-farfield-canvas-1d"></canvas>
			</div>
			<div class="canvas-footer">
				<div class="canvas-legend">
					<span class="legend-item" data-phi="0" data-v="0.0" data-az="0.0" data-visible="true">Phi = 0 deg</span>
					<span class="legend-item" data-phi="90" data-u="0.0" data-el="0.0" data-visible="true">Phi = 90 deg</span>
					<span style='font-size:0.8em'>Click to hide/show trace.</span>
				</div>
				<div>
					<label for="pa-farfield-1d-scale">Scale</label>
					<input type="number" max="200" min="5" value="40" id="pa-farfield-1d-scale" name="pa-farfield-1d-scale">
					<label for="pa-farfield-1d-colormap">Colormap</label>
					<select id="pa-farfield-1d-colormap" name="pa-farfield-1d-colormap"></select>
				</div>
			</div>
		</div>
	</div>
	<div class="canvas-full">
		<div class="canvas-container">
			<div class="canvas-header"><h2>Taper</h2></div>
			<div class="canvas-wrapper">
				<canvas id="pa-taper-canvas-1d"></canvas>
			</div>
			<div class="canvas-footer">
				<div class="canvas-legend">
					<span class="legend-item" data-axis="x" data-visible="true">X-Axis</span>
					<span class="legend-item" data-axis="y" data-visible="true">Y-Axis</span>
					<span style='font-size:0.8em'>Click to hide/show trace.</span>
				</div>
				<div>
					<label for="pa-taper-1d-colormap">Colormap</label>
					<select id="pa-taper-1d-colormap" name="pa-taper-1d-colormap"></select>
				</div>
			</div>
		</div>
	</div>
