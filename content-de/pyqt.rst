.. _pyqt-chapter:

##########################
Echtzeit-GUIs mit PyQt
##########################

In diesem Kapitel lernst du, wie du mit Python mithilfe von PyQt, den Python-Bindings für Qt, Echtzeit-Benutzeroberflächen (GUIs) erstellst. Im Rahmen dieses Kapitels bauen wir einen Spektrumanalysator mit Zeit-, Frequenz- und Spektrogramm/Wasserfall-Darstellungen sowie Eingabe-Widgets zum Einstellen der verschiedenen SDR-Parameter. Das Beispiel unterstützt PlutoSDR, USRP oder einen reinen Simulationsmodus.

****************
Einführung
****************

Qt (ausgesprochen „Cute") ist ein Framework zum Erstellen von GUI-Anwendungen, die unter Linux, Windows, macOS und sogar Android laufen können. Es ist ein sehr leistungsfähiges Framework, das in vielen kommerziellen Anwendungen eingesetzt wird, und ist für maximale Performance in C++ geschrieben. PyQt sind die Python-Bindings für Qt und bieten eine Möglichkeit, GUI-Anwendungen in Python zu erstellen, während die Leistung des effizienten C++-basierten Frameworks genutzt wird. In diesem Kapitel lernst du, wie du mit PyQt einen Echtzeit-Spektrumanalysator erstellen kannst, der mit einem SDR (oder mit einem simulierten Signal) verwendet werden kann. Der Spektrumanalysator verfügt über Zeit-, Frequenz- und Spektrogramm/Wasserfall-Darstellungen sowie Eingabe-Widgets zum Einstellen der verschiedenen SDR-Parameter. Wir verwenden `PyQtGraph <https://www.pyqtgraph.org/>`_, eine separate Bibliothek, die auf PyQt aufbaut, für die Darstellungen. Auf der Eingabeseite verwenden wir Schieberegler, Dropdown-Menüs und Druckknöpfe. Das Beispiel unterstützt PlutoSDR, USRP oder reinen Simulationsmodus. Obwohl der Beispielcode PyQt6 verwendet, ist jede einzelne Zeile identisch mit PyQt5 (abgesehen vom :code:`import`), aus API-Perspektive hat sich zwischen den beiden Versionen sehr wenig geändert. Dieses Kapitel ist naturgemäß sehr Python-Code-lastig, da wir anhand von Beispielen erklären. Am Ende dieses Kapitels wirst du mit den Bausteinen vertraut sein, die du zur Erstellung deiner eigenen interaktiven SDR-Anwendung benötigst!

****************
Qt-Überblick
****************

Qt ist ein sehr großes Framework, und wir werden nur an der Oberfläche kratzen, was es leisten kann. Es gibt jedoch einige Schlüsselkonzepte, die wichtig sind, wenn man mit Qt/PyQt arbeitet:

- **Widgets**: Widgets sind die Bausteine einer Qt-Anwendung und dienen zur Erstellung der GUI. Es gibt viele verschiedene Widget-Typen, darunter Schaltflächen, Schieberegler, Labels und Diagramme. Widgets können in Layouts angeordnet werden, die bestimmen, wie sie auf dem Bildschirm positioniert werden.

- **Layouts**: Layouts dienen zur Anordnung von Widgets in einem Fenster. Es gibt verschiedene Layout-Typen, darunter horizontale, vertikale, Gitter- und Formularlayouts. Layouts werden verwendet, um komplexe GUIs zu erstellen, die auf Änderungen der Fenstergröße reagieren.

- **Signals und Slots**: Signals und Slots sind eine Möglichkeit, zwischen verschiedenen Teilen einer Qt-Anwendung zu kommunizieren. Ein Signal wird von einem Objekt ausgegeben, wenn ein bestimmtes Ereignis eintritt, und mit einem Slot verbunden, der eine Callback-Funktion ist, die aufgerufen wird, wenn das Signal ausgesendet wird. Signals und Slots werden verwendet, um eine ereignisgesteuerte Struktur in einer Qt-Anwendung zu erzeugen und die GUI reaktionsfähig zu halten.

- **Style Sheets**: Style Sheets werden verwendet, um das Erscheinungsbild von Widgets in einer Qt-Anwendung anzupassen. Style Sheets werden in einer CSS-ähnlichen Sprache geschrieben und können verwendet werden, um Farbe, Schriftart und Größe von Widgets zu ändern.

- **Grafiken**: Qt verfügt über ein leistungsstarkes Grafik-Framework, das zur Erstellung benutzerdefinierter Grafiken in einer Qt-Anwendung verwendet werden kann. Das Grafik-Framework enthält Klassen zum Zeichnen von Linien, Rechtecken, Ellipsen und Text sowie Klassen zur Behandlung von Maus- und Tastaturereignissen.

- **Multithreading**: Qt hat eingebaute Unterstützung für Multithreading und stellt Klassen für die Erstellung von Worker-Threads bereit, die im Hintergrund laufen. Multithreading wird verwendet, um langwierige Operationen in einer Qt-Anwendung auszuführen, ohne den Haupt-GUI-Thread zu blockieren.

- **OpenGL**: Qt hat eingebaute Unterstützung für OpenGL und stellt Klassen für die Erstellung von 3D-Grafiken in einer Qt-Anwendung bereit. OpenGL wird für Anwendungen verwendet, die leistungsstarke 3D-Grafiken erfordern. In diesem Kapitel konzentrieren wir uns ausschließlich auf 2D-Anwendungen.

*************************
Grundlegendes App-Layout
*************************

Bevor wir uns mit den verschiedenen Qt-Widgets befassen, schauen wir uns das Layout einer typischen Qt-Anwendung an. Eine Qt-Anwendung besteht aus einem Hauptfenster, das ein zentrales Widget enthält, das wiederum den Hauptinhalt der Anwendung enthält. Mit PyQt können wir eine minimale Qt-Anwendung erstellen, die nur einen einzigen QPushButton enthält:

.. code-block:: python

    from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton

    # QMainWindow ableiten, um das Hauptfenster der Anwendung anzupassen
    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()

            # Beispiel-GUI-Komponente
            example_button = QPushButton('Push Me')
            def on_button_click():
                print("beep")
            example_button.clicked.connect(on_button_click)

            self.setCentralWidget(example_button)

    app = QApplication([])
    window = MainWindow()
    window.show() # Fenster sind standardmäßig ausgeblendet
    app.exec() # Ereignisschleife starten

Probiere den Code selbst aus – du wirst wahrscheinlich :code:`pip install PyQt6` ausführen müssen. Beachte, dass die allerletzte Zeile blockierend ist – alles, was du nach dieser Zeile hinzufügst, wird erst ausgeführt, wenn du das Fenster schließt. Der QPushButton, den wir erstellen, hat sein :code:`clicked`-Signal mit einer Callback-Funktion verbunden, die „beep" auf der Konsole ausgibt.

*******************************
Anwendung mit Worker-Thread
*******************************

Das minimale Beispiel oben hat ein Problem: Es lässt uns keinen Platz für SDR/DSP-orientierten Code. Das :code:`__init__` von :code:`MainWindow` ist der Ort, an dem die GUI konfiguriert und Callbacks definiert werden, aber du möchtest auf keinen Fall anderen Code (wie SDR- oder DSP-Code) dort hinzufügen. Der Grund ist, dass die GUI single-threaded ist, und wenn du den GUI-Thread mit langwierigem Code blockierst, friert die GUI ein oder ruckelt – und wir wollen eine möglichst flüssige GUI. Um das zu umgehen, können wir einen Worker-Thread verwenden, um den SDR/DSP-Code im Hintergrund auszuführen.

Das folgende Beispiel erweitert das minimale Beispiel um einen Worker-Thread, der Code (in der :code:`run`-Funktion) ununterbrochen ausführt. Wir verwenden jedoch kein :code:`while True:`, weil wir aufgrund der Art und Weise, wie PyQt intern funktioniert, wollen, dass unsere :code:`run`-Funktion periodisch beendet wird und von vorn beginnt. Um dies zu erreichen, wird das :code:`end_of_run`-Signal des Worker-Threads (das wir im nächsten Abschnitt näher besprechen) mit einer Callback-Funktion verbunden, die die :code:`run`-Funktion des Worker-Threads erneut auslöst. Wir müssen den Worker-Thread auch im :code:`MainWindow`-Code initialisieren, was die Erstellung eines neuen :code:`QThread` und die Zuweisung unseres benutzerdefinierten Workers dazu umfasst. Dieser Code mag kompliziert erscheinen, aber es ist ein sehr häufiges Muster in PyQt-Anwendungen. Das Wichtigste dabei: GUI-orientierter Code gehört in :code:`MainWindow`, SDR/DSP-orientierter Code in die :code:`run`-Funktion des Worker-Threads.

.. code-block:: python

    from PyQt6.QtCore import QThread, pyqtSignal, QObject, QTimer
    from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton
    import time

    # Nicht-GUI-Operationen (einschließlich SDR) müssen in einem separaten Thread laufen
    class SDRWorker(QObject):
        end_of_run = pyqtSignal()

        # Hauptschleife
        def run(self):
            print("Starting run()")
            time.sleep(1)
            self.end_of_run.emit() # MainWindow mitteilen, dass wir fertig sind

    # QMainWindow ableiten, um das Hauptfenster der Anwendung anzupassen
    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()

            # Worker und Thread initialisieren
            self.sdr_thread = QThread()
            worker = SDRWorker()
            worker.moveToThread(self.sdr_thread)

            # Beispiel-GUI-Komponente
            example_button = QPushButton('Push Me')
            def on_button_click():
                print("beep")
            example_button.clicked.connect(on_button_click)
            self.setCentralWidget(example_button)

            # Das lässt die run()-Funktion ununterbrochen wiederholen
            def end_of_run_callback():
                QTimer.singleShot(0, worker.run) # Worker sofort erneut ausführen
            worker.end_of_run.connect(end_of_run_callback)

            self.sdr_thread.started.connect(worker.run) # startet den ersten run()-Aufruf, wenn der Thread beginnt
            self.sdr_thread.start() # Thread starten

    app = QApplication([])
    window = MainWindow()
    window.show() # Fenster sind standardmäßig ausgeblendet
    app.exec() # Ereignisschleife starten

Führe den obigen Code aus – du solltest jede Sekunde ein „Starting run()" in der Konsole sehen, und der Druckknopf sollte weiterhin funktionieren (ohne jede Verzögerung). Im Worker-Thread machen wir jetzt nur einen Print und einen Sleep, aber bald werden wir die SDR-Verarbeitung und den DSP-Code dort hinzufügen.

*************************
Signals und Slots
*************************

Im obigen Beispiel haben wir das :code:`end_of_run`-Signal verwendet, um zwischen dem Worker-Thread und dem GUI-Thread zu kommunizieren. Dies ist ein häufiges Muster in PyQt-Anwendungen und wird als „Signals and Slots"-Mechanismus bezeichnet. Ein Signal wird von einem Objekt ausgesendet (in diesem Fall dem Worker-Thread) und mit einem Slot verbunden (in diesem Fall der Callback-Funktion :code:`end_of_run_callback` im GUI-Thread). Das Signal kann mit mehreren Slots verbunden werden, und der Slot kann mit mehreren Signals verbunden werden. Das Signal kann auch Argumente tragen, die an den Slot übergeben werden, wenn das Signal ausgesendet wird. Beachte, dass wir die Dinge auch umkehren können: Der GUI-Thread kann ein Signal an den Slot des Worker-Threads senden. Der Signal/Slot-Mechanismus ist eine leistungsstarke Möglichkeit, zwischen verschiedenen Teilen einer PyQt-Anwendung zu kommunizieren, eine ereignisgesteuerte Struktur zu erzeugen, und wird im folgenden Beispielcode ausgiebig genutzt. Denk einfach daran: Ein Slot ist lediglich eine Callback-Funktion, und ein Signal ist eine Möglichkeit, diese Callback-Funktion zu signalisieren.

*************************
PyQtGraph
*************************

PyQtGraph ist eine Bibliothek, die auf PyQt und NumPy aufbaut und schnelle und effiziente Darstellungsfähigkeiten bietet, da PyQt zu allgemein gehalten ist, um Darstellungsfunktionen mitzuliefern. Sie ist für den Einsatz in Echtzeitanwendungen konzipiert und auf Geschwindigkeit optimiert. Sie ähnelt in vieler Hinsicht Matplotlib, ist aber für Echtzeitanwendungen statt für Einzeldarstellungen gedacht. Mit dem einfachen Beispiel unten kannst du die Leistung von PyQtGraph mit Matplotlib vergleichen – ändere einfach :code:`if True:` zu :code:`False:`. Auf einem Intel Core i9-10900K @ 3,70 GHz aktualisierte der PyQtGraph-Code mit über 1000 FPS, während der Matplotlib-Code mit 40 FPS aktualisierte. Wenn du jedoch von Matplotlib profitierst (z. B. um Entwicklungszeit zu sparen oder weil du ein bestimmtes Feature möchtest, das PyQtGraph nicht unterstützt), kannst du Matplotlib-Diagramme in eine PyQt-Anwendung einbinden und den folgenden Code als Ausgangspunkt verwenden.

.. raw:: html

   <details>
   <summary>Expand for comparison code</summary>

.. code-block:: python

    import numpy as np
    import time
    import matplotlib
    matplotlib.use('Qt5Agg')
    from PyQt6 import QtCore, QtWidgets
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import pyqtgraph as pg # tested with pyqtgraph==0.13.7

    n_data = 1024

    if True:
        class MplCanvas(FigureCanvas):
            def __init__(self):
                fig = Figure(figsize=(13, 8), dpi=100)
                self.axes = fig.add_subplot(111)
                super(MplCanvas, self).__init__(fig)


        class MainWindow(QtWidgets.QMainWindow):
            def __init__(self):
                super(MainWindow, self).__init__()

                self.canvas = MplCanvas()
                self._plot_ref = self.canvas.axes.plot(np.arange(n_data), '.-r')[0]
                self.canvas.axes.set_xlim(0, n_data)
                self.canvas.axes.set_ylim(-5, 5)
                self.canvas.axes.grid(True)
                self.setCentralWidget(self.canvas)

                # Timer einrichten, der die Neuzeichnung durch Aufruf von update_plot auslöst
                self.timer = QtCore.QTimer()
                self.timer.setInterval(0) # Timer sofort starten
                self.timer.timeout.connect(self.update_plot) # Timer startet sich automatisch neu
                self.timer.start()
                self.start_t = time.time() # für Benchmarking

                self.show()

            def update_plot(self):
                self._plot_ref.set_ydata(np.random.randn(n_data))
                self.canvas.draw() # Canvas zum Aktualisieren und Neuzeichnen auslösen
                print('FPS:', 1/(time.time()-self.start_t)) # ~42 FPS auf einem i9-10900K
                self.start_t = time.time()

    else:
        class MainWindow(QtWidgets.QMainWindow):
            def __init__(self):
                super(MainWindow, self).__init__()

                self.time_plot = pg.PlotWidget()
                self.time_plot.setYRange(-5, 5)
                self.time_plot_curve = self.time_plot.plot([])
                self.setCentralWidget(self.time_plot)

                # Timer einrichten, der die Neuzeichnung durch Aufruf von update_plot auslöst
                self.timer = QtCore.QTimer()
                self.timer.setInterval(0) # Timer sofort starten
                self.timer.timeout.connect(self.update_plot) # Timer startet sich automatisch neu
                self.timer.start()
                self.start_t = time.time() # für Benchmarking

                self.show()

            def update_plot(self):
                self.time_plot_curve.setData(np.random.randn(n_data))
                print('FPS:', 1/(time.time()-self.start_t)) # ~42 FPS auf einem i9-10900K
                self.start_t = time.time()

    app = QtWidgets.QApplication([])
    w = MainWindow()
    app.exec()

.. raw:: html

    </details>

Was die Verwendung von PyQtGraph betrifft, importieren wir es mit :code:`import pyqtgraph as pg` und können dann ein Qt-Widget für ein 1D-Diagramm wie folgt erstellen (dieser Code gehört in das :code:`__init__` von :code:`MainWindow`):

.. code-block:: python

        # Beispiel-PyQtGraph-Plot
        time_plot = pg.PlotWidget(labels={'left': 'Amplitude', 'bottom': 'Time'})
        time_plot_curve = time_plot.plot(np.arange(1000), np.random.randn(1000)) # x und y
        time_plot.setYRange(-5, 5)

        self.setCentralWidget(time_plot)

.. image:: ../_images/pyqtgraph_example.png
   :scale: 80 %
   :align: center
   :alt: PyQtGraph example

Man sieht, wie unkompliziert es ist, ein Diagramm einzurichten, und das Ergebnis ist einfach ein weiteres Widget, das du deiner GUI hinzufügen kannst. Neben 1D-Diagrammen hat PyQtGraph auch ein Äquivalent zu Matplotlibs :code:`imshow()`, das 2D mithilfe einer Farbpalette darstellt – das werden wir für unser Echtzeit-Spektrogramm/Wasserfall verwenden. Ein schöner Aspekt von PyQtGraph ist, dass die erstellten Diagramme einfach Qt-Widgets sind, und wir können andere Qt-Elemente (z. B. ein Rechteck einer bestimmten Größe an einer bestimmten Koordinate) mit purem PyQt hinzufügen. Das liegt daran, dass PyQtGraph die :code:`QGraphicsScene`-Klasse von PyQt nutzt, die eine Oberfläche zur Verwaltung einer großen Anzahl von 2D-Grafikelementen bietet, und nichts hindert uns daran, Linien, Rechtecke, Text, Ellipsen, Polygone und Bitmaps mit reinem PyQt hinzuzufügen.

*******
Layouts
*******

In den obigen Beispielen haben wir :code:`self.setCentralWidget()` verwendet, um das Haupt-Widget des Fensters festzulegen. Das ist eine einfache Methode, erlaubt aber keine komplexeren Layouts. Für komplexere Layouts können wir Layouts verwenden, die eine Möglichkeit sind, Widgets in einem Fenster anzuordnen. Es gibt verschiedene Layout-Typen, darunter :code:`QHBoxLayout`, :code:`QVBoxLayout`, :code:`QGridLayout` und :code:`QFormLayout`. :code:`QHBoxLayout` und :code:`QVBoxLayout` ordnen Widgets horizontal bzw. vertikal an. :code:`QGridLayout` ordnet Widgets in einem Raster an, und :code:`QFormLayout` ordnet Widgets in einem zweispaltigen Layout an, mit Labels in der ersten und Eingabe-Widgets in der zweiten Spalte.

Um ein neues Layout zu erstellen und Widgets hinzuzufügen, füge folgendes in das :code:`__init__` deines :code:`MainWindow` ein:

.. code-block:: python

    layout = QHBoxLayout()
    layout.addWidget(QPushButton("Left-Most"))
    layout.addWidget(QPushButton("Center"), 1)
    layout.addWidget(QPushButton("Right-Most"), 2)
    self.setLayout(layout)

In diesem Beispiel stapeln wir die Widgets horizontal; durch Ersetzen von :code:`QHBoxLayout` durch :code:`QVBoxLayout` können wir sie stattdessen vertikal stapeln. Die Funktion :code:`addWidget` wird verwendet, um Widgets zum Layout hinzuzufügen, und das optionale zweite Argument ist ein Dehnungsfaktor, der bestimmt, wie viel Platz das Widget im Verhältnis zu den anderen Widgets im Layout einnehmen soll.

:code:`QGridLayout` hat zusätzliche Parameter, weil du Zeile und Spalte des Widgets angeben musst, und optional, wie viele Zeilen und Spalten das Widget überspannen soll (Standard ist jeweils 1). Hier ist ein Beispiel für ein :code:`QGridLayout`:

.. code-block:: python

    layout = QGridLayout()
    layout.addWidget(QPushButton("Button at (0, 0)"), 0, 0)
    layout.addWidget(QPushButton("Button at (0, 1)"), 0, 1)
    layout.addWidget(QPushButton("Button at (0, 2)"), 0, 2)
    layout.addWidget(QPushButton("Button at (1, 0)"), 1, 0)
    layout.addWidget(QPushButton("Button at (1, 1)"), 1, 1)
    layout.addWidget(QPushButton("Button at (1, 2)"), 1, 2)
    layout.addWidget(QPushButton("Button at (2, 0) spanning 2 columns"), 2, 0, 1, 2)
    self.setLayout(layout)

.. image:: ../_images/qt_layouts.svg
   :align: center
   :target: ../_images/qt_layouts.svg
   :alt: Qt Layouts showing examples of QHBoxLayout, QVBoxLayout, and QGridLayout

Für unseren Spektrumanalysator verwenden wir :code:`QGridLayout` für das Gesamtlayout, aber wir werden auch :code:`QHBoxLayout` hinzufügen, um Widgets innerhalb eines Gitterplatzes horizontal zu stapeln. Layouts können einfach verschachtelt werden, indem ein neues Layout erstellt und dem übergeordneten Layout hinzugefügt wird:

.. code-block:: python

    layout = QGridLayout()
    self.setLayout(layout)
    inner_layout = QHBoxLayout()
    layout.addLayout(inner_layout)

*******************
:code:`QPushButton`
*******************

Das erste eigentliche Widget, das wir besprechen, ist der :code:`QPushButton`, eine einfache Schaltfläche, die geklickt werden kann. Wir haben bereits gesehen, wie man einen :code:`QPushButton` erstellt und sein :code:`clicked`-Signal mit einer Callback-Funktion verbindet. Der :code:`QPushButton` hat einige weitere Signals, darunter :code:`pressed`, :code:`released` und :code:`toggled`. Das :code:`toggled`-Signal wird ausgesendet, wenn die Schaltfläche aktiviert oder deaktiviert wird, und ist nützlich zum Erstellen von Umschaltflächen. Der :code:`QPushButton` hat auch einige Eigenschaften, darunter :code:`text`, :code:`icon` und :code:`checkable`, sowie eine Methode namens :code:`click()`, die einen Klick auf die Schaltfläche simuliert. In unserem SDR-Spektrumanalysator verwenden wir Schaltflächen, um einen Auto-Range für Diagramme auszulösen, wobei die aktuellen Daten zur Berechnung der y-Grenzen verwendet werden. Da wir den :code:`QPushButton` bereits verwendet haben, gehen wir hier nicht weiter ins Detail, aber weitere Informationen findest du in der `QPushButton-Dokumentation <https://doc.qt.io/qtforpython/PySide6/QtWidgets/QPushButton.html>`_.

***************
:code:`QSlider`
***************

Der :code:`QSlider` ist ein Widget, mit dem der Benutzer einen Wert aus einem Wertebereich auswählen kann. Der :code:`QSlider` hat einige Eigenschaften, darunter :code:`minimum`, :code:`maximum`, :code:`value` und :code:`orientation`. Er hat auch einige Signals, darunter :code:`valueChanged`, :code:`sliderPressed` und :code:`sliderReleased`, sowie eine Methode :code:`setValue()`, die den Wert des Schiebereglers setzt – diese werden wir häufig verwenden. Die Dokumentationsseite für `QSlider ist hier <https://doc.qt.io/qtforpython/PySide6/QtWidgets/QSlider.html>`_.

In unserem Spektrumanalysator verwenden wir :code:`QSlider` zur Einstellung der Mittenfrequenz und der Verstärkung des SDR. Hier ist der Ausschnitt aus dem endgültigen Anwendungscode, der den Verstärkungs-Schieberegler erstellt:

.. code-block:: python

    # Verstärkungs-Schieberegler mit Label
    gain_slider = QSlider(Qt.Orientation.Horizontal)
    gain_slider.setRange(0, 73) # Min und Max, inklusiv. Schrittweite ist immer 1
    gain_slider.setValue(50) # Anfangswert
    gain_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
    gain_slider.setTickInterval(2) # nur für visuelle Darstellung
    gain_slider.sliderMoved.connect(worker.update_gain)
    gain_label = QLabel()
    def update_gain_label(val):
        gain_label.setText("Gain: " + str(val))
    gain_slider.sliderMoved.connect(update_gain_label)
    update_gain_label(gain_slider.value()) # Label initialisieren
    layout.addWidget(gain_slider, 5, 0)
    layout.addWidget(gain_label, 5, 1)

Sehr wichtig zu wissen: :code:`QSlider` verwendet Integer-Werte. Durch Setzen des Bereichs von 0 bis 73 erlauben wir dem Schieberegler, ganzzahlige Werte zwischen diesen Zahlen (einschließlich Start und Ende) auszuwählen. Das :code:`setTickInterval(2)` ist rein visuell. Aus diesem Grund verwenden wir kHz als Einheit für den Frequenz-Schieberegler, damit wir eine Auflösung bis auf 1 kHz erhalten.

In der Mitte des obigen Codes erstellen wir ein :code:`QLabel`, das einfach ein Text-Label zur Anzeige ist. Damit es den aktuellen Wert des Schiebereglers anzeigt, müssen wir eine Callback-Funktion (also einen Slot) erstellen, die das Label aktualisiert. Wir verbinden diese Callback-Funktion mit dem :code:`sliderMoved`-Signal, das automatisch ausgesendet wird, wenn der Schieberegler bewegt wird. Wir rufen die Callback-Funktion auch einmal auf, um das Label mit dem aktuellen Wert des Schiebereglers zu initialisieren (in unserem Fall 50). Außerdem müssen wir das :code:`sliderMoved`-Signal mit einem Slot im Worker-Thread verbinden, der die Verstärkung des SDR aktualisiert (denk daran: SDR und DSP gehören nicht in den Haupt-GUI-Thread). Die Callback-Funktion, die diesen Slot definiert, wird später besprochen.

*****************
:code:`QComboBox`
*****************

Die :code:`QComboBox` ist ein Dropdown-Widget, mit dem der Benutzer ein Element aus einer Liste von Elementen auswählen kann. Die :code:`QComboBox` hat einige Eigenschaften, darunter :code:`currentText`, :code:`currentIndex` und :code:`count`. Sie hat auch einige Signals, darunter :code:`currentTextChanged`, :code:`currentIndexChanged` und :code:`activated`, sowie Methoden wie :code:`addItem()` zum Hinzufügen eines Elements zur Liste und :code:`insertItem()` zum Einfügen eines Elements an einem bestimmten Index – diese werden wir in unserem Spektrumanalysator-Beispiel aber nicht verwenden. Die Dokumentationsseite für `QComboBox ist hier <https://doc.qt.io/qtforpython/PySide6/QtWidgets/QComboBox.html>`_.

In unserem Spektrumanalysator verwenden wir :code:`QComboBox`, um die Abtastrate aus einer vordefinierten Liste auszuwählen. Am Anfang unseres Codes definieren wir die möglichen Abtastraten mit :code:`sample_rates = [56, 40, 20, 10, 5, 2, 1, 0.5]`. Im :code:`__init__` von :code:`MainWindow` erstellen wir die :code:`QComboBox` wie folgt:

.. code-block:: python

    # Abtastraten-Dropdown mit QComboBox
    sample_rate_combobox = QComboBox()
    sample_rate_combobox.addItems([str(x) + ' MHz' for x in sample_rates])
    sample_rate_combobox.setCurrentIndex(0) # Index angeben, nicht String
    sample_rate_combobox.currentIndexChanged.connect(worker.update_sample_rate)
    sample_rate_label = QLabel()
    def update_sample_rate_label(val):
        sample_rate_label.setText("Sample Rate: " + str(sample_rates[val]) + " MHz")
    sample_rate_combobox.currentIndexChanged.connect(update_sample_rate_label)
    update_sample_rate_label(sample_rate_combobox.currentIndex()) # Label initialisieren
    layout.addWidget(sample_rate_combobox, 6, 0)
    layout.addWidget(sample_rate_label, 6, 1)

Der einzige wirkliche Unterschied zum Schieberegler ist :code:`addItems()`, dem du die Liste der Strings als Optionen übergibst, und :code:`setCurrentIndex()`, das den Startwert festlegt.

****************
Lambda-Funktionen
****************

Erinnere dich an den obigen Code, in dem wir Folgendes geschrieben haben:

.. code-block:: python

    def update_sample_rate_label(val):
        sample_rate_label.setText("Sample Rate: " + str(sample_rates[val]) + " MHz")
    sample_rate_combobox.currentIndexChanged.connect(update_sample_rate_label)

Wir erstellen eine Funktion mit nur einer einzigen Codezeile im Inneren und übergeben dann diese Funktion (Funktionen sind auch Objekte!) an :code:`connect()`. Um dies zu vereinfachen, schreiben wir dieses Code-Muster mit einfachem Python um:

.. code-block:: python

    def my_function(x):
        print(x)
    y.call_that_takes_in_function_obj(my_function)

In dieser Situation haben wir eine Funktion mit nur einer Codezeile im Inneren, und wir referenzieren diese Funktion nur einmal – beim Setzen des :code:`connect`-Callbacks. In solchen Situationen können wir eine Lambda-Funktion verwenden, eine Möglichkeit, eine Funktion in einer einzigen Zeile zu definieren. Hier ist der obige Code mit einer Lambda-Funktion umgeschrieben:

.. code-block:: python

    y.call_that_takes_in_function_obj(lambda x: print(x))

Wenn du noch nie eine Lambda-Funktion verwendet hast, mag das seltsam wirken, und du musst sie nicht unbedingt verwenden – aber sie spart zwei Zeilen Code und macht den Code kompakter. Der Name des temporären Arguments kommt nach „lambda", und alles nach dem Doppelpunkt ist der Code, der auf diese Variable angewendet wird. Es werden auch mehrere Argumente mit Kommas unterstützt, oder sogar keine Argumente mit :code:`lambda : <code>`. Versuche als Übung, die Funktion :code:`update_sample_rate_label` oben mit einer Lambda-Funktion umzuschreiben.

***********************
PyQtGraph's PlotWidget
***********************

PyQtGraphs :code:`PlotWidget` ist ein PyQt-Widget zur Erstellung von 1D-Diagrammen, ähnlich wie Matplotlibs :code:`plt.plot(x,y)`. Wir werden es für die Zeit- und Frequenz-PSD-Diagramme verwenden, obwohl es auch gut für IQ-Diagramme geeignet ist (die unser Spektrumanalysator nicht enthält). Für Interessierte: PlotWidget ist eine Unterklasse von PyQts `QGraphicsView <https://doc.qt.io/qtforpython-5/PySide2/QtWidgets/QGraphicsView.html>`_, einem Widget zur Darstellung des Inhalts einer `QGraphicsScene <https://doc.qt.io/qtforpython-5/PySide2/QtWidgets/QGraphicsScene.html#PySide2.QtWidgets.PySide2.QtWidgets.QGraphicsScene>`_, die eine Oberfläche zur Verwaltung einer großen Anzahl von 2D-Grafikelementen in Qt ist. Das Wichtigste über PlotWidget zu wissen ist, dass es einfach ein Widget ist, das ein einzelnes `PlotItem <https://pyqtgraph.readthedocs.io/en/latest/api_reference/graphicsItems/plotitem.html#pyqtgraph.PlotItem>`_ enthält. Aus Dokumentationsperspektive solltest du daher direkt die PlotItem-Dokumentation konsultieren: `<https://pyqtgraph.readthedocs.io/en/latest/api_reference/graphicsItems/plotitem.html>`_. Ein PlotItem enthält eine ViewBox zur Darstellung der darzustellenden Daten sowie AxisItems und Labels für Achsen und Titel.

Das einfachste Beispiel für die Verwendung eines PlotWidget lautet wie folgt (das im :code:`__init__` von :code:`MainWindow` hinzugefügt werden muss):

.. code-block:: python

 import pyqtgraph as pg
 plotWidget = pg.plot(title="My Title")
 plotWidget.plot(x, y)

wobei x und y typischerweise NumPy-Arrays sind, genau wie bei Matplotlibs :code:`plt.plot()`. Dies stellt jedoch ein statisches Diagramm dar, bei dem sich die Daten nie ändern. Für unseren Spektrumanalysator wollen wir die Daten im Worker-Thread aktualisieren. Daher müssen wir beim Initialisieren des Diagramms noch keine Daten übergeben – wir müssen es nur einrichten. So initialisieren wir das Zeitbereich-Diagramm in unserem Spektrumanalysator:

.. code-block:: python

    # Zeitbereich-Diagramm
    time_plot = pg.PlotWidget(labels={'left': 'Amplitude', 'bottom': 'Time [microseconds]'})
    time_plot.setMouseEnabled(x=False, y=True)
    time_plot.setYRange(-1.1, 1.1)
    time_plot_curve_i = time_plot.plot([])
    time_plot_curve_q = time_plot.plot([])
    layout.addWidget(time_plot, 1, 0)

Wir erstellen zwei verschiedene Kurven, eine für I und eine für Q. Der restliche Code erklärt sich von selbst. Um das Diagramm aktualisieren zu können, müssen wir im :code:`__init__` von :code:`MainWindow` einen Slot (eine Callback-Funktion) erstellen:

.. code-block:: python

    def time_plot_callback(samples):
        time_plot_curve_i.setData(samples.real)
        time_plot_curve_q.setData(samples.imag)

Wir verbinden diesen Slot mit dem Signal des Worker-Threads, das ausgesendet wird, wenn neue Samples verfügbar sind, wie später gezeigt.

Als letztes fügen wir im :code:`__init__` von :code:`MainWindow` zwei Schaltflächen rechts neben dem Diagramm hinzu, die einen Auto-Range auslösen. Eine verwendet das aktuelle Min/Max, eine andere setzt den Bereich auf -1,1 bis 1,1 (entspricht den ADC-Grenzen vieler SDRs plus 10% Spielraum). Wir erstellen ein inneres Layout vom Typ QVBoxLayout, um diese zwei Schaltflächen vertikal zu stapeln. Hier ist der Code zum Hinzufügen der Schaltflächen:

.. code-block:: python

    # Auto-Range-Schaltflächen für Zeitdiagramm
    time_plot_auto_range_layout = QVBoxLayout()
    layout.addLayout(time_plot_auto_range_layout, 1, 1)
    auto_range_button = QPushButton('Auto Range')
    auto_range_button.clicked.connect(lambda : time_plot.autoRange()) # lambda bedeutet unbenannte Funktion
    time_plot_auto_range_layout.addWidget(auto_range_button)
    auto_range_button2 = QPushButton('-1 to +1\n(ADC limits)')
    auto_range_button2.clicked.connect(lambda : time_plot.setYRange(-1.1, 1.1))
    time_plot_auto_range_layout.addWidget(auto_range_button2)

Und so sieht es am Ende aus:

.. image:: ../_images/pyqt_time_plot.png
   :scale: 50 %
   :align: center
   :alt: PyQtGraph Time Plot

Für das Frequenzbereich-Diagramm (PSD) verwenden wir ein ähnliches Muster.

*********************
PyQtGraph's ImageItem
*********************

Ein Spektrumanalysator ist ohne ein Wasserfall-Diagramm (auch Echtzeit-Spektrogramm genannt) nicht vollständig. Dafür verwenden wir PyQtGraphs ImageItem, das Bilder mit 1, 3 oder 4 „Kanälen" rendert. Ein Kanal bedeutet, dass du ihm ein 2D-Array aus Floats oder Ints übergibst, das dann eine Lookup-Tabelle (LUT) nutzt, um eine Farbpalette anzuwenden und letztlich das Bild zu erstellen. Alternativ kannst du RGB (3 Kanäle) oder RGBA (4 Kanäle) übergeben. Wir berechnen unser Spektrogramm als 2D-NumPy-Array aus Floats und übergeben es direkt an das ImageItem. Wir wählen eine Farbpalette und nutzen sogar die eingebaute Funktionalität zur Anzeige einer grafischen LUT, die die Werteverteilung unserer Daten und die Anwendung der Farbpalette darstellt.

Die eigentliche Initialisierung des Wasserfall-Diagramms ist recht unkompliziert: Wir verwenden ein PlotWidget als Container (damit Achsen angezeigt werden) und fügen dann ein ImageItem hinzu:

.. code-block:: python

    # Wasserfall-Diagramm
    waterfall = pg.PlotWidget(labels={'left': 'Time [s]', 'bottom': 'Frequency [MHz]'})
    imageitem = pg.ImageItem(axisOrder='col-major') # dieses Argument dient nur der Performance
    waterfall.addItem(imageitem)
    waterfall.setMouseEnabled(x=False, y=False)
    waterfall_layout.addWidget(waterfall)

Der Slot/Callback zum Aktualisieren der Wasserfall-Daten, der in das :code:`__init__` von :code:`MainWindow` gehört, sieht wie folgt aus:

.. code-block:: python

    def waterfall_plot_callback(spectrogram):
        imageitem.setImage(spectrogram, autoLevels=False)
        sigma = np.std(spectrogram)
        mean = np.mean(spectrogram)
        self.spectrogram_min = mean - 2*sigma # im Fensterzustand speichern
        self.spectrogram_max = mean + 2*sigma

Dabei ist spectrogram ein 2D-NumPy-Array aus Floats. Zusätzlich zum Setzen der Bilddaten berechnen wir ein Min und Max für die Farbpalette basierend auf Mittelwert und Varianz der Daten, das wir später verwenden. Der letzte Teil des GUI-Codes für das Spektrogramm ist die Erstellung der Farbskala, die auch die verwendete Farbpalette festlegt:

.. code-block:: python

    # Farbskala für Wasserfall
    colorbar = pg.HistogramLUTWidget()
    colorbar.setImageItem(imageitem) # verbindet die Skala mit dem Wasserfall-ImageItem
    colorbar.item.gradient.loadPreset('viridis') # Farbpalette setzen, auch für das ImageItem
    imageitem.setLevels((-30, 20)) # muss nach Erstellung der Farbskala kommen
    waterfall_layout.addWidget(colorbar)

Die zweite Zeile ist wichtig – sie verbindet diese Farbskala mit dem ImageItem. Hier wählen wir auch die Farbpalette und setzen die Anfangspegel (-30 dB bis +20 dB in unserem Fall). Im Worker-Thread-Code siehst du, wie das 2D-Spektrogramm-Array berechnet und gespeichert wird. Unten ist ein Screenshot dieses GUI-Teils, der die eingebaute Funktionalität der Farbskala und LUT-Anzeige zeigt. Beachte, dass die seitliche Glockenkurve die Verteilung der Spektrogramm-Werte zeigt – sehr nützlich.

.. image:: ../_images/pyqt_spectrogram.png
   :scale: 50 %
   :align: center
   :alt: PyQtGraph Spectrogram and colorbar

***********************
Worker-Thread
***********************

Erinnere dich, dass wir am Anfang dieses Kapitels gelernt haben, wie man einen separaten Thread erstellt, wobei wir eine Klasse namens SDRWorker mit einer :code:`run()`-Funktion verwendet haben. Hier werden wir den gesamten SDR- und DSP-Code unterbringen, mit Ausnahme der SDR-Initialisierung, die wir vorerst global vornehmen. Der Worker-Thread ist auch dafür verantwortlich, die drei Diagramme zu aktualisieren, indem er Signals aussendet, wenn neue Samples verfügbar sind, um die Callback-Funktionen auszulösen, die wir bereits in :code:`MainWindow` erstellt haben und die letztendlich die Diagramme aktualisieren. Die SDRWorker-Klasse lässt sich in drei Bereiche aufteilen:

#. :code:`init()` – zum Initialisieren von Zustand, z. B. des Spektrogramm-2D-Arrays
#. PyQt Signals – hier müssen wir unsere benutzerdefinierten Signals definieren, die ausgesendet werden sollen
#. PyQt Slots – die Callback-Funktionen, die durch GUI-Ereignisse wie einen Schieberegler-Bewegung ausgelöst werden
#. :code:`run()` – die Hauptschleife, die ununterbrochen läuft

***********************
PyQt Signals
***********************

Im GUI-Code mussten wir keine Signals definieren, weil sie in den verwendeten Widgets eingebaut waren, wie z. B. :code:`valueChanged` bei :code:`QSlider`. Unsere SDRWorker-Klasse ist benutzerdefiniert, und alle Signals, die wir senden möchten, müssen vor dem Start von :code:`run()` definiert werden. Hier ist der Code für die SDRWorker-Klasse, der vier Signals und ihre entsprechenden Datentypen definiert:

.. code-block:: python

    # PyQt Signals
    time_plot_update = pyqtSignal(np.ndarray)
    freq_plot_update = pyqtSignal(np.ndarray)
    waterfall_plot_update = pyqtSignal(np.ndarray)
    end_of_run = pyqtSignal() # tritt viele Male pro Sekunde auf

Die ersten drei Signals senden ein einzelnes Objekt – ein NumPy-Array. Das letzte Signal sendet kein Objekt mit. Du kannst auch mehrere Objekte gleichzeitig senden, indem du einfach Kommas zwischen Datentypen verwendest, aber wir benötigen das für unsere Anwendung nicht. Überall innerhalb von :code:`run()` können wir mit nur einer Codezeile ein Signal an den GUI-Thread senden, zum Beispiel:

.. code-block:: python

    self.time_plot_update.emit(samples)

Es gibt noch einen letzten Schritt, um alle Signal/Slot-Verbindungen herzustellen: Im GUI-Code (am Ende des :code:`__init__` von :code:`MainWindow`) müssen wir die Signals des Worker-Threads mit den Slots der GUI verbinden, zum Beispiel:

.. code-block:: python

    worker.time_plot_update.connect(time_plot_callback) # Signal mit Callback verbinden

Beachte, dass :code:`worker` die Instanz der SDRWorker-Klasse ist, die wir im GUI-Code erstellt haben. Wir verbinden also das Signal des Worker-Threads namens :code:`time_plot_update` mit dem Slot der GUI namens :code:`time_plot_callback`, den wir zuvor definiert haben. Jetzt ist ein guter Zeitpunkt, die bisher gezeigten Code-Ausschnitte noch einmal zu überprüfen und zu sehen, wie sie alle zusammenpassen, um sicherzustellen, dass du verstehst, wie GUI und Worker-Thread miteinander kommunizieren – das ist ein zentraler Bestandteil der PyQt-Programmierung.

***********************
Worker-Thread Slots
***********************

Die Slots des Worker-Threads sind Callback-Funktionen, die durch GUI-Ereignisse ausgelöst werden, z. B. wenn der Verstärkungs-Schieberegler bewegt wird. Sie sind recht unkompliziert – dieser Slot zum Beispiel aktualisiert die Verstärkung des SDR auf den neuen vom Schieberegler gewählten Wert:

.. code-block:: python

    def update_gain(self, val):
        print("Updated gain to:", val, 'dB')
        sdr.set_rx_gain(val)

***********************
Worker-Thread Run()
***********************

Die :code:`run()`-Funktion ist der Ort, an dem der ganze DSP-Spaß passiert! In unserer Anwendung beginnen wir jede :code:`run()`-Funktion damit, eine Reihe von Samples vom SDR zu empfangen (oder Samples zu simulieren, wenn du kein SDR hast).

.. code-block:: python

    # Hauptschleife
    def run(self):
        if sdr_type == "pluto":
            samples = sdr.rx()/2**11 # Samples empfangen
        elif sdr_type == "usrp":
            streamer.recv(recv_buffer, metadata)
            samples = recv_buffer[0] # wird np.complex64 sein
        elif sdr_type == "sim":
            tone = np.exp(2j*np.pi*self.sample_rate*0.1*np.arange(fft_size)/self.sample_rate)
            noise = np.random.randn(fft_size) + 1j*np.random.randn(fft_size)
            samples = self.gain*tone*0.02 + 0.1*noise
            # Auf -1 bis +1 begrenzen, um ADC-Bitgrenzen zu simulieren
            np.clip(samples.real, -1, 1, out=samples.real)
            np.clip(samples.imag, -1, 1, out=samples.imag)

        ...

Für das simulierte Beispiel erzeugen wir einen Ton mit etwas weißem Rauschen und begrenzen die Samples dann auf -1 bis +1.

Nun zum DSP! Wir wissen, dass wir die FFT für das Frequenzbereich-Diagramm und das Spektrogramm benötigen. Es stellt sich heraus, dass wir einfach die PSD für diesen Satz von Samples als eine Zeile des Spektrogramms verwenden können. Wir müssen also nur unser Spektrogramm/Wasserfall um eine Zeile nach oben verschieben und die neue Zeile unten (oder oben – das spielt keine Rolle) hinzufügen. Für jede Diagramm-Aktualisierung senden wir das Signal mit den aktualisierten Daten aus. Wir signalisieren auch das Ende der :code:`run()`-Funktion, damit der GUI-Thread sofort einen weiteren :code:`run()`-Aufruf startet. Insgesamt ist es eigentlich nicht viel Code:

.. code-block:: python

        ...

        self.time_plot_update.emit(samples[0:time_plot_samples])

        PSD = 10.0*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples)))**2/fft_size)
        self.PSD_avg = self.PSD_avg * 0.99 + PSD * 0.01
        self.freq_plot_update.emit(self.PSD_avg)

        self.spectrogram[:] = np.roll(self.spectrogram, 1, axis=1) # Wasserfall um 1 Zeile verschieben
        self.spectrogram[:,0] = PSD # letzte Zeile mit neuen FFT-Ergebnissen füllen
        self.waterfall_plot_update.emit(self.spectrogram)

        self.end_of_run.emit() # Signal senden, um die Schleife fortzusetzen
        # Ende von run()

Beachte, dass wir nicht den gesamten Satz von Samples an das Zeitdiagramm senden, weil es zu viele Punkte zum Anzeigen wären – stattdessen senden wir nur die ersten 500 Samples (am Anfang des Skripts konfigurierbar, hier nicht gezeigt). Für das PSD-Diagramm verwenden wir einen gleitenden Durchschnitt der PSD, indem wir die vorherige PSD speichern und 1% der neuen PSD hinzufügen. Das ist eine einfache Methode, um das PSD-Diagramm zu glätten. Beachte, dass die Reihenfolge der :code:`emit()`-Aufrufe für die Signals keine Rolle spielt; sie hätten genauso gut alle am Ende von :code:`run()` stehen können.

***********************
Vollständiger Beispiel-Code
***********************

Bisher haben wir uns Ausschnitte der Spektrumanalysator-App angesehen, aber jetzt schauen wir uns endlich den vollständigen Code an und probieren ihn aus. Er unterstützt derzeit PlutoSDR, USRP oder Simulationsmodus. Wenn du kein Pluto oder USRP hast, lass den Code einfach so wie er ist – er sollte den Simulationsmodus verwenden. Andernfalls ändere :code:`sdr_type`. Im Simulationsmodus wirst du bemerken, dass das Signal im Zeitbereich abgeschnitten wird, wenn du die Verstärkung ganz aufgedreht hast, was zu Störsignalen im Frequenzbereich führt.

Verwende diesen Code gerne als Ausgangspunkt für deine eigene Echtzeit-SDR-Anwendung! Unten ist auch eine Animation der App in Aktion, wobei ein Pluto verwendet wird, um das 750-MHz-Mobilfunkband und dann 2,4-GHz-WLAN zu betrachten. Eine Version in höherer Qualität ist auf YouTube `hier <https://youtu.be/hvofiY3Q_yo>`_ verfügbar.

.. image:: ../_images/pyqt_animation.gif
   :scale: 100 %
   :align: center
   :alt: Animated gif showing the PyQt spectrum analyzer app in action

Bekannte Fehler (um sie zu beheben, `bearbeite dies <https://github.com/777arc/PySDR/edit/master/figure-generating-scripts/pyqt_example.py>`_):

#. Die x-Achse des Wasserfalls aktualisiert sich nicht beim Ändern der Mittenfrequenz (das PSD-Diagramm schon)

Vollständiger Code:

.. code-block:: python

    from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal, QObject, QTimer
    from PyQt6.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QSlider, QLabel, QHBoxLayout, QVBoxLayout, QPushButton, QComboBox  # tested with PyQt6==6.7.0
    import pyqtgraph as pg # tested with pyqtgraph==0.13.7
    import numpy as np
    import time
    import signal # ermöglicht, dass Strg+C die App wirklich schließt

    # Standardwerte
    fft_size = 4096 # bestimmt Puffergröße
    num_rows = 200
    center_freq = 750e6
    sample_rates = [56, 40, 20, 10, 5, 2, 1, 0.5] # MHz
    sample_rate = sample_rates[0] * 1e6
    time_plot_samples = 500
    gain = 50 # 0 bis 73 dB. int

    sdr_type = "sim" # oder "usrp" oder "pluto"

    # SDR initialisieren
    if sdr_type == "pluto":
        import adi
        sdr = adi.Pluto("ip:192.168.1.10")
        sdr.rx_lo = int(center_freq)
        sdr.sample_rate = int(sample_rate)
        sdr.rx_rf_bandwidth = int(sample_rate*0.8) # Bandbreite des Anti-Aliasing-Filters
        sdr.rx_buffer_size = int(fft_size)
        sdr.gain_control_mode_chan0 = 'manual'
        sdr.rx_hardwaregain_chan0 = gain # dB
    elif sdr_type == "usrp":
        import uhd
        #usrp = uhd.usrp.MultiUSRP(args="addr=192.168.1.10")
        usrp = uhd.usrp.MultiUSRP(args="addr=192.168.1.201")
        usrp.set_rx_rate(sample_rate, 0)
        usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(center_freq), 0)
        usrp.set_rx_gain(gain, 0)

        # Stream und Empfangspuffer einrichten
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [0]
        metadata = uhd.types.RXMetadata()
        streamer = usrp.get_rx_stream(st_args)
        recv_buffer = np.zeros((1, fft_size), dtype=np.complex64)

        # Stream starten
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        stream_cmd.stream_now = True
        streamer.issue_stream_cmd(stream_cmd)

        def flush_buffer():
            for _ in range(10):
                streamer.recv(recv_buffer, metadata)

    class SDRWorker(QObject):
        def __init__(self):
            super().__init__()
            self.gain = gain
            self.sample_rate = sample_rate
            self.freq = 0 # in kHz, um mit QSlider-Integers und max. 2 Milliarden umzugehen
            self.spectrogram = -50*np.ones((fft_size, num_rows))
            self.PSD_avg = -50*np.ones(fft_size)

        # PyQt Signals
        time_plot_update = pyqtSignal(np.ndarray)
        freq_plot_update = pyqtSignal(np.ndarray)
        waterfall_plot_update = pyqtSignal(np.ndarray)
        end_of_run = pyqtSignal() # tritt viele Male pro Sekunde auf

        # PyQt Slots
        def update_freq(self, val): # TODO: SDR KÖNNTE AUCH IM GUI-THREAD GEÄNDERT WERDEN
            print("Updated freq to:", val, 'kHz')
            if sdr_type == "pluto":
                sdr.rx_lo = int(val*1e3)
            elif sdr_type == "usrp":
                usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(val*1e3), 0)
                flush_buffer()

        def update_gain(self, val):
            print("Updated gain to:", val, 'dB')
            self.gain = val
            if sdr_type == "pluto":
                sdr.rx_hardwaregain_chan0 = val
            elif sdr_type == "usrp":
                usrp.set_rx_gain(val, 0)
                flush_buffer()

        def update_sample_rate(self, val):
            print("Updated sample rate to:", sample_rates[val], 'MHz')
            if sdr_type == "pluto":
                sdr.sample_rate = int(sample_rates[val] * 1e6)
                sdr.rx_rf_bandwidth = int(sample_rates[val] * 1e6 * 0.8)
            elif sdr_type == "usrp":
                usrp.set_rx_rate(sample_rates[val] * 1e6, 0)
                flush_buffer()

        # Hauptschleife
        def run(self):
            start_t = time.time()

            if sdr_type == "pluto":
                samples = sdr.rx()/2**11 # Samples empfangen
            elif sdr_type == "usrp":
                streamer.recv(recv_buffer, metadata)
                samples = recv_buffer[0] # wird np.complex64 sein
            elif sdr_type == "sim":
                tone = np.exp(2j*np.pi*self.sample_rate*0.1*np.arange(fft_size)/self.sample_rate)
                noise = np.random.randn(fft_size) + 1j*np.random.randn(fft_size)
                samples = self.gain*tone*0.02 + 0.1*noise
                # Auf -1 bis +1 begrenzen, um ADC-Bitgrenzen zu simulieren
                np.clip(samples.real, -1, 1, out=samples.real)
                np.clip(samples.imag, -1, 1, out=samples.imag)

            self.time_plot_update.emit(samples[0:time_plot_samples])

            PSD = 10.0*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples)))**2/fft_size)
            self.PSD_avg = self.PSD_avg * 0.99 + PSD * 0.01
            self.freq_plot_update.emit(self.PSD_avg)

            self.spectrogram[:] = np.roll(self.spectrogram, 1, axis=1) # Wasserfall um 1 Zeile verschieben
            self.spectrogram[:,0] = PSD # letzte Zeile mit neuen FFT-Ergebnissen füllen
            self.waterfall_plot_update.emit(self.spectrogram)

            print("Frames per second:", 1/(time.time() - start_t))
            self.end_of_run.emit() # Signal senden, um die Schleife fortzusetzen


    # QMainWindow ableiten, um das Hauptfenster der Anwendung anzupassen
    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()

            self.setWindowTitle("The PySDR Spectrum Analyzer")
            self.setFixedSize(QSize(1500, 1000)) # Fenstergröße, sollte auf 1920x1080 passen

            self.spectrogram_min = 0
            self.spectrogram_max = 0

            layout = QGridLayout() # Gesamtlayout

            # Worker und Thread initialisieren
            self.sdr_thread = QThread()
            self.sdr_thread.setObjectName('SDR_Thread') # in htop sichtbar; F2 -> Display options -> Show custom thread names
            worker = SDRWorker()
            worker.moveToThread(self.sdr_thread)

            # Zeitdiagramm
            time_plot = pg.PlotWidget(labels={'left': 'Amplitude', 'bottom': 'Time [microseconds]'})
            time_plot.setMouseEnabled(x=False, y=True)
            time_plot.setYRange(-1.1, 1.1)
            time_plot_curve_i = time_plot.plot([])
            time_plot_curve_q = time_plot.plot([])
            layout.addWidget(time_plot, 1, 0)

            # Auto-Range-Schaltflächen für Zeitdiagramm
            time_plot_auto_range_layout = QVBoxLayout()
            layout.addLayout(time_plot_auto_range_layout, 1, 1)
            auto_range_button = QPushButton('Auto Range')
            auto_range_button.clicked.connect(lambda : time_plot.autoRange()) # lambda bedeutet unbenannte Funktion
            time_plot_auto_range_layout.addWidget(auto_range_button)
            auto_range_button2 = QPushButton('-1 to +1\n(ADC limits)')
            auto_range_button2.clicked.connect(lambda : time_plot.setYRange(-1.1, 1.1))
            time_plot_auto_range_layout.addWidget(auto_range_button2)

            # Frequenzdiagramm
            freq_plot = pg.PlotWidget(labels={'left': 'PSD', 'bottom': 'Frequency [MHz]'})
            freq_plot.setMouseEnabled(x=False, y=True)
            freq_plot_curve = freq_plot.plot([])
            freq_plot.setXRange(center_freq/1e6 - sample_rate/2e6, center_freq/1e6 + sample_rate/2e6)
            freq_plot.setYRange(-30, 20)
            layout.addWidget(freq_plot, 2, 0)

            # Auto-Range-Schaltfläche für Frequenzdiagramm
            auto_range_button = QPushButton('Auto Range')
            auto_range_button.clicked.connect(lambda : freq_plot.autoRange()) # lambda bedeutet unbenannte Funktion
            layout.addWidget(auto_range_button, 2, 1)

            # Layout-Container für Wasserfall-Elemente
            waterfall_layout = QHBoxLayout()
            layout.addLayout(waterfall_layout, 3, 0)

            # Wasserfall-Diagramm
            waterfall = pg.PlotWidget(labels={'left': 'Time [s]', 'bottom': 'Frequency [MHz]'})
            imageitem = pg.ImageItem(axisOrder='col-major') # dieses Argument dient nur der Performance
            waterfall.addItem(imageitem)
            waterfall.setMouseEnabled(x=False, y=False)
            waterfall_layout.addWidget(waterfall)

            # Farbskala für Wasserfall
            colorbar = pg.HistogramLUTWidget()
            colorbar.setImageItem(imageitem) # verbindet die Skala mit dem Wasserfall-ImageItem
            colorbar.item.gradient.loadPreset('viridis') # Farbpalette setzen, auch für das ImageItem
            imageitem.setLevels((-30, 20)) # muss nach Erstellung der Farbskala kommen
            waterfall_layout.addWidget(colorbar)

            # Auto-Range-Schaltfläche für Wasserfall
            auto_range_button = QPushButton('Auto Range\n(-2σ to +2σ)')
            def update_colormap():
                imageitem.setLevels((self.spectrogram_min, self.spectrogram_max))
                colorbar.setLevels(self.spectrogram_min, self.spectrogram_max)
            auto_range_button.clicked.connect(update_colormap)
            layout.addWidget(auto_range_button, 3, 1)

            # Frequenz-Schieberegler mit Label, alle Einheiten in kHz
            freq_slider = QSlider(Qt.Orientation.Horizontal)
            freq_slider.setRange(0, int(6e6))
            freq_slider.setValue(int(center_freq/1e3))
            freq_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            freq_slider.setTickInterval(int(1e6))
            freq_slider.sliderMoved.connect(worker.update_freq) # es gibt auch valueChanged
            freq_label = QLabel()
            def update_freq_label(val):
                freq_label.setText("Frequency [MHz]: " + str(val/1e3))
                freq_plot.autoRange()
            freq_slider.sliderMoved.connect(update_freq_label)
            update_freq_label(freq_slider.value()) # Label initialisieren
            layout.addWidget(freq_slider, 4, 0)
            layout.addWidget(freq_label, 4, 1)

            # Verstärkungs-Schieberegler mit Label
            gain_slider = QSlider(Qt.Orientation.Horizontal)
            gain_slider.setRange(0, 73)
            gain_slider.setValue(gain)
            gain_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            gain_slider.setTickInterval(2)
            gain_slider.sliderMoved.connect(worker.update_gain)
            gain_label = QLabel()
            def update_gain_label(val):
                gain_label.setText("Gain: " + str(val))
            gain_slider.sliderMoved.connect(update_gain_label)
            update_gain_label(gain_slider.value()) # Label initialisieren
            layout.addWidget(gain_slider, 5, 0)
            layout.addWidget(gain_label, 5, 1)

            # Abtastraten-Dropdown mit QComboBox
            sample_rate_combobox = QComboBox()
            sample_rate_combobox.addItems([str(x) + ' MHz' for x in sample_rates])
            sample_rate_combobox.setCurrentIndex(0) # muss mit dem Standard oben übereinstimmen
            sample_rate_combobox.currentIndexChanged.connect(worker.update_sample_rate)
            sample_rate_label = QLabel()
            def update_sample_rate_label(val):
                sample_rate_label.setText("Sample Rate: " + str(sample_rates[val]) + " MHz")
            sample_rate_combobox.currentIndexChanged.connect(update_sample_rate_label)
            update_sample_rate_label(sample_rate_combobox.currentIndex()) # Label initialisieren
            layout.addWidget(sample_rate_combobox, 6, 0)
            layout.addWidget(sample_rate_label, 6, 1)

            central_widget = QWidget()
            central_widget.setLayout(layout)
            self.setCentralWidget(central_widget)

            # Signals und Slots
            def time_plot_callback(samples):
                time_plot_curve_i.setData(samples.real)
                time_plot_curve_q.setData(samples.imag)

            def freq_plot_callback(PSD_avg):
                # TODO: Prüfen, ob man nur die visuellen Ticks ändern kann statt der x-Werte
                f = np.linspace(freq_slider.value()*1e3 - worker.sample_rate/2.0, freq_slider.value()*1e3 + worker.sample_rate/2.0, fft_size) / 1e6
                freq_plot_curve.setData(f, PSD_avg)
                freq_plot.setXRange(freq_slider.value()*1e3/1e6 - worker.sample_rate/2e6, freq_slider.value()*1e3/1e6 + worker.sample_rate/2e6)

            def waterfall_plot_callback(spectrogram):
                imageitem.setImage(spectrogram, autoLevels=False)
                sigma = np.std(spectrogram)
                mean = np.mean(spectrogram)
                self.spectrogram_min = mean - 2*sigma # im Fensterzustand speichern
                self.spectrogram_max = mean + 2*sigma

            def end_of_run_callback():
                QTimer.singleShot(0, worker.run) # Worker sofort erneut ausführen

            worker.time_plot_update.connect(time_plot_callback) # Signal mit Callback verbinden
            worker.freq_plot_update.connect(freq_plot_callback)
            worker.waterfall_plot_update.connect(waterfall_plot_callback)
            worker.end_of_run.connect(end_of_run_callback)

            self.sdr_thread.started.connect(worker.run) # startet Worker, wenn Thread beginnt
            self.sdr_thread.start()


    app = QApplication([])
    window = MainWindow()
    window.show() # Fenster sind standardmäßig ausgeblendet
    signal.signal(signal.SIGINT, signal.SIG_DFL) # ermöglicht, dass Strg+C die App wirklich schließt
    app.exec() # Ereignisschleife starten

    if sdr_type == "usrp":
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        streamer.issue_stream_cmd(stream_cmd)
