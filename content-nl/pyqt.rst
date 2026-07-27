.. _pyqt-chapter:

##########################
Realtime GUI's met PyQt
##########################

In dit hoofdstuk leren we hoe je realtime grafische gebruikersinterfaces (GUI's) in Python maakt met PyQt, de Python-bindings voor Qt. Als onderdeel van dit hoofdstuk bouwen we een spectrum analyzer met tijd-, frequentie- en spectrogram/waterfall-weergave, plus invoerwidgets om verschillende SDR-parameters aan te passen. Het voorbeeld ondersteunt PlutoSDR, USRP en een simulatiemodus.

****************
Introductie
****************

Qt (uitgesproken als het engelse woord "Cute") is een framework om GUI-applicaties te maken die op Linux, Windows, macOS en zelfs Android kunnen draaien. Het is een krachtig framework dat in veel commerciële applicaties wordt gebruikt en in C++ is geschreven voor hoge prestaties. PyQt is de Python-verbinding met Qt en biedt daarmee een manier om GUI-applicaties in Python te bouwen, terwijl je profiteert van de prestaties van het onderliggende C++-framework. In dit hoofdstuk gebruiken we PyQt om een realtime spectrum analyzer te bouwen die met een SDR (of met een gesimuleerd signaal) werkt. De analyzer krijgt tijd-, frequentie- en spectrogram/waterfall-weergaven, plus invoerwidgets om SDR-parameters bij te sturen. Voor het plotten gebruiken we `PyQtGraph <https://www.pyqtgraph.org/>`_, een aparte library bovenop PyQt. Voor invoer gebruiken we sliders, combo-boxes en push-buttons. Het voorbeeld ondersteunt PlutoSDR, USRP en simulatiemodus. Hoewel de voorbeeldcode PyQt6 gebruikt, is vrijwel elke regel identiek aan PyQt5 (op de :code:`import` na); qua API is er weinig veranderd tussen die versies. Dit hoofdstuk bevat daarom veel Python-code met uitleg via voorbeelden. Aan het einde heb je de belangrijkste bouwstenen in handen om je eigen interactieve SDR-app te maken.

****************
Qt-overzicht
****************

Qt is een groot framework en we behandelen slechts een klein deel van de mogelijkheden. Er zijn wel enkele kernconcepten die belangrijk zijn bij werken met Qt/PyQt:

- **Widgets**: Widgets zijn de bouwstenen van een Qt-applicatie en vormen de GUI. Er zijn veel soorten widgets, zoals knoppen, sliders, labels en plots. Widgets worden in layouts geplaatst, die bepalen hoe ze op het scherm staan.

- **Layouts**: Layouts worden gebruikt om widgets in een venster te ordenen. Er zijn meerdere types, waaronder horizontale, verticale, grid- en form-layouts. Layouts maken complexe GUI's mogelijk die goed reageren op veranderingen in venstergrootte.

- **Signals en Slots**: Signals en slots zijn een manier om tussen onderdelen van een Qt-applicatie te communiceren. Een signal wordt uitgezonden wanneer een gebeurtenis plaatsvindt en is gekoppeld aan een slot (een callbackfunctie) die dan wordt uitgevoerd. Dit maakt een event-driven structuur mogelijk en houdt de GUI responsief.

- **Style Sheets**: Style sheets worden gebruikt om het uiterlijk van widgets aan te passen. Ze zijn geschreven in een CSS-achtige taal en kunnen kleur, lettertype en grootte wijzigen.

- **Graphics**: Qt heeft een krachtig graphics-framework om custom grafische elementen te maken. Het bevat classes voor lijnen, rechthoeken, ellipsen en tekst, plus klassen voor muis- en toetsenbordevents.

- **Multithreading**: Qt ondersteunt multithreading ingebouwd en biedt classes om worker-threads op de achtergrond te draaien. Daarmee kun je langdurige taken uitvoeren zonder de hoofd-GUI-thread te blokkeren.

- **OpenGL**: Qt heeft ingebouwde OpenGL-ondersteuning en classes voor 3D-graphics. Dat is nuttig voor toepassingen die hoge 3D-prestaties vragen. In dit hoofdstuk richten we ons alleen op 2D-toepassingen.

*******************************
Basislayout van een Applicatie
*******************************

Voordat we de verschillende Qt-widgets behandelen, kijken we naar de layout van een typische Qt-applicatie. Een Qt-app bestaat uit een hoofdvenster met daarin een centrale widget, die op zijn beurt de hoofdinhoud bevat. Met PyQt kunnen we een minimale app maken met slechts een enkele QPushButton:

.. code-block:: python

    from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton

    # Subclass QMainWindow to customize your application's main window
    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            
            # Example GUI component
            example_button = QPushButton('Push Me')
            def on_button_click():
                print("beep")
            example_button.clicked.connect(on_button_click)

            self.setCentralWidget(example_button)

    app = QApplication([])
    window = MainWindow()
    window.show() # Windows are hidden by default
    app.exec() # Start the event loop

Probeer de code zelf uit; waarschijnlijk moet je :code:`pip install PyQt6` uitvoeren. Merk op dat de allerlaatste regel blokkerend is: alles wat je daaronder zet, draait pas nadat je het venster sluit. De gemaakte QPushButton heeft zijn :code:`clicked`-signal gekoppeld aan een callback die "beep" naar de console print.

*******************************
Applicatie met Worker-thread
*******************************

Er is een probleem met het minimale voorbeeld: er is geen goede plek voor SDR/DSP-code. De :code:`__init__` van :code:`MainWindow` is bedoeld voor GUI-configuratie en callbacks, maar daar wil je geen andere logica (zoals SDR of DSP) in stoppen. De reden: de GUI is single-threaded. Als je de GUI-thread blokkeert met langdurige code, bevriest of stottert de interface. Daarom gebruiken we een worker-thread om SDR/DSP op de achtergrond uit te voeren.

Het onderstaande voorbeeld breidt het minimale voorbeeld uit met een worker-thread die code in de :code:`run`-functie continu laat draaien. We gebruiken bewust geen :code:`while True:`, omdat we door de interne werking van PyQt willen dat :code:`run` periodiek afrondt en opnieuw start. Daarom koppelen we het :code:`end_of_run`-signal van de worker-thread (volgende sectie) aan een callback die :code:`run` opnieuw triggert. We initialiseren de worker-thread in :code:`MainWindow`, door een :code:`QThread` te maken en onze custom worker eraan toe te wijzen. Dit lijkt misschien complex, maar het is een veelgebruikt patroon in PyQt-apps. Belangrijkste punt: GUI-code hoort in :code:`MainWindow`, SDR/DSP-code in de worker-thread (:code:`run`).

.. code-block:: python

    from PyQt6.QtCore import QThread, pyqtSignal, QObject, QTimer
    from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton
    import time

    # Non-GUI operations (including SDR) need to run in a separate thread
    class SDRWorker(QObject):
        end_of_run = pyqtSignal()

        # Main loop
        def run(self):
            print("Starting run()")
            time.sleep(1)
            self.end_of_run.emit() # let MainWindow know we're done

    # Subclass QMainWindow to customize your application's main window
    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()

            # Initialize worker and thread
            self.sdr_thread = QThread()
            worker = SDRWorker()
            worker.moveToThread(self.sdr_thread)
            
            # Example GUI component
            example_button = QPushButton('Push Me')
            def on_button_click():
                print("beep")
            example_button.clicked.connect(on_button_click)
            self.setCentralWidget(example_button)

            # This is what keeps the run() function repeating nonstop
            def end_of_run_callback():
                QTimer.singleShot(0, worker.run) # Run worker again immediately
            worker.end_of_run.connect(end_of_run_callback)

            self.sdr_thread.started.connect(worker.run) # kicks off the first run() when the thread starts
            self.sdr_thread.start() # start thread

    app = QApplication([])
    window = MainWindow()
    window.show() # Windows are hidden by default
    app.exec() # Start the event loop

Probeer de code hierboven uit. Je zou elke seconde "Starting run()" in de console moeten zien en de knop moet zonder merkbare vertraging blijven werken. In de worker-thread doen we nu alleen print en sleep, maar zo voegen we straks eenvoudig SDR- en DSP-code toe.

*************************
Signals en Slots
*************************

In het vorige voorbeeld gebruikten we :code:`end_of_run` om tussen worker-thread en GUI-thread te communiceren. Dit is een veelvoorkomend patroon in PyQt, het "signals en slots"-mechanisme. Een signal wordt uitgezonden door een object (hier: de worker-thread) en gekoppeld aan een slot (hier: callback :code:`end_of_run_callback` in de GUI-thread). Een signal kan aan meerdere slots gekoppeld worden, en een slot aan meerdere signals. Een signal kan ook argumenten meedragen die aan het slot worden doorgegeven. Dit werkt ook andersom: de GUI-thread kan een signal naar een slot in de worker-thread sturen. Het signal/slot-mechanisme is een krachtige manier om onderdelen van een PyQt-app event-driven te laten samenwerken en komt veel terug in de code hieronder. Denk simpel: een slot is een callbackfunctie; een signal triggert die callback.

*************************
PyQtGraph
*************************

PyQtGraph is een library bovenop PyQt en NumPy die snelle, efficiente plotting biedt, omdat PyQt zelf te algemeen is om uitgebreide plotfunctionaliteit standaard te bevatten. De library is ontworpen voor realtime toepassingen en geoptimaliseerd voor snelheid. In veel opzichten lijkt het op Matplotlib, maar PyQtGraph is meer gericht op continue updates dan op losse statische plots. Met het eenvoudige voorbeeld hieronder kun je de prestaties van PyQtGraph vergelijken met Matplotlib door :code:`if True:` te wijzigen naar :code:`False:`. Op een Intel Core i9-10900K @ 3.70 GHz haalde PyQtGraph meer dan 1000 FPS, terwijl Matplotlib rond 40 FPS zat. Als Matplotlib jou toch voordeel geeft (bijvoorbeeld ontwikkeltijd of een specifieke feature), kun je Matplotlib-plots ook in een PyQt-app integreren, met de onderstaande code als startpunt.

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

                # Setup a timer to trigger the redraw by calling update_plot.
                self.timer = QtCore.QTimer()
                self.timer.setInterval(0) # causes the timer to start immediately
                self.timer.timeout.connect(self.update_plot) # causes the timer to start itself again automatically
                self.timer.start()
                self.start_t = time.time() # used for benchmarking

                self.show()

            def update_plot(self):
                self._plot_ref.set_ydata(np.random.randn(n_data))
                self.canvas.draw() # Trigger the canvas to update and redraw.
                print('FPS:', 1/(time.time()-self.start_t)) # got ~42 FPS on an i9-10900K
                self.start_t = time.time()

    else:
        class MainWindow(QtWidgets.QMainWindow):
            def __init__(self):
                super(MainWindow, self).__init__()
                
                self.time_plot = pg.PlotWidget()
                self.time_plot.setYRange(-5, 5)
                self.time_plot_curve = self.time_plot.plot([])
                self.setCentralWidget(self.time_plot)

                # Setup a timer to trigger the redraw by calling update_plot.
                self.timer = QtCore.QTimer()
                self.timer.setInterval(0) # causes the timer to start immediately
                self.timer.timeout.connect(self.update_plot) # causes the timer to start itself again automatically
                self.timer.start()
                self.start_t = time.time() # used for benchmarking

                self.show()

            def update_plot(self):
                self.time_plot_curve.setData(np.random.randn(n_data))
                print('FPS:', 1/(time.time()-self.start_t)) # got ~42 FPS on an i9-10900K
                self.start_t = time.time()

    app = QtWidgets.QApplication([])
    w = MainWindow()
    app.exec()

.. raw:: html

    </details>

Qua gebruik van PyQtGraph importeren we het met :code:`import pyqtgraph as pg` en maken daarna een Qt-widget voor een 1D-plot, zoals hieronder (deze code hoort in :code:`MainWindow.__init__`):

.. code-block:: python

        # Example PyQtGraph plot
        time_plot = pg.PlotWidget(labels={'left': 'Amplitude', 'bottom': 'Time'})
        time_plot_curve = time_plot.plot(np.arange(1000), np.random.randn(1000)) # x and y
        time_plot.setYRange(-5, 5)

        self.setCentralWidget(time_plot)

.. image:: ../_images/pyqtgraph_example.png
   :scale: 80 % 
   :align: center
   :alt: PyQtGraph-voorbeeld

Je ziet dat een plot opzetten relatief eenvoudig is en dat het resultaat gewoon een extra widget in je GUI is. Naast 1D-plots heeft PyQtGraph ook een equivalent van Matplotlib's :code:`imshow()` voor 2D-weergave met colormap, wat we gebruiken voor onze realtime spectrogram/waterfall. Een groot voordeel is dat de plots gewone Qt-widgets zijn, zodat je met pure PyQt extra elementen kunt toevoegen (bijvoorbeeld een rechthoek op een bepaalde locatie). Dat komt doordat PyQtGraph gebruikmaakt van PyQt's :code:`QGraphicsScene`, een oppervlak voor veel 2D-objecten. Je kunt dus zonder probleem lijnen, rechthoeken, tekst, ellipsen, polygonen en bitmaps toevoegen met standaard PyQt.

*******
Layouts
*******

In de voorbeelden hierboven gebruikten we :code:`self.setCentralWidget()` om de hoofdwidget van het venster te zetten. Dat is eenvoudig, maar beperkt voor complexere layouts. Daarvoor gebruik je layouts om widgets te rangschikken. Er zijn meerdere types, waaronder :code:`QHBoxLayout`, :code:`QVBoxLayout`, :code:`QGridLayout` en :code:`QFormLayout`. :code:`QHBoxLayout` en :code:`QVBoxLayout` plaatsen widgets respectievelijk horizontaal en verticaal. :code:`QGridLayout` plaatst widgets in een raster, en :code:`QFormLayout` in twee kolommen met labels links en invoerwidgets rechts.

Om een nieuwe layout te maken en widgets toe te voegen, probeer het volgende in :code:`MainWindow.__init__`:

.. code-block:: python

    layout = QHBoxLayout()
    layout.addWidget(QPushButton("Left-Most"))
    layout.addWidget(QPushButton("Center"), 1)
    layout.addWidget(QPushButton("Right-Most"), 2)
    self.setLayout(layout)

In dit voorbeeld stapelen we widgets horizontaal. Door :code:`QHBoxLayout` te vervangen door :code:`QVBoxLayout` stapel je ze verticaal. De functie :code:`addWidget` voegt widgets toe aan de layout, en het optionele tweede argument is een stretchfactor die bepaalt hoeveel ruimte de widget relatief inneemt.

:code:`QGridLayout` heeft extra parameters omdat je rij en kolom expliciet opgeeft. Je kunt optioneel ook aangeven over hoeveel rijen en kolommen een widget moet lopen (standaard 1 en 1). Voorbeeld:

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
    :alt: Qt-layouts met voorbeelden van QHBoxLayout, QVBoxLayout en QGridLayout

Voor onze spectrum analyzer gebruiken we :code:`QGridLayout` als hoofdlayout, maar voegen we ook :code:`QHBoxLayout` toe om widgets horizontaal te stapelen binnen een cel van het grid. Layouts kun je eenvoudig nesten door een nieuwe layout te maken en die aan de bovenliggende layout toe te voegen, bijvoorbeeld:

.. code-block:: python

    layout = QGridLayout()
    self.setLayout(layout)
    inner_layout = QHBoxLayout()
    layout.addLayout(inner_layout)

*******************
:code:`QPushButton`
*******************

De eerste widget die we behandelen is :code:`QPushButton`, een eenvoudige klikbare knop. We zagen al hoe je een :code:`QPushButton` maakt en het :code:`clicked`-signal aan een callback koppelt. :code:`QPushButton` heeft ook andere signals, zoals :code:`pressed`, :code:`released` en :code:`toggled`. Het :code:`toggled`-signal komt vrij wanneer een knop wordt in- of uitgeschakeld en is handig voor toggle-knoppen. Verder zijn er properties zoals :code:`text`, :code:`icon` en :code:`checkable`. Er is ook een :code:`click()`-methode om een klik te simuleren. In onze SDR spectrum analyzer gebruiken we knoppen om auto-range voor plots te triggeren op basis van actuele data. Omdat we :code:`QPushButton` al gebruikt hebben, gaan we hier niet dieper in op details; zie de `QPushButton-documentatie <https://doc.qt.io/qtforpython/PySide6/QtWidgets/QPushButton.html>`_.

***************
:code:`QSlider`
***************

De :code:`QSlider` is een widget waarmee de gebruiker een waarde uit een bereik kiest. Belangrijke properties zijn :code:`minimum`, :code:`maximum`, :code:`value` en :code:`orientation`. Belangrijke signals zijn :code:`valueChanged`, :code:`sliderPressed` en :code:`sliderReleased`. Met :code:`setValue()` zet je de sliderwaarde, wat we vaak gebruiken. De documentatie staat `hier voor QSlider <https://doc.qt.io/qtforpython/PySide6/QtWidgets/QSlider.html>`_.

In onze spectrum analyzer gebruiken we :code:`QSlider`-widgets om centerfrequentie en gain van de SDR aan te passen. Hieronder staat de snippet uit de uiteindelijke app voor de gain-slider:

.. code-block:: python

    # Gain slider with label
    gain_slider = QSlider(Qt.Orientation.Horizontal)
    gain_slider.setRange(0, 73) # min and max, inclusive. interval is always 1
    gain_slider.setValue(50) # initial value
    gain_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
    gain_slider.setTickInterval(2) # for visual purposes only
    gain_slider.sliderMoved.connect(worker.update_gain)
    gain_label = QLabel()
    def update_gain_label(val):
        gain_label.setText("Gain: " + str(val))
    gain_slider.sliderMoved.connect(update_gain_label)
    update_gain_label(gain_slider.value()) # initialize the label
    layout.addWidget(gain_slider, 5, 0)
    layout.addWidget(gain_label, 5, 1)

Een belangrijk punt bij :code:`QSlider`: deze werkt met gehele getallen. Met bereik 0 tot 73 kan de slider dus alleen integers tussen die waarden kiezen (inclusief begin en eind). :code:`setTickInterval(2)` is alleen visueel. Daarom gebruiken we kHz als eenheid voor de frequentieslider, zodat we tot 1 kHz resolutie hebben.

Halverwege de code zie je dat we een :code:`QLabel` maken, een tekstlabel voor weergave. Om daar de actuele sliderwaarde in te tonen, maken we een slot (callbackfunctie) die het label bijwerkt. Die callback koppelen we aan :code:`sliderMoved`, dat automatisch wordt uitgezonden bij het bewegen van de slider. We roepen de callback ook eenmalig aan om het label met de beginwaarde te initialiseren (50 in ons geval). Daarnaast koppelen we :code:`sliderMoved` aan een slot in de worker-thread die de SDR-gain aanpast (SDR-beheer en DSP willen we niet in de hoofd-GUI-thread doen). Die slot-callback bespreken we later.

*****************
:code:`QComboBox`
*****************

De :code:`QComboBox` is een dropdown-widget waarmee de gebruiker een item uit een lijst kiest. Belangrijke properties zijn :code:`currentText`, :code:`currentIndex` en :code:`count`. Belangrijke signals zijn :code:`currentTextChanged`, :code:`currentIndexChanged` en :code:`activated`. Daarnaast heeft :code:`QComboBox` methodes zoals :code:`addItem()` om items toe te voegen en :code:`insertItem()` om op een specifieke index in te voegen; die laatste gebruiken we in dit voorbeeld niet. De documentatie staat `hier voor QComboBox <https://doc.qt.io/qtforpython/PySide6/QtWidgets/QComboBox.html>`_.

In onze spectrum analyzer gebruiken we :code:`QComboBox` om de sample rate uit een vooraf gedefinieerde lijst te kiezen. Aan het begin van de code zetten we bijvoorbeeld :code:`sample_rates = [56, 40, 20, 10, 5, 2, 1, 0.5]`. Binnen :code:`MainWindow.__init__` maken we de :code:`QComboBox` als volgt:

.. code-block:: python

    # Sample rate dropdown using QComboBox
    sample_rate_combobox = QComboBox()
    sample_rate_combobox.addItems([str(x) + ' MHz' for x in sample_rates])
    sample_rate_combobox.setCurrentIndex(0) # must give it the index, not string
    sample_rate_combobox.currentIndexChanged.connect(worker.update_sample_rate)
    sample_rate_label = QLabel()
    def update_sample_rate_label(val):
        sample_rate_label.setText("Sample Rate: " + str(sample_rates[val]) + " MHz")
    sample_rate_combobox.currentIndexChanged.connect(update_sample_rate_label)
    update_sample_rate_label(sample_rate_combobox.currentIndex()) # initialize the label
    layout.addWidget(sample_rate_combobox, 6, 0)
    layout.addWidget(sample_rate_label, 6, 1)

Het belangrijkste verschil met de slider is :code:`addItems()` (je geeft een lijst strings als opties mee) en :code:`setCurrentIndex()` (je zet de startwaarde via index).

****************
Lambdafuncties
****************

Herinner je de code van hierboven waar we dit deden:

.. code-block:: python

    def update_sample_rate_label(val):
        sample_rate_label.setText("Sample Rate: " + str(sample_rates[val]) + " MHz")
    sample_rate_combobox.currentIndexChanged.connect(update_sample_rate_label)

We maken hier een functie met slechts een regel code en geven die functie (functies zijn ook objecten) door aan :code:`connect()`. Om dit patroon te vereenvoudigen, schrijven we het eerst om in basis-Python:

.. code-block:: python

    def my_function(x):
        print(x)
    y.call_that_takes_in_function_obj(my_function)

In deze situatie heeft de functie maar een regel code en gebruiken we die functie maar eenmaal, bij het zetten van de :code:`connect`-callback. In zulke gevallen kun je een lambdafunctie gebruiken, een manier om een functie in een regel te definieren. De code hierboven herschreven met lambda:

.. code-block:: python

    y.call_that_takes_in_function_obj(lambda x: print(x))

Als je nog niet eerder lambdafuncties hebt gebruikt, kan dit vreemd overkomen. Je bent niet verplicht ze te gebruiken, maar ze besparen vaak enkele regels en maken code compacter. Werking: de tijdelijke argumentnaam staat na "lambda", en alles na de dubbele punt is de code die op dat argument werkt. Dit ondersteunt ook meerdere argumenten met komma's, of zelfs geen argumenten met :code:`lambda : <code>`. Als oefening kun je :code:`update_sample_rate_label` hierboven herschrijven met een lambdafunctie.

************************
PlotWidget van PyQtGraph
************************

PyQtGraph's :code:`PlotWidget` is een PyQt-widget voor 1D-plots, vergelijkbaar met Matplotlib's :code:`plt.plot(x,y)`. Wij gebruiken deze voor tijd- en frequentieplots (PSD), al werkt hij ook goed voor IQ-plots (die onze analyzer niet bevat). Voor wie dieper wil: PlotWidget is een subclass van PyQt's `QGraphicsView <https://doc.qt.io/qtforpython-5/PySide2/QtWidgets/QGraphicsView.html>`_, een widget om de inhoud van een `QGraphicsScene <https://doc.qt.io/qtforpython-5/PySide2/QtWidgets/QGraphicsScene.html#PySide2.QtWidgets.PySide2.QtWidgets.QGraphicsScene>`_ te tonen. Die scene is een oppervlak voor veel 2D-grafische items in Qt. Belangrijk voor gebruik: PlotWidget is in de kern gewoon een widget met een enkel `PlotItem <https://pyqtgraph.readthedocs.io/en/latest/api_reference/graphicsItems/plotitem.html#pyqtgraph.PlotItem>`_. Vanuit documentatieperspectief kun je daarom vaak direct naar de PlotItem-documentatie gaan: `<https://pyqtgraph.readthedocs.io/en/latest/api_reference/graphicsItems/plotitem.html>`_. Een PlotItem bevat een ViewBox voor de data plus AxisItems en labels voor assen en titel.

Het eenvoudigste voorbeeld van PlotWidget-gebruik is als volgt (plaats dit in :code:`MainWindow.__init__`):

.. code-block:: python

 import pyqtgraph as pg
 plotWidget = pg.plot(title="My Title")
 plotWidget.plot(x, y)

waar x en y doorgaans NumPy-arrays zijn, net als bij Matplotlib's :code:`plt.plot()`. Dit is echter een statische plot waarin de data niet verandert. Voor onze spectrum analyzer willen we data in de worker-thread updaten, dus bij initialisatie van de plot hoeven we nog geen data mee te geven; alleen opzetten is genoeg. Zo initialiseren we de tijd-domeinplot:

.. code-block:: python

    # Time plot
    time_plot = pg.PlotWidget(labels={'left': 'Amplitude', 'bottom': 'Time [microseconds]'})
    time_plot.setMouseEnabled(x=False, y=True)
    time_plot.setYRange(-1.1, 1.1)
    time_plot_curve_i = time_plot.plot([]) 
    time_plot_curve_q = time_plot.plot([]) 
    layout.addWidget(time_plot, 1, 0)

Je ziet dat we twee curves maken: een voor I en een voor Q. De rest spreekt grotendeels voor zich. Om de plot te kunnen updaten, maken we een slot (callbackfunctie) in :code:`MainWindow.__init__`:

.. code-block:: python

    def time_plot_callback(samples):
        time_plot_curve_i.setData(samples.real)
        time_plot_curve_q.setData(samples.imag)

Dit slot koppelen we aan het signal van de worker-thread dat wordt uitgezonden wanneer nieuwe samples beschikbaar zijn, zoals later te zien is.

Het laatste dat we in :code:`MainWindow.__init__` doen is rechts naast de plot een paar knoppen toevoegen die auto-range triggeren. De ene gebruikt de huidige min/max, de andere zet het bereik op -1.1 tot 1.1 (de ADC-limieten van veel SDR's plus 10% marge). We maken hiervoor een geneste layout, specifiek QVBoxLayout, om de twee knoppen verticaal te stapelen. De code:

.. code-block:: python

    # Time plot auto range buttons
    time_plot_auto_range_layout = QVBoxLayout()
    layout.addLayout(time_plot_auto_range_layout, 1, 1)
    auto_range_button = QPushButton('Auto Range')
    auto_range_button.clicked.connect(lambda : time_plot.autoRange()) # lambda just means its an unnamed function
    time_plot_auto_range_layout.addWidget(auto_range_button)
    auto_range_button2 = QPushButton('-1 to +1\n(ADC limits)')
    auto_range_button2.clicked.connect(lambda : time_plot.setYRange(-1.1, 1.1))
    time_plot_auto_range_layout.addWidget(auto_range_button2)

En zo ziet het er uiteindelijk uit:

.. image:: ../_images/pyqt_time_plot.png
   :scale: 50 % 
   :align: center
   :alt: PyQtGraph tijdplot

Voor de frequentiedomeinplot (PSD) gebruiken we een vergelijkbaar patroon.

***********************
ImageItem van PyQtGraph
***********************

Een spectrum analyzer is niet compleet zonder waterfall (realtime spectrogram), en daarvoor gebruiken we PyQtGraph's ImageItem, dat beelden met 1, 3 of 4 "kanalen" rendert. Een kanaal betekent dat je een 2D-array met floats of ints aanbiedt; vervolgens wordt via een lookup table (LUT) een colormap toegepast om het beeld te maken. Je kunt ook RGB (3 kanalen) of RGBA (4 kanalen) aanleveren. Wij berekenen ons spectrogram als 2D NumPy-array met floats en geven die direct aan ImageItem. We kiezen een colormap en gebruiken ook de ingebouwde LUT-weergave die de waardeverdeling van de data en de kleurtoewijzing laat zien.

De initialisatie van de waterfall-plot is vrij eenvoudig: we gebruiken een PlotWidget als container (zodat x- en y-as zichtbaar blijven) en voegen daar een ImageItem aan toe:

.. code-block:: python

    # Waterfall plot
    waterfall = pg.PlotWidget(labels={'left': 'Time [s]', 'bottom': 'Frequency [MHz]'})
    imageitem = pg.ImageItem(axisOrder='col-major') # this arg is purely for performance
    waterfall.addItem(imageitem)
    waterfall.setMouseEnabled(x=False, y=False)
    waterfall_layout.addWidget(waterfall)

Het slot/de callback voor het updaten van de waterfall-data, eveneens in :code:`MainWindow.__init__`, is:

.. code-block:: python

    def waterfall_plot_callback(spectrogram):
        imageitem.setImage(spectrogram, autoLevels=False)
        sigma = np.std(spectrogram)
        mean = np.mean(spectrogram) 
        self.spectrogram_min = mean - 2*sigma # save to window state
        self.spectrogram_max = mean + 2*sigma

Hierbij is spectrogram een 2D NumPy-array met floats. Naast het zetten van de beelddata berekenen we een min en max voor de colormap op basis van gemiddelde en variantie van de data, die we later gebruiken. Het laatste GUI-deel voor het spectrogram is de colorbar, die ook de gebruikte colormap bepaalt:

.. code-block:: python

    # Colorbar for waterfall
    colorbar = pg.HistogramLUTWidget()
    colorbar.setImageItem(imageitem) # connects the bar to the waterfall imageitem
    colorbar.item.gradient.loadPreset('viridis') # set the color map, also sets the imageitem
    imageitem.setLevels((-30, 20)) # needs to come after colorbar is created for some reason
    waterfall_layout.addWidget(colorbar)

De tweede regel is belangrijk: die koppelt de colorbar daadwerkelijk aan het ImageItem. Hier kiezen we ook de colormap en de startniveaus (-30 dB tot +20 dB in ons geval). In de worker-thread-code zie je hoe de 2D spectrogramarray wordt berekend/opgeslagen. Hieronder staat een screenshot van dit GUI-deel; let op de sterke ingebouwde functionaliteit van colorbar en LUT-weergave. De zijwaartse klokvormige curve is de verdeling van spectrogramwaarden, wat erg nuttig is.

.. image:: ../_images/pyqt_spectrogram.png
   :scale: 50 % 
   :align: center
   :alt: PyQtGraph-spectrogram en colorbar

***********************
Worker-thread
***********************

Aan het begin van dit hoofdstuk zagen we hoe je een aparte thread maakt met een class genaamd SDRWorker en een run()-functie. Daar zetten we alle SDR- en DSP-code in, behalve de SDR-initialisatie die we voorlopig globaal doen. De worker-thread werkt ook de drie plots bij door signals uit te zenden zodra nieuwe samples beschikbaar zijn. Die triggeren callbacks in :code:`MainWindow` die de plots daadwerkelijk verversen. De SDRWorker-class is op te delen in drie onderdelen:

#. :code:`init()` - initialiseert status, bijvoorbeeld de 2D spectrogramarray
#. PyQt Signals - hier definieren we custom signals die we uitzenden
#. PyQt Slots - callbacks die reageren op GUI-events, zoals een bewegende slider
#. :code:`run()` - de hoofdloop die continu draait

***********************
PyQt-signals
***********************

In de GUI-code hoefden we geen eigen signals te definieren, omdat die al in widgets ingebouwd zijn, zoals :code:`QSlider.valueChanged`. Onze SDRWorker-class is custom, dus de signals die we willen uitzenden moeten we zelf definieren voordat :code:`run()` wordt gebruikt. Hieronder de vier gebruikte signals met hun datatypen:

.. code-block:: python

    # PyQt Signals
    time_plot_update = pyqtSignal(np.ndarray)
    freq_plot_update = pyqtSignal(np.ndarray)
    waterfall_plot_update = pyqtSignal(np.ndarray)
    end_of_run = pyqtSignal() # happens many times a second

De eerste drie signals sturen een enkel object mee (een NumPy-array). Het laatste signal stuurt geen object mee. Je kunt ook meerdere objecten tegelijk sturen door datatypen met komma's te scheiden, maar dat is hier niet nodig. Binnen :code:`run()` kun je op elke plek een signal naar de GUI-thread uitsturen met een regel code, bijvoorbeeld:

.. code-block:: python

    self.time_plot_update.emit(samples)

Er is nog een laatste stap voor alle signal/slot-koppelingen: in de GUI-code (helemaal aan het eind van :code:`MainWindow.__init__`) moeten we de signals van de worker-thread verbinden met slots in de GUI, bijvoorbeeld:

.. code-block:: python

    worker.time_plot_update.connect(time_plot_callback) # connect the signal to the callback

Onthoud dat :code:`worker` de instantie is van SDRWorker die we in de GUI-code hebben gemaakt. We koppelen hierboven dus het worker-signal :code:`time_plot_update` aan het GUI-slot :code:`time_plot_callback` dat eerder is gedefinieerd. Dit is een goed moment om de snippets terug te bekijken en te zien hoe alles samenwerkt, zodat duidelijk is hoe GUI-thread en worker-thread communiceren; dat is cruciaal in PyQt-programmering.

*************************
Slots in de Worker-thread
*************************

De slots van de worker-thread zijn callbacks die door GUI-events worden getriggerd, zoals het bewegen van de gain-slider. Ze zijn vrij rechttoe rechtaan; dit slot zet bijvoorbeeld de SDR-gain op de nieuwe sliderwaarde:

.. code-block:: python

    def update_gain(self, val):
        print("Updated gain to:", val, 'dB')
        sdr.set_rx_gain(val)

**************************
Run() van de Worker-thread
**************************

In de :code:`run()`-functie gebeurt het eigenlijke DSP-werk. In onze applicatie start elke run met het ophalen van samples uit de SDR (of met simulatie als je geen SDR hebt).

.. code-block:: python

    # Main loop
    def run(self):
        if sdr_type == "pluto":
            samples = sdr.rx()/2**11 # Receive samples
        elif sdr_type == "usrp":
            streamer.recv(recv_buffer, metadata)
            samples = recv_buffer[0] # will be np.complex64
        elif sdr_type == "sim":
            tone = np.exp(2j*np.pi*self.sample_rate*0.1*np.arange(fft_size)/self.sample_rate)
            noise = np.random.randn(fft_size) + 1j*np.random.randn(fft_size)
            samples = self.gain*tone*0.02 + 0.1*noise
            # Truncate to -1 to +1 to simulate ADC bit limits
            np.clip(samples.real, -1, 1, out=samples.real)
            np.clip(samples.imag, -1, 1, out=samples.imag)
        
        ...

Zoals je ziet genereren we in de simulatie een toon met wat witte ruis en begrenzen we samples daarna op -1 tot +1.

Nu het DSP-deel: we hebben een FFT nodig voor zowel frequentieplot als spectrogram. De PSD van deze sample-set kan direct dienen als een rij in het spectrogram. We schuiven dus de waterfall een rij op en vullen de nieuwe rij onderaan (of bovenaan) in. Voor elke plotupdate sturen we een signal met de bijgewerkte data. Daarna signaleren we het einde van :code:`run()`, zodat de GUI-thread meteen een nieuwe :code:`run()` start. Al met al is het weinig code:

.. code-block:: python

        ...

        self.time_plot_update.emit(samples[0:time_plot_samples])
        
        PSD = 10.0*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples)))**2/fft_size)
        self.PSD_avg = self.PSD_avg * 0.99 + PSD * 0.01
        self.freq_plot_update.emit(self.PSD_avg)

        self.spectrogram[:] = np.roll(self.spectrogram, 1, axis=1) # shifts waterfall 1 row
        self.spectrogram[:,0] = PSD # fill last row with new fft results
        self.waterfall_plot_update.emit(self.spectrogram)

        self.end_of_run.emit() # emit the signal to keep the loop going
        # end of run()

Let op dat we niet de hele samplebatch naar de tijdplot sturen; dat zijn te veel punten. In plaats daarvan sturen we alleen de eerste 500 samples (instelbaar bovenin het script, hier niet getoond). Voor de PSD-plot gebruiken we een running average door de vorige PSD op te slaan en daar 1% van de nieuwe PSD aan toe te voegen. Dat is een eenvoudige manier om de PSD-plot te egaliseren. De volgorde van :code:`emit()`-aanroepen maakt daarbij niet uit; ze hadden ook allemaal aan het einde van :code:`run()` kunnen staan.

****************************
Volledige Eindvoorbeeldcode
****************************

Tot nu toe bekeken we losse snippets van de spectrum analyzer-app, maar nu kijken we naar de volledige code en proberen we die te draaien. De code ondersteunt momenteel PlutoSDR, USRP en simulatiemodus. Heb je geen Pluto of USRP, laat de code dan zoals die is; dan gebruikt hij simulatiemodus. Anders wijzig je :code:`sdr_type`. In simulatiemodus zie je bij maximale gain dat het signaal in het tijddomein wordt afgeknipt, wat spurs in het frequentiedomein veroorzaakt.

Gebruik deze code gerust als startpunt voor je eigen realtime SDR-app. Hieronder staat ook een animatie van de app in actie: met een Pluto eerst op de 750 MHz cellulaire band en daarna op 2,4 GHz WiFi. Een hogere kwaliteit versie staat op YouTube `hier <https://youtu.be/hvofiY3Q_yo>`_.

.. image:: ../_images/pyqt_animation.gif
   :scale: 100 %
   :align: center
   :alt: Geanimeerde gif van de PyQt spectrum analyzer-app in actie

Bekende bugs (om te helpen oplossen kun je `dit bewerken <https://github.com/777arc/PySDR/edit/master/figure-generating-scripts/pyqt_example.py>`_):

#. De x-as van de waterfall wordt niet bijgewerkt bij wijzigen van centerfrequentie (de PSD-plot wel)

Volledige code:

.. code-block:: python

    from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal, QObject, QTimer
    from PyQt6.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QSlider, QLabel, QHBoxLayout, QVBoxLayout, QPushButton, QComboBox  # tested with PyQt6==6.7.0
    import pyqtgraph as pg # tested with pyqtgraph==0.13.7
    import numpy as np
    import time
    import signal # lets control-C actually close the app

    # Defaults
    fft_size = 4096 # determines buffer size
    num_rows = 200
    center_freq = 750e6
    sample_rates = [56, 40, 20, 10, 5, 2, 1, 0.5] # MHz
    sample_rate = sample_rates[0] * 1e6
    time_plot_samples = 500
    gain = 50 # 0 to 73 dB. int

    sdr_type = "sim" # or "usrp" or "pluto"

    # Init SDR
    if sdr_type == "pluto":
        import adi
        sdr = adi.Pluto("ip:192.168.1.10")
        sdr.rx_lo = int(center_freq)
        sdr.sample_rate = int(sample_rate)
        sdr.rx_rf_bandwidth = int(sample_rate*0.8) # antialiasing filter bandwidth
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

        # Set up the stream and receive buffer
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [0]
        metadata = uhd.types.RXMetadata()
        streamer = usrp.get_rx_stream(st_args)
        recv_buffer = np.zeros((1, fft_size), dtype=np.complex64)

        # Start Stream
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
            self.freq = 0 # in kHz, to deal with QSlider being ints and with a max of 2 billion
            self.spectrogram = -50*np.ones((fft_size, num_rows))
            self.PSD_avg = -50*np.ones(fft_size)

        # PyQt Signals
        time_plot_update = pyqtSignal(np.ndarray)
        freq_plot_update = pyqtSignal(np.ndarray)
        waterfall_plot_update = pyqtSignal(np.ndarray)
        end_of_run = pyqtSignal() # happens many times a second

        # PyQt Slots
        def update_freq(self, val): # TODO: WE COULD JUST MODIFY THE SDR IN THE GUI THREAD
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

        # Main loop
        def run(self):
            start_t = time.time()
                    
            if sdr_type == "pluto":
                samples = sdr.rx()/2**11 # Receive samples
            elif sdr_type == "usrp":
                streamer.recv(recv_buffer, metadata)
                samples = recv_buffer[0] # will be np.complex64
            elif sdr_type == "sim":
                tone = np.exp(2j*np.pi*self.sample_rate*0.1*np.arange(fft_size)/self.sample_rate)
                noise = np.random.randn(fft_size) + 1j*np.random.randn(fft_size)
                samples = self.gain*tone*0.02 + 0.1*noise
                # Truncate to -1 to +1 to simulate ADC bit limits
                np.clip(samples.real, -1, 1, out=samples.real)
                np.clip(samples.imag, -1, 1, out=samples.imag)

            self.time_plot_update.emit(samples[0:time_plot_samples])
            
            PSD = 10.0*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples)))**2/fft_size)
            self.PSD_avg = self.PSD_avg * 0.99 + PSD * 0.01
            self.freq_plot_update.emit(self.PSD_avg)
        
            self.spectrogram[:] = np.roll(self.spectrogram, 1, axis=1) # shifts waterfall 1 row
            self.spectrogram[:,0] = PSD # fill last row with new fft results
            self.waterfall_plot_update.emit(self.spectrogram)

            print("Frames per second:", 1/(time.time() - start_t))
            self.end_of_run.emit() # emit the signal to keep the loop going


    # Subclass QMainWindow to customize your application's main window
    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()

            self.setWindowTitle("The PySDR Spectrum Analyzer")
            self.setFixedSize(QSize(1500, 1000)) # window size, starting size should fit on 1920 x 1080

            self.spectrogram_min = 0
            self.spectrogram_max = 0

            layout = QGridLayout() # overall layout

            # Initialize worker and thread
            self.sdr_thread = QThread()
            self.sdr_thread.setObjectName('SDR_Thread') # so we can see it in htop, note you have to hit F2 -> Display options -> Show custom thread names
            worker = SDRWorker()
            worker.moveToThread(self.sdr_thread)

            # Time plot
            time_plot = pg.PlotWidget(labels={'left': 'Amplitude', 'bottom': 'Time [microseconds]'})
            time_plot.setMouseEnabled(x=False, y=True)
            time_plot.setYRange(-1.1, 1.1)
            time_plot_curve_i = time_plot.plot([]) 
            time_plot_curve_q = time_plot.plot([]) 
            layout.addWidget(time_plot, 1, 0)

            # Time plot auto range buttons
            time_plot_auto_range_layout = QVBoxLayout()
            layout.addLayout(time_plot_auto_range_layout, 1, 1)
            auto_range_button = QPushButton('Auto Range')
            auto_range_button.clicked.connect(lambda : time_plot.autoRange()) # lambda just means its an unnamed function
            time_plot_auto_range_layout.addWidget(auto_range_button)
            auto_range_button2 = QPushButton('-1 to +1\n(ADC limits)')
            auto_range_button2.clicked.connect(lambda : time_plot.setYRange(-1.1, 1.1))
            time_plot_auto_range_layout.addWidget(auto_range_button2)

            # Freq plot
            freq_plot = pg.PlotWidget(labels={'left': 'PSD', 'bottom': 'Frequency [MHz]'})
            freq_plot.setMouseEnabled(x=False, y=True)
            freq_plot_curve = freq_plot.plot([]) 
            freq_plot.setXRange(center_freq/1e6 - sample_rate/2e6, center_freq/1e6 + sample_rate/2e6)
            freq_plot.setYRange(-30, 20)
            layout.addWidget(freq_plot, 2, 0)
            
            # Freq auto range button
            auto_range_button = QPushButton('Auto Range')
            auto_range_button.clicked.connect(lambda : freq_plot.autoRange()) # lambda just means its an unnamed function
            layout.addWidget(auto_range_button, 2, 1)

            # Layout container for waterfall related stuff
            waterfall_layout = QHBoxLayout()
            layout.addLayout(waterfall_layout, 3, 0)

            # Waterfall plot
            waterfall = pg.PlotWidget(labels={'left': 'Time [s]', 'bottom': 'Frequency [MHz]'})
            imageitem = pg.ImageItem(axisOrder='col-major') # this arg is purely for performance
            waterfall.addItem(imageitem)
            waterfall.setMouseEnabled(x=False, y=False)
            waterfall_layout.addWidget(waterfall)

            # Colorbar for waterfall
            colorbar = pg.HistogramLUTWidget()
            colorbar.setImageItem(imageitem) # connects the bar to the waterfall imageitem
            colorbar.item.gradient.loadPreset('viridis') # set the color map, also sets the imageitem
            imageitem.setLevels((-30, 20)) # needs to come after colorbar is created for some reason
            waterfall_layout.addWidget(colorbar)

            # Waterfall auto range button
            auto_range_button = QPushButton('Auto Range\n(-2σ to +2σ)')
            def update_colormap():
                imageitem.setLevels((self.spectrogram_min, self.spectrogram_max))
                colorbar.setLevels(self.spectrogram_min, self.spectrogram_max)
            auto_range_button.clicked.connect(update_colormap)
            layout.addWidget(auto_range_button, 3, 1)

            # Freq slider with label, all units in kHz
            freq_slider = QSlider(Qt.Orientation.Horizontal)
            freq_slider.setRange(0, int(6e6))
            freq_slider.setValue(int(center_freq/1e3))
            freq_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            freq_slider.setTickInterval(int(1e6))
            freq_slider.sliderMoved.connect(worker.update_freq) # there's also a valueChanged option
            freq_label = QLabel()
            def update_freq_label(val):
                freq_label.setText("Frequency [MHz]: " + str(val/1e3))
                freq_plot.autoRange()
            freq_slider.sliderMoved.connect(update_freq_label)
            update_freq_label(freq_slider.value()) # initialize the label
            layout.addWidget(freq_slider, 4, 0)
            layout.addWidget(freq_label, 4, 1)

            # Gain slider with label
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
            update_gain_label(gain_slider.value()) # initialize the label
            layout.addWidget(gain_slider, 5, 0)
            layout.addWidget(gain_label, 5, 1)

            # Sample rate dropdown using QComboBox
            sample_rate_combobox = QComboBox()
            sample_rate_combobox.addItems([str(x) + ' MHz' for x in sample_rates])
            sample_rate_combobox.setCurrentIndex(0) # should match the default at the top
            sample_rate_combobox.currentIndexChanged.connect(worker.update_sample_rate)
            sample_rate_label = QLabel()
            def update_sample_rate_label(val):
                sample_rate_label.setText("Sample Rate: " + str(sample_rates[val]) + " MHz")
            sample_rate_combobox.currentIndexChanged.connect(update_sample_rate_label)
            update_sample_rate_label(sample_rate_combobox.currentIndex()) # initialize the label
            layout.addWidget(sample_rate_combobox, 6, 0)
            layout.addWidget(sample_rate_label, 6, 1)

            central_widget = QWidget()
            central_widget.setLayout(layout)
            self.setCentralWidget(central_widget)

            # Signals and slots stuff
            def time_plot_callback(samples):
                time_plot_curve_i.setData(samples.real)
                time_plot_curve_q.setData(samples.imag)
            
            def freq_plot_callback(PSD_avg):
                # TODO figure out if there's a way to just change the visual ticks instead of the actual x vals
                f = np.linspace(freq_slider.value()*1e3 - worker.sample_rate/2.0, freq_slider.value()*1e3 + worker.sample_rate/2.0, fft_size) / 1e6
                freq_plot_curve.setData(f, PSD_avg)
                freq_plot.setXRange(freq_slider.value()*1e3/1e6 - worker.sample_rate/2e6, freq_slider.value()*1e3/1e6 + worker.sample_rate/2e6)
            
            def waterfall_plot_callback(spectrogram):
                imageitem.setImage(spectrogram, autoLevels=False)
                sigma = np.std(spectrogram)
                mean = np.mean(spectrogram) 
                self.spectrogram_min = mean - 2*sigma # save to window state
                self.spectrogram_max = mean + 2*sigma

            def end_of_run_callback():
                QTimer.singleShot(0, worker.run) # Run worker again immediately
            
            worker.time_plot_update.connect(time_plot_callback) # connect the signal to the callback
            worker.freq_plot_update.connect(freq_plot_callback)
            worker.waterfall_plot_update.connect(waterfall_plot_callback)
            worker.end_of_run.connect(end_of_run_callback)

            self.sdr_thread.started.connect(worker.run) # kicks off the worker when the thread starts
            self.sdr_thread.start()


    app = QApplication([])
    window = MainWindow()
    window.show() # Windows are hidden by default
    signal.signal(signal.SIGINT, signal.SIG_DFL) # this lets control-C actually close the app
    app.exec() # Start the event loop

    if sdr_type == "usrp":
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        streamer.issue_stream_cmd(stream_cmd)
