.. _pyqt-chapter:

##########################
Interfaces Homme Machine temps-réel avec PyQt
##########################

Dans ce chapitre, nous apprenons à créer des interfaces graphiques utilisateur (GUI) en temps réel avec Python grâce à PyQt, l'interface Python pour Qt. Nous y construirons un analyseur de spectre avec affichage du temps, de la fréquence et d'un spectrogramme/diagramme en cascade, ainsi que des widgets de saisie pour ajuster les différents paramètres SDR. Cet exemple est compatible avec PlutoSDR, USRP et le mode simulation uniquement.

****************
Introduction
****************

Qt  (prononcé «  cute  ») est  un framework  permettant  de créer  des applications GUI compatibles avec Linux, Windows, macOS et Android. Ce framework   puissant,   utilisé   dans  de   nombreuses   applications commerciales, est écrit  en C++ pour des  performances optimales. PyQt est  l'interface Python  de Qt,  offrant la  possibilité de  créer des applications GUI en  Python tout en bénéficiant  des performances d'un framework  C++  performant.  Dans  ce  chapitre,  nous  apprendrons  à utiliser  PyQt pour  créer  un  analyseur de  spectre  en temps  réel, utilisable avec un SDR (ou  un signal simulé). Cet analyseur affichera le temps, la fréquence et un spectrogramme/diagramme en cascade, ainsi que des  widgets de saisie  pour ajuster les différents  paramètres du SDR.  Nous utiliserons  `PyQtGraph <https://www.pyqtgraph.org/>`,  une bibliothèque  distincte  basée sur  PyQt,  pour  la visualisation  des données.  Côté  saisie,  nous  utiliserons des  curseurs,  des  listes déroulantes et des boutons. Cet  exemple est compatible avec PlutoSDR, USRP  et le  mode simulation  uniquement. Bien  que le  code d'exemple utilise  PyQt6,  chaque  ligne  est  identique à  celle  de  PyQt5  (à l'exception  de :code:`import`),  les  différences entre  les deux  versions étant minimes du point de vue de l'API. Ce chapitre fait naturellement la  part  belle  au  code  Python, comme  nous  l'illustrons  par  des exemples. À  la fin de ce  chapitre, vous maîtriserez les  éléments de base  nécessaires  à  la  création de  votre  propre  application  SDR
interactive personnalisée !


****************
Aperçu de Qt 
****************

Qt est un framework très complet, et nous n'aborderons ici que quelques notions de base. Cependant, il est important de comprendre certains concepts clés pour travailler avec Qt/PyQt :

- **Widgets** : Les widgets sont les éléments constitutifs d'une application Qt et servent à créer l'interface graphique. Il existe différents types de widgets, comme les boutons, les curseurs, les étiquettes et les graphiques. Les widgets peuvent être organisés en mises en page, qui déterminent leur position à l'écran.

- **Mises en page** : Les mises en page permettent d'organiser les widgets dans une fenêtre. Il existe plusieurs types de mises en page, notamment horizontales, verticales, en grille et en formulaire. Les mises en page permettent de créer des interfaces graphiques complexes qui s'adaptent aux changements de taille de la fenêtre.

- **Signaux et slots** : Les signaux et les slots permettent la communication entre les différentes parties d'une application Qt. Un signal est émis par un objet lorsqu'un événement particulier se produit et est associé à un slot, une fonction de rappel appelée lors de l'émission du signal. Les signaux et les slots permettent de créer une structure événementielle dans une application Qt et de garantir la réactivité de l'interface graphique.

- **Feuilles de style** : Les feuilles de style servent à personnaliser l'apparence des widgets dans une application Qt. Écrites dans un langage similaire à CSS, elles permettent de modifier la couleur, la police et la taille des widgets.

- **Graphismes** : Qt dispose d'un puissant framework graphique permettant de créer des graphismes personnalisés dans une application Qt. Ce framework inclut des classes pour dessiner des lignes, des rectangles, des ellipses et du texte, ainsi que des classes pour gérer les événements de la souris et du clavier.

- **Multithreading** : Qt prend en charge nativement le multithreading et fournit des classes pour créer des threads de travail s'exécutant en arrière-plan. Le multithreading permet d'exécuter des opérations longues dans une application Qt sans bloquer le thread principal de l'interface graphique.

- **OpenGL** : Qt  intègre la prise en charge d’OpenGL  et fournit des classes pour la  création de graphismes 3D dans  une application Qt. OpenGL est utilisé pour créer des applications exigeant des performances  graphiques 3D  élevées.  Dans ce  chapitre, nous  nous concentrerons uniquement sur les applications 2D.
  

*************************
Structure de base d'une application
*************************

Avant d'explorer les différents widgets Qt, examinons la structure d'une application Qt typique. Une application Qt se compose d'une fenêtre principale contenant un widget central, lequel contient le contenu principal de l'application. Avec PyQt, nous pouvons créer une application Qt minimale, ne contenant qu'un seul QPushButton, comme suit :


.. code-block:: python

    from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton

    # Sous-classe QMainWindow pour paramétrer la fenêtre principale de
    l'application
    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            
            # Example de composant IHM
            example_button = QPushButton('Push Me')
            def on_button_click():
                print("beep")
            example_button.clicked.connect(on_button_click)

            self.setCentralWidget(example_button)

    app = QApplication([])
    window = MainWindow()
    window.show() # les fenêtres sont cachées par défaut
    app.exec() # Démarrage de la boucle d'événements

Essayez d'exécuter le code vous-même ; vous devrez probablement installer PyQt6 avec :code:`pip install PyQt6`. Notez que la dernière ligne est bloquante : tout ce que vous ajouterez après ne s'exécutera pas tant que vous n'aurez pas fermé la fenêtre. Le bouton QPushButton que nous créons a son signal :code:`clicked` connecté à une fonction de rappel qui affiche « beep » dans la console.


*******************************
Application avec thread de worker
*******************************

L'exemple minimal présenté ci-dessus pose problème : il ne laisse aucune place pour le code SDR/DSP. La méthode :code:`__init__` de la classe :code:`MainWindow` est configurée et les fonctions de rappel sont définies, mais il est absolument impératif de ne pas y ajouter d'autre code (SDR ou DSP, par exemple). En effet, l'interface graphique étant monothread, bloquer ce thread avec du code long entraînerait des blocages ou des saccades, or nous recherchons une interface aussi fluide que possible. Pour contourner ce problème, nous pouvons utiliser un thread de travail pour exécuter le code SDR/DSP en arrière-plan.

L'exemple ci-dessous étend l'exemple  minimal précédent en incluant un thread de worker qui exécute du code (dans la fonction :code:`run`) en continu.  Nous  n'utilisons pas de  boucle :code:`while True`,  car le fonctionnement interne de PyQt exige que la fonction :code:`run` se termine et redémarre périodiquement.  Pour ce faire, le signal :code:`end_of_run` du thread de worker (que nous détaillerons dans la section  suivante) est  associé  à une  fonction  de rappel  qui relance la fonction  :code:`run` de ce même thread. Il est également nécessaire d'initialiser  le thread  de worker  dans le  code de :code:`MainWindow`,  ce qui  implique la  création d'un  nouveau :code:`QThread` et l'affectation de notre thread de worker personnalisé. Ce  code peut paraître  complexe, mais il s'agit d'une pratique courante dans les applications PyQt. L'essentiel à retenir
est  que   le  code  orienté   interface  graphique  se   trouve  dans :code:`MainWindow`, tandis  que le  code orienté SDR/DSP  se trouve  dans la fonction :code:`run` du thread de travail.

.. code-block:: python

    from PyQt6.QtCore import QThread, pyqtSignal, QObject, QTimer
    from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton
    import time

    # opérations Non-IHM  (notamment SDR) néccessitant d'être lancées dans un thread spéaré.
    class SDRWorker(QObject):
        end_of_run = pyqtSignal()

        # Boucle principale
        def run(self):
            print("Starting run()")
            time.sleep(1)
            self.end_of_run.emit() # let MainWindow know we're done

    # Sous-classe QMainWindow pour personnaliser la fenêtre principale de votre application
    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()

            # Initialisation du worker et du thread
            self.sdr_thread = QThread()
            worker = SDRWorker()
            worker.moveToThread(self.sdr_thread)
            
            # Exemple de composant IHM
            example_button = QPushButton('Push Me')
            def on_button_click():
                print("beep")
            example_button.clicked.connect(on_button_click)
            self.setCentralWidget(example_button)

            # C'est ce qui permet à la fonction run() de se répéter en continu
            def end_of_run_callback():
                QTimer.singleShot(0, worker.run) # Run worker again immediately
            worker.end_of_run.connect(end_of_run_callback)

            self.sdr_thread.started.connect(worker.run) # kicks off the first run() when the thread starts
            self.sdr_thread.start() # start thread

    app = QApplication([])
    window = MainWindow()
    window.show() # Les fenêtres sont cachées par défaut
    app.exec() # Démarrer l'évenèment boucle

Essayez d'exécuter le code ci-dessus ; vous devriez voir « Starting run()» s'afficher dans la console toutes les secondes, et le bouton-poussoir devrait toujours fonctionner (sans délai). Dans le thread de travail, nous effectuons pour l'instant uniquement un affichage et une pause, mais nous y ajouterons prochainement la gestion du signal SDR et le code de traitement du signal numérique.

*************************
Signaux et slots
*************************

Dans   l'exemple    précédent,   nous   avons   utilisé    le   signal :code:`end_of_run` pour la communication entre le thread de travail et le thread d'interface graphique. Ce modèle, courant dans les applications PyQt, est connu  sous le nom  de mécanisme «  signaux et emplacements  ». Un signal  est émis  par un  objet  (ici, le  thread de  travail) et  est associé à un slot (/NDLR : emplacement en français/) (ici, la fonction de rappel :code:`end_of_run_callback` du thread d'interface graphique). Un signal peut être  associé à  plusieurs  slots,  et un  slot  peut  être associé  à plusieurs signaux. Le signal peut également transporter des arguments, qui  sont transmis  à l'emplacement  lors de  son émission.  Notez que l'opération  est réversible  :  le thread  d'interface graphique  peut envoyer un signal  à l'emplacement du thread de  travail. Le mécanisme de signaux/emplacements est un moyen puissant de communiquer entre les différentes  parties  d'une  application PyQt,  créant  une  structure événementielle.  Il  est  largement  utilisé dans  l'exemple  de  code suivant.  Retenez simplement qu'un slot est une fonction de rappel, et qu'un signal est un moyen de signaler cette fonction de rappel.
  

*************************
PyQtGraph
*************************

PyQtGraph est une bibliothèque basée sur PyQt et NumPy qui offre des capacités de traçage rapides et efficaces, PyQt étant trop généraliste pour intégrer des fonctionnalités de traçage. Conçue pour les applications temps réel, elle est optimisée pour la vitesse. Elle est similaire à Matplotlib à bien des égards, mais destinée aux applications temps réel plutôt qu'aux graphiques individuels. L'exemple simple ci-dessous permet de comparer les performances de PyQtGraph et de Matplotlib : il suffit de remplacer :code:`if True` par :code:`False`. Sur un processeur Intel Core i9-10900K à 3,70 GHz, le code PyQtGraph s'est mis à jour à plus de 1 000 images par seconde, tandis que le code Matplotlib s'est mis à jour à 40 images par seconde. Cela étant dit, si vous constatez que l'utilisation de Matplotlib vous est utile (par exemple, pour gagner du temps de développement ou parce que vous souhaitez une fonctionnalité spécifique que PyQtGraph ne prend pas en charge), vous pouvez intégrer des graphiques Matplotlib dans une application PyQt, en utilisant le code ci-dessous comme point de départ.

.. raw:: html

   <details>
   <summary>Développez pour afficher le code</summary>

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

                # Configurez une minuterie pour déclencher le redessin en appelant update_plot
                self.timer = QtCore.QTimer()
                self.timer.setInterval(0) # provoque le démarrage immédiat du minuteur
                self.timer.timeout.connect(self.update_plot) # provoque le redémarrage automatique du minuteur
                self.timer.start()
                self.start_t = time.time() # utilisé pour l'analyse comparative

                self.show()

            def update_plot(self):
                self._plot_ref.set_ydata(np.random.randn(n_data))
                self.canvas.draw() # Déclenchez la mise à jour et le redessin du canevas.
                print('FPS:', 1/(time.time()-self.start_t)) # on a obtenu environ 42 FPS sur un i9-10900K
                self.start_t = time.time()

    else:
        class MainWindow(QtWidgets.QMainWindow):
            def __init__(self):
                super(MainWindow, self).__init__()
                
                self.time_plot = pg.PlotWidget()
                self.time_plot.setYRange(-5, 5)
                self.time_plot_curve = self.time_plot.plot([])
                self.setCentralWidget(self.time_plot)

                # Configurez une minuterie pour déclencher le redessin en appelant update_plot.
                self.timer = QtCore.QTimer()
                self.timer.setInterval(0) # provoque le démarrage immédiat du timer
                self.timer.timeout.connect(self.update_plot) # provoque le redémarrage automatique du minuteur
                self.timer.start()
                self.start_t = time.time() # utilisé pour l'évaluation des performances.

                self.show()

            def update_plot(self):
                self.time_plot_curve.setData(np.random.randn(n_data))
                print('FPS:', 1/(time.time()-self.start_t)) # on a obtenu environ 42 FPS sur un i9-10900K
                self.start_t = time.time()

    app = QtWidgets.QApplication([])
    w = MainWindow()
    app.exec()

.. raw:: html

    </details>

Pour  ce   qui  est   d'utiliser  PyQtGraph,  nous   l'importons  avec :code:`import pyqtgraph as pg` et nous pouvons ensuite créer un widget Qt qui représente un graphique 1D comme suit (ce code va dans la méthode :code:`__init__` de :code:`MainWindow`).

.. code-block:: python

        # Exemple de graphique PyQtGraph
        time_plot = pg.PlotWidget(labels={'left': 'Amplitude', 'bottom': 'Time'})
        time_plot_curve        =       time_plot.plot(np.arange(1000),
        np.random.randn(1000)) # x et y
        time_plot.setYRange(-5, 5)

        self.setCentralWidget(time_plot)

.. image:: ../_images/pyqtgraph_example.png
   :scale: 80 % 
   :align: center
   :alt: PyQtGraph exemple


Vous pouvez constater qu'il est relativement simple de configurer un graphique, et le résultat est simplement un widget supplémentaire à ajouter à votre interface graphique. Outre les graphiques 1D, PyQtGraph possède également un équivalent de la fonction :code:`imshow()` de Matplotlib, qui permet de tracer des graphiques 2D à l'aide d'une palette de couleurs, que nous utiliserons pour notre spectrogramme/waterfall en temps réel. L'un des avantages de PyQtGraph est que les graphiques qu'il crée sont de simples widgets Qt, et que nous ajoutons d'autres éléments Qt (par exemple, un rectangle d'une certaine taille à une certaine coordonnée) en utilisant uniquement PyQt. En effet, PyQtGraph utilise la classe :code:`QGraphicsScene` de PyQt, qui fournit une interface pour gérer un grand nombre d'éléments graphiques 2D. Rien ne nous empêche donc d'ajouter des lignes, des rectangles, du texte, des ellipses, des polygones et des bitmaps, directement en utilisant PyQt.

*******
Dispositions
*******

Dans les exemples précédents, nous avons utilisé :code:`self.setCentralWidget()` pour définir le widget principal de la fenêtre. Cette méthode simple ne permet pas de créer des dispositions plus complexes. Pour cela, nous pouvons utiliser des dispositions, qui permettent d'organiser les widgets dans une fenêtre. Il existe plusieurs types de dispositions, notamment :code:`QHBoxLayout`, :code:`QVBoxLayout`, :code:`QGridLayout` et :code:`QFormLayout`. :code:`QHBoxLayout` et :code:`QVBoxLayout` disposent les widgets horizontalement et verticalement, respectivement. :code:`QGridLayout` les dispose sous forme de grille, et :code:`QFormLayout` les dispose sur deux colonnes : la première colonne contient les étiquettes et la seconde, les champs de saisie.

Pour créer une nouvelle mise en page et y ajouter des widgets, essayez d'ajouter ce qui suit dans la méthode :code:`__init__` de votre :code:`MainWindow` :

.. code-block:: python

    layout = QHBoxLayout()
    layout.addWidget(QPushButton("Left-Most"))
    layout.addWidget(QPushButton("Center"), 1)
    layout.addWidget(QPushButton("Right-Most"), 2)
    self.setLayout(layout)

    
Dans cet exemple, les widgets sont empilés horizontalement. Cependant, en remplaçant :code:`QHBoxLayout` par :code:`QVBoxLayout`, il est possible de les empiler verticalement. La fonction :code:`addWidget` permet d'ajouter des widgets à la mise en page. Son deuxième argument, optionnel, est un facteur d'étirement qui détermine l'espace occupé par le widget par rapport aux autres.

:code:`QGridLayout` possède des paramètres supplémentaires : il est nécessaire de spécifier la ligne et la colonne du widget, ainsi que le nombre de lignes et de colonnes qu'il doit occuper (par défaut : 1 et 1). Voici un exemple de :code:`QGridLayout` :

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
   :alt: Agencements  Qt  illustrant   des  exemples  de  QHBoxLayout, QVBoxLayout et QGridLayout

Pour notre analyseur de spectre, nous utiliserons :code:`QGridLayout` pour la mise en page générale, mais nous ajouterons également :code:`QHBoxLayout` pour empiler les widgets horizontalement dans un espace de la grille. Vous pouvez imbriquer des mises en page simplement en créant une nouvelle mise en page et en l'ajoutant à la mise en page de niveau supérieur (ou parente), par exemple :

.. code-block:: python

    layout = QGridLayout()
    self.setLayout(layout)
    inner_layout = QHBoxLayout()
    layout.addLayout(inner_layout)


*******************
:code:`QPushButton`
*******************

Le premier widget que nous allons aborder est le :code:`QPushButton`, un simple bouton cliquable. Nous avons déjà vu comment créer un :code:`QPushButton` et associer son signal :code:`clicked` à une fonction de rappel. Le :code:`QPushButton` possède d'autres signaux, notamment :code:`pressed`, :code:`released` et :code:`toggled`. Le signal :code:`toggled` est émis lorsque le bouton est activé ou désactivé, et est utile pour créer des boutons à bascule. Le :code:`QPushButton` possède également plusieurs propriétés, dont :code:`text`, :code:`icon` et :code:`checkable`. Enfin, le :code:`QPushButton` possède une méthode appelée :code:`click()` qui simule un clic sur le bouton. Pour notre application d'analyseur de spectre SDR, nous utiliserons des boutons pour déclencher un réglage automatique de la plage des graphiques, en utilisant les données actuelles pour calculer les limites de l'axe des y. Comme nous avons déjà utilisé le composant :code:`QPushButton`, nous n'entrerons pas dans les détails ici. Vous trouverez plus d'informations dans la `documentation de QPushButton  : <https://doc.qt.io/qtforpython/PySide6/QtWidgets/QPushButton.html>`_.


***************
:code:`QSlider`
***************
Le :code:`QSlider` est un widget qui permet à l'utilisateur de sélectionner une valeur dans une plage de valeurs. Le :code:`QSlider` possède plusieurs propriétés, notamment :code:`minimum`, :code:`maximum`, :code:`value` et :code:`orientation`. Le composant :code:`QSlider` possède également plusieurs signaux, notamment :code:`valueChanged`, :code:`sliderPressed` et :code:`sliderReleased`. Il dispose aussi d'une méthode :code:`setValue()` qui permet de définir la valeur du curseur ; nous l'utiliserons fréquemment. La documentation de :code:`QSlider` est disponible ici : `<https://doc.qt.io/qtforpython/PySide6/QtWidgets/QSlider.html>`_.


Pour notre application d'analyseur de spectre, nous utiliserons des curseurs QSlider pour ajuster la fréquence centrale et le gain du récepteur SDR. Voici un extrait du code final de l'application qui crée le curseur de gain :

.. code-block:: python

    # Slider de gain avec étiquette
    gain_slider = QSlider(Qt.Orientation.Horizontal)
    gain_slider.setRange(0, 73) # min et max inclus. L'intervalle est toujours de 1
    gain_slider.setValue(50) # valeur initiale
    gain_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
    gain_slider.setTickInterval(2) # à des fins visuelles uniquement
    gain_slider.sliderMoved.connect(worker.update_gain)
    gain_label = QLabel()
    def update_gain_label(val):
        gain_label.setText("Gain: " + str(val))
    gain_slider.sliderMoved.connect(update_gain_label)
    update_gain_label(gain_slider.value()) # initialisation du label
    layout.addWidget(gain_slider, 5, 0)
    layout.addWidget(gain_label, 5, 1)


Il est très important de savoir que :code:`QSlider` utilise des entiers. En définissant la plage de 0 à 73, on permet au curseur de choisir des valeurs entières comprises entre ces nombres (début et fin inclus). La fonction :code:`setTickInterval(2)` est purement visuelle. C'est pourquoi nous utiliserons le kHz comme unité pour le curseur de fréquence, afin d'obtenir une granularité jusqu'à 1 kHz.

Au milieu du code ci-dessus, vous remarquerez la création d'un :code:`QLabel`, une simple étiquette de texte. Pour afficher la valeur actuelle du curseur, nous devons créer un slot (c'est-à-dire une fonction de rappel) qui met à jour l'étiquette. Nous connectons cette fonction de rappel au signal :code:`sliderMoved`, émis automatiquement à chaque déplacement du curseur. Nous appelons également cette fonction une première fois pour initialiser l'étiquette avec la valeur actuelle du curseur (50 dans notre cas). Il faut également connecter le signal :code:`sliderMoved` à un slot situé dans le thread de travail, qui mettra à jour le gain du SDR (rappelons que nous préférons ne pas gérer le SDR ni effectuer de traitement du signal numérique dans le thread principal de l'interface graphique). La fonction de rappel définissant ce slot sera abordée ultérieurement.


*****************
:code:`QComboBox`
*****************
Le :code:`QComboBox` est un widget de type liste déroulante permettant à l'utilisateur de sélectionner un élément dans une liste. Il possède plusieurs propriétés, notamment :code:`currentText`, :code:`currentIndex` et :code:`count`. Il dispose également de signaux tels que :code:`currentTextChanged`, :code:`currentIndexChanged` et :code:`activated`. Enfin, il possède une méthode :code:`addItem()` pour ajouter un élément à la liste et une méthode :code:`insertItem()` pour insérer un élément à un index spécifique, bien que nous ne les utilisions pas dans notre exemple d'analyseur de spectre. La documentation de :code:`QComboBox` est disponible ici : `<https://doc.qt.io/qtforpython/PySide6/QtWidgets/QComboBox.html>`_.

Pour notre application d'analyseur de spectre, nous utiliserons un :code:`QComboBox` afin de sélectionner la fréquence d'échantillonnage dans une liste prédéfinie. Au début de notre code, nous définissons les fréquences d'échantillonnage possibles avec :code:`sample_rates = [56, 40, 20, 10, 5, 2, 1, 0.5]`. Dans la méthode :code:`__init__` de la fenêtre principale, nous créons le :code:`QComboBox` comme suit :

.. code-block:: python

    # Liste déroulante de fréquence d'échantillonnage utilisant QComboBox
    sample_rate_combobox = QComboBox()
    sample_rate_combobox.addItems([str(x) + ' MHz' for x in sample_rates])
    sample_rate_combobox.setCurrentIndex(0) # Il faut lui fournir l'index, et non une chaîne de caractères.
    sample_rate_combobox.currentIndexChanged.connect(worker.update_sample_rate)
    sample_rate_label = QLabel()
    def update_sample_rate_label(val):
        sample_rate_label.setText("Sample Rate: " + str(sample_rates[val]) + " MHz")
    sample_rate_combobox.currentIndexChanged.connect(update_sample_rate_label)
    update_sample_rate_label(sample_rate_combobox.currentIndex()) # initialisation du label
    layout.addWidget(sample_rate_combobox, 6, 0)
    layout.addWidget(sample_rate_label, 6, 1)


La seule véritable différence entre ceci et le curseur est le :code:`addItems()` où vous lui donnez la liste des chaînes à utiliser comme options, et :code:`setCurrentIndex()` qui définit la valeur de départ.
    

****************
Lonctions lambda
****************

Rappelez-vous dans le code ci-dessus où nous avons fait :

.. code-block:: python

    def update_sample_rate_label(val):
        sample_rate_label.setText("Sample Rate: " + str(sample_rates[val]) + " MHz")
    sample_rate_combobox.currentIndexChanged.connect(update_sample_rate_label)


Nous créons une fonction ne contenant qu'une seule ligne de code, puis nous passons cette fonction (les fonctions sont aussi des objets !) à :code:`connect()`. Pour simplifier, réécrivons ce modèle de code en utilisant du Python de base :

.. code-block:: python

    def my_function(x):
        print(x)
    y.call_that_takes_in_function_obj(my_function)

Dans ce cas précis, nous avons une fonction ne contenant qu'une seule ligne de code, et nous n'y faisons référence qu'une seule fois : lors de la définition du rappel :code:`connect`. Dans ce genre de situation, nous pouvons utiliser une fonction lambda, qui permet de définir une fonction sur une seule ligne. Voici le code ci-dessus réécrit à l'aide d'une fonction lambda :

.. code-block:: python

    y.call_that_takes_in_function_obj(lambda x: print(x))

Si vous n'avez jamais utilisé de fonction lambda, cela peut paraître étrange, et vous n'êtes d'ailleurs pas obligé de les utiliser, mais cela permet de gagner deux lignes de code et de le rendre plus concis. Le principe est le suivant : le nom de l'argument temporaire est indiqué après « lambda », et tout ce qui suit les deux-points correspond au code qui agira sur cette variable. Il est possible d'utiliser plusieurs arguments, séparés par des virgules, ou même aucun argument avec : :code:`lambda : <code>`. À titre d'exercice, essayez de réécrire la fonction :code:`update_sample_rate_label` ci-dessus en utilisant une fonction lambda.


***********************
Le widget de tracé de PyQtGraph
***********************

Le widget :code:`PlotWidget` de PyQtGraph permet de générer des graphiques 1D, à l'instar de :code:`plt.plot(x,y)` de Matplotlib. Nous l'utiliserons pour les graphiques dans le domaine temporel et fréquentiel (PSD), bien qu'il convienne également aux graphiques IQ (que notre analyseur de spectre ne prend pas en charge). Pour les curieux, PlotWidget est une sous-classe de `QGraphicsView <https://doc.qt.io/qtforpython-5/PySide2/QtWidgets/QGraphicsView.html>`_ de PyQt, qui est un widget permettant d'afficher le contenu d'une `QGraphicsScene <https://doc.qt.io/qtforpython-5/PySide2/QtWidgets/QGraphicsScene.html#PySide2.QtWidgets.PySide2.QtWidgets.QGraphicsScene>`_, qui est une surface permettant de gérer un grand nombre d'éléments graphiques 2D dans Qt. L'important à retenir concernant PlotWidget est qu'il s'agit simplement d'un widget contenant un unique `PlotItem <https://pyqtgraph.readthedocs.io/en/latest/api_reference/graphicsItems/plotitem.html#pyqtgraph.PlotItem>`_. Du point de vue de la documentation, il est donc préférable de se référer directement à la documentation de PlotItem : `<https://pyqtgraph.readthedocs.io/en/latest/api_reference/graphicsItems/plotitem.html>`_. Un PlotItem contient une ViewBox pour afficher les données à représenter graphiquement, ainsi que des AxisItems et des labels pour afficher les axes et le titre, comme on peut s'y attendre.

Voici un exemple simple d'utilisation d'un PlotWidget (à ajouter dans la méthode :code:`__init__` de :code:`MainWindow`) :


.. code-block:: python

 import pyqtgraph as pg
 plotWidget = pg.plot(title="My Title")
 plotWidget.plot(x, y)

où x et y sont généralement des tableaux NumPy, comme avec la fonction :code:`plt.plot()` de Matplotlib. Cependant, cela représente un graphique statique où les données ne changent jamais. Pour notre analyseur de spectre, nous souhaitons mettre à jour les données dans notre thread de travail. Par conséquent, lors de l'initialisation du graphique, nous n'avons même pas besoin de lui fournir de données ; il suffit de le configurer. Voici comment nous initialisons le graphique temporel dans notre application d'analyseur de spectre :

.. code-block:: python

    # Time plot
    time_plot = pg.PlotWidget(labels={'left': 'Amplitude', 'bottom': 'Time [microseconds]'})
    time_plot.setMouseEnabled(x=False, y=True)
    time_plot.setYRange(-1.1, 1.1)
    time_plot_curve_i = time_plot.plot([]) 
    time_plot_curve_q = time_plot.plot([]) 
    layout.addWidget(time_plot, 1, 0)

Vous pouvez constater que nous créons deux graphiques/courbes différents, un pour I et un pour Q. Le reste du code devrait être explicite. Pour pouvoir mettre à jour le graphique, nous devons créer un emplacement (c'est-à-dire une fonction de rappel) dans la méthode :code:`__init__` de la fenêtre principale.

.. code-block:: python

    def time_plot_callback(samples):
        time_plot_curve_i.setData(samples.real)
        time_plot_curve_q.setData(samples.imag)


Nous connecterons ce slot au signal du thread de travail émis lors de la disponibilité de nouveaux échantillons, comme indiqué plus loin.

La dernière étape dans la méthode :code:`__init__` de :code:`MainWindow` consiste à ajouter deux boutons à droite du graphique. Ces boutons activeront un réglage automatique de la plage. L'un utilisera les valeurs minimales et maximales actuelles, tandis que l'autre définira la plage entre -1,1 et 1,1 (correspondant aux limites de conversion analogique-numérique de nombreux SDR, plus une marge de 10 %). Nous créerons une mise en page interne, plus précisément un :code:`QVBoxLayout`, pour empiler verticalement ces deux boutons. Voici le code permettant d'ajouter les boutons :


.. code-block:: python

    # Boutons de plage automatique du graphique temporel
    time_plot_auto_range_layout = QVBoxLayout()
    layout.addLayout(time_plot_auto_range_layout, 1, 1)
    auto_range_button = QPushButton('Auto Range')
    auto_range_button.clicked.connect(lambda : time_plot.autoRange()) # lambda signifie simplement qu'il s'agit d'une fonction sans nom
    time_plot_auto_range_layout.addWidget(auto_range_button)
    auto_range_button2 = QPushButton('-1 to +1\n(ADC limits)')
    auto_range_button2.clicked.connect(lambda : time_plot.setYRange(-1.1, 1.1))
    time_plot_auto_range_layout.addWidget(auto_range_button2)

Et voici à quoi cela ressemble au final :

.. image:: ../_images/pyqt_time_plot.png
   :scale: 50 % 
   :align: center
   :alt: Graphique temporel PyQtGraph

Nous utiliserons un modèle similaire pour le graphique du domaine fréquentiel (PSD).


*********************
ImageItem de PyQtGraph
*********************

Un analyseur de spectre se doit d'afficher un spectrogramme en cascade (ou spectrogramme en temps réel). Pour cela, nous utiliserons l'objet ImageItem de PyQtGraph, qui génère des images à 1, 3 ou 4 canaux. Un canal correspond à un tableau 2D de nombres flottants ou entiers, qui utilise ensuite une table de correspondance (LUT) pour appliquer une palette de couleurs et créer l'image. On peut également utiliser les formats RGB (3 canaux) ou RGBA (4 canaux). Nous calculerons notre spectrogramme sous forme d'un tableau NumPy 2D de nombres flottants et le transmettrons directement à l'objet ImageItem. Nous choisirons une palette de couleurs et exploiterons la fonctionnalité intégrée d'affichage d'une LUT graphique permettant de visualiser la distribution des valeurs de nos données et l'application de la palette.

L'initialisation du spectrogramme watefall est assez simple : nous utilisons un PlotWidget comme conteneur (afin de conserver l'affichage des axes x et y) et y ajoutons un ImageItem.

.. code-block:: python

    # Waterfall plot
    waterfall = pg.PlotWidget(labels={'left': 'Time [s]', 'bottom': 'Frequency [MHz]'})
    imageitem = pg.ImageItem(axisOrder='col-major') # cet argument est simplement pour la performance
    waterfall.addItem(imageitem)
    waterfall.setMouseEnabled(x=False, y=False)
    waterfall_layout.addWidget(waterfall)

Le slot/callback associé à la mise à jour des données en cascade, qui se trouve dans :code:`MainWindow`'s :code:`__init__`, est le suivant :

.. code-block:: python

    def waterfall_plot_callback(spectrogram):
        imageitem.setImage(spectrogram, autoLevels=False)
        sigma = np.std(spectrogram)
        mean = np.mean(spectrogram) 
        self.spectrogram_min = mean - 2*sigma # save to window state
        self.spectrogram_max = mean + 2*sigma

Le spectrogramme sera un tableau NumPy 2D de nombres flottants. Outre la définition des données de l'image, nous calculerons les valeurs minimale et maximale de la palette de couleurs, en fonction de la moyenne et de la variance des données, que nous utiliserons ultérieurement. La dernière partie du code de l'interface graphique du spectrogramme consiste à créer la barre de couleurs, qui définit également la palette de couleurs utilisée.

.. code-block:: python

    # Colorbar for waterfall
    colorbar = pg.HistogramLUTWidget()
    colorbar.setImageItem(imageitem) # Connecte la barre à l'élément image du spectrogramme
    colorbar.item.gradient.loadPreset('viridis') #  définit la palette de couleurs, et définit également l'élément image
    imageitem.setLevels((-30, 20)) # doit être placé après la création de la barre de couleur (pour une raison inconnue)
    waterfall_layout.addWidget(colorbar)

La deuxième ligne est importante ; c’est elle qui relie la barre de couleurs à l’élément ImageItem. C’est également dans ce code que l’on choisit la palette de couleurs et que l’on définit les niveaux de départ (de -30 dB à +20 dB dans notre cas). Le code du thread de travail illustre le calcul et le stockage du tableau 2D du spectrogramme. Ci-dessous, une capture d’écran de cette partie de l’interface graphique montre l’incroyable fonctionnalité intégrée de la barre de couleurs et de l’affichage de la LUT. Notez que la courbe en cloche horizontale représente la distribution des valeurs du spectrogramme, une information très utile.

.. image:: ../_images/pyqt_spectrogram.png
   :scale: 50 % 
   :align: center
   :alt:  Spectrogramme et colorbar PyQtGraph

***********************
Worker Thread
***********************

Rappelez-vous, au début de ce chapitre, nous avons appris à créer un thread séparé à l'aide d'une classe nommée SDRWorker et de sa fonction run(). C'est dans ce thread que nous placerons tout notre code SDR et DSP, à l'exception de l'initialisation du SDR, que nous effectuerons globalement pour le moment. Ce thread de travail sera également chargé de mettre à jour les trois graphiques en émettant des signaux lorsque de nouveaux échantillons sont disponibles, afin de déclencher les fonctions de rappel que nous avons déjà créées dans :code:`MainWindow`, qui mettent finalement à jour les graphiques. La classe SDRWorker se divise en trois sections :

#. :code:`init()` -  utilisée pour initialiser un état, comme le tableau 2D du spectrogramme.
#. PyQt Signals - nous devons définir les signaux personnalisés qui seront émis
#. PyQt Slots - les fonctions de rappel déclenchées par des événements d'interface graphique, comme le déplacement d'un curseur
#. :code:`run()` - la boucle principale qui s'exécute en continu

***********************
Signaux PyQt
***********************

Dans le code de l'interface graphique, nous n'avions pas besoin de définir de signaux, car ils étaient intégrés aux widgets utilisés, comme le signal :code:`valueChanged` de :code:`QSlider`. Notre classe :code:`SDRWorker` est personnalisée, et tous les signaux que nous souhaitons émettre doivent être définis avant d'appeler :code:`run()`. Voici le code de la classe :code:`SDRWorker`, qui définit quatre signaux que nous utiliserons, ainsi que leurs types de données correspondants :

.. code-block:: python

    # Signaux PyQt
    time_plot_update = pyqtSignal(np.ndarray)
    freq_plot_update = pyqtSignal(np.ndarray)
    waterfall_plot_update = pyqtSignal(np.ndarray)
    end_of_run = pyqtSignal() # se produit plusieurs fois par seconde

Les trois premiers signaux envoient un seul objet : un tableau NumPy. Le dernier signal n'envoie aucun objet. Il est également possible d'envoyer plusieurs objets simultanément, en séparant les types de données par des virgules, mais cela n'est pas nécessaire pour notre application. À n'importe quel endroit de la fonction :code:`run()`, nous pouvons émettre un signal vers le thread d'interface graphique en une seule ligne de code, par exemple :

.. code-block:: python

    self.time_plot_update.emit(samples)

Il reste une dernière étape pour établir toutes les connexions signaux/slots : dans le code de l’interface graphique (qui se trouve à la toute fin de la méthode :code:`__init__` de :code:`MainWindow`), nous devons connecter les signaux du thread de travail aux slots de l’interface graphique, par exemple :

.. code-block:: python

    worker.time_plot_update.connect(time_plot_callback) # connection du signal à la fonction d'appel (callback)

Rappelez-vous que :code:`worker` est l'instance de la classe :code:`SDRWorker` créée dans le code de l'interface graphique. Nous connectons ici le signal du thread de travail, :code:`time_plot_update`, à l'emplacement de l'interface graphique, :code:`time_plot_callback`, défini précédemment. Revoyez les extraits de code présentés jusqu'ici et observez leur fonctionnement. Cela vous permettra de bien comprendre la communication entre l'interface graphique et le thread de travail, un aspect fondamental de la programmation PyQt.


***********************
Slots des Worker Threads
***********************

Les slots des worker threads sont les fonctions de rappel déclenchées par des événements d'interface graphique, comme le déplacement du curseur de gain. Leur fonctionnement est assez simple ; par exemple, cet emplacement met à jour la valeur de gain du SDR avec la nouvelle valeur sélectionnée par le curseur :

.. code-block:: python

    def update_gain(self, val):
        print("Updated gain to:", val, 'dB')
        sdr.set_rx_gain(val)

***********************
Worker Thread Run()
***********************

La fonction :code:`run()` est l'endroit où se déroule toute la partie DSP intéressante ! Dans notre application, chaque fonction :code:`run()` commencera par la réception d'un ensemble d'échantillons provenant du SDR (ou par la simulation d'échantillons si vous n'avez pas de SDR).

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

Comme vous pouvez le constater, pour l'exemple simulé, nous générons une tonalité avec du bruit blanc, puis nous tronquons les échantillons de -1 à +1.

Passons maintenant au traitement numérique du signal (DSP) ! Nous savons qu'il nous faudra effectuer la transformée de Fourier rapide (FFT) pour obtenir le graphique dans le domaine fréquentiel et le spectrogramme. Il s'avère que nous pouvons simplement utiliser la densité spectrale de puissance (DSP) de cet ensemble d'échantillons comme une ligne du spectrogramme. Il nous suffit donc de décaler notre spectrogramme/diagramme en cascade d'une ligne vers le haut et d'ajouter cette nouvelle ligne en bas (ou en haut, peu importe). À chaque mise à jour du graphique, nous émettons le signal contenant les données mises à jour. Nous signalons également la fin de la fonction :code:`run()` afin que le thread de l'interface graphique lance immédiatement un nouvel appel à :code:`run()`. Au final, le code est plutôt court.

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

Notez que nous n'envoyons pas l'intégralité des échantillons au graphique temporel, car cela représenterait un nombre excessif de points. Seuls les 500 premiers échantillons sont envoyés (paramétrable en début de script, non affiché ici). Pour le graphique de la densité spectrale de puissance (DSP), nous utilisons une moyenne mobile de la DSP, obtenue en stockant la DSP précédente et en y ajoutant 1 % de la nouvelle DSP. Cette méthode simple permet de lisser le graphique de la DSP. Notez que l'ordre d'appel de la fonction :code:`emit()` pour les signaux est indifférent ; ils auraient tout aussi bien pu être tous placés à la fin de la fonction :code:`run()`.


***********************
Exemple final : Code complet
***********************

Jusqu’à présent, nous avons examiné des extraits de code de l’application d’analyse de spectre. Nous allons maintenant étudier le code complet et l’exécuter. Il est compatible avec PlutoSDR, USRP et le mode simulation. Si vous ne possédez ni PlutoSDR ni USRP, laissez le code tel quel ; il utilisera alors le mode simulation. Sinon, modifiez :code:`sdr_type`. En mode simulation, si vous augmentez le gain au maximum, vous constaterez que le signal est tronqué dans le domaine temporel, ce qui provoque l’apparition de signaux parasites dans le domaine fréquentiel.

N’hésitez pas  à utiliser  ce code  comme point  de départ  pour votre propre application SDR  en temps réel ! Vous  trouverez ci-dessous une animation  de  l’application en  action,  utilisant  un PlutoSDR  pour analyser la bande cellulaire 750 MHz, puis la bande Wi-Fi 2,4 GHz. Une version de meilleure qualité est disponible sur YouTube ici `here <https://youtu.be/hvofiY3Q_yo>`_.

.. image:: ../_images/pyqt_animation.gif
   :scale: 100 %
   :align: center
   :alt:  gif animé montrant le fonctionnement l'application analyseur de spectre PyQt
  
         
Bogues connus  (pour aider à  les corriger, modifiez ce  fichier `edit
this
<https://github.com/777arc/PySDR/edit/master/figure-generating-scripts/pyqt_example.py>`_)
:

#. L'axe des x du spectrogramme ne se met pas à jour lorsque l'on modifie la fréquence centrale (contrairement au graphique PSD)

Code complet :

.. code-block:: python

    from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal, QObject, QTimer
    from PyQt6.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QSlider, QLabel, QHBoxLayout, QVBoxLayout, QPushButton, QComboBox  # tested with PyQt6==6.7.0
    import pyqtgraph as pg # tested with pyqtgraph==0.13.7
    import numpy as np
    import time
    import signal # lets control-C actually close the app

    # Valeurs par défaut
    fft_size = 4096 # determines buffer size
    num_rows = 200
    center_freq = 750e6
    sample_rates = [56, 40, 20, 10, 5, 2, 1, 0.5] # MHz
    sample_rate = sample_rates[0] * 1e6
    time_plot_samples = 500
    gain = 50 # 0 to 73 dB. int

    sdr_type = "sim" # or "usrp" or "pluto"

    # Initialisation du SDR
    if sdr_type == "pluto":
        import adi
        sdr = adi.Pluto("ip:192.168.1.10")
        sdr.rx_lo = int(center_freq)
        sdr.sample_rate = int(sample_rate)
        sdr.rx_rf_bandwidth = int(sample_rate*0.8) # bande-passante du filtre anti-repliement
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

        # Configuration du flux (stream) et du buiffer de réception
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [0]
        metadata = uhd.types.RXMetadata()
        streamer = usrp.get_rx_stream(st_args)
        recv_buffer = np.zeros((1, fft_size), dtype=np.complex64)

        # Démarrage du flux
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

        # Signaux PyQt
        time_plot_update = pyqtSignal(np.ndarray)
        freq_plot_update = pyqtSignal(np.ndarray)
        waterfall_plot_update = pyqtSignal(np.ndarray)
        end_of_run = pyqtSignal() # happens many times a second

        # Slots PyQt
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

        # Boucle principale
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


    # Sous-classe QMainWindow pour configurer la fenêtre principale de
    la fenêtre application
    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()

            self.setWindowTitle("The PySDR Spectrum Analyzer")
            self.setFixedSize(QSize(1500, 1000)) # window size, starting size should fit on 1920 x 1080

            self.spectrogram_min = 0
            self.spectrogram_max = 0

            layout = QGridLayout() # overall layout

            # Initialisation du worker et du thread
            self.sdr_thread = QThread()
            self.sdr_thread.setObjectName('SDR_Thread') # so we can see it in htop, note you have to hit F2 -> Display options -> Show custom thread names
            worker = SDRWorker()
            worker.moveToThread(self.sdr_thread)

            # Affichage temporel
            time_plot = pg.PlotWidget(labels={'left': 'Amplitude', 'bottom': 'Time [microseconds]'})
            time_plot.setMouseEnabled(x=False, y=True)
            time_plot.setYRange(-1.1, 1.1)
            time_plot_curve_i = time_plot.plot([]) 
            time_plot_curve_q = time_plot.plot([]) 
            layout.addWidget(time_plot, 1, 0)

            # Boutons de plage automatique du graphique temporel
            time_plot_auto_range_layout = QVBoxLayout()
            layout.addLayout(time_plot_auto_range_layout, 1, 1)
            auto_range_button = QPushButton('Auto Range')
            auto_range_button.clicked.connect(lambda : time_plot.autoRange()) # lambda just means its an unnamed function
            time_plot_auto_range_layout.addWidget(auto_range_button)
            auto_range_button2 = QPushButton('-1 to +1\n(ADC limits)')
            auto_range_button2.clicked.connect(lambda : time_plot.setYRange(-1.1, 1.1))
            time_plot_auto_range_layout.addWidget(auto_range_button2)

            # Graohique fréquentiel
            freq_plot = pg.PlotWidget(labels={'left': 'PSD', 'bottom': 'Frequency [MHz]'})
            freq_plot.setMouseEnabled(x=False, y=True)
            freq_plot_curve = freq_plot.plot([]) 
            freq_plot.setXRange(center_freq/1e6 - sample_rate/2e6, center_freq/1e6 + sample_rate/2e6)
            freq_plot.setYRange(-30, 20)
            layout.addWidget(freq_plot, 2, 0)
            
            # Bouton de sélection automatique de la plage de fréquence
            auto_range_button = QPushButton('Auto Range')
            auto_range_button.clicked.connect(lambda : freq_plot.autoRange()) # lambda just means its an unnamed function
            layout.addWidget(auto_range_button, 2, 1)

            # Conteneur pour les éléments liés au flux vidéo
            waterfall_layout = QHBoxLayout()
            layout.addLayout(waterfall_layout, 3, 0)

            # Affichage graphique du spectrogramme
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
