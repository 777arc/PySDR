.. _freq-domain-chapter:

##########################
Real-Time GUIs with PyQt
##########################

In this chapter we learn how to create real-time graphical user interfaces (GUIs) within Python by leveraging PyQt, the Python bindings for Qt.  As part of this chapter we build a spectrum analyzer with time, frequency, and spectrogram/waterfall graphics, as well as input widgets for adjusting the various SDR parameters.  The example supports the PlutoSDR, USRP, or simulation-only mode.

****************
Introduction
****************

Qt (pronounced "cute") is a framework for creating GUI applications that can run on Linux, Windows, macOS, and even Android.  It is a very powerful framework that is used in many commercial applications, and is written in C++ for maximum performance.  PyQt is the Python bindings for Qt, providing a way to create GUI applications in Python, while harnessing the performance of an efficient C++ based framework.  In this chapter we will learn how to use PyQt to create a real-time spectrum analyzer that can be used with an SDR (or with a simulated signal).  The spectrum analyzer will have time, frequency, and spectrogram/waterfall graphics, as well as input widgets for adjusting the various SDR parameters.  We use `PyQtGraph <https://www.pyqtgraph.org/>`_, which is a separate library built on top of PyQt, to perform plotting.  On the input side, we use sliders, combo-box, and push-buttons.  The example supports the PlutoSDR, USRP, or simulation-only mode.  Even though the example code uses PyQt6, every single line is identical to PyQt5 (besides the :code:`import`), very little changed between the two versions from an API perspective.  By the end of this chapter you will have gained familiarity with the building blocks used to create your own custom interactive SDR application!

****************
Qt Overview
****************

*************************
Basic Application Layout
*************************

Before we dive into the different Qt widgets, let's look at the layout of a typical Qt application.  A Qt application is composed of a main window, which contains a central widget, which in turn contains the main content of the application.  Using PyQt we can create a minimal Qt application, containing just a single QPushButton as follows:

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

Try running the code yourself, you will likely need to :code:`pip install PyQt6`.  Note how the very last line is blocking, anything you add after that line wont run until you close the window.  The QPushButton we create has its :code:`clicked` signal connected to a callback function that prints "beep" to the console.

*******************************
Application with Worker Thread
*******************************

There is one problem with the minimal example above- it doesn't leave us any spot to put SDR/DSP oriented code.  The :code:`MainWindow`'s :code:`__init__` is where the GUI is configured and callbacks are defined, but you absoluately do not want to add any other code (such as SDR or DSP code) to it.  The reason is that the GUI is single-threaded, and if you block the GUI thread with long-running code, the GUI will freeze/stutter, and we want the smoothest GUI possible.  To get around this, we can use a worker thread to run the SDR/DSP code in the background.

The example below extends the minimal example above to include a worker thread that runs code (in the :code:`run` function) nonstop.  We don't use a :code:`while True:` though, because of the way PyQt works under the hood, we want our :code:`run` function to finish and start over periodically.  In order to do this, the worker thread's :code:`end_of_run` signal (which we discuss more in the next section) is connected to a callback function that triggers the worker thread's :code:`run` function again.  We also must initialize the worker thread in the :code:`MainWindow` code, which involves creating a new :code:`QThread` and assigning our custom worker to it.  This code might seem complicated, but it is a very common pattern in PyQt applications and the main take-away is that the GUI-oriented code goes in :code`MainWindow`, and the SDR/DSP-oriented code goes in the worker thread's :code:`run` function.

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

Try running the above code, you should see a "Starting run()" in the console every 1 second, and the pushbutton should still work (without any delay).  Within the worker thread, all we are doing now is a print and a sleep, but soon we will be adding the SDR handling and DSP code to it.

*************************
Signals and Slots
*************************

In the above example, we used the :code:`end_of_run` signal to communicate between the worker thread and the GUI thread.  This is a common pattern in PyQt applications, and is known as the "signals and slots" mechanism.  A signal is emitted by an object (in this case, the worker thread) and is connected to a slot (in this case, the callback function :code:`end_of_run_callback` in the GUI thread).  The signal can be connected to multiple slots, and the slot can be connected to multiple signals.  The signal can also carry arguments, which are passed to the slot when the signal is emitted.  Note that we can also reverse things; the GUI thread is able to send a signal to the worker thread's slot.  The signal/slot mechanism is a powerful way to communicate between different parts of a PyQt application, creating an event-driven structure, and is used extensively in the example code that follows.  Just remember that a slot is simply a callback function, and a signal is a way to signal that callback function.  

*************************
PyQtGraph
*************************

PyQtGraph is a library built on top of PyQt and NumPy that provides fast and efficient plotting capabilities, as PyQt is too general purpose to come with plotting functionality.  It is designed to be used in real-time applications, and is optimized for speed.  It is similar in a lot of ways to matplotlib, but meant for real-time applications instead of single plots.  Using the simple example below you can compare the performance of PyQtGraph to matplotlib, simply change the :code:`if True:` to :code:`False:`.  On an Intel Core i9-10900K @ 3.70 GHz the PyQtGraph code updated at over 1000 FPS while the matplotlib code updated at 40 FPS.  That being said, if you find yourself benefiting from using matplotlib (e.g., to save development time, or because you want a specific feature that PyQtGraph doesn't support), you can incorporate matplotlib plots into a PyQt application, using the code below as a starting point.

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

As far as using PyQtGraph, we import it with :code:`import pyqtgraph as pg` and then we can create a Qt widget that represents a 1D plot as follows (this code goes in the :code:`MainWindow`'s :code:`__init__`):

.. code-block:: python

        # Example PyQtGraph plot
        time_plot = pg.PlotWidget(labels={'left': 'Amplitude', 'bottom': 'Time'})
        time_plot_curve = time_plot.plot(np.arange(1000), np.random.randn(1000)) # x and y
        time_plot.setYRange(-5, 5)

        self.setCentralWidget(time_plot)

.. image:: ../_images/pyqtgraph_example.png
   :scale: 80 % 
   :align: center
   :alt: PyQtGraph example

You can see how it's relatively straightforward to set up a plot, and the result is simply another widget to add to your GUI.  In addition to 1D plots, PyQtGraph also has an equivalent to matplotlib's :code:`imshow()` which plots 2D using a colormap, which we will use for our real-time spectrogram/waterfall.  One nice part about PyQtGraph is that the plots it creates are simply Qt widgets and we add other Qt elements (e.g. a rectangle of a certain size at a certain coordinate) using pure PyQt.  This is because PyQtGraph makes use of PyQt's :code:`QGraphicsScene` class, which provides a surface for managing a large number of 2D graphical items, and nothing is stopping us from adding lines, rectangles, text, ellipses, polygons, and bitmaps, using straight PyQt.

*******
Layouts
*******

In the above examples, we used :code:`self.setCentralWidget()` to set the main widget of the window.  This is a simple way to set the main widget, but it doesn't allow for more complex layouts.  For more complex layouts, we can use layouts, which are a way to arrange widgets in a window.  There are several types of layouts, including :code:`QHBoxLayout`, :code:`QVBoxLayout`, :code:`QGridLayout`, and :code:`QFormLayout`.  The :code:`QHBoxLayout` and :code:`QVBoxLayout` arrange widgets horizontally and vertically, respectively.  The :code:`QGridLayout` arranges widgets in a grid, and the :code:`QFormLayout` arranges widgets in a two-column layout, with labels in the first column and input widgets in the second column.

To create a new layout and add widgets to it, try adding the following inside your :code:`MainWindow`'s :code:`__init__`:

.. code-block:: python

    layout = QHBoxLayout()
    layout.addWidget(QPushButton("Left-Most"))
    layout.addWidget(QPushButton("Center"), 1)
    layout.addWidget(QPushButton("Right-Most"), 2)
    self.setLayout(layout)

In this example we are stacking the widgets horizontally, but by swapping :code:`QHBoxLayout` for :code:`QVBoxLayout` we can stack them vertically instead.  The :code:`addWidget` function is used to add widgets to the layout, and the optional second argument is a stretch factor that determines how much space the widget should take up relative to the other widgets in the layout.  

:code:`QGridLayout` has extra parameters because you must specify the row and column of the widget, and you can optionally specify how many rows and columns the widget should span (default is 1 and 1).  Here is an example of a :code:`QGridLayout`:

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

For our spectrum analyzer we will use the :code:`QGridLayout` for the overall layout, but we will also be adding :code:`QHBoxLayout` to stack widgets horizontally within a space in the grid.  You can nest layouts simply by create a new layout and adding it to the top-level (or parent) layout, e.g.:

.. code-block:: python

    layout = QGridLayout()
    self.setLayout(layout)
    inner_layout = QHBoxLayout()
    layout.addLayout(inner_layout)

************
QPushButton
************

The first actual widget we will cover is the QPushButton, which is a simple button that can be clicked.  We have already seen how to create a QPushButton and connect its :code:`clicked` signal to a callback function.  The QPushButton has a few other signals, including :code:`pressed`, :code:`released`, and :code:`toggled`.  The :code:`toggled` signal is emitted when the button is checked or unchecked, and is useful for creating toggle buttons.  The QPushButton also has a few properties, including :code:`text`, :code:`icon`, and :code:`checkable`.  The QPushButton also has a method called :code:`click()` which simulates a click on the button.  For our SDR spectrum analyzer application we will be using buttons to trigger an autorange for plots, using the current data to calculate the y limits.  Because we have already used the QPushButton, we won't go into more detail here, but you can find more information in the `QPushButton documentation <https://doc.qt.io/qtforpython/PySide6/QtWidgets/QPushButton.html>`_.

************
QSlider
************

The QSlider is a widget that allows the user to select a value from a range of values.  The QSlider has a few properties, including :code:`minimum`, :code:`maximum`, :code:`value`, and :code:`orientation`.  The QSlider also has a few signals, including :code:`valueChanged`, :code:`sliderPressed`, and :code:`sliderReleased`.  The QSlider also has a method called :code:`setValue()` which sets the value of the slider, we will be using this a lot.  The documentation page for `QSlider is here <https://doc.qt.io/qtforpython/PySide6/QtWidgets/QSlider.html>`_.

For our spectrum analyzer application we will be using QSliders to adjust the center frequency and gain of the SDR.  Here is the snippet from the final application code that creates the gain slider:

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

One very important thing to know about QSlider is it uses integers, so by setting the range from 0 to 73 we are allowing the slider to choose integer values between those numbers (inclusive of start and end).  The :code:`setTickInterval(2)` is purely a visual thing.  It is for this reason that we will use kHz as the units for the frequency slider, so that we can have granularity down to the 1 kHz.

Halfway into the code above you'll notice we create a :code:`QLabel`, which is just a text label for display purposes, but in order for it to display the current value of the slider we must create a slot (i.e., callback function) that updates the label.  We connect this callback function to the :code:`sliderMoved` signal, which is automatically emitted whenever the slider is moved.  We also call the callback function once to initialize the label with the current value of the slider (50 in our case).  We also have to connect the :code:`sliderMoved` signal to a slot that lives within the worker thread, which will update the gain of the SDR (remember, we don't like to manage the SDR or do DSP in the main GUI thread). The callback function that defines this slot will be discussed later.

************
QComboBox
************

The QComboBox is a dropdown-style widget that allows the user to select an item from a list of items.  The QComboBox has a few properties, including :code:`currentText`, :code:`currentIndex`, and :code:`count`.  The QComboBox also has a few signals, including :code:`currentTextChanged`, :code:`currentIndexChanged`, and :code:`activated`.  The QComboBox also has a method called :code:`addItem()` which adds an item to the list, and :code:`insertItem()` which inserts an item at a specific index, although we will not be using them in our spectrum analyzer example.  The documentation page for `QComboBox is here <https://doc.qt.io/qtforpython/PySide6/QtWidgets/QComboBox.html>`_.

For our spectrum analyzer application we will be using QComboBox to select the sample rate from a list we predefine.  At the beginning of our code we define the possible sample rates using :code:`sample_rates = [56, 40, 20, 10, 5, 2, 1, 0.5]`.  Within the :code:`MainWindow`'s :code:`__init__` we create the QComboBox as follows:

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

The only real difference between this and the slider is the :code:`addItems()` where you give it the list of strings to use as options, and :code:`setCurrentIndex()` which sets the starting value.

