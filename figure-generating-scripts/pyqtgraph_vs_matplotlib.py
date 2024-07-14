import numpy as np
import time
import matplotlib
matplotlib.use('Qt5Agg')
from PyQt6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pyqtgraph as pg # tested with pyqtgraph==0.13.7

n_data = 1024


if False:
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