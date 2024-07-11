from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow  # tested with PyQt6==6.7.0
import pyqtgraph as pg # tested with pyqtgraph==0.13.7
import numpy as np
import time
import threading # simpler code than using QThread with signals and slots

def worker_thread(window, quit_event):
    while not quit_event.is_set(): # will break when we close the window
        start_t = time.time()
        window.time_plot_curve_i.setData(np.arange(500), np.random.randn(500))
        window.time_plot_curve_q.setData(np.arange(500), np.random.randn(500))
        time.sleep(0.01)
        print("Frames per second:", 1/(time.time() - start_t))

# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("The PySDR Spectrum Analyzer")
        self.setFixedSize(QSize(400, 300)) # window size

        # create time plot
        self.time_plot = pg.PlotWidget(labels={'left': 'Amplitude', 'bottom': 'Time [microseconds]'}, enableMenu=False)
        #self.time_plot.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.time_plot.setMouseEnabled(x=False, y=True)
        self.time_plot_curve_i = self.time_plot.plot([]) 
        self.time_plot_curve_q = self.time_plot.plot([]) 
        #grid.addWidget(self.time_plot, 1, 0)

        self.setCentralWidget(self.time_plot) # Set the central widget of the Window

app = QApplication([])
window = MainWindow()
window.show()  # IMPORTANT!!!!! Windows are hidden by default.

# Spawn and start the worker thread
threads = []
quit_event = threading.Event() # Make a signal for the threads to stop running
rx_thread = threading.Thread(target=worker_thread, args=(window, quit_event))
threads.append(rx_thread)
rx_thread.start()
rx_thread.setName("worker_thread") # so we can monitor it using htop or system monitor

app.exec() # Start the event loop

# Interrupt and join the threads, so that when you close the window, the entire app stops
quit_event.set()
for thr in threads:
    thr.join()
