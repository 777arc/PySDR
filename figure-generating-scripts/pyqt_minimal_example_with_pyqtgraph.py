from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton
import pyqtgraph as pg
import numpy as np

# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Example PyQtGraph plot
        time_plot = pg.PlotWidget(labels={'left': 'Amplitude', 'bottom': 'Time'})
        time_plot_curve = time_plot.plot(np.arange(1000), np.random.randn(1000)) # x and y
        time_plot.setYRange(-5, 5)

        self.setCentralWidget(time_plot)

app = QApplication([])
window = MainWindow()
window.show() # Windows are hidden by default
app.exec() # Start the event loop
