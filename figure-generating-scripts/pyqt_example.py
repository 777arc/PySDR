from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget  # tested with PyQt6==6.7.0
import pyqtgraph as pg # tested with pyqtgraph==0.13.7
import numpy as np
import time
import threading # simpler code than using QThread with signals and slots

fft_size = 1024
num_rows = 500
center_freq = 100e6
sample_rate = 10e6
time_plot_samples = 500

def worker_thread(window, quit_event):
    spectrogram = np.zeros((fft_size, num_rows))
    PSD_avg = np.zeros(fft_size)
    f = np.linspace(center_freq - sample_rate/2.0, center_freq + sample_rate/2.0, fft_size) / 1e6
    t = np.arange(time_plot_samples)/sample_rate*1e6 # in microseconds
    first_time = True
    i = num_rows - 1 # counter to reset colormap, but do it at the start
    while not quit_event.is_set(): # will break when we close the window
        start_t = time.time()
        
        samples = np.random.randn(fft_size) + 0.5 +  1j*np.random.randn(fft_size) # generate some random samples

        window.time_plot_curve_i.setData(samples[0:time_plot_samples].real)
        window.time_plot_curve_q.setData(samples[0:time_plot_samples].imag)
        

        PSD = 10.0*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples)))**2)

        PSD_avg = PSD_avg * 0.99 + PSD * 0.01
        
        window.fft_plot_curve_fft.setData(f, PSD_avg) # FFT plot
        
    
        spectrogram[:] = np.roll(spectrogram, 1, axis=1) # shifts waterfall 1 row
        spectrogram[:,0] = PSD # fill last row with new fft results
            
        # Display waterfall
        i += 1
        if i == num_rows:
            window.imageitem.setImage(spectrogram, autoLevels=True) # auto range only on occasion
            window.fft_plot.autoRange()
            window.time_plot.autoRange()
            i = 0
        else:
            window.imageitem.setImage(spectrogram, autoLevels=False)

        #window.imageitem.translate((center_freq - sample_rate/2.0) / 1e6, 0)
        #window.imageitem.scale(sample_rate/fft_size/1e6, time_per_row)

        print("Frames per second:", 1/(time.time() - start_t))

# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("The PySDR Spectrum Analyzer")
        self.setFixedSize(QSize(1500, 1000)) # window size, starting size should fit on 1920 x 1080

        layout = QGridLayout()

        # Time plot
        self.time_plot = pg.PlotWidget(labels={'left': 'Amplitude', 'bottom': 'Time [microseconds]'}, enableMenu=False)
        #self.time_plot.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.time_plot.setMouseEnabled(x=False, y=True)
        self.time_plot_curve_i = self.time_plot.plot([]) 
        self.time_plot_curve_q = self.time_plot.plot([]) 
        layout.addWidget(self.time_plot, 1, 0)

        # create fft plot
        self.fft_plot = pg.PlotWidget(labels={'left': 'PSD', 'bottom': 'Frequency [MHz]'}, enableMenu=False)
        #self.fft_plot.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.fft_plot.setMouseEnabled(x=False, y=True)
        self.fft_plot_curve_fft = self.fft_plot.plot([]) 
        layout.addWidget(self.fft_plot, 2, 0)
        
        # Create waterfall plot
        self.waterfall = pg.PlotWidget(labels={'left': 'Time [s]', 'bottom': 'Frequency [MHz]'}, enableMenu=False)
        #self.waterfall.getPlotItem().getViewBox().translateBy(x=10.0)
        self.imageitem = pg.ImageItem(axisOrder='col-major') # some of these args are purely for performance
        self.imageitem.setColorMap(pg.colormap.get('viridis', source='matplotlib'))
        self.waterfall.addItem(self.imageitem)
        self.waterfall.setMouseEnabled(x=False, y=False)
        layout.addWidget(self.waterfall, 3, 0)

        # Colorbar for waterfall
        '''
        colorbar = pg.GraphicsLayoutWidget() # the bar needs a widget to be contained in
        bar = pg.ColorBarItem(colorMap=pg.colormap.get('viridis', source='matplotlib'))
        bar.setImageItem(self.imageitem)
        colorbar.addItem(bar)
        layout.addWidget(colorbar, 3, 1)
        '''

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def handleButton(self):
        self.time_plot.autoRange()
        self.fft_plot.autoRange()

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
