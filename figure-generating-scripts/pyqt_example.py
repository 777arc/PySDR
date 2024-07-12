from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal, QObject
from PyQt6.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QSlider, QLabel  # tested with PyQt6==6.7.0
import pyqtgraph as pg # tested with pyqtgraph==0.13.7
import numpy as np
import time

fft_size = 1024
num_rows = 500
center_freq = 100e6
sample_rate = 10e6
time_plot_samples = 500

class SDRWorker(QObject):
    time_plot_update = pyqtSignal(np.ndarray)
    fft_plot_update = pyqtSignal(np.ndarray)
    waterfall_plot_update = pyqtSignal(np.ndarray, bool)

    freq = 0 # in kHz, to deal with QSlider being ints and with a max of 2 billion

    # Slots
    def update_freq(self, val):
        print("Updated freq to:", val, 'kHz')
        self.freq = val
    
    def run(self):
        spectrogram = np.zeros((fft_size, num_rows))
        PSD_avg = np.zeros(fft_size)
        t = np.arange(time_plot_samples)/sample_rate*1e6 # in microseconds
        i = num_rows - 1 # counter to reset colormap, but do it at the start
        while True:
            
            start_t = time.time()

            QApplication.processEvents()
            
            samples = np.random.randn(fft_size) + 0.5 +  1j*np.random.randn(fft_size) # generate some random samples

            self.time_plot_update.emit(samples[0:time_plot_samples])
            
            PSD = 10.0*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples)))**2)

            PSD_avg = PSD_avg * 0.99 + PSD * 0.01
            self.fft_plot_update.emit(PSD_avg)
        
            spectrogram[:] = np.roll(spectrogram, 1, axis=1) # shifts waterfall 1 row
            spectrogram[:,0] = PSD # fill last row with new fft results

            i += 1
            if i == num_rows:
                self.waterfall_plot_update.emit(spectrogram, True)
                i = 0
            else:
                self.waterfall_plot_update.emit(spectrogram, False)
            
            time.sleep(0.01) # without a tiny delay the main GUI thread gets blocked and you cant move the slider
            
            #window.imageitem.translate((center_freq - sample_rate/2.0) / 1e6, 0)
            #window.imageitem.scale(sample_rate/fft_size/1e6, time_per_row)

            #print("Frames per second:", 1/(time.time() - start_t))


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("The PySDR Spectrum Analyzer")
        self.setFixedSize(QSize(1500, 1000)) # window size, starting size should fit on 1920 x 1080

        layout = QGridLayout()

        # Initialize worker and thread
        self.sdr_thread = QThread()
        self.worker = SDRWorker()
        self.worker.moveToThread(self.sdr_thread)

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
        self.imageitem.setLevels((20, 50))
        self.waterfall.addItem(self.imageitem)
        self.waterfall.setMouseEnabled(x=False, y=False)
        layout.addWidget(self.waterfall, 3, 0)

        # Colorbar for waterfall
        colorbar = pg.GraphicsLayoutWidget() # the bar needs a widget to be contained in
        bar = pg.ColorBarItem(colorMap=pg.colormap.get('viridis', source='matplotlib'))
        bar.setImageItem(self.imageitem)
        colorbar.addItem(bar)
        layout.addWidget(colorbar, 3, 1)

        # Label for freq slider
        freq_label = QLabel()
        def update_freq_label(val):
            freq_label.setText("Frequency [MHz]: " + str(val/1e3))
        layout.addWidget(freq_label, 4, 1)

        # Freq slider, all units in kHz
        self.freq_slider = QSlider(Qt.Orientation.Horizontal)
        self.freq_slider.setRange(0, int(6e6))
        self.freq_slider.setValue(int(750e3))
        self.freq_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.freq_slider.setTickInterval(int(1e6))
        self.freq_slider.sliderMoved.connect(self.worker.update_freq) # there's also a valueChanged option
        self.freq_slider.sliderMoved.connect(update_freq_label)
        update_freq_label(self.freq_slider.value()) # initialize the label
        layout.addWidget(self.freq_slider, 4, 0)


        
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Signals and slots stuff
        f = np.linspace(center_freq - sample_rate/2.0, center_freq + sample_rate/2.0, fft_size) / 1e6
        def time_plot_callback(samples):
            self.time_plot_curve_i.setData(samples.real)
            self.time_plot_curve_q.setData(samples.imag)
        def fft_plot_callback(PSD_avg):
            self.fft_plot_curve_fft.setData(f, PSD_avg)
        def waterfall_plot_callback(spectrogram, reset_range):
            self.imageitem.setImage(spectrogram, autoLevels=False) 
            if reset_range:
                self.fft_plot.autoRange()
                self.time_plot.autoRange()
        self.worker.time_plot_update.connect(time_plot_callback) # connect the signal to the callback
        self.worker.fft_plot_update.connect(fft_plot_callback)
        self.worker.waterfall_plot_update.connect(waterfall_plot_callback)
        self.sdr_thread.started.connect(self.worker.run) # kicks off the worker
        self.sdr_thread.start()
    
    #def update_colormap(self):
    #    self.imageitem.setLevels(self.range_slider.value())

app = QApplication([])
window = MainWindow()
window.show()  # IMPORTANT!!!!! Windows are hidden by default.
app.exec() # Start the event loop

