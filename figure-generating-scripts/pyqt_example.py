from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal, QObject
from PyQt6.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QSlider, QLabel, QHBoxLayout, QPushButton  # tested with PyQt6==6.7.0
import pyqtgraph as pg # tested with pyqtgraph==0.13.7
import numpy as np
import time
import adi

# Defaults
fft_size = 1024 # determines buffer size
num_rows = 500
center_freq = 100e6
sample_rate = 10e6
time_plot_samples = 500
gain = 65 # 0 to 73 dB. int

# Init SDR
sdr = adi.Pluto("ip:192.168.1.174")
sdr.rx_lo = int(center_freq)
sdr.rx_rf_bandwidth = int(2*sample_rate)
sdr.rx_buffer_size = int(fft_size)
sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = gain # dB


class SDRWorker(QObject):
    time_plot_update = pyqtSignal(np.ndarray)
    fft_plot_update = pyqtSignal(np.ndarray)
    waterfall_plot_update = pyqtSignal(np.ndarray, bool)

    freq = 0 # in kHz, to deal with QSlider being ints and with a max of 2 billion

    # Slots
    def update_freq(self, val): # TODO: WE COULD JUST MODIFY THE SDR IN THE GUI THREAD
        print("Updated freq to:", val, 'kHz')
        sdr.rx_lo = int(val*1e3)
    
    def update_gain(self, val):
        print("Updated gain to:", val, 'dB')
        sdr.rx_hardwaregain_chan0 = val

    def run(self):
        spectrogram = np.zeros((fft_size, num_rows))
        PSD_avg = np.zeros(fft_size)
        t = np.arange(time_plot_samples)/sample_rate*1e6 # in microseconds
        i = num_rows - 1 # counter to reset colormap, but do it at the start
        while True:
            
            start_t = time.time()

            QApplication.processEvents()
            
            #samples = np.random.randn(fft_size) + 0.5 +  1j*np.random.randn(fft_size) # generate some random samples
            samples = sdr.rx() # Receive samples
            samples = samples.astype(np.complex64) # type: ignore

            self.time_plot_update.emit(samples[0:time_plot_samples])
            
            PSD = 10.0*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples)))**2/(fft_size*sample_rate))

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
            
            time.sleep(0.001) # without a tiny delay the main GUI thread gets blocked and you cant move the slider
            
            #window.imageitem.translate((center_freq - sample_rate/2.0) / 1e6, 0)
            #window.imageitem.scale(sample_rate/fft_size/1e6, time_per_row)

            #print("Frames per second:", 1/(time.time() - start_t))


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("The PySDR Spectrum Analyzer")
        self.setFixedSize(QSize(1500, 1000)) # window size, starting size should fit on 1920 x 1080

        self.spectrogram_min = 0
        self.spectrogram_max = 0

        layout = QGridLayout()

        # Initialize worker and thread
        self.sdr_thread = QThread()
        self.worker = SDRWorker()
        self.worker.moveToThread(self.sdr_thread)

        # Time plot
        self.time_plot = pg.PlotWidget(labels={'left': 'Amplitude', 'bottom': 'Time [microseconds]'}, enableMenu=False)
        #self.time_plot.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.time_plot.setMouseEnabled(x=False, y=True)
        self.time_plot.setYRange(-2048, 2048) # Plutos sampling range
        self.time_plot_curve_i = self.time_plot.plot([]) 
        self.time_plot_curve_q = self.time_plot.plot([]) 
        layout.addWidget(self.time_plot, 1, 0)

        # create fft plot
        self.fft_plot = pg.PlotWidget(labels={'left': 'PSD', 'bottom': 'Frequency [MHz]'}, enableMenu=False)
        #self.fft_plot.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.fft_plot.setMouseEnabled(x=False, y=True)
        self.fft_plot_curve_fft = self.fft_plot.plot([]) 
        self.fft_plot.setXRange(center_freq/1e6 - sample_rate/2e6, center_freq/1e6 + sample_rate/2e6)
        self.fft_plot.setYRange(-60, 0)
        layout.addWidget(self.fft_plot, 2, 0)
        
        # Layout container for waterfall related stuff
        waterfall_layout = QHBoxLayout()
        layout.addLayout(waterfall_layout, 3, 0)

        # Waterfall plot
        self.waterfall = pg.PlotWidget(labels={'left': 'Time [s]', 'bottom': 'Frequency [MHz]'}, enableMenu=False)
        #self.waterfall.getPlotItem().getViewBox().translateBy(x=10.0)
        self.imageitem = pg.ImageItem(axisOrder='col-major') # some of these args are purely for performance
        self.imageitem.setColorMap(pg.colormap.get('viridis', source='matplotlib'))
        self.imageitem.setLevels((-60, 0))
        self.waterfall.addItem(self.imageitem)
        self.waterfall.setMouseEnabled(x=False, y=False)
        waterfall_layout.addWidget(self.waterfall)

        # Colorbar for waterfall
        colorbar = pg.GraphicsLayoutWidget() # the bar needs a widget to be contained in
        colorbar.setMaximumWidth(80)
        bar = pg.ColorBarItem(colorMap=pg.colormap.get('viridis', source='matplotlib'), width=40)
        bar.setImageItem(self.imageitem)
        colorbar.addItem(bar)
        waterfall_layout.addWidget(colorbar)

        # Auto range button
        auto_range_button = QPushButton('Auto\nRange')
        auto_range_button.setMaximumWidth(50)
        def update_colormap():
            new_min = self.spectrogram_min + 5
            new_max = self.spectrogram_max - 5
            self.imageitem.setLevels((new_min, new_max))
            bar.setLevels((new_min, new_max))
        auto_range_button.clicked.connect(update_colormap)
        waterfall_layout.addWidget(auto_range_button)

        # Freq slider with label, all units in kHz
        freq_slider = QSlider(Qt.Orientation.Horizontal)
        freq_slider.setRange(0, int(6e6))
        freq_slider.setValue(int(center_freq/1e3))
        freq_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        freq_slider.setTickInterval(int(1e6))
        freq_slider.sliderMoved.connect(self.worker.update_freq) # there's also a valueChanged option
        freq_label = QLabel()
        def update_freq_label(val):
            freq_label.setText("Frequency [MHz]: " + str(val/1e3))
            self.fft_plot.autoRange()
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
        gain_slider.sliderMoved.connect(self.worker.update_gain)
        gain_label = QLabel()
        def update_gain_label(val):
            gain_label.setText("Gain: " + str(val))
        gain_slider.sliderMoved.connect(update_gain_label)
        update_gain_label(gain_slider.value()) # initialize the label
        layout.addWidget(gain_slider, 5, 0)
        layout.addWidget(gain_label, 5, 1)


        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Signals and slots stuff
        def time_plot_callback(samples):
            self.time_plot_curve_i.setData(samples.real)
            self.time_plot_curve_q.setData(samples.imag)
        def fft_plot_callback(PSD_avg):
            # TODO figure out if there's a way to just change the visual ticks instead of the actual x vals
            f = np.linspace(freq_slider.value()*1e3 - sample_rate/2.0, freq_slider.value()*1e3 + sample_rate/2.0, fft_size) / 1e6
            self.fft_plot_curve_fft.setData(f, PSD_avg)
            self.fft_plot.setXRange(freq_slider.value()*1e3/1e6 - sample_rate/2e6, freq_slider.value()*1e3/1e6 + sample_rate/2e6)
        def waterfall_plot_callback(spectrogram, reset_range):
            self.imageitem.setImage(spectrogram, autoLevels=False) 
            if reset_range:
                self.fft_plot.autoRange()
            self.spectrogram_min = np.min(spectrogram)
            self.spectrogram_max = np.max(spectrogram)
        self.worker.time_plot_update.connect(time_plot_callback) # connect the signal to the callback
        self.worker.fft_plot_update.connect(fft_plot_callback)
        self.worker.waterfall_plot_update.connect(waterfall_plot_callback)
        self.sdr_thread.started.connect(self.worker.run) # kicks off the worker
        self.sdr_thread.start()


app = QApplication([])
window = MainWindow()
window.show()  # IMPORTANT!!!!! Windows are hidden by default.
app.exec() # Start the event loop

