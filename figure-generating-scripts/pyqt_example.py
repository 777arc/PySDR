from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal, QObject, QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QSlider, QLabel, QHBoxLayout, QVBoxLayout, QPushButton, QComboBox  # tested with PyQt6==6.7.0
import pyqtgraph as pg # tested with pyqtgraph==0.13.7
import numpy as np
import time
import adi
import signal

# SDR thread is using 18% with old method

# Defaults
fft_size = 4096 # determines buffer size
num_rows = 200
center_freq = 2400e6
sample_rates = [20, 10, 5, 2.5, 1] # MHz
sample_rate = sample_rates[0] * 1e6
time_plot_samples = 500
gain = 50 # 0 to 73 dB. int

# Init SDR
sdr = adi.Pluto("ip:192.168.1.233")
sdr.rx_lo = int(center_freq)
sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(sample_rate*0.8) # antialiasing filter bandwidth
sdr.rx_buffer_size = int(fft_size)
sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = gain # dB


class SDRWorker(QObject):
    # PyQt Signals
    time_plot_update = pyqtSignal(np.ndarray)
    freq_plot_update = pyqtSignal(np.ndarray)
    waterfall_plot_update = pyqtSignal(np.ndarray)
    end_of_run = pyqtSignal() # happens many times a second

    # State
    freq = 0 # in kHz, to deal with QSlider being ints and with a max of 2 billion
    spectrogram = -50*np.ones((fft_size, num_rows))
    PSD_avg = -50*np.ones(fft_size)
    
    # PyQt Slots
    def update_freq(self, val): # TODO: WE COULD JUST MODIFY THE SDR IN THE GUI THREAD
        print("Updated freq to:", val, 'kHz')
        sdr.rx_lo = int(val*1e3)
    
    def update_gain(self, val):
        print("Updated gain to:", val, 'dB')
        sdr.rx_hardwaregain_chan0 = val

    def update_sample_rate(self, val):
        print("Updated sample rate to:", sample_rates[val], 'MHz')
        sdr.sample_rate = int(sample_rates[val] * 1e6)
        sdr.rx_rf_bandwidth = int(sample_rates[val] * 1e6 * 0.8)

    # Main loop
    def run(self):
        start_t = time.time()

        QApplication.processEvents()
        
        samples = sdr.rx() # Receive samples
        samples = samples.astype(np.complex64) # type: ignore

        self.time_plot_update.emit(samples[0:time_plot_samples]/2**11) # make it go from -1 to 1 at highest gain
        
        PSD = 10.0*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples)))**2/(fft_size*sample_rate))
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
        auto_range_button2 = QPushButton('-1 to +1')
        auto_range_button2.clicked.connect(lambda : time_plot.setYRange(-1.1, 1.1))
        time_plot_auto_range_layout.addWidget(auto_range_button2)

        # Freq plot
        freq_plot = pg.PlotWidget(labels={'left': 'PSD', 'bottom': 'Frequency [MHz]'})
        freq_plot.setMouseEnabled(x=False, y=True)
        freq_plot_curve = freq_plot.plot([]) 
        freq_plot.setXRange(center_freq/1e6 - sample_rate/2e6, center_freq/1e6 + sample_rate/2e6)
        freq_plot.setYRange(-60, -40)
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
        imageitem.setLevels((-60, -40)) # needs to come after colorbar is created for some reason
        waterfall_layout.addWidget(colorbar)

        # Waterfall auto range button
        auto_range_button = QPushButton('Auto Range')
        def update_colormap():
            imageitem.setLevels((self.spectrogram_min, self.spectrogram_max))
            colorbar.setLevels((self.spectrogram_min, self.spectrogram_max))
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

        # Sample rate dropdown (QComboBox)
        sample_rate_combobox = QComboBox()
        sample_rate_combobox.addItems([str(x) + ' MHz' for x in sample_rates])
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
            f = np.linspace(freq_slider.value()*1e3 - sample_rate/2.0, freq_slider.value()*1e3 + sample_rate/2.0, fft_size) / 1e6
            freq_plot_curve.setData(f, PSD_avg)
            freq_plot.setXRange(freq_slider.value()*1e3/1e6 - sample_rate/2e6, freq_slider.value()*1e3/1e6 + sample_rate/2e6)
        
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
window.show()  # IMPORTANT!!!!! Windows are hidden by default
signal.signal(signal.SIGINT, signal.SIG_DFL) # this lets control-C actually close the app
app.exec() # Start the event loop

