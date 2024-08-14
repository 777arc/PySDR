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
