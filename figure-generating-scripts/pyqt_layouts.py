from PyQt6.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QGridLayout, QPushButton, QWidget

class Window(QWidget):
    def __init__(self):
        super().__init__()
        if False:
            self.setWindowTitle("QHBoxLayout Example")
            layout = QHBoxLayout()
            layout.addWidget(QPushButton("Left-Most"))
            layout.addWidget(QPushButton("Center"), 1)
            layout.addWidget(QPushButton("Right-Most"), 2)
        if False:
            self.setWindowTitle("QVBoxLayout Example")
            layout = QVBoxLayout()
            layout.addWidget(QPushButton("Top"))
            layout.addWidget(QPushButton("Center"))
            layout.addWidget(QPushButton("Bottom"))
        if True:
            self.setWindowTitle("QGridLayout Example")
            layout = QGridLayout()
            layout.addWidget(QPushButton("Button at (0, 0)"), 0, 0)
            layout.addWidget(QPushButton("Button at (0, 1)"), 0, 1)
            layout.addWidget(QPushButton("Button at (0, 2)"), 0, 2)
            layout.addWidget(QPushButton("Button at (1, 0)"), 1, 0)
            layout.addWidget(QPushButton("Button at (1, 1)"), 1, 1)
            layout.addWidget(QPushButton("Button at (1, 2)"), 1, 2)
            layout.addWidget(QPushButton("Button at (2, 0) spanning 2 columns"), 2, 0, 1, 2)
        self.setLayout(layout)

app = QApplication([])
window = Window()
window.show()
app.exec() 
