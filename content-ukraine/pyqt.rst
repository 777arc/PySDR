.. _pyqt-chapter:

###############################
Графічні інтерфейси реального часу з PyQt
###############################

У цьому розділі ми навчимося створювати графічні інтерфейси користувача (GUI) реального часу на Python, використовуючи PyQt — Python-обгортку до Qt.  У межах розділу ми збудуємо аналізатор спектра з графіками часу, частоти та спектрограми/«водоспаду», а також віджетами введення для налаштування різних параметрів SDR.  Приклад підтримує PlutoSDR, USRP або режим лише моделювання.

****************
Вступ
****************

Qt (вимовляється як «к'ют») — це фреймворк для створення GUI-додатків, які можуть працювати в Linux, Windows, macOS і навіть Android.  Це дуже потужний фреймворк, який використовується в багатьох комерційних застосунках, і написаний на C++ для максимальної продуктивності.  PyQt — це Python-обгортка до Qt, яка дає змогу створювати GUI-додатки на Python, водночас користуючись ефективністю фреймворку на C++.  У цьому розділі ми навчимося використовувати PyQt для створення аналізатора спектра реального часу, який може працювати з SDR (або із змодельованим сигналом).  Аналізатор спектра матиме графіки часу, частоти та спектрограми/«водоспаду», а також елементи введення для налаштування різних параметрів SDR.  Для побудови графіків ми використовуємо `PyQtGraph <https://www.pyqtgraph.org/>`_, це окрема бібліотека поверх PyQt.  Для введення використовуємо повзунки, випадаючі списки та кнопки.  У прикладі використано PyQt6, але кожен рядок коду (окрім :code:`import`) ідентичний PyQt5, з погляду API зміни мінімальні.  Не дивно, що цей розділ значною мірою складається з Python-коду й прикладів.  До кінця розділу ви познайомитеся з будівельними блоками, потрібними для створення власного інтерактивного SDR-додатка!

****************
Огляд Qt
****************

Qt — дуже великий фреймворк, і ми лише злегка торкнемося його можливостей.  Проте є кілька ключових концепцій, які важливо розуміти, працюючи з Qt/PyQt:

- **Віджети**: Віджети — це будівельні блоки додатка Qt, вони використовуються для створення GUI.  Існує багато різновидів віджетів, зокрема кнопки, повзунки, написи й графіки.  Віджети можна розміщувати у лейаутах, які визначають їх положення на екрані.

- **Лейаути**: Лейаути використовують для компонування віджетів у вікні.  Є кілька типів лейаутів: горизонтальні, вертикальні, сіткові та форм-лейаути.  Лейаути допомагають створювати складні GUI, які реагують на зміну розміру вікна.

- **Сигнали та слоти**: Сигнали та слоти — це спосіб взаємодії різних частин додатка Qt.  Об'єкт випромінює сигнал, коли відбувається певна подія, а сигнал з'єднано зі слотом — функцією зворотного виклику, яку викликають, коли сигнал випромінюється.  Сигнали й слоти забезпечують подієво-орієнтовану структуру в Qt і тримають GUI відгукливим.

- **Таблиці стилів**: Таблиці стилів використовують, щоб налаштувати вигляд віджетів у Qt-додатку.  Вони пишуться в стилі CSS і дозволяють змінювати колір, шрифт та розміри віджетів.

- **Графіка**: Qt має потужний графічний фреймворк для створення користувацької графіки.  Він містить класи для малювання ліній, прямокутників, еліпсів і тексту, а також обробки подій миші та клавіатури.

- **Багатопоточність**: Qt має вбудовану підтримку багатопоточності та надає класи для створення робочих потоків, що працюють у фоновому режимі.  Багатопоточність дозволяє виконувати тривалі операції, не блокуючи основний GUI-потік.

- **OpenGL**: Qt має вбудовану підтримку OpenGL і надає класи для створення 3D-графіки.  OpenGL використовують у застосунках, що потребують високопродуктивної 3D-графіки.  У цьому розділі ми зосередимося лише на 2D.

*************************
Базова структура застосунку
*************************

Перш ніж занурюватися у різні віджети Qt, розглянемо структуру типового Qt-застосунку.  Qt-застосунок складається з головного вікна, яке містить центральний віджет, а той у свою чергу містить основний вміст застосунку.  За допомогою PyQt ми можемо створити мінімальний Qt-застосунок із єдиною кнопкою QPushButton так:

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

Спробуйте запустити цей код, імовірно, доведеться виконати :code:`pip install PyQt6`.  Зверніть увагу, що останній рядок блокує виконання; усе, що ви додасте після нього, не запуститься, поки ви не закриєте вікно.  Кнопка QPushButton, яку ми створили, має сигнал :code:`clicked`, під’єднаний до функції зворотного виклику, яка друкує «beep» у консолі.

*******************************
Застосунок із робочим потоком
*******************************

У мінімальному прикладі вище є одна проблема: він не залишає місця для SDR/DSP-коду.  Метод :code:`__init__` класу :code:`MainWindow` відповідає за конфігурування GUI та визначення зворотних викликів, але додавати туди інший код (наприклад, SDR чи DSP) не варто.  Причина в тому, що GUI однопотоковий, і якщо ви заблокуєте GUI-потік довготривалим кодом, інтерфейс «замерзне» або почне «смикатися», а нам потрібна максимально плавна робота.  Щоб це обійти, можна використати робочий потік, який виконуватиме SDR/DSP у фоні.

Наступний приклад розширює мінімальний код, додаючи робочий потік, що запускає функцію :code:`run` безперервно.  Ми не використовуємо :code:`while True:`, адже через те, як PyQt працює «під капотом», нам потрібно, щоб :code:`run` завершувалась і періодично запускалася знову.  Щоб це реалізувати, сигнал :code:`end_of_run` робочого потоку (обговоримо його у наступному розділі) з'єднано з функцією зворотного виклику, яка повторно запускає :code:`run`.  Також ми маємо ініціалізувати робочий потік у коді :code:`MainWindow`, створивши новий :code:`QThread` і призначивши йому нашого робітника.  Цей код може виглядати складно, але це дуже поширений шаблон у PyQt-додатках, і головна ідея полягає в тому, що GUI-код живе в :code:`MainWindow`, а SDR/DSP-код — у методі :code:`run` робочого потоку.

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

Запустіть цей код: у консолі що секунду з’являтиметься «Starting run()», і кнопка все ще працюватиме без затримок.  Поки що в робочому потоці ми лише друкуємо та «спимо», але скоро додамо керування SDR та DSP.

*************************
Сигнали та слоти
*************************

У прикладі вище ми використали сигнал :code:`end_of_run`, щоб організувати взаємодію між робочим потоком і GUI-потоком.  Це типовий шаблон у PyQt і він відомий як механізм «сигналів і слотів».  Об’єкт випромінює сигнал (у нашому випадку робочий потік) і з’єднується зі слотом (функцією зворотного виклику :code:`end_of_run_callback` у GUI-потоці).  Сигнал можна під’єднати до кількох слотів, і слот може обробляти кілька сигналів.  Сигнал може передавати аргументи, які отримує слот.  Зверніть увагу, що можна організувати взаємодію і в протилежному напрямку: GUI-потік здатен надсилати сигнал у слот робочого потоку.  Механізм сигналів і слотів — потужний спосіб організувати взаємодію частин PyQt-застосунку, створюючи подієву структуру, і ми активно використовуємо його в подальшому прикладі.  Просто пам'ятайте, що слот — це функція зворотного виклику, а сигнал — це спосіб викликати цю функцію.

*************************
PyQtGraph
*************************

PyQtGraph — це бібліотека поверх PyQt та NumPy, яка надає швидкі та ефективні можливості побудови графіків, адже сам PyQt занадто загальний і не містить функціоналу для графіків.  Її створено для використання в реальному часі, і вона оптимізована на швидкість.  У багатьох аспектах вона схожа на Matplotlib, але орієнтована на реальний час, а не на статичні графіки.  У наведеному нижче простому прикладі ви можете порівняти продуктивність PyQtGraph і Matplotlib, просто змініть :code:`if True:` на :code:`False:`.  На Intel Core i9-10900K @ 3.70 GHz код з PyQtGraph оновлювався з частотою понад 1000 FPS, а код з Matplotlib — 40 FPS.  Водночас, якщо вам вигідніше використовувати Matplotlib (наприклад, щоб зекономити час розробки чи скористатися функцією, якої нема в PyQtGraph), можна вбудувати графіки Matplotlib у Qt-застосунок, використавши наведений код як відправну точку.

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

Щоб скористатися PyQtGraph, імпортуємо його як :code:`import pyqtgraph as pg`, після чого можемо створити Qt-віджет для 1D-графіка так (цей код додається у :code:`__init__` :code:`MainWindow`):

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

Ви бачите, що налаштувати графік доволі просто, а результат — ще один віджет, який можна додати до GUI.  Окрім 1D-графіків, PyQtGraph має еквівалент :code:`imshow()` з Matplotlib для побудови 2D-даних за допомогою колірної карти, і ми використаємо його для реальної спектрограми/«водоспаду».  Приємний момент у PyQtGraph полягає в тому, що створені графіки — це просто Qt-віджети, і ми можемо додавати інші елементи Qt (наприклад, прямокутник потрібного розміру в певних координатах) чистим PyQt.  Причина в тому, що PyQtGraph використовує клас PyQt :code:`QGraphicsScene`, який забезпечує поверхню для керування великою кількістю 2D-графічних об’єктів, і ніщо не заважає нам додавати лінії, прямокутники, текст, еліпси, багатокутники та растрові зображення безпосередньо PyQt.

*******
Лейаути
*******

У наведених вище прикладах ми використовували :code:`self.setCentralWidget()` для встановлення головного віджета вікна.  Це простий спосіб задати центральний віджет, але він не дозволяє створювати складніші компоновки.  Для цього ми можемо використати лейаути — структури, що розташовують віджети у вікні.  Є кілька типів лейаутів: :code:`QHBoxLayout`, :code:`QVBoxLayout`, :code:`QGridLayout` та :code:`QFormLayout`.  :code:`QHBoxLayout` і :code:`QVBoxLayout` розташовують віджети відповідно горизонтально та вертикально.  :code:`QGridLayout` розміщує віджети у сітці, а :code:`QFormLayout` створює двоколонну компоновку з написами в першій колонці та віджетами введення в другій.

Щоб створити новий лейаут і додати до нього віджети, спробуйте вставити в :code:`__init__` :code:`MainWindow` такий код:

.. code-block:: python

    layout = QHBoxLayout()
    layout.addWidget(QPushButton("Left-Most"))
    layout.addWidget(QPushButton("Center"), 1)
    layout.addWidget(QPushButton("Right-Most"), 2)
    self.setLayout(layout)

У цьому прикладі ми розміщуємо віджети горизонтально, але замінивши :code:`QHBoxLayout` на :code:`QVBoxLayout`, можна розмістити їх вертикально.  Функція :code:`addWidget` додає віджети до лейауту, а необов’язковий другий аргумент задає коефіцієнт розтягування, який визначає, скільки місця займе віджет відносно інших.

:code:`QGridLayout` має додаткові параметри, оскільки треба вказати рядок і колонку віджета, а також (необов’язково) кількість рядків і колонок, які він має займати (за замовчуванням по 1).  Ось приклад :code:`QGridLayout`:

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
   :alt: Компоновки Qt з прикладами QHBoxLayout, QVBoxLayout та QGridLayout

Для нашого аналізатора спектра ми використаємо :code:`QGridLayout` як основний лейаут, але також додаватимемо :code:`QHBoxLayout`, щоб розміщувати віджети горизонтально в певних комірках сітки.  Ви можете вкладати лейаути, просто створивши новий і додавши його до батьківського, наприклад:

.. code-block:: python

    layout = QGridLayout()
    self.setLayout(layout)
    inner_layout = QHBoxLayout()
    layout.addLayout(inner_layout)

*******************
:code:`QPushButton`
*******************

Перший віджет, який ми розглянемо — :code:`QPushButton`, проста кнопка, на яку можна натискати.  Ми вже бачили, як створити :code:`QPushButton` і під'єднати її сигнал :code:`clicked` до функції зворотного виклику.  :code:`QPushButton` має також сигнали :code:`pressed`, :code:`released` та :code:`toggled`.  Сигнал :code:`toggled` випромінюється, коли кнопку позначають або знімають позначку, і корисний для кнопок-перемикачів.  Серед властивостей :code:`QPushButton` — :code:`text`, :code:`icon` і :code:`checkable`.  Також є метод :code:`click()`, який імітує натискання.  У нашому аналізаторі ми використовуватимемо кнопки, щоб запускати автоматичне масштабування графіків за поточними даними.  Оскільки ми вже бачили :code:`QPushButton`, не заглиблюватимемось, деталі дивіться в `документації QPushButton <https://doc.qt.io/qtforpython/PySide6/QtWidgets/QPushButton.html>`_.

***************
:code:`QSlider`
***************

:code:`QSlider` — це віджет, який дозволяє користувачу обрати значення з певного діапазону.  Він має властивості :code:`minimum`, :code:`maximum`, :code:`value` та :code:`orientation`.  Серед сигналів — :code:`valueChanged`, :code:`sliderPressed` та :code:`sliderReleased`.  Також є метод :code:`setValue()`, який встановлює значення повзунка; ми використовуватимемо його часто.  Документацію можна знайти `тут <https://doc.qt.io/qtforpython/PySide6/QtWidgets/QSlider.html>`_.

У нашому застосунку аналізатора спектра ми використовуватимемо :code:`QSlider` для налаштування центральної частоти та підсилення SDR.  Ось фрагмент кінцевого коду, який створює повзунок підсилення:

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

Дуже важливо пам’ятати, що :code:`QSlider` працює з цілими числами, тож, задаючи діапазон 0…73, ми дозволяємо повзунку обирати лише цілі значення.  :code:`setTickInterval(2)` — це суто візуальний ефект.  Саме тому ми використовуємо кілогерци як одиниці для повзунка частоти, щоб отримати крок у 1 кГц.

У середині коду ви, мабуть, помітили створення :code:`QLabel`.  Це просто текстова мітка, але щоб вона показувала поточне значення повзунка, нам потрібно створити слот (тобто функцію зворотного виклику), який оновлює текст.  Ми з’єднуємо цей зворотний виклик із сигналом :code:`sliderMoved`, який автоматично випромінюється під час переміщення повзунка.  Також ми викликаємо функцію один раз, щоб ініціалізувати мітку поточним значенням (у нашому випадку 50).  Крім того, треба під’єднати :code:`sliderMoved` до слота в робочому потоці, який оновить підсилення SDR (пам’ятайте, ми не хочемо керувати SDR чи виконувати DSP у головному GUI-потоці).  Цю функцію ми розглянемо пізніше.

*****************
:code:`QComboBox`
*****************

:code:`QComboBox` — це випадаючий список, який дозволяє користувачу обрати елемент зі списку.  Він має властивості :code:`currentText`, :code:`currentIndex` та :code:`count`.  Серед сигналів — :code:`currentTextChanged`, :code:`currentIndexChanged` та :code:`activated`.  Також є метод :code:`addItem()`, який додає елемент до списку, та :code:`insertItem()`, що вставляє елемент у певну позицію, хоча ми їх не використовуватимемо в нашому прикладі.  Документація доступна `тут <https://doc.qt.io/qtforpython/PySide6/QtWidgets/QComboBox.html>`_.

У нашому аналізаторі спектра ми використовуємо :code:`QComboBox`, щоб обирати частоту дискретизації зі списку, який ми заздалегідь визначили.  На початку коду ми задаємо можливі частоти як :code:`sample_rates = [56, 40, 20, 10, 5, 2, 1, 0.5]`.  У :code:`__init__` :code:`MainWindow` створюємо :code:`QComboBox` так:

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

Основна відмінність від повзунка — це виклик :code:`addItems()`, куди передаємо список рядків як опції, та :code:`setCurrentIndex()`, який задає початкове значення.

****************
Лямбда-функції
****************

Згадайте фрагмент коду вище:

.. code-block:: python

    def update_sample_rate_label(val):
        sample_rate_label.setText("Sample Rate: " + str(sample_rates[val]) + " MHz")
    sample_rate_combobox.currentIndexChanged.connect(update_sample_rate_label)

Ми створюємо функцію з одним рядком коду всередині й передаємо цю функцію (адже функції в Python — теж об’єкти) у :code:`connect()`.  Щоб спростити, перепишемо цей шаблон базовою Python-нотацією:

.. code-block:: python

    def my_function(x):
        print(x)
    y.call_that_takes_in_function_obj(my_function)

У цьому випадку у нас є функція з одним рядком коду, і ми посилаємось на неї лише один раз — коли передаємо у :code:`connect`.  У таких ситуаціях можна використати лямбда-функцію — спосіб визначити функцію в одному рядку.  Ось попередній код, переписаний з лямбда-функцією:

.. code-block:: python

    y.call_that_takes_in_function_obj(lambda x: print(x))

Якщо ви не працювали з лямбда-функціями, це може виглядати незвично, і користуватися ними не обов’язково, але вони забирають два рядки коду та роблять його компактнішим.  Синтаксис такий: після слова «lambda» задаємо тимчасові імена аргументів, а після двокрапки — код, який їх обробляє.  Підтримується кілька аргументів через кому або навіть відсутність аргументів (:code:`lambda : <code>`).  Як вправу, спробуйте переписати функцію :code:`update_sample_rate_label` вище за допомогою лямбда-функції.

***********************
PlotWidget із PyQtGraph
***********************

:code:`PlotWidget` у PyQtGraph — це віджет Qt для побудови 1D-графіків, подібно до :code:`plt.plot(x,y)` у Matplotlib.  Ми використовуватимемо його для графіків у часовій та частотній (PSD) областях, хоча він також підходить для IQ-графіків (яких у нашому аналізаторі немає).  Для цікавих читачів: PlotWidget є підкласом `QGraphicsView <https://doc.qt.io/qtforpython-5/PySide2/QtWidgets/QGraphicsView.html>`_, віджета для відображення вмісту `QGraphicsScene <https://doc.qt.io/qtforpython-5/PySide2/QtWidgets/QGraphicsScene.html#PySide2.QtWidgets.PySide2.QtWidgets.QGraphicsScene>`_, яка є поверхнею для роботи з великою кількістю 2D-графічних елементів у Qt.  Але важливо знати, що PlotWidget — це просто віджет, який містить один `PlotItem <https://pyqtgraph.readthedocs.io/en/latest/api_reference/graphicsItems/plotitem.html#pyqtgraph.PlotItem>`_, тож найкраще звертатися до документації PlotItem: `<https://pyqtgraph.readthedocs.io/en/latest/api_reference/graphicsItems/plotitem.html>`_.  PlotItem містить ViewBox для відображення даних, а також AxisItem та підписи, як ви й очікували.

Найпростіший приклад використання PlotWidget виглядає так (код має бути доданий у :code:`__init__` :code:`MainWindow`):

.. code-block:: python

    import pyqtgraph as pg
    plotWidget = pg.plot(title="My Title")
    plotWidget.plot(x, y)

де x та y зазвичай є масивами NumPy, так само, як і у Matplotlib :code:`plt.plot()`.  Однак це статичний графік, дані не змінюються.  У нашому аналізаторі ми хочемо оновлювати дані в робочому потоці, тож, ініціалізуючи графік, можна поки що не передавати дані, а лише налаштувати його.  Ось як ми ініціалізуємо графік у часовій області в нашому застосунку:

.. code-block:: python

    # Time plot
    time_plot = pg.PlotWidget(labels={'left': 'Amplitude', 'bottom': 'Time [microseconds]'})
    time_plot.setMouseEnabled(x=False, y=True)
    time_plot.setYRange(-1.1, 1.1)
    time_plot_curve_i = time_plot.plot([])
    time_plot_curve_q = time_plot.plot([])
    layout.addWidget(time_plot, 1, 0)

Бачимо, що ми створюємо два графіки/криві: одну для I, іншу для Q.  Інший код має бути зрозумілим.  Щоб оновлювати графік, нам потрібен слот (функція зворотного виклику) у :code:`__init__` :code:`MainWindow`:

.. code-block:: python

    def time_plot_callback(samples):
        time_plot_curve_i.setData(samples.real)
        time_plot_curve_q.setData(samples.imag)

Ми з'єднаємо цей слот із сигналом робочого потоку, який випромінюється, коли доступні нові вибірки, як показано далі.

Останнє, що ми зробимо в :code:`__init__` :code:`MainWindow`, — додамо кілька кнопок праворуч від графіка для автоматичного масштабування.  Одна кнопка встановить діапазон за поточними мінімумом/максимумом, інша задасть межі -1.1…1.1 (обмеження АЦП багатьох SDR із 10% запасом).  Ми створимо внутрішній лейаут, конкретно QVBoxLayout, щоб вертикально розмістити ці кнопки.  Ось код, який додає кнопки:

.. code-block:: python

    # Time plot auto range buttons
...
    def run(self):
        if sdr_type == "pluto":
            samples = sdr.rx()/2**11 # Receive samples
        elif sdr_type == "usrp":
            streamer.recv(recv_buffer, metadata)
            samples = recv_buffer[0] # will be np.complex64
        elif sdr_type == "sim":
            tone = np.exp(2j*np.pi*self.sample_rate*0.1*np.arange(fft_size)/self.sample_rate)
            noise = np.random.randn(fft_size) + 1j*np.random.randn(fft_size)
            samples = self.gain*tone*0.02 + 0.1*noise
            # Truncate to -1 to +1 to simulate ADC bit limits
            np.clip(samples.real, -1, 1, out=samples.real)
            np.clip(samples.imag, -1, 1, out=samples.imag)

        ...

Як бачите, для змодельованого режиму ми генеруємо тон із білим шумом і обмежуємо вибірки в діапазоні -1…+1.

Тепер перейдемо до DSP!  Ми знаємо, що нам потрібне FFT для графіка частотної області та спектрограми.  Насправді ми можемо використати PSD для одного набору вибірок як один рядок спектрограми, тож достатньо зсунути спектрограму/«водоспад» на один рядок і додати новий рядок знизу (або зверху — неважливо).  Для кожного оновлення графіків ми випромінюємо сигнал із даними для відображення.  Ми також випромінюємо сигнал завершення :code:`run()`, щоб GUI негайно запускав його знову.  Загалом, це не так уже й багато коду:

.. code-block:: python

        ...

        self.time_plot_update.emit(samples[0:time_plot_samples])

        PSD = 10.0*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples)))**2/fft_size)
        self.PSD_avg = self.PSD_avg * 0.99 + PSD * 0.01
        self.freq_plot_update.emit(self.PSD_avg)

        self.spectrogram[:] = np.roll(self.spectrogram, 1, axis=1) # shifts waterfall 1 row
        self.spectrogram[:,0] = PSD # fill last row with new fft results
        self.waterfall_plot_update.emit(self.spectrogram)

        self.end_of_run.emit() # emit the signal to keep the loop going
        # end of run()

Зауважте, що ми не надсилаємо на графік часу весь пакет вибірок, адже це надто багато точок, натомість відправляємо перші 500 (це налаштовується на початку скрипта, тут не показано).  Для графіка PSD ми використовуємо ковзне середнє: зберігаємо попередній PSD і додаємо до нього 1% нового.  Це простий спосіб згладити графік.  Порядок викликів :code:`emit()` не має значення — всі вони могли бути наприкінці :code:`run()`.

***********************
Повний код фінального прикладу
***********************

До цього моменту ми розглядали окремі фрагменти застосунку аналізатора спектра, а тепер подивимося на повний код і спробуємо його запустити.  Наразі підтримуються PlutoSDR, USRP або режим моделювання.  Якщо у вас немає Pluto чи USRP, залиште код як є, і він використає режим моделювання, інакше змініть :code:`sdr_type`.  У режимі моделювання, якщо збільшити підсилення до максимуму, ви помітите, що сигнал у часовій області зрізається, що призводить до появи спурів у частотній області.

Сміливо використовуйте цей код як відправну точку для власного SDR-додатку реального часу!  Нижче також наведено анімацію роботи застосунку: Pluto використовується для перегляду стільникового діапазону 750 МГц, а потім — Wi-Fi на 2.4 ГГц.  Версію вищої якості можна переглянути на YouTube `тут <https://youtu.be/hvofiY3Q_yo>`_.

.. image:: ../_images/pyqt_animation.gif
   :scale: 100 %
   :align: center
   :alt: Анімація роботи застосунку аналізатора спектра PyQt

Відомі вади (щоб допомогти їх виправити, `відредагуйте цей файл <https://github.com/777arc/PySDR/edit/master/figure-generating-scripts/pyqt_example.py>`_):

#. Вісь x «водоспаду» не оновлюється під час зміни центральної частоти (натомість оновлюється графік PSD)

Повний код:

.. code-block:: python

    from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal, QObject, QTimer
    from PyQt6.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QSlider, QLabel, QHBoxLayout, QVBoxLayout, QPushButton, QComboBox  # tested with PyQt6==6.7.0
    import pyqtgraph as pg # tested with pyqtgraph==0.13.7
    import numpy as np
    import time
    import signal # lets control-C actually close the app

    # Defaults
    fft_size = 4096 # determines buffer size
    num_rows = 200
    center_freq = 750e6
    sample_rates = [56, 40, 20, 10, 5, 2, 1, 0.5] # MHz
    sample_rate = sample_rates[0] * 1e6
    time_plot_samples = 500
    gain = 50 # 0 to 73 dB. int

    sdr_type = "sim" # or "usrp" or "pluto"

    # Init SDR
    if sdr_type == "pluto":
        import adi
        sdr = adi.Pluto("ip:192.168.1.10")
        sdr.rx_lo = int(center_freq)
        sdr.sample_rate = int(sample_rate)
        sdr.rx_rf_bandwidth = int(sample_rate*0.8) # antialiasing filter bandwidth
        sdr.rx_buffer_size = int(fft_size)
        sdr.gain_control_mode_chan0 = 'manual'
        sdr.rx_hardwaregain_chan0 = gain # dB
    elif sdr_type == "usrp":
        import uhd
        #usrp = uhd.usrp.MultiUSRP(args="addr=192.168.1.10")
        usrp = uhd.usrp.MultiUSRP(args="addr=192.168.1.201")
        usrp.set_rx_rate(sample_rate, 0)
        usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(center_freq), 0)
        usrp.set_rx_gain(gain, 0)

        # Set up the stream and receive buffer
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [0]
        metadata = uhd.types.RXMetadata()
        streamer = usrp.get_rx_stream(st_args)
        recv_buffer = np.zeros((1, fft_size), dtype=np.complex64)

        # Start Stream
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        stream_cmd.stream_now = True
        streamer.issue_stream_cmd(stream_cmd)

        def flush_buffer():
            for _ in range(10):
                streamer.recv(recv_buffer, metadata)

    class SDRWorker(QObject):
        def __init__(self):
            super().__init__()
            self.gain = gain
            self.sample_rate = sample_rate
            self.freq = 0 # in kHz, to deal with QSlider being ints and with a max of 2 billion
            self.spectrogram = -50*np.ones((fft_size, num_rows))
            self.PSD_avg = -50*np.ones(fft_size)

        # PyQt Signals
        time_plot_update = pyqtSignal(np.ndarray)
        freq_plot_update = pyqtSignal(np.ndarray)
        waterfall_plot_update = pyqtSignal(np.ndarray)
        end_of_run = pyqtSignal() # happens many times a second

        # PyQt Slots
        def update_freq(self, val): # TODO: WE COULD JUST MODIFY THE SDR IN THE GUI THREAD
            print("Updated freq to:", val, 'kHz')
            if sdr_type == "pluto":
                sdr.rx_lo = int(val*1e3)
            elif sdr_type == "usrp":
                usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(val*1e3), 0)
                flush_buffer()

        def update_gain(self, val):
            print("Updated gain to:", val, 'dB')
            self.gain = val
            if sdr_type == "pluto":
                sdr.rx_hardwaregain_chan0 = val
            elif sdr_type == "usrp":
                usrp.set_rx_gain(val, 0)
                flush_buffer()

        def update_sample_rate(self, val):
            print("Updated sample rate to:", sample_rates[val], 'MHz')
            if sdr_type == "pluto":
                sdr.sample_rate = int(sample_rates[val] * 1e6)
                sdr.rx_rf_bandwidth = int(sample_rates[val] * 1e6 * 0.8)
            elif sdr_type == "usrp":
                usrp.set_rx_rate(sample_rates[val] * 1e6, 0)
                flush_buffer()

        # Main loop
        def run(self):
            start_t = time.time()

            if sdr_type == "pluto":
                samples = sdr.rx()/2**11 # Receive samples
            elif sdr_type == "usrp":
                streamer.recv(recv_buffer, metadata)
                samples = recv_buffer[0] # will be np.complex64
            elif sdr_type == "sim":
                tone = np.exp(2j*np.pi*self.sample_rate*0.1*np.arange(fft_size)/self.sample_rate)
                noise = np.random.randn(fft_size) + 1j*np.random.randn(fft_size)
                samples = self.gain*tone*0.02 + 0.1*noise
                # Truncate to -1 to +1 to simulate ADC bit limits
                np.clip(samples.real, -1, 1, out=samples.real)
                np.clip(samples.imag, -1, 1, out=samples.imag)

            self.time_plot_update.emit(samples[0:time_plot_samples])

            PSD = 10.0*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples)))**2/fft_size)
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
            auto_range_button2 = QPushButton('-1 to +1\n(ADC limits)')
            auto_range_button2.clicked.connect(lambda : time_plot.setYRange(-1.1, 1.1))
            time_plot_auto_range_layout.addWidget(auto_range_button2)

            # Freq plot
            freq_plot = pg.PlotWidget(labels={'left': 'PSD', 'bottom': 'Frequency [MHz]'})
            freq_plot.setMouseEnabled(x=False, y=True)
            freq_plot_curve = freq_plot.plot([])
            freq_plot.setXRange(center_freq/1e6 - sample_rate/2e6, center_freq/1e6 + sample_rate/2e6)
            freq_plot.setYRange(-30, 20)
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
            imageitem.setLevels((-30, 20)) # needs to come after colorbar is created for some reason
            waterfall_layout.addWidget(colorbar)

            # Waterfall auto range button
            auto_range_button = QPushButton('Auto Range\n(-2σ to +2σ)')
            def update_colormap():
                imageitem.setLevels((self.spectrogram_min, self.spectrogram_max))
                colorbar.setLevels(self.spectrogram_min, self.spectrogram_max)
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

            # Sample rate dropdown using QComboBox
            sample_rate_combobox = QComboBox()
            sample_rate_combobox.addItems([str(x) + ' MHz' for x in sample_rates])
            sample_rate_combobox.setCurrentIndex(0) # should match the default at the top
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
                f = np.linspace(freq_slider.value()*1e3 - worker.sample_rate/2.0, freq_slider.value()*1e3 + worker.sample_rate/2.0, fft_size) / 1e6
                freq_plot_curve.setData(f, PSD_avg)
                freq_plot.setXRange(freq_slider.value()*1e3/1e6 - worker.sample_rate/2e6, freq_slider.value()*1e3/1e6 + worker.sample_rate/2e6)

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
    window.show() # Windows are hidden by default
    signal.signal(signal.SIGINT, signal.SIG_DFL) # this lets control-C actually close the app
    app.exec() # Start the event loop

    if sdr_type == "usrp":
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        streamer.issue_stream_cmd(stream_cmd)
