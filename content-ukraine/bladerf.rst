.. _bladerf-chapter:

##################
BladeRF у Python
##################

bladeRF 2.0 (a.k.a. bladeRF 2.0 micro) від компанії `Nuand <https://www.nuand.com>`_ — це SDR на базі USB 3.0 з двома каналами прийому, двома каналами передавання, робочим діапазоном налаштування від 47 МГц до 6 ГГц та можливістю дискретизації до 61 МГц або навіть до 122 МГц після модифікації. Він використовує радіочастотну інтегральну схему (RFIC) AD9361 так само, як USRP B210 та PlutoSDR, тому його радіочастотні характеристики будуть подібними. bladeRF 2.0 було випущено у 2021 році, він зберіг компактний форм-фактор 2.5" x 4.5" і постачається з двома варіантами FPGA (xA4 та xA9). Хоча цей розділ зосереджується на bladeRF 2.0, більшість наведеного коду також застосовна до оригінального bladeRF, який `з'явився у 2013 році <https://www.kickstarter.com/projects/1085541682/bladerf-usb-30-software-defined-radio>`_.

.. image:: ../_images/bladeRF_micro.png
   :scale: 35 %
   :align: center
   :alt: bladeRF 2.0 glamour shot

********************************
Архітектура bladeRF
********************************

На високому рівні bladeRF 2.0 побудований на базі RFIC AD9361 у поєднанні з FPGA Cyclone V (або 49 kLE :code:`5CEA4`, або 301 kLE :code:`5CEA9`) та контролером Cypress FX3 USB 3.0 із ядром ARM9 на частоті 200 МГц, що працює під керуванням спеціальної прошивки. Нижче наведена блок-схема bladeRF 2.0:

.. image:: ../_images/bladeRF-2.0-micro-Block-Diagram-4.png
   :scale: 80 %
   :align: center
   :alt: bladeRF 2.0 block diagram

FPGA керує RFIC, виконує цифрову фільтрацію та формує пакети для передавання через USB (серед іншого). `Вихідний код <https://github.com/Nuand/bladeRF/tree/master/hdl>`_ для образу FPGA написаний VHDL і вимагає безкоштовного програмного забезпечення Quartus Prime Lite для компіляції власних образів. Готові образи доступні `тут <https://www.nuand.com/fpga_images/>`_.

`Вихідний код <https://github.com/Nuand/bladeRF/tree/master/fx3_firmware>`_ для прошивки Cypress FX3 з відкритим кодом і містить реалізацію:

1. Завантаження образу FPGA
2. Передавання IQ-вибірок між FPGA та хостом через USB 3.0
3. Керування GPIO FPGA через UART

З точки зору потоку сигналу є два канали прийому та два канали передавання, і кожен канал має низькочастотний та високочастотний вхід/вихід до RFIC залежно від обраного діапазону. Саме тому між RFIC та SMA-роз'ємами потрібен електронний ВКП перемикач (single pole double throw, SPDT). Bias tee — це вбудована схема, що подає ~4,5 В постійного струму на SMA-роз'єм і дозволяє зручно живити зовнішній підсилювач чи інші ВЧ-компоненти. Ця додаткова напруга постійного струму знаходиться на радіочастотній стороні SDR, тому вона не заважає базовій роботі прийому/передавання.

JTAG — це інтерфейс налагодження, який дозволяє тестувати та перевіряти проєкти під час розробки.

Наприкінці цього розділу ми обговоримо генератор VCTCXO, PLL та розширювальний порт.

********************************
Налаштування програмного та апаратного забезпечення
********************************

Ubuntu (або Ubuntu у WSL)
#############################

В Ubuntu та інших системах на базі Debian можна встановити програмне забезпечення bladeRF наступними командами:

.. code-block:: bash

 sudo apt update
 sudo apt install cmake python3-pip libusb-1.0-0
 cd ~
 git clone --depth 1 https://github.com/Nuand/bladeRF.git
 cd bladeRF/host
 mkdir build && cd build
 cmake ..
 make -j8
 sudo make install
 sudo ldconfig
 cd ../libraries/libbladeRF_bindings/python
 sudo python3 setup.py install

Це встановить бібліотеку libbladerf, Python-біндінги, інструменти командного рядка bladeRF, засіб завантаження прошивки та засіб завантаження бітстрімів FPGA. Щоб перевірити встановлену версію бібліотеки, скористайтеся :code:`bladerf-tool version` (цей посібник написано для libbladeRF версії v2.5.0).

Якщо ви використовуєте Ubuntu через WSL, на стороні Windows необхідно переслати USB-пристрій bladeRF у WSL. Спочатку встановіть останній `msi-дистрибутив утиліти usbipd <https://github.com/dorssel/usbipd-win/releases>`_ (цей посібник припускає, що у вас usbipd-win 4.0.0 або новіший), потім відкрийте PowerShell від імені адміністратора та виконайте:

.. code-block:: bash

    usbipd list
    # (знайдіть BUSID з позначкою bladeRF 2.0 і підставте його в команду нижче)
    usbipd bind --busid 1-23
    usbipd attach --wsl --busid 1-23

У WSL ви маєте змогу виконати :code:`lsusb` і побачити новий елемент :code:`Nuand LLC bladeRF 2.0 micro`. Зауважте, що до команди :code:`usbipd attach` можна додати прапорець :code:`--auto-attach`, якщо потрібне автоматичне повторне підключення.

(Може не знадобитися) Для нативного Linux і WSL потрібно встановити правила udev, щоб уникнути помилок доступу:

.. code-block::

 sudo nano /etc/udev/rules.d/88-nuand.rules

та вставити такий рядок:

.. code-block::

 ATTRS{idVendor}=="2cf0", ATTRS{idProduct}=="5250", MODE="0666"

Щоб зберегти та вийти з nano, натисніть: control-o, потім Enter, потім control-x. Для оновлення udev виконайте:

.. code-block:: bash

    sudo udevadm control --reload-rules && sudo udevadm trigger

Якщо ви використовуєте WSL і бачите повідомлення :code:`Failed to send reload request: No such file or directory`, це означає, що служба udev не запущена, і вам потрібно виконати :code:`sudo nano /etc/wsl.conf` та додати рядки:

.. code-block:: bash

 [boot]
 command="service udev start"

Потім перезавантажте WSL такою командою в PowerShell з правами адміністратора: :code:`wsl.exe --shutdown`.

Від'єднайте та знову під'єднайте bladeRF (користувачам WSL доведеться повторно приєднати пристрій) і перевірте права доступу командами:

.. code-block:: bash

 bladerf-tool probe
 bladerf-tool info

Ви зрозумієте, що все спрацювало, якщо ваш bladeRF 2.0 буде в списку і ви **не** побачите повідомлення :code:`Found a bladeRF via VID/PID, but could not open it due to insufficient permissions`. Якщо все вдалося, зверніть увагу на версії FPGA та прошивки, що виводяться.

(Опційно) Встановіть найновішу прошивку та образи FPGA (відповідно v2.4.0 та v0.15.0 на момент написання посібника) командами:

.. code-block:: bash

 cd ~/Downloads
 wget https://www.nuand.com/fx3/bladeRF_fw_latest.img
 bladerf-tool flash_fw bladeRF_fw_latest.img

 # для xA4 використовуйте:
 wget https://www.nuand.com/fpga/hostedxA4-latest.rbf
 bladerf-tool flash_fpga hostedxA4-latest.rbf

 # для xA9 використовуйте:
 wget https://www.nuand.com/fpga/hostedxA9-latest.rbf
 bladerf-tool flash_fpga hostedxA9-latest.rbf

Від'єднайте та знову під'єднайте bladeRF, щоб перезапустити живлення.

Тепер перевіримо працездатність, прийнявши 1 млн вибірок у FM-діапазоні радіо з частотою дискретизації 10 МГц у файл /tmp/samples.sc16:

.. code-block:: bash

 bladerf-tool rx --num-samples 1000000 /tmp/samples.sc16 100e6 10e6

Кілька повідомлень :code:`Hit stall for buffer` наприкінці — це нормально, а успішним результатом буде поява файлу /tmp/samples.sc16 обсягом 4 МБ.

Нарешті, перевіримо Python API:

.. code-block:: bash

 python3
 import bladerf
 bladerf.BladeRF()
 exit()

Ви зрозумієте, що все працює, якщо побачите щось на кшталт :code:`<BladeRF(<DevInfo(...)>)>` і не з'являться попередження чи помилки.

Windows і macOS
###################

Користувачам Windows (які не бажають використовувати WSL) див. https://github.com/Nuand/bladeRF/wiki/Getting-Started%3A-Windows, а користувачам macOS — https://github.com/Nuand/bladeRF/wiki/Getting-started:-Mac-OSX.

********************************
Основи Python API bladeRF
********************************

Почнімо з опитування bladeRF щодо корисної інформації за допомогою такого скрипта. **Не називайте свій скрипт bladerf.py**, інакше він конфліктуватиме з самим модулем Python bladeRF!

.. code-block:: python

 from bladerf import _bladerf
 import numpy as np
 import matplotlib.pyplot as plt

 sdr = _bladerf.BladeRF()

 print("Device info:", _bladerf.get_device_list()[0])
 print("libbladeRF version:", _bladerf.version()) # v2.5.0
 print("Firmware version:", sdr.get_fw_version()) # v2.4.0
 print("FPGA version:", sdr.get_fpga_version())   # v0.15.0

 rx_ch = sdr.Channel(_bladerf.CHANNEL_RX(0)) # задайте 0 або 1
 print("sample_rate_range:", rx_ch.sample_rate_range)
 print("bandwidth_range:", rx_ch.bandwidth_range)
 print("frequency_range:", rx_ch.frequency_range)
 print("gain_modes:", rx_ch.gain_modes)
 print("manual gain range:", sdr.get_gain_range(_bladerf.CHANNEL_RX(0))) # канал 0 або 1

Для bladeRF 2.0 xA9 результат виглядатиме приблизно так:

.. code-block:: python

    Device info: Device Information
        backend  libusb
        serial   f80a27b1010448dfb7a003ef7fa98a59
        usb_bus  2
        usb_addr 5
        instance 0
    libbladeRF version: v2.5.0 ("2.5.0-git-624994d")
    Firmware version: v2.4.0 ("2.4.0-git-a3d5c55f")
    FPGA version: v0.15.0 ("0.15.0")
    sample_rate_range: Range
        min   520834
        max   61440000
        step  2
        scale 1.0

    bandwidth_range: Range
        min   200000
        max   56000000
        step  1
        scale 1.0

    frequency_range: Range
        min   70000000
        max   6000000000
        step  2
        scale 1.0

    gain_modes: [<GainMode.Default: 0>, <GainMode.Manual: 1>, <GainMode.FastAttack_AGC: 2>, <GainMode.SlowAttack_AGC: 3>, <GainMode.Hybrid_AGC: 4>]

    manual gain range: Range
        min   -15
        max   60
        step  1
        scale 1.0

Параметр bandwidth встановлює фільтр, який використовує SDR під час прийому, тому зазвичай його задають рівним або трохи меншим за sample_rate/2. Режими підсилення важливо розуміти: SDR може працювати або в режимі ручного підсилення, коли ви задаєте значення в дБ, або в режимі автоматичного керування підсиленням (AGC), що має три налаштування (швидке, повільне, гібридне). Для задач на кшталт моніторингу спектра рекомендується ручний режим (щоб бачити, коли сигнали з'являються і зникають), а для задач на кшталт прийому конкретного сигналу, який ви очікуєте, AGC буде кориснішим, бо автоматично підбиратиме підсилення так, щоб сигнал максимально заповнював аналогово-цифровий перетворювач (ADC).

Щоб задати основні параметри SDR, додайте такий код:

.. code-block:: python

 sample_rate = 10e6
 center_freq = 100e6
 gain = 50 # від -15 до 60 дБ
 num_samples = int(1e6)

 rx_ch.frequency = center_freq
 rx_ch.sample_rate = sample_rate
 rx_ch.bandwidth = sample_rate/2
 rx_ch.gain_mode = _bladerf.GainMode.Manual
 rx_ch.gain = gain

********************************
Приймання вибірок у Python
********************************

Далі, спираючись на попередній приклад, приймемо 1 млн вибірок у FM-діапазоні з частотою дискретизації 10 МГц, як ми робили раніше. Будь-яка антена на порту RX1 має приймати FM, бо сигнал дуже потужний. Код нижче демонструє роботу синхронного потокового API bladeRF: перед початком прийому його потрібно налаштувати і створити буфер. Цикл :code:`while True:` продовжить приймати вибірки, доки не буде досягнуто потрібної кількості. Отримані вибірки зберігаються в окремому масиві numpy, щоб ми могли обробити їх після завершення циклу.

.. code-block:: python

 # Налаштування синхронного потоку
 sdr.sync_config(layout = _bladerf.ChannelLayout.RX_X1, # або RX_X2
                 fmt = _bladerf.Format.SC16_Q11, # int16
                 num_buffers    = 16,
                 buffer_size    = 8192,
                 num_transfers  = 8,
                 stream_timeout = 3500)

 # Створення приймального буфера
 bytes_per_sample = 4 # не змінюйте, завжди використовуються int16
 buf = bytearray(1024 * bytes_per_sample)

 # Увімкнення модуля
 print("Starting receive")
 rx_ch.enable = True

 # Цикл прийому
 x = np.zeros(num_samples, dtype=np.complex64) # сховище для IQ-вибірок
 num_samples_read = 0
 while True:
     if num_samples > 0 and num_samples_read == num_samples:
         break
     elif num_samples > 0:
         num = min(len(buf) // bytes_per_sample, num_samples - num_samples_read)
     else:
         num = len(buf) // bytes_per_sample
     sdr.sync_rx(buf, num) # зчитування у буфер
     samples = np.frombuffer(buf, dtype=np.int16)
     samples = samples[0::2] + 1j * samples[1::2] # перетворення у комплексний тип
     samples /= 2048.0 # масштабування до -1...1 (використовується 12-бітний ADC)
     x[num_samples_read:num_samples_read+num] = samples[0:num] # збереження буфера у масиві вибірок
     num_samples_read += num

 print("Stopping")
 rx_ch.enable = False
 print(x[0:10]) # подивіться на перші 10 IQ-вибірок
 print(np.max(x)) # якщо значення близьке до 1, ADC перевантажено і слід зменшити підсилення

Кілька повідомлень :code:`Hit stall for buffer` наприкінці — це очікувано. Останнє виведене число показує максимальну отриману вибірку; вам варто налаштувати підсилення так, щоб це значення було в діапазоні 0.5–0.8. Якщо ж воно дорівнює 0.999, приймач перевантажений/насичений, і сигнал спотворюється (у частотній області він виглядатиме розмитим).

Щоб візуалізувати отриманий сигнал, побудуймо спектрограму (детальніше про спектрограми див. :ref:`spectrogram-section`). Додайте наприкінці попереднього блоку коду:

.. code-block:: python

 # Побудова спектрограми
 fft_size = 2048
 num_rows = len(x) // fft_size # // — цілочисельне ділення з округленням донизу
 spectrogram = np.zeros((num_rows, fft_size))
 for i in range(num_rows):
     spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)
 extent = [(center_freq + sample_rate/-2)/1e6, (center_freq + sample_rate/2)/1e6, len(x)/sample_rate, 0]
 plt.imshow(spectrogram, aspect='auto', extent=extent)
 plt.xlabel("Frequency [MHz]")
 plt.ylabel("Time [s]")
 plt.show()

.. image:: ../_images/bladerf-waterfall.svg
   :align: center
   :target: ../_images/bladerf-waterfall.svg
   :alt: bladeRF spectrogram example

Кожна вертикальна хвиляста лінія — це сигнал FM-радіо. Невідомо, звідки береться пульсація праворуч; зменшення підсилення її не прибрало.

********************************
Передавання вибірок у Python
********************************

Процес передавання вибірок на bladeRF дуже схожий на прийом. Головна відмінність у тому, що потрібно згенерувати вибірки для передавання, а потім записати їх у bladeRF за допомогою методу :code:`sync_tx`, який може обробити весь пакет вибірок одразу (до ~4 млрд вибірок). Наведений нижче код демонструє, як передати простий тон і повторити його 30 разів. Тон генерується за допомогою numpy, потім масштабується до діапазону від -2048 до 2048, щоб відповідати 12-бітному цифро-аналоговому перетворювачу (DAC). Далі тон перетворюється на байти, що представляють int16, і використовується як буфер передавання. Синхронний потоковий API використовується для передавання вибірок, а цикл :code:`while True:` триватиме, доки не буде зроблено потрібну кількість повторів. Якщо бажаєте передавати вибірки з файлу, просто виконайте :code:`samples = np.fromfile('yourfile.iq', dtype=np.int16)` (або з відповідним типом даних), щоб прочитати вибірки, а потім перетворіть їх у байти за допомогою :code:`samples.tobytes()`, пам'ятаючи про діапазон DAC від -2048 до 2048.

.. code-block:: python

 from bladerf import _bladerf
 import numpy as np

 sdr = _bladerf.BladeRF()
 tx_ch = sdr.Channel(_bladerf.CHANNEL_TX(0)) # задайте 0 або 1

 sample_rate = 10e6
 center_freq = 100e6
 gain = 0 # від -15 до 60 дБ. починайте з малого, поступово збільшуйте й переконайтеся, що антена під'єднана
 num_samples = int(1e6)
 repeat = 30 # кількість повторів сигналу
 print('duration of transmission:', num_samples/sample_rate*repeat, 'seconds')

 # Генерація IQ-вибірок для передавання (у цьому випадку простого тону)
 t = np.arange(num_samples) / sample_rate
 f_tone = 1e6
 samples = np.exp(1j * 2 * np.pi * f_tone * t) # значення від -1 до +1
 samples = samples.astype(np.complex64)
 samples *= 2048.0 # масштабування до -1...1 (використовується 12-бітний DAC)
 samples = samples.view(np.int16)
 buf = samples.tobytes() # перетворення вибірок у байти та використання їх як буфера передавання

 tx_ch.frequency = center_freq
 tx_ch.sample_rate = sample_rate
 tx_ch.bandwidth = sample_rate/2
 tx_ch.gain = gain

 # Налаштування синхронного потоку
 sdr.sync_config(layout=_bladerf.ChannelLayout.TX_X1, # або TX_X2
                 fmt=_bladerf.Format.SC16_Q11, # int16
                 num_buffers=16,
                 buffer_size=8192,
                 num_transfers=8,
                 stream_timeout=3500)

 print("Starting transmit!")
 repeats_remaining = repeat - 1
 tx_ch.enable = True
 while True:
     sdr.sync_tx(buf, num_samples) # запис у bladeRF
     print(repeats_remaining)
     if repeats_remaining > 0:
         repeats_remaining -= 1
     else:
         break

 print("Stopping transmit")
 tx_ch.enable = False

Кілька повідомлень :code:`Hit stall for buffer` наприкінці — це нормально.

Щоб одночасно передавати та приймати, необхідно використовувати потоки. У такому випадку краще взяти приклад Nuand `txrx.py <https://github.com/Nuand/bladeRF/blob/624994d65c02ad414a01b29c84154260912f4e4f/host/examples/python/txrx/txrx.py>`_, який робить саме це.

***********************************
Генератори, PLL та калібрування
***********************************

Усі SDR із прямим перетворенням (у тому числі всі пристрої на AD9361, як-от USRP B2X0, Analog Devices Pluto та bladeRF) покладаються на один генератор, що забезпечує стабільну тактову частоту для радіоприймача. Будь-які зміщення чи джитер частоти цього генератора перетворюються на частотне зміщення та джитер у прийнятому або переданому сигналі. Цей генератор розташований на платі, але його можна «дисциплінувати», підключивши окремий квадратний або синусоїдальний сигнал до bladeRF через роз'єм U.FL на платі.

У bladeRF встановлено `VCTCXO Abracon <https://abracon.com/Oscillators/ASTX12_ASVTX12.pdf>`_ (керований напругою температурно-компенсований генератор) із частотою 38,4 МГц. «Температурна компенсація» означає, що він розрахований на стабільність у широкому діапазоні температур. Керування напругою означає, що частота генератора може змінюватися залежно від прикладеної напруги, і в bladeRF цю напругу подає окремий 10-бітний цифро-аналоговий перетворювач (DAC), показаний зеленим кольором на блок-схемі нижче. Це дає змогу програмно виконувати тонке підлаштування частоти генератора, і саме так ми калібруємо (тобто підлаштовуємо) VCTCXO bladeRF. На щастя, пристрої bladeRF калібруються на заводі, як ми обговоримо далі, але якщо у вас є відповідне вимірювальне обладнання, ви можете додатково відрегулювати це значення, особливо через роки, коли частота генератора може дрейфувати.

.. image:: ../_images/bladeRF-2.0-micro-Block-Diagram-4-oscillator.png
   :scale: 80 %
   :align: center
   :alt: bladeRF 2.0 glamour shot

Під час використання зовнішнього частотного еталону (який може мати майже будь-яку частоту до 300 МГц) референсний сигнал подається безпосередньо на PLL `Analog Devices ADF4002 <http://www.analog.com/en/adf4002>`_, що встановлена на bladeRF. Ця PLL захоплює еталонний сигнал і подає на VCTCXO (позначено синім) сигнал, пропорційний різниці частоти та фази між (масштабованим) еталонним входом і виходом VCTCXO. Коли PLL захоплює синхронізм, цей сигнал між PLL і VCTCXO стає стабільною постійною напругою, що підтримує вихід VCTCXO на «точно» 38,4 МГц (за умови, що еталон точний) і фазово синхронізує його з еталонним сигналом. Під час використання зовнішнього еталона потрібно ввімкнути :code:`clock_ref` (через Python або CLI) і задати частоту еталонного входу (:code:`refin_freq`), яка за замовчуванням дорівнює 10 МГц. Причини використовувати зовнішній еталон — це покращена точність частоти та можливість синхронізувати кілька SDR одним еталоном.

Для кожного bladeRF значення підстроювання VCTCXO DAC калібрується на заводі з точністю до 1 Гц на частоті 38,4 МГц при кімнатній температурі, і ви можете ввести свій серійний номер на `цій сторінці <https://www.nuand.com/calibration/>`_, щоб дізнатися заводське значення (серійний номер вказано на платі або доступний через :code:`bladerf-tool probe`). За словами Nuand, нова плата має точність набагато кращу за 0.5 ppm і, ймовірно, ближче до 0.1 ppm. Якщо у вас є обладнання для вимірювання точності частоти або ви хочете встановити заводське значення, використовуйте команди:

.. code-block:: bash

 $ bladeRF-cli -i
 bladeRF> flash_init_cal 301 0x2049

Замінивши :code:`301` на розмір вашого bladeRF і :code:`0x2049` на шістнадцяткове представлення вашого значення підстроювання VCTCXO DAC. Щоб зміни набули чинності, потрібно перезапустити живлення.

***********************************
Дискретизація на 122 МГц
***********************************

Незабаром!

***********************************
Порти розширення
***********************************

bladeRF 2.0 містить порт розширення з використанням роз'єма BSH-030. Докладніша інформація про використання цього порту з'явиться пізніше!

********************************
Додаткові матеріали
********************************

#. `Wiki bladeRF <https://github.com/Nuand/bladeRF/wiki>`_
#. `Приклад Nuand txrx.py <https://github.com/Nuand/bladeRF/blob/master/host/examples/python/txrx/txrx.py>`_
