.. _hackrf-chapter:

########################
Python 玩转 HackRF One
########################

Great Scott Gadgets 推出的 `HackRF One <https://greatscottgadgets.com/hackrf/one/>`_ 是一款基于 USB 2.0 的 SDR，能够在 1 MHz 到 6 GHz 之间进行发射或接收，采样率范围为 2 到 20 MHz。
它于 2014 年发布，并在这些年里经历了几次小幅改进。
它是少数能够下探到 1 MHz 的低成本可发射 SDR 之一，因此除了更高频段的玩法之外，也非常适合 HF 应用（例如业余无线电）。
它的最大发射功率 15 dBm 也高于大多数其他 SDR；完整的发射功率规格请参见 `这个页面 <https://hackrf.readthedocs.io/en/latest/faq.html#what-is-the-transmit-power-of-hackrf>`_ 。
它采用半双工工作方式，这意味着任意时刻它只能处于发射或接收其中一种模式，并且它使用的是 8 位 ADC/DAC。

.. image:: ../_images/hackrf1.jpeg
   :scale: 60 %
   :align: center
   :alt: HackRF One

********************************
HackRF 架构
********************************

HackRF 的核心是 Analog Devices 的 MAX2839 芯片，它本质上是一颗 2.3 GHz 到 2.7 GHz 的收发器，最初是为 WiMAX 设计的。
它还搭配了一颗 MAX5864 射频前端芯片（本质上就是 ADC 和 DAC），以及一颗 RFFC5072 宽带合成器/VCO（用于在频率上对信号进行上变频和下变频）。
这与大多数其他低成本 SDR 不同，因为后者通常会使用一颗被称为 RFIC 的单芯片方案。
除了设置 RFFC5072 内部生成的频率之外，我们后续会调整的其他参数，比如衰减和模拟滤波，基本都发生在 MAX2839 中。
HackRF 没有像许多 SDR 那样使用 FPGA 或片上系统（SoC），而是使用了一颗复杂可编程逻辑器件（CPLD）作为简单的胶合逻辑，以及一颗基于 ARM 的 LPC4320 微控制器来完成所有板载 DSP 和通过 USB 与主机的交互（包括双向 IQ 样本传输以及 SDR 参数控制）。
下面这张来自 Great Scott Gadgets 的精美方框图展示了最新版 HackRF One 的架构：

.. image:: ../_images/hackrf_block_diagram.webp
   :align: center
   :alt: HackRF One 方框图
   :target: ../_images/hackrf_block_diagram.webp

HackRF One 具有很强的可扩展性，也很适合拿来折腾。
塑料外壳内部有四组排针（P9、P20、P22 和 P28），具体说明可以 `在这里查看 <https://hackrf.readthedocs.io/en/latest/expansion_interface.html>`_ 。
其中，8 路 GPIO 和 4 路 ADC 输入位于 P20 排针，而 SPI、I2C 和 UART 位于 P22 排针。
P28 排针则可以通过触发输入和输出，与其他设备（例如 TR 开关、外部放大器，或另一台 HackRF）同步触发发射/接收操作，其延迟小于一个采样周期。

.. image:: ../_images/hackrf2.jpeg
   :scale: 50 %
   :align: center
   :alt: HackRF One PCB

HackRF 用于本振和 ADC/DAC 的时钟，既可以来自板上的 25 MHz 振荡器，也可以来自通过 SMA 输入的外部 10 MHz 参考时钟。
无论使用哪一种时钟源，HackRF 都会在 CLKOUT 上输出一个 10 MHz 时钟信号；这是一个标准的 3.3V、10 MHz 方波，面向高阻抗负载。
CLKIN 接口则用于输入类似的 10 MHz、3.3V 方波；当检测到输入时钟时，HackRF One 会在开始发射或接收操作时切换为使用这个外部输入，而不是内部晶振。

********************************
HackRF 的软件与硬件配置
********************************

软件安装流程分为两步：首先安装 Great Scott Gadgets 提供的 HackRF 主库，然后安装 Python API。

安装 HackRF 主库
#############################

以下步骤已经在 Ubuntu 22.04 上验证可用（使用的是 2025 年 3 月的 commit hash `17f3943`）：

.. code-block:: bash

    git clone https://github.com/greatscottgadgets/hackrf.git
    cd hackrf
    git checkout 17f3943
    cd host
    mkdir build
    cd build
    cmake ..
    make
    sudo make install
    sudo ldconfig
    sudo cp /usr/local/bin/hackrf* /usr/bin/.

安装好 :code:`hackrf` 后，你将能够使用以下工具：

* :code:`hackrf_info` - 读取 HackRF 设备信息，例如序列号和固件版本。
* :code:`hackrf_transfer` - 使用 HackRF 发送和接收信号。输入/输出文件采用 8 位有符号正交采样格式。
* :code:`hackrf_sweep` - 命令行频谱分析仪。
* :code:`hackrf_clock` - 读取和写入时钟输入/输出配置。
* :code:`hackrf_operacake` - 配置连接到 HackRF 的 Opera Cake 天线切换器。
* :code:`hackrf_spiflash` - 用于给 HackRF 刷写新固件，参见 `HackRF 官方固件更新说明 <https://hackrf.readthedocs.io/en/stable/updating_firmware.html>`_ 。
* :code:`hackrf_debug` - 用于读取和写入寄存器以及其他底层调试配置。

如果你是通过 WSL 使用 Ubuntu，那么在 Windows 侧需要先把 HackRF 的 USB 设备转发到 WSL。
首先安装最新版本的 `usbipd utility msi <https://github.com/dorssel/usbipd-win/releases>`_ （本文默认你使用的是 usbipd-win 4.0.0 或更高版本），然后以管理员模式打开 PowerShell 并运行：

.. code-block:: bash

    usbipd list
    # 找到标记为 HackRF One 的 BUSID，并代入下面两条命令
    usbipd bind --busid 1-10
    usbipd attach --wsl --busid 1-10

在 WSL 这边，你应该可以通过 :code:`lsusb` 看到一个新设备项，名称类似 :code:`Great Scott Gadgets HackRF One`。
注意，如果你希望它自动重新连接，可以在 :code:`usbipd attach` 命令后加上 :code:`--auto-attach` 参数。
最后，你还需要通过下面的命令添加 udev 规则：

.. code-block:: bash

    echo 'ATTR{idVendor}=="1d50", ATTR{idProduct}=="6089", SYMLINK+="hackrf-one-%k", MODE="660", TAG+="uaccess"' | sudo tee /etc/udev/rules.d/53-hackrf.rules
    sudo udevadm trigger

然后把 HackRF One 拔掉再重新插上（并重新执行一次 :code:`usbipd attach` 这一步）。
需要注意的是，在做下面的测试步骤之前，我一度遇到了权限问题，直到我在 Windows 侧改用 `WSL USB Manager <https://gitlab.com/alelec/wsl-usb-gui/-/releases>`_ 来管理转发到 WSL 的 USB 设备；它似乎也顺便处理了 udev 规则相关的问题。

无论你使用的是原生 Linux 还是 WSL，到这里你都应该可以运行 :code:`hackrf_info`，并看到类似下面的输出：

.. code-block:: bash

    hackrf_info version: git-17f39433
    libhackrf version: git-17f39433 (0.9)
    Found HackRF
    Index: 0
    Serial number: 00000000000000007687865765a765
    Board ID Number: 2 (HackRF One)
    Firmware Version: 2024.02.1 (API:1.08)
    Part ID Number: 0xa000cb3c 0x004f4762
    Hardware Revision: r10
    Hardware appears to have been manufactured by Great Scott Gadgets.
    Hardware supported by installed firmware: HackRF One

我们再顺手录制一段 FM 频段的 IQ 数据，带宽设为 10 MHz、中心频率设为 100 MHz，并抓取 100 万个样本：

.. code-block:: bash

    hackrf_transfer -r out.iq -f 100000000 -s 10000000 -n 1000000 -a 0 -l 30 -g 50

这个工具会生成一个 int8 格式的二进制 IQ 文件（每个 IQ 样本占 2 字节），在我们的这个例子里文件应当是 2 MB。
如果你感兴趣，这个录制下来的信号可以通过下面的 Python 代码读入：

.. code-block:: python

    import numpy as np
    samples = np.fromfile('out.iq', dtype=np.int8)
    samples = samples[::2] + 1j * samples[1::2]
    print(len(samples))
    print(samples[0:10])
    print(np.max(samples))

如果你的最大值是 127（也就是 ADC 已经饱和），那么就要把命令末尾那两个增益值调低一些。

安装 HackRF Python API
#############################

最后，我们还需要安装 HackRF One 的 `Python 绑定 <https://github.com/GvozdevLeonid/python_hackrf>`_ ，它由 `GvozdevLeonid <https://github.com/GvozdevLeonid>`_ 维护。
以下步骤已在 Ubuntu 22.04 上于 2024 年 11 月 4 日使用当时最新的 main 分支验证可用。

.. code-block:: bash

    sudo apt install libusb-1.0-0-dev
    pip install python_hackrf==1.2.7

我们可以用下面这段代码来测试安装是否成功；如果运行时没有报错（同时也不会有任何输出），那基本就说明一切正常了。

.. code-block:: python

    from python_hackrf import pyhackrf  # type: ignore
    pyhackrf.pyhackrf_init()
    sdr = pyhackrf.pyhackrf_open()
    sdr.pyhackrf_set_sample_rate(10e6)
    sdr.pyhackrf_set_antenna_enable(False)
    sdr.pyhackrf_set_freq(100e6)
    sdr.pyhackrf_set_amp_enable(False)
    sdr.pyhackrf_set_lna_gain(30) # LNA 增益 - 0 到 40 dB，步进为 8 dB
    sdr.pyhackrf_set_vga_gain(50) # VGA 增益 - 0 到 62 dB，步进为 2 dB
    sdr.pyhackrf_close()

至于真正接收样本的测试，请看下文给出的示例代码。

********************************
HackRF 收发增益
********************************

HackRF 接收端增益
############################

HackRF One 在接收端有三个不同的增益级：

* RF (:code:`amp`，只能是 0 或 11 dB)
* IF (:code:`lna`，0 到 40 dB，步进为 8 dB)
* 基带 (:code:`vga`，0 到 62 dB，步进为 2 dB)

在接收大多数信号时，通常建议保持 RF 放大器关闭（0 dB），除非你面对的是一个极其微弱的信号，并且附近可以确定没有强信号存在。
IF（LNA）增益是最关键的那一级，它决定了你能否在避免 ADC 饱和的同时最大化信噪比，因此它是第一优先级要调的旋钮。
基带增益则通常可以保持在较高的值，例如我们这里就直接保持在 50 dB。

HackRF 发射端增益
############################

在发射端，HackRF 有两个增益级：

* RF（只能是 0 或 11 dB）
* IF（0 到 47 dB，步进为 1 dB）

你大概率会希望打开 RF 放大器，然后再根据实际需求调整 IF 增益。

**************************************************
在 Python 中通过 HackRF 接收 IQ 样本
**************************************************

目前 :code:`python_hackrf` 这个 Python 包并没有提供用于接收样本的便捷函数；它本质上只是把 HackRF 的 C++ API 映射成了一组 Python 绑定。
这意味着，如果我们要接收 IQ 样本，就必须写不少代码。
这个 Python 包是通过回调函数来接收更多样本的，也就是说，我们必须自己准备一个回调函数；但一旦设置好，只要 HackRF 有更多样本可读，这个函数就会被自动调用。
这个回调函数始终必须接收四个特定参数，并且如果我们还想继续接收下一批样本，它就必须返回 :code:`0`。
在下面的代码中，每次回调函数被调用时，我们都会把样本转换成 NumPy 的复数类型，缩放到 -1 到 +1，然后存入一个更大的 :code:`samples` 数组中。

运行下面这段代码之后，如果你在时域图中看到样本已经碰到了 ADC 的上下限 -1 和 +1，那么就应当把 :code:`lna_gain` 每次减少 8 dB，直到它明显不再打到边界为止。

.. code-block:: python

    from python_hackrf import pyhackrf  # type: ignore
    import matplotlib.pyplot as plt
    import numpy as np
    import time

    # 这些设置应与书中 hackrf_transfer 示例保持一致，得到的瀑布图应该看起来也差不多
    recording_time = 1  # 秒
    center_freq = 100e6  # Hz
    sample_rate = 10e6
    baseband_filter = 7.5e6
    lna_gain = 30 # 0 到 40 dB，步进为 8 dB
    vga_gain = 50 # 0 到 62 dB，步进为 2 dB

    pyhackrf.pyhackrf_init()
    sdr = pyhackrf.pyhackrf_open()

    allowed_baseband_filter = pyhackrf.pyhackrf_compute_baseband_filter_bw_round_down_lt(baseband_filter) # 根据期望值计算最接近且受支持的带宽

    sdr.pyhackrf_set_sample_rate(sample_rate)
    sdr.pyhackrf_set_baseband_filter_bandwidth(allowed_baseband_filter)
    sdr.pyhackrf_set_antenna_enable(False)  # 这个设置看起来是启用或禁用天线口供电。默认值为 False。固件在回到 IDLE 模式后会自动关闭它

    sdr.pyhackrf_set_freq(center_freq)
    sdr.pyhackrf_set_amp_enable(False)  # 默认值为 False
    sdr.pyhackrf_set_lna_gain(lna_gain)  # LNA 增益 - 0 到 40 dB，步进为 8 dB
    sdr.pyhackrf_set_vga_gain(vga_gain)  # VGA 增益 - 0 到 62 dB，步进为 2 dB

    print(f'center_freq: {center_freq} sample_rate: {sample_rate} baseband_filter: {allowed_baseband_filter}')

    num_samples = int(recording_time * sample_rate)
    samples = np.zeros(num_samples, dtype=np.complex64)
    last_idx = 0

    def rx_callback(device, buffer, buffer_length, valid_length):  # 这个回调函数必须始终带这四个参数
        global samples, last_idx

        accepted = valid_length // 2
        accepted_samples = buffer[:valid_length].astype(np.int8) # -128 到 127
        accepted_samples = accepted_samples[0::2] + 1j * accepted_samples[1::2]  # 转换为复数类型（将交织的 IQ 解交织）
        accepted_samples /= 128 # 缩放到 -1 到 +1
        samples[last_idx: last_idx + accepted] = accepted_samples

        last_idx += accepted

        return 0

    sdr.set_rx_callback(rx_callback)
    sdr.pyhackrf_start_rx()
    print('is_streaming', sdr.pyhackrf_is_streaming())

    time.sleep(recording_time)

    sdr.pyhackrf_stop_rx()
    sdr.pyhackrf_close()
    pyhackrf.pyhackrf_exit()

    samples = samples[100000:] # 为了稳妥起见，丢弃最前面的 100k 样本，因为它们可能包含瞬态

    fft_size = 2048
    num_rows = len(samples) // fft_size
    spectrogram = np.zeros((num_rows, fft_size))
    for i in range(num_rows):
        spectrogram[i, :] = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples[i * fft_size:(i+1) * fft_size]))) ** 2)
    extent = [(center_freq + sample_rate / -2) / 1e6, (center_freq + sample_rate / 2) / 1e6, len(samples) / sample_rate, 0]

    plt.figure(0)
    plt.imshow(spectrogram, aspect='auto', extent=extent) # type: ignore
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Time [s]")

    plt.figure(1)
    plt.plot(np.real(samples[0:10000]))
    plt.plot(np.imag(samples[0:10000]))
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend(["Real", "Imaginary"])

    plt.show()

如果你使用的是能够接收 FM 频段的天线，那么应当会得到类似下面这样的结果，瀑布图里能看到多个 FM 广播电台：

.. image:: ../_images/hackrf_time_screenshot.png
   :align: center
   :scale: 50 %
   :alt: 从 HackRF 抓取样本后的时域图

.. image:: ../_images/hackrf_freq_screenshot.png
   :align: center
   :scale: 50 %
   :alt: 从 HackRF 抓取样本后的时频谱图
