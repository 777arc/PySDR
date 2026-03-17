.. _rtlsdr-chapter:

######################
Python 玩转 RTL-SDR
######################

RTL-SDR 是目前为止最便宜的 SDR，价格大约在 30 美元左右，也是非常适合入门的一款 SDR。
虽然它只能接收、最高只能调谐到约 1.75 GHz，但依然有大量应用场景可以使用它。
本章中，我们将学习如何配置 RTL-SDR 软件，并使用它的 Python API。

.. image:: ../_images/rtlsdrs.svg
   :align: center
   :target: ../_images/rtlsdrs.svg
   :alt: RTL-SDR 示例

********************************
RTL-SDR 背景
********************************

RTL-SDR 大约诞生于 2010 年，当时有人发现可以 hack 那些内置 Realtek RTL2832U 芯片的低成本 DVB-T 电视棒。
DVB-T 是一种主要在欧洲使用的数字电视标准，而 RTL2832U 真正有意思的地方在于，它可以直接访问原始 IQ 样本，因此这颗芯片就可以被拿来构建一台通用的、只接收的 SDR。

RTL2832U 芯片本身集成了模数转换器（ADC）和 USB 控制器，但它必须搭配一个射频调谐器一起工作。
常见的调谐器芯片包括 Rafael Micro 的 R820T、R828D，以及 Elonics 的 E4000。
可调谐频率范围取决于所用调谐器芯片，通常在 50 到 1700 MHz 左右。
另一方面，最大采样率则由 RTL2832U 以及你电脑的 USB 总线决定，通常在 2.4 MHz 左右，再高就容易开始丢样本。
还要记住，这类调谐器极其低成本，因此射频灵敏度通常比较差；如果你要接收较弱信号，往往需要额外加一个低噪声放大器（LNA）和带通滤波器。

RTL2832U 始终使用 8 位样本，因此主机每接收一个 IQ 样本会得到两个字节。
高端一些的 RTL-SDR 通常会用温控振荡器（TCXO）替代更廉价的晶振，以获得更好的频率稳定性。
另一个可选特性是 Bias Tee（也叫 Bias-T），它是一种板载电路，会在 SMA 接头上提供大约 4.5V 的直流电，用来方便地给外部 LNA 或其他射频器件供电。
这部分额外直流偏置位于 SDR 的射频侧，因此不会干扰基本的接收操作。

如果你对到达方向估计（DOA）或其他波束形成应用感兴趣，那么 `KrakenSDR <https://www.crowdsupply.com/krakenrf/krakensdr>`_ 是一个值得关注的产品：它是由五个共享振荡器和采样时钟的 RTL-SDR 组成的一台相位相干 SDR。

********************************
RTL-SDR 软件配置
********************************

在 Ubuntu 上安装 RTL-SDR（或 WSL 中的 Ubuntu）
############################################################

在 Ubuntu 20、22 以及其他基于 Debian 的系统上，可以使用下面这条命令安装 RTL-SDR 软件。

.. code-block:: bash

 sudo apt install rtl-sdr

这会安装 librtlsdr 库，以及诸如 :code:`rtl_sdr`、:code:`rtl_tcp`、:code:`rtl_fm` 和 :code:`rtl_test` 这样的命令行工具。

接下来，用下面的命令安装 librtlsdr 的 Python 封装：

.. code-block:: bash

 sudo pip install pyrtlsdr

如果你是通过 WSL 使用 Ubuntu，那么在 Windows 侧需要先下载最新版本的 `Zadig <https://zadig.akeo.ie/>`_ ，并运行它来为 RTL-SDR 安装 “WinUSB” 驱动（可能会看到两个 Bulk-In Interface，如果是这样，就两个都安装 “WinUSB”）。
Zadig 完成后，把 RTL-SDR 拔掉再重新插上。

接下来，你需要把 RTL-SDR 的 USB 设备转发到 WSL。
首先安装最新版本的 `usbipd utility msi <https://github.com/dorssel/usbipd-win/releases>`_ （本文默认你使用的是 usbipd-win 4.0.0 或更高版本），然后以管理员模式打开 PowerShell 并执行：

.. code-block:: bash

    # （先拔掉 RTL-SDR）
    usbipd list
    # （再插上 RTL-SDR）
    usbipd list
    # （找到新出现的设备，并把它的 busid 代入下面的命令）
    usbipd bind --busid 1-5
    usbipd attach --wsl --busid 1-5

在 WSL 这边，你应该可以通过 :code:`lsusb` 看到一个新设备项，名称类似 RTL2838 DVB-T。

如果你遇到了权限问题（例如下面的测试只有在使用 :code:`sudo` 时才工作），那么你需要配置 udev 规则。
先运行 :code:`lsusb` 找到 RTL-SDR 的 ID，然后创建文件 :code:`/etc/udev/rules.d/10-rtl-sdr.rules`，写入以下内容；如果你的 RTL-SDR 的 idVendor 或 idProduct 不同，请自行替换：

.. code-block::

 SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2838", MODE="0666"

要刷新 udev，请执行：

.. code-block:: bash

    sudo udevadm control --reload-rules
    sudo udevadm trigger

如果你使用的是 WSL，并且它提示 :code:`Failed to send reload request: No such file or directory`，那就说明 udev 服务没有运行。
你需要执行 :code:`sudo nano /etc/wsl.conf` 并加入以下内容：

.. code-block:: bash

 [boot]
 command="service udev start"

然后在管理员 PowerShell 中执行下面的命令重启 WSL： :code:`wsl.exe --shutdown` 。

你可能还需要把 RTL-SDR 拔掉再重新插上（WSL 下则需要重新执行 :code:`usbipd attach`）。

Windows 下安装 RTL-SDR
########################################

如果你使用 Windows，请参考 https://www.rtl-sdr.com/rtl-sdr-quick-start-guide/ 。

********************************
测试 RTL-SDR 软件栈
********************************

如果软件配置没有问题，你应该可以运行下面的测试。
它会把 RTL-SDR 调到 FM 广播频段，并录制 100 万个样本到 :code:`/tmp` 下一个名为 :code:`recording.iq` 的文件中。

.. code-block:: bash

    rtl_sdr /tmp/recording.iq -s 2e6 -f 100e6 -n 1e6

如果你看到 :code:`No supported devices found`，即使在命令前面加上 :code:`sudo` 也是如此，那么说明 Linux 根本看不到 RTL-SDR 设备。
如果它在 :code:`sudo` 下可以工作，那就说明是 udev 规则的问题；请按照前面的 udev 配置步骤处理后，再尝试重启电脑。
当然，你也可以简单粗暴地对所有命令都加 :code:`sudo`，包括运行 Python。

你还可以用下面这段脚本测试 Python 是否能看到 RTL-SDR：

.. code-block:: python

 from rtlsdr import RtlSdr

 sdr = RtlSdr()
 sdr.sample_rate = 2.048e6 # Hz
 sdr.center_freq = 100e6   # Hz
 sdr.freq_correction = 60  # PPM
 sdr.gain = 'auto'

 print(len(sdr.read_samples(1024)))
 sdr.close()

其输出应类似：

.. code-block:: bash

 Found Rafael Micro R820T tuner
 [R82XX] PLL not locked!
 1024

********************************
RTL-SDR Python 代码
********************************

上面的代码其实就可以算是一个 RTL-SDR Python 基本使用示例。
接下来的几个小节会更详细地介绍各种设置以及一些使用技巧。

避免 RTL-SDR 卡死
###############################

在脚本结尾，或者每次用完 RTL-SDR 准备停止抓取样本时，我们都应该调用 :code:`sdr.close()`。
这样有助于避免 RTL-SDR 进入某种异常卡死状态，否则你可能不得不把它拔掉再插上。
即使调用了 :code:`close()`，这种情况仍然可能发生；如果它发生了，你通常会在 :code:`read_samples()` 调用期间发现 RTL-SDR 卡住不动。
这时你就需要把 RTL-SDR 拔掉重插，必要时甚至重启电脑。
如果你使用的是 WSL，还需要通过 usbipd 重新 attach 这个设备。

RTL-SDR 增益设置
#############################

通过设置 :code:`sdr.gain = 'auto'`，我们启用了自动增益控制（AGC）。
这样 RTL-SDR 会根据接收到的信号自动调整接收增益，尽量在不让 8 位 ADC 饱和的前提下把动态范围填满。
但在很多场景下，例如制作一个频谱分析仪，让增益保持为固定值反而更有用，这就意味着我们需要设置手动增益。
RTL-SDR 的增益不是连续可调的；你可以通过 :code:`print(sdr.valid_gains_db)` 查看所有可用的增益值。
不过即便你设置了一个不在这个列表中的增益值，它也会自动选择最接近的合法值。
你也可以随时用 :code:`print(sdr.gain)` 查看当前实际设置的增益。
下面这个例子中，我们把增益设置为 49.6 dB，接收 4096 个样本，然后在时域中绘制它们：

.. code-block:: python

 from rtlsdr import RtlSdr
 import numpy as np
 import matplotlib.pyplot as plt

 sdr = RtlSdr()
 sdr.sample_rate = 2.048e6 # Hz
 sdr.center_freq = 100e6   # Hz
 sdr.freq_correction = 60  # PPM
 print(sdr.valid_gains_db)
 sdr.gain = 49.6
 print(sdr.gain)

 x = sdr.read_samples(4096)
 sdr.close()

 plt.plot(x.real)
 plt.plot(x.imag)
 plt.legend(["I", "Q"])
 plt.savefig("../_images/rtlsdr-gain.svg", bbox_inches='tight')
 plt.show()

.. image:: ../_images/rtlsdr-gain.svg
   :align: center
   :target: ../_images/rtlsdr-gain.svg
   :alt: RTL-SDR 手动增益示例

这里有几点值得注意。
首先，前面大约 2k 个样本似乎没什么信号功率，因为它们主要是瞬态部分。
因此通常建议你在每个脚本开始时先丢弃前 2k 个样本，例如调用 :code:`sdr.read_samples(2048)`，但不要对输出做任何处理。
其次，我们会注意到 pyrtlsdr 返回给我们的样本是浮点数，范围在 -1 到 +1 之间。
虽然 RTL-SDR 使用的是 8 位 ADC，本来产生的是整数值，但 pyrtlsdr 为了方便使用，已经自动帮我们除以了 127.0。

RTL-SDR 允许的采样率
###############################

大多数 RTL-SDR 要求采样率必须设置在 230 到 300 kHz 之间，或者 900 kHz 到 3.2 MHz 之间。
请注意，较高的采样率，尤其是超过 2.4 MHz 时，未必能通过 USB 连接传回 100% 的样本。
如果你给它设置了一个不支持的采样率，它只会直接报错，例如： :code:`rtlsdr.rtlsdr.LibUSBError: Error code -22: Could not set sample rate to 899000 Hz` 。
当你设置一个合法的采样率时，你会在终端中看到实际采用的精确采样率；这个值也可以通过读取 :code:`sdr.sample_rate` 获得。
有些应用在做计算时，使用这个更精确的实际值会更有帮助。

作为一个练习，我们把采样率设置为 2.4 MHz，并创建 FM 广播频段的时频谱：

.. code-block:: python

 # ...
 sdr.sample_rate = 2.4e6 # Hz
 # ...

 fft_size = 512
 num_rows = 500
 x = sdr.read_samples(2048) # 丢弃前面这些空样本
 x = sdr.read_samples(fft_size*num_rows) # 读取时频谱所需的全部样本
 spectrogram = np.zeros((num_rows, fft_size))
 for i in range(num_rows):
     spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)
 extent = [(sdr.center_freq + sdr.sample_rate/-2)/1e6,
             (sdr.center_freq + sdr.sample_rate/2)/1e6,
             len(x)/sdr.sample_rate, 0]
 plt.imshow(spectrogram, aspect='auto', extent=extent)
 plt.xlabel("Frequency [MHz]")
 plt.ylabel("Time [s]")
 plt.show()

.. image:: ../_images/rtlsdr-waterfall.svg
   :align: center
   :target: ../_images/rtlsdr-waterfall.svg
   :alt: RTL-SDR 瀑布图（时频谱）示例

RTL-SDR 的 PPM 设置
############################

如果你好奇 ppm 设置到底是什么：每一台 RTL-SDR 都会因为调谐器芯片成本低、缺乏校准，而存在一个小的频率偏移/误差。
这个频率偏移在整个频谱上通常近似线性（而不是一个恒定的频移），因此我们可以通过输入一个以百万分之一（parts per million）为单位的 PPM 值来修正它。
例如，如果你调到 100 MHz，并把 PPM 设为 25，那么接收到的信号将会上移 :math:`100e6/1e6*25=2500` Hz。
对于更窄带的信号，频率误差带来的影响会更明显。
不过，很多现代信号在解调过程中本身就包含频率同步步骤，因此无论频偏来自发射端、接收端还是多普勒效应，它最终都会被纠正掉。

********************************
RTL-SDR 延伸阅读
********************************

#. `RTL-SDR.com 的 About 页面 <https://www.rtl-sdr.com/about-rtl-sdr/>`_
#. https://hackaday.com/2019/07/31/rtl-sdr-seven-years-later/
#. https://osmocom.org/projects/rtl-sdr/wiki/Rtl-sdr
