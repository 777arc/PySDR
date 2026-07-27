.. _bladerf-chapter:

######################
Python 玩转 bladeRF
######################

`Nuand <https://www.nuand.com>`_ 推出的 bladeRF 2.0（也叫 bladeRF 2.0 micro）是一款基于 USB 3.0 的 SDR，具备两路接收通道、两路发射通道，可调谐频率范围为 47 MHz 到 6 GHz，最高可支持 61 MHz 的采样率，经过 hack 后甚至可以到 122 MHz。
它和 USRP B210 以及 PlutoSDR 一样，使用的是 AD9361 射频集成电路（RFIC），因此射频性能会比较接近。
bladeRF 2.0 于 2021 年发布，尺寸保持在 2.5" x 4.5" 的小体积，并提供两种不同 FPGA 容量的版本（xA4 和 xA9）。
虽然本章聚焦于 bladeRF 2.0，但其中不少代码同样适用于最早 `于 2013 年推出 <https://www.kickstarter.com/projects/1085541682/bladerf-usb-30-software-defined-radio>`_ 的初代 bladeRF。

.. image:: ../_images/bladeRF_micro.png
   :scale: 35 %
   :align: center
   :alt: bladeRF 2.0 宣传照

********************************
bladeRF 架构
********************************

从高层来看，bladeRF 2.0 基于 AD9361 RFIC，搭配 Cyclone V FPGA（49 kLE 的 :code:`5CEA4` 或 301 kLE 的 :code:`5CEA9`），以及一颗 Cypress FX3 USB 3.0 控制器，其内部带有 200 MHz 的 ARM9 核，并加载了定制固件。
bladeRF 2.0 的方框图如下所示：

.. image:: ../_images/bladeRF-2.0-micro-Block-Diagram-4.png
   :scale: 80 %
   :align: center
   :alt: bladeRF 2.0 方框图

FPGA 负责控制 RFIC、执行数字滤波、将数据打包后通过 USB 传输（以及其他一些工作）。
FPGA 镜像的 `源代码 <https://github.com/Nuand/bladeRF/tree/master/hdl>`_ 使用 VHDL 编写，如果你想编译自定义镜像，则需要使用免费的 Quartus Prime Lite 设计软件。
预编译镜像可以在 `这里 <https://www.nuand.com/fpga_images/>`_ 获取。

Cypress FX3 固件的 `源代码 <https://github.com/Nuand/bladeRF/tree/master/fx3_firmware>`_ 也是开源的，其中包含以下功能代码：

1. 加载 FPGA 镜像
2. 通过 USB 3.0 在 FPGA 与主机之间传输 IQ 样本
3. 通过 UART 控制 FPGA 的 GPIO

从信号流的角度看，它有两路接收通道和两路发射通道，而且每个通道根据所用频段的不同，都对应 RFIC 的低频和高频输入/输出路径。
正因为如此，在 RFIC 与 SMA 接头之间需要一个单刀双掷（SPDT）电子射频开关。
Bias Tee（也叫 Bias-T）是板载电路，它会在 SMA 接头上提供约 4.5V 的直流电，用来方便地为外部放大器或其他射频器件供电。
这部分额外的直流偏置位于 SDR 的射频侧，因此不会干扰基本的收发操作。

JTAG 是一种调试接口，可用于在开发过程中测试和验证设计。

在本章末尾，我们还会讨论 VCTCXO 振荡器、PLL，以及扩展接口。

********************************
bladeRF 的软件与硬件配置
********************************

在 Ubuntu 上安装 bladeRF（或 WSL 中的 Ubuntu）
############################################################

在 Ubuntu 以及其他基于 Debian 的系统上，可以使用下面的命令安装 bladeRF 软件：

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

这会安装 libbladerf 库、Python 绑定、bladeRF 命令行工具、固件下载器，以及 FPGA 比特流下载器。
要检查你安装的库版本，可以使用 :code:`bladerf-tool version` (本文写作时使用的是 libbladeRF v2.5.0)。

如果你是通过 WSL 使用 Ubuntu，那么在 Windows 侧需要先把 bladeRF 的 USB 设备转发到 WSL。
首先安装最新版本的 `usbipd utility msi <https://github.com/dorssel/usbipd-win/releases>`_ （本文默认你使用的是 usbipd-win 4.0.0 或更高版本），然后以管理员模式打开 PowerShell 并执行：

.. code-block:: bash

    usbipd list
    # （找到标记为 bladeRF 2.0 的 BUSID，并代入下面的命令）
    usbipd bind --busid 1-23
    usbipd attach --wsl --busid 1-23

在 WSL 这边，你应该能通过 :code:`lsusb` 看到一个新设备项，名称类似 :code:`Nuand LLC bladeRF 2.0 micro`。
注意，如果你希望它自动重新连接，可以在 :code:`usbipd attach` 命令后加上 :code:`--auto-attach` 参数。

（可能不需要）对于原生 Linux 和 WSL，我们都需要安装 udev 规则，以避免权限错误：

.. code-block::

 sudo nano /etc/udev/rules.d/88-nuand.rules

然后把下面这行内容粘贴进去：

.. code-block::

 ATTRS{idVendor}=="2cf0", ATTRS{idProduct}=="5250", MODE="0666"

保存并退出 nano 的方法是：先按 control-o，然后回车，再按 control-x。
要刷新 udev，请执行：

.. code-block:: bash

    sudo udevadm control --reload-rules && sudo udevadm trigger

如果你使用的是 WSL，并且它提示 :code:`Failed to send reload request: No such file or directory`，那就说明 udev 服务没有运行。
这时你需要执行 :code:`sudo nano /etc/wsl.conf`，并加入以下内容：

.. code-block:: bash

 [boot]
 command="service udev start"

然后在管理员 PowerShell 中执行以下命令来重启 WSL： :code:`wsl.exe --shutdown` 。

拔下再重新插上你的 bladeRF（WSL 用户还需要重新 attach），然后用下面的命令测试权限：

.. code-block:: bash

 bladerf-tool probe
 bladerf-tool info

如果看到列出了你的 bladeRF 2.0，并且 **没有** 出现 :code:`Found a bladeRF via VID/PID, but could not open it due to insufficient permissions`，就说明配置成功了。
如果成功，请顺便记下输出里的 FPGA Version 和 Firmware Version。

（可选）安装最新版本的固件和 FPGA 镜像（本文写作时分别是 v2.4.0 和 v0.15.0）：

.. code-block:: bash

 cd ~/Downloads
 wget https://www.nuand.com/fx3/bladeRF_fw_latest.img
 bladerf-tool flash_fw bladeRF_fw_latest.img

 # xA4 版本使用：
 wget https://www.nuand.com/fpga/hostedxA4-latest.rbf
 bladerf-tool flash_fpga hostedxA4-latest.rbf

 # xA9 版本使用：
 wget https://www.nuand.com/fpga/hostedxA9-latest.rbf
 bladerf-tool flash_fpga hostedxA9-latest.rbf

然后给 bladeRF 断电重启一次，也就是拔掉再插上。

接下来我们通过在 FM 广播频段接收 100 万个样本、采样率设为 10 MHz，并把数据写入 :code:`/tmp/samples.sc16`，来测试它的基本功能：

.. code-block:: bash

 bladerf-tool rx --num-samples 1000000 /tmp/samples.sc16 100e6 10e6

出现少量 :code:`Hit stall for buffer` 是正常的；如果最终你看到了一个 4 MB 的 :code:`/tmp/samples.sc16` 文件，就说明它工作正常。

最后，使用下面的命令测试 Python API：

.. code-block:: bash

 python3
 import bladerf
 bladerf.BladeRF()
 exit()

如果你看到类似 :code:`<BladeRF(<DevInfo(...)>)>` 的输出，并且没有 warning/error，那么就说明 Python API 也工作正常。

Windows 与 macOS 下安装 bladeRF
########################################

如果你使用 Windows（并且不打算走 WSL），请参考 https://github.com/Nuand/bladeRF/wiki/Getting-Started%3A-Windows ；如果你使用 macOS，请参考 https://github.com/Nuand/bladeRF/wiki/Getting-started:-Mac-OSX 。

********************************
bladeRF Python API 基础
********************************

首先，让我们用下面这段脚本从 bladeRF 中查询一些有用的信息。 **不要把你的脚本命名为** ``bladerf.py``，否则它会和 bladeRF 的 Python 模块本身发生冲突！

.. code-block:: python

 from bladerf import _bladerf
 import numpy as np
 import matplotlib.pyplot as plt

 sdr = _bladerf.BladeRF()

 print("Device info:", _bladerf.get_device_list()[0])
 print("libbladeRF version:", _bladerf.version()) # v2.5.0
 print("Firmware version:", sdr.get_fw_version()) # v2.4.0
 print("FPGA version:", sdr.get_fpga_version())   # v0.15.0

 rx_ch = sdr.Channel(_bladerf.CHANNEL_RX(0)) # 这里传入 0 或 1
 print("sample_rate_range:", rx_ch.sample_rate_range)
 print("bandwidth_range:", rx_ch.bandwidth_range)
 print("frequency_range:", rx_ch.frequency_range)
 print("gain_modes:", rx_ch.gain_modes)
 print("manual gain range:", sdr.get_gain_range(_bladerf.CHANNEL_RX(0))) # 通道 0 或 1

对于 bladeRF 2.0 xA9，输出应大致类似下面这样：

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

带宽参数决定了 SDR 在接收时使用的滤波器，因此我们通常把它设置为与 :code:`sample_rate/2` 相等或略小。
增益模式也很重要：设备既支持手动增益模式（由你以 dB 指定增益），也支持自动增益控制（AGC），并提供 fast、slow、hybrid 三种设置。
对于频谱监测这类应用，通常更建议使用手动增益（这样你能看出信号什么时候进来、什么时候消失）；而对于接收某个你预期一定存在的特定信号时，AGC 更实用，因为它会自动调节增益，使信号尽可能填满模数转换器（ADC）的动态范围。

要设置 SDR 的主要参数，我们可以加上下面这些代码：

.. code-block:: python

 sample_rate = 10e6
 center_freq = 100e6
 gain = 50 # -15 到 60 dB
 num_samples = int(1e6)

 rx_ch.frequency = center_freq
 rx_ch.sample_rate = sample_rate
 rx_ch.bandwidth = sample_rate/2
 rx_ch.gain_mode = _bladerf.GainMode.Manual
 rx_ch.gain = gain

********************************
在 Python 中接收 bladeRF 样本
********************************

接下来，我们基于上一段代码，在 FM 广播频段以 10 MHz 的采样率接收 100 万个样本，就像前面在命令行里做的那样。
只要天线接在 RX1 口上，通常都应该能收到 FM 广播，因为它的信号一般很强。
下面这段代码展示了 bladeRF 的同步流（synchronous stream）API 是如何工作的；在开始接收之前，必须先完成流配置，并创建接收缓冲区。
:code:`while True:` 循环会一直接收，直到达到所请求的样本数。
接收到的样本会被存入一个单独的 numpy 数组中，这样我们就可以在循环结束之后再处理它们。

.. code-block:: python

 # 配置同步流
 sdr.sync_config(layout = _bladerf.ChannelLayout.RX_X1, # 或 RX_X2
                 fmt = _bladerf.Format.SC16_Q11, # int16
                 num_buffers    = 16,
                 buffer_size    = 8192,
                 num_transfers  = 8,
                 stream_timeout = 3500)

 # 创建接收缓冲区
 bytes_per_sample = 4 # 不要修改，它始终使用 int16
 buf = bytearray(1024 * bytes_per_sample)

 # 启用模块
 print("Starting receive")
 rx_ch.enable = True

 # 接收循环
 x = np.zeros(num_samples, dtype=np.complex64) # 用来存储 IQ 样本
 num_samples_read = 0
 while True:
     if num_samples > 0 and num_samples_read == num_samples:
         break
     elif num_samples > 0:
         num = min(len(buf) // bytes_per_sample, num_samples - num_samples_read)
     else:
         num = len(buf) // bytes_per_sample
     sdr.sync_rx(buf, num) # 读入缓冲区
     samples = np.frombuffer(buf, dtype=np.int16)
     samples = samples[0::2] + 1j * samples[1::2] # 转换为复数类型
     samples /= 2048.0 # 缩放到 -1 到 1（它使用的是 12 位 ADC）
     x[num_samples_read:num_samples_read+num] = samples[0:num] # 将 buf 存入 samples 数组
     num_samples_read += num

 print("Stopping")
 rx_ch.enable = False
 print(x[0:10]) # 查看前 10 个 IQ 样本
 print(np.max(x)) # 如果这个值接近 1，说明 ADC 过载了，应当减小增益

在结束时出现少量 :code:`Hit stall for buffer` 是正常的。
最后打印出来的那个数值表示接收到的最大样本值；你应当调节增益，让它大约落在 0.5 到 0.8 之间。
如果它接近 0.999，就说明接收机已经过载/饱和，信号会发生失真（在频域中会表现为被涂抹开来）。

为了可视化接收到的信号，我们接下来用时频谱来显示这些 IQ 样本（关于时频谱如何工作的更多细节，请参见 :ref:`spectrogram-section` 小节）。
把下面这段代码追加到前一个代码块的末尾：

.. code-block:: python

 # 创建时频谱
 fft_size = 2048
 num_rows = len(x) // fft_size # // 是向下取整的整数除法
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
   :alt: bladeRF 时频谱示例

图中每一条竖着扭动的线都是一个 FM 广播信号。
右侧那团脉冲状的东西具体是什么我也不清楚，把增益调低也没有让它消失。


********************************
在 Python 中发射 bladeRF 样本
********************************

使用 bladeRF 发射样本的流程与接收非常相似。
最主要的区别在于，我们必须先生成要发射的样本，然后通过 :code:`sync_tx` 方法把它们写入 bladeRF；这个方法可以一次处理整批样本（最多大约 40 亿个样本）。
下面的代码展示了如何发射一个简单的单音信号，并将它重复 30 次。
这个单音信号用 numpy 生成，然后被缩放到 -2048 到 2048 之间，以适配 12 位数模转换器（DAC）。
随后它会被转换成表示 int16 的字节串，并作为发射缓冲区使用。
同步流 API 用来发射这些样本，而 :code:`while True:` 循环则会持续发射，直到达到指定的重复次数。
如果你想发射文件中的样本，也可以直接使用 :code:`samples = np.fromfile('yourfile.iq', dtype=np.int16)` (或者对应的实际数据类型) 读入样本，然后用 :code:`samples.tobytes()` 把它们转成字节。
不过请记得 DAC 对应的数值范围是 -2048 到 2048。

.. code-block:: python

 from bladerf import _bladerf
 import numpy as np

 sdr = _bladerf.BladeRF()
 tx_ch = sdr.Channel(_bladerf.CHANNEL_TX(0)) # 这里传入 0 或 1

 sample_rate = 10e6
 center_freq = 100e6
 gain = 0 # -15 到 60 dB。发射时应从较低值开始慢慢往上加，并确保天线已连接
 num_samples = int(1e6)
 repeat = 30 # 重复发射次数
 print('duration of transmission:', num_samples/sample_rate*repeat, 'seconds')

 # 生成待发射的 IQ 样本（这里是一个简单的单音）
 t = np.arange(num_samples) / sample_rate
 f_tone = 1e6
 samples = np.exp(1j * 2 * np.pi * f_tone * t) # 范围是 -1 到 +1
 samples = samples.astype(np.complex64)
 samples *= 2048.0 # 缩放到 -2048 到 2048（它使用的是 12 位 DAC）
 samples = samples.view(np.int16)
 buf = samples.tobytes() # 转成字节，并作为发射缓冲区

 tx_ch.frequency = center_freq
 tx_ch.sample_rate = sample_rate
 tx_ch.bandwidth = sample_rate/2
 tx_ch.gain = gain

 # 配置同步流
 sdr.sync_config(layout=_bladerf.ChannelLayout.TX_X1, # 或 TX_X2
                 fmt=_bladerf.Format.SC16_Q11, # int16
                 num_buffers=16,
                 buffer_size=8192,
                 num_transfers=8,
                 stream_timeout=3500)

 print("Starting transmit!")
 repeats_remaining = repeat - 1
 tx_ch.enable = True
 while True:
     sdr.sync_tx(buf, num_samples) # 写入 bladeRF
     print(repeats_remaining)
     if repeats_remaining > 0:
         repeats_remaining -= 1
     else:
         break

 print("Stopping transmit")
 tx_ch.enable = False

在结束时出现少量 :code:`Hit stall for buffer` 也是正常的。

如果你想同时发射和接收，那么就必须使用线程。
这种情况下，你基本上可以直接使用 Nuand 提供的示例 `txrx.py <https://github.com/Nuand/bladeRF/blob/624994d65c02ad414a01b29c84154260912f4e4f/host/examples/python/txrx/txrx.py>`_ ，它就是专门做这件事的。

***********************************
振荡器、PLL 与 bladeRF 校准
***********************************

所有直接变频 SDR（包括所有基于 AD9361 的 SDR，例如 USRP B2X0、Analog Devices Pluto 和 bladeRF）都依赖一个单一振荡器来为射频收发器提供稳定时钟。
这个振荡器输出频率中的任何偏移或抖动，都会转化为接收或发射信号中的频率偏移和频率抖动。
这个振荡器本身位于板上，但也可以选择通过板上的 U.FL 接头输入一个额外的方波或正弦参考信号，对它进行约束。

bladeRF 板上使用的是一颗频率为 38.4 MHz 的 `Abracon VCTCXO <https://abracon.com/Oscillators/ASTX12_ASVTX12.pdf>`_ （压控温度补偿晶体振荡器，Voltage-controlled temperature-compensated oscillator）。
其中 “温度补偿” 表示它被设计成在较宽的温度范围内都保持稳定。
而 “压控” 则表示可以通过施加电压对振荡器频率进行细微调节；在 bladeRF 上，这个电压来自另一颗独立的 10 位 DAC，如下图中绿色部分所示。
这意味着我们可以通过软件对振荡器频率做细调，而这正是校准（也叫 trim）bladeRF 的 VCTCXO 的方式。
幸运的是，bladeRF 在出厂时就已经做过校准了，后文会详细提到；但如果你手头有测试设备，随着时间推移、振荡器频率发生漂移后，你依然可以进一步手动微调这一数值。

.. image:: ../_images/bladeRF-2.0-micro-Block-Diagram-4-oscillator.png
   :scale: 80 %
   :align: center
   :alt: bladeRF 2.0 振荡器方框图

当使用外部频率参考时（几乎可以是任意不超过 300 MHz 的频率），参考信号会被直接送入 bladeRF 板上的 `Analog Devices ADF4002 <http://www.analog.com/en/adf4002>`_ PLL。
这个 PLL 会锁定参考信号，并向 VCTCXO 发送一个控制信号（上图蓝色部分），其大小与（经过比例缩放后的）参考输入和 VCTCXO 输出之间的频率差与相位差成正比。
当 PLL 锁定之后，PLL 和 VCTCXO 之间的这个信号就会变成一个稳态直流电压，使 VCTCXO 输出维持在 “精确的” 38.4 MHz（前提是参考源本身足够准确），并与参考输入保持相位锁定。
在使用外部参考时，你必须启用 :code:`clock_ref` (无论是在 Python 中还是在 CLI 中)，并设置输入参考频率（也叫 :code:`refin_freq`），其默认值是 10 MHz。
使用外部参考的理由包括更高的频率精度，以及让多台 SDR 共享同一个参考源从而实现同步。

每台 bladeRF 的 VCTCXO DAC trim 值在出厂时都会被校准到室温下 38.4 MHz 误差不超过 1 Hz。
你可以把序列号输入到 `这个页面 <https://www.nuand.com/calibration/>`_ 查看出厂校准值（序列号可以在板上找到，也可以通过 :code:`bladerf-tool probe` 查看）。
根据 Nuand 的说法，一块新板卡通常会优于 0.5 ppm，很多时候甚至接近 0.1 ppm。
如果你手头有频率精度测试设备，或者只是想把它恢复到出厂值，可以使用下面的命令：

.. code-block:: bash

 $ bladeRF-cli -i
 bladeRF> flash_init_cal 301 0x2049

其中，把 :code:`301` 替换为你的 bladeRF 容量型号，把 :code:`0x2049` 替换为你的 VCTCXO DAC trim 十六进制值。
修改后必须重新上电才会生效。

***********************************
以 122 MHz 采样
***********************************

敬请期待！

***********************************
bladeRF 扩展接口
***********************************

bladeRF 2.0 提供了一个使用 BSH-030 连接器的扩展接口。
关于如何使用这个接口，后续再补充。

********************************
bladeRF 延伸阅读
********************************

#. `bladeRF Wiki <https://github.com/Nuand/bladeRF/wiki>`_
#. `Nuand 的 txrx.py 示例 <https://github.com/Nuand/bladeRF/blob/master/host/examples/python/txrx/txrx.py>`_
