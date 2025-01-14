.. _usrp-chapter:

####################################
Python 玩转 USRP
####################################

.. image:: ../_images/usrp.png
   :scale: 50 % 
   :align: center
   :alt: Ettus Research 提供的 USRP 无线电系列 

在本章中，我们将学习如何使用 UHD Python API 来控制 `USRP <https://www.ettus.com/>`_ 收发信号，它是由 Ettus Research（现在是 NI 的一部分）制造的 SDR 设备。
我们将讨论如何在 Python 中进行 USRP 的信号收发，并深入探讨其他 USRP 使用细节，包括流参数（Stream Args）、子设备（Subdevices）、通道（Channels）、10 MHz 参考信号和 PPS 同步。

***************************
软件与驱动安装
***************************

本书提供的 Python 代码可在 Windows、Mac 和 Linux 下运行，但我们只会提供针对 Ubuntu 22 的软件与驱动安装指南（虽然理论上也适用于大多数同样基于 Debian 的发行版）。
我们将从创建一个 Ubuntu 22 的 VirtualBox 虚拟机（VM） 开始指南。
如果你已经准备好操作系统，可以跳过 VM 部分。
如果你使用的是 Windows 11，那么 Windows Linux 子系统（WSL）的 Ubuntu 22 版本通常运行良好，并且同样支持开箱即用的图形界面。

安装 Ubuntu 22 VM
#############################

(可选)

1. 下载 Ubuntu 22.04 Desktop 的 .iso 文件 - https://ubuntu.com/download/desktop
2. 安装并启动 `VirtualBox <https://www.virtualbox.org/wiki/Downloads>`_ 。
3. 创建一个 VM，建议你使用主机内存的 50% 作为 VM 内存大小。
4. 创建虚拟硬盘，选择 VDI，并选择动态分配空间。15 GB 应该足够，如果你想万无一失，也可以选择更大的空间。
5. 启动 VM。它会要求你选择安装媒体（Installation Media），请选择 Ubuntu 22 desktop 的 .iso 文件，选择 “install ubuntu”，使用默认选项，屏幕中会弹出一个警告窗口，点击继续，输入用户名/密码，然后等待 VM 完成初始化。初始化完成后，VM 将重新启动，请在重新启动后让 VM 关机。
6. 进入 VM 设置（齿轮图标）。
7. 在系统 > 处理器 下选择至少 3 个 CPU。如果你使用独立显卡，请在显示 > 显存 中选择一个更高的值。
8. 启动 VM。
9. 对于 USB 类型的 USRPs（译者注：如 B210），还需要在 VirtualBox 中安装 VM 附加工具。然后在 VM 中转到 设备 > 插入客户附加 CD > 弹出框时点击运行。最后重启 VM，尝试将 USRP 转发到 VM。此外，共享剪贴板功能可以这样启用：设备 > 共享剪贴板 > 双向启用。

安装 UHD 和 Python API
################################

这些终端命令将从源码构建并安装最新版本的 UHD，其中包括 Python API。

.. code-block:: bash

 sudo apt-get install git cmake libboost-all-dev libusb-1.0-0-dev python3-docutils python3-mako python3-numpy python3-requests python3-ruamel.yaml python3-setuptools build-essential # 译者注：这里可以提前配置 APT 源为清华源等镜像以加速下载
 cd ~
 git clone https://github.com/EttusResearch/uhd.git # 译者注：这里需要注意网络环境
 mkdir build
 cd build
 cmake -DENABLE_TESTS=OFF -DENABLE_C_API=OFF -DENABLE_PYTHON_API=ON -DENABLE_MANUAL=OFF ..
 make -j8
 sudo make install
 sudo ldconfig

查看 Ettus 官方的 `从源代码构建和安装UHD <https://files.ettus.com/manual/page_build_guide.html>`_ 页面以获取更多帮助。 请注意，以上方法为从源码安装并构建，也有其他的安装方法（译者注：比如使用 anaconda 安装最新版 GNURadio，这将同时自动安装最新版 UHD）。

测试 UHD 驱动以及 Python API
######################################

新开一个终端并执行以下命令：

.. code-block:: bash

 python3
 import uhd
 usrp = uhd.usrp.MultiUSRP()
 samples = usrp.recv_num_samps(10000, 100e6, 1e6, [0], 50)
 print(samples[0:10])

如果没有报错，那么恭喜你搞定安装了！


用 Python 对 USRP 进行速度基准测试
#######################################

(可选)

如果你使用标准的源码安装方式，下面的命令可以让你用 Python API 对 USRP 接收速率进行基准测试。如果命令中的 56e6 导致了采样点丢弃或者溢出（Overflow）警告，可以尝试降低这个数字。采样点丢弃不是什么大问题，但是标志着你的 VM/主机性能可能存在瓶颈。例如，如果使用 B2X0，一台带有 USB 3.0 端口的正常运行的主机应该能够在不丢失采样点的情况下以 56 MHz 的采样率运行，尤其是当 :code:`num_recv_frames` 设置的如此高时。

.. code-block:: bash

 python /usr/lib/uhd/examples/python/benchmark_rate.py --rx_rate 56e6 --args "num_recv_frames=1000"


************************
接收
************************

使用 USRP 接收信号是非常简单的，比如你可以使用 UHD 的内置快捷函数 :code:`recv_num_samps()` 。
下面是 Python 代码示例：USRP 被调谐到 100 MHz，采样率为 1 MHz，并使用 50 dB 的接收增益，接收 10,000 个采样点。

.. code-block:: python

 import uhd
 usrp = uhd.usrp.MultiUSRP()
 samples = usrp.recv_num_samps(10000, 100e6, 1e6, [0], 50) # 单位: 需要接收的采样点总数（无单位）, Hz, Hz, channel IDs 的列表, dB
 print(samples[0:10])

其中，[0] 是指让 USRP 通过且仅通过第一个通道（射频口）接收信号（例如，B210 具备两个通道，若想同时通过两个通道接收，那么改为 [0, 1] 即可）。

如果采样率设置的太高以至于不断出现溢出（表现为终端打印字符 “O”），在初始化 USRP 时可以这样配置：

.. code-block:: python

 usrp = uhd.usrp.MultiUSRP("num_recv_frames=1000")

来增大接收缓冲区的大小（默认大小是 32，单位为 Byte ），以减少溢出的发生概率。在执行这行代码后，实际配置的大小还取决于 USRP 的信号以及其与你的主机相连的方式，但是将 :code:`num_recv_frames` 设置为远大于 32 的值通常会有所帮助。

不建议在更严肃的 SDR 应用中使用快捷函数 :code:`recv_num_samps()` ，因为它隐藏了底层行为，而且在循环调用中每轮都会进行初始化，这是不必要的，对于长时间的采样而言是一个不小的开销。下面的代码与 :code:`recv_num_samps()` 具有相同的功能，相当于把它的细节展开了，但现在我们就有机会修改更多细节行为了：

.. code-block:: python

 import uhd
 import numpy as np

 usrp = uhd.usrp.MultiUSRP()

 num_samps = 10000 # 需要接收的采样点总数
 center_freq = 100e6 # Hz
 sample_rate = 1e6 # Hz
 gain = 50 # dB

 usrp.set_rx_rate(sample_rate, 0)
 usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(center_freq), 0)
 usrp.set_rx_gain(gain, 0)

 # 设置流（Stream） 和接收缓存（Receive Buffer）
 st_args = uhd.usrp.StreamArgs("fc32", "sc16")
 st_args.channels = [0]
 metadata = uhd.types.RXMetadata()
 streamer = usrp.get_rx_stream(st_args)
 recv_buffer = np.zeros((1, 1000), dtype=np.complex64)

 # 启动 Stream
 stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
 stream_cmd.stream_now = True
 streamer.issue_stream_cmd(stream_cmd)

 # 开始接收信号
 samples = np.zeros(num_samps, dtype=np.complex64)
 for i in range(num_samps//1000):
     streamer.recv(recv_buffer, metadata)
     samples[i*1000:(i+1)*1000] = recv_buffer[0]

 # 停止 Stream
 stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
 streamer.issue_stream_cmd(stream_cmd)

 print(len(samples))
 print(samples[0:10])

将 :code:`num_samps` 设置为 10,000，:code:`recv_buffer` 设置为 1000 后，for 循环将运行 10 次，即会调用 10 次 :code:`streamer.recv` 。
请注意，这里我们将 :code:`recv_buffer` 硬编码为 1000，但实际上你也可以使用 :code:`streamer.get_max_num_samps()` 自动找到最大可能的值，通常在 3000 左右。
同时注意，:code:`recv_buffer` 必须是 2D 的，因为在接收多个通道时会使用相同的 API。在我们的情况下，我们只接收了一个通道，所以 :code:`recv_buffer[0]` 才是我们想要的 1D 采样点数组。
目前你不需要理解流的启动/停止方式，但需要知道除了 “连续” 模式之外还有其他选项，比如接收特定数量的样本后让流自动停止。
你可以通过查看每次循环中的 :code:`metadata.error_code` 来检查运行元数据，这包含任何发生过的错误以及其他信息（错误通常会自动在终端中显示，所以这个步骤不是必须的）。

接收增益（Receive Gain）
############################

这个列表展示了不同 USRP 的接收增益范围，它们都从 0 dB 开始，直至下面列出的数字。注意这里不是 dBm，严格来说是 dBm 加上一些未知的偏移量，因为这些设备通常没有经过校准。

* B200/B210/B200-mini: 76 dB
* X310/N210 with WBX/SBX/UBX: 31.5 dB
* X310 with TwinRX: 93 dB
* E310/E312: 76 dB
* N320/N321: 60 dB

你也可以通过在终端中使用命令 :code:`uhd_usrp_probe` 来查看当前 USRP 的接收增益，其显示在输出信息的 RX Frontend （接收前端）部分。

在指定增益时，你可以使用常规的 :code:`set_rx_gain()` 函数，它接受以 dB 为单位的增益值。你也可以使用 :code:`set_normalized_rx_gain()`，它接受一个 0 到 1 的值，并自动将其转换为你正在使用的 USRP 支持范围内对应比例的增益。
当开发兼容多种 USRP 型号的应用时，这种方式更方便。使用归一化的增益的缺点在于它不再以 dB 表示，因此，如果你想定量地设置增益（比如 10dB），那么你得先手动计算出相应的数值。

自动增益控制
############################

一些 USRP，包括 B200 以及 E310 系列，支持自动增益控制（Automatic Gain Control，AGC），它会根据接收到的信号水平自动调整接收增益，以尽可能地填满 ADC 的位数。可以使用以下代码打开 AGC：

.. code-block:: python

 usrp.set_rx_agc(True, 0) # 0 指 Channel0, 即 USRP 的第一个通道

如果你的 USRP 并不支持 AGC，那么运行上面的代码时会抛出异常。在 AGC 开启的情况下，其他设置增益的代码是无效的。

流参数（Stream Args）
**************************

在上文的完整示例代码中，你能看到这样一行代码 :code:`st_args = uhd.usrp.StreamArgs("fc32", "sc16")` 。这个代码中的函数输入就是流参数，其中第一个流参数指的是 UHD 运行时指定的 CPU 数据格式，即采样点在主机中进行处理时的数据类型。UHD 在 Python API 中支持以下 CPU 数据类型：

.. list-table::
   :widths: 15 20 30
   :header-rows: 1
   
   * - 流参数
     - Numpy 数据类型
     - 描述
   * - fc64
     - np.complex128
     - 双精度复数
   * - fc32
     - np.complex64
     - 单精度复数

在 UHD C++ API 的文档中你可能能看到不在上文表格中的数据类型，但是截止本文撰写时，它们尚未在 Python API 中被实现。

第二个流参数是 “过线传输（over-the-wire）” 数据格式，即采样点通过 USB/以太网/SFP 线缆在主机和 USRP 之间传输状态下的数据类型。对于 Python API，选项包括：“sc16”、“sc12” 和 “sc8”，其中 “sc12” 只被部分 USRP 支持。这个参数很重要，因为 USRP 和主机之间的传输带宽通常是采样率的瓶颈，通过从 16 位切换到 8 位，你可能得以采用更高的采样率。还要注意，许多 USRP 的 ADC 仅限于 12 或 14 位，能设置 “sc16” 并不意味着 ADC 变成了 16 位的。

关于 :code:`st_args` 以及通道，请参阅下面的子设备（Subdevice）和通道（Channels） 小节。

************************
发射
************************

与 :code:`recv_num_samps()` 方便函数类似，UHD 提供了 :code:`send_waveform()` 快捷函数用于传输一批采样点，下面展示了一个例子。如果你指定的时长（以秒为单位）长于所提供的信号，它将简单地重复该信号。此外，采样点的值将被保持在 -1.0 到 1.0 之间。

.. code-block:: python

 import uhd
 import numpy as np
 usrp = uhd.usrp.MultiUSRP()
 samples = 0.1*np.random.randn(10000) + 0.1j*np.random.randn(10000) # 创造随机信号
 duration = 10 # 以秒为单位
 center_freq = 915e6
 sample_rate = 1e6
 gain = 20 # [dB] 建议一开始设置小一点，按照实际情况调整
 usrp.send_waveform(samples, duration, center_freq, sample_rate, [0], gain)

如果你想了解这个发射快捷函数的更多细节，可以查看 `这里 <https://github.com/EttusResearch/uhd/blob/master/host/python/uhd/usrp/multi_usrp.py>`_ 。


发射增益（Transmit Gain）
###########################

发射增益的可配置范围与接收增益类似，根据 USRP 型号的不同而变化，从 0 dB 到下面指定的数字：

* B200/B210/B200-mini: 90 dB
* N210 with WBX: 25 dB
* N210 with SBX or UBX: 31.5 dB
* E310/E312: 90 dB
* N320/N321: 60 dB

如果希望使用 0 到 1 的归一化值来指定发射增益，也有 :code:`set_normalized_tx_gain()` 函数供你使用。

**************
同时收发
**************

如果你想用同一台 USRP 同时进行信号发射与接收，那么关键操作在于你必须在同一个进程的不同线程中进行，因为 USRP 不能被多进程同时调用。举个例子，在 UHD 源代码 `txrx_loopback_to_file.cpp <https://github.com/EttusResearch/uhd/blob/master/host/examples/txrx_loopback_to_file.cpp>`_ 中，一个单独的线程被创建用于运行发射机，而接收则在主线程中进行。
你也可以像 Python 示例代码 `benchmark_rate <https://github.com/EttusResearch/uhd/blob/master/host/examples/python/benchmark_rate.py>`_ 中那样，分别创建两个线程，一个用于发射，一个用于接收。
因为篇幅的原因，这里没有展示完整的示例代码，但是 Ettus 的 benchmark_rate.py 示例代码的确是一个很好的学习起点。

*********************************************************
子设备，通道，与天线
*********************************************************

在使用 USRP 时，初学者常常因为子设备（Subdevice）和通道（Channels）的选择而疑惑。你可能注意到了，上面的每个例子中我们都使用了通道 0，而没有指定任何与子设备相关的内容。如果你使用的是 B210，只想使用 RF:B 而不是 RF:A，那么你只需要选择通道 1 而不是 0。但是在像 X310 这样具有两个射频子板（即子设备）的 USRP 上，你必须告诉 UHD 你想使用其中的子板 A 还是 B，以及该子板上的哪个通道，例如：

.. code-block:: python

 usrp.set_rx_subdev_spec("B:0")

如果你想使用 TX/RX 射频端口代替 RX2 射频端口（默认配置），可以这样轻松解决：

.. code-block:: python

 usrp.set_rx_antenna('TX/RX', 0) # 把 channel 0 定义为 'TX/RX'

这行代码本质上是配置了 USRP 上的射频交换器，使得其从另一个 SMA 射频端口传导信号。

为了通过两个通道同时进行收/发，你需要将 :code:`st_args.channels = [0]` 改为 :code:`[0,1]`。这时接收样本缓冲区的大小将变为（2, N）而不是（1, N）。请记住，大多数 USRP 的两个通道共享一个本地振荡器（LO），所以你通常不能把两个通道分别调谐到不同的中心频率上。

**********************************
与 10 MHz 参考信号以及 PPS 同步
**********************************

相比其他 SDR 设备，使用 USRP 的巨大优势之一是其能够与外部源或机载 `GPSDO <https://www.ettus.com/all-products/gpsdo-tcxo-module/>`_ 同步，从而支持 TDOA 等多接收机应用。如果你已经将外部 10 MHz 参考信号和 PPS （每秒脉冲）源连接至你的 USRP，你需要确保在初始化 USRP 后调用以下两行代码：

.. code-block:: python

 usrp.set_clock_source("external")
 usrp.set_time_source("external")

如果你使用的是机载 GPSDO ，代码则是：

.. code-block:: python

 usrp.set_clock_source("gpsdo")
 usrp.set_time_source("gpsdo")

这样一来，频率同步就搞定了：USRP 混频器中的本振（LO）现在将会与外部源或机载 `GPSDO <https://www.ettus.com/all-products/gpsdo-tcxo-module/>`_ 相连。
在定时同步方面，你可能希望 USRP 精确地在 PPS 上开始采样，代码可以这样写：

.. code-block:: python

 # 请复制上文接收示例代码，包括 # Start Stream 之前的所有内容

 # 等待 1 个 PPS 发生，然后将下一个 PPS 的时间设置为 0.0
 time_at_last_pps = usrp.get_time_last_pps().get_real_secs()
 while time_at_last_pps == usrp.get_time_last_pps().get_real_secs():
     time.sleep(0.1) # 等待 1 个 PPS 发生，如果这个循环永远不结束，表示 PPS 信号没有被检测到
 usrp.set_time_next_pps(uhd.libpyuhd.types.time_spec(0.0))
 
 # 配置接收参数：从上一个 PPS 信号恰好的 3 秒后接收由 num_samps 指定数量的采样点
 stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
 stream_cmd.num_samps = num_samps
 stream_cmd.stream_now = False
 stream_cmd.time_spec = uhd.libpyuhd.types.time_spec(3.0) # 设置起始时间（你可以尝试调整此参数）
 streamer.issue_stream_cmd(stream_cmd)
 
 # 开始接收：recv() 将交替返回 0 和采样点，0 可以作为采样结束的标志
 waiting_to_start = True # 用于识别循环状态（参见上一行注释）
 nsamps = 0
 i = 0
 samples = np.zeros(num_samps, dtype=np.complex64)
 while nsamps != 0 or waiting_to_start:
     nsamps = streamer.recv(recv_buffer, metadata)
     if nsamps and waiting_to_start:
         waiting_to_start = False
     elif nsamps:
         samples[i:i+nsamps] = recv_buffer[0][0:nsamps]
     i += nsamps

如果以上代码没有按照预期运行，但是又没有报错，你可以尝试把 3.0 改为 1.0 到 5.0 之间的任意值试试。你也可以在调用 :code:`recv()` 后检查元数据，即检查 :code:`if metadata.error_code != uhd.types.RXMetadataErrorCode.none:` 。

为了 Debug，你可以通过检查 :code:`usrp.get_mboard_sensor("ref_locked", 0)` 的返回值来验证 10 MHz 信号是否传递到了 USRP。而对于 PPS 信号而言，如果它没有传递到 USRP，那么上面代码中的第一个 while 循环将永远不会结束。

****
GPIO
****

大多数 USRP 都包含一个 GPIO 接头。在 B200/B210 上，它是 J504 接头，而在 X310 上则位于前面板。

为了介绍 GPIO，首先我们定义一些 Ettus 的术语： **CTRL** 设置引脚是否 ATR（automatic，自动）控制（1 表示 ATR 控制，0 表示手动控制）。**DDR** （Data Direction Register， 数据方向寄存器）设置 GPIO 是输出（0）还是输入（1）。 **OUT** 用于手动设置引脚的值（仅在手动 CTRL 模式下使用）。

以使用 X310 前面的 “AUX I/O” 接头作为 GPIO 输出为例，更多信息请参见 `此文档 <https://files.ettus.com/manual/page_gpio_api.html>`_ 。

.. code-block:: python

  import uhd
  import time
  usrp = uhd.usrp.MultiUSRP()
  usrp.set_gpio_attr('FP0A', 'CTRL', 0x000, 0xFFF)
  usrp.set_gpio_attr('FP0A', 'DDR', 0xFFF, 0xFFF)
  for i in range(10):
      print("Off")
      usrp.set_gpio_attr('FP0A', 'OUT', 0x000, 0xFFF)
      time.sleep(1)
      print("On")
      usrp.set_gpio_attr('FP0A', 'OUT', 0xFFF, 0xFFF)
      time.sleep(1)