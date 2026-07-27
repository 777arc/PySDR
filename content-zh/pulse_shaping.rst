.. _pulse-shaping-chapter:

#######################
脉冲整型
#######################

在本章中我们将介绍脉冲整形（Pulse Shaping），符号间干扰（Inter-Symbol-Interference, ISI），匹配滤波器（Matched Filter）和升余弦滤波器（Raised-Cosine Filter）。
最后我们将使用 Python 为 BPSK 符号添加脉冲整形。
你可以把本章节视为滤波器章节的后续章节，在此我们将深入讨论脉冲整形。

**********************************
符号间干扰（ISI）
**********************************

在 :ref:`filters-chapter` 章节中，我们学习到了方块形状的符号/脉冲会占用相当宽的频谱，我们可以通过"整形"我们的脉冲来大大减少使用的频谱。然而，你不能使用任何低通滤波器，否则可能会出现符号间干扰（Inter-Symbol-Interference，ISI），即符号相互干扰。

当我们传输数字符号时，我们是连续地逐个传输它们的（而不是在符号之间等待一段时间）。当你施加脉冲整形滤波器时，它会在时域中拉长脉冲（以便在频域中压缩），从而导致相邻符号彼此重叠。只要你的脉冲整形滤波器满足以下准则，这种重叠就不会有问题：除了其中一个脉冲之外，所有脉冲在符号周期 :math:`T` 的每一个整数倍处的叠加值必须为零。通过下面这幅图可以最直观地理解这个概念：

.. image:: ../_images/pulse_train.svg
   :align: center
   :target: ../_images/pulse_train.svg
   :alt: A pulse train of sinc pulses

如图所示，在 :math:`T` 的每一个整数倍处，只有一个脉冲存在峰值，而其余所有脉冲的值都为 0（它们穿过了 x 轴）。当接收机对信号进行采样时，它恰好在完美的时刻（即脉冲的峰值处）进行采样，这意味着只有那个时间点才真正重要。通常，接收机中会有一个符号同步模块来确保在峰值处对符号进行采样。

**********************************
匹配滤波器
**********************************

在无线通信中，我们使用的一个技巧叫做匹配滤波（Matched Filtering）。要理解匹配滤波，你首先需要理解以下两点：

1. 上面讨论的那些脉冲只需要在 *接收端* 采样之前完美对齐即可。在那之前，即使存在 ISI 也没关系，也就是说，信号可以带着 ISI 在空中传播，这是完全没问题的。

2. 我们希望在发射机中使用低通滤波器来减少信号占用的频谱。但接收机同样需要一个低通滤波器来尽可能消除信号旁边的噪声/干扰。因此，我们在发射机（Tx）端有一个低通滤波器，在接收机（Rx）端也有一个低通滤波器，而采样发生在这两个滤波器（以及无线信道的影响）之后。

在现代通信中，我们会将脉冲整形滤波器均等地拆分到 Tx 和 Rx 两端。虽然它们 *不一定* 非得是相同的滤波器，但从理论上讲，在 AWGN 环境下最大化 SNR 的最优线性滤波器就是在 Tx 和 Rx 使用 *相同* 的滤波器。这种策略被称为 "匹配滤波器" 的概念。

另一种理解匹配滤波器的方式是：接收机将接收到的信号与已知的模板信号进行相关运算。这里的模板信号本质上就是发射机发送的脉冲，与施加在其上的相位/幅度偏移无关。回忆一下，滤波是通过卷积来完成的，而卷积基本上就是相关运算（事实上，当模板是对称的时候，它们在数学上是完全等价的）。将接收信号与模板进行相关运算，能让我们最大限度地恢复所发送的内容，这也是为什么它在理论上是最优的。打个比方，想象一个图像识别系统，它使用一个人脸模板并通过 2D 相关运算来寻找人脸：

.. image:: ../_images/face_template.png
   :scale: 70 %
   :align: center

**********************************
将滤波器拆分为两半
**********************************

我们具体要怎么把一个滤波器拆成两半呢？卷积运算具有结合律，即：

.. math::
 (f * g) * h = f * (g * h)

假设 :math:`f` 是我们的输入信号，而 :math:`g` 和 :math:`h` 是两个滤波器。先用 :math:`g` 滤波 :math:`f` ，再用 :math:`h` 滤波，等效于使用一个为 :math:`g * h` 的滤波器进行滤波。

另外，回忆一下时域卷积等于频域相乘：

.. math::
 g(t) * h(t) \leftrightarrow G(f)H(f)

要将一个滤波器拆分为两半，你可以对其频率响应取平方根。

.. math::
 X(f) = X_H(f) X_H(f) \quad \mathrm{where} \quad X_H(f) = \sqrt{X(f)}

下图展示了一个简化的发射和接收链路图。其中升余弦（RC）滤波器被拆分为两个根升余弦（RRC）滤波器：发射端的一个是脉冲整形滤波器，接收端的一个是匹配滤波器。它们共同作用，使得解调器处的脉冲看起来就像经过了单个 RRC 滤波器进行脉冲整形一样。

.. image:: ../_images/splitting_rc_filter.svg
   :align: center
   :target: ../_images/splitting_rc_filter.svg
   :alt: A diagram of a transmit and receive chain, with a Raised Cosine (RC) filter being split into two Root Raised Cosine (RRC) filters

**********************************
具体的脉冲整形滤波器
**********************************

我们已经知道我们需要做到：

1. 设计一个能降低信号带宽（以减少频谱占用）的滤波器，同时要求在每个符号间隔处，除了一个脉冲之外，其余脉冲的叠加值均为零。

2. 将该滤波器拆分为两半，一半放在 Tx，另一半放在 Rx。

让我们来看看一些常用的脉冲整形滤波器。

升余弦滤波器
#########################

最常用的脉冲整形滤波器似乎是 "升余弦" 滤波器。它是一个很好的低通滤波器，能够限制信号占用的带宽，同时它还具有在 :math:`T` 的整数倍处求和为零的特性：

.. image:: ../_images/raised_cosine.svg
   :align: center
   :target: ../_images/raised_cosine.svg
   :alt: The raised cosine filter in the time domain with a variety of roll-off values

请注意，上图是时域图，描绘的是滤波器的脉冲响应。参数 :math:`\beta` 是升余弦滤波器唯一的参数，它决定了滤波器在时域中衰减到零的速度，而这与其在频域中衰减的速度成反比：

.. image:: ../_images/raised_cosine_freq.svg
   :align: center
   :target: ../_images/raised_cosine_freq.svg
   :alt: The raised cosine filter in the frequency domain with a variety of roll-off values

之所以称之为升余弦滤波器，是因为当 :math:`\beta = 1` 时，其频域响应是一个半周期的余弦波，被抬升到 x 轴上。

升余弦滤波器的脉冲响应的数学表达式为：

.. math::
 h(t) = \mathrm{sinc}\left( \frac{t}{T} \right) \frac{\cos\left(\frac{\pi\beta t}{T}\right)}{1 - \left( \frac{2 \beta t}{T}   \right)^2}

关于 :math:`\mathrm{sinc}()` 函数的更多信息可以参阅 `这里 <https://en.wikipedia.org/wiki/Sinc_function>`_ 。你可能在其他地方看到包含 :math:`\frac{1}{T}` 缩放因子的公式；这个因子使滤波器具有单位增益，即输出信号的功率与输入信号相同（这是设计滤波器时的常见做法）。然而，我们要将其应用于由符号组成的脉冲序列（例如 1 和 -1），我们不希望这些符号的幅度在脉冲整形后发生变化，因此我们省略了这个缩放因子。当我们深入到 Python 示例并绘制输出时，这一点会更加清晰。

记住：我们要将这个滤波器均等地拆分到 Tx 和 Rx。接下来介绍根升余弦（RRC）滤波器！

根升余弦滤波器
#########################

根升余弦（Root Raised-Cosine，RRC）滤波器才是我们实际在 Tx 和 Rx 中实现的滤波器。正如前文所述，两者组合起来就构成了一个完整的升余弦滤波器。因为将滤波器拆分为两半涉及到频域上的开平方运算，所以脉冲响应看起来会稍微复杂一些：

.. image:: ../_images/rrc_filter.png
   :scale: 70 %
   :align: center

好在这是一个非常常用的滤波器，有大量现成的实现可以使用，包括 `Python 实现 <https://commpy.readthedocs.io/en/latest/generated/commpy.filters.rrcosfilter.html>`_ 。

其他脉冲整形滤波器
###########################

其他滤波器包括高斯滤波器（Gaussian Filter），其脉冲响应类似高斯函数。还有 sinc 滤波器，它等效于 :math:`\beta = 0` 时的升余弦滤波器。sinc 滤波器更接近理想滤波器，即它能够在几乎没有过渡区域的情况下消除不需要的频率。

**********************************
滚降因子
**********************************

让我们仔细研究一下参数 :math:`\beta` 。它是一个介于 0 到 1 之间的数值，称为 "滚降因子（Roll-off Factor）"，有时也被称为 "过量带宽（Excess Bandwidth）"。它决定了滤波器在时域中衰减到零的速度。回忆一下，要用作滤波器，脉冲响应必须在两侧衰减到零：

.. image:: ../_images/rrc_rolloff.svg
   :align: center
   :target: ../_images/rrc_rolloff.svg
   :alt: Plot of the raised cosine roll-off parameter

:math:`\beta` 越小，所需的滤波器抽头数就越多。当 :math:`\beta = 0` 时，脉冲响应永远不会完全衰减到零，因此我们试图将 :math:`\beta` 设置得尽可能低，同时又不会引发其他问题。滚降因子越低，对于给定的符号速率，我们能够在频率上把信号压缩得越紧凑，这一点始终很重要。

一个常用的公式用于估算给定符号速率和滚降因子下的带宽（单位为 Hz）：

.. math::
    \mathrm{BW} = R_S(\beta + 1)

:math:`R_S` 是符号速率（单位为 Hz）。在无线通信中，我们通常选择 0.2 到 0.5 之间的滚降因子。作为经验法则，一个使用符号速率 :math:`R_S` 的数字信号将占用略多于 :math:`R_S` 的频谱，这里包括正频率和负频率部分。一旦我们将信号上变频并发射出去，两侧的频谱都很重要。如果我们以每秒 100 万个符号（MSps）的速率传输 QPSK 信号，它将占用大约 1.3 MHz 的频谱。数据速率将是 2 Mbps（回忆一下 QPSK 每个符号携带 2 比特），其中包括信道编码和帧头等开销。

**********************************
Python 练习
**********************************

作为 Python 练习，让我们来对一些脉冲进行滤波和整形。我们将使用 BPSK 符号，因为它更容易可视化——在脉冲整形步骤之前，BPSK 就是传输 1 或 -1，其 "Q" 分量等于零。由于 Q 为零，我们只需绘制 I 分量即可，这样看起来更简单。

在这个仿真中，我们将使用每个符号 8 个采样点，并且不使用看起来像方波的 1 和 -1 信号，而是使用由脉冲（冲激）组成的脉冲序列。当你把一个脉冲通过滤波器时，输出就是脉冲响应（这也是 "脉冲响应" 这个名称的由来）。因此，如果你想要一系列脉冲，就应该使用中间填充零的冲激序列，从而避免产生方形脉冲。

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal

    num_symbols = 10
    sps = 8

    bits = np.random.randint(0, 2, num_symbols) # Our data to be transmitted, 1's and 0's

    x = np.array([])
    for bit in bits:
        pulse = np.zeros(sps)
        pulse[0] = bit*2-1 # set the first value to either a 1 or -1
        x = np.concatenate((x, pulse)) # add the 8 samples to the signal
    plt.figure(0)
    plt.plot(x, '.-')
    plt.grid(True)
    plt.show()

.. image:: ../_images/pulse_shaping_python1.png
   :scale: 80 %
   :align: center
   :alt: A pulse train of impulses in the time domain simulated in Python

此时我们的符号仍然是 1 和 -1。不要纠结于我们使用了冲激这件事。实际上，*不要* 去可视化冲激响应，而是把它当作一个数组来理解可能会更简单：

.. code-block:: python

 bits: [0, 1, 1, 1, 1, 0, 0, 0, 1, 1]
 BPSK symbols: [-1, 1, 1, 1, 1, -1, -1, -1, 1, 1]
 Applying 8 samples per symbol: [-1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, ...]

我们将使用 :math:`\beta` 为 0.35 的升余弦滤波器，并将其设为 101 个抽头长度，以给信号足够的时间衰减到零。虽然升余弦方程需要我们提供符号周期和时间向量 :math:`t` ，但我们可以假设 **采样** 周期为 1 秒来 "归一化" 我们的仿真。这意味着我们的符号周期 :math:`Ts` 为 8，因为我们有每个符号 8 个采样点。于是我们的时间向量将是一个整数列表。根据升余弦方程的特性，我们希望 :math:`t=0` 位于中心。我们将生成从 -51 到 +51 的长度为 101 的时间向量。

.. code-block:: python

    # Create our raised-cosine filter
    num_taps = 101
    beta = 0.35
    Ts = sps # Assume sample rate is 1 Hz, so sample period is 1, so *symbol* period is 8
    t = np.arange(num_taps) - (num_taps-1)//2
    h = np.sinc(t/Ts) * np.cos(np.pi*beta*t/Ts) / (1 - (2*beta*t/Ts)**2)
    plt.figure(1)
    plt.plot(t, h, '.')
    plt.grid(True)
    plt.show()


.. image:: ../_images/pulse_shaping_python2.png
   :scale: 80 %
   :align: center

注意输出确实衰减到了零。我们使用每个符号 8 个采样点这一设定决定了这个滤波器看起来有多窄以及它衰减到零的速度有多快。上面的脉冲响应看起来就像一个典型的低通滤波器，我们实际上无法仅从外观上判断它是专门的脉冲整形滤波器还是其他普通的低通滤波器。

最后，我们可以用该滤波器对信号 :math:`x` 进行滤波并查看结果。不要过分关注代码中引入的 for 循环，我们会在代码块之后讨论它存在的原因。

.. code-block:: python

    # Filter our signal, in order to apply the pulse shaping
    x_shaped = np.convolve(x, h)
    plt.figure(2)
    plt.plot(x_shaped, '.-')
    for i in range(num_symbols):
        plt.plot([i*sps+num_taps//2,i*sps+num_taps//2], [0, x_shaped[i*sps+num_taps//2]])
    plt.grid(True)
    plt.show()

.. image:: ../_images/pulse_shaping_python3.svg
   :align: center
   :target: ../_images/pulse_shaping_python3.svg

这个结果信号是由许多脉冲响应叠加而成的，其中大约一半的脉冲响应先乘以了 -1。虽然看起来可能很复杂，但我们会一起来分析。

首先，由于滤波器和卷积运算的特性，数据前后会存在一些瞬态采样点。这些额外的采样点会包含在我们的传输中，但它们并不包含真正的脉冲 "峰值"。

其次，竖直线是在 for 循环中创建的，用于辅助可视化。它们旨在标示 :math:`Ts` 间隔出现的位置，这些间隔代表接收机对信号进行采样的位置。观察可以发现，在每个 :math:`Ts` 间隔处，曲线的值恰好为 1.0 或 -1.0，这使得它们成为理想的采样时刻。

如果我们要将这个信号上变频并发射出去，接收机就需要确定 :math:`Ts` 的边界在哪里，例如使用符号同步算法。这样接收机才能 *精确地* 知道何时采样以获得正确的数据。如果接收机采样稍早或稍晚，由于 ISI 的存在，它会看到略有偏差的值；如果偏差太大，接收到的就会是一堆奇怪的数字。

下面是一个使用 GNU Radio 创建的示例，它展示了在正确和错误的时刻采样时 IQ 图（即星座图）的样子。原始脉冲的比特值已标注在图上。

.. image:: ../_images/symbol_sync1.png
   :scale: 50 %
   :align: center

下图代表理想的采样时刻以及对应的 IQ 图：

.. image:: ../_images/symbol_sync2.png
   :scale: 40 %
   :align: center
   :alt: GNU Radio simulation showing perfect sampling as far as timing

与之对比，下图展示了最差的采样时刻。注意星座图中出现了三个聚类。我们恰好在每两个符号的正中间进行采样，因此采样值会有很大偏差。

.. image:: ../_images/symbol_sync3.png
   :scale: 40 %
   :align: center
   :alt: GNU Radio simulation showing imperfect sampling as far as timing

这是另一个不良采样时刻的例子，它介于理想情况和最差情况之间。注意此时出现了四个聚类。在高 SNR 下，我们或许勉强能用这个采样时刻，但这并不可取。

.. image:: ../_images/symbol_sync4.png
   :scale: 40 %
   :align: center

请记住，时域图中没有显示 Q 值，因为它们近似为零，所以 IQ 图仅在水平方向上展开。
