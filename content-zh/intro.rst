.. _intro-chapter:

#############
简介
#############

***************************
目标及受众
***************************

首先，需要介绍几个重要术语：

**软件定义无线电（SDR，Software-Defined Radio）：**
    一种无线电广播通信技术，它基于软件定义的无线通信协议而非通过硬连线实现。

**数字信号处理（DSP， Digital Signal Processing）：**
    利用数字计算方法对信号进行处理。具体到我们的案例，就是针对射频（RF）信号的处理。

本教程是对 DSP、SDR 和无线通信的实用入门，适合以下读者：

#. 对使用 SDR 进行一些炫酷项目感兴趣的人
#. 对 Python 编程有一定了解的人
#. 在 DSP、SDR、无线通信领域刚起步的人
#. 喜欢使用视觉动画学习，而不仅是公式的人
#. 通常喜欢先理解概念，然后再看公式的人
#. 更喜欢阅读简明扼要的解释，而不是浏览长达 1000 页的教程的人

例如，如果你是一名计算机科学专业的学生，并对毕业后从事无线通信的工作有兴趣，那么本书将会是你的理想选择。
此书不仅适合你，同样适合对学习 SDR 有浓厚兴趣并具有编程经验的任何人。
为了实现这个目标，我们在这本书中包含了理解 DSP 技术所需的理论基础，但剔除了传统 DSP 课程中的复杂数学部分。
为了便于理解，我们并未沉浸在公式中，而是通过大量的图示和动画来解释概念。
例如，下图展示了在复平面上制造的傅立叶级数动画。
我认为，理解公式的真正含义需要在通过视觉和实践理解了相关概念之后。
由于本书大量地使用了动画，因此你不会在亚马逊等地方找到这本 PySDR 的纸质版。

.. image:: ../_images/fft_logo_wide.gif
   :scale: 70 %
   :align: center

此教科书目标在于，快速且清晰地阐述概念，让读者能够有效利用 DSP 和 SDR。然而，它并非一部详尽无遗的工具书，涵盖所有 DSP/SDR 细节。因为已有许多优质共类参考：
比如 `Analog Device出版的SDR教材 <https://www.analog.com/en/education/education-library/software-defined-radio-for-engineers.html>`_
和 `dspguide.com <http://www.dspguide.com/>`_ 网站。
而对于三角恒等式或香农极限等知识，亦可 Google 查阅。
因此，此教科书应当被视为深入 DSP 和 SDR 领域的引导之门，相较传统课程和教科书，它更为简洁和更省时间金钱。

本书将电子工程专业中典型课程《信号与系统》所覆盖的基本 DSP 理论压缩成几章内容。
在讲解完 DSP 的基础知识后，我们将转入 SDR 的主题，然而，DSP 和无线通信的概念会时刻穿插其中。

示例代码均采用 Python 语言编写，且使用了 Python 的标准库 NumPy 来处理数组和进行高级数学计算，
同时依赖于 Matplotlib 这一图形库来可视化信号、数组和复数。
虽然 Python 的执行速度一般低于 C++，但其大部分数学函数已用 C/C++ 实现并进行了优化。
此外，我们使用的 SDR API 其实是编写 C/C++ 函数/类的 Python 绑定。
因此，对于拥有 MATLAB、Ruby 或 Perl 基础但对 Python 不熟悉的读者，只要掌握 Python 的语法就无需担心。

***************
参与贡献
***************

如果你从 PySDR 中获得了帮助，别忘了与你的同事、学生和其他可能对这些资料感兴趣的终身学习者分享。你还可以通过在 PySDR 的 `Patreon <https://www.patreon.com/PySDR>`_ 上捐款来表达你的支持，你的名字将会显示在每章下面的章节列表的左侧。

此外，欢迎你通过 pysdr@vt.edu 向我发送问题、评论或者建议，你将为这本教科书的完善做出贡献！
你还可以直接在教科书的 `GitHub页面 <https://github.com/777arc/PySDR/tree/master/content>`_ 上编辑源代码，欢迎你提交 Issue 或 Pull Request。
为本书提供有价值反馈或修复的人将被永久列在下面的致谢部分。如果你有建议但不熟悉  Git的操作，也可以直接通过 pysdr@vt.edu 与我取得联系。

*****************
致谢
*****************

谢谢所有提供反馈的读者，特别感谢以下人员：

- `Barry Duggan <http://github.com/duggabe>`_
- Matthew Hannon
- James Hayek
- Deidre Stuffer
- Tarik Benaddi 将 PySDR 翻译为 `法语 <https://pysdr.org/fr/index-fr.html>`_
- `Daniel Versluis <https://versd.bitbucket.io/content/about.html>`_ 将 PySDR 翻译为 `荷兰语 <https://pysdr.org/nl/index-nl.html>`_
- `mrbloom <https://github.com/mrbloom>`_  将 PySDR 翻译为 `乌克兰语 <https://pysdr.org/ukraine/index-ukraine.html>`_
- `Yimin Zhao <https://github.com/doctormin>`_ 将 PySDR 翻译为 `简体中文 <https://pysdr.org/zh/index-zh.html>`_
- `Eduardo Chancay <https://github.com/edulchan>`_ 将 PySDR 翻译为 `西班牙语 <https://pysdr.org/es/index-es.html>`_

以及所有 `PySDR Patreon <https://www.patreon.com/PySDR>`_ 支持者!
