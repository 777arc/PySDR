.. _intro-chapter:

#############
はじめに
#############

***************************
目的と対象読者
***************************

まずはじめに、いくつかの重要な用語を説明します。

**ソフトウェア無線 (Software-Defined Radio、SDR):**
    *概念* としてのSDRとは、従来ハードウェアで行われていた無線通信や高周波(RF)アプリケーションに特有の信号処理作業を、ソフトウェアによって行うことです。
    このソフトウェアは汎用コンピュータ(CPU)、FPGA、あるいはGPU上で動作し、リアルタイムアプリケーションや、保存済みの信号のオフライン処理に利用できます。
    類似する用語にsoftware radioやRF digital signal processingがあります。
    
    *物理的な物* としてのSDRは一般的に、アンテナを接続してRF信号を受信し、
    そのデジタル化されたRF信号をコンピュータに送信し処理または記録できるデバイスを指します(コンピュータとの通信にはUSB、Ethernet、PCIなどが使用されます)。
    多くのSDRは送信機能も備えており、コンピュータが送ったSDRに送った信号を、指定したRF周波数で送信できます。
    また、組み込み型のSDRには、オンボードコンピュータが搭載されているものもあります。


**デジタル信号処理 (Digital Signal Processing、DSP):**
    信号をデジタル処理すること。ここでは、特にRF信号を対象としています。

この教材は、DSP、SDR、そして無線通信への実践的な入門ガイドとして作られています。
想定している読者は次のような人です：

#. SDRを *使って* 面白いことをしてみたい人
#. Pythonに慣れている人
#. DSPや無線通信、SDRについては比較的初心者の人
#. 数式よりもアニメーションなど視覚的な説明を好む人
#. まず概念を学んだ *あとで* 数式を理解するほうが得意な人
#. 1,000ページの分厚い教科書ではなく、簡明な解説を求めている人

例えば、無線通信関連の仕事に就きたいと考えているコンピュータサイエンス専攻の学生が対象になりますが、プログラミング経験がありSDRを学びたい人なら誰でも活用できます。
そのため、通常のDSPの講義で扱われるような難解な数学に頼ることなく、DSP技術を理解するために必要な理論をカバーしています。
難解な数式を多用する代わりに、概念を伝えるために数多くの図やアニメーションを活用しています。
例えば、下に示したフーリエ級数の複素平面アニメーションのようなものです。
私は、ビジュアルや実践的な演習を通じて概念を学んだ *あとに* 数式を理解するのが最も効果的だと考えています。
アニメーションを多用しているため、PySDRが印刷物としてAmazonで売られることはないでしょう。

.. image:: ../_images/fft_logo_wide.gif
   :scale: 70 %   
   :align: center
   :alt: The PySDR logo created using a Fourier transform
   
この教材は、DSPやSDRの概念を素早くスムーズに紹介し、読者が賢くDSPを実践し、SDRを使いこなせるようになることを目的としています。
この教材は、全てのDSP/SDR分野を網羅するリファレンステキストを目指しているわけではありません。
そのような優れた教科書はすでに数多く存在しており、例えば `Analog DevicesのSDRの教科書
<https://www.analog.com/en/education/education-library/software-defined-radio-for-engineers.html>`_と `dspguide.com <http://www.dspguide.com/>`_ が挙げられます。
三角関数の公式やシャノン限界などは、いつでもGoogleを使って思い出せば大丈夫です。
この教材はDSPやSDRの世界への入り口と考えてください。
従来の講義や教科書に比べて、時間もお金もかけずに始められる内容になっています。

基礎的なDSP理論をカバーするために、電気工学分野で一般的な「Signals and Systems(信号とシステム)」という1学期分の内容を、数章に凝縮しています。
DSPの基礎が一通り終わった後はSDRの話題に進みますが、DSPや無線通信に関連する概念も引き続き教材に登場します。

コード例はPythonで提供されています。
これらの例では、配列操作や高等数学の標準的ライブラリであるNumPyを利用しています。
また、信号や配列、複素数を簡単に可視化できるPythonの描画ライブラリMatplotlibも使用しています。
なお、一般にPythonはC++よりも「遅い」ものの、Python/NumPy内のほとんどの数学関数はC/C++で実装され、高度に最適化されています。
同様に、使用しているSDR APIも、C/C++で書かれた関数やクラスに対するPythonバインディングです。
Pythonの経験があまりない方でも、MATLAB、Ruby、Perlなどに慣れているなら、Pythonの文法に慣れるだけで問題なく取り組めるでしょう。

***************
貢献
***************

もしPySDRが役に立ったと感じたら、この教材に興味を持ちそうな同僚や学生、生涯学習者(lifelong learners)たちにぜひシェアしてください。
また、PySDRのPatreonページ `PySDR Patreon <https://www.patreon.com/PySDR>`_  から寄付を行い、感謝の意を表明することもできます。
寄付をすると、あなたの名前が各ページ左側の章リスト下に掲載されます。

この教材のどこかを読み進め、質問・コメント・提案などを marc@pysdr.org にメールしてくれた場合、それだけであなたはこの教材に貢献したことになります！
さらに、教材のGitHubページ `textbook's GitHub page <https://github.com/777arc/PySDR/tree/master/content>`_  上で直接ソースを編集することも可能です(変更を加えるにはプルリクエストが作成することになります)。
バグ修正や改善案について、IssueやPull Request(PR)を送るのも大歓迎です。
価値あるフィードバックや修正を提供してくれた方は、このあとに続く謝辞セクションに永続的に名前を記載します。
Gitの操作に自信がないけれど提案したい内容がありますか? そういう時は、遠慮なく marc@pysdr.org までメールしてください。

*****************
謝辞
*****************

この教材の一部でも読んでフィードバックを寄せてくださったすべての方々に感謝します。特に以下の皆さまに感謝いたします。

- `Barry Duggan <http://github.com/duggabe>`_
- Matthew Hannon
- James Hayek
- Deidre Stuffer
- Tarik Benaddi for `translating PySDR to French <https://pysdr.org/fr/index-fr.html>`_
- `Daniel Versluis <https://versd.bitbucket.io/content/about.html>`_ for `translating PySDR to Dutch <https://pysdr.org/nl/index-nl.html>`_
- `mrbloom <https://github.com/mrbloom>`_ for `translating PySDR to Ukrainian <https://pysdr.org/ukraine/index-ukraine.html>`_
- `Yimin Zhao <https://github.com/doctormin>`_ for `translating PySDR to Simplified Chinese <https://pysdr.org/zh/index-zh.html>`_
- `Eduardo Chancay <https://github.com/edulchan>`_ for `translating PySDR to Spanish <https://pysdr.org/es/index-es.html>`_

As well as all `PySDR Patreon <https://www.patreon.com/PySDR>`_ supporters!
