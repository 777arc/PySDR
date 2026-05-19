.. _intro-chapter:

#############
Einführung
#############

***************************
Zweck und Zielgruppe
***************************

Zunächst einige wichtige Begriffe:

**Software-Defined Radio (SDR):**
    Als *Konzept* bezieht es sich auf die Verwendung von Software zur Durchführung von Signalverarbeitungsaufgaben, die traditionell von Hardware durchgeführt wurden, die speziell für Radio-/HF-Anwendungen angedacht wurden. Diese Software kann auf einem Mikroprozessor (CPU), FPGA oder sogar GPU ausgeführt werden und kann für Echtzeitanwendungen oder die Offline-Verarbeitung aufgezeichneter Signale verwendet werden. Synonyme sind „Software-Radio" und „digitale HF-Signalverarbeitung".

    Als *Gerät* (z. B. „ein SDR") bezieht es sich typischerweise auf ein Gerät, an das du eine Antenne anschließen und HF-Signale empfangen kannst, wobei die digitalisierten HF-Samples zur Verarbeitung oder Aufzeichnung an einen Computer gesendet werden (z. B. der Transfer über USB, Ethernet, PCI). Viele SDRs haben auch Sendefähigkeiten, die es dem Computer ermöglichen, Samples an das SDR zu senden, das dann das Signal auf einer bestimmten HF-Frequenz zu übertragen. Einige eingebettete SDRs enthalten einen integrierten Computer.

**Digitale Signalverarbeitung (DSP):**
    Die digitale Verarbeitung von Signalen; in unserem Fall HF-Signale.

Dieses Lehrbuch dient als praktische Einführung in die Bereiche DSP, SDR und drahtlose Kommunikation. Es richtet sich an jemanden, der:

#. Daran interessiert ist, SDRs für coole Dinge zu *verwenden*
#. Gut mit Python umgehen kann
#. Relativ neu in den Bereichen DSP, drahtlose Kommunikation und SDR ist
#. Visuell lernt und Animationen gegenüber Gleichungen bevorzugt
#. Gleichungen am besten versteht, *nachdem* die Konzepte gelernt wurden
#. Auf der Suche nach prägnanten Erklärungen ist, nicht nach einem 1.000-seitigen Lehrbuch

Ein Beispiel wäre ein Informatikstudent, der nach dem Abschluss an einem Job im Bereich drahtlose Kommunikation interessiert ist, obwohl es von jedem genutzt werden kann, der SDR lernen möchte und Programmiererfahrung hat. Als solches deckt es die notwendige Theorie zum Verständnis von DSP-Techniken ab, ohne die intensive Mathematik, die normalerweise in DSP-Kursen enthalten ist. Anstatt uns in Gleichungen zu vergraben, werden viele Bilder und Animationen verwendet, um die Konzepte zu vermitteln, wie die Animation der komplexen Ebene der Fourier-Reihe unten. Ich glaube, dass Gleichungen am besten *nach* dem Erlernen der Konzepte durch Visualisierungen und praktische Übungen verstanden werden. Die intensive Nutzung von Animationen ist der Grund, warum PySDR nie eine gedruckte Version haben wird, die bei Amazon verkauft wird.

.. image:: ../_images_de/fft_logo_wide.gif
   :scale: 70 %
   :align: center
   :alt: Das PySDR-Logo, erstellt mit einer Fourier-Transformation

Dieses Lehrbuch soll Konzepte schnell und reibungslos einführen und den Lesern ermöglichen, DSP durchzuführen und SDRs intelligent zu nutzen. Es ist nicht als Referenzlehrbuch für alle DSP/SDR-Themen gedacht; es gibt bereits viele großartige Lehrbücher, wie das `SDR-Lehrbuch von Analog Devices
<https://www.analog.com/en/education/education-library/software-defined-radio-for-engineers.html>`_ und `dspguide.com <http://www.dspguide.com/>`_. Du kannst immer Google verwenden, um trigonometrische Identitäten oder die Shannon-Grenze nachzuschlagen. Betrachte dieses Lehrbuch als Einstieg in die Welt von DSP und SDR: Es ist leichter und weniger zeitaufwendig und kostenintensiv im Vergleich zu traditionelleren Kursen und Lehrbüchern.

Um grundlegende DSP-Theorie abzudecken, wird ein gesamtes Semester „Signale und Systeme", ein typischer Kurs in der Elektrotechnik, in wenige Kapitel komprimiert. Sobald die DSP-Grundlagen abgedeckt sind, steigen wir in SDRs ein, obwohl DSP- und drahtlose Kommunikationskonzepte im gesamten Lehrbuch weiterhin auftauchen.

Codebeispiele werden in Python bereitgestellt. Sie verwenden NumPy, Pythons Standardbibliothek für Arrays und höhere Mathematik. Die Beispiele basieren auch auf Matplotlib, einer Python-Plotbibliothek, die eine einfache Möglichkeit bietet, Signale, Arrays und komplexe Zahlen zu visualisieren. Beachte, dass Python zwar im Allgemeinen „langsamer" als C++ ist, die meisten mathematischen Funktionen in Python/NumPy jedoch in C/C++ implementiert und stark optimiert sind. Ebenso ist die von uns verwendete SDR-API lediglich eine Reihe von Python-Bindungen für C/C++-Funktionen/-Klassen. Diejenigen, die wenig Python-Erfahrung haben, aber eine solide Grundlage in MATLAB, Ruby oder Perl haben, werden wahrscheinlich nach dem Kennenlernen der Python-Syntax gut zurechtkommen.


***************
Mitwirken
***************

Wenn du von PySDR profitiert hast, teile es bitte mit Kollegen, Studenten und anderen lebenslangen Lernenden, die an dem Material interessiert sein könnten. Du kannst auch über das `PySDR Patreon <https://www.patreon.com/PySDR>`_ spenden, um dich zu bedanken und deinen Namen links auf jeder Seite unter der Kapitelliste zu erhalten. Es gibt auch die Möglichkeit, `eine einmalige Spende zu machen <https://www.paypal.com/donate/?hosted_button_id=FH3LQCJRUVPWL>`_.

Wenn du einen Teil dieses Lehrbuchs durcharbeitest und mir eine E-Mail an marc@pysdr.org mit Fragen/Kommentaren/Vorschlägen schickst, dann herzlichen Glückwunsch, du hast zu diesem Lehrbuch beigetragen! Du kannst das Quellmaterial auch direkt auf der `GitHub-Seite des Lehrbuchs <https://github.com/777arc/PySDR/tree/master/content>`_ bearbeiten (deine Änderung startet einen neuen Pull-Request). Zögere nicht, ein Issue oder sogar einen Pull Request (PR) mit Korrekturen oder Verbesserungen einzureichen. Diejenigen, die wertvolles Feedback/Korrekturen einreichen, werden dauerhaft zum Abschnitt „Danksagungen" unten hinzugefügt. Nicht gut mit Git, aber hast Änderungsvorschläge? Schreib mir gerne eine E-Mail an marc@pysdr.org.

*****************
Danksagungen
*****************

Vielen Dank an alle, die einen Teil dieses Lehrbuchs gelesen und Feedback gegeben haben, und insbesondere an:

- `Barry Duggan <http://github.com/duggabe>`_
- Matthew Hannon
- James Hayek
- Deidre Stuffer
- Tarik Benaddi für die `Übersetzung von PySDR ins Französische <https://pysdr.org/fr/index-fr.html>`_
- `Daniel Versluis <https://versd.bitbucket.io/content/about.html>`_ für die `Übersetzung von PySDR ins Niederländische <https://pysdr.org/nl/index-nl.html>`_
- `mrbloom <https://github.com/mrbloom>`_ für die `Übersetzung von PySDR ins Ukrainische <https://pysdr.org/ukraine/index-ukraine.html>`_
- `Yimin Zhao <https://github.com/doctormin>`_ für die `Übersetzung von PySDR ins vereinfachte Chinesisch <https://pysdr.org/zh/index-zh.html>`_
- `Eduardo Chancay <https://github.com/edulchan>`_ für die `Übersetzung von PySDR ins Spanische <https://pysdr.org/es/index-es.html>`_
- `Dipl. Ing. (FH) Viet Dang <https://www.linkedin.com/in/viet-dang-a09bb01b6/>`_ für die `Übersetzung von PySDR ins Deutsche <https://pysdr.org/es/index-de.html>`_
- John Marcovici
- `Vishwaksen Reddy Dhareddy <https://www.linkedin.com/in/vishwaksen-/>`_ für den Beitrag zum Abschnitt „Echtzeit-Paketerkennung" im Erkennungskapitel

Sowie allen `PySDR Patreon <https://www.patreon.com/PySDR>`_-Unterstützern!
