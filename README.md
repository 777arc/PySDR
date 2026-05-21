# PySDR

<p align="center">
  <img width="200" src="https://raw.githubusercontent.com/777arc/PySDR/master/_images/fft_logo_wide.gif" />
</p>

[PySDR: Ein Leitfaden zu SDR und DSP mit Python](https://pysdr.org) ist ein Leitfaden zur softwaredefinierten Funktechnik (SDR) und HF-Signalverarbeitung mit Python-Codebeispielen, verfügbar unter https://pysdr.org. Es handelt sich um ein kostenloses Online-Lehrbuch, das eine sanfte Einführung in drahtlose Kommunikation und SDR bietet – mit zahlreichen Diagrammen, Animationen und Codebeispielen. Von FFTs über Filter bis hin zu digitaler Modulation sowie dem Empfangen und Senden über SDRs in Python – PySDR hat alles abgedeckt! Dieses Repository enthält speziell den Quellinhalt zur Generierung des Lehrbuchs, einschließlich des Haupttexts und der Python-Skripte zur Erzeugung der Abbildungen. Bei Fragen, Anmerkungen oder Vorschlägen können Sie oben auf dieser Seite ein Issue einreichen. Wenn Sie eine Änderung am Lehrbuch vorschlagen möchten (z. B. eine Korrektur oder Verbesserung), können Sie einen Pull Request verwenden. Wer wertvolles Feedback oder Korrekturen einreicht, wird dauerhaft im Danksagungsabschnitt aufgeführt. Nicht vertraut mit Git, aber Änderungen vorschlagen? Schreiben Sie Marc gerne eine E-Mail an marc@pysdr.org.

Sie können PySDR auch über die [PySDR Patreon-Seite](https://www.patreon.com/c/PySDR) oder eine [Einmalspende](https://www.paypal.com/donate/?hosted_button_id=FH3LQCJRUVPWL) unterstützen.

## Erstellen

Die Website wird nun bei jedem Push/Merge in den Master-Branch automatisch erstellt und bereitgestellt, mithilfe der GitHub-Aktion [build-and-deploy.yml](https://github.com/777arc/PySDR/blob/master/.github/workflows/build-and-deploy.yml) und dem GitHub Pages-System zum Hosten des eigentlichen Lehrbuchs.

Um Änderungen am Lehrbuch lokal zu testen, kann mit folgenden Schritten gebaut werden:

### Ubuntu/Debian

Schauen Sie sich `.github/workflows/build-and-deploy.yml` an und führen Sie die apt/pip-Installationen aus, dann:

```bash
make html
make html-fr
make html-nl
make html-ukraine
make html-zh
make html-ja
```

In `_build` sollte eine `index.html` vorhanden sein, die die Hauptseite der englischen Website darstellt.

Hinweis: Auf einem Rechner musste ich `~/.local/bin` zum PATH hinzufügen.

### Windows

Installieren Sie die erforderliche Software wie folgt:

1. Installieren Sie Python 3.10 aus dem Microsoft Store (3.8–3.12 ist ebenfalls in Ordnung, falls bereits installiert).
1. Führen Sie in einem PowerShell-Terminal (Startmenü öffnen, dann „powershell" eingeben, oder ein Terminal in VSCode öffnen) den Befehl `pip install sphinx sphinxcontrib-tikz patreon setuptools` aus.
1. Wechseln Sie mit `cd` in das Verzeichnis, in das Sie PySDR geklont haben.

Erstellen Sie nur die englische Version mit:

```
python -m sphinx.cmd.build -b html . _build
```

Beim ersten Ausführen kann dies etwas länger dauern, da LaTeX-Pakete heruntergeladen werden müssen.

Testen Sie den JavaScript-Teil mit folgendem Befehl, um CORS-Fehler zu vermeiden:

```
cd _build
python -m http.server
```

## PDF-Export erstellen

Noch nicht vollständig funktionsfähig aufgrund animierter GIFs – diese müssen alle entfernt werden, damit kein Fehler auftritt:

```
sudo apt-get install -y latexmk
sphinx-build -b latex . _build/latex
make latexpdf
```

## Sonstiges

Ideen für zukünftige Kapitel:

* Entzerrung – wäre der letzte fehlende Schritt zum Abschluss der Ende-zu-Ende-Kommunikationsstrecke
* OFDM – Simulation von OFDM und CP, Darstellung per Python, wie frequenzselektives Fading in flaches Fading umgewandelt wird
* Erstellung von Echtzeit-SDR-Anwendungen mit GUIs in Python mithilfe von pyqt und pyqtgraph oder sogar mit aktualisierendem matplotlib
* Python-Code, der den Pluto (oder RTL-SDR) als UKW-Empfänger mit Audioausgabe betreibt
* End-to-End-Beispiel, das zeigt, wie der Paketanfang erkannt wird und andere Konzepte, die im RDS-Kapitel nicht behandelt werden
* Einführung in Radar

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons Lizenz" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />Dieses Werk ist lizenziert unter einer <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Namensnennung-NichtKommerziell-WeitergebeUnterGleichenBedingungen 4.0 International Lizenz</a>.
