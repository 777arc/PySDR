# PySDR Textbook Source Material

This repo contains the source content used to generate the textbook [PySDR: A Guide to SDR and DSP using Python](https://pysdr.org) hosted at https://pysdr.org.

Feel free to submit an issue, or even a Pull Request (PR) with fixes or improvements.  Those who submit valuable feedback/fixes be permanently added to the acknowledgments section.  Not good at Git but have changes to suggest?  Feel free to email Marc at marc@pysdr.org.

<p align="center">
  <img width="200" src="https://raw.githubusercontent.com/777arc/PySDR/master/_images/fft_logo_wide.gif" />
</p>

## Building

Note that the website is now automatically built and deployed with each push/merge into master branch, using the GitHub action [build-and-deploy.yml](https://github.com/777arc/PySDR/blob/master/.github/workflows/build-and-deploy.yml) and the GitHub pages system for hosting the actual textbook.

For testing changes to the textbook locally, you can build using the following steps:

### Ubuntu/Debian

Look at `.github/workflows/build-and-deploy.yml` and run the apt/pip installs, then:

```bash
make html
make html-fr
make html-nl
make html-ukraine
make html-zh
make html-ja
```

In _build there should be an index.html that represents the main page of the English site

Note: on one machine I had to add `~/.local/bin` to PATH

### Windows

Install pre-requisite software with:

1. From the Microsoft Store install Python 3.10 (3.8-3.12 is fine too if you already have it installed).
1. In a PowerShell terminal (click start menu then type powershell, or open a terminal in VSCode) run `pip install sphinx sphinxcontrib-tikz patreon setuptools`
1. `cd` to the directory you cloned PySDR

Build the English version only using:

```
python -m sphinx.cmd.build -b html . _build
```

The first time running this it might take a while because it has to download LaTeX packages.

Test the javascript part with the following to avoid CORS errors:
```
cd _build
python -m http.server
```

## Creating a PDF Export

Not fully working yet due to animated gifs, they all need to be removed for this to not error out:

```
sudo apt-get install -y latexmk
sphinx-build -b latex . _build/latex
make latexpdf
```

## Misc

Ideas for future chapters:

* Equalization, would be the last step needed to finish the end-to-end communications link
* OFDM, simulating OFDM and CP, show via Python how it turns freq selective fading into flat fading
* How to create real-time SDR apps with GUIs in Python using pyqt and pyqtgraph, or even just matplotlib with updating
* Python code that lets the Pluto (or RTL-SDR) act as an FM receiver, like with sound output
* End-to-end example that shows how to detect start of packet and other concepts not covered in RDS chapter
* Intro to radar

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 Unported License</a>.
