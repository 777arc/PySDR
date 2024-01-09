# PySDR Textbook Source Material

This repo contains the source content used to generate the textbook [PySDR: A Guide to SDR and DSP using Python](https://pysdr.org) hosted at https://pysdr.org.

Feel free to submit an issue, or even a Pull Request (PR) with fixes or improvements.  Those who submit valuable feedback/fixes be permanently added to the acknowledgments section.  Not good at Git but have changes to suggest?  Feel free to email Marc at pysdr@vt.edu.

<p align="center">
  <img width="200" src="https://raw.githubusercontent.com/777arc/PySDR/master/_images/fft_logo_wide.gif" />
</p>

## Building

Note that the website is now automatically built and deployed with each push/merge into the main branch, using the github action .github/workflows/build-and-deploy.yml and the GitHub pages system for hosting the website.

### Ubuntu

```bash
sudo apt update
sudo apt install -y texlive-latex-extra pdf2svg
pip install -r requirements.txt
make html
make html-fr
make html-nl
make html-ukraine
```

In _build there should be an index.html that represents the main page of the site

Note: on one machine I had to add `~/.local/bin` to PATH

### Windows

(this wont include the French version):

```
sphinx-build -b html -D imgmath_latex="C:\Program Files\MiKTeX 2.9\miktex\bin\x64\latex.exe" . _build
```

## Creating a PDF Export

Not fully working yet due to gifs

```
sudo apt-get install -y latexmk
sphinx-build -b latex . _build/latex
make latexpdf
NEED TO REMOVE ALL GIFS FOR IT TO NOT ERROR OUT
```

## Misc

Ideas for future chapters:

* Equalization, would be the last step needed to finish the end-to-end comms link
* Cyclostationary analysis
* OFDM, simulating OFDM and CP, show via Python how it turns freq selective fading into flat fading
* How to create real-time SDR apps with GUIs in Python using pyqt and pyqtgraph, or even just matplotlib with updating
* Python code that lets the Pluto (or RTL-SDR) act as an FM receiver, like with sound output
* End to end example that shows how to detect start of packet and other concepts not covered in RDS chapter

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/3.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/3.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/3.0/">Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License</a>.
