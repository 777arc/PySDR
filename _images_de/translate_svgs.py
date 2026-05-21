"""
Translate English text in SVG files to German.
Handles two types:
  1. SVGs with real <text>/<tspan> elements (46 files)
  2. Matplotlib SVGs with comments - these require script regeneration
"""
import re
import os
import glob

# Translation dictionary for SVG text labels
# Keys are exact English strings, values are German translations
# Order matters: longer/more specific strings first to avoid partial replacements
TRANSLATIONS = {
    # === AXIS LABELS ===
    "Frequency [GHz]": "Frequenz [GHz]",
    "Frequency [MHz]": "Frequenz [MHz]",
    "Frequency [kHz]": "Frequenz [kHz]",
    "Frequency [Hz]": "Frequenz [Hz]",
    "Frequency [Normalized Hz]": "Frequenz [normiert Hz]",
    "Frequency Domain": "Frequenzbereich",
    "Frequency Response of Filter": "Frequenzgang des Filters",
    "Attenuation [dB / km]": "Dämpfung [dB / km]",
    "Channel Magnitude [dB]": "Kanalbetrag [dB]",
    "Signal Amplitude": "Signalamplitude",
    "Signal Magnitude [Linear]": "Signalbetrag [linear]",
    "Power Spectral Density [dB]": "Leistungsdichtespektrum [dB]",
    "Magnitude [dB]": "Betrag [dB]",
    "Beam Pattern [dB]": "Strahlmuster [dB]",
    "FFT Magnitude": "FFT-Betrag",
    "FFT Phase [radians]": "FFT-Phase [Radianten]",
    "FFT Index": "FFT-Index",
    "Sample Index": "Sample-Index",
    "Time [s]": "Zeit [s]",
    "Time Domain": "Zeitbereich",
    "Eigenvalue [dB]": "Eigenwert [dB]",
    "Azimuth angle (deg)": "Azimutwinkel (Grad)",
    "Azimuth angle [degrees]": "Azimutwinkel [Grad]",
    "Elevation angle [degrees]": "Elevationswinkel [Grad]",
    "Theta [Degrees]": "Theta [Grad]",
    "Angle [deg]": "Winkel [Grad]",
    "B bandwidth [Hz]": "B Bandbreite [Hz]",
    "T seconds": "T Sekunden",
    "X Position [m]": "X Position [m]",
    "Y Position [m]": "Y Position [m]",
    "Z Position [m]": "Z Position [m]",
    "Shannon Limit [bits/s/Hz]": "Shannon-Grenze [Bits/s/Hz]",
    "Correlation Magnitude [Linear]": "Korrelationsbetrag [linear]",

    # === PLOT TITLES / DESCRIPTIONS ===
    "Beamforming Taxonomy": "Strahlformungs-Taxonomie",
    "Direction of Arrival (DOA)": "Einfallsrichtung (DOA)",
    "Direction of Arrival": "Einfallsrichtung",
    "Beam Pattern and DOA Results, With Training": "Strahlmuster und DOA-Ergebnisse, mit Training",
    "Beam Pattern and DOA Results, Without Training": "Strahlmuster und DOA-Ergebnisse, ohne Training",
    "Time Adaptive Processing (STAP)": "Raum-Zeit-adaptive Verarbeitung (STAP)",
    "Spectrogram": "Spektrogramm",
    "Time Domain": "Zeitbereich",
    "Modulation Scheme Used": "Verwendetes Modulationsschema",

    # === SYSTEM COMPONENTS ===
    "Direct Sampling (a.k.a. Direct RF)": "Direktabtastung (auch: Direct RF)",
    "Direct Conversion (a.k.a. Zero IF)": "Direktmischung (auch: Zero IF)",
    "Superheterodyne": "Überlagerungsempfänger",
    "FM radio in your old car": "UKW-Radio im alten Auto",
    "Expensive": "Teuer",
    "expensive ADC": "teurer ADC",
    "Digital Filtering and Equalization": "Digitale Filterung und Entzerrung",
    "RF Channel Bandwidth": "HF-Kanalbandbreite",
    "Rx Decimation": "Empfänger-Dezimation",
    "Tx Interpolation": "Sender-Interpolation",
    "Calibration and": "Kalibrierung und",
    "Correction": "Korrektur",
    "Enable State": "Zustandsmaschine",
    "Machine (ENSM)": "",
    "Automatic": "Automatische",
    "Phase\nSplitter": "Phasenteiler",
    "Input Mux": "Eingangs-Mux",
    "Output Mux": "Ausgangs-Mux",
    "Rx Channel 1": "Empfangskanal 1",
    "Rx Channel 2": "Empfangskanal 2",
    "Tx Channel 1": "Sendekanal 1",
    "Tx Channel 2": "Sendekanal 2",
    "Temperature": "Temperatur",
    "Baseband": "Basisband",
    "Splitter": "Teiler",
    "Dual": "Dual",

    # === SIGNAL FLOW ===
    "Transmit Antenna": "Sendeantenne",
    "TX Antenna": "Sendeantenne",
    "RX Antenna": "Empfangsantenne",
    "Transmitter": "Sender",
    "Receiver": "Empfänger",
    "Transmit\nPower": "Sendeleistung",
    "Transmit": "Senden",
    "Received": "Empfangen",
    "LOS Path": "Direktweg",
    "Multipath": "Mehrwegausbreitung",
    "Path Loss": "Pfadverlust",
    "(Compression)": "(Kompression)",
    "(Error correcting": "(Fehlerkorrektur",
    "codes)": "Codes)",
    "PSK, QAM)": "PSK, QAM)",
    "RF Circuit": "HF-Schaltung",
    "Up": "Aufwärts",
    "Converter,": "Wandler,",
    "Amplifiers)": "Verstärker)",
    "Down": "Abwärts",
    "Digital Converter)": "Digitalwandler)",
    "Converter)": "Wandler)",
    "Synchronization and": "Synchronisation und",
    "processing": "Verarbeitung",
    "often happens here": "oft hier durchgeführt",
    "Source\nData": "Quelldaten",
    "Source": "Quelle",
    "Data": "Daten",
    "(hopefully)": "(hoffentlich)",
    "Wireless": "Drahtlos",

    # === BEAMFORMING ===
    "Conventional Beamformer": "Konventioneller Strahlformer",
    "(aka Delay and Sum)": "(auch: Delay-and-Sum)",
    "Null Steering": "Nullsteuerung",
    "Switched Beam": "Geschaltete Strahlung",
    "Spatial Multiplexing": "Räumliches Multiplexing",
    "Pattern Synthesis": "Mustersynth.",
    "Subspace": "Unterraum",
    "Traditional": "Traditionell",
    "(Data Independent/Deterministic)": "(Datenunabhängig/Deterministisch)",
    "Adaptive": "Adaptiv",
    "Iterative": "(Snapshot/Update-basiert)",
    "(Snapshot/Update Based)": "(Snapshot/Update-basiert)",
    "Block\nbased": "Blockbasiert",
    "Tapering": "Fensterfunktionen",
    "(Optional Addon)": "(Optionale Erweiterung)",
    "Input includes": "Eingang beinhaltet",
    "(expected) angle": "(erwarteten) Winkel",
    "of SOI": "der Nutzquelle",
    "Needs pilots/exact": "Braucht Pilots/exaktes",
    "Most techniques with": "Die meisten Techniken unter",
    "under": "",
    "Beamforming": "Strahlformung",
    "can": "können",
    "be directly used to perform": "direkt zur DOA-Bestimmung verwendet werden",
    "Space": "Raum-",
    "Blind": "Blind",
    "Sidelobe": "Nebenkeulen-",
    "Canceller": "Unterdrücker",
    "Multiple Sidelobe": "Mehrfach-Nebenkeulen-",
    "Decomposition aka": "Zerlegung aka",
    "Max SNR": "Max SNR",
    "Max SINR": "Max SINR",
    "Dynamic Multiple": "Dynamischer Mehrfach-",
    "Sidelobe Canceller": "Nebenkeule-Unterdrücker",
    "Woodward Lawson": "Woodward-Lawson",
    "Technique": "Technik",
    "Max": "Max",
    "Likelihood": "Likelihood",
    "Beamform": "Strahl-",
    "ers": "former",
    "based": "basiert",

    # === PLOT ELEMENTS ===
    "Noise Floor": "Rauschpegel",
    "Signal(s)": "Signal(e)",
    "Signal we don't want": "Unerwünschtes Signal",
    "Signal in": "Eingangssignal",
    "Signal 1": "Signal 1",
    "Signal 2": "Signal 2",
    "Signal 3": "Signal 3",
    "FFT Shift": "FFT-Verschiebung",
    "FFT of Slice 1": "FFT von Abschnitt 1",
    "FFT of Slice 2": "FFT von Abschnitt 2",
    "FFT of Slice 3": "FFT von Abschnitt 3",
    "FFT of Slice 4": "FFT von Abschnitt 4",
    "FFT of Slice 5": "FFT von Abschnitt 5",
    "FFT of Slice 6": "FFT von Abschnitt 6",
    "Slice\n1": "Abschnitt 1",
    "Slice 2": "Abschnitt 2",
    "Slice 3": "Abschnitt 3",
    "Slice 4": "Abschnitt 4",
    "Slice 5": "Abschnitt 5",
    "Slice 6": "Abschnitt 6",
    "Slice": "Abschnitt",
    "(IQ Samples)": "(IQ-Samples)",
    "Stereo Audio (L-R)": "Stereo-Audio (L-R)",
    "Mono": "Mono",
    "Audio": "Audio",
    "Tone": "Ton",
    "Bit Position": "Bitposition",
    "Encoded Bits": "Kodierte Bits",
    "Encoded": "Kodiert",
    "Decoded": "Dekodiert",
    "Parity": "Parität",
    "Coverage": "Abdeckung",
    "boresight": "Hauptstrahlrichtung",
    "Target": "Ziel",
    "Beam 1": "Strahl 1",
    "Beam 2": "Strahl 2",
    '"Sum"': '"Summe"',
    "Beam": "Strahl",
    "Taps": "Koeffizienten",
    "Magnitude": "Betrag",
    "Sample Rate": "Abtastrate",
    "Sample:": "Sample:",
    "Sample\n": "Sample\n",
    "Input Vector": "Eingangsvektor",
    "Input 1": "Eingang 1",
    "Input 2": "Eingang 2",
    "Input": "Eingang",
    "Output": "Ausgang",
    "Delay": "Verzögerung",
    "Filter": "Filter",
    "Channel": "Kanal",
    "Modulation": "Modulation",
    "Demodulation": "Demodulation",
    "Pulse": "Impuls",
    "Matched": "Angepasst",
    "Coarse": "Grob",
    "Time Sync": "Zeitsynchronisation",
    "Fine Freq": "Feine Frequenz",
    "Frame": "Rahmen",
    "Gain": "Verstärkung",
    "Power": "Leistung",

    # === PARTIAL WORDS (for axis labels split across elements) ===
    "requency": "requenz",
    "ime": "eit",
    "frequency": "Frequenz",
    "time": "Zeit",
    "spectrum": "Spektrum",
    "signal": "Signal",
    "shifted": "verschoben",
    "sinusoid": "Sinusschwingung",
    "shift": "Verschiebung",
    "more\nbits per\nsecond": "mehr\nBits pro\nSekunde",
    "more\nspectrum\nrequired": "mehr\nSpektrum\nbenötigt",
    "bits per": "Bits pro",
    "second": "Sekunde",
    "more": "mehr",
    "required": "benötigt",
    "index:": "Index:",
    "modulated\noutput": "modulierter\nAusgang",
    "modulated": "moduliert",
    "output": "Ausgang",

    # === ADAPTIVE MCS ===
    "Throughpu": "Durchsat",
    "(L+R)": "(L+R)",

    # === SPHERICAL COORDINATES ===
    "(azimuth)": "(Azimut)",
    "(elevation)": "(Elevation)",

    # === TRELLIS ===
    "Level j=": "Ebene j=",

    # === REMAINING LOWERCASE / SPECIFIC CASES ===
    "Received": "Empfangen",
    "TX Antenna": "Sendeantenne",
    "RX Antenna": "Empfangsantenne",
    "Transmit\nPower": "Sendeleistung",
    "Automatic\nGain\nControl": "Automatische\nVerstärkungs-\nRegelung",
}

# Files where we should NOT translate "T" → "Z" standalone letter (it's part of Time axis)
# We handle spectrogram_diagram.svg specially
SPECTROGRAM_SPECIAL = {
    ">T<": ">Z<",  # Only the T that's part of "Time" axis
}


def translate_text_content(text, context=""):
    """Translate English text to German."""
    # Try exact match first (longest first)
    for en, de in sorted(TRANSLATIONS.items(), key=lambda x: -len(x[0])):
        if text == en:
            return de
    return text


def translate_svg_text_elements(content, filename=""):
    """Replace English text in SVG text nodes with German translations.

    Uses targeted replacement: for each (en, de) pair we replace all occurrences of
    '>en<' and '>en ' (text node delimiters), which only matches actual XML text nodes,
    not attribute values (which are enclosed in quotes) or path data.
    """
    result = content

    # Apply translations ordered by length (longest first to avoid partial matches)
    for en, de in sorted(TRANSLATIONS.items(), key=lambda x: -len(x[0])):
        if not en or not de or en == de:
            continue

        # Replace as exact text node: >en< or >en (followed by < or newline/space then <)
        # This covers: <text>en</text> and <text>en <tspan>
        result = result.replace(f'>{en}<', f'>{de}<')
        result = result.replace(f'>{en} <', f'>{de} <')
        result = result.replace(f'>{en}\n', f'>{de}\n')
        result = result.replace(f'>{en}\r\n', f'>{de}\r\n')

    # Special handling for spectrogram_diagram.svg: "T" axis label → "Z"
    if 'spectrogram_diagram' in filename:
        result = result.replace(
            'matrix(-1.83697e-16 -1 1 -1.83697e-16 113.526 276)">T</text>',
            'matrix(-1.83697e-16 -1 1 -1.83697e-16 113.526 276)">Z</text>'
        )

    return result


def process_svg_file(filepath):
    """Translate a single SVG file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content
    filename = os.path.basename(filepath)
    translated = translate_svg_text_elements(content, filename)

    if translated != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(translated)
        return True
    return False


# List of SVGs with real text elements (directly translatable)
REAL_TEXT_FILES = [
    'Costas_loop_model.svg', 'Spherical_Coordinates.svg', 'ad9361.svg',
    'adaptive_mcs.svg', 'adaptive_mcs2.svg', 'atmospheric_attenuation.svg',
    'bandpass_filter_taps.svg', 'beamforming_examples.svg', 'beamforming_taxonomy.svg',
    'bpsk.svg', 'bpsk2.svg', 'costas-loop-freq-tracking.svg', 'differential_coding2.svg',
    'doa.svg', 'doa_trig.svg', 'ethernet.svg', 'fft-block-diagram.svg', 'fft-io.svg',
    'fft-python3.svg', 'fft-python4.svg', 'fm_psd.svg', 'fm_psd_labeled.svg',
    'freq-shift-diagram.svg', 'freq-shift.svg', 'fsk2.svg', 'hamming.svg', 'hamming2.svg',
    'masking.svg', 'max_freq.svg', 'monopulse.svg', 'multipath.svg', 'multipath2.svg',
    'negative-frequencies.svg', 'negative-frequencies2.svg', 'negative-frequencies3.svg',
    'rayleigh.svg', 'receiver_arch_diagram.svg', 'spectrogram_diagram.svg',
    'splitting_rc_filter.svg', 'sync-diagram.svg', 'time-scaling.svg', 'trellis.svg',
    'two-signals.svg', 'tx_rx_chain.svg', 'tx_rx_system.svg', 'tx_rx_system_params.svg'
]


if __name__ == '__main__':
    images_dir = os.path.dirname(os.path.abspath(__file__))

    changed = 0
    unchanged = 0

    for fname in REAL_TEXT_FILES:
        fpath = os.path.join(images_dir, fname)
        if not os.path.exists(fpath):
            print(f'  MISSING: {fname}')
            continue

        if process_svg_file(fpath):
            print(f'  TRANSLATED: {fname}')
            changed += 1
        else:
            print(f'  unchanged: {fname}')
            unchanged += 1

    print(f'\nDone: {changed} files translated, {unchanged} unchanged.')
