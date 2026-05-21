"""
Post-process Matplotlib-generated SVGs to translate English axis labels to German.

Matplotlib SVGs store text as glyph paths, not <text> elements. This script:
1. Finds text groups identified by HTML comments (<!-- label -->)
2. Extracts the position from the transform attribute
3. Replaces the glyph group with an SVG <text> element in German

Supports three transform formats used by different matplotlib/SVG versions:
  - translate(x y) scale(a b)
  - translate(x y) rotate(angle) scale(a b)
  - matrix(a,b,c,d,e,f)
"""
import re
import os
import glob

# =========================================================================
# Translation dictionary (English → German)
# =========================================================================
TRANSLATIONS = {
    # --- Axis labels ---
    "Sample Index": "Sample-Index",
    "Signal Amplitude": "Signalamplitude",
    "FFT Magnitude": "FFT-Betrag",
    "FFT Phase [radians]": "FFT-Phase [Radianten]",
    "FFT Index": "FFT-Index",
    "Frequency [Hz]": "Frequenz [Hz]",
    "Frequency [MHz]": "Frequenz [MHz]",
    "Frequency [kHz]": "Frequenz [kHz]",
    "Frequency [GHz]": "Frequenz [GHz]",
    "Frequency [Normalized Hz]": "Frequenz [norm. Hz]",
    "Frequency Domain": "Frequenzbereich",
    "Frequency Response of Filter": "Frequenzgang des Filters",
    "Frequency offset (kHz)": "Frequenzversatz (kHz)",
    "Frequency": "Frequenz",
    "Time [s]": "Zeit [s]",
    "Time Domain": "Zeitbereich",
    "Time": "Zeit",
    "Amplitude / Value": "Amplitude / Wert",
    "Amplitude": "Amplitude",
    "Magnitude [dB]": "Betrag [dB]",
    "Magnitude": "Betrag",
    "Power [linear]": "Leistung [linear]",
    "Power Spectral Density [dB]": "Leistungsdichtespektrum [dB]",
    "Power, dB": "Leistung [dB]",
    "Power": "Leistung",
    "Phase Estimate [degrees]": "Phasenschätzung [Grad]",
    "Phase": "Phase",
    "Sample Value (I)": "Samplewert (I)",
    "Samples": "Samples",
    "Sample": "Sample",
    "Index": "Index",
    "Angle [deg]": "Winkel [Grad]",
    "Azimuth angle (deg)": "Azimutwinkel (Grad)",
    "Azimuth angle [degrees]": "Azimutwinkel [Grad]",
    "Elevation angle [degrees]": "Elevationswinkel [Grad]",
    "Theta (azimuth, degrees)": "Theta (Azimut, Grad)",
    "theta (20 deg)": "Theta (20 Grad)",
    "Phi (elevation, degrees)": "Phi (Elevation, Grad)",
    "Theta [Degrees]": "Theta [Grad]",
    "Eigenvalue [dB]": "Eigenwert [dB]",
    "Beam Pattern [dB]": "Strahlmuster [dB]",
    "DOA Metric": "DOA-Metrik",
    "Channel Magnitude [dB]": "Kanalbetrag [dB]",
    "Signal Magnitude [Linear]": "Signalbetrag [linear]",
    "Normalized Frequency": "Normierte Frequenz",
    "Cyclic Frequency [Normalized Hz]": "Zyklische Frequenz [norm. Hz]",
    "SCF Magnitude": "SCF-Betrag",
    "SCF Power": "SCF-Leistung",
    "CAF Power": "CAF-Leistung",
    "CAF (real part)": "CAF (Realteil)",
    "Tau": "Tau",
    "Alpha": "Alpha",
    "PSD [dB]": "LDS [dB]",
    "PSD Before FM Demod [dB]": "LDS vor FM-Demod [dB]",
    "PSD": "LDS",
    "Shannon Limit [bits/s/Hz]": "Shannon-Grenze [Bits/s/Hz]",
    "SNR [dB]": "SNR [dB]",
    "SNR (dB)": "SNR (dB)",
    "X Position [m]": "X-Position [m]",
    "Y Position [m]": "Y-Position [m]",
    "Z Position [m]": "Z-Position [m]",
    "Correlation Magnitude [Linear]": "Korrelationsbetrag [linear]",
    "Normalized Correlation Peak Magnitude": "Normierter Korrelations-Spitzenwert",
    "Normalized correlation peak (dB)": "Normierter Korrelationspeak [dB]",
    "Normalized Correlation": "Normierte Korrelation",
    "Correlation Power": "Korrelationsleistung",
    "Number of taps": "Anzahl der Koeffizienten",
    "Time per call (ms)": "Zeit pro Aufruf (ms)",
    "Code Phase (chips)": "Codephase (Chips)",
    "Offset (Fraction of a Chip)": "Versatz (Chip-Bruchteil)",
    "Freq Offset": "Frequenzversatz",
    "Freq": "Frequenz",
    "Pd": "Pd",
    "Pfa": "Pfa",
    "Period   (1/Frequency)": "Periode   (1/Frequenz)",
    "20 degrees": "20 Grad",

    # --- Plot titles ---
    "Beam Pattern and DOA Results, With Training": "Strahlmuster und DOA-Ergebnisse, mit Training",
    "Beam Pattern and DOA Results, Without Training": "Strahlmuster und DOA-Ergebnisse, ohne Training",
    "Conventional Pattern": "Konventionelles Muster",
    "MVDR Pattern": "MVDR-Muster",
    "Detection probability vs SNR for various frequency offsets": "Erkennungswahrscheinlichkeit vs. SNR",
    "Correlation degradation vs frequency offset": "Korrelationsdegradation vs. Frequenzversatz",
    "DSSS Correlation Peak vs. Fractional Chip Timing Offset": "DSSS-Korrelationspeak vs. Chip-Versatz",
    "Preamble Correlator Output with Adaptive CFAR Threshold": "Präambel-Korrelationsausgang mit CFAR",
    "ROC Curves": "ROC-Kurven",
    "Probability of detection": "Erkennungswahrscheinlichkeit",
    "Pd vs SNR (Pfa=0.01)": "Pd vs. SNR (Pfa=0,01)",
    "SV 11  —  Code-Phase Slice  (Doppler = +2500 Hz)": "SV 11  —  Codephase  (Doppler = +2500 Hz)",

    # --- Legend / annotation labels ---
    "Input Signal Length: 1000 samples": "Eingangssignallänge: 1000 Samples",
    "Input Signal Length: 100000 samples": "Eingangssignallänge: 100000 Samples",
    "Input": "Eingang",
    "Output": "Ausgang",
    "Error": "Fehler",
    "Signal 1": "Signal 1",
    "Signal 2": "Signal 2",
    "Signal 3": "Signal 3",
    "Real part of signal": "Realteil des Signals",
    "Wireless Signal": "Drahtloses Signal",
    "No Fading": "Kein Fading",
    "Rayleigh Fading": "Rayleigh-Fading",
    "Symbols": "Symbole",
    "Decoded": "Dekodiert",
    "Encoded": "Kodiert",
    "Our Data": "Unsere Daten",
    "Combined": "Kombiniert",
    "Pulses (before being combined)": "Impulse (vor Kombination)",
    "Starting With 0": "Beginnend mit 0",
    "Starting With 1": "Beginnend mit 1",
    "True Offset": "Wahrer Versatz",
    "Perfect Alignment": "Perfekte Ausrichtung",
    "After Costas Loop": "Nach Costas-Schleife",
    "Before Costas Loop": "Vor Costas-Schleife",
    "After Freq Offset": "Nach Frequenzversatz",
    "Before Freq Offset": "Vor Frequenzversatz",
    "After Interpolation": "Nach Interpolation",
    "Before Interpolation": "Vor Interpolation",
    "After Time Sync": "Nach Zeitsynchronisation",
    "Before Time Sync": "Vor Zeitsynchronisation",
    "Detections (Preamble Found)": "Erkennungen (Präambel gefunden)",
    "CFAR Adaptive Threshold": "CFAR-Adaptivschwelle",
    "0.15 Hz": "0,15 Hz",
    "Zoomed in below": "Vergrößert unten",
    "Regular SCF": "Reguläres SCF",
    "Spectral Coherence Function (COH)": "Spektrale Kohärenzfunktion (COH)",
    "Time-Domain Received Signal": "Empfangssignal im Zeitbereich",
    "Correlator Output $|r(t) * p^*(-t)|^2$": "Korrelatorausgang",
    "Rx Signal Power ($|r(t)|^2$)": "Empfangssignalleistung",
    "Offset=0.0 kHz": "Versatz=0,0 kHz",
    "Offset=2.0 kHz": "Versatz=2,0 kHz",
    "Offset=5.0 kHz": "Versatz=5,0 kHz",
}

# Window names are proper names, no translation needed:
# Hamming, Hanning, Blackman, Bartlett, Kaiser, Rectangular → Rechteck


def find_text_groups(content):
    """Find matplotlib text groups by scanning for comment+transform patterns.

    Returns list of dicts with keys: label, tx, ty, rotate, scale, start, end, g_open
    """
    results = []

    # A text group looks like:
    #   <g id="text_N" [optional attrs]>
    #     <!-- label text -->
    #     [optional <defs>...</defs>]
    #     <g [id=...] transform="...">
    #       ... glyph uses ...
    #     </g>
    #   </g>
    #
    # The transform can be:
    #   translate(x y) [rotate(angle)] scale(a b)
    #   matrix(a,b,c,d,e,f)  where (e,f) = translate and (a,d) = scale

    # Build combined regex that handles both transform types
    TRANSFORM_PAT = (
        r'(?:'
        r'translate\(([^)]+)\)(?:\s*rotate\(([^)]+)\))?\s*scale\(([^)]+)\)'   # variant 1
        r'|'
        r'matrix\(([0-9.eE+\-]+),([0-9.eE+\-]+),([0-9.eE+\-]+),([0-9.eE+\-]+),([0-9.eE+\-]+),([0-9.eE+\-]+)\)'  # variant 2
        r')'
    )

    pattern = re.compile(
        r'(<g\s[^>]*id="text_\d+"[^>]*>)\s*'        # outer g with text id
        r'<!--\s*([^-]+?)\s*-->\s*'                   # comment = label
        r'(?:<defs[^>]*>.*?</defs>\s*)?'              # optional inline defs
        r'<g\s[^>]*transform="' + TRANSFORM_PAT + r'"',
        re.DOTALL
    )

    for m in pattern.finditer(content):
        label = m.group(2).strip()
        g_open = m.group(1)

        # Groups 3-5: translate variant; groups 6-11: matrix variant
        if m.group(3) is not None:  # translate variant
            pos_parts = m.group(3).split()
            rotate_str = m.group(4)
            scale_str = m.group(5) or "0.1 -0.1"
            try:
                tx, ty = float(pos_parts[0]), float(pos_parts[1])
            except (ValueError, IndexError):
                continue
        else:  # matrix variant
            try:
                ma = float(m.group(6))   # scale_x
                # mb = m.group(7)  # 0 normally
                # mc = m.group(8)  # 0 normally
                md = float(m.group(9))   # scale_y (negative = y-flip)
                tx = float(m.group(10))
                ty = float(m.group(11))
            except (ValueError, TypeError):
                continue
            rotate_str = None
            scale_str = f"{abs(ma)} {abs(md)}"

        # Find end of the outer <g> group
        start = m.start()
        depth = 0
        group_end = -1
        for i in range(start, min(start + 50000, len(content))):
            if content[i:i+2] == '<g':
                depth += 1
            elif content[i:i+4] == '</g>':
                depth -= 1
                if depth == 0:
                    group_end = i + 4
                    break
        if group_end == -1:
            continue

        results.append({
            'label': label,
            'tx': tx,
            'ty': ty,
            'rotate': rotate_str,
            'scale': scale_str,
            'start': start,
            'end': group_end,
            'g_open': g_open,
        })

    return results


def estimate_font_size(scale_str):
    """Estimate font-size in SVG units from the scale string."""
    parts = scale_str.split()
    try:
        scale = abs(float(parts[0]))
        return max(6.0, scale * 100)
    except (ValueError, IndexError):
        return 10.0


def make_text_element(group_info, german_text, font_size):
    """Build an SVG <text> element to replace a glyph group."""
    tx = group_info['tx']
    ty = group_info['ty']
    rotate = group_info['rotate']

    id_match = re.search(r'id="([^"]+)"', group_info['g_open'])
    group_id = id_match.group(1) if id_match else 'text_x'

    safe_text = (german_text
                 .replace('&', '&amp;')
                 .replace('<', '&lt;')
                 .replace('>', '&gt;'))

    if rotate is not None:
        transform = f'translate({tx:.3f},{ty:.3f}) rotate({rotate})'
        return (
            f'<g id="{group_id}">'
            f'<text x="0" y="0" transform="{transform}" '
            f'font-family="DejaVu Sans,sans-serif" font-size="{font_size:.1f}" '
            f'text-anchor="middle">{safe_text}</text>'
            f'</g>'
        )
    else:
        return (
            f'<g id="{group_id}">'
            f'<text x="{tx:.3f}" y="{ty:.3f}" '
            f'font-family="DejaVu Sans,sans-serif" font-size="{font_size:.1f}" '
            f'text-anchor="middle">{safe_text}</text>'
            f'</g>'
        )


def process_matplotlib_svg(filepath, translations):
    """Translate text labels in a matplotlib SVG file. Returns True if changed."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Only process matplotlib SVGs (glyph-path based)
    if 'DejaVuSans' not in content and 'DejaVu Sans' not in content:
        return False

    groups = find_text_groups(content)
    if not groups:
        return False

    # Collect replacements (apply in reverse order to preserve offsets)
    replacements = []
    for g in groups:
        german = translations.get(g['label'])
        if german and german != g['label']:
            fs = estimate_font_size(g['scale'])
            replacements.append((g['start'], g['end'], make_text_element(g, german, fs)))

    if not replacements:
        return False

    # Apply in reverse order
    replacements.sort(key=lambda x: x[0], reverse=True)
    result = content
    for start, end, elem in replacements:
        result = result[:start] + elem + result[end:]

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(result)
    return True


# SVGs already handled by translate_svgs.py (real <text> elements)
REAL_TEXT_FILES = {
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
    'two-signals.svg', 'tx_rx_chain.svg', 'tx_rx_system.svg', 'tx_rx_system_params.svg',
}


if __name__ == '__main__':
    images_dir = os.path.dirname(os.path.abspath(__file__))
    changed = 0
    unchanged = 0

    # Files with ONLY real text (no matplotlib glyphs) — skip entirely
    PURE_REAL_TEXT = {
        'Costas_loop_model.svg', 'Spherical_Coordinates.svg', 'ad9361.svg',
        'adaptive_mcs.svg', 'adaptive_mcs2.svg', 'atmospheric_attenuation.svg',
        'bandpass_filter_taps.svg', 'beamforming_examples.svg', 'beamforming_taxonomy.svg',
        'bpsk.svg', 'bpsk2.svg', 'differential_coding2.svg', 'doa.svg', 'doa_trig.svg',
        'ethernet.svg', 'fft-block-diagram.svg', 'fft-io.svg', 'fft-python3.svg',
        'fft-python4.svg', 'freq-shift-diagram.svg', 'freq-shift.svg', 'fsk2.svg',
        'hamming.svg', 'hamming2.svg', 'masking.svg', 'monopulse.svg', 'multipath.svg',
        'multipath2.svg', 'negative-frequencies.svg', 'negative-frequencies2.svg',
        'negative-frequencies3.svg', 'receiver_arch_diagram.svg', 'spectrogram_diagram.svg',
        'splitting_rc_filter.svg', 'sync-diagram.svg', 'time-scaling.svg', 'trellis.svg',
        'two-signals.svg', 'tx_rx_chain.svg', 'tx_rx_system.svg', 'tx_rx_system_params.svg',
    }

    for fpath in sorted(glob.glob(os.path.join(images_dir, '*.svg'))):
        fname = os.path.basename(fpath)
        if fname in PURE_REAL_TEXT or fname.startswith('translate_'):
            continue
        if process_matplotlib_svg(fpath, TRANSLATIONS):
            print(f'  TRANSLATED: {fname}')
            changed += 1
        else:
            unchanged += 1

    print(f'\nDone: {changed} matplotlib SVGs translated, {unchanged} unchanged.')
