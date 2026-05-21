.. _detection-chapter:

#####################################################
Detectie met Correlatie
#####################################################

.. raw:: html

 <span style="display: table; margin: 0 auto; font-size: 20px;">Mede-auteur: <a href="https://www.linkedin.com/in/samuel-brown-vt">Sam Brown</a></span>

In dit hoofdstuk leren we hoe we de aanwezigheid van signalen kunnen detecteren en hun timing terug kunnen vinden door ontvangen samples kruis te correleren met een voor ons bekend deel van het signaal, zoals de preamble van een pakket. Deze methode leidt van nature tot een eenvoudige vorm van classificatie met een rij correlators. We introduceren de basisconcepten van signaaldetectie, met focus op de beslissing of een specifiek signaal wel of niet aanwezig is in een ruisachtige omgeving. Daarbij behandelen we zowel de theoretische basis als praktische technieken om onder onzekerheid zo goed mogelijk te beslissen.

****************************************************
Basis van Signaaldetectie en Correlators
****************************************************

Signaaldetectie is de taak waarbij wordt besloten of een waargenomen energiepiek een betekenisvol signaal is of alleen achtergrondruis.

De Uitdaging - In systemen zoals radar of sonar is ruis overal aanwezig. Als de detector te gevoelig is, krijg je "valse alarmen". Is hij niet gevoelig genoeg, dan "mist" hij het echte doel.

De Oplossingen - De eerste en eenvoudigste optie is de Neyman-Pearson-detector, die een wiskundige "sweet spot" geeft: maximale kans op detectie bij een strikt begrensde kans op vals alarm. CFAR-detectors (CFAR: Constant False Alarm Rate) bouwen hierop voort door adaptief te reageren op veranderingen in het ruisniveau. Meer specifiek worden CFAR-detectors gebruikt wanneer de ruisstatistiek niet stationair is, dus wanneer ruisvloer en ruisverdeling veranderen door interferentie en veranderende kanaalomstandigheden. Het doel is om de detectiedrempel automatisch mee te laten bewegen met de achtergrondruis, zodat een gewenste vals-alarmkans behouden blijft. Dat vereist een continue schatting van de ruisvloer.

Zodra een systeem weet dat er iets aanwezig is, moet het precies bepalen waar de data start. Digitale pakketten in LTE, 5G en wifi beginnen met een "preamble": een bekend en herhaald patroon. Een preamble-correlator werkt als een "slot-en-sleutel"-mechanisme, waarbij de sleutel een symboolreeks is die de ontvanger kent en die uniek is voor het te herstellen signaal. Door een kopie van die preamble over het inkomende signaal te schuiven en op elke vertraging een inwendig product te nemen, meet de ontvanger de overeenkomst tussen sjabloon en ontvangen reeks. Als beide bijna perfect uitlijnen, ontstaat een scherpe piek die exact aangeeft waar de data begint. Geavanceerde varianten houden ook rekening met frequentie-offset door afstemverschillen of Doppler-verschuiving.

Wanneer een bekend signaal (de preamble) over een kanaal met alleen Additive White Gaussian Noise (AWGN) wordt verzonden, is de taak simpel: beslissen of het signaal aanwezig is. Dit is het eenvoudigste, maar ook meest fundamentele detectieprobleem.

De Kruiscorrelatiefunctie
###############################

Een correlator in de eenvoudigste vorm is gewoon een kruiscorrelatie tussen een ontvangen signaal en een sjabloon van wat je zoekt. Kruiscorrelatie is een inwendig product tussen twee vectoren terwijl één vector over de andere schuift. Als je convolutie kent: het is bijna hetzelfde, behalve dat je de tweede vector niet omkeert, dus net iets eenvoudiger. Voor complexe signalen, waar we hier mee werken, neem je ook de complex geconjugeerde van één ingang. In Python kan dat zo:

.. code-block:: python

    def correlate(a, v):
        n = len(a)
        m = len(v)
        result = []
        for i in range(n - m + 1):
            s = 0
            for j in range(m):
                s += a[i + j] * v[j].conjugate()
            result.append(s)
        return result

    # Voorbeeldgebruik:
    a = [1+2j, 2+1j, 3+0j, 4-1j, 5-2j]
    v = [0+1j, 1+0j, 0.5-0.5j]
    correlate(a, v)

Let op hoe :code:`a` schuift en :code:`v` complex geconjugeerd wordt, en hoe de loop met :code:`j` en :code:`s` in feite gewoon een inwendig vector-product is. Gelukkig hoeven we kruiscorrelatie niet zelf van nul te implementeren: in Python kunnen we NumPy's :code:`correlate` gebruiken (er is ook een SciPy-variant).

Python-voorbeeld van een Kruiscorrelatie
########################################################

Om een basisvoorbeeld van een correlator in Python te maken, bouwen we eerst een voorbeeldsignaal met een bekende preamble in ruis. We gebruiken een Zadoff-Chu-sequentie als bekende preamble vanwege de uitstekende autocorrelatie-eigenschappen en het veelvuldige gebruik in communicatiesystemen. We negeren hier de rest van de payload-data, al volgt in echte systemen na de preamble meestal onbekende data. Een Zadoff-Chu-sequentie genereren we zo:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    N = 839  # Length of Zadoff-Chu sequence
    u = 25  # Root of ZC sequence
    t = np.arange(N)
    zadoff_chu = np.exp(-1j * np.pi * u * t * (t + 1) / N)

De resulterende sequentie *is* een signaal: de IQ-samples van :code:`zadoff_chu` vormen een complex basisbandsignaal zoals we vaak in dit boek zien, alleen encodeert het hier geen bits. We kunnen een realistischer scenario nabootsen door dit Zadoff-Chu-signaal op een willekeurige offset in een langere AWGN-stroom te plaatsen:

.. code-block:: python

    signal_length = 10 * N # overall simulated signal length
    offset = np.random.randint(N, signal_length - N)
    print(f"True offset: {offset}")
    snr_db = -15
    noise_power = 1 / (2 * (10**(snr_db / 10)))
    signal = np.sqrt(noise_power/2) * (np.random.randn(signal_length) + 1j * np.random.randn(signal_length))
    signal[offset:offset+N] += zadoff_chu # place our ZC signal at the random offset

Let op dat we hier een *zeer* lage SNR gebruiken. Die is zo laag dat je de Zadoff-Chu-sequentie in het tijdsdomein helemaal niet terugziet. De sequentie is 839 samples lang op ongeveer 8000 gesimuleerde samples, en zit zo diep in de ruis dat je zelfs geen kleine toename in signaalamplitude ziet.

.. image:: ../_images/detection_basic_1.svg
   :align: center 
   :target: ../_images/detection_basic_1.svg
   :alt: Time Domain Signal with Zadoff-Chu Sequence

Nu kunnen we de correlator implementeren door een kruiscorrelatie uit te voeren tussen het ontvangen signaal en onze bekende Zadoff-Chu-sequentie met :code:`np.correlate()`. Dit veronderstelt dat de ontvanger de exacte preamble kent. :code:`zadoff_chu` werd eerst gebruikt om het scenario te simuleren, maar fungeert nu ook als sjabloon dat de ontvanger in de correlator gebruikt. In Python kan dit in één regel:

.. code-block:: python

 correlation = np.correlate(signal, zadoff_chu, mode='valid')

De :code:`valid`-modus lichten we zo toe. We normaliseren de uitgang ook met de sequentielengte en nemen de magnitude in het kwadraat om vermogen te krijgen. Je kunt ook alleen de magnitude nemen; dat werkt meestal ook. Het belangrijkste blijft de :code:`np.correlate()`-operatie.

.. code-block:: python

 correlation = np.abs(correlation / N)**2 # normalize by N, and take magnitude squared

Hieronder plotten we de magnitude in het kwadraat en markeren we de echte startpositie van de sequentie om te zien of de correlator die vindt:

.. image:: ../_images/detection_basic_2.svg
   :align: center 
   :target: ../_images/detection_basic_2.svg
   :alt: Correlator Output

Ondanks de zeer lage SNR zien we een duidelijke piek in de correlator-uitgang precies waar de Zadoff-Chu-sequentie is geplaatst. Dat is de *start* van de sequentie: de 839 samples vanaf die piek bevatten het patroon. Dit laat de kracht van correlatiegebaseerde detectie zien, zeker in combinatie met een lange preamble. We hebben nog geen expliciete drempel ingesteld om te beslissen of de piek een echt signaal of ruis is; we inspecteren nu visueel. Voor automatische detectie is een drempel nodig. De rest van dit hoofdstuk gaat grotendeels over hoe je die drempel goed kiest, vooral wanneer ruisvloer en interferentie continu veranderen.

Modi: Valid, Same en Full
#######################################

Je hebt misschien gezien dat :code:`np.correlate()` en :code:`np.convolve()` drie modi hebben: :code:`valid`, :code:`same` en :code:`full`. Die bepalen de lengte van de output-array ten opzichte van de inputs. In ons geval gebruikten we :code:`valid`, wat betekent dat alleen punten worden teruggegeven waar beide arrays volledig overlappen. De outputlengte wordt dan :code:`len(signal) - len(zadoff_chu) + 1`. Met :code:`same` krijg je een output met dezelfde lengte als het (langste) ingangssignaal. Met :code:`full` krijg je de volledige lineaire discrete convolutie, met een iets langere output van lengte :code:`max(M, N) - min(M, N) + 1`, waarbij :code:`M` en :code:`N` de lengtes van beide arrays zijn. In veel RF-signaalbewerking gebruiken we convolutie om een FIR-filter toe te passen, en dan is :code:`same` handig omdat input en output even lang blijven. Voor correlatiegebaseerde detectie willen we meestal :code:`valid`, omdat vooral de posities interessant zijn waar de preamble volledig overlapt met het ontvangen signaal.

De Neyman-Pearson-detector
############################

De gouden standaard voor het kiezen van een goede drempel voor correlatoruitgang is de Neyman-Pearson-detector. Deze theorie helpt een optimale beslissing nemen onder een specifieke randvoorwaarde: maximaliseer de detectiekans, :math:`P_{D}`, bij een vaste en acceptabele vals-alarmkans, :math:`P_{FA}`. Simpel gezegd: jij kiest hoeveel valse detecties je maximaal accepteert (bijvoorbeeld één per uur), en Neyman-Pearson geeft de beste drempel om zoveel mogelijk echte signalen te vinden. Voor detectie van een bekende preamble in AWGN is de aanpak eenvoudig: bereken een correlatiewaarde tussen het ontvangen signaal en een bekend preamble-patroon. Overschrijdt die waarde de drempel :math:`\tau`, dan verklaar je het signaal aanwezig; anders ga je ervan uit dat er alleen ruis is.

De prestatie van deze detector, gemeten met :math:`P_{D}` en :math:`P_{FA}`, hangt af van de drempel :math:`\tau`, de SNR en de preamblelengte :math:`L`. De vals-alarmkans is een functie van de drempel en ruisvariantie :math:`\sigma_n^2`:

:math:`P_{FA} = Q\left(\frac{\tau}{\sigma_n}\right)`

De detectiekans is een functie van drempel, ruisvariantie en preamble-energie (:math:`E_s = L \cdot S`, met :math:`S` als gemiddeld symboolvermogen):

:math:`P_{D} = Q\left(\frac{\tau - \sqrt{E_s}}{\sigma_n}\right) = Q\left(\frac{\tau - \sqrt{L \cdot S}}{\sigma_n}\right)`

Hier is :math:`Q(x)` de Q-functie (staartkans van de standaardnormale verdeling), oftewel de kans dat een standaardnormale variabele groter is dan :math:`x`.

Prestatie-analyse: ROC-curves en Pd-vs-SNR-curves
#################################################################

Om te kwantificeren hoe goed een correlatie-detector presteert in ruis, gebruiken engineers twee hoofdvisualisaties: de Receiver Operating Characteristic (ROC)-curve en de Probability of Detection (:math:`P_{d}`)-tegen-SNR-curve.

De ROC-curve zet :math:`P_{d}` uit tegen :math:`P_{fa}` bij vaste SNR. Door de detectiedrempel op de correlatoruitgang te variëren kies je een punt op deze curve; het blijft een afweging. Een lagere drempel verhoogt :math:`P_{d}` (meer signalen gevonden), maar ook :math:`P_{fa}` (meer ruis-triggers). Hoe sterker de curve naar linksboven buigt, hoe beter de detector. Een perfecte detector zit linksboven (100% :math:`P_{d}`, 0% :math:`P_{fa}`), terwijl een diagonaal overeenkomt met gokken.

.. image:: ../_images/detection_pd_vs_snr.svg
   :align: center 
   :target: ../_images/detection_pd_vs_snr.svg
   :alt: Pd vs SNR Curve and ROC curve

Uit de vergelijkingen (en intuïtie) volgt dat preamblelengte :math:`L` een cruciale ontwerpparameter is, omdat die direct de processing gain en daarmee de detectieprestatie bepaalt. Bij vaste drempel en SNR groeit :math:`P_{D}` met :math:`L`. Een langere preamble verzamelt meer signaalenergie, waardoor scheiding tussen signaal en achtergrondruis eenvoudiger wordt. Deze prestatieverbetering heet "processing gain" en wordt vaak in dB uitgedrukt als :math:`10\log_{10}(L)`. Dit is essentieel voor zwakke signalen die anders gemist worden. Door energie over meer samples te integreren kun je signalen detecteren die onder de ruisvloer liggen.

****************************************************
CFAR-detectors: Robuust in Veranderende Omgevingen
****************************************************

Hoewel de Neyman-Pearson-detector optimaal is bij een vaste ruisvloer, zijn praktijkomstandigheden zelden zo stabiel. In een dynamische omgeving, zoals radar door regen of een draadloze ontvanger in een drukke stad, schommelen achtergrondruis en interferentie voortdurend. Hier wordt een Constant False Alarm Rate (CFAR)-detector essentieel.

CFAR-detectors zijn de werkpaarden van systemen waar een onvoorspelbare achtergrond een vaste drempel onbruikbaar maakt:

- Radar en sonar detecteren doelen (vliegtuigen, onderzeeers) tegen "clutter": reflecties van golven, regen of land die veranderen terwijl de sensor beweegt.
- Draadloze communicatie, zoals cognitieve radio en LTE/5G-systemen, gebruikt CFAR om beschikbaar spectrum te vinden of inkomende pakketten te detecteren bij grillige interferentie van andere apparaten.
- Medische beeldvorming gebruikt CFAR in automatische analyse van echo- of MRI-data om echte weefselstructuren te onderscheiden van variërende elektronische ruis.

De "C" in CFAR staat voor Constant, omdat het doel is om de vals-alarmkans (:math:`P_{FA}`) op een stabiel, voorspelbaar niveau te houden.

Om een drempelwaarde te kiezen, moet je een statistisch ruismodel aannemen (de ruisverdeling). In eenvoudige AWGN is dat een Gauss-verdeling. In radarclutter kan het bijvoorbeeld een Rayleigh- of Weibull-verdeling zijn. Als je model niet klopt, gaat :math:`P_{FA}` "driften", waardoor het systeem óf blind wordt óf overspoeld raakt door valse triggers.

In plaats van een vaste waarde schat een CFAR-detector het ruisvermogen in de lokale "omgeving" van het signaal en vermenigvuldigt die schatting met een schaalfactor (:math:`T`) afgeleid van de gewenste :math:`P_{FA}`. Daardoor stijgt de drempel automatisch mee als de ruisvloer stijgt.

Per-lag versus Systeemniveau Vals-alarmkans
####################################################

Dit is een cruciaal onderscheid dat beginners vaak missen. Bij preamble-zoekacties voer je meestal een schuivende correlatie uit, waarbij je de drempel op duizenden tijdsverschuivingen ("lags") per seconde controleert.

Per-lag :math:`P_{FA}`: de kans dat één specifieke correlatietoets een vals alarm oplevert. Stel je :math:`P_{FA}` op 0,001, dan heeft elke losse lag 1 op 1000 kans op een "spooksignaal".

Systeemniveau (globaal) :math:`P_{FA}`: de kans dat het systeem minstens één vals alarm geeft in een volledig zoekvenster (bijv. over 2048 lags).

Wiskundig geldt: als je per-lag :math:`P_{FA}` gelijk is aan :math:`p`, dan is de kans op minstens één vals alarm over :math:`N` lags ongeveer :math:`1-(1-p)^{N}`.

Gevolg: bij 1000 lags en per-lag :math:`P_{FA}` van 0,001 rapporteert het systeem in bijna 63% van de zoekacties minstens één vals alarm. Om de systeemniveau-kans laag te houden moet per-lag :math:`P_{FA}` dus extreem klein zijn.

Python-voorbeeld
#################

Om zelf met een CFAR-detector te experimenteren, simuleren we eerst een scenario met herhaalde QPSK-pakketten met bekende preamble over een kanaal met tijdsvariërende ruisvloer. Daarna implementeren we een eenvoudige Cell-Averaging CFAR (CA-CFAR)-detector om preambles in het ontvangen signaal te vinden. De volgende Python-code genereert het ontvangen signaal:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import correlate

    def generate_qpsk_packets(num_packets, sps, preamble):
        """Generates repeating QPSK packets with gaps and varying noise."""
        qpsk_map = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        data_len = 200
        gap_len = 100
        full_signal = []
        
        # Pre-calculate preamble upsampled for correlation
        upsampled_preamble = np.repeat(preamble, sps)
        
        for _ in range(num_packets):
            data = qpsk_map[np.random.randint(0, 4, data_len)]
            packet = np.concatenate([preamble, data])
            full_signal.extend(np.repeat(packet, sps))
            full_signal.extend(np.zeros(gap_len * sps))
        
        return np.array(full_signal), upsampled_preamble

    # Setup Parameters
    sps = 4
    preamble_syms = np.array([1+1j, 1+1j, -1-1j, -1-1j, 1-1j, -1+1j]) / np.sqrt(2)
    tx_signal, ref_preamble = generate_qpsk_packets(5, sps, preamble_syms)

    # Channel: Time-Varying Noise Floor
    t = np.arange(len(tx_signal))
    noise_env = 0.05 + 0.3 * np.sin(2 * np.pi * 0.0003 * t)**2
    noise = (np.random.randn(len(tx_signal)) + 1j*np.random.randn(len(tx_signal))) * noise_env
    rx_signal = tx_signal + noise

De eerste stap is één correlatie van het ontvangen signaal met de bekende preamble. In de praktijk gebeurt dit vaak in batches, maar hier doen we het in één batch:

.. code-block:: python

    # Preamble-correlatie, een correlatiepiek ontstaat wanneer referentie en ontvangen segment matchen
    corr_out = correlate(rx_signal, ref_preamble, mode='same')
    corr_power = np.abs(corr_out)**2

TODO: kijk naar de ruwe output van alleen deze stap

Nu implementeren we de CFAR-detector, passen die toe op de correlatoruitgang en visualiseren de resultaten:

.. code-block:: python

    # CFAR Detection on Correlator Output
    def ca_cfar_adaptive(data, num_train, num_guard, pfa):
        num_cells = len(data)
        thresholds = np.zeros(num_cells)
        alpha = num_train * (pfa**(-1/num_train) - 1)  # Scaling factor
        half_window = (num_train + num_guard) // 2
        guard_half = num_guard // 2
        for i in range(half_window, num_cells - half_window):
            # Extract training cells (excluding guard cells and CUT)
            lagging_win = data[i - half_window : i - guard_half]
            leading_win = data[i + guard_half + 1 : i + half_window + 1]
            noise_floor_est = np.mean(np.concatenate([lagging_win, leading_win]))
            thresholds[i] = alpha * noise_floor_est
        return thresholds

    # Detect on correlator power
    cfar_thresholds = ca_cfar_adaptive(corr_power, num_train=60, num_guard=20, pfa=1e-5)
    detections = np.where(corr_power > cfar_thresholds)[0]
    # Filter detections to only include those where threshold is non-zero (avoid edges)
    detections = detections[cfar_thresholds[detections] > 0]

    # Subplot 1: Received Signal and Raw Power
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.plot(np.abs(rx_signal)**2, color='gray', alpha=0.4, label='Rx Signal Power ($|r(t)|^2$)')
    plt.title("Time-Domain Received Signal")
    plt.ylabel("Power")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Correlator Output vs Adaptive Threshold
    plt.subplot(2, 1, 2)
    plt.plot(corr_power, label='Correlator Output $|r(t) * p^*(-t)|^2$', color='blue')
    plt.plot(cfar_thresholds, label='CFAR Adaptive Threshold', color='red', linestyle='--', linewidth=1.5)
    if len(detections) > 0: # Overlay detections
        plt.scatter(detections, corr_power[detections], color='lime', edgecolors='black', label='Detections (Preamble Found)', zorder=5)
    plt.title("Preamble Correlator Output with Adaptive CFAR Threshold")
    plt.xlabel("Sample Index")
    plt.ylabel("Correlation Power")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

.. image:: ../_images/detection_cfar.svg
   :align: center 
   :target: ../_images/detection_cfar.svg
   :alt: CFAR Detector Output Example



Frequentie-offset-robuuste Preamble-correlators
####################################################

Het detecteren van een preamble wordt een meerdimensionaal zoekprobleem wanneer de middenfrequentie onbekend is. In een perfect gesynchroniseerd systeem werkt een coherente correlator als matched filter en maximaliseert die de SNR. Frequentie-offset introduceert echter een tijdsafhankelijke faserotatie die het signaal loskoppelt van het lokale sjabloon, met potentieel dramatisch verlies van detectiegevoeligheid als gevolg.

De impact van frequentie-offset :math:`\Delta f` hangt af van de grootte ervan ten opzichte van de preambleduur (:math:`T_{p}`):

Licht verschoven (Doppler/clock drift): meestal veroorzaakt door ppm-onnauwkeurigheid van de lokale oscillator (LO) of beweging met lage snelheid. Hier geldt :math:`\Delta f \cdot T_{p} \ll 1`. De correlatiepiek verzwakt iets, maar de timing is nog steeds terug te winnen.

In gevallen waar de frequentie-offset volledig onbekend is, zoals bij "cold start"-satellietacquisitie of sterk dynamische UAV-links, kan de coherente som zelfs naar nul uitdoven als de fase over de preamble meer dan :math:`180^{\circ}` roteert (:math:`\Delta f > 1/(2T_{p})`). Detectie wordt dan praktisch onmogelijk, ongeacht de SNR.

Het verlies in correlatiemagnitude door frequentie-offset wordt beschreven door de Dirichlet-kern (de periodieke sinc-functie). Naarmate de frequentie-offset toeneemt, volgt de coherente som van geroteerde vectoren deze sinc-achtige afrol.

Het verlies in dB door frequentie-offset kan benaderd worden met:

:math:`L_{dB}(\Delta f) = 20 \log_{10} \left| \frac{\sin(\pi \Delta f N T_{s})}{N \sin(\pi \Delta f T_{s})} \right|`

Waarbij:

    - :math:`N`: aantal symbolen in de preamble.
    - :math:`T_{s}`: symboolperiode.
    - :math:`\Delta f`: frequentie-offset in Hz.

Als :math:`\Delta f` toeneemt, oscilleert de teller terwijl de noemer groeit, waardoor "nullen" in de gevoeligheid ontstaan. Voor een standaard correlator ligt de eerste nul bij :math:`\Delta f = 1/(N T_{s})`. Zit je offset op een halve binbreedte, dan verlies je ongeveer 3,9 dB, wat je effectieve SNR en :math:`P_{d}` merkbaar verslechtert.

Methoden voor Robuustheid tegen Frequentie-offset
#################################################

A. Coherente Gesegmenteerde Correlator

De preamble met lengte :math:`N` wordt opgesplitst in :math:`M` segmenten van lengte :math:`L = N/M`. Elk segment wordt coherent gecorreleerd, waarna de resultaten worden gecombineerd met compensatie voor de fasedrift tussen segmenten.

:math:`Y_{coh} = \sum_{m=0}^{M-1} \left( \sum_{k=0}^{L-1} r[k+mL] \cdot p^{*}[k] \right) e^{-j \hat{\phi}_m}`

Hierbij is :math:`\hat{\phi}_m` een schatting van de faserotatie voor dat segment. Dit behoudt de SNR-gain van een preamble over volledige lengte, maar vraagt een nauwkeurige frequentieschatting om fasen goed uit te lijnen.

B. Niet-coherente Gesegmenteerde Correlator

Segmenten worden coherent gecorreleerd, maar de magnitudes worden opgeteld, waarbij fase-informatie wordt weggegooid.

:math:`Y_{non-coh} = \sum_{m=0}^{M-1} \left| \sum_{k=0}^{L-1} r[k+mL] \cdot p^{*}[k] \right|^{2}`

Deze aanpak is zeer robuust tegen frequentie-offset (tot ongeveer :math:`1/(L T_{s})`). Nadeel is Non-Coherent Integration Loss. Door magnitudes op te tellen in plaats van complexe waarden stapelt ruis sneller op dan signaal, wat de "post-detection" SNR effectief verlaagt.

C. Brute-force Frequentiezoektocht

De ontvanger draait meerdere parallelle correlators, elk verschoven met een discrete frequentie :math:`\Delta f_{i}`.

Deze methode biedt de beste SNR-prestatie (volledige coherente gain), maar is ook het meest rekentechnisch kostbaar. De "bin spacing" moet klein genoeg zijn (volgens de Dirichlet-formule) zodat het worst-case verlies tussen bins acceptabel blijft (bijv. < 1 dB).

Bij time-domain tapping worden samples geconvolueerd met een vaste set gewichten. Voor een frequentiezoekactie heb je dan een aparte FIR-bank per frequentiebin nodig. Dat is efficiënt voor korte preambles op FPGA's met Xilinx DSP48-slices.
Frequentiedomeinverwerking (FFT): voor een zoekactie neem je de FFT van het inkomende signaal en de preamble. Vermenigvuldiging in het frequentiedomein is equivalent aan correlatie.
De "frequency shift trick": om verschillende frequentie-offsets te testen heb je geen meerdere FFT's nodig. Je kunt de FFT-bins van de preamble circulair verschuiven ten opzichte van het signaal vóór puntsgewijze vermenigvuldiging en IFFT.
Voor continue stromen gebruik je chunkmethoden zoals Overlap-Save of Overlap-Add, zodat correlatiepieken aan de randen van FFT-vensters niet verloren gaan.

Robuustheid tegen frequentie-offset is een afruil tussen processing gain en rekentechnische complexiteit. Niet-coherente gesegmenteerde correlatie is het meest robuust in omgevingen met veel onzekerheid, maar vraagt een hogere linkmarge. Coherente segmentmethoden en brute-force FFT-zoekacties bieden betere gevoeligheid, maar vereisen aanzienlijk meer hardwarebronnen. Begrijpen hoe het Dirichlet-verlies werkt is cruciaal om de benodigde "bin density" van een frequentiezoekende ontvanger te bepalen.

TODO: Licht deze figuur toe en voeg een relevant stuk Python toe aan deze sectie

.. image:: ../_images/detection_freq_offset.svg
   :align: center 
   :target: ../_images/detection_freq_offset.svg
    :alt: Invloed van frequentie-offset op correlatie

*****************************************************************
DSSS-signalen (Direct Sequence Spread Spectrum) Detecteren
*****************************************************************

In een DSSS-systeem is de correlator-detector de vitale schakel die een bruikbaar signaal uit schijnbaar willekeurige ruis haalt. Met een chipsequentie op hoge snelheid ("chipping code") spreidt het systeem de signaalenergie over een veel bredere band dan de oorspronkelijke data nodig heeft. Omdat het totale vermogen gelijk blijft, daalt de vermogensspectrale dichtheid (PSD) sterk. Dit "spectraal verdunnen" kan het signaal onder de thermische ruisvloer brengen, waardoor het voor klassieke smalbandontvangers bijna onzichtbaar wordt. Voor buitenstaanders lijkt het op achtergrondruis, maar de bedoelde ontvanger gebruikt dezelfde chipsequentie om te "ontspreiden", waardoor de energie terug samenkomt in de oorspronkelijke smalle band en smalbandinterferentie juist uitgesmeerd wordt. Dat maakt betrouwbare detectie mogelijk, zelfs in zeer ruisrijke omstandigheden.

De Rol van Autocorrelatie-eigenschappen
########################################

De juiste sequentie kiezen is cruciaal voor synchronisatie en multipad-onderdrukking. Idealiter heeft een sequentie perfecte autocorrelatie: een hoge piek bij perfecte uitlijning en bijna nul op alle andere tijdsverschuivingen. Scherpe autocorrelatiepieken laten de ontvanger locken met sub-chip timingnauwkeurigheid. Als een signaal via een reflectie later binnenkomt, zorgt goede autocorrelatie dat de ontvanger die vertraagde kopie als ongecorreleerde ruis behandelt in plaats van destructieve interferentie.


Veelgebruikte Spreidingssequenties
####################################

Verschillende toepassingen vragen verschillende wiskundige eigenschappen van hun sequenties. Voorbeelden zijn:

- Barker-codes, bekend om de best mogelijke autocorrelatie bij korte lengtes (tot 13), en klassiek gebruikt in 802.11b-wifi.
- M-sequenties (maximale lengte), opgewekt met linear-feedback shift registers (LFSR's), bieden uitstekende pseudo-willekeurigheid en autocorrelatie over lange periodes.
- Gold-codes, afgeleid van paren m-sequenties, leveren een grote set sequenties met gecontroleerde kruiscorrelatie, en zijn daarom standaard in GPS en CDMA met meerdere gelijktijdige signalen.
- Zadoff-Chu (ZC)-sequenties, complexwaardig met constante amplitude en nul autocorrelatie voor alle niet-nul shifts, zijn nu een hoeksteen van LTE en 5G-synchronisatie.
- Kasami-codes, vergelijkbaar met Gold-codes maar vaak met nog lagere kruiscorrelatie bij gegeven lengte, nuttig in hoge-dichtheidsomgevingen.

Chip-timing-synchronisatie in DSSS
####################################################

In een DSSS-systeem hangt het kunnen terughalen van data volledig af van synchronisatie met de inkomende chipsequentie. Omdat chips veel korter zijn dan databits kan zelfs een kleine fractionele timingfout, waarbij de ontvanger "tussen" chips samplet, de correlatiepiek sterk verlagen. We verkennen dit effect met een simpele DSSS-simulatie en plotten de correlatie-output terwijl we de timing-offset variëren van 0 tot 1 chip. Let op: we doen hier geen volledige correlatie, maar een dotproduct bij lag 0, omdat we weten dat daar de piek hoort te zitten.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    # Barker 11 sequence: +1, -1, +1, +1, -1, +1, +1, +1, -1, -1, -1
    barker11 = np.array([1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1])
    samples_per_chip = 100

    # Upsample the sequence to simulate continuous-ish time
    sig = np.repeat(barker11, samples_per_chip)

    offsets = np.linspace(-1.5, 1.5, 500) # Fractional chip offsets
    peaks = []

    for offset in offsets:
        # Shift the signal by a fractional number of chips (converted to samples)
        shift_samples = int(offset * samples_per_chip)
        if shift_samples > 0:
            shifted_sig = np.pad(sig, (shift_samples, 0))[:len(sig)]
        elif shift_samples < 0:
            shifted_sig = np.pad(sig, (0, abs(shift_samples)))[abs(shift_samples):]
        else:
            shifted_sig = sig
            
        # Compute normalized correlation at zero lag for this specific offset
        correlation = np.vdot(sig, shifted_sig) / np.vdot(sig, sig)
        peaks.append(np.abs(correlation))

    plt.figure(figsize=(10, 5))
    plt.plot(offsets, peaks, label='Normalized Correlation', color='blue', linewidth=2)
    plt.axvline(0, color='red', linestyle='--', alpha=0.5, label='Perfect Alignment')
    plt.title('DSSS Correlation Peak vs. Fractional Chip Timing Offset')
    plt.xlabel('Offset (Fraction of a Chip)')
    plt.ylabel('Normalized Correlation Peak Magnitude')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('../_images/detection_dsss.svg', bbox_inches='tight')
    plt.show()

.. image:: ../_images/detection_dsss.svg
   :align: center 
   :target: ../_images/detection_dsss.svg
   :alt: DSSS

De piek ligt zoals verwacht bij offset nul en daalt ongeveer lineair; bij een halve chip-offset zit je rond de helft van de piekwaarde. Na meer dan één chip-offset kan het lijken alsof de correlatie weer stijgt, maar de echte piek blijft laag omdat de uitlijning met de sequentie dan weg is.

****************************************************
Realtime Pakketdetectie in Continue IQ-stromen
****************************************************

Tot nu toe hebben we de theoretische basis van signaaldetectie verkend, van correlators via CFAR-detectors tot spread-spectrumsystemen. Nu brengen we alles samen voor een veelvoorkomend praktijkprobleem: **pakketten detecteren in een continue stroom IQ-samples van een SDR**. Stel je dit scenario voor: een modem of IoT-apparaat verstuurt eens per seconde (of onregelmatig) een datapakket. Je SDR ontvangt continu samples, bijvoorbeeld op 1 MHz. Pakketten komen op onvoorspelbare momenten binnen, verborgen in ruis en interferentie. Je moet:

1. Detecteren wanneer een pakket aankomt
2. De exacte sample-index bepalen waar het start
3. Het pakket uitknippen voor verdere verwerking (demodulatie, decodering, enz.)
4. Dit realtime doen zonder pakketten te missen

Dit is fundamenteel anders dan een vooraf opgenomen IQ-bestand verwerken, waarbij je het hele signaal in één keer kunt analyseren. Hier komen samples continu binnen en moet je met beperkte rekenmiddelen realtime beslissingen nemen. We combineren hiervoor meerdere technieken uit dit hoofdstuk:

1. **Kruiscorrelatie**: om het bekende preamblepatroon te vinden
2. **CFAR-detectie**: om drempels adaptief te zetten bij variërende ruis
3. **Bufferbeheer**: om continue streamdata af te handelen
4. **Piekdetectie**: om precieze pakkettiming te bepalen

Om realtime te kunnen werken verzamelen we samples in **buffers** (bijvoorbeeld chunks van 100.000 samples), draaien de detector op elke buffer en houden toestand bij over buffergrenzen heen, zodat pakketten die twee buffers overspannen niet gemist worden.

Implementatie
##############

Onze detector volgt deze workflow:

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────┐
    │  Continue IQ-stroom van SDR (bijv. 1 MHz sample rate)       │
    └────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  Buffer-opbouw (bijv. 100k samples = 0,1 s)                 │
    └────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  Kruiscorrelatie met bekende preamble                       │
    │  → Levert correlatie versus sample-index op                 │
    └────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  CFAR-drempelberekening                                     │
    │  → Adaptieve drempel die de ruisvloer volgt                 │
    └────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  Piekdetectie (correlatie > drempel)                        │
    │  → Lijst met kandidaat startindices van pakketten           │
    └────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  Pakketextractie & validatie                                │
    │  → Samples uitknippen, doorgeven aan demodulator            │
    └─────────────────────────────────────────────────────────────┘

Om pakketten die over buffergrenzen gaan niet te missen gebruiken we een **overlap-save**-aanpak, waarbij elke buffer de laatste ``N_preamble`` samples van de vorige buffer bevat. Zo zit elk pakket dat aan het einde van buffer ``i`` start volledig in buffer ``i+1``. Dat kost iets extra rekentijd, maar voorkomt gemiste pakketten op buffergrenzen.

Laten we stap voor stap een complete pakketdetector in Python bouwen. We gebruiken een Zadoff-Chu-preamble zoals eerder, maar korter, en implementeren een adaptieve CFAR-detector.

Stap 1: Definieer de Preamble en Parameters
*******************************************

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import correlate
    
    # Preamble: Zadoff-Chu sequence (excellent correlation properties)
    N_zc = 63  # ZC sequence length (typically prime or power of 2 - 1)
    u = 5      # ZC root
    t = np.arange(N_zc)
    preamble = np.exp(-1j * np.pi * u * t * (t + 1) / N_zc)
    
    # System parameters
    sample_rate = 1e6  
    buffer_size = 100000
    overlap_size = len(preamble)  # Overlap to catch boundary packets
    
    # CFAR parameters
    cfar_guard = 10
    cfar_train = 50
    pfa_target = 1e-6
    
    # Packet parameters (for simulation)
    packet_length = 500  # Total packet length in samples (preamble + data)
    snr_db = -5

Stap 2: CFAR-detectorfunctie
*******************************

We gebruiken de Cell-Averaging CFAR (CA-CFAR) van eerder, licht geoptimaliseerd:

.. code-block:: python

    def ca_cfar_1d(signal, num_train, num_guard, pfa):
        """
        1D Cell-Averaging CFAR detector.
        
        Args:
            signal: Input signal (typically correlation magnitude)
            num_train: Number of training cells (on each side)
            num_guard: Number of guard cells (on each side)
            pfa: Target probability of false alarm
            
        Returns:
            threshold: Adaptive threshold array
        """
        n = len(signal)
        threshold = np.zeros(n)
        alpha = num_train * (pfa**(-1/num_train) - 1)
        
        for i in range(n):
            # Define training window indices
            train_start_left = max(0, i - num_guard - num_train)
            train_end_left = max(0, i - num_guard)
            train_start_right = min(n, i + num_guard + 1)
            train_end_right = min(n, i + num_guard + num_train + 1)
            
            # Collect training cells (avoid guard cells and CUT)
            train_cells = np.concatenate([
                signal[train_start_left:train_end_left],
                signal[train_start_right:train_end_right]
            ])
            
            if len(train_cells) > 0:
                noise_est = np.mean(train_cells)
                threshold[i] = alpha * noise_est
        
        return threshold

Stap 3: Pakketdetectiefunctie
**********************************

.. code-block:: python

    def detect_packets(buffer, preamble, cfar_guard, cfar_train, pfa, 
                      min_spacing=None):
        """
        Detect packets in a buffer of IQ samples.
        
        Args:
            buffer: Complex IQ samples
            preamble: Known preamble sequence
            cfar_guard: CFAR guard cells
            cfar_train: CFAR training cells
            pfa: Target false alarm probability
            min_spacing: Minimum samples between detections (prevents duplicates)
            
        Returns:
            detections: List of sample indices where packets start
        """
        # Correlate buffer with preamble
        corr = correlate(buffer, preamble, mode='same')
        corr_power = np.abs(corr)**2
        
        # Compute adaptive threshold
        threshold = ca_cfar_1d(corr_power, cfar_train, cfar_guard, pfa)
        
        # Find peaks above threshold
        detections_raw = np.where(corr_power > threshold)[0]

        # Compensate for correlation offset (peak occurs at len(preamble)//2 after true start)
        half_preamble = len(preamble) // 2
        detections_raw = detections_raw - half_preamble
        
        # Remove edge detections (unreliable)
        half_preamble = len(preamble) // 2
        detections_raw = detections_raw[
            (detections_raw > half_preamble) & 
            (detections_raw < len(buffer) - half_preamble)
        ]
        
        # Remove duplicate detections (peaks close together)
        if min_spacing is None:
            min_spacing = len(preamble)
        
        detections = []
        if len(detections_raw) > 0:
            detections.append(detections_raw[0])
            for det in detections_raw[1:]:
                if det - detections[-1] > min_spacing:
                    detections.append(det)
        
        return detections, corr_power, threshold

Stap 4: Simulatie - Genereer Testsignaal
******************************************

.. code-block:: python

    def generate_packet_stream(preamble, packet_length, num_packets, 
                               sample_rate, snr_db):
        """
        Generate a simulated IQ stream with intermittent packets.
        
        Returns:
            signal: Complex IQ samples
            true_starts: Ground truth packet start indices
        """
        # Calculate noise power from SNR
        signal_power = 1.0  # Normalized preamble power
        noise_power = signal_power / (10**(snr_db/10))
        noise_std = np.sqrt(noise_power / 2)  # Complex noise
        
        # Generate QPSK data (random payload after preamble)
        qpsk_map = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        
        # Time between packets (1 second +/- 20% jitter)
        packets_per_sec = 1
        avg_gap = int(sample_rate / packets_per_sec)
        
        signal = []
        true_starts = []
        
        for i in range(num_packets):
            # Add gap (noise only)
            if i == 0:
                gap_length = np.random.randint(avg_gap//2, avg_gap)
            else:
                gap_length = np.random.randint(int(avg_gap*0.8), int(avg_gap*1.2))
            
            noise = noise_std * (np.random.randn(gap_length) + 
                                1j*np.random.randn(gap_length))
            signal.extend(noise)
            
            # Record true packet start
            true_starts.append(len(signal))
            
            # Add packet (preamble + data)
            data_length = packet_length - len(preamble)
            data = qpsk_map[np.random.randint(0, 4, data_length)]
            packet = np.concatenate([preamble, data])
            
            # Add noise to packet
            packet_noisy = packet + noise_std * (np.random.randn(len(packet)) + 
                                                 1j*np.random.randn(len(packet)))
            signal.extend(packet_noisy)
        
        # Add final gap
        gap_length = np.random.randint(avg_gap//2, avg_gap)
        noise = noise_std * (np.random.randn(gap_length) + 
                            1j*np.random.randn(gap_length))
        signal.extend(noise)
        
        return np.array(signal), true_starts

    # Generate 5 seconds of signal with ~5 packets
    signal, true_starts = generate_packet_stream(
        preamble, packet_length, num_packets=5, 
        sample_rate=sample_rate, snr_db=snr_db
    )
    
    print(f"Generated {len(signal)} samples ({len(signal)/sample_rate:.1f} sec)")
    print(f"True packet starts: {true_starts}")

Stap 5: Detectie in Streaming-modus
****************************************

Nu verwerken we het signaal in stukken en simuleren daarmee realtime streaming:

.. code-block:: python

    def process_stream(signal, preamble, buffer_size, overlap_size,
                      cfar_guard, cfar_train, pfa):
        """
        Process continuous IQ stream in buffers (simulates real-time).
        
        Returns:
            all_detections: List of detected packet starts (global indices)
        """
        all_detections = []
        n_samples = len(signal)
        current_pos = 0
        
        while current_pos < n_samples:
            # Define buffer with overlap
            buffer_start = max(0, current_pos - overlap_size)
            buffer_end = min(n_samples, current_pos + buffer_size)
            buffer = signal[buffer_start:buffer_end]
            
            # Detect packets in this buffer
            detections, corr_power, threshold = detect_packets(
                buffer, preamble, cfar_guard, cfar_train, pfa
            )
            
            # Convert buffer-relative indices to global indices
            for det in detections:
                global_idx = buffer_start + det
                
                # Avoid duplicate detections from overlap region
                if len(all_detections) == 0 or \
                   global_idx - all_detections[-1] > len(preamble):
                    all_detections.append(global_idx)
            
            current_pos += buffer_size
        
        return all_detections
    

    detected_starts = process_stream(
        signal, preamble, buffer_size, overlap_size,
        cfar_guard, cfar_train, pfa_target
    )
    
    print(f"\nDetection Results:")
    print(f"True packets:     {len(true_starts)}")
    print(f"Detected packets: {len(detected_starts)}")
    print(f"Detected starts:  {detected_starts}")

Stap 6: Evalueer Prestaties
*****************************

.. code-block:: python

    # Calculate detection statistics
    tolerance = len(preamble)
    
    matched_detections = []
    false_alarms = []
    
    for det in detected_starts:
        # Check if detection matches any true packet
        matched = False
        for true_start in true_starts:
            if abs(det - true_start) <= tolerance:
                matched_detections.append(det)
                matched = True
                break
        if not matched:
            false_alarms.append(det)
    
    missed_packets = len(true_starts) - len(matched_detections)
    
    print(f"\nPerformance Metrics:")
    print(f"  Correct detections: {len(matched_detections)}/{len(true_starts)}")
    print(f"  Missed packets:     {missed_packets}")
    print(f"  False alarms:       {len(false_alarms)}")
    
    # Calculate timing errors
    timing_errors = []
    for det in matched_detections:
        errors = [abs(det - ts) for ts in true_starts]
        timing_errors.append(min(errors))
    
    if len(timing_errors) > 0:
        print(f"  Timing error (avg): {np.mean(timing_errors):.1f} samples")
        print(f"  Timing error (max): {np.max(timing_errors):.1f} samples")

Stap 7: Visualiseer Resultaten
*******************************

.. code-block:: python

    # Process one buffer for detailed visualization
    buffer_start = max(0, true_starts[0] - 5000)
    buffer_end = min(len(signal), true_starts[0] + 20000)
    viz_buffer = signal[buffer_start:buffer_end]
    
    detections_viz, corr_viz, thresh_viz = detect_packets(
        viz_buffer, preamble, cfar_guard, cfar_train, pfa_target
    )
    
    # Convert to global indices for plotting
    detections_viz_global = [d + buffer_start for d in detections_viz]
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    time_axis = (np.arange(len(viz_buffer)) + buffer_start) / sample_rate * 1000  # ms
    
    # Subplot 1: Received signal power
    axes[0].plot(time_axis, np.abs(viz_buffer)**2, 'gray', alpha=0.6, linewidth=0.5)
    axes[0].set_ylabel('Power')
    axes[0].set_title('Received IQ Signal Power')
    axes[0].grid(True, alpha=0.3)
    
    # Mark true packet locations
    for ts in true_starts:
        if buffer_start <= ts <= buffer_end:
            t_ms = ts / sample_rate * 1000
            axes[0].axvline(t_ms, color='green', linestyle='--', alpha=0.7, 
                          label='True Packet' if ts == true_starts[0] else '')
    axes[0].legend()
    
    # Subplot 2: Correlation output
    axes[1].plot(time_axis, corr_viz, 'blue', linewidth=1, label='Correlation')
    axes[1].plot(time_axis, thresh_viz, 'red', linestyle='--', linewidth=1.5, 
                label='CFAR Threshold')
    axes[1].set_ylabel('Correlation Power')
    axes[1].set_title('Preamble Correlation with Adaptive CFAR Threshold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Subplot 3: Detections
    detection_mask = np.zeros(len(viz_buffer))
    for det in detections_viz:
        detection_mask[det] = corr_viz[det]
    
    axes[2].plot(time_axis, corr_viz, 'blue', alpha=0.4, linewidth=0.8)
    axes[2].scatter(time_axis[detection_mask > 0], detection_mask[detection_mask > 0],
                   color='lime', edgecolors='black', s=100, zorder=5, 
                   label='Detected Packets')
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_ylabel('Correlation Power')
    axes[2].set_title('Detected Packet Locations')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('../_images/detection_realtime.svg', bbox_inches='tight')
    plt.show()

De visualisatie zou het volgende moeten laten zien:

1. **Bovenste plot**: ruwe signaalpower met gemarkeerde echte pakketlocaties
2. **Middelste plot**: correlatie-output met adaptieve CFAR-drempel die de ruisvloer volgt
3. **Onderste plot**: gedetecteerde pakketten gemarkeerd als pieken boven de drempel

.. image:: ../_images/detection_realtime.svg
   :align: center 
   :target: ../_images/detection_realtime.svg
    :alt: Realtime pakketdetectieresultaten

Praktische Overwegingen en Tuning
####################################

Afweging op basis van buffergrootte
************************************

**Grotere buffers (bijv. 1M samples):**

- ✅ Betere CFAR-ruisschatting (meer trainingscellen)
- ✅ Lagere rekentechnische overhead (minder functie-aanroepen)
- ❌ Hogere latency (buffer moet eerst gevuld worden)
- ❌ Meer geheugen nodig

**Kleinere buffers (bijv. 10k samples):**

- ✅ Lagere latency (snellere respons)
- ✅ Minder geheugengebruik
- ❌ CFAR-prestatie verslechtert (minder trainingscellen)
- ❌ Hoger CPU-gebruik (vaker verwerken)

**Aanbeveling**: begin met buffergrootte = 10× tot 100× je preamblelengte. Voor een preamble van 63 samples bij 1 Msps kun je 10k-100k samples proberen.

CFAR-parametertuning
**********************

De drie CFAR-parameters bepalen het detectorgedrag:

**num_guard** (guard-cellen):

- Doel: voorkomt dat signaalenergie in de ruisschatting lekt
- Te klein: signaal lekt in trainingsregio → hogere drempel → gemiste detecties
- Te groot: minder trainingscellen → slechtere ruisschatting
- **Vuistregel**: zet op ongeveer 0,5 tot 1,0× de preamblelengte

**num_train** (training-cellen):

- Doel: schat de lokale ruisvloer
- Te klein: ruisachtige drempel → valse alarmen of gemiste detecties
- Te groot: drempel past zich te traag aan ruisveranderingen aan
- **Vuistregel**: zet op ongeveer 3 tot 5× de preamblelengte

**pfa** (kans op vals alarm):

- Doel: regelt de detectiegevoeligheid
- Te hoog (bijv. 1e-2): veel valse alarmen
- Te laag (bijv. 1e-10): zwakke pakketten worden gemist
- **Vuistregel**: start met 1e-5 voor per-lag PFA en stuur bij op basis van vals-alarmkans op systeemniveau

Onthoud de relatie tussen per-lag en systeemniveau vals-alarmkansen uit het eerdere deel van dit hoofdstuk.
