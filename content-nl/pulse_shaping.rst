.. _pulse-shaping-chapter:

#######################
Pulse Shaping
#######################

Dit hoofdstuk gaat over pulsvorming, inter-symbool-interferentie, matched filters, en raised-cosine filters.
We zullen uiteindelijk Python gebruiken om pulsvorming toe te passen op BPSK-symbolen.
Je kunt dit hoofdstuk als deel 2 van het Filters hoofdstuk opvatten, waarin we een duik nemen in het vormgeven van pulsen.

**********************************
Inter-Symbool-Interferentie (ISI)
**********************************

In het :ref:`filters-chapter` hoofdstuk hebben we geleerd dat blokvormige symbolen/pulsen een groot deel van het spectrum gebruiken, en dat we het gebruik van het spectrum drastisch kunnen verminderen door het *vormgeven* van onze pulsen.
Maar, je kunt niet zomaar elk laagdoorlaatfilter toepassen want dan krijg je last van inter-symbool-interferentie (ISI). Dit is wanneer symbolen elkaar storen en overlappen.

Wanneer we digitale symbolen versturen, dan versturen we ze zij-aan-zij (i.t.t. een bepaalde tijd te wachten tussen pulsen). Wanneer je een pulsvormend filter toepast worden deze pulsen uitgerekt in het tijddomein (om het samen te drukken in frequentie), waardoor aangrenzende symbolen elkaar in de tijd overlappen. Dit overlappen is niet erg zolang het pulsvormende filter aan een eis voldoet: alle pulsen behalve een, moeten optellen tot 0 op elke veelvoud van de symboolperiode :math:`T`. Dit is het beste te begrijpen door een figuur:

.. image:: ../_images/pulse_train.svg
   :align: center 
   :target: ../_images/pulse_train.svg
   :alt: A pulse train of sinc pulses

Zoals je ziet is op elke interval van :math:`T` er maar een puls hoog, terwijl alle andere pulsen 0 zijn en de x-as kruisen. Wanneer de ontvanger het signaal samplet doet het dit op het perfecte moment (wanneer de puls het hoogst is), dus alleen dat moment in tijd is belangrijk. Meestal vindt er nog een vorm van symboolsynchronisatie plaats bij de ontvanger om ervoor te zorgen dat de symbolen inderdaad bij de toppen worden gesampled.

**********************************
Matched Filter
**********************************

Een truc dat in draadloze communicatie wordt toegepast heet matched filters (op elkaar afgestemde filters).
Om deze afstemming van filters te begrijpen zul je eerst deze twee punten moeten snappen:

1. De besproken pulsen hoeven *alleen bij de ontvanger* voor het samplen perfect te zijn uitgelijnd. Tot dat punt maakt het niet uit of er ISI plaatsvindt, de signalen kunnen met ISI zonder problemen door het luchtruim vliegen.

2. We willen een laagdoorlaatfilter bij de zender om te voorkomen dat ons signaal te veel van het spectrum gebruikt. De ontvanger heeft echter ook een laagdoorlaatfilter nodig om zoveel mogelijk ruis/interferentie op on signaal weg te filteren. Dit resulteert in een laagdoorlaatfilter bij de zender (Tx) alsmede de ontvanger (Rx). De ontvanger samplet het signaal dan na beide filters (en natuurlijk de effecten van het draadloze kanaal).

Wat we in moderne communicatie doen, is het opsplitsen van het vormgevende filter tussen Tx en Rx. Ze *moeten* niet identiek zijn, maar, theoretisch gezien, is het *optimaal* om identieke filters te gebruiken bij de aanwezigheid van AWGN, om de SNR te maximaliseren. Deze vorm van filteren heet het "matched filter" concept.

Een andere manier om hierover na te denken is dat de ontvanger het signaal correleert met een bekend signaal. Dit bekende signaal is in wezen de pulsen die worden verzonden, ongeacht de fase- en amplitudeverschuivingen die erop zijn toegepast. Bedenk dat filteren een convolutie actie is, wat in feite gewoon correlatie is (ze geven wiskundig gezien hetzelfde wanneer het voorbeeldsignaal symmetrisch is).
Dit correleren van het ontvangen is signaal met het voorbeeld geeft ons de beste kans om echt te ontvangen wat is verzonden, daarom is het optimaal.
Als analogie kun je denken aan een beeldherkenningssysteem dat gezichten zoekt aan de hand van een sjabloon- of voorbeeldgezicht en deze correleert (2D) met een figuur:

.. image:: ../_images/face_template.png
   :scale: 70 % 
   :align: center 
   :alt: A diagram of a transmit and receive chain, with a Raised Cosine (RC) filter being split into two Root Raised Cosine (RRC) filters

**********************************
Een filter opsplitsen
**********************************

Hoe splitsen we eigenlijk een filter in tweeën? Convolutie is associatief, dit betekent:

.. math::
 (f * g) * h = f * (g * h)

Stel dat :math:`f` onze ingang is, en :math:`g` en :math:`h` de filters.  Nu maakt het niet uit of we eerst :math:`f` filteren met :math:`g` en daarna met :math:`h`, of :math:`f` filteren met een enkel filter gelijk aan :math:`g * h`.

Als je nu ook bedenkt dat convolutie in het tijddomein gelijk is aan vermenigvuldigen in het frequentiedomein:

.. math::
 g(t) * h(t) \leftrightarrow G(f)H(f)

Dan komen we tot de conclusie dat we simpelweg de wortel kunnen nemen in het frequentiedomein om het filter op te splitsen. 

.. math::
 X(f) = X_H(f) X_H(f) \quad \mathrm{where} \quad X_H(f) = \sqrt{X(f)}

Hieronder zie je weer een simpel diagram van een zend- en ontvangstketen waarbij de een Raised-Cosine (RC) filter in tweeën is gesplitst tot twee Root Raised Cosine (RRC) filters; het filter van de zender dient om het signaal te vormen en bandbreedte te beperken, het filter bij de ontvanger dient om ruis- en interferentie te beperken. Samen zorgen ze ervoor dat het signaal bij de demodulator gevormd lijkt te zijn door een enkel RC-filter.

.. image:: ../_images/splitting_rc_filter.svg
   :align: center 
   :target: ../_images/splitting_rc_filter.svg

**********************************
Specifieke pulsvormende filters
**********************************

We weten nu dat we:

1. een filter willen ontwerpen dat de bandbreedte beperkt en dat alle pulsen (behalve een) optellen tot nul bij elke symboolperiode;

2. het filter willen opsplitsen en een helft bij de zender en de andere helft bij de ontvanger willen plaatsen.

Laten we eens naar wat specifieke filters kijken die aan deze eisen voldoen:

Raised-Cosine Filter
#########################

Het meest populaire pulsvormende filter lijkt het "raised-cosine" filter te zijn. Het is inderdaad een goed laagdoorlaatfilter en tegelijkertijd somt het inderdaad op tot 0 bij elke interval van :math:`T`.

.. image:: images/raised_cosine.svg
   :align: center 
   :alt: The raised cosine filter in the time domain with a variety of roll-off values

Het bovenstaande figuur laat de impulsresponsie van het filter zien.
Met :math:`\beta` kun je de steilheid van het filter instellen in het tijddomein, en dus ook omgekeerd evenredig de steilheid in het frequentiedomein:

.. image:: images/raised_cosine_freq.svg
   :align: center 
   :alt: The raised cosine filter in the frequency domain with a variety of roll-off values

Het wordt een raised-cosine filter genoemd omdat bij een :math:`\beta=1` het frequentiedomein een halve cosinus laat zien, raised (opgeduwd) tot boven de x-as.

De impulsresponsie van het filter kun je beschrijven met:

.. math::
 h(t) = \mathrm{sinc}\left( \frac{t}{T} \right) \frac{\cos\left(\frac{\pi\beta t}{T}\right)}{1 - \left( \frac{2 \beta t}{T}   \right)^2}

Je kunt `hier <https://en.wikipedia.org/wiki/Sinc_function>`_ meer lezen over de :math:`\mathrm{sinc}()` functie.
Op andere plekken vind je misschien de vergelijking met een :math:`\frac{1}{T}` ervoor; dit zorgt ervoor dat het filter een versterking heeft zodat het uitgangssignaal dezelfde amplitude heeft als het ingangssignaal (dit is over het algemeen een gewoonte bij het ontwerpen van filters). Maar, omdat we het op een pulstrein van symbolen (bijv. 1'en en -1'en) toepassen en we niet willen dat de amplitude van die symbolen na de pulsvorming verandert, laten we dus die deling achterwege. Dit zal duidelijker worden wanneer we in het Pythonvoorbeeld duiken en de uitkomst weergeven.

Onthoud dat we dit filter egaal opsplitsen tussen Tx en Rx. Dan nu het Root Raised Cosine (RRC) filter!

Root Raised-Cosine Filter
#########################

Bij de zender en ontvanger plaatsen we dus een RRC-filter. Zoals besproken vormen die samen weer een RC-filter.
Helaas wordt de impulsresponsie een rommel omdat we de wortel hebben genomen in het (complexe) frequentiedomein:

.. image:: ../_images/rrc_filter.png
   :scale: 70 % 
   :align: center 
   :alt: Plot of the raised cosine roll-off parameter

Gelukkig wordt het filter zoveel toegepast dat er vele implementaties van te vinden zijn, zelfs `in Python <https://commpy.readthedocs.io/en/latest/generated/commpy.filters.rrcosfilter.html>`_.

Andere pulsvormende filters
###########################

Een ander filter wat aan de eisen voldoet is het Gaussische filter, met een impulsresponsie dat op een Gaussische functie lijkt.
Er is ook nog een sinc filter, een subset van het RC filter met :math:`\beta=0`. Dit is in feite de ideale vorm met een oneindige impulsresponsie en dus ook een filterovergang van praktisch 0 Hz in het frequentiedomein.

**********************************
Roll-Off Factor
**********************************

Laten we :math:`\beta` wat beter gaan bekijken.  
Het is een getal tussen de 0 en 1 en wordt de "roll-off", of soms "excess bandwith", factor genoemd. Dit bepaalt hoe snel het filter afzakt naar nul in het tijddomein. Om het als een filter te kunnen gebruiken moet de impulsresponsie naar 0 gaan aan beide kanten:

.. image:: images/rrc_rolloff.svg
   :align: center 
   :alt: A pulse train of impulses in the time domain simulated in Python

Als resultaat heeft het filter dus meer coëfficiënten nodig naargelang :math:`\beta` lager wordt.
Wanneer :math:`\beta` nul bereikt zal de impulsresponsie nooit meer afzwakken naar 0, dus in de praktijk proberen we :math:`\beta` zo dicht mogelijk bij de nul te krijgen, zonder andere problemen te veroorzaken.
Hoe langzamer de impulsresponsie afzwakt, hoe smaller de bandbreedte van het signaal voor een gegeven symboolsnelheid, wat natuurlijk altijd erg belangrijk is.

Je kunt de bandbreedte in Hz met deze veel gebruikte vergelijking vinden:

.. math::
    \mathrm{BW} = R_S(\beta + 1)

:math:`R_S` is de symboolsnelheid in Hz.  
Voor draadloze communicatie willen we meestal een "roll-off" tussen de 0.2 en 0.5 gebruiken. 
Een goede vuistregel is dat een signaal met een snelheid van :math:`R_s` Hz slecht een beetje meer dan :math:`R_s` aan spectrum zal innemen.
Dus wanneer we met QPSK een miljoen symbolen per seconde (MSps) versturen, zal het rond de 1.3 MHz aan bandbreedte innemen.
In geval van QPSK (2 bits per symbool) levert dat dan een doorvoersnelheid op van 2 Mbps, inclusief de overhead van kanaalcodering en pakketinformatie.

**********************************
Python Oefeningen
**********************************
Laten we eens met Python wat pulsen gaan vormgeven. We zullen hiervoor BPSK-symbolen gebruiken omdat dit reële symbolen zijn en we dus alleen het I-deel hoeven te weergeven, wat iets makkelijker is om te volgen.

.. todo - dit is nog een vage onderbouwing
We gaan 8 samples per symbool toepassen. In plaats van een blokgolf die varieert tussen 1 en -1 zullen we een rij aan pulsen gebruiken. Wanneer je een impuls in een filter stopt zul je de impulsresponsie eruit krijgen. Dus, als je een rij aan pulsen wilt hebben dan zul je het moeten opvullen met nullen zodat je niet een blokgolf krijgt.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal

    num_symbols = 10
    sps = 8

    bits = np.random.randint(0, 2, num_symbols) # De te verzenden bits

    x = np.array([])
    for bit in bits:
        pulse = np.zeros(sps)
        pulse[0] = bit*2-1 # alleen eerste waarde gelijk aan bitwaarde
        x = np.concatenate((x, pulse)) # de 8 samples toevoegen aan x
    plt.figure(0)
    plt.plot(x, '.-')
    plt.grid(True)
    plt.show()

.. image:: ../_images/pulse_shaping_python1.png
   :scale: 80 % 
   :align: center 

Op dit moment bestaan onze symbolen nog uit 1'en en -1'en.
Raak niet verstrikt in het feit dat we impulsen gebruiken, het is waarschijnlijk makkelijker om het te zien als een array:

.. code-block:: python

 bits: [0, 1, 1, 1, 1, 0, 0, 0, 1, 1]
 BPSK symbolen: [-1, 1, 1, 1, 1, -1, -1, -1, 1, 1]
 8 samples per symbool toepassen: [-1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, ...]

We zullen een RC-filter bouwen met een :math:`\beta` van 0.35 en 101 coëfficiënten zodat het signaal genoeg tijd heeft om naar 0 te gaan.
De RC vergelijking vraagt om een periodetijd met een tijdvector, maar voor het gemak zullen we uitgaan van een periodetijd van 1 seconde.
Dit betekent dat onze symboolperiode :math:`T_s` dan 8 is omdat we 8 samples per symbool hebben gebruikt.
Onze tijdvector zal dan gewoon een oplopende lijst van gehele getallen zijn.
Met de manier waarop de filtervergelijking werkt willen we het tijdstip 0 in het midden hebben. De 101 coëfficiënten zullen dan starten bij -51 en eindigen bij +52.

.. code-block:: python

    # het RC filter bouwen
    num_taps = 101
    beta = 0.35
   Ts = sps # samplerate is 1 Hz, sampleperiode is 1, *symbool*periode is 8
    t = np.arange(num_taps) - (num_taps-1)//2 # neemt laatste nummer niet mee
    h = 1/Ts*np.sinc(t/Ts) * np.cos(np.pi*beta*t/Ts) / (1 - (2*beta*t/Ts)**2)
    plt.figure(1)
    plt.plot(t, h, '.')
    plt.grid(True)
    plt.show()

.. image:: ../_images/pulse_shaping_python2.png
   :scale: 80 % 
   :align: center 

De uitgang zakt zeker naar 0 aan beide kanten. De hoeveelheid samples per symbool bepaalt hoe smal dit filter lijkt en hoe snel het naar 0 afzwakt.
De bovenstaande impulsresponsie lijkt op een typisch laagdoorlaatfilter. Er is vrijwel geen onderscheid te maken tussen een vormgevend filter en een algemeen laagdoorlaatfilter.

Nu zullen we het filter op ons signaal :math:`x` toepassen en het resultaat bestuderen.
De for-loop tekent alleen wat extra lijntjes in het figuur, maak je hier niet druk om.

.. code-block:: python 
 
    # signaal x filteren.
    x_shaped = np.convolve(x, h)
    plt.figure(2)
    plt.plot(x_shaped, '.-')
    #wat lijntjes toevoegen op de juiste momenten
    for i in range(num_symbols):
        plt.plot([i*sps+num_taps//2,i*sps+num_taps//2], [0, x_shaped[i*sps+num_taps//2]])
    plt.grid(True)
    plt.show()

.. image:: ../_images/pulse_shaping_python3.svg
   :align: center 
   :target: ../_images/pulse_shaping_python3.svg

Het resultaat is een opsomming van alle impulsresponsen waarbij ongeveer de helft met -1 is vermenigvuldigd. Het ziet er ingewikkeld uit dus we zullen er samen doorheen lopen.

Als eerste zie je samples voor en achter de data vanwege hoe convolutie werkt. De extra samples worden wel meegestuurd, maar bevatten geen 'pieken' van de impulsen.

Als tweede zijn de verticale lijnen aangebracht voor de uitleg. Ze laten zien waar elk samplemoment :math:`T_s` plaatsvindt.
Het zijn de momenten waarop de ontvanger het signaal moet samplen. 
Op elk samplemoment is het signaal precies 1.0 of -1.0: het ideale tijdstip om te samplen.

Zouden we dit signaal op een draaggolf moduleren en verzenden, dan moet de ontvanger zelf bepalen waar de samplemomenten vallen met bijvoorbeeld een symboolsynchronisatie-algoritme. Mocht de ontvanger net te vroeg of te laat samples nemen dan krijgen we waarden die door ISI een beetje afwijken, mochten we veel te vroeg of laat samplen dan krijgen we alleen een boel rare getallen.

Hieronder laten we in een IQ-diagram zien hoe het op tijd (of niet) samplen eruitziet. 

.. image:: ../_images/symbol_sync1.png
   :scale: 50 % 
   :align: center 

Onderstaande diagram laat de ideale samplemomenten zien:

.. image:: ../_images/symbol_sync2.png
   :scale: 40 % 
   :align: center 
   :alt: GNU Radio simulation showing perfect sampling as far as timing

Vergelijk dat eens met de slechtste samplemomenten. We zien nu 3 clusters aan samples in het IQ-diagram. Doordat we midden elk symbool samplen krijgen we totaal verkeerde samples binnen.

.. image:: ../_images/symbol_sync3.png
   :scale: 40 % 
   :align: center 
   :alt: GNU Radio simulation showing imperfect sampling as far as timing

En hier is nog een voorbeeld, ergens tussen bovenstaande voorbeelden in. Nu hebben we vier clusters. Met een hoge SNR zou deze timing net voldoende kunnen zijn, maar het wordt niet aangeraden.

.. image:: ../_images/symbol_sync4.png
   :scale: 40 % 
   :align: center 

De Q waarden worden niet getoond op de tijdsdomein plot omdat ze ongeveer nul zijn, waardoor de IQ-plots alleen horizontaal kunnen spreiden.


*************
OQPSK en MSK
*************

Gewone QPSK kan flinke amplitudeschommelingen hebben, omdat de I- en Q-component soms tegelijk veranderen. Dat kan een probleem zijn voor vermogensversterkers die juist goed werken met een zo constant mogelijke envelop. Hieronder zie je een voorbeeld van QPSK met raised-cosine pulsvorming: boven staan baseband I en Q in het tijddomein apart weergegeven, onder staat de magnitude. Let op de grote schommelingen in magnitude door de bijna-nuldoorgangen wanneer I en Q tegelijk omschakelen. Let ook op de verticale stippellijnen, die de symboolgrenzen aangeven; op die punten zijn zowel I als Q exact 1 of -1. Je ziet ook dat de magnitude op sommige momenten heel dicht bij nul komt.

.. image:: ../_images/qpsk_magnitude.svg
   :align: center 
   :target: ../_images/qpsk_magnitude.svg
   :alt: Voorbeeld van QPSK-magnitude met grote schommelingen door bijna-nuldoorgangen

**Offset QPSK (OQPSK)** is een kleine variatie op standaard QPSK die dit probleem vermindert. Dat werkt door de Q-component een halve symboolperiode te vertragen, zodat I en Q nooit tegelijk veranderen. Het resultaat is dat het signaal op elk moment alleen fase-overgangen van 90 graden maakt (in plaats van mogelijke sprongen van 180 graden), waardoor de envelop veel stabieler blijft. Hieronder zie je OQPSK; we hebben verticale stippellijnen toegevoegd met een halve symboolperiode offset om te laten zien waar de Q-component verandert (dus in het midden van het symbool).

.. image:: ../_images/oqpsk_magnitude.svg
   :align: center 
   :target: ../_images/oqpsk_magnitude.svg
   :alt: Voorbeeld van OQPSK-magnitude met veel kleinere schommelingen door de offset tussen I en Q

De Python-code om OQPSK met raised-cosine pulsvorming te genereren is als volgt:

.. code-block:: python

   # Parameters
   num_symbols = 200
   sps = 32         # samples per symbool
   beta = 0.35      # roll-offfactor
   span = 6         # filterlengte in symbolen (per kant)

   # Genereer QPSK-symbolen
   bits = np.random.randint(0, 4, num_symbols)
   symbols = np.exp(1j * (np.pi/4 + bits * np.pi/2)).astype(complex)  # punten op 45, 135, 225 en 315 graden

   # RC-filter
   t = np.arange(-span * sps, span * sps + 1) / sps  # in symboolperioden
   h = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t)**2 + 1e-20)

   # Vertraag Q-impulsen met een halve symboolperiode voor filtering, zodat het
   # pulsvormingsfilter de opbouw natuurlijk afhandelt (geen opvulartefacten na filtering)
   half = sps // 2
   I_up = np.zeros(num_symbols * sps)
   Q_up = np.zeros(num_symbols * sps)
   I_up[::sps] = np.real(symbols)
   Q_up[half::sps] = np.imag(symbols)
   I_filt = np.convolve(I_up, h, mode='same')
   Q_filt = np.convolve(Q_up, h, mode='same')
   signal = I_filt + 1j * Q_filt

We kunnen nog een stap verder gaan: als we raised-cosine pulsvorming vervangen door een ander type pulsvorming, namelijk half-sine, krijgen we een perfect constante envelop. Het half-sine pulsvormingsfilter is gedefinieerd als :math:`h(t) = \sin\left(\frac{\pi t}{T}\right)`, en deze vorm laat elk symbool vloeiend in- en uitlopen zodat de fase continu en lineair van het ene naar het volgende symbool verandert. Het resultaat heet **Minimum Shift Keying (MSK)** en is een speciaal geval van OQPSK. Als we in de vorige code het raised-cosine filter vervangen door de volgende half-sine filtercode, krijgen we MSK:

.. code-block:: python

 # ...

 # Half-sine pulsvorm (plaats dit op de plek van de RC-filterregels)
 t = np.arange(sps)
 h = np.sin(np.pi * t / sps)

 # ...

.. image:: ../_images/msk_magnitude.svg
   :align: center 
   :target: ../_images/msk_magnitude.svg
   :alt: Example of MSK magnitude showing a constant envelope

De geprinte envelop hierboven zal in essentie constant zijn; dat is precies het kenmerk van MSK.

Let op dat bij OQPSK en MSK de termen "symboolperiode" en "samples per symbool" verwarrend kunnen zijn, omdat een symbool zowel kan verwijzen naar een volledig I+Q-tijdsblok als naar alleen de tijd tussen veranderingen in I of Q (dus half zo lang). In de code hierboven gebruiken we de eerste definitie: een symbool is het volledige I+Q-tijdsblok. Dat is echter niet altijd zo, en je kunt daarom factoren 2 tegenkomen in vergelijkingen zoals die van half-sine.

Een korte blik op de vorm in het frequentiedomein (power spectral density) van deze signalen: voor QPSK of OQPSK met raised-cosine pulsvorming is het spectrum hetzelfde. Het is compact en rolt af volgens de roll-off factor, precies waarom raised-cosine pulsvorming zo populair is.

.. image:: ../_images/qpsk_psd.svg
   :align: center 
   :target: ../_images/qpsk_psd.svg
   :alt: Voorbeeld van QPSK- of OQPSK-PSD wanneer een RC-filter voor pulsvorming wordt gebruikt

Bij MSK zorgt de raised-sine vorm ervoor dat de hoofdlob veel breder is, en het signaal duidelijk hogere sidelobes heeft. Bij signalen met lage SNR zie je die sidelobes vaak niet eens, omdat ze onder de ruisvloer liggen (meer dan 20 dB lager). De afruil is dat we wel een perfect constante envelop krijgen.

.. image:: ../_images/msk_psd.svg
   :align: center 
   :target: ../_images/msk_psd.svg
   :alt: Voorbeeld van MSK-PSD waarbij een raised-sine filter voor pulsvorming wordt gebruikt

MSK wordt vaak gebruikt in toepassingen zoals satellietcommunicatie en deep-space communicatie, waar een constante envelop efficiëntere vermogensversterking mogelijk maakt en spectrumbesparing minder belangrijk is dan maximaal vermogensrendement. Zowel OQPSK als MSK vragen wel om een iets complexere ontvanger dan gewone QPSK, vanwege de offset tussen I en Q.

MSK kun je ook vanuit een heel andere invalshoek afleiden: als een speciaal geval van **Continuous-Phase FSK (CPFSK)**. In CPFSK wordt elk symbool met een van twee frequenties verzonden, en belangrijk is dat de fase nooit wordt gereset; die loopt vloeiend door vanaf het vorige symbool. Die continuiteit houdt de envelop constant en het spectrum compact. MSK is CPFSK met modulatie-index :math:`h = 0.5`, wat betekent dat de twee tonen precies :math:`\Delta f = \frac{1}{2T}` Hz uit elkaar liggen, waarbij :math:`T` de symboolperiode is. Het basebandsignaal is:

.. math::

  s(t) = e^{j 2\pi \frac{h}{2T} \int_{-\infty}^{t} d(\tau)\, d\tau}

waar :math:`d(\tau) \in \{-1, +1\}` de NRZ-datastroom is. In de praktijk stapelt de integraal simpelweg fase op: elk bit roteert de fase met :math:`\pm \frac{\pi}{2}` over een symboolperiode. De Python-code om MSK via de CPFSK-aanpak te genereren is als volgt. Let op dat :code:`sps` overal door 2 is gedeeld, omdat de symboolperiode half zo lang is in de CPFSK-aanpak: elk symbool komt dan overeen met een verandering in I of Q, niet in beide tegelijk.

.. code-block:: python

   bits = np.random.randint(0, 2, num_symbols)
   symbols = 2 * bits - 1 # map {0,1} naar {-1, +1}

   # Bouw de instantane frequentieafwijking op
   mod_index = 0.5
   t = np.arange(num_symbols * sps / 2) / (sps / 2)
   freq_dev = np.zeros(num_symbols * sps // 2)
   for k, a in enumerate(symbols):
      freq_dev[k * sps // 2 : (k + 1) * sps // 2] = a * mod_index / 2.0

   phase = 2.0 * np.pi * np.cumsum(freq_dev) / (sps / 2) # fase cumulatief opbouwen
   signal = np.exp(1j * phase)

En zoals je ziet, ziet het er exact hetzelfde uit als onze eerdere MSK, maar nu gegenereerd via een volledig andere aanpak.

.. image:: ../_images/cpfsk_magnitude.svg
   :align: center 
   :target: ../_images/cpfsk_magnitude.svg
   :alt: Voorbeeld van CPFSK-magnitude die laat zien dat het overeenkomt met MSK
