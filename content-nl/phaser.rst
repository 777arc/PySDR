.. _phaser-chapter:

####################################
Phased Arrays met Phaser
####################################
   
In dit hoofdstuk gebruiken we de `Analog Devices Phaser <https://wiki.analog.com/resources/eval/user-guides/circuits-from-the-lab/cn0566>`_ (ook bekend als CN0566 of ADALM-PHASER), een voordelige 8-kanaals phased-array-SDR die een PlutoSDR, Raspberry Pi en ADAR1000-bundelvormers combineert en ontworpen is voor gebruik rond 10,25 GHz. We behandelen de installatie- en calibratiestappen en lopen daarna door enkele voorbeelden van bundelvorming in Python. Voor wie geen Phaser heeft, zijn screenshots en animaties toegevoegd van wat je normaal zou zien.

.. image:: ../_images/phaser_on_tripod.png
   :scale: 60 % 
   :align: center
   :alt: De Phaser (CN0566) van Analog Devices

************************
Hardware-overzicht
************************

.. image:: ../_images/phaser_front_and_back.png
   :scale: 40 % 
   :align: center
   :alt: Voor- en achterkant van de Phaser-unit

De Phaser is een enkel bord met daarop de phased array en diverse andere componenten, met aan de ene zijde een Raspberry Pi en aan de andere zijde een Pluto. Het blokschema op hoofdlijnen staat hieronder. Enkele belangrijke punten:

1. Hoewel het op een 32-element 2D-array lijkt, is het in werkelijkheid een 8-element 1D-array
2. Beide ontvangstkanalen van de Pluto worden gebruikt (het tweede kanaal gebruikt een u.FL-connector)
3. De onboard LO wordt gebruikt om het ontvangen signaal van rond 10,25 GHz naar rond 2 GHz te downconverten, zodat de Pluto het kan ontvangen
4. Elke ADAR1000 heeft vier faseschuivers met instelbare gain, en alle vier kanalen worden opgeteld voordat ze naar de Pluto gaan
5. De Phaser bevat in essentie twee "subarrays", elk met vier kanalen
6. Niet getoond: GPIO- en seriele signalen van de Raspberry Pi die verschillende onderdelen op de Phaser aansturen

.. image:: ../_images/phaser_components.png
   :scale: 40 % 
   :align: center
   :alt: Componenten van de Phaser (CN0566), inclusief ADF4159, LTC5548 en ADAR1000

Voor nu negeren we de zendkant van de Phaser, omdat we in dit hoofdstuk alleen de HB100 als testzender gebruiken. De ADF4159 is een frequentiesynthesizer die een toon tot 13 GHz kan maken; dit is onze lokale oscillator (LO). Deze LO gaat naar de mixer LTC5548, die zowel upconversion als downconversion kan doen, maar wij gebruiken downconversion. Daarbij worden LO en een signaal tussen 2 en 14 GHz met elkaar vermenigvuldigd, wat een frequentieverschuiving geeft. Het resulterende downconverted signaal kan tussen DC en 6 GHz liggen, al mikken wij op ongeveer 2 GHz. De ADAR1000 is een 4-kanaals analoge bundelvormer; daarom gebruikt de Phaser er twee. Een analoge bundelvormer heeft per kanaal onafhankelijk instelbare fase en gain, zodat elk kanaal tijdsvertraging en attenuatie kan krijgen voordat alle kanalen in het analoge domein worden opgeteld (tot een enkel kanaal). Op de Phaser levert elke ADAR1000 een signaal dat wordt downconverted en daarna door de Pluto wordt ontvangen. Met de Raspberry Pi kunnen we fase en gain van alle acht kanalen realtime regelen voor bundelvorming. We hebben ook de optie voor tweekanaals digitale bundelvorming/arrayverwerking, besproken in het volgende hoofdstuk.

Voor geinteresseerden staat hieronder een iets gedetailleerder blokschema.

.. image:: ../_images/phaser_detailed_block_diagram.png
   :scale: 80 % 
   :align: center
   :alt: Gedetailleerd blokschema van de Phaser (CN0566)


************************
SD-kaartvoorbereiding
************************

We gaan ervan uit dat je de Raspberry Pi op de Phaser gebruikt (direct, met monitor/toetsenbord/muis). Dat vereenvoudigt de setup, omdat Analog Devices een kant-en-klaar SD-kaartimage aanbiedt met alle benodigde drivers en software. Je kunt het SD-image downloaden en instructies voor het flashen vinden `hier <https://wiki.analog.com/resources/tools-software/linux-software/kuiper-linux>`_. Het image is gebaseerd op Raspberry Pi OS en bevat de benodigde software al vooraf geinstalleerd.

************************
Hardwarevoorbereiding
************************

1. Verbind de MIDDELSTE micro-USB-poort van de Pluto met de Raspberry Pi
2. Optioneel: schroef voorzichtig het statief in de statiefaansluiting
3. We gaan ervan uit dat je een HDMI-scherm, USB-toetsenbord en USB-muis op de Raspberry Pi gebruikt
4. Voed de Pi en het Phaser-bord via de USB-C-poort van de Phaser (CN0566), dus sluit GEEN aparte voeding op de USB-C van de Raspberry Pi aan

************************
Software-installatie
************************

Nadat je met het voorgebouwde image bent opgestart op de Raspberry Pi (standaard gebruiker/wachtwoord: analog/analog), is het aanbevolen om de volgende stappen uit te voeren:

.. code-block:: bash

 wget https://github.com/mthoren-adi/rpi_setup_stuff/raw/main/phaser/phaser_sdcard_setup.sh
 sudo chmod +x phaser_sdcard_setup.sh
 ./phaser_sdcard_setup.sh
 sudo reboot
 
 sudo raspi-config

Voor extra hulp bij het opzetten van de Phaser, zie de `Phaser wiki quickstart-pagina <https://wiki.analog.com/resources/eval/user-guides/circuits-from-the-lab/cn0566/quickstart>`_.

************************
HB100-setup
************************

.. image:: ../_images/phaser_hb100.png
   :scale: 50 % 
   :align: center
   :alt: HB100 die met de Phaser wordt meegeleverd

De HB100 die bij de Phaser wordt geleverd is een voordelige Doppler-radarmodule die we als testzender gebruiken, omdat deze een continue toon rond 10 GHz uitzendt. Hij werkt op 2 AA-batterijen of een 3V-labvoeding, en bij inschakelen brandt er een constante rode LED.

Omdat de HB100 goedkoop is en eenvoudige RF-componenten gebruikt, varieert de zendfrequentie per exemplaar met honderden MHz, een bereik groter dan de maximale bandbreedte die de Pluto kan ontvangen (56 MHz). Om de Pluto en downconverter zo af te stemmen dat we het HB100-signaal zeker ontvangen, moeten we dus eerst de zendfrequentie van de HB100 bepalen. Dat doen we met een voorbeeldapp van Analog Devices die een frequentiesweep uitvoert en FFT's berekent om een piek te vinden. Zorg dat de HB100 aan staat en in de buurt van de Phaser is, en voer daarna het hulpprogramma uit met:

.. code-block:: bash

 cd ~/pyadi-iio/examples/phaser
 python phaser_find_hb100.py

Dit zou in dezelfde map een bestand genaamd hb100_freq_val.pkl moeten maken. Dat bestand bevat de HB100-zendfrequentie in Hz (gepickled, dus niet als platte tekst leesbaar), die we in de volgende stap gebruiken.

************************
Calibration
************************

Tot slot moeten we de phased array calibreren. Daarvoor houd je de HB100 op boresight van de array (0 graden). De zijde van de HB100 met de barcode is de zendzijde; houd die op enige afstand recht voor en gecentreerd op de Phaser en richt hem direct op de Phaser. In de volgende stap kun je met verschillende hoeken en orientaties experimenteren, maar voer nu eerst de calibratietool uit:

.. code-block:: bash

 python phaser_examples.py cal

Dit maakt in dezelfde map nog twee picklebestanden aan: phase_cal_val.pkl en gain_cal_val.pkl. Elk bestand bevat een array met 8 waarden die de fase- en gain-correcties per kanaal aangeven. Deze waarden zijn uniek per Phaser, omdat productievariaties een rol spelen. Herhaalde runs van deze tool geven normaal gesproken licht verschillende waarden.

************************
Voorgebouwde Voorbeeldapp
************************

Nu we de Phaser hebben gecalibreerd en de HB100-frequentie kennen, kunnen we de voorbeeldapp van Analog Devices starten.

.. code-block:: bash

 python phaser_gui.py

Als je linksonder het vakje "Auto Refresh Data" aanvinkt, zou de app moeten starten. Wanneer je de HB100 op boresight van de Phaser houdt, zou je iets als het volgende moeten zien.

.. image:: ../_images/phaser_gui.png
   :scale: 50 % 
   :align: center
   :alt: Phaser-voorbeeldtool met GUI van Analog Devices

************************
Phaser in Python
************************

We gaan nu naar het praktische Python-gedeelte. Voor wie geen Phaser heeft, zijn screenshots en animaties toegevoegd.

Phaser en Pluto initialiseren
##############################

De volgende Python-code zet onze Phaser en Pluto op. Op dit punt heb je de calibratiestappen al uitgevoerd, die drie picklebestanden opleveren. Zorg dat je het onderstaande script uitvoert vanuit dezelfde map als die picklebestanden.

Er zijn veel instellingen, dus het is prima als je niet direct de hele code begrijpt. Let vooral op dat we een sample rate van 30 MHz gebruiken, handmatige gain op een lage waarde zetten, alle elementgains gelijk maken en de array op boresight (0 graden) richten.

.. code-block:: python

 import time
 import sys
 import matplotlib.pyplot as plt
 import numpy as np
 import pickle
 from adi import ad9361
 from adi.cn0566 import CN0566
 
 phase_cal = pickle.load(open("phase_cal_val.pkl", "rb"))
 gain_cal = pickle.load(open("gain_cal_val.pkl", "rb"))
 signal_freq = pickle.load(open("hb100_freq_val.pkl", "rb"))
 d = 0.014  # element to element spacing of the antenna
 
 phaser = CN0566(uri="ip:localhost")
 sdr = ad9361(uri="ip:192.168.2.1")
 phaser.sdr = sdr
 print("PlutoSDR and CN0566 connected!")
 
 time.sleep(0.5) # recommended by Analog Devices
 
 phaser.configure(device_mode="rx")
 
 # Set all antenna elements to half scale - a typical HB100 will have plenty of signal power.
 gain = 64 # 64 is about half scale
 for i in range(8):
     phaser.set_chan_gain(i, gain, apply_cal=False)
 
 # Aim the beam at boresight (zero degrees)
 phaser.set_beam_phase_diff(0.0)
 
 # Misc SDR settings, not super critical to understand
 sdr._ctrl.debug_attrs["adi,frequency-division-duplex-mode-enable"].value = "1"
 sdr._ctrl.debug_attrs["adi,ensm-enable-txnrx-control-enable"].value = "0" # Disable pin control so spi can move the states
 sdr._ctrl.debug_attrs["initialize"].value = "1"
 sdr.rx_enabled_channels = [0, 1] # enable Rx1 and Rx2
 sdr._rxadc.set_kernel_buffers_count(1) # No stale buffers to flush
 sdr.tx_hardwaregain_chan0 = int(-80) # Make sure the Tx channels are attenuated (or off)
 sdr.tx_hardwaregain_chan1 = int(-80)
 
 # These settings are basic PlutoSDR settings we have seen before
 sample_rate = 30e6
 sdr.sample_rate = int(sample_rate)
 sdr.rx_buffer_size = int(1024)  # samples per buffer
 sdr.rx_rf_bandwidth = int(10e6)  # analog filter bandwidth
 
 # Manually gain (no automatic gain control) so that we can sweep angle and see peaks/nulls
 sdr.gain_control_mode_chan0 = "manual"
 sdr.gain_control_mode_chan1 = "manual"
 sdr.rx_hardwaregain_chan0 = 10 # dB, 0 is the lowest gain.  the HB100 is pretty loud
 sdr.rx_hardwaregain_chan1 = 10 # dB
 
 sdr.rx_lo = int(2.2e9) # The Pluto will tune to this freq
 
 # Set the Phaser's PLL (the ADF4159 onboard) to downconvert the HB100 to 2.2 GHz plus a small offset
 offset = 1000000 # add a small arbitrary offset just so we're not right at 0 Hz where there's a DC spike
 phaser.lo = int(signal_freq + sdr.rx_lo - offset)


Samples Ontvangen van de Pluto
################################

Op dit punt zijn de Phaser en Pluto geconfigureerd en klaar. We kunnen nu data van de Pluto ontvangen. Laten we een enkele batch van 1024 samples ophalen en daarna van beide kanalen de FFT nemen.

.. code-block:: python

 # Grab some samples (whatever we set rx_buffer_size to), remember we are receiving on 2 channels at the same time
 data = sdr.rx()
 
 # Take FFT
 PSD0 = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(data[0])))**2)
 PSD1 = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(data[1])))**2)
 f = np.linspace(-sample_rate/2, sample_rate/2, len(data[0]))
 
 # Time plot helps us check that we see the HB100 and that we're not saturated (ie gain isnt too high)
 plt.subplot(2, 1, 1)
 plt.plot(data[0].real) # Only plot real part
 plt.plot(data[1].real)
 plt.xlabel("Data Point")
 plt.ylabel("ADC output")
 
 # PSDs show where the HB100 is and verify both channels are working
 plt.subplot(2, 1, 2)
 plt.plot(f/1e6, PSD0)
 plt.plot(f/1e6, PSD1)
 plt.xlabel("Frequency [MHz]")
 plt.ylabel("Signal Strength [dB]")
 plt.tight_layout()
 plt.show()

Wat je hier ziet hangt af van of de HB100 aan staat en waar hij op gericht is. Als je hem op enige afstand van de Phaser houdt en naar het midden richt, zou je ongeveer dit moeten zien:

.. image:: ../_images/phaser_rx_psd.png
   :scale: 100 % 
   :align: center
   :alt: Eerste Phaser-voorbeeld

Let op de sterke piek rond 0 Hz; de tweede, kleinere piek is een artefact dat je kunt negeren, omdat die ongeveer 40 dB lager ligt. De bovenste plot in het tijddomein toont het reele deel van de twee kanalen, waardoor de relatieve amplitude iets varieert afhankelijk van de positie van de HB100.

Bundelvorming Uitvoeren
##############################

Nu gaan we echt de fase sweepen. In de volgende code sweepen we de fase van -180 tot +180 graden, met stappen van 2 graden. Let op: dit is niet direct de hoek waar de bundelvormer naartoe wijst; het is het faseverschil tussen aangrenzende kanalen. We moeten de bijbehorende aankomsthoek per fasestap berekenen met de lichtsnelheid, de RF-frequentie van het ontvangen signaal en de elementafstand van de Phaser. Het faseverschil tussen aangrenzende elementen is:

.. math::

 \phi = \frac{2 \pi d}{\lambda} \sin(\theta_{AOA})

waar :math:`\theta_{AOA}` de aankomsthoek van het signaal is ten opzichte van boresight, :math:`d` de antenneafstand in meter, en :math:`\lambda` de golflengte van het signaal. Met de formule voor golflengte en opgelost naar :math:`\theta_{AOA}` krijgen we:

.. math::

 \theta_{AOA} = \sin^{-1}\left(\frac{c \phi}{2 \pi f d}\right)

Dat zie je terug bij de berekening van :code:`steer_angle` hieronder:

.. code-block:: python

 powers = [] # main DOA result
 angle_of_arrivals = []
 for phase in np.arange(-180, 180, 2): # sweep over angle
     print(phase)
     # set phase difference between the adjacent channels of devices
     for i in range(8):
         channel_phase = (phase * i + phase_cal[i]) % 360.0 # Analog Devices had this forced to be a multiple of phase_step_size (2.8125 or 360/2**6bits) but it doesn't seem nessesary
         phaser.elements.get(i + 1).rx_phase = channel_phase
     phaser.latch_rx_settings() # apply settings
 
     steer_angle = np.degrees(np.arcsin(max(min(1, (3e8 * np.radians(phase)) / (2 * np.pi * signal_freq * phaser.element_spacing)), -1))) # arcsin argument must be between 1 and -1, or numpy will throw a warning
     # If you're looking at the array side of Phaser (32 squares) then add a *-1 to steer_angle
     angle_of_arrivals.append(steer_angle) 
     data = phaser.sdr.rx() # receive a batch of samples
     data_sum = data[0] + data[1] # sum the two subarrays (within each subarray the 4 channels have already been summed)
     power_dB = 10*np.log10(np.sum(np.abs(data_sum)**2))
     powers.append(power_dB)
     # in addition to just taking the power in the signal, we could also do the FFT then grab the value of the max bin, effectively filtering out noise, results came out almost exactly the same in my tests
     #PSD = 10*np.log10(np.abs(np.fft.fft(data_sum * np.blackman(len(data_sum))))**2) # in dB
 
 powers -= np.max(powers) # normalize so max is at 0 dB
 
 plt.plot(angle_of_arrivals, powers, '.-')
 plt.xlabel("Angle of Arrival")
 plt.ylabel("Magnitude [dB]")
 plt.show()

Voor elke :code:`phase`-waarde (dit is dus het faseverschil tussen aangrenzende elementen) zetten we de faseschuivers, na optellen van de fasecalibratiewaarden en normalisatie van graden naar 0-360. Daarna halen we met :code:`rx()` een batch samples op, sommeren we de twee kanalen en berekenen we het signaalvermogen. Vervolgens plotten we vermogen tegen aankomsthoek. Het resultaat ziet er ongeveer zo uit:

.. image:: ../_images/phaser_sweep.png
   :scale: 100 % 
   :align: center
   :alt: Phaser enkele sweep

In dit voorbeeld werd de HB100 iets naast boresight gehouden.

Als je een polaire plot wilt, kun je in plaats daarvan het volgende gebruiken:

.. code-block:: python

 # Polar plot
 fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
 ax.plot(np.deg2rad(angle_of_arrivals), powers) # x axis in radians
 ax.set_rticks([-40, -30, -20, -10, 0])  # Less radial ticks
 ax.set_thetamin(np.min(angle_of_arrivals)) # in degrees
 ax.set_thetamax(np.max(angle_of_arrivals))
 ax.set_theta_direction(-1) # increase clockwise
 ax.set_theta_zero_location('N') # make 0 degrees point up
 ax.grid(True)
 plt.show()

.. image:: ../_images/phaser_sweep_polar.png
   :scale: 100 % 
   :align: center
   :alt: Phaser enkele sweep met polaire plot

Door het maximum te nemen kunnen we de aankomstrichting van het signaal schatten.

Realtime en met Ruimtelijke Tapering
######################################

Laten we nu kort stilstaan bij ruimtelijke tapering. Tot nu toe hielden we de gaininstellingen van elk kanaal gelijk, zodat alle acht kanalen gelijk worden opgeteld. Net zoals we een venster toepassen voor een FFT, kunnen we in het ruimtelijke domein een venster toepassen door gewichten op deze acht kanalen te zetten. We gebruiken dezelfde vensterfuncties zoals Hanning, Hamming, enzovoort. We passen de code ook aan voor realtime uitvoering:

.. code-block:: python

 plt.ion() # needed for real-time view
 print("Starting, use control-c to stop")
 try:
     while True:
         powers = [] # main DOA result
         angle_of_arrivals = []
         for phase in np.arange(-180, 180, 6): # sweep over angle
             # set phase difference between the adjacent channels of devices
             for i in range(8):
                 channel_phase = (phase * i + phase_cal[i]) % 360.0 # Analog Devices had this forced to be a multiple of phase_step_size (2.8125 or 360/2**6bits) but it doesn't seem nessesary
                 phaser.elements.get(i + 1).rx_phase = channel_phase
            
             # set gains, incl the gain_cal, which can be used to apply a taper.  try out each one!
             gain_list = [127] * 8 # rectangular window          [127, 127, 127, 127, 127, 127, 127, 127]
             #gain_list = np.rint(np.hamming(8) * 127)         # [ 10,  32,  82, 121, 121,  82,  32,  10]
             #gain_list = np.rint(np.hanning(10)[1:-1] * 127)  # [ 15,  52,  95, 123, 123,  95,  52,  15]
             #gain_list = np.rint(np.blackman(10)[1:-1] * 127) # [  6,  33,  80, 121, 121,  80,  33,   6]
             #gain_list = np.rint(np.bartlett(10)[1:-1] * 127) # [ 28,  56,  85, 113, 113,  85,  56,  28]
             for i in range(8):
                 channel_gain = int(gain_list[i] * gain_cal[i])
                 phaser.elements.get(i + 1).rx_gain = channel_gain
 
             phaser.latch_rx_settings() # apply settings
 
             steer_angle = np.degrees(np.arcsin(max(min(1, (3e8 * np.radians(phase)) / (2 * np.pi * signal_freq * phaser.element_spacing)), -1))) # arcsin argument must be between 1 and -1, or numpy will throw a warning
             angle_of_arrivals.append(steer_angle) 
             data = phaser.sdr.rx() # receive a batch of samples
             data_sum = data[0] + data[1] # sum the two subarrays (within each subarray the 4 channels have already been summed)
             power_dB = 10*np.log10(np.sum(np.abs(data_sum)**2))
             powers.append(power_dB)
 
         powers -= np.max(powers) # normalize so max is at 0 dB
 
         # Real-time view
         plt.plot(angle_of_arrivals, powers, '.-')
         plt.xlabel("Angle of Arrival")
         plt.ylabel("Magnitude [dB]")
         plt.draw()
         plt.pause(0.001)
         plt.clf()
 
 except KeyboardInterrupt:
     sys.exit() # quit python

Je zou nu een realtimeversie van de vorige oefening moeten zien. Wissel eens van :code:`gain_list` om met verschillende vensters te experimenteren. Hier is een voorbeeld met het rechthoekige venster (dus zonder vensterfunctie):

.. image:: ../_images/phaser_animation_rect.gif
   :scale: 100 % 
   :align: center
   :alt: Bundelvormingsanimatie met de Phaser en rechthoekig venster

en hier een voorbeeld met het Hamming-venster:

.. image:: ../_images/phaser_animation_hamming.gif
   :scale: 100 % 
   :align: center
   :alt: Bundelvormingsanimatie met de Phaser en Hamming-venster

Let op het ontbreken van sidelobes bij Hamming. In feite zal elk venster behalve Rectangular de sidelobes sterk verminderen, maar in ruil daarvoor wordt de hoofdlob iets breder.

************************
Monopulse Tracking
************************

Tot nu toe voerden we losse sweeps uit om de aankomsthoek van een testzender (de HB100) te vinden. Stel nu dat we continu een communicatie- of radarsignaal willen ontvangen dat beweegt en daardoor een veranderende aankomsthoek heeft. Dit noemen we tracking, en het veronderstelt dat we al een ruwe schatting van de aankomsthoek hebben (de eerste sweep heeft dus een interessant signaal gevonden). We gebruiken monopulse-tracking om de gewichten adaptief bij te werken en de hoofdlob in de tijd op het signaal gericht te houden, al zijn er ook andere trackingmethoden.

Monopulse-tracking werd in 1943 bedacht door Robert Page bij het Naval Research Laboratory (NRL). Het basisidee is om twee bundels te gebruiken die beide iets afwijken van de huidige aankomsthoek (of onze schatting daarvan), maar aan tegengestelde kanten zoals in het diagram hieronder.

.. image:: ../_images/monopulse.svg
   :align: center 
   :target: ../_images/monopulse.svg
   :alt: Monopulse-diagram met twee bundels en de sombundel

Vervolgens nemen we digitaal zowel de som als het verschil (delta) van deze twee bundels. Dat betekent dat we twee digitale kanalen van de Phaser gebruiken, dus dit is een hybride array-aanpak (al kun je som en verschil ook analoog realiseren met aangepaste hardware). De sombundel is gecentreerd rond de huidige aankomsthoekschatting, zoals hierboven, en kan worden gebruikt voor demodulatie/decodering van het doelsignaal. De delta-bundel is lastiger te visualiseren, maar heeft een null op de geschatte aankomsthoek. We kunnen de verhouding tussen sombundel en delta-bundel (de error) gebruiken voor tracking. Dit wordt het duidelijkst met een korte Python-snippet; de :code:`rx()`-functie geeft een batch samples van beide kanalen terug. In de code hieronder is :code:`data[0]` het eerste Pluto-kanaal (eerste set van vier Phaser-elementen) en :code:`data[1]` het tweede kanaal (tweede set van vier elementen). Om twee bundels te maken sturen we deze twee sets apart aan. Som, delta en error berekenen we als volgt:

.. code-block:: python

   data = phaser.sdr.rx()
   sum_beam = data[0] + data[1]
   delta_beam = data[0] - data[1]
   error = np.mean(np.real(delta_beam / sum_beam))

Het teken van de error vertelt ons aan welke kant het signaal werkelijk zit, en de grootte van de error geeft aan hoe ver we van het signaal af zitten. Met die informatie werken we de aankomsthoekschatting en de gewichten bij. Door dit realtime te herhalen kunnen we het signaal volgen.

In het volledige Python-voorbeeld beginnen we met de code van de eerdere 180-gradensweep. De enige toevoeging is dat we de fase nemen waarbij het ontvangen vermogen maximaal was:

.. code-block:: python

   # Sweep phase once to get initial estimate for AOA, using code above
   # ...
   current_phase = phase_angles[np.argmax(powers)]
   print("max_phase:", current_phase)

Vervolgens maken we twee bundels: eerst 5 graden lager en 5 graden hoger dan de huidige schatting. Let op dat dit in fase-eenheden is; we hebben nog niet naar stuurhoek omgerekend, al zijn die vergelijkbaar. De volgende code is in essentie twee kopieen van de eerdere code voor faseschuivers per kanaal, met dit verschil: de eerste 4 elementen voor de lage bundel en de laatste 4 voor de hoge bundel:

.. code-block:: python

   # Now we create the two beams on either side of our current estimate
   phase_offset = np.radians(5) # TRY TWEAKING THIS - specify offset from center in degrees
   phase_lower = current_phase - phase_offset
   phase_upper = current_phase + phase_offset
   # first 4 elements will be used for lower beam
   for i in range(0, 4): 
      channel_phase = (phase_lower * i + phase_cal[i]) % 360.0
      phaser.elements.get(i + 1).rx_phase = channel_phase
   # last 4 elements will be used for upper beam
   for i in range(4, 8): 
      channel_phase = (phase_upper * i + phase_cal[i]) % 360.0
      phaser.elements.get(i + 1).rx_phase = channel_phase
   phaser.latch_rx_settings() # apply settings

Voordat we echte tracking doen, testen we dit eerst door de bundelgewichten constant te houden en de HB100 links en rechts te bewegen (nadat de initialisatie de starthoek heeft bepaald):

.. code-block:: python

   print("START MOVING THE HB100 A LITTLE LEFT AND RIGHT")
   error_log = []
   for i in range(1000):
      data = phaser.sdr.rx() # receive a batch of samples
      sum_beam = data[0] + data[1]
      delta_beam = data[0] - data[1]
      error = np.mean(np.real(delta_beam / sum_beam))
      error_log.append(error)
      print(error)
      time.sleep(0.01)

   plt.plot(error_log)
   plt.plot([0,len(error_log)], [0,0], 'r--')
   plt.xlabel("Time")
   plt.ylabel("Error")
   plt.show()

.. image:: ../_images/monopulse_waving.svg
   :align: center 
   :target: ../_images/monopulse_waving.svg
   :alt: Errorfunctie voor monopulse-tracking zonder de gewichten bij te werken

Wat hier gebeurt: ik beweeg de HB100 rond. Ik begin met een vaste positie terwijl de 180-gradensweep loopt, daarna beweeg ik hem iets naar rechts en wiebel ik ermee, vervolgens naar links van de startpositie en weer wat beweging. Rond tijd = 400 in de plot ga ik weer naar de andere kant en houd ik hem kort stil, daarna nogmaals wat beweging. De kern: hoe verder de HB100 van de starthoek zit, hoe groter de error, en het teken van de error geeft aan aan welke kant de HB100 zich bevindt ten opzichte van de starthoek.

Laten we nu de error gebruiken om de gewichten bij te werken. We vervangen de vorige for-loop door een nieuwe for-loop rond het volledige proces. Voor de duidelijkheid staat hieronder het complete codevoorbeeld, behalve het initiele deel met de 180-gradensweep:

.. code-block:: python

   # Sweep phase once to get initial estimate for AOA
   # ...
   current_phase = phase_angles[np.argmax(powers)]
   print("max_phase:", current_phase)

   # Now we'll actually update the current_phase based on the error
   print("START MOVING THE HB100 A LITTLE LEFT AND RIGHT")
   phase_log = []
   error_log = []
   for ii in range(500):
      # Now we create the two beams on either side of our current estimate, using the specified offset
      phase_offset = np.radians(5)
      phase_lower = current_phase - phase_offset
      phase_upper = current_phase + phase_offset
      # first 4 elements will be used for lower beam
      for i in range(0, 4): 
            channel_phase = (phase_lower * i + phase_cal[i]) % 360.0
            phaser.elements.get(i + 1).rx_phase = channel_phase
      # last 4 elements will be used for upper beam
      for i in range(4, 8): 
            channel_phase = (phase_upper * i + phase_cal[i]) % 360.0
            phaser.elements.get(i + 1).rx_phase = channel_phase
      phaser.latch_rx_settings() # apply settings

      data = phaser.sdr.rx() # receive a batch of samples
      sum_beam = data[0] + data[1]
      delta_beam = data[0] - data[1]
      error = np.mean(np.real(delta_beam / sum_beam))
      error_log.append(error)
      print(error)

      # Update our estimated angle of arrival based on error
      current_phase += -10 * error # was manually tweaked until it seemed to track at a nice speed
      steer_angle = np.degrees(np.arcsin(max(min(1, (3e8 * np.radians(current_phase)) / (2 * np.pi * signal_freq * phaser.element_spacing)), -1)))
      phase_log.append(steer_angle) # looks nicer to plot steer angle instead of straight phase
      
      time.sleep(0.01)

   fig, [ax0, ax1] = plt.subplots(2, 1, figsize=(8, 10))

   ax0.plot(phase_log)
   ax0.plot([0,len(phase_log)], [0,0], 'r--')
   ax0.set_xlabel("Time")
   ax0.set_ylabel("Phase Estimate [degrees]")

   ax1.plot(error_log)
   ax1.plot([0,len(error_log)], [0,0], 'r--')
   ax1.set_xlabel("Time")
   ax1.set_ylabel("Error")

   plt.show()

.. image:: ../_images/monopulse_tracking.svg
   :align: center 
   :target: ../_images/monopulse_tracking.svg
   :alt: Monopulse-trackingdemo met een Phaser en een bewegende HB100 ervoor

Je ziet dat de error in essentie de afgeleide van de faseschatting is; omdat tracking hier werkt, benadert de faseschatting de werkelijke aankomsthoek. Alleen op basis van deze plots is dat niet altijd direct zichtbaar, maar bij een plotselinge beweging heeft het systeem een kleine fractie van een seconde nodig om bij te sturen. Het doel is dat de verandering in aankomsthoek nooit zo snel gaat dat het signaal buiten de hoofdlobben van de twee bundels terechtkomt.

Het proces is veel makkelijker te visualiseren met een 1D-array, maar praktische toepassingen van monopulse-tracking zijn vrijwel altijd 2D (met een 2D/planaire array in plaats van een lineaire array zoals de Phaser). In het 2D-geval maak je vier bundels in plaats van twee, en na verwerking houd je een enkele sombundel en vier delta-bundels over voor sturing in beide dimensies.

************************
Radar with Phaser
************************

Komt binnenkort!

************************
Conclusie
************************

Alle code die is gebruikt om de figuren in dit hoofdstuk te genereren is beschikbaar op de GitHub-pagina van het leerboek.

