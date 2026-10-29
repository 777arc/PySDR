.. _phaser-chapter:

####################
Prise en main du Phaser
####################
   
Dans ce chapitrte, nous utilisons `le Phaser d'Analog Devices <https://wiki.analog.com/resources/eval/user-guides/circuits-from-the-lab/cn0566>`_, (également appelé CN0566 ou ADALM-PHASER), un système SDR à 8 canaux et à coût réduit doté d'une antenne à balayage de phase (phased array). Il associe un PlutoSDR, un Raspberry Pi et des circuits de formation de faisceau ADAR1000, le tout conçu pour fonctionner autour de 10,25 GHz. Nous aborderons les étapes de configuration et d'étalonnage, puis nous examinerons quelques exemples de formation de faisceau en Python. Pour ceux qui ne disposent pas de Phaser, nous avons inclus des captures d'écran et des animations illustrant ce que l'utilisateur verrait.

.. image:: ../_images/phaser_on_tripod.png
   :scale: 60 % 
   :align: center
   :alt: Le Phaser (CN0566) d'Analog Devices

************************
Aperçu du matériel
************************

.. image:: ../_images/phaser_front_and_back.png
   :scale: 40 % 
   :align: center
   :alt: Les faces avant et arrière du Phaser

Le Phaser est une carte électronique consituté d'une carte regroupant le réseau d'antennes à commande de phase et divers autres composants : un Raspberry Pi 4 connecté d'un côté et un Adalm-Pluto est monté de l'autre. Le schéma fonctionnel simplifié est présenté ci-dessous. Quelques point à noter :

1. Bien qu'il ressemble à un réseau bidimensionnel de 32 éléments, il s'agit en réalité d'un réseau unidimensionnel de 8 éléments.
2. Les deux canaux de réception du Pluto sont utilisés (le second canal utilise un connecteur u.FL).
3. L'oscillateur local (LO) intégré est utilisé pour abaisser la fréquence du signal reçu d'environ 10,25 GHz à environ 2 GHz, afin que le Pluto puisse le recevoir.
4. Chaque puce ADAR1000 dispose de quatre déphaseurs à gain réglable, et les quatre canaux sont additionnés avant d'être transmis au Pluto.
5. Le Phaser se compose essentiellement de deux « sous-réseaux », chacun comprenant quatre canaux.
6. Les signaux GPIO et série provenant du Raspberry Pi, utilisés pour commander divers composants du Phaser, ne sont pas représentés ci-dessous.

.. image:: ../_images/phaser_components.png
   :scale: 40 % 
   :align: center
   :alt: Les composants du Phaser (CN0566) notamment ADF4159, LTC5548, ADAR1000

Pour l'instant, mettons de côté la partie émission du Phaser, car dans ce chapitre, nous utiliserons uniquement le module HB100 comme émetteur de test. L'ADF4159 est un synthétiseur de fréquence générant un signal pouvant atteindre 13 GHz, appelé oscillateur local (ou LO). Ce signal LO est injecté dans un mélangeur, le LTC5548, capable d'effectuer une transposition de fréquence vers le haut ou vers le bas (upconversion ou downconversion) ; nous l'utiliserons ici pour une transposition vers le bas. Pour cette opération, le composant reçoit le signal LO ainsi qu'un signal compris entre 2 et 14 GHz, puis multiplie les deux, réalisant ainsi un décalage de fréquence. Le signal résultant peut se situer dans une plage allant du courant continu (DC) jusqu'à 6 GHz, bien que nous viserons une fréquence d'environ 2 GHz. L'ADAR1000 est un formateur de faisceau analogique à 4 canaux ; le Phaser en utilise donc deux. Un formateur de faisceau analogique dispose de déphaseurs et de réglages de gain indépendants pour chaque canal, permettant d'appliquer un retard temporel et une atténuation à chaque canal avant leur sommation dans le domaine analogique (pour aboutir à un canal unique). Sur le Phaser, chaque ADAR1000 délivre un signal qui est transposé vers le bas puis reçu par le Pluto. Grâce au Raspberry Pi, nous pouvons contrôler en temps réel la phase et le gain des huit canaux afin de réaliser la formation de faisceau. Nous avons également la possibilité d'effectuer une formation de faisceau numérique ou un traitement d'antenne sur deux canaux, sujets abordés dans le chapitre suivant.

Pour les personnes intéressées, un schéma fonctionnel un peu plus détaillé est fourni ci-dessou.s

.. image:: ../_images/phaser_detailed_block_diagram.png
   :scale: 80 % 
   :align: center
   :alt: Schéma fonctionnel détaillé du Phaser (CN0566)


************************
Préparation de la carte SD
************************

Nous supposerons que vous utilisez le Raspberry Pi intégré au Phaser (directement, avec un écran, un clavier et une souris). Cela simplifie la configuration, car Analog Devices met à disposition une image de carte SD préconfigurée contenant tous les pilotes et logiciels nécessaires. Vous pouvez télécharger cette image et consulter les instructions pour la préparer `ici <https://wiki.analog.com/resources/tools-software/linux-software/kuiper-linux>`_. L'image est basée sur Raspberry Pi OS et inclut déjà tous les logiciels dont vous aurez besoin.

************************
Préparation du matériel
************************

1. Connectez le port micro-USB central du Pluto au Raspberry Pi.
2. Vous pouvez également visser délicatement le trépied dans son support.
3. Nous supposons que vous utilisez un écran HDMI, un clavier USB et une souris USB connectés au Raspberry Pi.

4. Alimentez le Raspberry Pi et la carte Phaser via le port USB-C de la Phaser (CN0566). Ne connectez pas d'alimentation au port USB-C du Raspberry Pi.

************************
Installation du logiciel
************************

Une fois que vous avez démarré le Raspberry Pi avec l'image pré-construite, en utilisant les identifiants par défaut « analog/analog », il est recommandé d'effectuer les étapes suivantes :

.. code-block:: bash

 wget https://github.com/mthoren-adi/rpi_setup_stuff/raw/main/phaser/phaser_sdcard_setup.sh
 sudo chmod +x phaser_sdcard_setup.sh
 ./phaser_sdcard_setup.sh
 sudo reboot
 
 sudo raspi-config

For more assistance setting up the Phaser, reference the `Phaser wiki quickstart page <https://wiki.analog.com/resources/eval/user-guides/circuits-from-the-lab/cn0566/quickstart>`_.

************************
Configuration du HB100
************************

.. image:: ../_images/phaser_hb100.png
   :scale: 50 % 
   :align: center
   :alt: HB100 that comes with Phaser

Le module HB100 fourni avec le Phaser est un module radar Doppler économique que nous utiliserons comme émetteur de test, car il émet un signal continu aux alentours de 10 GHz. Il fonctionne avec deux piles AA ou une alimentation de laboratoire de 3 V ; lorsqu'il est sous tension, une LED rouge reste allumée en continu.

Comme le HB100 est peu coûteux et utilise des composants RF bon marché, sa fréquence d'émission varie d'une unité à l'autre sur une plage de plusieurs centaines de mégahertz, ce qui dépasse la largeur de bande maximale que nous pouvons recevoir avec le Pluto (56 MHz). Par conséquent, pour nous assurer de régler correctement le Pluto et le convertisseur de fréquence afin de toujours capter le signal du HB100, nous devons déterminer la fréquence d'émission exacte de ce dernier. Pour ce faire, on utilise une application d'exemple fournie par Analog Devices, qui effectue un balayage de fréquence et calcule des FFT tout en recherchant un pic de signal. Assurez-vous que le HB100 est sous tension et placé à proximité immédiate du Phaser, puis lancez l'utilitaire avec la commande suivante :

.. code-block:: bash

 cd ~/pyadi-iio/examples/phaser
 python phaser_find_hb100.py

Cela devrait créer un fichier nommé :code:`hb100_freq_val.pkl` dans le même répertoire. Ce fichier contient la fréquence d'émission du HB100 en Hz (au format « pickle », et donc non lisible en texte brut), que nous utiliserons à l'étape suivante.

************************
Etalonnage
************************

Enfin, nous devons calibrer le réseau à commande de phase. Pour cela, il faut placer le module HB100 dans l'axe de visée du réseau (à 0 degré). La face du HB100 portant le code-barres étant celle qui émet le signal, elle doit être maintenue à quelques pieds du Phaser, bien centrée et directement en face de celui-ci, puis orientée précisément vers lui. Vous pourrez tester différents angles et orientations à l'étape suivante, mais pour l'instant, lançons l'utilitaire de calibration :

.. code-block:: bash

 python phaser_examples.py cal

Cela générera deux fichiers pickle supplémentaires dans le même répertoire : phase_cal_val.pkl et gain_cal_val.pkl. Chacun contient un tableau de 8 valeurs correspondant aux ajustements de phase et de gain nécessaires pour calibrer chaque canal. Ces valeurs sont propres à chaque Phaser, car elles peuvent varier lors de la fabrication. Les exécutions ultérieures de cet utilitaire produiront des valeurs légèrement différentes, ce qui est normal.

************************
Exemple d'application pré-conçue
************************

Maintenant que nous avons calibré notre Phaser et déterminé la fréquence du HB100, nous pouvons exécuter l'application d'exemple fournie par Analog Devices.
.. code-block:: bash

 python phaser_gui.py

Si vous cochez la case « Auto Refresh Data » (Actualisation automatique des données) en bas à gauche, le processus devrait démarrer. Vous devriez voir un affichage similaire à ce qui suit lorsque vous placez le HB100 dans l'axe de visée du Phaser.

.. image:: ../_images/phaser_gui.png
   :scale: 50 % 
   :align: center
   :alt: Exemple d'outil d'interface graphique du Phaser d'Analog Devices

************************
Phaser en Python
************************

Nous allons maintenant passer à la partie pratique en Python. Pour ceux qui ne disposent pas de Phaser, des captures d'écran et des animations sont fournies..

Initialisation du Phaser et de la Pluto
##############################

Le code Python suivant configure Phaser et Pluto. À ce stade, vous devriez déjà avoir exécuté les étapes d'étalonnage, qui génèrent trois fichiers pickle. Assurez-vous d'exécuter le script Python ci-dessous depuis le même répertoire que ces fichiers pickle.

De nombreux paramètres sont à configurer ; vous pouvez donc ne pas assimiler l'intégralité du code ci-dessous. Notez simplement que nous utilisons une fréquence d'échantillonnage de 30 MHz, un gain manuel très faible, que tous les gains des éléments sont identiques et que le réseau est orienté vers l'axe optique (0°).

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
 
 # Met la PLL du Phaser (the ADF4159 onboard) pour downconvert le HB100 à 2.2 GHz plus un petit offset
 offset = 1000000 # add a small arbitrary offset just so we're not right at 0 Hz where there's a DC spike
 phaser.lo = int(signal_freq + sdr.rx_lo - offset)


Réception d'échantillons de la Pluto
################################

À ce stade, le Phaser et le Pluto sont configurés et prêts à fonctionner. Nous pouvons désormais commencer à recevoir des données du Pluto. Récupérons un lot unique de 1024 échantillons, puis effectuons la FFT de chacun des deux canaux.

.. code-block:: python

 # Récupération des échantillons (selon la valeur définie pour rx_buffer_size) ; ne pas oublier que la réception s'effectue simultanément sur deux canaux.
 data = sdr.rx()
 
 # Take FFT
 PSD0 = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(data[0])))**2)
 PSD1 = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(data[1])))**2)
 f = np.linspace(-sample_rate/2, sample_rate/2, len(data[0]))
 
 # Le graphique dans le domaine temporel permet de vérifier que le HB100 est correctement détecté et qu’il n’y a pas de saturation (c’est-à-dire que le gain n’est pas trop élevé).
 plt.subplot(2, 1, 1)
 plt.plot(data[0].real) # Only plot real part
 plt.plot(data[1].real)
 plt.xlabel("Data Point")
 plt.ylabel("ADC output")
 
 # La Densité Spectrale de Puissance montre où est le  HB100 et vérifie que les deux canaux fonctionnent
 plt.subplot(2, 1, 2)
 plt.plot(f/1e6, PSD0)
 plt.plot(f/1e6, PSD1)
 plt.xlabel("Frequency [MHz]")
 plt.ylabel("Signal Strength [dB]")
 plt.tight_layout()
 plt.show()

Ce que vous voyez à ce stade dépendra de si votre HB100 est allumé et de la direction vers laquelle il est orienté. Si vous le tenez à une certaine distance (environ un mètre) du Phaser et que vous le dirigez vers son centre, vous devriez observer quelque chose comme ceci :

.. image:: ../_images/phaser_rx_psd.png
   :scale: 100 % 
   :align: center
   :alt: Example initial du phaser

Observez le pic près de 0 Hz ; le second pic, plus petit, n’est qu’un artefact qui peut être ignoré, car il se situe environ 40 dB en dessous. Le graphique du haut, qui représente le domaine temporel, affiche la partie réelle des deux canaux ; l’amplitude relative entre les deux variera donc légèrement selon la position dans laquelle vous tenez le HB100.

Mise en œuvre de la formation de faisceau (beamforming)
##############################
Passons maintenant au balayage de la phase ! Dans le code suivant, nous faisons varier la phase de -180 à +180 degrés, par pas de 2 degrés. Il est important de noter qu'il ne s'agit pas de l'angle vers lequel le faisceau est orienté, mais de la différence de phase entre des canaux adjacents. Nous devons calculer l'angle d'arrivée correspondant à chaque incrément de phase, en nous appuyant sur la vitesse de la lumière, la fréquence RF du signal reçu et l'espacement entre les éléments du dispositif (Phaser). La différence de phase entre des éléments adjacents est donnée par :

.. math::

 \phi = \frac{2 \pi d}{\lambda} \sin(\theta_{AOA})

où :math:`\theta_{AOA}`est l'angle d'arrivée du signal par rapport à l'axe de visée, :math:`d` est l'espacement entre les antennes en mètres, et :math:`\lambda` est la longueur d'onde du signal. En utilisant la formule de la longueur d'onde et en isolant :math:`\theta_{AOA}`, on obtient :
.. math::

 \theta_{AOA} = \sin^{-1}\left(\frac{c \phi}{2 \pi f d}\right)

Vous constaterez cela lorsque nous calculerons :code:`steer_angle` ci-dessous :

.. code-block:: python

 powers = [] # main DOA result
 angle_of_arrivals = []
 for phase in np.arange(-180, 180, 2): # balayage angulaire
     print(phase)
     # définit le déphasage entre les canaux adjacents des appareils
     for i in range(8):
         channel_phase = (phase * i + phase_cal[i]) % 360.0 # Analog Devices a forcé ceci à être un multiple de phase_step_size (2.8125 or 360/2**6bits) mais cela ne semble pas nécessaire
         phaser.elements.get(i + 1).rx_phase = channel_phase
     phaser.latch_rx_settings() # appliquer les paramètres
 
     steer_angle = np.degrees(np.arcsin(max(min(1, (3e8 * np.radians(phase)) / (2 * np.pi * signal_freq * phaser.element_spacing)), -1))) # l'argument d'arcsin doit être compris entre  -1 et 1, ou numpy enverra un warning
     # Si vous vous intéressez à l'aspect « tableau » de Phaser (32 cases), alors ajoutez un « *-1 » à steer_angle.
     angle_of_arrivals.append(steer_angle) 
     data = phaser.sdr.rx() # reception d'un lot d’échantillons
     data_sum = data[0] + data[1] # additionner les deux sous-tableaux (au sein de chaque sous-tableau, les 4 canaux ont déjà été additionnés)
     power_dB = 10*np.log10(np.sum(np.abs(data_sum)**2))
     powers.append(power_dB)
     # Au lieu de simplement considérer la puissance du signal, on pourrait également effectuer une FFT, puis récupérer la valeur du « bin » (case fréquentielle) correspondant au maximum ; cela permet, en pratique, d'éliminer le bruit. Dans les tests, les résultats obtenus ont été quasiment identiques.
     #PSD = 10*np.log10(np.abs(np.fft.fft(data_sum * np.blackman(len(data_sum))))**2) # en dB
 
 powers -= np.max(powers) # normalize so max is at 0 dB
 
 plt.plot(angle_of_arrivals, powers, '.-')
 plt.xlabel("Angle of Arrival")
 plt.ylabel("Magnitude [dB]")
 plt.show()

Pour chaque valeur de :code:`phase` (rappelons qu'il s'agit du déphasage entre des éléments adjacents), nous configurons les déphaseurs, après avoir ajouté les valeurs de calibration de phase et ramené les angles dans l'intervalle [0, 360] degrés. Nous acquérons ensuite un ensemble d'échantillons via :code:`rx()`, sommons les deux canaux, puis calculons la puissance du signal. Enfin, nous traçons la puissance en fonction de l'angle d'arrivée. Le résultat devrait ressembler à ceci :

.. image:: ../_images/phaser_sweep.png
   :scale: 100 % 
   :align: center
   :alt: Balayage unique du phaser

Dans cet exemple, le HB100 a été maintenu légèrement décalé par rapport à l'axe de visée.

Si vous souhaitez obtenir un diagramme polaire, vous pouvez utiliser ce qui suit :

.. code-block:: python

 # Diagramme polaire
 fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
 ax.plot(np.deg2rad(angle_of_arrivals), powers) # axe des x en radians
 ax.set_rticks([-40, -30, -20, -10, 0])  # moins de graduations radiales
 ax.set_thetamin(np.min(angle_of_arrivals)) # en degrés
 ax.set_thetamax(np.max(angle_of_arrivals))
 ax.set_theta_direction(-1) # augmenter dans le sens des aiguilles d'une montre
 ax.set_theta_zero_location('N') # orienter 0 degré vers le haut
 ax.grid(True)
 plt.show()

.. image:: ../_images/phaser_sweep_polar.png
   :scale: 100 % 
   :align: center
   :alt: Balayage unique de type « phaser » à l’aide d’un diagramme polaire

En retenant la valeur maximale, on peut estimer la direction d’arrivée du signal !


Temps réel avec atténuation spatiale
#####################################

Intéressons-nous maintenant à la pondération spatiale (spatial tapering). Jusqu'ici, nous avons maintenu les gains de chaque canal à des valeurs identiques, de sorte que les huit canaux soient sommés à parts égales. Tout comme nous avons appliqué une fenêtre avant d'effectuer une transformée de Fourier rapide (FFT), nous pouvons appliquer une fenêtre dans le domaine spatial en affectant des coefficients de pondération à ces huit canaux. Nous utiliserons les mêmes fonctions de fenêtrage — Hanning, Hamming, etc. Modifions également le code pour qu'il s'exécute en temps réel ; ce sera bien plus ludique !

.. code-block:: python

 plt.ion() # nécessaire pour la vue en temps-réel
 print("Démarrage, utilisez Control-c pour arrêter")
 try:
     while True:
         powers = [] # résultat principal de DOA
         angle_of_arrivals = []
         for phase in np.arange(-180, 180, 6): # balayage angulaire
             # définir le déphasage entre les canaux adjacents des appareils
             for i in range(8):
                 channel_phase = (phase * i + phase_cal[i]) % 360.0 # Analog Devices a forcé ceci à être un multiple de phase_step_size (2.8125 or 360/2**6bits) mais cela ne semble pas nécessaire
                 phaser.elements.get(i + 1).rx_phase = channel_phase
            
             #  régler les gains, y compris le « gain_cal », qui peut servir à appliquer une pondération (taper). Testez-les tous !
             gain_list = [127] * 8 # rectangular window          [127, 127, 127, 127, 127, 127, 127, 127]
             #gain_list = np.rint(np.hamming(8) * 127)         # [ 10,  32,  82, 121, 121,  82,  32,  10]
             #gain_list = np.rint(np.hanning(10)[1:-1] * 127)  # [ 15,  52,  95, 123, 123,  95,  52,  15]
             #gain_list = np.rint(np.blackman(10)[1:-1] * 127) # [  6,  33,  80, 121, 121,  80,  33,   6]
             #gain_list = np.rint(np.bartlett(10)[1:-1] * 127) # [ 28,  56,  85, 113, 113,  85,  56,  28]
             for i in range(8):
                 channel_gain = int(gain_list[i] * gain_cal[i])
                 phaser.elements.get(i + 1).rx_gain = channel_gain
 
             phaser.latch_rx_settings() # application des paramètres
 
             steer_angle = np.degrees(np.arcsin(max(min(1, (3e8 * np.radians(phase)) / (2 * np.pi * signal_freq * phaser.element_spacing)), -1))) # l'argument d'arcsin doit être compris entre -1 et 1, ou numpy enverra un warning
             angle_of_arrivals.append(steer_angle) 
             data = phaser.sdr.rx() # réception d'un lot d'échantillons
             data_sum = data[0] + data[1] # sum the two subarrays (within each subarray the 4 channels have already been summed)
             power_dB = 10*np.log10(np.sum(np.abs(data_sum)**2))
             powers.append(power_dB)
 
         powers -= np.max(powers) # normalize so max is at 0 dB
 
         # Vue en temps-réel
         plt.plot(angle_of_arrivals, powers, '.-')
         plt.xlabel("Angle d'arrivée")
         plt.ylabel("Amplitude [dB]")
         plt.draw()
         plt.pause(0.001)
         plt.clf()
 
 except KeyboardInterrupt:
     sys.exit() # quitter python

Vous devriez voir une version en temps réel de l'exercice précédent. Essayez de modifier la liste `gain_list` utilisée, afin d'explorer les différentes fenêtres. Voici un exemple avec la fenêtre rectangulaire (c'est-à-dire sans fonction de fenêtrage) :     

.. image:: ../_images/phaser_animation_rect.gif
   :scale: 100 % 
   :align: center
   :alt: Animation de formation de faisceau (beamforming) utilisant le déphaseur et une fenêtre rectangulaire

et voici un exemple de fenêtre de Hamming :

.. image:: ../_images/phaser_animation_hamming.gif
   :scale: 100 % 
   :align: center
   :alt: Animation de formation de faisceau (beamforming) utilisant le déphaseur et une fenêtre de Hamming

Notez l'absence de lobes secondaires avec la fenêtre de Hamming. En réalité, toutes les fenêtres, à l'exception de la fenêtre rectangulaire, réduisent considérablement les lobes secondaires ; en contrepartie, le lobe principal devient un peu plus large.

************************
Poursuite monopulse
************************

Jusqu’à présent, nous avons effectué des balayages ponctuels afin de déterminer l’angle d’arrivée d’un émetteur de test (le HB100). Supposons maintenant que nous souhaitions recevoir en continu un signal — qu’il s’agisse d’un signal de communication ou d’un signal radar — susceptible de se déplacer, entraînant ainsi une variation de son angle d’arrivée au cours du temps. Ce processus est appelé « poursuite » (ou *tracking*) ; il suppose que nous disposons déjà d’une estimation approximative de l’angle d’arrivée (c’est-à-dire qu’un balayage initial a permis d’identifier un signal d’intérêt). Nous utiliserons la technique de poursuite monopulse pour ajuster de manière adaptative les pondérations, de sorte que le lobe principal reste orienté vers le signal au fil du temps. Il convient toutefois de noter qu’il existe d’autres méthodes de poursuite en dehors de la technique monopulse.

Inventé en 1943 par Robert Page au Naval Research Laboratory (NRL), le concept fondamental de la poursuite monopulse repose sur l’utilisation de deux faisceaux, chacun étant légèrement décalé par rapport à l’angle d’arrivée réel (ou, plus précisément, par rapport à notre estimation de celui-ci) et situés de part et d’autre de cet angle, comme illustré dans le schéma ci-dessous.

.. image:: ../_images/monopulse.svg
   :align: center 
   :target: ../_images/monopulse.svg
   :alt: Diagramme de faisceaux monopulse illustrant deux faisceaux ainsi que le faisceau somme

Nous calculons ensuite, par voie numérique, la somme et la différence (aussi appelée « delta ») de ces deux faisceaux ; cela nécessite l'utilisation de deux canaux numériques du Phaser, ce qui en fait une approche hybride (bien qu'il soit tout à fait possible d'effectuer ces opérations de somme et de différence en mode analogique, à l'aide de matériel dédié). Le faisceau « somme » correspond à un faisceau centré sur l'angle d'arrivée estimé (comme illustré ci-dessus) ; il peut donc être utilisé pour la démodulation ou le décodage du signal d'intérêt. Le faisceau « delta » — pour reprendre ce terme — est plus difficile à visualiser, mais il présente un creux (ou « zéro ») au niveau de l'angle d'arrivée estimé. Le rapport entre le faisceau somme et le faisceau delta (que l'on appelle « erreur ») peut être exploité pour assurer le suivi du signal. Ce processus est illustré au mieux par un court extrait de code Python ; rappelons que la fonction :code:`rx()` renvoie un ensemble d'échantillons issus des deux canaux. Ainsi, dans le code ci-dessous, :code:`data[0]` correspond au premier canal du Pluto (le premier groupe de quatre éléments du Phaser) et :code:`data[1]` au second canal (le second groupe de quatre éléments). Pour générer deux faisceaux, nous orientons chacun de ces deux groupes séparément. Nous pouvons alors calculer la somme, le delta et l'erreur comme suit :

.. code-block:: python

   data = phaser.sdr.rx()
   sum_beam = data[0] + data[1]
   delta_beam = data[0] - data[1]
   error = np.mean(np.real(delta_beam / sum_beam))

Le signe de l'erreur indique la direction réelle de provenance du signal, tandis que son amplitude précise l'écart par rapport à la position du signal. Ces informations permettent ensuite de mettre à jour l'estimation de l'angle d'arrivée ainsi que les pondérations. En répétant ce processus en temps réel, il est possible de suivre le signal.

Pour mieux comprendre ce mécanisme, rappelons qu'un déphasage de 180 degrés équivaut à une multiplication par -1 ; ainsi, le faisceau « delta » (différence) correspond, pour l'essentiel, à la somme du premier faisceau et du second faisceau déphasé de 180 degrés. Si le signal se situe principalement dans le second faisceau, il présentera un déphasage de 180 degrés par rapport au signal reçu par le faisceau « somme ». Par ailleurs, rappelons que la division de deux nombres complexes revient à calculer le rapport de leurs amplitudes et la différence de leurs phases. Par conséquent, si le signal est principalement présent dans le second faisceau, l'erreur sera négative, et son amplitude sera proportionnelle à la part du signal contenue dans le second faisceau par rapport au premier.

Passons maintenant à l'exemple complet en Python : nous commencerons par reprendre le code utilisé précédemment pour effectuer un balayage sur 180 degrés. Le seul ajout consistera à extraire la phase correspondant à la puissance reçue maximale :

.. code-block:: python

   # Effectuer un balayage pour obtenir une estimation initiale de l'AOA (Angle Of Arrival = Angle d'arrivée), en utilisant le code ci-dessus.
   # ...
   current_phase = phase_angles[np.argmax(powers)]
   print("max_phase:", current_phase)

Ensuite, nous allons créer deux faisceaux ; nous commencerons par tester des valeurs situées 5 degrés en dessous et 5 degrés au-dessus de l'estimation actuelle. Notez toutefois qu'il s'agit ici d'unités de phase — nous n'avons pas effectué la conversion en angle de pointage, bien que les deux grandeurs soient proches. Le code suivant se compose essentiellement de deux instances du code utilisé précédemment pour configurer les déphaseurs de chaque canal ; nous utilisons ici les 4 premiers éléments pour le faisceau inférieur et les 4 derniers éléments pour le faisceau supérieur :

.. code-block:: python

   # Nous créons maintenant les deux faisceaux de part et d’autre de notre estimation actuelle
   phase_offset = np.radians(5) # TRY TWEAKING THIS - specify offset from center in degrees
   phase_lower = current_phase - phase_offset
   phase_upper = current_phase + phase_offset
   # Les 4 premiers éléments serviront pour le faisceau inférieur
   for i in range(0, 4): 
      channel_phase = (phase_lower * i + phase_cal[i]) % 360.0
      phaser.elements.get(i + 1).rx_phase = channel_phase
   # les 4 derniers éléments serviront pour le faisceau supérieur
   for i in range(4, 8): 
      channel_phase = (phase_upper * i + phase_cal[i]) % 360.0
      phaser.elements.get(i + 1).rx_phase = channel_phase
   phaser.latch_rx_settings() # application des paramètres

Avant de procéder au suivi proprement dit, testons ce qui précède en maintenant les pondérations du faisceau constantes et en déplaçant le HB100 vers la gauche et vers la droite (une fois son initialisation terminée et l'angle de départ déterminé) :

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
   :alt: Affichage de la fonction d'erreur pour la poursuite monopulse, sans pour autant mettre à jour les pondérations.

Dans cet exemple, je déplace le module HB100. Je commence par le maintenir dans une position fixe pendant un balayage de 180 degrés ; une fois celui-ci terminé, je le déplace légèrement vers la droite en le faisant bouger un peu, puis je le déplace vers la gauche de sa position initiale en le faisant également bouger. Ensuite, vers t = 400 sur le graphique, je le ramène de l'autre côté et l'y maintiens un instant, avant de le faire bouger à nouveau. Ce qu'il faut retenir, c'est que plus le HB100 s'éloigne de l'angle de départ, plus l'erreur est importante ; le signe de cette erreur nous indique de quel côté du point de départ se situe le HB100.

Utilisons maintenant la valeur de l'erreur pour mettre à jour les poids. Nous allons supprimer la boucle :code:`for` précédente et en créer une nouvelle qui englobe l'ensemble du processus. Pour plus de clarté, voici l'exemple de code complet, à l'exception de la partie initiale correspondant au balayage de 180 degrés :


.. code-block:: python

   #  Effectuer un balayage pour obtenir une estimation initiale de l'AOA
   # ...
   current_phase = phase_angles[np.argmax(powers)]
   print("phase_max:", current_phase)

   # À présent, nous allons mettre à jour la phase actuelle (current_phase) en fonction de l'erreur
   print("START MOVING THE HB100 A LITTLE LEFT AND RIGHT")
   phase_log = []
   error_log = []
   for ii in range(500):
      # Créer les deux faisceaux de part et d'autre de l'estimation actuelle, en utilisant le décalage spécifié
      phase_offset = np.radians(5)
      phase_lower = current_phase - phase_offset
      phase_upper = current_phase + phase_offset

      # Les 4 premiers éléments serviront pour le faisceau inférieur
      for i in range(0, 4): 
            channel_phase = (phase_lower * i + phase_cal[i]) % 360.0
            phaser.elements.get(i + 1).rx_phase = channel_phase
      # Les 4 derniers éléments serviront pour le faisceau inférieur
      for i in range(4, 8): 
            channel_phase = (phase_upper * i + phase_cal[i]) % 360.0
            phaser.elements.get(i + 1).rx_phase = channel_phase
      phaser.latch_rx_settings() # application des paramètres

      data = phaser.sdr.rx() # réception d'un lot d'échantillons
      sum_beam = data[0] + data[1]
      delta_beam = data[0] - data[1]
      error = np.mean(np.real(delta_beam / sum_beam))
      error_log.append(error)
      print(error)

      # Mise à jour de l’angle d’arrivée estimé en fonction de l’erreur.
      current_phase += -10 * error # ajusté manuellement jusqu’à obtenir une vitesse de poursuite satisfaisante 
      steer_angle = np.degrees(np.arcsin(max(min(1, (3e8 * np.radians(current_phase)) / (2 * np.pi * signal_freq * phaser.element_spacing)), -1)))
      phase_log.append(steer_angle) #  Il est préférable de représenter graphiquement l'angle de pointage plutôt que la phase brute.
      
      time.sleep(0.01)

   fig, [ax0, ax1] = plt.subplots(2, 1, figsize=(8, 10))

   ax0.plot(phase_log)
   ax0.plot([0,len(phase_log)], [0,0], 'r--')
   ax0.set_xlabel("Temps")
   ax0.set_ylabel("Estimation de la Phase [degrés]")

   ax1.plot(error_log)
   ax1.plot([0,len(error_log)], [0,0], 'r--')
   ax1.set_xlabel("Temps")
   ax1.set_ylabel("Erreur")

   plt.show()

.. image:: ../_images/monopulse_tracking.svg
   :align: center 
   :target: ../_images/monopulse_tracking.svg
   :alt: Démonstration de poursuite monopulse utilisant un module Phaser, devant lequel on agite un composant HB100.

On peut constater que l'erreur correspond, pour l'essentiel, à la dérivée de l'estimation de phase ; comme le suivi s'opère correctement, cette estimation de phase équivaut, peu ou prou, à l'angle d'arrivée réel. Bien que cela ne soit pas évident à la simple lecture de ces graphiques, tout mouvement soudain nécessite une brève fraction de seconde pour que le système s'ajuste et se recalibre. L'objectif est de garantir que la variation de l'angle d'arrivée ne soit jamais si rapide que le signal se retrouve en dehors des lobes principaux des deux faisceaux.

Il est bien plus aisé de visualiser ce processus dans le cas d'une antenne unidimensionnelle (1D) ; toutefois, les applications concrètes du suivi monopulse relèvent presque systématiquement de la 2D (en utilisant une antenne plane plutôt qu'une antenne linéaire, telle que le système « Phaser »). Dans le cas bidimensionnel (2D), quatre faisceaux sont générés au lieu de deux ; le traitement aboutit alors à un faisceau « somme » unique et à quatre faisceaux « différence » (ou « delta »), permettant un pilotage selon les deux dimensions.


************************
Radar avec le Phaseur
************************

À venir !

************************
Conclusion
************************

L’ensemble du code utilisé pour générer les figures de ce chapitre est disponible sur la page GitHub du manuel.
