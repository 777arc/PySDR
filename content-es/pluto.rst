.. _pluto-chapter:

####################################
PlutoSDR en Python
####################################

.. image:: ../_images/pluto.png
   :scale: 50 % 
   :align: center
   :alt: The PlutoSDR by Analog Devices
   
En este capítulo aprenderemos cómo usar la API de Python para `PlutoSDR <https://www.analog.com/en/design-center/evaluation-hardware-and-software/evaluation-boards-kits/adalm-pluto.html>`_, que es un SDR de bajo costo de Analog Devices. Cubriremos los pasos de instalación de PlutoSDR para ejecutar los controladores/software y luego analizaremos la transmisión y recepción con PlutoSDR en Python.

*******************************
Instalación de Software/Drivers
*******************************

Configurando la máquina virtual
###############################

Si bien el código Python proporcionado en este libro debería funcionar en Windows, Mac y Linux, las instrucciones de instalación a continuación son específicas para Ubuntu 22. Si tiene problemas para instalar el software en su sistema operativo, siga `las instrucciones proporcionadas por Analog Devices <https://wiki.analog.com/university/tools/pluto/users/quick_start>`_, Recomiendo instalar una máquina virtual Ubuntu 22 y probar las instrucciones a continuación. Alternativamente, si está en Windows 11, el Subsistema de Windows para Linux (WSL) que usa Ubuntu 22 tiende a funcionar bastante bien y admite gráficos listos para usar.

1. Instalar y abrir `VirtualBox <https://www.virtualbox.org/wiki/Downloads>`_.
2. Cree una nueva máquina virtual. Para el tamaño de la memoria, recomiendo utilizar el 50% de la RAM de su computadora.
3. Cree el disco duro virtual, elija VDI y asigne tamaño dinámicamente. 15 GB deberían ser suficientes. Si quieres estar realmente seguro, puedes usar más.
4. Descarga Ubuntu 22 Desktop .iso- https://ubuntu.com/download/desktop
5. Inicie la máquina virtual. Le pedirá medios de instalación. Elija el archivo .iso del escritorio de Ubuntu 22. Elija "instalar ubuntu", use las opciones predeterminadas y una ventana emergente le advertirá sobre los cambios que está a punto de realizar. Pulsa continuar. Elija nombre/contraseña y luego espere a que la VM termine de inicializarse. Después de finalizar, la VM se reiniciará, pero debe apagarla después del reinicio.
6. Vaya a la configuración de VM (el ícono de ajustes).
7. En sistema > procesador > elija al menos 3 CPU. Si tiene una tarjeta de video real, en pantalla > memoria de video > elija algo mucho más alto.
8. Inicie su máquina virtual.
9. Recomiendo instalar adiciones de invitados de VM. Dentro de la máquina virtual, vaya a Dispositivos > Insertar CD de Guest Additions > presione ejecutar cuando aparezca un cuadro. Sigue las instrucciones. Reinicie la máquina virtual. El portapapeles compartido se puede habilitar a través de Dispositivos > Portapapeles compartido > Bidireccional.

Conectando el PlutoSDR
######################

1. Si ejecuta OSX, dentro de OSX, no en la VM, en las preferencias del sistema, habilite las "extensiones del kernel". Luego instale HoRNDIS (es posible que deba reiniciar después).
2. Si ejecuta Windows, instale este controlador: https://github.com/analogdevicesinc/plutosdr-m2k-drivers-win/releases/download/v0.7/PlutoSDR-M2k-USB-Drivers.exe
3. Si ejecuta Linux, no debería tener que hacer nada especial.
4. Conecte el Pluto a la máquina host a través de USB. Asegúrate de usar el puerto USB del medio en el Pluto porque el otro es solo para alimentación. Al conectar Plutón se debería crear un adaptador de red virtual, es decir, Plutón aparece como un adaptador Ethernet USB.
5. En la máquina host (no en la VM), abra una terminal o su herramienta de ping preferida y haga ping a 192.168.2.1. Si eso no funciona, detenga y depure la interfaz de red.
6. Dentro de la VM, abra una nueva terminal
7. Haga ping a 192.168.2.1. Si eso no funciona, deténgase aquí y depure. Mientras hace ping, desconecte su Pluto y asegúrese de que el ping se detenga; si continúa haciendo ping, entonces hay algo más en esa dirección IP en la red y tendrá que cambiar la IP de Pluto (u otro dispositivo) antes de continuar.
8. Anota la dirección IP del Pluto porque la necesitarás cuando empecemos a usar el Pluto en Python.

Instalación del Driver para el PlutoSDR
#######################################

Los siguientes comandos de terminal deberían compilar e instalar la última versión de:

1. **libiio**, Biblioteca “multiplataforma” de Analog Device para interfaz de hardware
2. **libad9361-iio**, AD9361 es el chip RF específico dentro del PlutoSDR
3. **pyadi-iio**, la API Python de Plutón, *este es nuestro objetivo final*, pero depende de las dos bibliotecas anteriores


.. code-block:: bash

 sudo apt-get install build-essential git libxml2-dev bison flex libcdk5-dev cmake python3-pip libusb-1.0-0-dev libavahi-client-dev libavahi-common-dev libaio-dev
 cd ~
 git clone --branch v0.23 https://github.com/analogdevicesinc/libiio.git
 cd libiio
 mkdir build
 cd build
 cmake -DPYTHON_BINDINGS=ON ..
 make -j$(nproc)
 sudo make install
 sudo ldconfig
 
 cd ~
 git clone https://github.com/analogdevicesinc/libad9361-iio.git
 cd libad9361-iio
 mkdir build
 cd build
 cmake ..
 make -j$(nproc)
 sudo make install
 
 cd ~
 git clone --branch v0.0.14 https://github.com/analogdevicesinc/pyadi-iio.git
 cd pyadi-iio
 pip3 install --upgrade pip
 pip3 install -r requirements.txt
 sudo python3 setup.py install

Probando los Driver del PlutoSDR
################################

Abra una nueva terminal (en su VM) y escriba los siguientes comandos:

.. code-block:: bash

 python3
 import adi
 sdr = adi.Pluto('ip:192.168.2.1') # or whatever your Pluto's IP is
 sdr.sample_rate = int(2.5e6)
 sdr.rx()

Si llega hasta aquí sin ningún error, continúe con los siguientes pasos.

Cambiando la dirección IP de Plutón
####################################

Si por alguna razón la IP predeterminada de 192.168.2.1 no funciona porque ya tienes una subred 192.168.2.0, o porque quieres conectar varios Pluto al mismo tiempo, puedes cambiar la IP siguiendo estos pasos:

1. Edite el archivo config.txt en el dispositivo de almacenamiento masivo PlutoSDR (es decir, la unidad USB que aparece después de conectar Pluto). Introduce la nueva IP que desees.
2. Expulse el dispositivo de almacenamiento masivo (¡no desconecte el Plutón!). En Ubuntu 22 hay un símbolo de expulsión al lado del dispositivo PlutoSDR, cuando se mira el explorador de archivos.
3. Espere unos segundos y luego apague y encienda desconectando el Pluto y volviéndolo a enchufar. Vuelva al config.txt para determinar si sus cambios se guardaron.

Tenga en cuenta que este procedimiento también se utiliza para mostrar una imagen de firmware diferente en el Pluto. Para más detalles ver https://wiki.analog.com/university/tools/pluto/users/firmware.

"Hackear" PlutoSDR para aumentar el alcance de RF
#################################################

El PlutoSDR tiene un rango de frecuencia central y una frecuencia de muestreo limitados, pero el chip subyacente es capaz de alcanzar frecuencias mucho más altas. Siga estos pasos para desbloquear todo el rango de frecuencia del chip. Tenga en cuenta que este proceso lo proporciona Analog Devices, por lo que es el riesgo más bajo posible. La limitación de frecuencia de PlutoSDR tiene que ver con que Analog Devices "agrupe" el AD9364 según estrictos requisitos de rendimiento en las frecuencias más altas. .... Como entusiastas y experimentadores de SDR, no nos preocupan demasiado dichos requisitos de rendimiento.

¡Es hora de hackear! Abra una terminal (ya sea host o VM, no importa):

.. code-block:: bash

 ssh root@192.168.2.1

La contraseña predeterminada es :code:`analog`

Deberías ver la pantalla de bienvenida de PlutoSDR. ¡Ahora ha conectado SSH a la CPU ARM en el propio Pluto!
Si tiene un Pluto con la versión de firmware 0.31 o inferior, escriba los siguientes comandos en:

.. code-block:: bash

 fw_setenv attr_name compatible
 fw_setenv attr_val ad9364
 reboot

Y para uso de 0.32 y superiores:

.. code-block:: bash
 
 fw_setenv compatible ad9364
 reboot

Ahora deberías poder sintonizar hasta 6 GHz y bajar hasta 70 MHz, ¡sin mencionar usar una frecuencia de muestreo de hasta 56 MHz! ¡Hurra!

************************
Recepción
************************

El muestreo utilizando la API Python de PlutoSDR es sencillo. Con cualquier aplicación SDR sabemos que debemos indicarle la frecuencia central, la frecuencia de muestreo y la ganancia (o si usar el control automático de ganancia). Puede haber otros detalles, pero esos tres parámetros son necesarios para que el SDR tenga suficiente información para recibir muestras. Algunos SDR tienen un comando que le indica que comience a muestrear, mientras que otros, como Plutón, comenzarán a muestrear tan pronto como lo inicialice. Una vez que el búfer interno del SDR se llena, se descartan las muestras más antiguas. Todas las API de SDR tienen algún tipo de función de "recibir muestras", y para el Pluto es rx(), que devuelve un lote de muestras. El número específico de muestras por lote está definido por el tamaño del búfer establecido de antemano.

El siguiente código supone que tiene instalada la API Python de Plutón. Este código se inicializa el Pluto, establece la frecuencia de muestreo en 1 MHz, establece la frecuencia central en 100 MHz y establece la ganancia en 70 dB con el control automático de ganancia desactivado. Tenga en cuenta que normalmente no importa el orden en el que establezca la frecuencia central, la ganancia y la frecuencia de muestreo. En el siguiente fragmento de código, le decimos al Pluto que queremos que nos dé 10.000 muestras por llamada a rx(). Imprimimos las primeras 10 muestras.

.. code-block:: python

    import numpy as np
    import adi
    
    sample_rate = 1e6 # Hz
    center_freq = 100e6 # Hz
    num_samps = 10000 # number of samples returned per call to rx()
    
    sdr = adi.Pluto()
    sdr.gain_control_mode_chan0 = 'manual'
    sdr.rx_hardwaregain_chan0 = 70.0 # dB
    sdr.rx_lo = int(center_freq)
    sdr.sample_rate = int(sample_rate)
    sdr.rx_rf_bandwidth = int(sample_rate) # filter width, just set it to the same as sample rate for now
    sdr.rx_buffer_size = num_samps
    
    samples = sdr.rx() # receive samples off Pluto
    print(samples[0:10])


Por ahora no vamos a hacer nada interesante con estos ejemplos, pero el resto de este libro de texto está lleno de código Python que funciona en ejemplos de IQ tal como lo recibimos anteriormente.


Ganancia de Recepción
#####################

El Pluto se puede configurar para que tenga una ganancia de recepción fija o automática. Un control automático de ganancia (AGC) ajustará automáticamente la ganancia de recepción para mantener un nivel de señal fuerte (-12 dBFS para cualquiera que tenga curiosidad). AGC no debe confundirse con el convertidor analógico a digital (ADC) que digitaliza la señal. Técnicamente hablando, AGC es un circuito de retroalimentación de circuito cerrado que controla la ganancia del amplificador en respuesta a la señal recibida. Su objetivo es mantener un nivel de potencia de salida constante a pesar de un nivel de potencia de entrada variable. Normalmente, el AGC ajustará la ganancia para evitar saturar el receptor (es decir, alcanzar el límite superior del rango del ADC) y al mismo tiempo permitirá que la señal "llene" tantos bits de ADC como sea posible.

El circuito integrado de radiofrecuencia, o RFIC, dentro del PlutoSDR tiene un módulo AGC con algunas configuraciones diferentes. (Un RFIC es un chip que funciona como un transceptor: transmite y recibe ondas de radio). Primero, tenga en cuenta que la ganancia de recepción en el Pluto tiene un rango de 0 a 74,5 dB. Cuando está en modo AGC "manual", el AGC se apaga y debe indicarle a Pluto qué ganancia de recepción usar, por ejemplo:

.. code-block:: python

  
  sdr.gain_control_mode_chan0 = "manual" # turn off AGC
  gain = 50.0 # allowable range is 0 to 74.5 dB
  sdr.rx_hardwaregain_chan0 = gain # set receive gain

Si desea habilitar el AGC, debe elegir uno de dos modos:

1. :code:`sdr.gain_control_mode_chan0 = "slow_attack"`
2. :code:`sdr.gain_control_mode_chan0 = "fast_attack"`

Y con AGC habilitado no proporciona un valor a :code:`rx_hardwaregain_chan0`. Se ignorará porque el propio Pluto ajusta la ganancia de la señal. El Pluto tiene dos modos para AGC: ataque rápido y ataque lento, como se muestra en el código recortado arriba. La diferencia entre los dos es intuitiva, si lo piensas bien. El modo de ataque rápido reacciona más rápido a las señales. En otras palabras, el valor de ganancia cambiará más rápido cuando la señal recibida cambie de nivel. Ajustar los niveles de potencia de la señal puede ser importante, especialmente para los sistemas dúplex por división de tiempo (TDD) que utilizan la misma frecuencia para transmitir y recibir. Configurar el control de ganancia en modo de ataque rápido para este escenario limita la atenuación de la señal. Con cualquiera de los modos, si no hay señal presente y solo ruido, el AGC maximizará la configuración de ganancia; cuando aparece una señal, saturará el receptor brevemente, hasta que el AGC pueda reaccionar y reducir la ganancia. Siempre puedes comprobar el nivel de ganancia actual en tiempo real con:

.. code-block:: python
 
 sdr._get_iio_attr('voltage0','hardwaregain', False)

Para obtener más detalles sobre el AGC del Pluto SDR, como por ejemplo cómo cambiar la configuración avanzada del AGC, consulte `the "RX Gain Control" section of this page <https://wiki.analog.com/resources/tools-software/linux-drivers/iio-transceiver/ad9361>`_.

************************
Transmitiendo
************************

Antes de transmitir cualquier señal con su Pluto, asegúrese de conectar un cable SMA entre el puerto TX de Pluto y cualquier dispositivo que actúe como receptor. Es importante comenzar siempre transmitiendo a través de un cable, especialmente mientras aprendes *cómo* transmitir, para asegurarte de que el SDR se comporta como deseas. Mantenga siempre la potencia de transmisión extremadamente baja para no sobrecargar al receptor, ya que el cable no atenúa la señal como lo hace el canal inalámbrico. Si posee un atenuador (por ejemplo, 30 dB), ahora sería un buen momento para usarlo. Si no tienes otro SDR o un analizador de espectro que actúe como receptor, en teoría puedes usar el puerto RX en el mismo Pluto, pero puede complicarse. Recomendaría adquirir un RTL-SDR de $10 para que actúe como SDR receptor.

Transmitir es muy similar a recibir, excepto que en lugar de decirle al SDR que reciba una cierta cantidad de muestras, le daremos una cierta cantidad de muestras para transmitir. En lugar de estar configurando :code:`rx_lo` lo haremos con :code:`tx_lo`, para especificar en qué frecuencia portadora transmitir. La frecuencia de muestreo se comparte entre RX y TX, por lo que la configuraremos como de costumbre. A continuación se muestra un ejemplo completo de transmisión, donde generamos una sinusoide a +100 kHz, luego transmitimos la señal compleja a una frecuencia portadora de 915 MHz, lo que hace que el receptor vea una portadora a 915,1 MHz. Realmente no hay ninguna razón práctica para hacer esto, podríamos simplemente haber configurado center_freq en 915.1e6 y transmitir una matriz de unos, pero queríamos generar muestras complejas con fines de demostración.

.. code-block:: python
    
    import numpy as np
    import adi

    sample_rate = 1e6 # Hz
    center_freq = 915e6 # Hz

    sdr = adi.Pluto("ip:192.168.2.1")
    sdr.sample_rate = int(sample_rate)
    sdr.tx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
    sdr.tx_lo = int(center_freq)
    sdr.tx_hardwaregain_chan0 = -50 # Increase to increase tx power, valid range is -90 to 0 dB

    N = 10000 # number of samples to transmit at once
    t = np.arange(N)/sample_rate
    samples = 0.5*np.exp(2.0j*np.pi*100e3*t) # Simulate a sinusoid of 100 kHz, so it should show up at 915.1 MHz at the receiver
    samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs

    # Transmit our batch of samples 100 times, so it should be 1 second worth of samples total, if USB can keep up
    for i in range(100):
        sdr.tx(samples) # transmit the batch of samples once

Aquí hay algunas notas sobre este código. Primero, desea simular sus muestras de IQ para que estén entre -1 y 1, pero luego, antes de transmitirlas, tenemos que escalarlas en 2^14 debido a cómo Analog Devices implementó la función del :code:`tx()`.  Si no está seguro de cuáles son sus valores mínimos/máximos, simplemente imprímalos con :code:`print(np.min(samples), np.max(samples))` o escriba una declaración if para asegurarse de que nunca superen 1 o sean inferiores a -1 (asumiendo que el código viene antes de la escala 2^14). En cuanto a la ganancia de transmisión, el rango es de -90 a 0 dB, por lo que 0 dB es la potencia de transmisión más alta. Siempre queremos comenzar con una potencia de transmisión baja y luego aumentar si es necesario, por lo que tenemos la ganancia configurada en -50 dB de forma predeterminada, que está hacia el extremo bajo. No lo establezca simplemente en 0 dB sólo porque su señal no aparece; Puede que haya algo más que este mal y no querrás quemar tu receptor. 

Transmisión de muestras repetidas
#################################

Si desea transmitir continuamente el mismo conjunto de muestras repetidas, en lugar de usar un bucle for/ while dentro de Python como hicimos anteriormente, puede decirle al Pluto que lo haga usando solo una línea:

.. code-block:: python

 sdr.tx_cyclic_buffer = True # Enable cyclic buffers

Luego transmitirías tus muestras como de costumbre: :code:`sdr.tx(samples)` solo una vez, y el Pluto seguirá transmitiendo la señal indefinidamente, hasta que se llame al destructor de objetos sdr. Para cambiar las muestras que se transmiten continuamente, no puede simplemente llamar :code:`sdr.tx(samples)` nuevamente con un nuevo conjunto de muestras, primero debe llamar :code:`sdr.tx_destroy_buffer()`, entonces llamar a :code:`sdr.tx(samples)`.

Transmitir por aire legalmente
#################################

Innumerables veces los estudiantes me han preguntado en qué frecuencias pueden transmitir con una antena (en los Estados Unidos). La respuesta corta es ninguna, hasta donde yo sé. Por lo general, cuando las personas señalan regulaciones específicas que hablan sobre límites de potencia de transmisión, se refieren a `las regulaciones del "Título 47, Parte 15" de la FCC (47 CFR 15) <https://www.ecfr.gov/cgi-bin/text-idx?SID=7ce538354be86061c7705af3a5e17f26&mc=true&node=pt47.1.15&rgn=div5>`_.  Pero esas son regulaciones para los fabricantes que construyen y venden dispositivos que operan en las bandas ISM, y las regulaciones discuten cómo deben probarse. Un dispositivo Parte 15 es aquel en el que una persona no necesita una licencia para operar el dispositivo en cualquier espectro que esté usando, pero el dispositivo en sí debe estar autorizado/certificado para demostrar que está operando siguiendo las regulaciones de la FCC antes de comercializarlo y venderlo. Las regulaciones de la Parte 15 especifican los niveles máximos de potencia de transmisión y recepción para las diferentes partes del espectro, pero nada de eso se aplica realmente a una persona que transmite una señal con un SDR o su radio casera. Las únicas regulaciones que pude encontrar relacionadas con radios que en realidad no son productos que se venden fueron específicas para operar una estación de radio AM o FM de baja potencia en las bandas AM/FM. También hay una sección sobre "dispositivos de fabricación casera", pero dice específicamente que no se aplica a nada construido a partir de un kit, y sería exagerado decir que un equipo de transmisión que utiliza un SDR es un dispositivo de fabricación casera. En resumen, las regulaciones de la FCC no son tan simples como "puedes transmitir en estas frecuencias sólo por debajo de estos niveles de potencia", sino que son un enorme conjunto de reglas destinadas a pruebas y cumplimiento.

Otra forma de verlo sería decir "bueno, estos no son dispositivos de la Parte 15, pero sigamos las reglas de la Parte 15 como si lo fueran". Para la banda ISM de 915 MHz, las reglas son que "La intensidad de campo de cualquier emisión radiada dentro de la banda de frecuencia especificada no excederá los 500 microvoltios/metro a 30 metros. El límite de emisión en este párrafo se basa en instrumentos de medición que emplean un detector promedio ". Como puede ver, no es tan simple como una potencia máxima de transmisión en vatios.

Ahora, si tiene su licencia de radioaficionado (ham), la FCC le permite utilizar ciertas bandas reservadas para la radioafición. Todavía hay pautas a seguir y potencias máximas de transmisión, pero al menos estos números se especifican en vatios de
potencia radiada efectiva.  `Esta infografía <http://www.arrl.org/files/file/Regulatory/Band%20Chart/Band%20Chart%20-%2011X17%20Color.pdf>`_ muestra qué bandas están disponibles para usar según su clase de licencia (Técnico, General y Extra). Recomendaría a cualquier persona interesada en transmitir con SDR que obtenga su licencia de radioaficionado, consulte `ARRL's Getting Licensed page <http://www.arrl.org/getting-licensed>`_ para mas información. 

Si alguien tiene más detalles sobre lo que está permitido y lo que no, por favor envíeme un correo electrónico.

************************************************
Transmitir y recibir simultáneamente
************************************************

Usando el truco tx_cyclic_buffer puedes recibir y transmitir fácilmente al mismo tiempo, apagando el transmisor y luego recibiendo.
El siguiente código muestra un ejemplo práctico de cómo transmitir una señal QPSK en la banda de 915 MHz, recibirla y trazar el PSD.

.. code-block:: python

    import numpy as np
    import adi
    import matplotlib.pyplot as plt

    sample_rate = 1e6 # Hz
    center_freq = 915e6 # Hz
    num_samps = 100000 # number of samples per call to rx()

    sdr = adi.Pluto("ip:192.168.2.1")
    sdr.sample_rate = int(sample_rate)

    # Config Tx
    sdr.tx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
    sdr.tx_lo = int(center_freq)
    sdr.tx_hardwaregain_chan0 = -50 # Increase to increase tx power, valid range is -90 to 0 dB

    # Config Rx
    sdr.rx_lo = int(center_freq)
    sdr.rx_rf_bandwidth = int(sample_rate)
    sdr.rx_buffer_size = num_samps
    sdr.gain_control_mode_chan0 = 'manual'
    sdr.rx_hardwaregain_chan0 = 0.0 # dB, increase to increase the receive gain, but be careful not to saturate the ADC

    # Create transmit waveform (QPSK, 16 samples per symbol)
    num_symbols = 1000
    x_int = np.random.randint(0, 4, num_symbols) # 0 to 3
    x_degrees = x_int*360/4.0 + 45 # 45, 135, 225, 315 degrees
    x_radians = x_degrees*np.pi/180.0 # sin() and cos() takes in radians
    x_symbols = np.cos(x_radians) + 1j*np.sin(x_radians) # this produces our QPSK complex symbols
    samples = np.repeat(x_symbols, 16) # 16 samples per symbol (rectangular pulses)
    samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs

    # Start the transmitter
    sdr.tx_cyclic_buffer = True # Enable cyclic buffers
    sdr.tx(samples) # start transmitting

    # Clear buffer just to be safe
    for i in range (0, 10):
        raw_data = sdr.rx()
        
    # Receive samples
    rx_samples = sdr.rx()
    print(rx_samples)

    # Stop transmitting
    sdr.tx_destroy_buffer()

    # Calculate power spectral density (frequency domain version of signal)
    psd = np.abs(np.fft.fftshift(np.fft.fft(rx_samples)))**2
    psd_dB = 10*np.log10(psd)
    f = np.linspace(sample_rate/-2, sample_rate/2, len(psd))

    # Plot time domain
    plt.figure(0)
    plt.plot(np.real(rx_samples[::100]))
    plt.plot(np.imag(rx_samples[::100]))
    plt.xlabel("Time")

    # Plot freq domain
    plt.figure(1)
    plt.plot(f/1e6, psd_dB)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("PSD")
    plt.show()


Debería ver algo parecido a esto, suponiendo que tenga las antenas adecuadas o un cable conectado:

.. image:: ../_images/pluto_tx_rx.svg
   :align: center 

Es un buen ejercicio para adaptarse lentamente. :code:`sdr.tx_hardwaregain_chan0` y :code:`sdr.rx_hardwaregain_chan0` para asegurarse de que la señal recibida se esté debilitando o fortaleciendo como se esperaba.

************************
API de referencia
************************

Para obtener la lista completa de propiedades y funciones de sdr que puede llamar, consulte la `pyadi-iio Pluto Python code (AD936X) <https://github.com/analogdevicesinc/pyadi-iio/blob/master/adi/ad936x.py>`_.

************************
Ejercicios de Python
************************

En lugar de proporcionarle código para ejecutar, he creado varios ejercicios en los que se proporciona el 95% del código y el código restante es Python bastante sencillo de crear. Los ejercicios no pretenden ser difíciles. Les falta suficiente código para hacerte pensar.

Ejercicio 1: determine el rendimiento de su USB
###############################################

Intentemos recibir muestras del PlutoSDR y, en el proceso, veamos cuántas muestras por segundo podemos enviar a través de la conexión USB 2.0. 

**Su tarea es crear un script de Python que determine la tasa de recepción de muestras en Python, es decir, contar las muestras recibidas y realizar un seguimiento del tiempo para calcular la tasa. Luego, intente usar diferentes sample_rate y buffer_size para ver cómo impacta la tasa más alta posible.**

Tenga en cuenta que si recibe menos muestras por segundo que la frecuencia de muestreo especificada, significa que está perdiendo/eliminando una fracción de muestras, lo que probablemente ocurrirá con una frecuencia de muestreo alta. El Pluto sólo utiliza USB 2.0.

El siguiente código actuará como punto de partida pero contiene las instrucciones que necesita para realizar esta tarea.

.. code-block:: python

 import numpy as np
 import adi
 import matplotlib.pyplot as plt
 import time
 
 sample_rate = 10e6 # Hz
 center_freq = 100e6 # Hz
 
 sdr = adi.Pluto("ip:192.168.2.1")
 sdr.sample_rate = int(sample_rate)
 sdr.rx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
 sdr.rx_lo = int(center_freq)
 sdr.rx_buffer_size = 1024 # this is the buffer the Pluto uses to buffer samples
 samples = sdr.rx() # receive samples off Pluto

Además, para calcular el tiempo que tarda algo, puede utilizar el siguiente código:

.. code-block:: python

 start_time = time.time()
 # do stuff
 end_time = time.time()
 print('seconds elapsed:', end_time - start_time)

A continuación se ofrecen varios consejos para empezar.

Sugerencia 1: deberá colocar la línea "samples = sdr.rx()" en un bucle que se ejecute muchas veces (por ejemplo, 100 veces). Debe contar cuántas muestras recibe en cada llamada a sdr.rx() mientras realiza un seguimiento de cuánto tiempo ha transcurrido.

Sugerencia 2: El hecho de que esté calculando muestras por segundo no significa que tenga que realizar exactamente 1 segundo para recibir muestras. Puede dividir la cantidad de muestras que recibió por la cantidad de tiempo que pasó.

Sugerencia 3: comience en sample_rate = 10e6 como muestra el código porque esta velocidad es mucho más de lo que USB 2.0 puede admitir. Podrás ver cuántos datos pasan. Entonces puedes modificar rx_buffer_size. Hazlo mucho más grande y mira qué pasa. Una vez que tenga un script que funcione y haya manipulado rx_buffer_size, intente ajustar sample_rate. Determine qué tan bajo debe llegar hasta poder recibir el 100% de las muestras en Python (es decir, muestras en un ciclo de trabajo del 100%).

Sugerencia 4: en su bucle donde llama a sdr.rx(), intente hacer lo menos posible para no agregar un retraso adicional en el tiempo de ejecución. No hagas nada intensivo como imprimir desde dentro del bucle.

Como parte de este ejercicio, obtendrá una idea del rendimiento máximo de USB 2.0. Puede buscar en línea para verificar sus hallazgos.

Como beneficio adicional, intente cambiar center_freq y rx_rf_bandwidth para ver si afecta la velocidad con la que puede recibir muestras del Pluto.

Ejercicio 2: crear un espectrograma/cascada
###########################################

Para este ejercicio, creará un espectrograma, también conocido como cascada, como aprendimos al final del capitulo :ref:`freq-domain-chapter` .  Un espectrograma es simplemente un conjunto de FFT que se muestran apiladas una encima de otra. En otras palabras, es una imagen en la que un eje representa la frecuencia y el otro eje representa el tiempo.

En el capitulo :ref:`freq-domain-chapter` aprendimos el código Python para realizar una FFT. Para este ejercicio puedes usar fragmentos de código del ejercicio anterior, así como un poco de código Python básico.

Consejos:

1. Intente configurar sdr.rx_buffer_size al tamaño de FFT para que siempre realice 1 FFT por cada llamada del `sdr.rx()`.
2. Cree una matriz 2D para contener todos los resultados de FFT donde cada fila sea 1 FFT. Se puede crear una matriz 2D llena de ceros con: `np.zeros((num_rows, fft_size))`.  Acceda a la fila i de la matriz con: `waterfall_2darray[i,:]`.
3. `plt.imshow()` es una forma conveniente de mostrar una matriz 2D. Esta escala el color automáticamente.

Como objetivo ambicioso, muestre el espectrograma en tiempo real.




