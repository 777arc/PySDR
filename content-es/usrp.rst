.. _usrp-chapter:

####################################
USRP en Python
####################################

.. image:: ../_images/usrp.png
   :scale: 50 % 
   :align: center
   :alt: The family of USRP radios from Ettus Research
   
En este capítulo aprenderemos cómo usar la API UHD Python para controlar y recibir/transmitir señales con un `USRP <https://www.ettus.com/>`_ que es una serie de SDR fabricados por Ettus Research (ahora parte de NI). Analizaremos la transmisión y recepción en USRP en Python y profundizaremos en temas específicos de USRP, incluidos argumentos de transmisión, subdispositivos, canales, sincronización de 10 MHz y PPS. 

************************************
Instalación de Software/Drivers USRP
************************************

Si bien el código Python proporcionado en este libro de texto debería funcionar en Windows, Mac y Linux, solo proporcionaremos instrucciones de instalación de controladores/API específicas para Ubuntu 22 (aunque las instrucciones a continuación deberían funcionar en la mayoría de las distribuciones basadas en Debian). Comenzaremos creando una VM VirtualBox Ubuntu 22; no dude en omitir la parte de VM si ya tiene su sistema operativo listo para funcionar. Alternativamente, si está en Windows 11, el Subsistema de Windows para Linux (WSL) que usa Ubuntu 22 tiende a funcionar bastante bien y admite gráficos listos para usar.

Configurar la máquina virtual en Ubuntu 22
##########################################

(Opcional)

1. Descarge Ubuntu 22.04 Desktop .iso- https://ubuntu.com/download/desktop
2. Instale y abra `VirtualBox <https://www.virtualbox.org/wiki/Downloads>`_.
3. Cree una nueva máquina virtual. Para el tamaño de la memoria, recomiendo utilizar el 50% de la RAM de su computadora.
4. Cree el disco duro virtual, elija VDI y asigne tamaño dinámicamente. 15 GB deberían ser suficientes. Si quieres estar realmente seguro, puedes usar más.
5. Inicie la máquina virtual. Le pedirá medios de instalación. Elija el archivo .iso del escritorio de Ubuntu 22. Elija "instalar ubuntu", use las opciones predeterminadas y una ventana emergente le advertirá sobre los cambios que está a punto de realizar. Pulsa continuar. Elija nombre/contraseña y luego espere a que la VM termine de inicializarse. Después de finalizar, la VM se reiniciará, pero debe apagarla después del reinicio.
6. Vaya a la configuración de VM (el ícono de ajustes).
7. En sistema > procesador > elija al menos 3 CPU. Si tiene una tarjeta de video real, en pantalla > memoria de video > elija algo mucho más alto.
8. Inicie su máquina virtual.
9. Para USRP de tipo USB, deberá instalar adiciones de invitados de VM. Dentro de la máquina virtual, vaya a Dispositivos > Insertar CD de Guest Additions > presione ejecutar cuando aparezca un cuadro. Sigue las instrucciones. Reinicie la VM, luego intente reenviar el USRP a la VM, suponiendo que aparezca en la lista en Dispositivos > USB. El portapapeles compartido se puede habilitar a través de Dispositivos > Portapapeles compartido > Bidireccional.

Instalación de UHD y API de Python
##################################

Los siguientes comandos de terminal deberían compilar e instalar la última versión de UHD, incluida la API de Python:

.. code-block:: bash

 sudo apt-get install git cmake libboost-all-dev libusb-1.0-0-dev python3-docutils python3-mako python3-numpy python3-requests python3-ruamel.yaml python3-setuptools build-essential
 cd ~
 git clone https://github.com/EttusResearch/uhd.git
 cd uhd/host
 mkdir build
 cd build
 cmake -DENABLE_TESTS=OFF -DENABLE_C_API=OFF -DENABLE_MANUAL=OFF ..
 make -j8
 sudo make install
 sudo ldconfig

Para obtener más ayuda, consulte el sitio oficial de Ettus. La pagina `Construyendo e instalando UHD desde la fuente <https://files.ettus.com/manual/page_build_guide.html>`_ .  Tenga en cuenta que también existen métodos para instalar los controladores que no requieren compilación desde la fuente.

Prueba de controladores UHD y API de Python
###########################################

Abra una nueva terminal y escriba los siguientes comandos:

.. code-block:: bash

 python3
 import uhd
 usrp = uhd.usrp.MultiUSRP()
 samples = usrp.recv_num_samps(10000, 100e6, 1e6, [0], 50)
 print(samples[0:10])

Si no se producen errores, ¡está listo para comenzar!


Evaluación comparativa de la velocidad USRP en Python
#####################################################

(Opcional)

Si utilizó la instalación estándar desde el origen, el siguiente comando debería comparar la tasa de recepción de su USRP utilizando la API de Python. Si el uso de 56e6 provocó la caída de muchas muestras o excesos, intente reducir el número. Las muestras caídas no necesariamente arruinarán nada, pero es una buena manera de probar las ineficiencias que pueden surgir al usar una máquina virtual o una computadora más antigua, por ejemplo. Si usa un B 2X0, una computadora bastante moderna con un puerto USB 3.0 funcionando correctamente debería lograr funcionar a 56 MHz sin pérdida de muestras, especialmente con num_recv_frames configurado en un nivel tan alto.

.. code-block:: bash

 python /usr/lib/uhd/examples/python/benchmark_rate.py --rx_rate 56e6 --args "num_recv_frames=1000"


************************
Recepción USRP
************************

Recibir muestras de un USRP es extremadamente fácil usando la función incorporada "recv_num_samps()", a continuación se muestra el código Python que sintoniza el USRP a 100 MHz, usando una frecuencia de muestreo de 1 MHz, y toma 10,000 muestras del USRP, usando una ganancia de recepción de 50 dB:

.. code-block:: python

 import uhd
 usrp = uhd.usrp.MultiUSRP()
 samples = usrp.recv_num_samps(10000, 100e6, 1e6, [0], 50) # units: N, Hz, Hz, list of channel IDs, dB
 print(samples[0:10])

El [0] le dice al USRP que use su primer puerto de entrada y que solo reciba muestras de un canal (para que un B210 reciba dos canales a la vez, por ejemplo, puede usar [0, 1]).

Aquí tienes un consejo si estás intentando recibir a una velocidad alta pero obtienes desbordamientos (aparecen 0's en tu consola). En su lugar usa :code:`usrp = uhd.usrp.MultiUSRP()` :

.. code-block:: python

 usrp = uhd.usrp.MultiUSRP("num_recv_frames=1000")

lo que hace que el búfer de recepción sea mucho más grande (el valor predeterminado es 32), lo que ayuda a reducir los desbordamientos. El tamaño real del búfer en bytes depende del USRP y del tipo de conexión, pero simplemente configurando :code:`num_recv_frames` a un valor mucho mayor que 32 tiende a ayudar.

Para aplicaciones más serias, recomiendo no usar la función de conveniencia recv_num_samps(), porque oculta algunos de los comportamientos interesantes que ocurren bajo el capó, y hay algunas configuraciones que ocurren en cada llamada que quizás solo queramos hacer una vez al principio. , por ejemplo, si queremos recibir muestras de forma indefinida. El siguiente código tiene la misma funcionalidad que recv_num_samps(); de hecho, es casi exactamente lo que se llama cuando usas la función de conveniencia, pero ahora tenemos la opción de modificar el comportamiento:

.. code-block:: python

 import uhd
 import numpy as np
 
 usrp = uhd.usrp.MultiUSRP()
 
 num_samps = 10000 # number of samples received
 center_freq = 100e6 # Hz
 sample_rate = 1e6 # Hz
 gain = 50 # dB
 
 usrp.set_rx_rate(sample_rate, 0)
 usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(center_freq), 0)
 usrp.set_rx_gain(gain, 0)
 
 # Set up the stream and receive buffer
 st_args = uhd.usrp.StreamArgs("fc32", "sc16")
 st_args.channels = [0]
 metadata = uhd.types.RXMetadata()
 streamer = usrp.get_rx_stream(st_args)
 recv_buffer = np.zeros((1, 1000), dtype=np.complex64)
 
 # Start Stream
 stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
 stream_cmd.stream_now = True
 streamer.issue_stream_cmd(stream_cmd)
 
 # Receive Samples
 samples = np.zeros(num_samps, dtype=np.complex64)
 for i in range(num_samps//1000):
     streamer.recv(recv_buffer, metadata)
     samples[i*1000:(i+1)*1000] = recv_buffer[0]
 
 # Stop Stream
 stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
 streamer.issue_stream_cmd(stream_cmd)
 
 print(len(samples))
 print(samples[0:10])

Con num_samps configurado en 10,000 y recv_buffer configurado en 1000, el bucle for se ejecutará 10 veces, es decir, habrá 10 llamadas a streamer.recv. Tenga en cuenta que codificamos recv_buffer en 1000, pero puede encontrar el valor máximo permitido usando :code:`streamer.get_max_num_samps()`, que suele rondar los 3000 y tantos. También tenga en cuenta que recv_buffer debe ser 2d porque se usa la misma API cuando se reciben múltiples canales a la vez, pero en nuestro caso solo recibimos un canal, por lo que recv_buffer[0] nos dio la matriz 1D de muestras que queríamos. No es necesario que entiendas mucho sobre cómo inicia y finaliza la transmisión por ahora, pero debes saber que hay otras opciones además del modo "continuo", como recibir una cantidad específica de muestras y hacer que la transmisión se detenga automáticamente. Aunque no procesamos metadatos en este código de ejemplo, contiene los errores que ocurren, entre otras cosas, que puede verificar mirando metadata.error_code en cada iteración del bucle, si lo desea (los errores tienden a aparecer también en la propia consola, como resultado de UHD, así que no sienta que tiene que buscarlos en su código Python). 

Ganancia del Receptor
#####################

La siguiente lista muestra el rango de ganancia de los diferentes USRP, todos van desde 0 dB hasta el número especificado a continuación. Tenga en cuenta que esto no es dBm, es esencialmente dBm combinado con algún desplazamiento desconocido porque estos no son dispositivos calibrados. 

* B200/B210/B200-mini: 76 dB
* X310/N210 with WBX/SBX/UBX: 31.5 dB
* X310 with TwinRX: 93 dB
* E310/E312: 76 dB
* N320/N321: 60 dB

También puedes usar el comando :code:`uhd_usrp_probe` en un terminal y en la sección RX Frontend mencionará el rango de ganancia.

Al especificar la ganancia, puede usar la función normal set_rx_gain() que toma el valor de ganancia en dB, pero también puede usar set_normalized_rx_gain() que toma un valor de 0 a 1 y lo convierte automáticamente al rango del USRP. estás usando. Esto resulta útil a la hora de crear una aplicación que admita diferentes modelos de USRP. La desventaja de usar ganancia normalizada es que ya no tienes tus unidades en dB, por lo que si quieres aumentar tu ganancia en 10 dB, por ejemplo, ahora tienes que calcular la cantidad.

Control de ganancia automática
##############################

Algunos USRP, incluidas las series B200 y E310, admiten el control automático de ganancia (AGC), que ajustará automáticamente la ganancia de recepción en respuesta al nivel de la señal recibida, en un intento de "llenar" mejor los bits del ADC. AGC se puede activar usando:

.. code-block:: python

 usrp.set_rx_agc(True, 0) # 0 for channel 0, i.e. the first channel of the USRP

Si tiene un USRP que no implementa un AGC, se generará una excepción al ejecutar la línea anterior. Con AGC activado, configurar la ganancia no hará nada.

Argumentos de transmisión
*************************

En el ejemplo completo anterior verás la línea :code:`st_args = uhd.usrp.StreamArgs("fc32", "sc16")`.  El primer argumento es el formato de datos de la CPU, que es el tipo de datos de las muestras una vez que están en su computadora. UHD admite los siguientes tipos de datos de CPU cuando se utiliza la API de Python:

.. list-table::
   :widths: 15 20 30
   :header-rows: 1
   
   * - Stream Arg
     - Numpy Data Type
     - Description
   * - fc64
     - np.complex128
     - Complex-valued double-precision data
   * - fc32
     - np.complex64
     - Complex-valued single-precision data

Es posible que vea otras opciones en la documentación de la API UHD C++, pero nunca se implementaron dentro de la API de Python, al menos en el momento de escribir este artículo.

El segundo argumento es el formato de datos "por cable", es decir, el tipo de datos a medida que las muestras se envían a través de USB/Ethernet/SFP al host. Para la API de Python, las opciones son: "sc16", "sc12" y "sc8", y la opción de 12 bits solo es compatible con ciertos USRP. Esta elección es importante porque la conexión entre el USRP y la computadora host suele ser el cuello de botella, por lo que al cambiar de 16 bits a 8 bits se puede lograr una velocidad más alta. Recuerde también que muchos USRP tienen ADC limitados a 12 o 14 bits; usar "sc16" no significa que el ADC sea de 16 bits. 

Para la parte del canal :code:`st_args`, consulte la subsección Subdispositivos y canales a continuación.

************************
Transmisor
************************

De manera similar a la función de conveniencia recv_num_samps(), UHD proporciona la función send_waveform() para transmitir un lote de muestras; a continuación se muestra un ejemplo. Si especifica una duración (en segundos) mayor que la señal proporcionada, simplemente la repetirá. Ayuda a mantener los valores de las muestras entre -1,0 y 1,0.

.. code-block:: python

 import uhd
 import numpy as np
 usrp = uhd.usrp.MultiUSRP()
 samples = 0.1*np.random.randn(10000) + 0.1j*np.random.randn(10000) # create random signal
 duration = 10 # seconds
 center_freq = 915e6
 sample_rate = 1e6
 gain = 20 # [dB] start low then work your way up
 usrp.send_waveform(samples, duration, center_freq, sample_rate, [0], gain)

Para obtener detalles sobre cómo funciona esta práctica función interna, consulte el código fuente. `aqui <https://github.com/EttusResearch/uhd/blob/master/host/python/uhd/usrp/multi_usrp.py>`_. 


Ganancia del transmisor
#######################

De manera similar al lado de recepción, el rango de ganancia de transmisión varía según el modelo USRP, desde 0 dB hasta el número especificado a continuación:

* B200/B210/B200-mini: 90 dB
* N210 with WBX: 25 dB
* N210 with SBX or UBX: 31.5 dB
* E310/E312: 90 dB
* N320/N321: 60 dB

También hay una función set_normalized_tx_gain() si desea especificar la ganancia de transmisión usando el rango de 0 a 1.

************************************************
Transmitir y recibir simultáneamente con USRP
************************************************

Si deseas transmitir y recibir usando el mismo USRP al mismo tiempo, la clave es hacerlo usando múltiples hilos dentro del mismo proceso; el USRP no puede abarcar múltiples procesos. Por ejemplo, en el `txrx_loopback_to_file <https://github.com/EttusResearch/uhd/blob/master/host/examples/txrx_loopback_to_file.cpp>`_ en el ejemplo de C++ se crea un hilo separado para ejecutar el transmisor y la recepción se realiza en el hilo principal. También puedes generar dos hilos, uno para transmitir y otro para recibir, como se hace en el `benchmark_rate <https://github.com/EttusResearch/uhd/blob/master/host/examples/python/benchmark_rate.py>`_ del ejemplo en Python. Aquí no se muestra un ejemplo completo, simplemente porque sería un ejemplo bastante largo y benchmark_rate.py de Ettus siempre puede actuar como punto de partida para alguien.


*********************************
Subdispositivo, canales y antenas
*********************************

Una fuente común de confusión al utilizar USRP es cómo elegir el subdispositivo y la ID de canal correctos. Es posible que hayas notado que en todos los ejemplos anteriores utilizamos el canal 0 y no especificamos nada relacionado con el subdesarrollo. Si está usando un B210 y solo quiere usar RF:B en lugar de RF:A, todo lo que tiene que hacer es elegir el canal 1 en lugar de 0. Pero en USRP como el X310 que tienen dos ranuras para placa secundaria, debe indicarlo. UHD si desea utilizar la ranura A o B y qué canal en esa placa secundaria, por ejemplo:

.. code-block:: python

 usrp.set_rx_subdev_spec("B:0")

Si desea utilizar el puerto TX/RX en lugar del RX2 (el predeterminado), es tan simple como:

.. code-block:: python

 usrp.set_rx_antenna('TX/RX', 0) # set channel 0 to 'TX/RX'

que básicamente solo controla un interruptor de RF a bordo del USRP, para enrutarlo desde el otro conector SMA.

Para recibir o transmitir en dos canales a la vez, en lugar de utilizar :code:`st_args.channels = [0]` se proporciona una lista, como :code:`[0,1]`.  El búfer de muestras de recepción tendrá que ser de tamaño (2, N) en este caso, en lugar de (1,N). Sólo recuerde que con la mayoría de los USRP, ambos canales comparten un LO, por lo que no puede sintonizar diferentes frecuencias a la vez.

*****************************
Sincronización a 10 MHz y PPS
*****************************

Una de las grandes ventajas de utilizar un USRP sobre otros SDR es su capacidad de sincronizarse con una fuente externa o integrada. `GPSDO <https://www.ettus.com/all-products/gpsdo-tcxo-module/>`_, permitiendo aplicaciones multireceptor como TDOA. Si ha conectado una fuente externa de 10 MHz y PPS a su USRP, querrá asegurarse de llamar a estas dos líneas después de inicializar su USRP:

.. code-block:: python

 usrp.set_clock_source("external")
 usrp.set_time_source("external")

Si está utilizando un GPSDO a bordo, utilizará en su lugar:

.. code-block:: python

 usrp.set_clock_source("gpsdo")
 usrp.set_time_source("gpsdo")

En cuanto a la sincronización de frecuencia, no hay mucho más que hacer; El LO utilizado en el mezclador del USRP ahora estará vinculado a la fuente externa o `GPSDO <https://www.ettus.com/all-products/gpsdo-tcxo-module/>`_.  Pero en lo que respecta al tiempo, es posible que desee ordenar al USRP que comience a muestrear exactamente en el PPS, por ejemplo. Esto se puede hacer con el siguiente código:

.. code-block:: python

 # copy the receive example above, everything up until # Start Stream

 # Wait for 1 PPS to happen, then set the time at next PPS to 0.0
 time_at_last_pps = usrp.get_time_last_pps().get_real_secs()
 while time_at_last_pps == usrp.get_time_last_pps().get_real_secs():
     time.sleep(0.1) # keep waiting till it happens- if this while loop never finishes then the PPS signal isn't there
 usrp.set_time_next_pps(uhd.libpyuhd.types.time_spec(0.0))
 
 # Schedule Rx of num_samps samples exactly 3 seconds from last PPS
 stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
 stream_cmd.num_samps = num_samps
 stream_cmd.stream_now = False
 stream_cmd.time_spec = uhd.libpyuhd.types.time_spec(3.0) # set start time (try tweaking this)
 streamer.issue_stream_cmd(stream_cmd)
 
 # Receive Samples.  recv() will return zeros, then our samples, then more zeros, letting us know it's done
 waiting_to_start = True # keep track of where we are in the cycle (see above comment)
 nsamps = 0
 i = 0
 samples = np.zeros(num_samps, dtype=np.complex64)
 while nsamps != 0 or waiting_to_start:
     nsamps = streamer.recv(recv_buffer, metadata)
     if nsamps and waiting_to_start:
         waiting_to_start = False
     elif nsamps:
         samples[i:i+nsamps] = recv_buffer[0][0:nsamps]
     i += nsamps

Si parece que no funciona, pero no arroja ningún error, intente cambiar ese número 3.0 entre 1.0 y 5.0. También puede verificar los metadatos después de la llamada a recv(), simplemente verifique :code:`if metadata.error_code != uhd.types.RXMetadataErrorCode.none:`.  
     
Por motivos de depuración, puede verificar que la señal de 10 MHz se muestra en el USRP verificando el retorno de :code:`usrp.get_mboard_sensor("ref_locked", 0)`.  Si la señal PPS no aparece, lo sabrás porque el primer bucle while del código anterior nunca finalizará.
     
     
     
     
