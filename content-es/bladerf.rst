.. _bladerf-chapter:

##################
BladeRF en Python
##################

El bladeRF 2.0 (también conocido como bladeRF 2.0 micro) de la empresa `Nuand <https://www.nuand.com>`_ es un SDR basado en USB 3.0 con dos canales de recepción, dos canales de transmisión, un rango sintonizable de 47 MHz a 6 GHz y la capacidad de muestrear hasta 61 MHz o hasta 122 MHz cuando se piratea. Utiliza el circuito integrado de RF (RFIC) AD9361 al igual que el USRP B210 y PlutoSDR, por lo que el rendimiento de RF será similar. El bladeRF 2.0 se lanzó en 2021, mantiene un factor de forma pequeño de 2,5" x 4,5" y viene en dos tamaños de FPGA diferentes (xA4 y xA9). Si bien este capítulo se centra en el bladeRF 2.0, gran parte del código también se aplicará al bladeRF original, que `salió en 2013 <https://www.kickstarter.com/projects/1085541682/bladerf-usb-30-software-defined-radio>`_.

.. image:: ../_images/bladeRF_micro.png
   :scale: 35 %
   :align: center 
   :alt: bladeRF 2.0 glamour shot

********************************
Arquitectura del bladeRF 
********************************

En un nivel alto, el bladeRF 2.0 se basa en el RFIC AD9361, combinado con un FPGA Cyclone V (ya sea el 49 kLE :code:`5CEA4` o el 301 kLE :code:`5CEA9`) y un controlador Cypress FX3 USB 3.0. que tiene un núcleo ARM9 de 200 MHz en su interior, cargado con firmware personalizado. El diagrama de bloques del bladeRF 2.0 se muestra a continuación:

.. image:: ../_images/bladeRF-2.0-micro-Block-Diagram-4.png
   :scale: 80 %
   :align: center 
   :alt: bladeRF 2.0 block diagram

La FPGA controla el RFIC, realiza filtrado digital y enmarca paquetes para transferirlos a través de USB (entre otras cosas). El `codigo fuente de la imagen FPGA <https://github.com/Nuand/bladeRF/tree/master/hdl>`_ está escrita en VHDL y requiere el software de diseño gratuito Quartus Prime Lite para compilar imágenes personalizadas. Imágenes precompiladas disponibles `aqui <https://www.nuand.com/fpga_images/>`_.

El `codigo fuente del firmware Cypress FX3 <https://github.com/Nuand/bladeRF/tree/master/fx3_firmware>`_  es de código abierto e incluye código para:

1. Cargue la imagen FPGA
2. Transfiera muestras de IQ entre la FPGA y el host a través de USB 3.0
3. Controlar GPIO de la FPGA sobre UART

Desde una perspectiva de flujo de señal, hay dos canales de recepción y dos canales de transmisión, y cada canal tiene una entrada/salida de baja y alta frecuencia al RFIC, según la banda que se esté utilizando. Es por esta razón que se necesita un interruptor electrónico de RF unipolar y bidireccional (SPDT) entre los conectores RFIC y SMA. La T de polarización es un circuito integrado que proporciona ~4,5 V DC en el conector SMA y se utiliza para alimentar cómodamente un amplificador externo u otros componentes de RF. Este desplazamiento de DC adicional se encuentra en el lado de RF del SDR, por lo que no interfiere con la operación básica de recepción/transmisión.

JTAG es un tipo de interfaz de depuración que permite probar y verificar diseños durante el proceso de desarrollo.

Al final de este capítulo, analizamos el oscilador VCTCXO, PLL y el puerto de expansión.

*************************************
Configuración del software y hardware
*************************************

Ubuntu (o Ubuntu con WSL)
#############################

En Ubuntu y otros sistemas basados en Debian, puede instalar el software bladeRF con los siguientes comandos:

.. code-block:: bash

 sudo apt update
 sudo apt install cmake python3-pip libusb-1.0-0
 cd ~
 git clone --depth 1 https://github.com/Nuand/bladeRF.git
 cd bladeRF/host
 mkdir build && cd build
 cmake ..
 make -j8
 sudo make install
 sudo ldconfig
 cd ../libraries/libbladeRF_bindings/python
 sudo python3 setup.py install

Esto instalará la biblioteca libbladerf, los enlaces de Python, las herramientas de línea de comandos de bladerf, el descargador de firmware y el descargador de flujo de bits FPGA. Para verificar qué versión de la biblioteca instaló, use :code:`bladerf-tool version` (esta guía fue escrita usando libbladeRF versión v2.5.0).

Si está utilizando Ubuntu a través de WSL, en el lado de Windows deberá reenviar el dispositivo USB bladeRF a WSL, primero instalando la última versión `usbipd utility msi <https://github.com/dorssel/usbipd-win/releases>`_ (esta guía supone que tiene usbipd-win 4.0.0 o superior), luego abre PowerShell en modo administrador y ejecuta:

.. code-block:: bash

    usbipd list
    # (find the BUSID labeled bladeRF 2.0 and substitute it in the command below)
    usbipd bind --busid 1-23
    usbipd attach --wsl --busid 1-23

En el lado WSL, debería poder ejecutar :code:`lsusb` y ver un nuevo elemento llamado :code:`Nuand LLC bladeRF 2.0 micro`. Tenga en cuenta que puede agregar el indicador :code:`--auto-attach` al comando :code:`usbipd adjunto` si desea que se vuelva a conectar automáticamente.

(Puede que no sea necesario) Tanto para Linux nativo como para WSL, debemos instalar las reglas udev para no obtener errores de permisos:

.. code-block::

 sudo nano /etc/udev/rules.d/88-nuand.rules

y pegue la siguiente línea:

.. code-block::

 ATTRS{idVendor}=="2cf0", ATTRS{idProduct}=="5250", MODE="0666"

Para guardar y salir de nano, use: control-o, luego Enter, luego control-x. Para actualizar udev, ejecute:

.. code-block:: bash

    sudo udevadm control --reload-rules && sudo udevadm trigger

Si estás usando WSL y dice :code:`Failed to send reload request: No such file or directory`, eso significa que el servicio udev no se está ejecutando y necesitarás :code:`sudo nano /etc/wsl.conf` y agrega las líneas:

.. code-block:: bash

 [boot]
 command="service udev start"

luego reinicie WSL usando el siguiente comando en PowerShell con admin: :code:`wsl.exe --shutdown`.

Desenchufe y vuelva a enchufar su bladeRF (los usuarios de WSL deberán volver a conectarlo) y pruebe los permisos con:

.. code-block:: bash

 bladerf-tool probe
 bladerf-tool info

y sabrás que funcionó si ves tu bladeRF 2.0 en la lista y **si no** se ve :code:`Found a bladeRF via VID/PID, but could not open it due to insufficient permissions`.  Si funcionó, anote la versión de FPGA y la versión de firmware.

(Opcional) Instale el firmware y las imágenes FPGA más recientes (v2.4.0 y v0.15.0 respectivamente cuando se escribió esta guía) usando:

.. code-block:: bash

 cd ~/Downloads
 wget https://www.nuand.com/fx3/bladeRF_fw_latest.img
 bladerf-tool flash_fw bladeRF_fw_latest.img

 # for xA4 use:
 wget https://www.nuand.com/fpga/hostedxA4-latest.rbf
 bladerf-tool flash_fpga hostedxA4-latest.rbf

 # for xA9 use:
 wget https://www.nuand.com/fpga/hostedxA9-latest.rbf
 bladerf-tool flash_fpga hostedxA9-latest.rbf

Desenchufe y enchufe su bladeRF para realizar un ciclo de energía.

Ahora probaremos su funcionalidad recibiendo 1 millón de muestras en la banda de radio FM, a una frecuencia de muestreo de 10 MHz, en un archivo /tmp/samples.sc16:

.. code-block:: bash

 bladerf-tool rx --num-samples 1000000 /tmp/samples.sc16 100e6 10e6

un par :code:`Hit stall for buffer` se espera, pero sabrá si funcionó si ve un archivo /tmp/samples.sc16 de 4 MB.

Por último, probaremos la API de Python con:

.. code-block:: bash

 python3
 import bladerf
 bladerf.BladeRF()
 exit()

Sabrás que funcionó si ves algo como :code:`<BladeRF(<DevInfo(...)>)>` y sin advertencias/errores.

Windows y MacOS
###################

Para usuarios Windows, ver https://github.com/Nuand/bladeRF/wiki/Getting-Started%3A-Windows, y para usuarios MacOS, ver https://github.com/Nuand/bladeRF/wiki/Getting-started:-Mac-OSX.

**********************************
API basicas para bladeRF en Python
**********************************

Para empezar, sondeemos el bladeRF para obtener información útil, utilizando el siguiente script. **¡No asigne a su script el nombre bladerf.py** o entrará en conflicto con el módulo bladeRF Python!

.. code-block:: python

 from bladerf import _bladerf
 import numpy as np
 import matplotlib.pyplot as plt

 sdr = _bladerf.BladeRF()
 
 print("Device info:", _bladerf.get_device_list()[0])
 print("libbladeRF version:", _bladerf.version()) # v2.5.0
 print("Firmware version:", sdr.get_fw_version()) # v2.4.0
 print("FPGA version:", sdr.get_fpga_version())   # v0.15.0
 
 rx_ch = sdr.Channel(_bladerf.CHANNEL_RX(0)) # give it a 0 or 1
 print("sample_rate_range:", rx_ch.sample_rate_range)
 print("bandwidth_range:", rx_ch.bandwidth_range)
 print("frequency_range:", rx_ch.frequency_range)
 print("gain_modes:", rx_ch.gain_modes)
 print("manual gain range:", sdr.get_gain_range(_bladerf.CHANNEL_RX(0))) # ch 0 or 1

Para bladeRF 2.0 xA9, la salida debería verse así:

.. code-block:: python
 
    Device info: Device Information
        backend  libusb
        serial   f80a27b1010448dfb7a003ef7fa98a59
        usb_bus  2
        usb_addr 5
        instance 0
    libbladeRF version: v2.5.0 ("2.5.0-git-624994d")
    Firmware version: v2.4.0 ("2.4.0-git-a3d5c55f")
    FPGA version: v0.15.0 ("0.15.0")
    sample_rate_range: Range
        min   520834
        max   61440000
        step  2
        scale 1.0

    bandwidth_range: Range
        min   200000
        max   56000000
        step  1
        scale 1.0

    frequency_range: Range
        min   70000000
        max   6000000000
        step  2
        scale 1.0

    gain_modes: [<GainMode.Default: 0>, <GainMode.Manual: 1>, <GainMode.FastAttack_AGC: 2>, <GainMode.SlowAttack_AGC: 3>, <GainMode.Hybrid_AGC: 4>]

    manual gain range: Range
        min   -15
        max   60
        step  1
        scale 1.0

El parámetro de ancho de banda establece el filtro utilizado por el SDR al realizar la operación de recepción, por lo que normalmente lo configuramos para que sea igual o ligeramente menor que sample_rate/2. Es importante comprender los modos de ganancia, el SDR utiliza un modo de ganancia manual donde usted proporciona la ganancia en dB o un control de ganancia automático (AGC) que tiene tres configuraciones diferentes (rápido, lento, híbrido). Para aplicaciones como la monitorización del espectro, se recomienda la ganancia manual (para que pueda ver cuándo van y vienen las señales), pero para aplicaciones como la recepción de una señal específica que espera que exista, el AGC será más útil porque ajustará automáticamente la ganancia a permitir que la señal llene el convertidor analógico a digital (ADC).

Para configurar los parámetros principales del SDR, podemos agregar el siguiente código:

.. code-block:: python

 sample_rate = 10e6
 center_freq = 100e6
 gain = 50 # -15 to 60 dB
 num_samples = int(1e6)
 
 rx_ch.frequency = center_freq
 rx_ch.sample_rate = sample_rate
 rx_ch.bandwidth = sample_rate/2
 rx_ch.gain_mode = _bladerf.GainMode.Manual
 rx_ch.gain = gain

********************************
Recibir muestras en Python
********************************

A continuación, trabajaremos con el bloque de código anterior para recibir 1 millón de muestras en la banda de radio FM, a una frecuencia de muestreo de 10 MHz, tal como lo hicimos antes. Cualquier antena en el puerto RX1 debería poder recibir FM, ya que es muy potente. El siguiente código muestra cómo funciona la API de flujo síncrono bladeRF; se debe configurar y crear un búfer de recepción antes de que comience la recepción. El bucle :code:` while True:` continuará recibiendo muestras hasta que se alcance el número de muestras solicitadas. Las muestras recibidas se almacenan en una matriz numpy separada, para que podamos procesarlas una vez finalizado el ciclo.

.. code-block:: python

 # Setup synchronous stream
 sdr.sync_config(layout = _bladerf.ChannelLayout.RX_X1, # or RX_X2
                 fmt = _bladerf.Format.SC16_Q11, # int16s
                 num_buffers    = 16,
                 buffer_size    = 8192,
                 num_transfers  = 8,
                 stream_timeout = 3500)
 
 # Create receive buffer
 bytes_per_sample = 4 # don't change this, it will always use int16s
 buf = bytearray(1024 * bytes_per_sample)
 
 # Enable module
 print("Starting receive")
 rx_ch.enable = True
 
 # Receive loop
 x = np.zeros(num_samples, dtype=np.complex64) # storage for IQ samples
 num_samples_read = 0
 while True:
     if num_samples > 0 and num_samples_read == num_samples:
         break
     elif num_samples > 0:
         num = min(len(buf) // bytes_per_sample, num_samples - num_samples_read)
     else:
         num = len(buf) // bytes_per_sample
     sdr.sync_rx(buf, num) # Read into buffer
     samples = np.frombuffer(buf, dtype=np.int16)
     samples = samples[0::2] + 1j * samples[1::2] # Convert to complex type
     samples /= 2048.0 # Scale to -1 to 1 (its using 12 bit ADC)
     x[num_samples_read:num_samples_read+num] = samples[0:num] # Store buf in samples array
     num_samples_read += num
 
 print("Stopping")
 rx_ch.enable = False
 print(x[0:10]) # look at first 10 IQ samples
 print(np.max(x)) # if this is close to 1, you are overloading the ADC, and should reduce the gain

Se esperan algunos :code:`Hit stop for buffer` al final. El último número impreso muestra la muestra máxima recibida; querrás ajustar tu ganancia para intentar obtener ese valor entre 0,5 y 0,8. Si es 0,999, significa que su receptor está sobrecargado/saturado y la señal se distorsionará (se verá manchada en todo el dominio de la frecuencia).

Para visualizar la señal recibida, mostremos las muestras de IQ usando un espectrograma (consulte :ref:`spectrogram-section` para obtener más detalles sobre cómo funcionan los espectrogramas). Agregue lo siguiente al final del bloque de código anterior:

.. code-block:: python

 # Create spectrogram
 fft_size = 2048
 num_rows = len(x) // fft_size # // is an integer division which rounds down
 spectrogram = np.zeros((num_rows, fft_size))
 for i in range(num_rows):
     spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)
 extent = [(center_freq + sample_rate/-2)/1e6, (center_freq + sample_rate/2)/1e6, len(x)/sample_rate, 0]
 plt.imshow(spectrogram, aspect='auto', extent=extent)
 plt.xlabel("Frequency [MHz]")
 plt.ylabel("Time [s]")
 plt.show()

.. image:: ../_images/bladerf-waterfall.svg
   :align: center 
   :target: ../_images/bladerf-waterfall.svg
   :alt: bladeRF spectrogram example

Cada línea ondulada vertical es una señal de radio FM. No tengo idea de a qué se debe el pulso en el lado derecho, reducir la ganancia no hizo que desapareciera.


*********************************
Transmisión de muestras en Python
*********************************

El proceso de transmisión de muestras con bladeRF es muy similar al de recepción. La principal diferencia es que debemos generar las muestras para transmitir y luego escribirlas en bladeRF usando el método :code:`sync_tx` que puede manejar todo nuestro lote de muestras a la vez (hasta ~4B muestras). El siguiente código muestra cómo transmitir un tono simple y luego repetirlo 30 veces. El tono se genera usando numpy y luego se escala para que esté entre -32767 y 32767, de modo que pueda almacenarse como int16s. Luego, el tono se convierte en bytes y se utiliza como búfer de transmisión. La API de flujo síncrono se utiliza para transmitir las muestras, y el bucle :code:` while True:` continuará transmitiendo muestras hasta que se alcance el número de repeticiones solicitadas. Si desea transmitir muestras desde un archivo, simplemente use :code:`samples = np.fromfile('yourfile.iq', dtype=np.int16)` (o cualquier tipo de datos que sean) para leer las muestras, y luego conviértalos a bytes usando :code:`samples.tobytes()`.

.. code-block:: python

 from bladerf import _bladerf
 import numpy as np
 
 sdr = _bladerf.BladeRF()
 tx_ch = sdr.Channel(_bladerf.CHANNEL_TX(0)) # give it a 0 or 1
 
 sample_rate = 10e6
 center_freq = 100e6
 gain = 0 # -15 to 60 dB. for transmitting, start low and slowly increase, and make sure antenna is connected
 num_samples = int(1e6)
 repeat = 30 # number of times to repeat our signal
 print('duration of transmission:', num_samples/sample_rate*repeat, 'seconds')
 
 # Generate IQ samples to transmit (in this case, a simple tone)
 t = np.arange(num_samples) / sample_rate
 f_tone = 1e6
 samples = np.exp(1j * 2 * np.pi * f_tone * t) # will be -1 to +1
 samples = samples.astype(np.complex64)
 samples *= 32767 # scale so they can be stored as int16s
 samples = samples.view(np.int16)
 buf = samples.tobytes() # convert our samples to bytes and use them as transmit buffer
 
 tx_ch.frequency = center_freq
 tx_ch.sample_rate = sample_rate
 tx_ch.bandwidth = sample_rate/2
 tx_ch.gain = gain
  
 # Setup synchronous stream
 sdr.sync_config(layout=_bladerf.ChannelLayout.TX_X1, # or TX_X2
                 fmt=_bladerf.Format.SC16_Q11, # int16s
                 num_buffers=16,
                 buffer_size=8192,
                 num_transfers=8,
                 stream_timeout=3500)
 
 print("Starting transmit!")
 repeats_remaining = repeat - 1
 tx_ch.enable = True
 while True:
     sdr.sync_tx(buf, num_samples) # write to bladeRF
     print(repeats_remaining)
     if repeats_remaining > 0:
         repeats_remaining -= 1
     else:
         break
 
 print("Stopping transmit")
 tx_ch.enable = False

Se esperan algunos :code:`Pulse parada para el buffer` al final.

Para transmitir y recibir al mismo tiempo, debes usar hilos, y también puedes usar el ejemplo de Nuand. `txrx.py <https://github.com/Nuand/bladeRF/blob/624994d65c02ad414a01b29c84154260912f4e4f/host/examples/python/txrx/txrx.py>`_ que hace exactamente eso.

***********************************
Osciladores, PLL y calibración
***********************************

Todos los SDR de conversión directa (incluidos todos los SDR basados en AD9361 como USRP B2X0, Analog Devices Pluto y bladeRF) dependen de un único oscilador para proporcionar un reloj estable para el transceptor de RF. Cualquier compensación o fluctuación en la frecuencia producida por este oscilador se traducirá en compensación de frecuencia y fluctuación de frecuencia en la señal recibida o transmitida. Este oscilador está integrado, pero opcionalmente se puede "disciplinar" usando una onda cuadrada o sinusoidal independiente alimentada al bladeRF a través de un conector U.FL en la placa.

La placa bladeRF es una `Abracon VCTCXO <https://abracon.com/Oscillators/ASTX12_ASVTX12.pdf>`_ (controlado por voltaje
oscilador con compensación de temperatura) con una frecuencia de 38,4 MHz. El aspecto de "temperatura compensada" significa que está diseñado para ser estable en un amplio rango de temperaturas. El aspecto controlado por voltaje significa que se usa un nivel de voltaje para provocar ligeros ajustes en la frecuencia del oscilador, y en el bladeRF este voltaje es proporcionado por un convertidor digital a analógico (DAC) de 10 bits separado, como se muestra en verde en el bloque. diagrama a continuación. Esto significa que a través del software podemos hacer ajustes finos a la frecuencia del oscilador, y así es como calibramos (también conocido como recortamos) el VCTCXO del bladeRF. Afortunadamente, los bladeRF están calibrados en fábrica, como veremos más adelante en esta sección, pero si tiene el equipo de prueba disponible, siempre puede ajustar este valor, especialmente a medida que pasan los años y la frecuencia del oscilador cambia.

.. image:: ../_images/bladeRF-2.0-micro-Block-Diagram-4-oscillator.png
   :scale: 80 %
   :align: center 
   :alt: bladeRF 2.0 glamour shot

Cuando se utiliza una referencia de frecuencia externa (que puede ser casi cualquier frecuencia hasta 300 MHz), la señal de referencia se envía directamente al `Analog Devices ADF4002 <http://www.analog.com/en/adf4002>`_ PLL integrado la cuchillaRF. Este PLL se bloquea en la señal de referencia y envía una señal al VCTCXO (como se muestra en azul arriba) que es proporcional a la diferencia de frecuencia y fase entre la entrada de referencia (escalada) y la salida del VCTCXO. Una vez que el PLL está bloqueado, esta señal entre el PLL y el VCTCXO es un voltaje de CC de estado estable que mantiene la salida del VCTCXO en "exactamente" 38,4 MHz (suponiendo que la referencia fuera correcta) y bloqueada en fase con la entrada de referencia. Como parte del uso de una referencia externa, debe habilitar :code:`clock_ref` (ya sea a través de Python o CLI) y configurar la frecuencia de referencia de entrada (también conocida como :code:`refin_freq`), que es 10 MHz de forma predeterminada. Las razones para utilizar una referencia externa incluyen una mejor precisión de frecuencia y la capacidad de sincronizar múltiples SDR con la misma referencia.

Cada valor de ajuste de bladeRF VCTCXO DAC está calibrado en fábrica para estar dentro de 1 Hz a 38,4 MHz a temperatura ambiente, y puede ingresar su número de serie en `esta página <https://www.nuand.com/calibration/>`_ para ver cuál era el valor calibrado de fábrica (busque su número de serie en la placa o usando :code:`bladerf-tool probe`). Según Nuand, una placa nueva debería estar dentro de los 0,5 ppm y probablemente más cerca de los 0,1 ppm. Si tiene un equipo de prueba para medir la precisión de la frecuencia o desea configurarlo al valor de fábrica, puede usar los comandos:

.. code-block:: bash

 $ bladeRF-cli -i
 bladeRF> flash_init_cal 301 0x2049

intercambiando :code:`301` con el tamaño de su bladeRF y :code:`0x2049` con el formato hexadecimal de su valor de ajuste VCTCXO DAC. Debes realizar un ciclo de energía para que entre en vigor.

***********************************
Muestreo a 122 MHz
***********************************

Proximamente!

***********************************
Expansion de puertos
***********************************

El bladeRF 2.0 incluye un puerto de expansión mediante un conector BSH-030. ¡Más información sobre el uso de este puerto próximamente!

********************************
Lecturas Futuras
********************************

#. `bladeRF Wiki <https://github.com/Nuand/bladeRF/wiki>`_
#. `Nuand's txrx.py example <https://github.com/Nuand/bladeRF/blob/master/host/examples/python/txrx/txrx.py>`_
