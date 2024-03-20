.. _rtlsdr-chapter:

##################
RTL-SDR en Python
##################

El RTL-SDR es, con diferencia, el SDR más barato, alrededor de 30 dólares, y un excelente SDR para empezar. Si bien es solo de recepción y solo puede sintonizar hasta ~1,75 GHz, existen numerosas aplicaciones para las que se puede utilizar. En este capítulo, aprendemos cómo configurar el software RTL-SDR y utilizar su API Python.

.. image:: ../_images/rtlsdrs.svg
   :align: center 
   :target: ../_images/rtlsdrs.svg
   :alt: Example RTL-SDRs

********************************
Antecedentes RTL-SDR
********************************

El RTL-SDR surgió alrededor de 2010 cuando la gente descubrió que podían piratear dongles DVB-T de bajo costo que contenían el chip Realtek RTL2832U. DVB-T es un estándar de televisión digital utilizado principalmente en Europa, pero lo interesante del RTL2832U fue que se podía acceder directamente a las muestras de IQ sin procesar, lo que permitía utilizar el chip para construir un SDR de uso general y solo para recepción. 

El chip RTL2832U incluye el convertidor analógico a digital (ADC) y el controlador USB, pero debe emparejarse con un sintonizador de RF. Los chips sintonizadores populares incluyen Rafael Micro R820T, R828D y Elonics E4000. El rango de frecuencia sintonizable se basa en el chip sintonizador y suele estar entre 50 y 1700 MHz. La frecuencia de muestreo máxima, por otro lado, está determinada por el RTL2832U y el bus USB de su computadora, y suele ser de alrededor de 2,4 MHz sin perder demasiadas muestras. Tenga en cuenta que estos sintonizadores son de costo extremadamente bajo y tienen una sensibilidad de RF muy pobre, por lo que a menudo es necesario agregar un amplificador de bajo ruido (LNA) y un filtro de paso de banda para recibir señales débiles.

El RTL2832U siempre utiliza muestras de 8 bits, por lo que la máquina host recibirá dos bytes por muestra de IQ. Los RTL-SDR premium suelen venir con un oscilador con temperatura controlada (también conocido como TCXO) en lugar del oscilador de cristal más económico, que proporciona una mejor estabilidad de frecuencia. Otra característica opcional es una T de polarización (también conocida como diagonal-T), que es un circuito integrado que proporciona ~4,5 V CC en el conector SMA, que se utiliza para alimentar cómodamente un LNA externo u otros componentes de RF. Esta compensación de DC adicional se encuentra en el lado de RF del SDR, por lo que no interfiere con la operación de recepción básica.

Para aquellos interesados en la dirección de llegada (DOA) u otras aplicaciones de beamforming, el `KrakenSDR <https://www.crowdsupply.com/krakenrf/krakensdr>`_ es un SDR de fase coherente fabricado a partir de cinco RTL-SDR que comparten un oscilador y un reloj de muestra.

********************************
Configuración del software
********************************

Ubuntu (o Ubuntu con WSL)
#############################

En Ubuntu 20, 22 y otros sistemas basados en Debian, puede instalar el software RTL-SDR con el siguiente comando.

.. code-block:: bash

 sudo apt install rtl-sdr

Esto instalará la biblioteca librtlsdr y herramientas de línea de comandos como rtl_sdr, rtl_tcp, rtl_fm y rtl_test.

A continuación, instale el contenedor Python para librtlsdr usando:

.. code-block:: bash

 sudo pip install pyrtlsdr

Si está utilizando Ubuntu a través de WSL, en el lado de Windows descargue la última versión `Zadig <https://zadig.akeo.ie/>`_ y ejecútelo para instalar el controlador "WinUSB" para RTL-SDR (puede haber dos interfaces masivas, en cuyo caso instale "WinUSB" en ambas). Desenchufe y vuelva a enchufar el RTL-SDR una vez que finalice Zadig. 

A continuación, deberá reenviar el dispositivo USB RTL-SDR a WSL, primero instalando la última versión `usbipd utility msi <https://github.com/dorssel/usbipd-win/releases>`_ (esta guía supone que tiene usbipd-win 4.0.0 o superior), luego abre PowerShell en modo administrador y ejecuta:

.. code-block:: bash

    # (unplug RTL-SDR)
    usbipd list
    # (plug in RTL-SDR)
    usbipd list
    # (find the new device and substitute its index in the command below)
    usbipd bind --busid 1-5
    usbipd attach --wsl --busid 1-5

En el lado WSL, debería poder ejecutar :code:`lsusb` y ver un nuevo elemento llamado RTL2838 DVB-T o algo similar.

Si tiene problemas de permisos (por ejemplo, la prueba a continuación solo funciona cuando se usa :code:`sudo`), necesitará configurar reglas de udev. Primero ejecute :code:`lsusb` para encontrar el ID del RTL-SDR, luego cree el archivo :code:`/etc/udev/rules.d/10-rtl-sdr.rules` con el siguiente contenido, sustituyendo el idVendor e idProduct de tu RTL-SDR si el tuyo es diferente:

.. code-block::

 SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2838", MODE="0666"

Para actualizar udev, ejecute:

.. code-block:: bash

    sudo udevadm control --reload-rules
    sudo udevadm trigger

Es posible que también necesite desconectar y volver a conectar el RTL-SDR (para WSL tendrá que :code:`usbipd adjunto` nuevamente).

Windows
###################

Para usuarios windows, ver https://www.rtl-sdr.com/rtl-sdr-quick-start-guide/.  

********************************
Probando el RTL-SDR
********************************

Si la configuración del software funcionó, debería poder ejecutar la siguiente prueba, que sintonizará el RTL-SDR a la banda de radio FM y grabará 1 millón de muestras en un archivo llamado Recording.iq en /tmp.

.. code-block:: bash

    rtl_sdr /tmp/recording.iq -s 2e6 -f 100e6 -n 1e6

Si obtiene :code:`No supported devices found`, incluso cuando agrega un :code:`sudo` al principio, entonces Linux no puede ver el RTL-SDR en absoluto. Si funciona con :code:`sudo`, entonces es un problema de reglas de udev, intente reiniciar la computadora después de seguir las instrucciones de configuración de udev anteriores. Alternativamente, puedes usar :code:`sudo` para todo, incluso ejecutar Python.

Puede probar la capacidad de Python para ver RTL-SDR utilizando el siguiente script:

.. code-block:: python

 from rtlsdr import RtlSdr

 sdr = RtlSdr()
 sdr.sample_rate = 2.048e6 # Hz
 sdr.center_freq = 100e6   # Hz
 sdr.freq_correction = 60  # PPM
 sdr.gain = 'auto'
 
 print(len(sdr.read_samples(1024)))
 sdr.close()

lo cual debería mostrar:

.. code-block:: bash

 Found Rafael Micro R820T tuner
 [R82XX] PLL not locked!
 1024

********************************
Codigo RTL-SDR en Python
********************************

El código anterior puede considerarse un ejemplo de uso básico de RTL-SDR en Python. Las siguientes secciones entrarán en más detalles sobre las diversas configuraciones y trucos de uso.

Evitar fallas en RTL-SDR
###############################

Al final de nuestro script, o cuando hayamos terminado de tomar muestras del RTL-SDR, llamaremos a :code:`sdr.close()`, lo que ayudará a evitar que el RTL-SDR entre en un estado de falla en el que es necesario desconectarlo/volverlo a enchufar. Incluso usando close() todavía puede suceder, lo sabrás si el RTL-SDR se detiene durante la llamada read_samples(). Si esto sucede, deberá desconectar y volver a conectar el RTL-SDR y posiblemente reiniciar su computadora. Si está utilizando WSL, deberá volver a conectar el RTL-SDR mediante usbipd.

Configuración de ganancia
#########################

Al configurar :code:`sdr.gain = 'auto'` estamos habilitando el control automático de ganancia (AGC), lo que hará que el RTL-SDR ajuste la ganancia de recepción en función de las señales que recibe, intentando completar el 8- bit ADC sin saturarlo. Para muchas situaciones, como por ejemplo hacer un analizador de espectro, es útil mantener la ganancia en un valor constante, lo que significa que tenemos que configurar una ganancia manual. El RTL-SDR no tiene una ganancia infinitamente ajustable; puede ver la lista de valores de ganancia válidos usando :code:`print(sdr.valid_gains_db)`. Dicho esto, si lo configura en una ganancia que no está en esta lista, seleccionará automáticamente el valor permitido más cercano. Siempre puedes comprobar cuál está configurada la ganancia actual con :code:`print(sdr.gain)`. En el siguiente ejemplo, configuramos la ganancia en 49,6 dB y recibimos 4096 muestras, luego las trazamos en el dominio del tiempo:

.. code-block:: python

 from rtlsdr import RtlSdr
 import numpy as np
 import matplotlib.pyplot as plt
 
 sdr = RtlSdr()
 sdr.sample_rate = 2.048e6 # Hz
 sdr.center_freq = 100e6   # Hz
 sdr.freq_correction = 60  # PPM
 print(sdr.valid_gains_db)
 sdr.gain = 49.6
 print(sdr.gain)
 
 x = sdr.read_samples(4096)
 sdr.close()
 
 plt.plot(x.real)
 plt.plot(x.imag)
 plt.legend(["I", "Q"])
 plt.savefig("../_images/rtlsdr-gain.svg", bbox_inches='tight')
 plt.show()

.. image:: ../_images/rtlsdr-gain.svg
   :align: center 
   :target: ../_images/rtlsdr-gain.svg
   :alt: RTL-SDR manual gain example

Hay un par de cosas a tener en cuenta aquí. Las primeras muestras de ~2k no parecen tener mucha potencia de señal, porque representan transitorios. Se recomienda desechar las primeras 2k muestras de cada script, por ejemplo, usando :code:`sdr.read_samples(2048)` y no hacer nada con la salida. La otra cosa que notamos es que pyrtlsdr nos devuelve las muestras como flotantes, entre -1 y +1. Aunque utiliza un ADC de 8 bits y produce valores enteros, pyrtlsdr divide por 127,0 para nuestra conveniencia.

Frecuencias de muestreo permitidas
##################################

La mayoría de los RTL-SDR requieren que la frecuencia de muestreo se establezca entre 230 y 300 kHz o entre 900 y 3,2 MHz. Tenga en cuenta que es posible que las velocidades más altas, especialmente por encima de 2,4 MHz, no obtengan el 100% de las muestras a través de la conexión USB. Si le asigna una frecuencia de muestreo no compatible, simplemente regresará con el error :code:`rtlsdr.rtlsdr.LibUSBError: Error code -22: Could not set sample rate to 899000 Hz`. Al configurar una frecuencia de muestreo permitida, notará el mensaje de la consola que muestra la frecuencia de muestreo exacta; este valor exacto también se puede recuperar llamando a :code:`sdr.sample_rate`. Algunas aplicaciones pueden beneficiarse al utilizar un valor más exacto en los cálculos.

Como ejercicio, estableceremos la frecuencia de muestreo en 2,4 MHz y crearemos un espectrograma de la banda de radio FM:

.. code-block:: python

 # ...
 sdr.sample_rate = 2.4e6 # Hz
 # ...
 
 fft_size = 512
 num_rows = 500
 x = sdr.read_samples(2048) # get rid of initial empty samples
 x = sdr.read_samples(fft_size*num_rows) # get all the samples we need for the spectrogram
 spectrogram = np.zeros((num_rows, fft_size))
 for i in range(num_rows):
     spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)
 extent = [(sdr.center_freq + sdr.sample_rate/-2)/1e6,
             (sdr.center_freq + sdr.sample_rate/2)/1e6,
             len(x)/sdr.sample_rate, 0]
 plt.imshow(spectrogram, aspect='auto', extent=extent)
 plt.xlabel("Frequency [MHz]")
 plt.ylabel("Time [s]")
 plt.show()

.. image:: ../_images/rtlsdr-waterfall.svg
   :align: center 
   :target: ../_images/rtlsdr-waterfall.svg
   :alt: RTL-SDR waterfall (aka spectrogram) example

Configuración de PPM
####################

Para aquellos curiosos sobre la configuración de ppm, cada RTL-SDR tiene un pequeño desfase/error de frecuencia, debido al bajo costo de los chips sintonizadores y la falta de calibración. El desplazamiento de frecuencia debe ser relativamente lineal (no un cambio de frecuencia constante) en todo el espectro, por lo que podemos corregirlo ingresando un valor de PPM en partes por millón. Por ejemplo, si sintoniza 100 MHz y configura PPM en 25, la señal recibida aumentará en 100e6/1e6*25=2500 Hz. Las señales más estrechas tendrán un mayor impacto por error de frecuencia. Dicho esto, muchas señales modernas implican un paso de sincronización de frecuencia que corregirá cualquier compensación de frecuencia en el transmisor, el receptor o debido al desplazamiento Doppler.

********************************
Lecturas Futuras
********************************

#. `Página Acerca de de RTL-SDR.com <https://www.rtl-sdr.com/about-rtl-sdr/>`_
#. https://hackaday.com/2019/07/31/rtl-sdr-seven-years-later/
#. https://osmocom.org/projects/rtl-sdr/wiki/Rtl-sdr
