.. _iq-files-chapter:

###################
Archivos IQ y SigMF
###################

En todos nuestros ejemplos anteriores de Python almacenamos señales como matrices 1D NumPy de tipo "flotante complejo". En este capítulo aprenderemos cómo se pueden almacenar las señales en un archivo y luego volver a leerlas en Python, además de presentar el estándar SigMF. Almacenar datos de señales en un archivo es extremadamente útil; es posible que desee grabar una señal en un archivo para analizarla manualmente sin conexión, compartirla con un colega o crear un conjunto de datos completo.

*************************
Archivos Binarios
*************************

Recuerde que una señal digital en banda base es una secuencia de números complejos.

Ejemplo: [0,123 + j0,512, 0,0312 + j0,4123, 0,1423 + j0,06512, ...]

Estos números corresponden a [I+jQ, I+jQ, I+jQ, I+jQ, I+jQ, I+jQ, I+jQ, ...]

Cuando queremos guardar números complejos en un archivo, los guardamos en el formato IQIQIQIQIQIQIQIQ. Es decir, almacenamos un montón de flotantes en una fila, y cuando los volvemos a leer debemos separarlos nuevamente en [I+jQ, I+jQ, ...].

Si bien es posible almacenar los números complejos en un archivo de texto o csv, preferimos guardarlos en lo que se llama un "archivo binario" para ahorrar espacio. Con frecuencias de muestreo altas, sus grabaciones de señales podrían fácilmente ocupar varios GB y queremos utilizar la memoria lo más eficientemente posible. Si alguna vez abrió un archivo en un editor de texto y parecía incomprensible como en la captura de pantalla a continuación, probablemente era binario. Los archivos binarios contienen una serie de bytes y usted mismo debe realizar un seguimiento del formato. Los archivos binarios son la forma más eficiente de almacenar datos, suponiendo que se haya realizado toda la compresión posible. Debido a que nuestras señales generalmente aparecen como una secuencia aleatoria de flotantes, normalmente no intentamos comprimir los datos. Los archivos binarios se utilizan para muchas otras cosas, por ejemplo, programas compilados (llamados "binarios"). Cuando se utilizan para guardar señales, los llamamos "archivos IQ" binarios y utilizamos la extensión de archivo .iq.

.. image:: ../_images/binary_file.png
   :scale: 70 % 
   :align: center 

En Python, el tipo complejo predeterminado es np.complex128, que utiliza dos flotantes de 64 bits por muestra. Pero en DSP/SDR, tendemos a usar flotantes de 32 bits porque los ADC de nuestros SDR no tienen **esa** precisión para garantizar flotantes de 64 bits. En Python usaremos **np.complex64**, que usa dos flotantes de 32 bits. Cuando simplemente estás procesando una señal en Python, realmente no importa, pero cuando vas a guardar la matriz 1d en un archivo, primero debes asegurarte de que sea una matriz de np.complex64.

*************************
Ejemplos en Python
*************************

En Python, y numpy específicamente, usamos la función :code:`tofile()` para almacenar una matriz numpy en un archivo. Aquí hay un breve ejemplo de cómo crear una señal BPSK simple más ruido y guardarla en un archivo en el mismo directorio desde donde ejecutamos nuestro script:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    num_symbols = 10000

    x_symbols = np.random.randint(0, 2, num_symbols)*2-1 # -1 and 1's
    n = (np.random.randn(num_symbols) + 1j*np.random.randn(num_symbols))/np.sqrt(2) # AWGN with unity power
    r = x_symbols + n * np.sqrt(0.01) # noise power of 0.01
    print(r)
    plt.plot(np.real(r), np.imag(r), '.')
    plt.grid(True)
    plt.show()

    # Now save to an IQ file
    print(type(r[0])) # Check data type.  Oops it's 128 not 64!
    r = r.astype(np.complex64) # Convert to 64
    print(type(r[0])) # Verify it's 64
    r.tofile('bpsk_in_noise.iq') # Save to file


Ahora examine los detalles del archivo producido y verifique cuántos bytes tiene. Debería ser num_symbols * 8 porque usamos np.complex64, que son 8 bytes por muestra, 4 bytes por flotante (2 flotantes por muestra).

Usando un nuevo script de Python, podemos leer este archivo usando :code:`np.fromfile()`, al igual que:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    samples = np.fromfile('bpsk_in_noise.iq', np.complex64) # Read in file.  We have to tell it what format it is
    print(samples)

    # Plot constellation to make sure it looks right
    plt.plot(np.real(samples), np.imag(samples), '.')
    plt.grid(True)
    plt.show()

Un gran error es olvidar decirle a np.fromfile() el formato del archivo. Los archivos binarios no incluyen ninguna información sobre su formato. De forma predeterminada, np.fromfile() asume que está leyendo en una matriz de float64.

La mayoría de los otros lenguajes tienen métodos para leer archivos binarios, por ejemplo, en MATLAB puedes usar fread(). Para analizar visualmente un archivo RF, consulte la sección siguiente.

Si alguna vez te encuentras tratando con int16 (también conocidos como ints cortos), o cualquier otro tipo de datos para el que numpy no tenga un equivalente complejo, te verás obligado a leer las muestras como reales, incluso si en realidad son complejas. El truco consiste en leerlos como reales, pero luego intercalarlos nuevamente en el formato IQIQIQ... usted mismo; a continuación se muestran un par de formas diferentes de hacerlo:

.. code-block:: python

 samples = np.fromfile('iq_samples_as_int16.iq', np.int16).astype(np.float32).view(np.complex64)

or

.. code-block:: python

 samples = np.fromfile('iq_samples_as_int16.iq', np.int16)
 samples /= 32768 # convert to -1 to +1 (optional)
 samples = samples[::2] + 1j*samples[1::2] # convert to IQIQIQ...

********************************
Análisis visual de un archivo RF
********************************

Aunque aprendimos cómo crear nuestro propio diagrama de espectrograma en el capitulo  :ref:`freq-domain-chapter` , no hay nada mejor que utilizar un software ya creado. Cuando se trata de analizar grabaciones de RF sin tener que instalar nada, el `sitio web IQEngine <https://iqengine.org>`_ que es un conjunto de herramientas para analizar, procesar y compartir grabaciones de RF.

Para aquellos que quieran una aplicación de escritorio, también existe `inspectrum <https://github.com/miek/inspectrum>`_.  Inspectrum es una herramienta gráfica bastante simple pero poderosa para escanear visualmente un archivo RF, con un control preciso sobre el rango del mapa de colores y el tamaño FFT (cantidad de zoom). Puede mantener presionada la tecla Alt y usar la rueda de desplazamiento para desplazarse en el tiempo. Tiene cursores opcionales para medir el tiempo delta entre dos ráfagas de energía y la capacidad de exportar una porción del archivo RF a un archivo nuevo. Para la instalación en plataformas basadas en Debian como Ubuntu, utilice los siguientes comandos:

.. code-block:: bash

 sudo apt-get install qt5-default libfftw3-dev cmake pkg-config libliquid-dev
 git clone https://github.com/miek/inspectrum.git
 cd inspectrum
 mkdir build
 cd build
 cmake ..
 make
 sudo make install
 inspectrum

.. image:: ../_images/inspectrum.jpg
   :scale: 30 % 
   :align: center 
   
****************************
Valores máximos y saturación
****************************

Al recibir muestras de un SDR, es importante conocer el valor máximo de muestra. Muchos SDR generarán muestras como flotantes utilizando un valor máximo de 1,0 y un valor mínimo de -1,0. Otros SDR le darán muestras como números enteros, generalmente de 16 bits, en cuyo caso los valores máximo y mínimo serán +32767 y -32768 (a menos que se especifique lo contrario), y puede optar por dividirlos entre 32,768 para convertirlos en flotantes desde - 1,0 a 1,0. La razón para estar atento al valor máximo de su SDR se debe a la saturación: al recibir una señal extremadamente alta (o si la ganancia está demasiado alta), el receptor se "saturará" y truncará los valores altos a cualquiera que sea el valor máximo de muestra. Los ADC de nuestros SDR tienen un número limitado de bits. Al crear una aplicación SDR, es aconsejable comprobar siempre la saturación y, cuando esto suceda, debes indicarlo de alguna manera.

Una señal saturada se verá entrecortada en el dominio del tiempo, así:

.. image:: ../_images/saturated_time.png
   :scale: 30 % 
   :align: center
   :alt: Example of a saturated receiver where the signal is clipped

Debido a los cambios repentinos en el dominio del tiempo, debido al truncamiento, el dominio de la frecuencia puede verse borroso. En otras palabras, el dominio de la frecuencia incluirá características falsas; características que resultaron de la saturación y que en realidad no son parte de la señal, lo que puede desorientar a las personas al analizar una señal.

********************************
SigMF y Anotación de archivos IQ
********************************

Dado que el archivo IQ en sí no tiene ningún metadato asociado, es común tener un segundo archivo que contenga información sobre la señal, con el mismo nombre de archivo pero con .txt u otra extensión de archivo. Esto debe incluir, como mínimo, la frecuencia de muestreo utilizada para recopilar la señal y la frecuencia a la que se sintonizó el SDR. Después de analizar la señal, el archivo de metadatos podría incluir información sobre rangos de muestra de características interesantes, como ráfagas de energía. El índice de muestra es simplemente un número entero que comienza en 0 e incrementa cada muestra compleja. Si supiera que hay energía desde la muestra 492342 a la 528492, entonces podría leer el archivo y extraer esa parte de la matriz: :code:`samples[492342:528493]`.

Afortunadamente, ahora existe un estándar abierto que especifica un formato de metadatos utilizado para describir grabaciones de señales, conocido como `SigMF <https://github.com/gnuradio/SigMF>`_.  Al utilizar un estándar abierto como SigMF, varias partes pueden compartir grabaciones de RF más fácilmente y utilizar diferentes herramientas para operar en los mismos conjuntos de datos, como `IQEngine <https://iqengine.org/sigmf>`_.  También evita el "bitrot" de conjuntos de datos de RF donde los detalles de la captura se pierden con el tiempo debido a que los detalles de la grabación no se ubican con la grabación misma.

La forma más sencilla (y mínima) de utilizar el estándar SigMF para describir un archivo IQ binario que haya creado es cambiar el nombre del archivo .iq a .sigmf-data y crear un nuevo archivo con el mismo nombre pero con la extensión .sigmf-meta. y asegúrese de que el campo de tipo de datos en el metaarchivo coincida con el formato binario de su archivo de datos. Este metaarchivo es un archivo de texto sin formato lleno de json, por lo que simplemente puede abrirlo con un editor de texto y completarlo manualmente (más adelante discutiremos cómo hacerlo mediante programación). A continuación se muestra un archivo .sigmf-meta de ejemplo que puede utilizar como plantilla:

.. code-block::

 {
     "global": {
         "core:datatype": "cf32_le",
         "core:sample_rate": 1000000,
         "core:hw": "PlutoSDR with 915 MHz whip antenna",
         "core:author": "Art Vandelay",
         "core:version": "1.0.0"
     },
     "captures": [
         {
             "core:sample_start": 0,
             "core:frequency": 915000000
         }
     ],
     "annotations": []
 }

Note que :code:`core:cf32_le` indica que sus datos .sigmf son del tipo IQIQIQIQ... con flotantes de 32 bits, es decir, np.complex64 como usamos anteriormente. Consulte las especificaciones para otros tipos de datos disponibles, como si tiene datos reales en lugar de complejos o si utiliza enteros de 16 bits en lugar de flotantes para ahorrar espacio.

Aparte del tipo de datos, las líneas más importantes a completar son :code:`core:sample_rate` y :code:`core:frequency`.  Es una buena práctica introducir también información sobre el hardware.(:code:`core:hw`) utilizado para capturar la grabación, como el tipo SDR y la antena, así como una descripción de lo que se sabe sobre la(s) señal(es) en la grabación en :code:`core:description`.  El :code:`core:version` es simplemente la versión del estándar SigMF que se utiliza en el momento en que se creó el archivo de metadatos.

Si está capturando su grabación de RF desde Python, por ejemplo, utilizando la API de Python para su SDR, puede evitar tener que crear manualmente estos archivos de metadatos utilizando el paquete SigMF Python. Esto se puede instalar en un sistema operativo basado en Ubuntu/Debian de la siguiente manera:

.. code-block:: bash

 cd ~
 git clone https://github.com/gnuradio/SigMF.git
 cd SigMF
 sudo pip install .

El código Python para escribir el archivo .sigmf-meta para el ejemplo del comienzo de este capítulo, donde guardamos bpsk_in_noise.iq, se muestra a continuación:

.. code-block:: python

 import numpy as np
 import datetime as dt
 from sigmf import SigMFFile
 
 # <code from example>
 
 # r.tofile('bpsk_in_noise.iq')
 r.tofile('bpsk_in_noise.sigmf-data') # replace line above with this one
 
 # create the metadata
 meta = SigMFFile(
     data_file='example.sigmf-data', # extension is optional
     global_info = {
         SigMFFile.DATATYPE_KEY: 'cf32_le',
         SigMFFile.SAMPLE_RATE_KEY: 8000000,
         SigMFFile.AUTHOR_KEY: 'Your name and/or email',
         SigMFFile.DESCRIPTION_KEY: 'Simulation of BPSK with noise',
         SigMFFile.VERSION_KEY: sigmf.__version__,
     }
 )
 
 # create a capture key at time index 0
 meta.add_capture(0, metadata={
     SigMFFile.FREQUENCY_KEY: 915000000,
     SigMFFile.DATETIME_KEY: dt.datetime.utcnow().isoformat()+'Z',
 })
 
 # check for mistakes and write to disk
 meta.validate()
 meta.tofile('bpsk_in_noise.sigmf-meta') # extension is optional

Simplemente reemplace :code:`8000000` y :code:`915000000` con las variables que utilizó para almacenar la frecuencia de muestreo y la frecuencia central respectivamente.

Para leer una grabación SigMF en Python, utilice el siguiente código. En este ejemplo, los dos archivos SigMF deben denominarse :code:`bpsk_in_noise.sigmf-meta` y :code:`bpsk_in_noise.sigmf-data`.

.. code-block:: python

 from sigmf import SigMFFile, sigmffile
 
 # Load a dataset
 filename = 'bpsk_in_noise'
 signal = sigmffile.fromfile(filename)
 samples = signal.read_samples().view(np.complex64).flatten()
 print(samples[0:10]) # lets look at the first 10 samples
 
 # Get some metadata and all annotations
 sample_rate = signal.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
 sample_count = signal.sample_count
 signal_duration = sample_count / sample_rate

Para más detalles consulte `the SigMF documentation <https://github.com/gnuradio/SigMF>`_.

Una pequeña ventaja para quienes hayan leído hasta aquí; El logotipo de SigMF en realidad se almacena como una grabación de SigMF y cuando la señal se traza como una constelación (gráfico IQ) a lo largo del tiempo, produce la siguiente animación:

.. image:: ../_images/sigmf_logo.gif
   :scale: 100 %   
   :align: center
   :alt: The SigMF logo animation

El código Python utilizado para leer el archivo del logotipo (ubicado `aqui <https://github.com/gnuradio/SigMF/tree/master/logo>`_) y produzca el gif animado que se muestra a continuación, para aquellos curiosos:

.. code-block:: python

 import numpy as np
 import matplotlib.pyplot as plt
 import imageio
 from sigmf import SigMFFile, sigmffile
 
 # Load a dataset
 filename = 'sigmf_logo' # assume its in the same directory as this script
 signal = sigmffile.fromfile(filename)
 samples = signal.read_samples().view(np.complex64).flatten()
 
 # Add zeros to the end so its clear when the animation repeats
 samples = np.concatenate((samples, np.zeros(50000)))
 
 sample_count = len(samples)
 samples_per_frame = 5000
 num_frames = int(sample_count/samples_per_frame)
 filenames = []
 for i in range(num_frames):
     print("frame", i, "out of", num_frames)
     # Plot the frame
     fig, ax = plt.subplots(figsize=(5, 5))
     samples_frame = samples[i*samples_per_frame:(i+1)*samples_per_frame]
     ax.plot(np.real(samples_frame), np.imag(samples_frame), color="cyan", marker=".", linestyle="None", markersize=1)
     ax.axis([-0.35,0.35,-0.35,0.35]) # keep axis constant
     ax.set_facecolor('black') # background color
     
     # Save the plot to a file
     filename = '/tmp/sigmf_logo_' + str(i) + '.png'
     fig.savefig(filename, bbox_inches='tight')
     filenames.append(filename)
 
 # Create animated gif
 images = []
 for filename in filenames:
     images.append(imageio.imread(filename))
 imageio.mimsave('/tmp/sigmf_logo.gif', images, fps=20)



