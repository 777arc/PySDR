.. _multipath-chapter:

###############################
Desvanecimiento Multicamino
###############################

En este capítulo presentamos el multitrayecto, un fenómeno de propagación que da como resultado que las señales lleguen al receptor por dos o más caminos, que experimentamos en los sistemas inalámbricos del mundo real. Hasta ahora sólo hemos hablado del "Canal AWGN", es decir, un modelo de canal inalámbrico donde la señal simplemente se suma al ruido, lo que en realidad sólo se aplica a las señales a través de un cable y algunos sistemas de comunicaciones por satélite.

*************************
Multicamino
*************************

Todos los canales inalámbricos realistas incluyen muchos "reflectores", dado que las señales de RF rebotan. Cualquier objeto entre o cerca del transmisor (Tx) o del receptor (Rx) puede causar caminos adicionales por los que viaja la señal. Cada camino experimenta un cambio de fase (retraso) y una atenuación (escalamiento de amplitud) diferentes. En el receptor, todos los caminos se suman. Pueden sumar de manera constructiva, destructiva o una combinación de ambas. A este concepto de múltiples rutas de señal lo llamamos "multicamino". Está la ruta de línea de visión (LOS) y luego todas las demás rutas. En el siguiente ejemplo, mostramos la ruta LOS y una única ruta que no es LOS:

.. image:: ../_images/multipath.svg
   :align: center 
   :target: ../_images/multipath.svg
   :alt: Simple depiction of multipath, showing the line-of-sight (LOS) path and a single multipath

Puede ocurrir una interferencia destructiva si no tienes suerte con la forma en que se combinan los caminos. Considere el ejemplo anterior con sólo dos caminos. Dependiendo de la frecuencia y la distancia exacta de las rutas, las dos rutas pueden recibirse con un desfase de 180 grados y aproximadamente con la misma amplitud, lo que hace que se anulen entre sí (como se muestra a continuación). Es posible que hayas aprendido sobre la interferencia constructiva y destructiva en la clase de física. En los sistemas inalámbricos, cuando las rutas se combinan de manera destructiva, llamamos a esta interferencia "desvanecimiento profundo" porque nuestra señal desaparece brevemente.

.. image:: ../_images/destructive_interference.svg
   :align: center 
   :target: ../_images/destructive_interference.svg

Las rutas también pueden acumularse de manera constructiva, provocando que se reciba una señal fuerte. Cada camino tiene un cambio de fase y una amplitud diferentes, que podemos visualizar en un gráfico en el dominio del tiempo llamado "perfil de retardo de potencia (PDP)":

.. image:: ../_images/multipath2.svg
   :align: center 
   :target: ../_images/multipath2.svg
   :alt: Multipath visualized including the power delay profile plot over time

La primera ruta, la más cercana al eje y, siempre será la ruta LOS (suponiendo que exista una) porque no hay forma de que otra ruta llegue al receptor más rápido que la ruta LOS. Normalmente, la magnitud disminuirá a medida que aumenta el retraso, ya que una ruta que tardó más en aparecer en el receptor habrá viajado más.

*************************
Desvanecimiento
*************************

Lo que tiende a suceder es que obtenemos una combinación de interferencia constructiva y destructiva, y cambia con el tiempo a medida que el Rx, el Tx o el entorno se mueven/cambian. Usamos el término "desvanecimiento" cuando nos referimos a los efectos de un canal multitrayecto que **cambia** con el tiempo. Es por eso que a menudo nos referimos a él como "desvanecimiento por trayectos múltiples"; en realidad es la combinación de interferencia constructiva/destructiva y un entorno cambiante. Lo que terminamos con una SNR que varía con el tiempo; Los cambios suelen ser del orden de milisegundos a microsegundos, dependiendo de qué tan rápido se mueve el Tx/Rx. A continuación se muestra un gráfico de SNR a lo largo del tiempo en milisegundos que demuestra el desvanecimiento por multicaminos.

.. image:: ../_images/multipath_fading.png
   :scale: 100 % 
   :align: center
   :alt: Multipath fading causes deep fades or nulls periodically where the SNR drops extremely low

Hay dos tipos de desvanecimiento desde la perspectiva del dominio del **tiempo**:

- **Desvanecimiento lento:** El canal no cambia dentro del valor de un paquete de datos. Es decir, un nulo profundo durante el desvanecimiento lento borrará todo el paquete.
- **Desvanecimiento rápido:** El canal cambia muy rápidamente en comparación con la longitud de un paquete. La corrección de errores directa, combinada con el *interleaving*, puede combatir el desvanecimiento rápido.

También hay dos tipos de desvanecimiento desde la perspectiva del dominio de **frecuencia**:

**Desvanecimiento de frecuencia selectiva**: La interferencia constructiva/destructiva cambia dentro del rango de frecuencia de la señal. Cuando tenemos una señal de banda ancha, abarcamos una amplia gama de frecuencias. Recuerde que la longitud de onda determina si es constructiva o destructiva. Bueno, si nuestra señal abarca un amplio rango de frecuencia, también abarca un amplio rango de longitud de onda (ya que la longitud de onda es la inversa de la frecuencia). En consecuencia, podemos obtener diferentes calidades de canal en diferentes partes de nuestra señal (en el dominio de la frecuencia). De ahí el nombre de desvanecimiento selectivo en frecuencia.

**Desvanecimiento plano**: ocurre cuando el ancho de banda de la señal es lo suficientemente estrecho como para que todas las frecuencias experimenten aproximadamente el mismo canal. Si hay un desvanecimiento profundo, toda la señal desaparecerá (mientras dure el desvanecimiento profundo).  

En la figura siguiente, la forma en :red:`rojo` muestra nuestra señal en el dominio de la frecuencia, y la línea curva negra muestra la condición actual del canal en función de la frecuencia. Debido a que la señal más estrecha experimenta las mismas condiciones de canal en toda la señal, experimenta un desvanecimiento plano. La señal más amplia está experimentando un desvanecimiento selectivo de frecuencia.

.. image:: ../_images/flat_vs_freq_selective.png
   :scale: 70 % 
   :align: center
   :alt: Flat fading vs frequency selective fading

A continuación se muestra un ejemplo de una señal de 16 MHz de ancho que se transmite continuamente. Hay varios momentos en el medio donde hay un período de tiempo que falta una señal. Este ejemplo muestra un desvanecimiento selectivo de frecuencia, que provoca agujeros en la señal que eliminan algunas frecuencias pero no otras.

.. image:: ../_images/fading_example.jpg
   :scale: 60 % 
   :align: center
   :alt: Example of frequency selective fading on a spectrogram (a.k.a. waterfall plot) showing smearing and a hole in the spectrogram where a deep null is
   
****************************************
Simulando el desvanecimiento de Rayleigh
****************************************

El desvanecimiento de Rayleigh se utiliza para modelar el desvanecimiento a lo largo del tiempo, cuando no existe una ruta LOS significativa. Cuando hay una ruta LOS dominante, el modelo de desvanecimiento de Rician se vuelve más adecuado, pero nos centraremos en Rayleigh. Tenga en cuenta que los modelos de Rayleigh y Rician no incluyen principalmente la pérdida de ruta entre el transmisor y el receptor (como la pérdida de ruta calculada como parte del presupuesto del enlace), ni ninguna sombra causada por objetos grandes. Su función es modelar el desvanecimiento por trayectos múltiples que se produce con el tiempo, como resultado del movimiento y la dispersión en el medio ambiente.

Hay mucha teoría que surge del modelo de desvanecimiento de Rayleigh, como expresiones para la tasa de cruce a nivel y la duración promedio del desvanecimiento. Pero el modelo de desvanecimiento de Rayleigh no nos dice directamente cómo simular realmente un canal usando el modelo. Para generar el desvanecimiento de Rayleigh en la simulación, tenemos que usar uno de los muchos métodos publicados, y en el siguiente ejemplo de Python usaremos el método de "suma de sinusoides" de Clarke.

Para generar un canal de desvanecimiento de Rayleigh en Python, primero debemos especificar el desplazamiento Doppler máximo, en Hz, que se basa en la rapidez con la que se mueve el transmisor y/o el receptor, denotado por :math:`\Delta v`.  Cuando la velocidad es pequeña en comparación con la velocidad de la luz, lo que siempre será el caso en las comunicaciones inalámbricas, el desplazamiento Doppler se puede calcular como:

.. math::

  f_D = \frac{\Delta v f_c}{c} 
  
donde :math:`c` es la velocidad de la luz, aproximadamente 3e8 m/s, y :math:`f_c` es la frecuencia portadora en la que se transmite.  

También elegimos cuántas sinusoides simular y no hay una respuesta correcta porque se basa en la cantidad de dispersores en el entorno, que en realidad nunca sabemos. Como parte de los cálculos, asumimos que la fase de la señal recibida de cada ruta es uniformemente aleatoria entre 0 y :math:`2\pi`.  El siguiente código simula un canal con desvanecimiento de Rayleigh utilizando el método de Clarke:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    # Simulation Params, feel free to tweak these
    v_mph = 60 # velocity of either TX or RX, in miles per hour
    center_freq = 200e6 # RF carrier frequency in Hz
    Fs = 1e5 # sample rate of simulation
    N = 100 # number of sinusoids to sum

    v = v_mph * 0.44704 # convert to m/s
    fd = v*center_freq/3e8 # max Doppler shift
    print("max Doppler shift:", fd)
    t = np.arange(0, 1, 1/Fs) # time vector. (start, stop, step)
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    for i in range(N):
        alpha = (np.random.rand() - 0.5) * 2 * np.pi
        phi = (np.random.rand() - 0.5) * 2 * np.pi
        x = x + np.random.randn() * np.cos(2 * np.pi * fd * t * np.cos(alpha) + phi)
        y = y + np.random.randn() * np.sin(2 * np.pi * fd * t * np.cos(alpha) + phi)

    # z is the complex coefficient representing channel, you can think of this as a phase shift and magnitude scale
    z = (1/np.sqrt(N)) * (x + 1j*y) # this is what you would actually use when simulating the channel
    z_mag = np.abs(z) # take magnitude for the sake of plotting
    z_mag_dB = 10*np.log10(z_mag) # convert to dB

    # Plot fading over time
    plt.plot(t, z_mag_dB)
    plt.plot([0, 1], [0, 0], ':r') # 0 dB
    plt.legend(['Rayleigh Fading', 'No Fading'])
    plt.axis([0, 1, -15, 5])
    plt.show()

Si tiene la intención de utilizar este modelo de canal como parte de una simulación más grande, simplemente multiplicaría la señal recibida por el número complejo. :code:`z`, representando un desvanecimiento plano. El valor :code:`z` Luego se actualizaría en cada paso. Esto significa que todos los componentes de frecuencia de la señal experimentan el mismo canal en un momento dado, por lo que **no** estarías simulando un desvanecimiento selectivo de frecuencia, que requiere una respuesta de impulso de canal de múltiples tomas en la que no entraremos en este capítulo. Si nos fijamos en la magnitud de :code:`z`, Podemos ver que Rayleigh se desvanece con el tiempo:

.. image:: ../_images/rayleigh.svg
   :align: center 
   :target: ../_images/rayleigh.svg
   :alt: Simulation of Rayleigh Fading

Tenga en cuenta los desvanecimientos profundos que ocurren brevemente, así como la pequeña fracción de tiempo en la que el canal realmente funciona mejor que si no hubiera ningún desvanecimiento.


*******************************************
Mitigar el desvanecimiento por multicaminos
*******************************************

En las comunicaciones modernas, hemos desarrollado formas de combatir el desvanecimiento por trayectos múltiples.

CDMA
#####

La telefonía móvil 3G utiliza una tecnología llamada acceso múltiple por división de código (CDMA). Con CDMA, se toma una señal de banda estrecha y se la distribuye en un ancho de banda amplio antes de transmitirla (utilizando una técnica de espectro ensanchado llamada DSSS). Bajo el desvanecimiento selectivo de frecuencia, es poco probable que todas las frecuencias estén en un nulo profundo al mismo tiempo. En el receptor, la dispersión se invierte y este proceso de reducción de la dispersión mitiga en gran medida una nulidad profunda.

.. image:: ../_images/cdma.png
   :scale: 100 % 
   :align: center 

OFDM 
#####

La telefonía celular 4G, WiFi y muchas otras tecnologías utilizan un esquema llamado multiplexación por división de frecuencia ortogonal (OFDM). OFDM utiliza algo llamado subportadoras, donde dividimos la señal en el dominio de la frecuencia en un montón de señales estrechas compactas. Para combatir el desvanecimiento por trayectos múltiples podemos evitar asignar datos a subportadoras que se encuentran en un desvanecimiento profundo, aunque requiere que el extremo receptor envíe información del canal al transmisor lo suficientemente rápido. También podemos asignar esquemas de modulación de alto orden a subportadoras con gran calidad de canal para maximizar nuestra velocidad de datos.





