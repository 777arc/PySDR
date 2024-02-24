.. _freq-domain-chapter:

########################
Dominio de la Frecuencia
########################

Este capítulo presenta el dominio de la frecuencia y cubre series de Fourier, transformada de Fourier, propiedades de Fourier, FFT, ventanas y espectrogramas, utilizando ejemplos de Python.  

Uno de los efectos secundarios más interesantes de aprender acerca de DSP y las comunicaciones inalámbricas es que también aprenderá a pensar en el dominio de la frecuencia. La experiencia de la mayoría de las personas *trabajando* en el dominio de la frecuencia se limita a ajustar las perillas de graves/medios/agudos en el sistema de audio de un automóvil. La experiencia de la mayoría de las personas al *ver* algo en el dominio de la frecuencia se limita a ver un ecualizador de audio, como este clip:

.. image:: ../_images/audio_equalizer.webp
   :align: center
   
Al final de este capítulo comprenderá lo que realmente significa el dominio de la frecuencia, cómo convertir entre tiempo y frecuencia (además de lo que sucede cuando lo hacemos) y algunos principios interesantes que usaremos a lo largo de nuestros estudios de DSP y SDR. Al final de este libro de texto, serás un maestro trabajando en el dominio de la frecuencia, ¡garantizado!

Primero, ¿por qué nos gusta observar señales en el dominio de la frecuencia? Bueno, aquí hay dos señales de ejemplo, mostradas tanto en el dominio del tiempo como en el de la frecuencia.

.. image:: ../_images/time_and_freq_domain_example_signals.png
   :scale: 40 %
   :align: center
   :alt: Two signals in the time domain may look like noise, but in the frequency domain we see additional features

Como puede ver, en el dominio del tiempo ambos parecen ruido, pero en el dominio de la frecuencia podemos ver características diferentes. Todo está en el dominio del tiempo en su forma natural; cuando muestreamos señales, las estaremos muestreando en el dominio del tiempo, porque no se puede muestrear *directamente* una señal en el dominio de la frecuencia. Pero lo interesante suele ocurrir en el dominio de la frecuencia. 

*****************
Series de Fourier
*****************

Los conceptos básicos del dominio de la frecuencia comienzan con la comprensión de que cualquier señal puede representarse mediante la suma de ondas sinusoidales. Cuando descomponemos una señal en sus componentes sinusoidales, la llamamos serie de Fourier. A continuación se muestra un ejemplo de una señal que se compone sólo de dos ondas sinusoidales:

.. image:: ../_images/summing_sinusoids.svg
   :align: center
   :target: ../_images/summing_sinusoids.svg
   :alt: Simple example of how a signal can be made up of multiple sinusoids, demonstrating the Fourier Series
   
Aquí hay otro ejemplo; La curva roja a continuación se aproxima a una onda en diente de sierra sumando hasta 10 ondas sinusoidales. Podemos ver que no es una reconstrucción perfecta: se necesitaría un número infinito de ondas sinusoidales para reproducir esta onda en dientes de sierra debido a las transiciones bruscas:

.. image:: ../_images/fourier_series_triangle.gif
   :scale: 70 %   
   :align: center
   :alt: Animation of the Fourier series decomposition of a triangle wave (a.k.a. sawtooth)
   
Algunas señales requieren más ondas sinusoidales que otras y algunas requieren una cantidad infinita, aunque siempre se pueden aproximar con un número limitado. Aquí hay otro ejemplo de una señal que se descompone en una serie de ondas sinusoidales:

.. image:: ../_images/fourier_series_arbitrary_function.gif
   :scale: 70 %   
   :align: center  
   :alt: Animation of the Fourier series decomposition of an arbitrary function made up of square pulses

Para comprender cómo podemos descomponer una señal en ondas sinusoidales o sinusoides, primero debemos revisar los tres atributos de una onda sinusoidal:

#. Amplitud
#. Frecuencia
#. Fase

**Amplitud** indica la "fuerza" de la onda, mientras que **frecuencia** es el número de ondas por segundo. **Fase** se utiliza para representar cómo se desplaza la onda sinusoidal en el tiempo, entre 0 y 360 grados (o 0 a :math:`2\pi`), pero debe ser relativo a algo para que tenga algún significado, como dos señales con la misma frecuencia que estén desfasadas 30 grados entre sí.

.. image:: ../_images/amplitude_phase_period.svg
   :align: center
   :target: ../_images/amplitude_phase_period.svg
   :alt: Reference diagram of amplitude, phase, and frequency of a sine wave (a.k.a. sinusoid)
   
En este punto, es posible que se haya dado cuenta de que una "señal" es esencialmente solo una función, generalmente representada "a lo largo del tiempo" (es decir, el eje x). Otro atributo que es fácil de recordar es **período**, que es el inverso de **frecuencia**. El **período** de una sinusoide es la cantidad de tiempo, en segundos, que tarda la onda en finalizar un ciclo. Por tanto, la unidad de frecuencia es 1/segundo o Hz.
   
Cuando descomponemos una señal en una suma de ondas sinusoidales, cada una tendrá una determinada **amplitud**, **fase** y **frecuencia**. La **amplitud** de cada onda sinusoidal nos dirá qué tan fuerte existía la **frecuencia** en la señal original. No te preocupes demasiado por la **fase** por ahora, aparte de darte cuenta de que la única diferencia entre sin() y cos() es un cambio de fase (cambio de tiempo).

Es más importante comprender el concepto subyacente que las ecuaciones reales para resolver una serie de Fourier, pero para aquellos que estén interesados en las ecuaciones, los remito a la explicación concisa de Wolfram: https://mathworld.wolfram.com/FourierSeries.html.  

***********************
Pares Tiempo-Frecuencia
***********************

Hemos establecido que las señales se pueden representar como ondas sinusoidales, que tienen varios atributos. Ahora, aprendamos a trazar señales en el dominio de la frecuencia. Mientras que el dominio del tiempo demuestra cómo cambia una señal con el tiempo, el dominio de la frecuencia muestra qué parte de una señal descansa en qué frecuencias. En lugar de que el eje x sea el tiempo, será la frecuencia. Podemos trazar una señal dada tanto en tiempo * como * en frecuencia. Veamos algunos ejemplos simples para comenzar.

Así es como se ve una onda sinusoidal, con frecuencia f, en el dominio del tiempo y la frecuencia:

.. image:: ../_images/sine-wave.png
   :scale: 70 % 
   :align: center
   :alt: The time-frequency Fourier pair of a sine wave, which is an impulse in the frequency domain

El dominio del tiempo debería resultarle muy familiar. Es una función oscilante. No te preocupes por en qué momento del ciclo comienza ni por cuánto dura. La conclusión es que la señal tiene una **frecuencia única**, por lo que vemos un único pico en el dominio de la frecuencia. Cualquiera que sea la frecuencia a la que oscile esa onda sinusoidal será donde veremos el pico en el dominio de la frecuencia. El nombre matemático para un pico como este se llama "impulso".

Ahora, ¿qué pasaría si tuviéramos un impulso en el dominio del tiempo? Imagine una grabación de sonido de alguien aplaudiendo o golpeando un clavo con un martillo. Este par tiempo-frecuencia es un poco menos intuitivo.

.. image:: ../_images/impulse.png
   :scale: 70 % 
   :align: center  
   :alt: The time-frequency Fourier pair of an impulse in the time domain, which is a horizontal line (all frequencies) in the frequency domain

Como podemos ver, un pico/impulso en el dominio del tiempo es plano en el dominio de la frecuencia y, en teoría, contiene todas las frecuencias. No existe un impulso teóricamente perfecto porque tendría que ser infinitamente corto en el dominio del tiempo. Al igual que la onda sinusoidal, no importa en qué parte del dominio del tiempo se produzca el impulso. Lo importante aquí es que los cambios rápidos en el dominio del tiempo dan como resultado que ocurran muchas frecuencias.

A continuación, veamos los gráficos en el dominio del tiempo y la frecuencia de una onda cuadrada:

.. image:: ../_images/square-wave.svg
   :align: center 
   :target: ../_images/square-wave.svg
   :alt: The time-frequency Fourier pair of a square wave, which is a sinc (sin(x)/x function) in the frequency domain

Este también es menos intuitivo, pero podemos ver que el dominio de la frecuencia tiene un pico fuerte, que resulta estar en la frecuencia de la onda cuadrada, pero hay más picos a medida que aumentamos en frecuencia. Se debe al cambio rápido en el dominio del tiempo, como en el ejemplo anterior. Pero su frecuencia no es uniforme. Tiene picos a intervalos y el nivel decae lentamente (aunque continuará para siempre). Una onda cuadrada en el dominio del tiempo tiene un patrón sin(x)/x en el dominio de la frecuencia (también conocida como función sinc).

¿Y ahora qué pasa si tenemos una señal constante en el dominio del tiempo? Una señal constante no tiene "frecuencia". Vamos a ver:

.. image:: ../_images/dc-signal.png
   :scale: 80 % 
   :align: center 
   :alt: The time-frequency Fourier pair of a DC signal, which is an impulse at 0 Hz in the frequency domain

Como no hay frecuencia, en el dominio de la frecuencia tenemos un pico a 0 Hz. Tiene sentido si lo piensas. El dominio de la frecuencia no estará "vacío" porque eso solo sucede cuando no hay señal presente (es decir, dominio del tiempo de 0). Llamamos a 0 Hz en el dominio de la frecuencia "DC", porque es causado por una señal DC en el tiempo (una señal constante que no cambia). Tenga en cuenta que si aumentamos la amplitud de nuestra señal de CC en el dominio del tiempo, el pico a 0 Hz en el dominio de la frecuencia también aumentará.

Más adelante aprenderemos qué significa exactamente el eje y en el gráfico del dominio de la frecuencia, pero por ahora puedes considerarlo como una especie de amplitud que te indica cuánta de esa frecuencia estaba presente en la señal del dominio del tiempo.
   
***********************
Transformada de Fourier
***********************

Matemáticamente, la "transformada" que utilizamos para pasar del dominio del tiempo al dominio de la frecuencia y viceversa se llama Transformada de Fourier. Se define de la siguiente manera:

.. math::
   X(f) = \int x(t) e^{-j2\pi ft} dt

Para una señal x(t) podemos obtener la versión en el dominio de la frecuencia, X(f), usando esta fórmula. Representaremos la versión en el dominio del tiempo de una función con x(t) o y(t), y la versión correspondiente en el dominio de la frecuencia con X(f) e Y(f). Tenga en cuenta la "t" para el tiempo y la "f" para la frecuencia. La "j" es simplemente la unidad imaginaria. Es posible que lo hayas visto como "i" en la clase de matemáticas de la escuela secundaria. Usamos "j" en ingeniería e informática porque "i" a menudo se refiere a corriente y en programación a menudo se usa como iterador.

Volver al dominio del tiempo desde la frecuencia es casi lo mismo, aparte de un factor de escala y un signo negativo:

.. math::
   x(t) = \frac{1}{2 \pi} \int X(f) e^{j2\pi ft} df

Tenga en cuenta que muchos libros de texto y otros recursos utilizan :math:`w` en lugar de :math:`2\pi f`.  :math:`w` es la frecuencia angular en radianes por segundo, mientras que :math:`f` es en Hz.  Todo lo que tienes que saber es que

.. math::
   \omega = 2 \pi f

Aunque añade termino :math:`2 \pi` para muchas ecuaciones, es más fácil seguir con la frecuencia en Hz. En última instancia, trabajará con Hz en su aplicación SDR.

La ecuación anterior para la Transformada de Fourier es la forma continua, que sólo verás en problemas de matemáticas. La forma discreta está mucho más cerca de lo que se implementa en el código:

.. math::
   X_k = \sum_{n=0}^{N-1} x_n e^{-\frac{j2\pi}{N}kn}
   
Tenga en cuenta que la principal diferencia es que reemplazamos la integral con una suma. El índice :math:`k` va desde 0 a N-1.  

Está bien si ninguna de estas ecuaciones significa mucho para ti. ¡En realidad no necesitamos usarlos directamente para hacer cosas interesantes con DSP y SDR!

********************************
Propiedades en Tiempo-Frecuencia
********************************

Anteriormente examinamos ejemplos de cómo aparecen las señales en el dominio del tiempo y en el dominio de la frecuencia. Ahora cubriremos cinco "propiedades de Fourier" importantes. Estas son propiedades que nos dicen que si hacemos ____ con nuestra señal en el dominio del tiempo, entonces ____ sucede con nuestra señal en el dominio de la frecuencia. Nos dará una idea importante del tipo de procesamiento de señales digitales (DSP) que realizaremos en las señales en el dominio del tiempo en la práctica.

1. Propiedad de linealidad:

.. math::
   a x(t) + b y(t) \leftrightarrow a X(f) + b Y(f)

Esta propiedad es probablemente la más fácil de entender. Si agregamos dos señales en el tiempo, entonces la versión en el dominio de la frecuencia también será la suma de las dos señales en el dominio de la frecuencia. También nos dice que si multiplicamos cualquiera de ellos por un factor de escala, el dominio de la frecuencia también escalará en la misma cantidad. La utilidad de esta propiedad será más evidente cuando sumamos múltiples señales.

2. Propiedad de desplazamiento en frecuencia:

.. math::
   e^{2 \pi j f_0 t}x(t) \leftrightarrow X(f-f_0)

El término a la izquierda de x(t) es lo que llamamos "sinusoide compleja" o "exponencial compleja". Por ahora, todo lo que necesitamos saber es que es esencialmente sólo una onda sinusoidal en la frecuencia :math:`f_0`.  Esta propiedad nos dice que si tomamos una señal :math:`x(t)` y lo multiplicamos por una onda sinusoidal, entonces en el dominio de la frecuencia obtenemos :math:`X(f)` excepto que desplazado por una cierta frecuencia, :math:`f_0`.  Este cambio de frecuencia puede ser más fácil de visualizar:

.. image:: ../_images/freq-shift.svg
   :align: center 
   :target: ../_images/freq-shift.svg
   :alt: Depiction of a frequency shift of a signal in the frequency domain

El cambio de frecuencia es parte integral del DSP porque querremos cambiar la frecuencia de las señales hacia arriba y hacia abajo por muchas razones. Esta propiedad nos dice cómo hacerlo (multiplicar por una onda sinusoidal). Aquí hay otra forma de visualizar esta propiedad:

.. image:: ../_images/freq-shift-diagram.svg
   :align: center
   :target: ../_images/freq-shift-diagram.svg
   :alt: Visualization of a frequency shift by multiplying by a sine wave or sinusoid
   
3. Propiedad de escalamiento en tiempo:

.. math::
   x(at) \leftrightarrow X\left(\frac{f}{a}\right)

En el lado izquierdo de la ecuación, podemos ver que estamos escalando nuestra señal x(t) en el dominio del tiempo. A continuación se muestra un ejemplo de una señal que se escala en el tiempo y luego qué sucede con las versiones en el dominio de la frecuencia de cada una.

.. image:: ../_images/time-scaling.svg
   :align: center
   :target: ../_images/time-scaling.svg
   :alt: Depiction of the time scaling Fourier transform property in both time and frequency domain

El escalado en el tiempo esencialmente reduce o expande la señal en el eje x. Lo que esta propiedad nos dice es que el escalamiento en el dominio del tiempo provoca un escalamiento inverso en el dominio de la frecuencia. Por ejemplo, cuando transmitimos bits más rápido tenemos que utilizar más ancho de banda. La propiedad ayuda a explicar por qué las señales de mayor velocidad de datos ocupan más ancho de banda/espectro. Si la escala de tiempo-frecuencia fuera proporcional en lugar de inversamente proporcional, los operadores de telefonía celular podrían transmitir todos los bits por segundo que quisieran sin pagar miles de millones por el espectro. Lamentablemente ese no es el caso.

Quienes ya estén familiarizados con esta propiedad pueden notar que falta un factor de escala; se omite por razones de simplicidad. A efectos prácticos, no hace ninguna diferencia.

4. Propiedad de convolución en el tiempo:

.. math::
   \int x(\tau) y(t-\tau) d\tau  \leftrightarrow X(f)Y(f)

Se llama propiedad de convolución porque en el dominio del tiempo estamos convolucionando x(t) e y(t). Es posible que aún no conozcas la operación de convolución, así que por ahora imagínala como una correlación cruzada, aunque profundizaremos en las convoluciones en :ref:`esta sección <convolution-section>`. Cuando convolucionamos señales en el dominio del tiempo, es equivalente a multiplicar las versiones en el dominio de la frecuencia de esas dos señales. Es muy diferente a sumar dos señales. Cuando agregas dos señales, como vimos, en realidad no sucede nada, simplemente sumas la versión en el dominio de la frecuencia. Pero cuando convolucionas dos señales, es como crear una tercera señal nueva a partir de ellas. La convolución es la técnica más importante en DSP, aunque primero debemos comprender cómo funcionan los filtros para comprenderla por completo.

Antes de continuar, para explicar brevemente por qué esta propiedad es tan importante, considere esta situación: tiene una señal que desea recibir y hay una señal de interferencia junto a ella.

.. image:: ../_images/two-signals.svg
   :align: center
   :target: ../_images/two-signals.svg
   
El concepto de enmascaramiento se usa mucho en programación, así que usémoslo aquí. ¿Qué pasaría si pudiéramos crear la máscara de abajo y multiplicarla por la señal de arriba para enmascarar la que no queremos?

.. image:: ../_images/masking.svg
   :align: center
   :target: ../_images/masking.svg

Normalmente realizamos operaciones DSP en el dominio del tiempo, así que utilicemos la propiedad de convolución para ver cómo podemos hacer este enmascaramiento en el dominio del tiempo. Digamos que x(t) es nuestra señal recibida. Sea Y(f) la máscara que queremos aplicar en el dominio de la frecuencia. Bueno, eso significa que y(t) es la representación en el dominio del tiempo de nuestra máscara, y si la convolucionamos con x(t), podemos "filtrar" la señal que no queremos.

.. tikz:: [font=\Large\bfseries\sffamily]
   \definecolor{babyblueeyes}{rgb}{0.36, 0.61, 0.83}
   \draw (0,0) node[align=center,babyblueeyes]           {E.g., our received signal};
   \draw (0,-4) node[below, align=center,babyblueeyes]   {E.g., the mask}; 
   \draw (0,-2) node[align=center,scale=2]{$\int x(\tau)y(t-\tau)d\tau \leftrightarrow X(f)Y(f)$};   
   \draw[->,babyblueeyes,thick] (-4,0) -- (-5.5,-1.2);
   \draw[->,babyblueeyes,thick] (2.5,-0.5) -- (3,-1.3);
   \draw[->,babyblueeyes,thick] (-2.5,-4) -- (-3.8,-2.8);
   \draw[->,babyblueeyes,thick] (3,-4) -- (5.2,-2.8);
   :xscale: 70

When we discuss filtering, the convolution property will make more sense.

5. Convolution in Frequency Property:

Lastly, I want to point out that the convolution property works in reverse, although we won't be using it as much as the time domain convolution:

.. math::
   x(t)y(t)  \leftrightarrow  \int X(g) Y(f-g) dg

There are other properties, but the above five are the most crucial to understand in my opinion.  Even though we didn't step through the proof for each property, the point is we use the mathematical properties to gain insight into what happens to real signals when we do analysis and processing.  Don't get caught up on the equations. Make sure you understand the description of each property.


******************************
Fast Fourier Transform (FFT)
******************************

Now back to the Fourier Transform. I showed you the equation for the discrete Fourier Transform, but what you will be using while coding 99.9% of the time will be the FFT function, fft().  The Fast Fourier Transform (FFT) is simply an algorithm to compute the discrete Fourier Transform.  It was developed decades ago, and even though there are variations on the implementation, it's still the reigning leader for computing a discrete Fourier transform. Lucky, considering they used "Fast" in the name.

The FFT is a function with one input and one output.  It converts a signal from time to frequency: 

.. image:: ../_images/fft-block-diagram.svg
   :align: center
   :target: ../_images/fft-block-diagram.svg
   :alt: FFT is a function with one input (time domain) and one output (frequency domain) 
   
We will only be dealing with 1 dimension FFTs in this textbook (2D is used for image processing and other applications). For our purposes, think of the FFT function as having one input: a vector of samples, and one output: the frequency domain version of that vector of samples.  The size of the output is always the same as the size of the input. If I feed 1,024 samples into the FFT, I will get 1,024 out.  The confusing part is that the output will always be in the frequency domain, and thus the "span" of the x-axis if we were to plot it doesn't change based on the number of samples in the time domain input.  Let's visualize that by looking at the input and output arrays, along with the units of their indices:

.. image:: ../_images/fft-io.svg
   :align: center
   :target: ../_images/fft-io.svg
   :alt: Reference diagram for the input (seconds) and output (bandwidth) format of the FFT function showing frequency bins and delta-t and delta-f

Because the output is in the frequency domain, the span of the x-axis is based on the sample rate, which we will cover next chapter.  When we use more samples for the input vector, we get a better resolution in the frequency domain (in addition to processing more samples at once).  We don't actually "see" more frequencies by having a larger input. The only way would be to increase the sample rate (decrease the sample period :math:`\Delta t`).

How do we actually plot this output?  As an example let's say that our sample rate was 1 million samples per second (1 MHz).  As we will learn next chapter, that means we can only see signals up to 0.5 MHz, regardless of how many samples we feed into the FFT.  The way the output of the FFT gets represented is as follows:

.. image:: ../_images/negative-frequencies.svg
   :align: center
   :target: ../_images/negative-frequencies.svg
   :alt: Introducing negative frequencies

It is always the case; the output of the FFT will always show :math:`\text{-} f_s/2` to :math:`f_s/2` where :math:`f_s` is the sample rate.  I.e., the output will always have a negative portion and positive portion.  If the input is complex, the negative and positive portions will be different, but if it's real then they will be identical. 

Regarding the frequency interval, each bin corresponds to :math:`f_s/N` Hz, i.e., feeding in more samples to each FFT will lead to more granular resolution in your output.  A very minor detail that can be ignored if you are new: mathematically, the very last index does not correspond to *exactly* :math:`f_s/2`, rather it's :math:`f_s/2 - f_s/N` which for a large :math:`N` will be approximately :math:`f_s/2`.

********************
Negative Frequencies
********************

What in the world is a negative frequency?  For now, just know that they have to do with using complex numbers (imaginary numbers)--there isn't really such thing as a "negative frequency" when it comes to transmitting/receiving RF signals, it's just a representation we use.  Here's an intuitive way to think about it.  Consider we tell our SDR to tune to 100 MHz (the FM radio band) and sample at a rate of 10 MHz.  In other words, we will view the spectrum from 95 MHz to 105 MHz.  Perhaps there are three signals present:

.. image:: ../_images/negative-frequencies2.svg
   :align: center
   :target: ../_images/negative-frequencies2.svg
   
Now, when the SDR gives us the samples, it will appear like this:

.. image:: ../_images/negative-frequencies3.svg
   :align: center
   :target: ../_images/negative-frequencies3.svg
   :alt: Negative frequencies are simply the frequencies below the center (a.k.a. carrier) frequency that the radio tuned to

Remember that we tuned the SDR to 100 MHz.  So the signal that was at about 97.5 MHz shows up at -2.5 MHz when we represent it digitally, which is technically a negative frequency.  In reality it's just a frequency lower than the center frequency.  This will make more sense when we learn more about sampling and obtain experience using our SDRs.

****************************
Order in Time Doesn't Matter
****************************
One last property before we jump into FFTs.  The FFT function sort of "mixes around" the input signal to form the output, which has a different scale and units. We are no longer in the time domain after all.  A good way to internalize this difference between domains is realizing that changing the order things happen in the time domain doesn't change the frequency components in the signal.  I.e., the FFT of the following two signals will both have the same two spikes because the signal is just two sine waves at different frequencies.  Changing the order the sine waves occur doesn't change the fact that they are two sine waves at different frequencies.

.. image:: ../_images/fft_signal_order.png
   :scale: 50 % 
   :align: center
   :alt: When performing an FFT on a set of samples, the order in time that different frequencies occurred within those samples doesn't change the resulting FFT output

Technically, the phase of the FFT values will change because of the time-shift of the sinusoids.  However, for the first several chapters of this textbook we will mostly be concerned with the magnitude of the FFT.

*******************
FFT in Python
*******************

Now that we have learned about what an FFT is and how the output is represented, let's actually look at some Python code and use Numpy's FFT function, np.fft.fft().  It is recommended that you use a full Python console/IDE on your computer, but in a pinch you can use the online web-based Python console linked at the bottom of the navigation bar on the left.

First we need to create a signal in the time domain.  Feel free to follow along with your own Python console. To keep things simple, we will make a simple sine wave at 0.15 Hz.  We will also use a sample rate of 1 Hz, meaning in time we sample at 0, 1, 2, 3 seconds, etc.

.. code-block:: python

 import numpy as np
 t = np.arange(100)
 s = np.sin(0.15*2*np.pi*t)

If we plot :code:`s` it looks like:

.. image:: ../_images/fft-python1.png
   :scale: 70 % 
   :align: center 

Next let's use Numpy's FFT function:

.. code-block:: python

 S = np.fft.fft(s)

If we look at :code:`S` we see it's an array of complex numbers:

.. code-block:: python

    S =  array([-0.01865008 +0.00000000e+00j, -0.01171553 -2.79073782e-01j,0.02526446 -8.82681208e-01j,  3.50536075 -4.71354150e+01j, -0.15045671 +1.31884375e+00j, -0.10769903 +7.10452463e-01j, -0.09435855 +5.01303240e-01j, -0.08808671 +3.92187956e-01j, -0.08454414 +3.23828386e-01j, -0.08231753 +2.76337148e-01j, -0.08081535 +2.41078885e-01j, -0.07974909 +2.13663710e-01j,...

Hint: regardless of what you’re doing, if you ever run into complex numbers, try calculating the magnitude and the phase and see if they make more sense.  Let's do exactly that, and plot the magnitude and phase.  In most languages, abs() is a function for magnitude of a complex number.  The function for phase varies, but in Python it's :code:`np.angle()`.

.. code-block:: python

 import matplotlib.pyplot as plt
 S_mag = np.abs(S)
 S_phase = np.angle(S)
 plt.plot(t,S_mag,'.-')
 plt.plot(t,S_phase,'.-')

.. image:: ../_images/fft-python2.png
   :scale: 80 % 
   :align: center 

Right now we aren't providing any x-axis to the plots, it's just the index of the array (counting up from 0).  Due to mathematical reasons, the output of the FFT has the following format:

.. image:: ../_images/fft-python3.svg
   :align: center
   :target: ../_images/fft-python3.svg
   :alt: Arrangement of the output of an FFT before doing an FFT shift
   
But we want 0 Hz (DC) in the center and negative freqs to the left (that's just how we like to visualize things).  So any time we do an FFT we need to perform an "FFT shift", which is just a simple array rearrangement operation, kind of like a circular shift but more of a "put this here and that there".  The diagram below fully defines what the FFT shift operation does:

.. image:: ../_images/fft-python4.svg
   :align: center
   :target: ../_images/fft-python4.svg
   :alt: Reference diagram of the FFT shift function, showing positive and negative frequencies and DC

For our convenience, Numpy has an FFT shift function, :code:`np.fft.fftshift()`.  Replace the np.fft.fft() line with:

.. code-block:: python

 S = np.fft.fftshift(np.fft.fft(s))

We also need to figure out the x-axis values/label.  Recall that we used a sample rate of 1 Hz to keep things simple.  That means the left edge of the frequency domain plot will be -0.5 Hz and the right edge will be 0.5 Hz.  If that doesn't make sense, it will after you get through the chapter on :ref:`sampling-chapter`.  Let's stick to that assumption that our sample rate was 1 Hz, and plot the FFT output's magnitude and phase with a proper x-axis label.  Here is the final version of this Python example and the output:

.. code-block:: python

 import numpy as np
 import matplotlib.pyplot as plt
 
 Fs = 1 # Hz
 N = 100 # number of points to simulate, and our FFT size
 
 t = np.arange(N) # because our sample rate is 1 Hz
 s = np.sin(0.15*2*np.pi*t)
 S = np.fft.fftshift(np.fft.fft(s))
 S_mag = np.abs(S)
 S_phase = np.angle(S)
 f = np.arange(Fs/-2, Fs/2, Fs/N)
 plt.figure(0)
 plt.plot(f, S_mag,'.-')
 plt.figure(1)
 plt.plot(f, S_phase,'.-')
 plt.show()

.. image:: ../_images/fft-python5.png
   :scale: 80 % 
   :align: center 

Note that we see our spike at 0.15 Hz, which is the frequency we used when creating the sine wave. So that means our FFT worked!  If we did not know the code used to generate that sine wave, but we were just given the list of samples, we could use the FFT to determine the frequency. The reason why we see a spike also at -0.15 Hz has to do with the fact it was a real signal, not complex, and we will get deeper into that later. 

******************************
Windowing
******************************

When we use an FFT to measure the frequency components of our signal, the FFT assumes that it's being given a piece of a *periodic* signal.  It behaves as if the piece of signal we provided continues to repeat indefinitely. It's as if the last sample of the slice connects back to the first sample.  It stems from the theory behind the Fourier Transform.  It means that we want to avoid sudden transitions between the first and last sample because sudden transitions in the time domain look like many frequencies, and in reality our last sample doesn't actually connect back to our first sample.  To put it simply: if we are doing an FFT of 100 samples, using :code:`np.fft.fft(x)`, we want :code:`x[0]` and :code:`x[99]` to be equal or close in value.

The way we make up for this cyclic property is through "windowing".  Right before the FFT, we multiply the slice of signal by a window function, which is just any function that tapers to zero on both ends.  That ensures the slice of signal will begin and end at zero and connect.  Common window functions include Hamming, Hanning, Blackman, and Kaiser.  When you don't apply any windowing, it's called using a "rectangular" window because it's like multiplying by an array of ones.   Here is what several window functions look like:

.. image:: ../_images/windows.svg
   :align: center
   :target: ../_images/windows.svg
   :alt: Windowing function in time and frequency domain of rectangular, hamming, hanning, bartlet, blackman, and kaiser windows

A simple approach for beginners is to just stick with a Hamming window, which can be created in Python with :code:`np.hamming(N)` where N is the number of elements in the array, which is your FFT size.  In the above exercise, we would apply the window right before the FFT. After the 2nd line of code we would insert:

.. code-block:: python

 s = s * np.hamming(100)

If you are afraid of choosing the wrong window, don't be.  The difference between Hamming, Hanning, Blackman, and Kaiser is very minimal compared to not using a window at all since they all taper to zero on both sides and solve the underlying problem.


*******************
FFT Sizing
*******************

The last thing to note is FFT sizing.  The best FFT size is always an order of 2 because of the way the FFT is implemented.  You can use a size that is not an order of 2, but it will be slower. Common sizes are between 128 and 4,096, although you can certainly go larger.  In practice we may have to process signals that are millions or billions of samples long, so we need to break up the signal and do many FFTs.  That means we will get many outputs. We can either average them up or plot them over time (especially when our signal is changing over time).  You don't have to put *every* sample of a signal through an FFT to get a good frequency domain representation of that signal. For example you could only FFT 1,024 out of every 100k samples in a signal and it will still probably look fine, as long as the signal is always on.

*********************
Spectrogram/Waterfall
*********************

A spectrogram is the plot that shows frequency over time.  It is simply a bunch of FFTs stacked together (vertically, if you want frequency on the horizontal axis).  We can also show it in real-time, often referred to as a waterfall.  A spectrum analyzer is the piece of equipment that shows this spectrogram/waterfall.  The diagram below shows how an array of IQ samples can be sliced up to form a spectrogram:

.. image:: ../_images/spectrogram_diagram.svg
   :align: center
   :target: ../_images/spectrogram_diagram.svg
   :alt: Spectrogram (a.k.a. waterfall) diagram showing how FFT slices are arrange/stacked to form a time-frequency plot

Because a spectrogram involves plotting 2D data, it's effectively a 3D plot, so we have to use a colormap to represent the FFT magntiudes, which are the "values" we want to plot.  Here is an example of a spectrogram, with frequency on the horizontal/x-axis and time on the vertical/y-axis.  Blue represents the lowest energy and red is the highest. We can see that there is a strong spike at DC (0 Hz) in the center with a varying signal around it.  Blue represents our noise floor.

.. image:: ../_images/waterfall.png
   :scale: 120 % 
   :align: center 

Remember, it's just rows of FFTs stacked on top of each other, each row is 1 FFT (technically, the magnitude of 1 FFT).  Be sure to time-slice your input signal in slices of your FFT size (e.g., 1024 samples per slice).   Before jumping into the code to produce a spectrogram, here is an example signal we will use, it is simply a tone in white noise:

.. code-block:: python

 import numpy as np
 import matplotlib.pyplot as plt
 
 sample_rate = 1e6
 
 # Generate tone plus noise
 t = np.arange(1024*1000)/sample_rate # time vector
 f = 50e3 # freq of tone
 x = np.sin(2*np.pi*f*t) + 0.2*np.random.randn(len(t))

Here is what it looks like in the time domain (first 200 samples):

.. image:: ../_images/spectrogram_time.svg
   :align: center
   :target: ../_images/spectrogram_time.svg

In Python we can generate a spectrogram as follows:

.. code-block:: python

 # simulate the signal above, or use your own signal
  
 fft_size = 1024
 num_rows = len(x) // fft_size # // is an integer division which rounds down
 spectrogram = np.zeros((num_rows, fft_size))
 for i in range(num_rows):
     spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)
 
 plt.imshow(spectrogram, aspect='auto', extent = [sample_rate/-2/1e6, sample_rate/2/1e6, 0, len(x)/sample_rate])
 plt.xlabel("Frequency [MHz]")
 plt.ylabel("Time [s]")
 plt.show()

Which should produce the following, which is not the most interesting spectrogram because there is no time-varying behavior.  There are two tones because we simulated a real signal, and real signals always have a negative PSD that matches the positive side.  For more interesting examples of spectrograms, checkout https://www.IQEngine.org!

.. image:: ../_images/spectrogram.svg
   :align: center
   :target: ../_images/spectrogram.svg

*********************
FFT Implementation
*********************

Even though NumPy has already implemented the FFT for us, it's nice to know the basics of how it works under the hood.  The most popular FFT algorithm is the Cooley-Tukey FFT algorithm, first invented around 1805 by Carl Friedrich Gauss and then later rediscovered and popularized by James Cooley and John Tukey in 1965.

The basic version of this algorithm works on power-of-two size FFTs, and is intended for complex inputs but can also work on real inputs.   The building block of this algorithm is known as the butterfly, which is essentially a N = 2 size FFT, consisting of two multiplies and two summations: 

.. image:: ../_images/butterfly.svg
   :align: center
   :target: ../_images/butterfly.svg
   :alt: Cooley-Tukey FFT algorithm butterfly radix-2

or

.. math::
   y_0 = x_0 + x_1 w^k_N

   y_1 = x_0 - x_1 w^k_N

where :math:`w^k_N = e^{j2\pi k/N}` are known as twiddle factors (:math:`N` is the size of the sub-FFT and :math:`k` is the index).  Note that the input and output is intended to be complex, e.g., :math:`x_0` might be 0.6123 - 0.5213j, and the sums/multiplies are complex.

The algorithm is recursive and breaks itself in half until all that is left is a series of butterflies, this is depicted below using a size 8 FFT:

.. image:: ../_images/butterfly2.svg
   :align: center
   :target: ../_images/butterfly2.svg
   :alt: Cooley-Tukey FFT algorithm size 8

Each column in this pattern is a set of operations that can be done in parallel, and :math:`log_2(N)` steps are performed, which is why the computational complexity of the FFT is O(:math:`N\log N`) while a DFT is O(:math:`N^2`).

For those who prefer to think in code rather than equations, the following shows a simple Python implementation of the FFT, along with an example signal consisting of a tone plus noise, to try the FFT out with.

.. code-block:: python

 import numpy as np
 import matplotlib.pyplot as plt
 
 def fft(x):
     N = len(x)
     if N == 1:
         return x
     twiddle_factors = np.exp(-2j * np.pi * np.arange(N//2) / N)
     x_even = fft(x[::2]) # yay recursion!
     x_odd = fft(x[1::2])
     return np.concatenate([x_even + twiddle_factors * x_odd,
                            x_even - twiddle_factors * x_odd])
 
 # Simulate a tone + noise
 sample_rate = 1e6
 f_offset = 0.2e6 # 200 kHz offset from carrier
 N = 1024
 t = np.arange(N)/sample_rate
 s = np.exp(2j*np.pi*f_offset*t)
 n = (np.random.randn(N) + 1j*np.random.randn(N))/np.sqrt(2) # unity complex noise
 r = s + n # 0 dB SNR
 
 # Perform fft, fftshift, convert to dB
 X = fft(r)
 X_shifted = np.roll(X, N//2) # equivalent to np.fft.fftshift
 X_mag = 10*np.log10(np.abs(X_shifted)**2)
 
 # Plot results
 f = np.linspace(sample_rate/-2, sample_rate/2, N)/1e6 # plt in MHz
 plt.plot(f, X_mag)
 plt.plot(f[np.argmax(X_mag)], np.max(X_mag), 'rx') # show max
 plt.grid()
 plt.xlabel('Frequency [MHz]')
 plt.ylabel('Magnitude [dB]')
 plt.show()


.. image:: ../_images/fft_in_python.svg
   :align: center
   :target: ../_images/fft_in_python.svg
   :alt: python implementation of fft example

For those interested in JavaScript and/or WebAssembly based implementations, check out the `WebFFT <https://github.com/IQEngine/WebFFT>`_ library for performing FFTs in web or NodeJS applications, it includes several implementations under the hood, and there is a `benchmarking tool <https://webfft.com>`_ used to compare the performance of each implementation.
