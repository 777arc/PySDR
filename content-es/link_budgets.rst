.. _link-budgets-chapter:

#######################
Presupuesto de enlace
#######################

Este capítulo cubre los presupuestos de enlace, una gran parte del cual consiste en comprender la potencia de transmisión/recepción, la pérdida de camino, la ganancia de la antena, el ruido y la SNR. Terminamos construyendo un ejemplo de presupuesto de enlace para ADS-B, que son señales transmitidas por aviones comerciales para compartir su posición y otra información.

*************************
Introducción
*************************

El proposito de un presupuesto de enlace es contabilizar todas las ganancias y pérdidas del transmisor al receptor en un sistema de comunicación. El presupuesto de enlace describe una dirección del enlace inalámbrico. La mayoría de los sistemas de comunicaciones son bidireccionales, por lo que debe haber un presupuesto separado para el enlace ascendente y el enlace descendente. El "resultado" del presupuesto del enlace le indicará aproximadamente cuánta relación señal-ruido (abreviada como SNR, que utiliza este libro de texto, o S/N) debe esperar tener en su receptor. Se necesitarían más análisis para comprobar si esa SNR es lo suficientemente alta para su aplicación.

Se estudian los presupuestos de enlace no con el propósito de poder hacer un presupuesto de enlace para alguna situación, sino para aprender y desarrollar un punto de vista de la capa de sistema de las comunicaciones inalámbricas.

Primero cubriremos el presupuesto de potencia de la señal recibida, luego el presupuesto de potencia del ruido y finalmente combinaremos los dos para encontrar la SNR (potencia de la señal dividida por la potencia del ruido).

***********************************
Presupuesto de potencia de la señal
***********************************

A continuación se muestra el diagrama más básico de un enlace inalámbrico genérico. En este capítulo nos centraremos en una dirección, es decir, de un transmisor (Tx) a un receptor (Rx). Para un sistema dado, conocemos la potencia de *transmisión*; suele ser una configuración en el transmisor. ¿Cómo calculamos la potencia *recibida* en el receptor?

.. image:: ../_images/tx_rx_system.svg
   :align: center 
   :target: ../_images/tx_rx_system.svg

Necesitamos cuatro parámetros del sistema para determinar la potencia recibida, que se proporcionan a continuación con sus abreviaturas comunes. En este capítulo se profundizará en cada uno de ellos.

- **Pt** - Potencia de transmisión
- **Gt** - Ganancia de la antena transmisora
- **Gr** - Gananacia de la antena receptora
- **Lp** - Perdida de camino entre Tx y Rx (i.e., cuanta perdida de la señal inalámbrica existe)

.. image:: ../_images/tx_rx_system_params.svg
   :align: center 
   :target: ../_images/tx_rx_system_params.svg
   :alt: Parameters within a link budget depicted

Potencia de transmisión
#######################

La potencia de transmisión es bastante sencilla; será un valor en vatios, dBW o dBm (recuerde que dBm es una abreviatura de dBmW). Cada transmisor tiene uno o más amplificadores y la potencia de transmisión es principalmente función de esos amplificadores. Una analogía con la potencia de transmisión sería la potencia nominal en vatios de una bombilla: cuanto mayor sea la potencia, más luz transmitirá la bombilla. A continuación se muestran ejemplos de potencia de transmisión aproximada para diferentes tecnologías:

==================  =====  =======
\                       Potencia    
------------------  --------------
Bluetooth           10 mW  -20 dBW   
WiFi                100mW  -10 dBW
Estación Base LTE   1W       0 dBW
Estación FM         10kW    40 dBW
==================  =====  =======

Ganancia de antenas
#####################

Las ganancias de las antenas de transmisión y recepción son cruciales para calcular los presupuestos de enlace. ¿Cuál es la ganancia de la antena, te preguntarás? Indica la directividad de la antena. Es posible que vea que se refiere a esto como ganancia de potencia de antena, pero no deje que eso lo engañe, la única forma de que una antena tenga una ganancia mayor es dirigir la energía en una región más enfocada.

Las ganancias se representarán en dB (sin unidades); siéntase libre de aprender o recordar por qué dB no tiene unidades para nuestro escenario en el capitulo :ref:`noise-chapter` .  Normalmente, las antenas son omnidireccionales, lo que significa que su potencia irradia en todas las direcciones, o direccionales, lo que significa que su potencia irradia en una dirección específica. Si son omnidireccionales su ganancia será de 0 dB a 3 dB. Una antena direccional tendrá una ganancia más alta, generalmente de 5 dB o más, y hasta 60 dB aproximadamente.

.. image:: ../_images/antenna_gain_patterns.png
   :scale: 80 % 
   :align: center 

When a directional antenna is used, it must be either installed facing the correct direction or attached to a mechanical gimbal. It could also be a phased array, which can be electronically steered (i.e., by software).

.. image:: ../_images/antenna_steering.png
   :scale: 80 % 
   :align: center 
   
Las antenas omnidireccionales se utilizan cuando no es posible apuntar en la dirección correcta, como su teléfono celular y su computadora portátil. En 5G, los teléfonos pueden funcionar en bandas de frecuencia más altas como 28 GHz (Verizon) y 39 GHz (AT&T) utilizando una serie de antenas y dirección electrónica del haz.

En un presupuesto de enlace, debemos suponer que cualquier antena direccional, ya sea de transmisión o recepción, apunta en la dirección correcta. Si no apunta correctamente, nuestro presupuesto de enlace no será preciso y podría haber una pérdida de comunicación (por ejemplo, la antena parabólica de su techo es golpeada por una pelota de baloncesto y se mueve). En general, nuestros presupuestos de enlaces asumen circunstancias ideales y agregan una pérdida miscelánea para tener en cuenta factores del mundo real.

Perdida de camino
#####################

A medida que una señal se mueve a través del aire (o el vacío), su fuerza se reduce. Imagínese sostener un pequeño panel solar frente a una bombilla. Cuanto más lejos esté el panel solar, menos energía absorberá de la bombilla. **Flujo** es un término en física y matemáticas, definido como "cuánta de pasa por ti". Para nosotros, es la cantidad de campo electromagnético que pasa a nuestra antena receptora. Queremos saber cuánta potencia se pierde en una distancia determinada.

.. image:: ../_images/flux.png
   :scale: 80 % 
   :align: center 

La pérdida de camino en el espacio libre (FSPL) nos indica la pérdida de camino cuando no hay obstáculos en una distancia determinada. En su forma general, :math:`\mathrm{FSPL} = ( 4\pi d / \lambda )^2`. Googlee la fórmula de transmisión de Friis para más información. (Dato curioso: las señales encuentran una impedancia de 377 ohmios al moverse a través del espacio libre). Para generar presupuestos de enlace, podemos usar esta misma ecuación pero convertida a dB:

.. math::
 \mathrm{FSPL}_{dB} = 20 \log_{10} d + 20 \log_{10} f - 147.55 \left[ dB \right]

En los presupuestos de enlaces aparecerán en dB, sin unidades porque es una pérdida.  :math:`d` está en metros y es la distancia entre el transmisor y el receptor.  :math:`f` está en Hz y es la frecuencia portadora. Sólo hay un problema con esta sencilla ecuación; No siempre tendremos espacio libre entre el transmisor y el receptor. Las frecuencias rebotan mucho en interiores (la mayoría de las frecuencias pueden atravesar paredes, pero no metales o mampostería gruesa). Para estas situaciones existen varios modelos de espacio no libre. Uno común para ciudades y áreas suburbanas (por ejemplo, celular) es el modelo Okumura-Hata:

.. math::
 L_{path} = 69.55 + 26.16 \log_{10} f - 13.82 \log_{10} h_B - C_H + \left[ 44.9 - 6.55 \log_{10} h_B \right] \log_{10} d

donde :math:`L_{path}` la perdida de camino es en dB, :math:`h_B` es la altura de la antena transmisora sobre el nivel del suelo en metros, :math:`f` es la frecuencia portadora en MHz, :math:`d` es la distancia entre Tx y Rx en Km, y :math:`C_H` se denomina "factor de corrección alta de la antena" y se define en función del tamaño de la ciudad y el rango de frecuencia de la portadora:

:math:`C_H` para ciudades pequeñas/medianas:

.. math::
 C_H = 0.8 + (1.1 \log_{10} f - 0.7 ) h_M - 1.56 \log_{10} f

:math:`C_H` para grandes ciudades cuando :math:`f` está por debajo de 200 MHz:

.. math::
 C_H = 8.29 ( log_{10}(1.54 h_M))^2 - 1.1
 
:math:`C_H` para grandes ciudades cuando :math:`f` está por encima de 200 MHz pero menos de 1,5 GHz:

.. math::
 C_H = 3.2 ( log_{10}(11.75 h_M))^2 - 4.97

donde :math:`h_M` es la altura de la antena receptora sobre el nivel del suelo en metros.

No se preocupe si el modelo anterior de Okumura-Hata le parece confuso; Aquí se muestra principalmente para demostrar cómo los modelos de pérdida de trayectoria fuera del espacio libre son mucho más complicados que nuestra simple ecuación FSPL. El resultado final de cualquiera de estos modelos es un número único que podemos usar para la porción de pérdida de ruta de nuestro presupuesto de enlace. Seguiremos usando FSPL durante el resto de este capítulo.

Clases de Pérdidas
#####################

En nuestro presupuesto de enlaces también queremos tener en cuenta diversos tipos de pérdidas. Los agruparemos en un solo término, generalmente entre 1 y 3 dB. Ejemplos de tipos de pérdidas:

- Pérdidas por cable
- Pérdidas Atmosfericas
- Imperfecciones en la orientación de la antena
- Precipitación

El siguiente gráfico muestra la pérdida atmosférica en dB/km sobre la frecuencia (normalmente estaremos < 40 GHz). Si se toma un tiempo para comprender el eje y, verá que las comunicaciones de corto alcance por debajo de 40 GHz **y** menos de 1 km tienen 1 dB o menos de pérdida atmosférica y, por lo tanto, generalmente lo ignoramos. Cuando realmente entra en juego la pérdida atmosférica es en las comunicaciones por satélite, donde la señal tiene que viajar muchos kilómetros a través de la atmósfera.

.. image:: ../_images/atmospheric_attenuation.svg
   :align: center 
   :target: ../_images/atmospheric_attenuation.svg
   :alt: Plot of atmospheric attenuation in dB/km over frequency showing the spikes from H2O (water) and O2 (oxygen)

Ecuación de potencia de señal
#############################

Ahora es el momento de juntar todas estas ganancias y pérdidas para calcular la potencia de nuestra señal en el receptor. :math:`P_r`:

.. math::
 P_r = P_t + G_t + G_r - L_p - L_{misc} \quad \mathrm{dBW}

En general, es una ecuación fácil. Sumamos las ganancias y las pérdidas. Es posible que algunos ni siquiera lo consideren una ecuación. Generalmente mostramos las ganancias, pérdidas y el total en una tabla, similar a la contabilidad, como esta:

.. list-table::
   :widths: 15 10
   :header-rows: 0
   
   * - Pt = 1.0 W
     - 0 dBW
   * - Gt = 100
     - 20.0 dB
   * - Gr = 1
     - 0 dB
   * - Lp
     - -162.0 dB
   * - Lmisc
     - -1.0 dB
   * - **Pr**
     - **-143.0 dBW**

PIRE
#####

Como comentario breve, es posible que vea la métrica de potencia radiada isotrópica efectiva (PIRE), que se define como :math:`P_t + G_t - L_{cable}` y en unidades de dBW. Sumando la potencia de transmisión con la ganancia de la antena de transmisión y restando las pérdidas del cable del lado de transmisión, obtenemos una cifra útil que representa la potencia "hipotética" que tendría que radiar una antena isotrópica (omnidireccional perfecta) para dar la misma intensidad de señal. **en la dirección del haz principal de la antena**. Esta última parte se enfatiza porque cualquier antena con alta ganancia (:math:`G_t`) sólo da esa alta ganancia cuando se apunta correctamente. Entonces, suponiendo que esté bien orientado, la PIRE le brinda todo lo que necesita saber sobre el lado de transmisión del presupuesto del enlace y, por lo tanto, es una métrica que a menudo se encuentra en hojas de datos de transmisores direccionales, como estaciones terrestres satelitales (generalmente en forma de "máx. PIRE").

********************************
Presupuesto de potencia de ruido
********************************

Ahora que conocemos la potencia de la señal recibida, cambiemos de tema al ruido recibido, ya que, después de todo, necesitamos ambos para calcular la SNR. Podemos encontrar ruido recibido con un presupuesto de energía de estilo similar.

Ahora es un buen momento para hablar sobre dónde entra el ruido en nuestro enlace de comunicaciones. Respuesta: **¡En el receptor!** La señal no se corrompe con ruido hasta que vamos a recibirla. ¡Es *extremadamente* importante entender este hecho! Muchos estudiantes no lo internalizan del todo y, como resultado, terminan cometiendo un error tonto. No hay ruido flotando a nuestro alrededor en el aire. El ruido proviene del hecho de que nuestro receptor tiene un amplificador y otros componentes electrónicos que no son perfectos y no están a 0 grados Kelvin (K).

Una formulación popular y sencilla para el presupuesto de ruido utiliza el enfoque "kTB":

.. math::
 P_{noise} = kTB

- :math:`k` – Constante de Boltzmann = 1,38 x 10-23 J/K = **-228,6 dBW/K/Hz**. Para cualquiera que tenga curiosidad, la constante de Boltzmann es una constante física que relaciona la energía cinética promedio de las partículas en un gas con la temperatura del gas.
- :math:`T` – Temperatura de ruido del sistema en K (¿alguien tiene crioenfriadores?), basada en gran medida en nuestro amplificador. Este es el término que resulta más difícil de encontrar, y suele ser muy aproximado. Es posible que pague más por un amplificador con una temperatura de ruido más baja. 
- :math:`B` – Ancho de banda de la señal en Hz, suponiendo que filtre el ruido alrededor de su señal. Entonces, una señal de enlace descendente LTE de 10 MHz de ancho tendrá :math:`B` establecido en 10 MHz o 70 dBHz.

Multiplicar (o sumar dB) kTB da nuestra potencia de ruido, es decir, el término inferior de nuestra ecuación SNR.

*************************
SNR
*************************

Ahora que tenemos ambos números, podemos tomar la relación para encontrar la SNR (consulte el capitulo :ref:`noise-chapter` para más información sobre SNR):

.. math::
   \mathrm{SNR} = \frac{P_{signal}}{P_{noise}}

.. math::
   \mathrm{SNR_{dB}} = P_{signal\_dB} - P_{noise\_dB}

Normalmente buscamos una SNR > 10 dB, aunque realmente depende de la aplicación. En la práctica, la SNR se puede verificar observando la FFT de la señal recibida o calculando la potencia con y sin la señal presente (varianza de recuperación = potencia). Cuanto mayor sea la SNR, más bits por símbolo podrá gestionar sin demasiados errores.

***************************************
Ejemplo de presupuesto de enlace: ADS-B
***************************************

La transmisión-vigilancia dependiente automática (ADS-B) es una tecnología utilizada por las aeronaves para transmitir señales que comparten su posición y otros estados con las estaciones terrestres de control del tráfico aéreo y otras aeronaves. ADS-B es automático porque no requiere piloto ni entrada externa; Depende de los datos del sistema de navegación de la aeronave y de otras computadoras. Los mensajes no están cifrados (¡sí!). El equipo ADS-B es actualmente obligatorio en partes del espacio aéreo australiano, mientras que Estados Unidos exige que algunas aeronaves estén equipadas, según el tamaño.

.. image:: ../_images/adsb.jpg
   :scale: 120 % 
   :align: center 
   
La Capa Física (PHY) de ADS-B tiene las siguientes características:

- Transmitido en 1.090 MHz
- Ancho de banda de señal alrededor de 2 MHz
- Modulación PPM
- Velocidad de datos de 1 Mbit/s, con mensajes entre 56 - 112 microsegundos
- Los mensajes transportan 15 bytes de datos cada uno, por lo que normalmente se necesitan varios mensajes para toda la información de la aeronave.
- El acceso múltiple se consigue emitiendo mensajes con un periodo que oscila aleatoriamente entre 0,4 y 0,6 segundos. Esta aleatorización está diseñada para evitar que los aviones tengan todas sus transmisiones una encima de la otra (algunas aún pueden colisionar, pero está bien)
- Las antenas ADS-B están polarizadas verticalmente
- La potencia de transmisión varía, pero debe rondar los 100 W (20 dBW)
- La ganancia de la antena de transmisión es omnidireccional pero solo apunta hacia abajo, así que digamos 3 dB
- Los receptores ADS-B también tienen una ganancia de antena omnidireccional, por lo que digamos 0 dB.

La pérdida de camino depende de qué tan lejos esté el avión de nuestro receptor. Por ejemplo, hay unos 30 km entre la Universidad de Maryland (donde se impartió el curso del que se originó el contenido de este libro de texto) y el aeropuerto BWI. Calculemos FSPL para esa distancia y una frecuencia de 1.090 MHz:

.. math::
    \mathrm{FSPL}_{dB} = 20 \log_{10} d + 20 \log_{10} f - 147.55  \left[ \mathrm{dB} \right]
    
    \mathrm{FSPL}_{dB} = 20 \log_{10} 30e3 + 20 \log_{10} 1090e6 - 147.55  \left[ \mathrm{dB} \right]

    \mathrm{FSPL}_{dB} = 122.7 \left[ \mathrm{dB} \right]

Otra opción es dejar :math:`d` como una variable en el presupuesto del enlace y determinar a qué distancia podemos escuchar señales en función de una SNR requerida.

Ahora bien, como definitivamente no tendremos espacio libre, agreguemos otros 3 dB de pérdida. Haremos que la pérdida sea de 6 dB en total, para tener en cuenta que nuestra antena no está bien adaptada y las pérdidas del cable/conector. Teniendo en cuenta todos estos criterios, nuestro presupuesto de enlace de señal se ve así:

.. list-table::
   :widths: 15 10
   :header-rows: 0
   
   * - Pt
     - 20 dBW
   * - Gt
     - 3 dB
   * - Gr
     - 0 dB
   * - Lp
     - -122.7 dB
   * - Lmisc
     - -6 dB
   * - **Pr**
     - **-105.7 dBW**

Para nuestro presupuesto de ruido:

- B = 2 MHz = 2e6 = 63 dBHz
- T tenemos que aproximarnos, digamos 300 K, que son 24,8 dBK. Variará según la calidad del receptor.
- k es siempre -228,6 dBW/K/Hz

.. math::
 P_{noise} = k + T + B = -140.8 \quad \mathrm{dBW}
 
Por lo tanto, nuestra SNR es -105,7 - (-140,8) = **35,1 dB**. No es sorprendente que sea un número enorme, considerando que afirmamos estar a sólo 30 km del avión en el espacio libre. Si las señales ADS-B no pudieran alcanzar los 30 km, entonces ADS-B no sería un sistema muy efectivo: nadie se escucharía hasta que estuvieran muy cerca. Con este ejemplo podemos decodificar fácilmente las señales; La modulación de posición de pulso (PPM) es bastante robusta y no requiere una SNR tan alta. Lo difícil es cuando intentas recibir ADS-B dentro de un salón de clases, con una antena muy mal adaptada y una estación de radio FM potente cerca que causa interferencias. Esos factores podrían fácilmente provocar pérdidas de entre 20 y 30 dB.

Este ejemplo fue en realidad solo un cálculo aproximado, pero demostró los conceptos básicos de la creación de un presupuesto de enlace y la comprensión de los parámetros importantes de un enlace de comunicaciones.
