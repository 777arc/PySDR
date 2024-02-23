.. _intro-chapter:

#############
Introducción
#############

****************************
Propósito y público objetivo
****************************

Primero y más importante, un par de términos importantes:

**Software-Defined Radio (SDR):**
    Un dispositivo de radio que utiliza software para realizar tareas de procesamiento de señales que tradicionalmente se realizaban mediante hardware.
  
**Digital Signal Processing (DSP):**
    El procesamiento digital de señales, en nuestro caso señales RF.

Este libro de texto actúa como una introducción práctica a las áreas de DSP, SDR y comunicaciones inalámbricas. Está diseñado para alguien que es:

#. Interesado en *usar* SDRs para hacer cosas interesantes
#. Bueno con Python
#. Relativamente nuevo en DSP, comunicaciones inalámbricas y SDR
#. Un estudiante visual que prefiere las animaciones a las ecuaciones.
#. Mejor comprensión de ecuaciones *después* de aprender los conceptos
#. Buscando explicaciones concisas, no un libro de texto de 1.000 páginas

Un ejemplo es un estudiante de Ciencias de la Computación interesado en un trabajo que involucre comunicaciones inalámbricas después de graduarse, aunque puede ser utilizado por cualquiera que desee aprender sobre SDR y tenga experiencia en programación. Como tal, cubre la teoría necesaria para comprender las técnicas de DSP sin las intensas matemáticas que normalmente se incluyen en los cursos de DSP. En lugar de sumergirnos en ecuaciones, se utiliza una gran cantidad de imágenes y animaciones para ayudar a transmitir los conceptos, como la animación de plano complejo de la serie de Fourier que aparece a continuación. Creo que las ecuaciones se entienden mejor *después* de aprender los conceptos a través de imágenes y ejercicios prácticos. El uso intensivo de animaciones es la razón por la que PySDR nunca tendrá una versión impresa a la venta en Amazon. 

.. image:: ../_images/fft_logo_wide.gif
   :scale: 70 %   
   :align: center
   :alt: The PySDR logo created using a Fourier transform
   
Este libro de texto está destinado a presentar conceptos de forma rápida y fluida, permitiendo al lector utilizar conceptos de DSP y implementar en una SDR de forma inteligente. No pretende ser un libro de texto de referencia para todos los temas de DSP/SDR; Ya existen muchos libros de texto excelentes, como `Analog Device's SDR textbook
<https://www.analog.com/en/education/education-library/software-defined-radio-for-engineers.html>`_ y `dspguide.com <http://www.dspguide.com/>`_.  Siempre puedes utilizar Google para recordar las identidades trigonométricas o el límite de Shannon. Piense en este libro de texto como una puerta de entrada al mundo de DSP y SDR: es más liviano y requiere menos tiempo y compromiso monetario, en comparación con cursos y libros de texto más tradicionales.

Para cubrir la teoría fundamental del DSP, un semestre completo de "Señales y Sistemas", un curso típico dentro de la ingeniería eléctrica, se condensa en unos pocos capítulos. Una vez que se cubren los fundamentos de DSP, nos lanzamos a los SDR, aunque los conceptos de DSP y comunicaciones inalámbricas continúan apareciendo a lo largo del libro de texto.

Se proporcionan ejemplos de código en Python. Utilizan NumPy, que es la biblioteca estándar de Python para matrices y matemáticas de alto nivel. Los ejemplos también se basan en Matplotlib, que es una biblioteca para gráficar en Python que proporciona una forma sencilla de visualizar señales, matrices y números complejos. Tenga en cuenta que, si bien Python es "más lento" que C++ en general, la mayoría de las funciones matemáticas dentro de Python/NumPy se implementan en C/C++ y están muy optimizadas. Del mismo modo, la API SDR que utilizamos es simplemente un conjunto de enlaces de Python para funciones/clases de C/C++. Aquellos que tienen poca experiencia en Python pero una base sólida en MATLAB, Ruby o Perl probablemente estarán bien después de familiarizarse con la sintaxis de Python.


***************
Contribuciones
***************

Si obtuvo valor de PySDR, compártalo con colegas, estudiantes y otros estudiantes permanentes que puedan estar interesados en el material. También puedes donar a través del `PySDR Patreon <https://www.patreon.com/PySDR>`_ como una forma de decir gracias y hacer que su nombre aparezca a la izquierda de cada página debajo de la lista de capítulos.

Si lee cualquier cantidad de este libro de texto y me envía un correo electrónico a pysdr@vt.edu con preguntas/comentarios/sugerencias, entonces felicidades, ¡habrá contribuido a este libro de texto! También puedes editar el material fuente directamente en el `textbook's GitHub page <https://github.com/777arc/PySDR/tree/master/content>`_ (su cambio iniciará con una nueva Pull Request). No dudes en enviar un **Issue** o incluso una **Pull Request (PR)** con correcciones o mejoras. Aquellos que envíen comentarios/correcciones valiosas se agregarán permanentemente a la sección de agradecimientos a continuación. ¿No eres bueno en Git pero tienes cambios que sugerir? No dude en enviarme un correo electrónico a pysdr@vt.edu.

*****************
Agradecimientos
*****************

Gracias a cualquiera que haya leído alguna parte de este libro de texto y haya brindado comentarios, y especialmente a:

- `Barry Duggan <http://github.com/duggabe>`_
- Matthew Hannon
- James Hayek
- Deidre Stuffer
- Tarik Benaddi por `traducir PySDR al Frances <https://pysdr.org/fr/index-fr.html>`_
- `Daniel Versluis <https://versd.bitbucket.io/content/about.html>`_ por `traducir PySDR al Aleman <https://pysdr.org/nl/index-nl.html>`_
- `mrbloom <https://github.com/mrbloom>`_ por `tranducir PySDR al ucraniano <https://pysdr.org/ukraine/index-ukraine.html>`_
- `Yimin Zhao <https://github.com/doctormin>`_ por `traducir PySDR al chino simplificado <https://pysdr.org/zh/index-zh.html>`_
- Eduardo Chancay por `traducir PySDR al español <https://pysdr.org/es/index-es.html>`_
Así como todos nuestros `PySDR Patreon <https://www.patreon.com/PySDR>`_ !
