.. _intro-chapter:

#############
Introduction
#############

***************************
Objectif et public cible
***************************

Tout d'abord, quelques termes importants:

**Software-Defined Radio (SDR):**
    ou Radio Logicielle en bon français. C'est une radio qui utilise un logiciel pour effectuer des tâches de traitement du signal qui étaient traditionnellement effectuées par des composants éléctroniques.
  
**Digital Signal Processing (DSP):**
    ou traitements numérique du signal (RF dans notre cas) en bon français.

Ce manuel est une introduction pratique aux domaines du DSP, de la radio logicielle et des communications sans fil. Il est conçu pour quelqu'un qui:

#. Est intéressé à *utiliser* les SDR pour faire des trucs cool
#. a une bonne connaissance de Python
#. est relativement nouveau dans le domaine du DSP, des communications sans fil et de la radio logicielle.
#. est un apprenant plutôt visuel, préférant les animations aux équations.
#. comprends mieux les équations *après* avoir appris les concepts de base.
#. cherche des explications concises, et non un manuel de 1000 pages.

Il s'adresse par exemple à un étudiant en informatique intéressé par un emploi dans le domaine des communications sans fil après l'obtention de son diplôme. Ou toute autre personne désireuse d'en savoir plus sur la DSP et ayant une expérience en programmation. En tant que tel, il couvre la théorie nécessaire pour comprendre les techniques DSP sans les mathématiques intenses qui sont généralement incluses dans les cours universitaires. Au lieu de vous enterrer sous de nombreuses équations, une abondance d'images et d'animations sont utilisées pour aider à transmettre les concepts, comme l'animation du plan complexe de la série de Fourier ci-dessous. Je pense que les équations sont mieux comprises *après* l'apprentissage des concepts par des images et des exercices pratiques.  L'utilisation intensive d'animations est la raison pour laquelle PySDR n'aura jamais de version papier vendue sur Amazon.  

.. image:: ../_images/fft_logo_wide.gif
   :scale: 70 %   
   :align: center
   
Ce manuel a pour but d'introduire les concepts rapidement et en douceur, permettant au lecteur de réaliser des DSP et d'utiliser les SDRs intelligemment.  Il n'a pas pour but d'être un manuel de référence pour tous les sujets DSP/SDR ; il existe déjà beaucoup d'excellents manuels en anglais, tels que `Analog Device's SDR textbook'.
<https://www.analog.com/en/education/education-library/software-defined-radio-for-engineers.html>`_ et `dspguide.com <http://www.dspguide.com/>`_.  Vous pouvez toujours utiliser Google pour rappeler les identités trigonométriques ou la limite de Shannon. Considérez ce manuel comme une porte d'entrée dans le monde de la DSP et de la SDR: c'est un engagement plus léger et moins coûteux en temps et en argent, comparé aux cours et manuels plus traditionnels.

Pour couvrir la théorie fondamentale du DSP, un semestre entier de "Signaux et systèmes", cours typique d'ingénierie, est condensé en quelques chapitres. Une fois les fondements du DSP couverts, nous nous lançons dans les SDR, bien que les concepts de DSP et de communications sans fil continuent d'être abordés tout au long du manuel.

Les exemples de code sont fournis en Python.  Ils utilisent NumPy, qui est la bibliothèque standard de Python pour les tableaux et les mathématiques de haut niveau. Les exemples s'appuient également sur Matplotlib, qui est une bibliothèque de traçage Python permettant de visualiser facilement des signaux, des tableaux et des nombres complexes. Notez que si Python est plus "lent" que C++ en général, la plupart des fonctions mathématiques de Python/NumPy sont implémentées en C/C++ et sont fortement optimisées. De même, l'API SDR que nous utilisons est simplement un ensemble de liaisons Python pour des fonctions/classes C/C++. Ceux qui ont peu d'expérience de Python mais de solides bases en MATLAB, Ruby ou Perl se débrouilleront sans problème après s'être familiarisés avec la syntaxe de Python.


***************
Contribution
***************

Si vous parvenez à lire une partie de ce manuel et que vous m'envoyez un courriel à l'adresse marc@pysdr.org avec des questions/commentaires/suggestions, alors félicitations, vous aurez contribué à ce manuel!

Mais à une plus grande échelle, vous pouvez contribuer à ce manuel de la même manière que n'importe quel projet de logiciel open source - à travers Git. Ce manuel prend la forme d'un site Web, mais le matériel source à partir duquel il est généré vit sur la page GitHub du manuel `<https://github.com/777arc/PySDR>`_.  N'hésitez pas à soumettre un problème ou même une demande de transfert (Pull Request, PR) avec des corrections ou des améliorations. Ceux qui soumettent des commentaires/réparations de valeur seront ajoutés de façon permanente à la section des remerciements ci-dessous. Vous n'êtes pas doué pour Git mais vous avez des changements à suggérer? N'hésitez pas à m'envoyer un courriel à marc@pysdr.org.

Le site Web sur lequel ce manuel est hébergé est exempt de publicité, car nous détestons tous les publicités. Je ne fournis pas non plus d'adresse PayPal ou Bitcoin où j'accepte les dons. Il n'y a littéralement aucun moyen pour moi d'être payé pour ce manuel. Au lieu de cela, je suggère simplement de partager ce manuel avec des collègues, des étudiants et d'autres apprenants tout au long de la vie qui pourraient être intéressés par ce matériel.

*****************
Remerciements
*****************

Nous remercions tous ceux qui ont lu une partie de ce manuel et nous ont fait part de leurs commentaires, et tout particulièrement:

- `Barry Duggan <http://github.com/duggabe>`_
- Matthew Hannon
- James Hayek
- Deidre Stuffer
- `Tarik Benaddi <https://tarikbenaddi.github.io>`_ pour la `traduction de PySDR en français <https://pysdr.org/fr/index-fr.html>`_


