Ugosa
========================================================

This is the code repository for the paper 
*User-guided one-shot deep model adaptation for music source separation*
by Giorgia Cantisani, Alexey Ozerov, Slim Essid, and Gaël Richard.

Copyright 2020 InterDigital R&D and Télécom Paris.

The source code will be made publicly available after the approval by the company

#### The paper
In this work, we propose to exploit a temporal segmentation provided by the user indicating when each instrument is active, in order to fine-tune a pre-trained deep model for source separation and adapt it to one specific mixture. This paradigm can be referred to as **user-guided one-shot deep model adaptation for music source separation**, as the adaptation acts on the target song instance only.
The adaptation is possible thanks to a proposed loss function which aims to **minimize the energy of the silent sources** while at the same time forcing the perfect reconstruction of the mixture. 
The results are promising and show that state-of-the-art source separation models have large margins of improvement especially for those instruments which are underrepresented in the training data. Below you can find some audio examples from the MUSDB18 test set.

#### Links
- [Paper preprint](https://hal.telecom-paris.fr/hal-03219350)
- [GitHub Code](https://github.com/giorgiacantisani/ugosa)
- [Demo](https://adasp.telecom-paris.fr/resources/2021-06-01-ugosa-paper)

Acknowledgment
--------------
This work has received funding from the European Union’s Horizon 2020 research and innovation
programme under the Marie Skłodowska-Curie grant agreement No. 765068.
