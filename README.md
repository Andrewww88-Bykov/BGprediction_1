# Band Gap Prediction for Low-Dimensional Hybride Metal(III) Halides by Mashine Learning

This work is dedicated to solve QSPR task for Band Gap (BG) of hybrid antimony(III) and bismuth(III) halides with $\\{MHal_{4}\\}^{-}\ (M = Bi^{+3}, Sb^{+3}; Hal = I^{-}, Br^{-}, Cl^{-})$ anion type. The repository contains the sources data, trained models, code for training and prediction, and a small part on data analysis. It is expected that this will be an addition to a scientific article.

At the moment, only a part of the hybrid metal(III) halides dataset is presented, it will be fully available after the article is rejected.

## Abstrac

The design of new functional materials is based on the search for compounds whose physicochemical properties should provide a significant improvement in the efficiency and economy of existing devices. Experimental approaches, including those based on the trial and error method, do not provide high performance when considering a huge number of potential candidates and selecting the most suitable ones, the study of which requires a large amount of resources. One of the most rational and currently feasible strategies is the "inverse design of materials" - establishing correlations between various characteristics of compounds and their structure (Quantitative Structure-Property Relationship, QSPR task) based on the accumulated data and further reasonable choice of the direction in which to focus efforts. Developing this approach for potential materials with a wide range of attractive optoelectronic properties, in this work we applied machine learning methods to establish quantitative-structure-property-relationships between the main characteristic of semiconductor materials - the band gap and the structural parameters of non-toxic low-dimensional organic inorganic halobismuthates(III) and haloantimonates(III) with an anionic substructure of the α-$\\{MHal_{4}\\}^{-}$ type. Based on the most available data - geometric, obtained from crystallographic data, we have investigated possible descriptors for solving the QSPR problem and determined their importance. We have proposed two machine learning models for two different feature spaces, and also shown that the distortions of the MHal6 octahedrons that make up anions in terms of their effect on the value of band gap are comparable to non-covalent interactions between anions. The first dataset of hybrid halometallates of Group 15 metals with a 1D anion, models and some auxiliary data are published in the public domain here.

## What's Presented Here

1) Dataset (notebook "Data analysis" contains a detailed description, initial data analysis and preparation for training).

2) Model training code (notebook "BG prediction").

3) Trained models (in pickle-format, you can use notebook "Models to use" to load models).

4) Additional data (geometric calculations in dia folder).
   
## What's New
-
## About Authors
Andrey Bykov, Ph.D. student, Chemistry Department Moscow State University (bykov.andrey.sw@gmail.com).
## Funding
This research was funded by "Non-commercial Foundation for the Advancement of Science and Education INTELLECT".
