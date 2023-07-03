# Intrinsic Alignment Models

This directory contains classes and functions for modelling the intrinsic alignment(s) of galaxies.


## Empirical IA Models

The code necessary to build empirical models for IA are located in the following files:

* `ia_model_componenets.py`

The alignment models in `ia_model_componenets.py` use an implemntation of the Watson distribution available [here](https://github.com/duncandc/watson_dist), which must be in your `PYTHONPATH`.

A notebook demonstrating of the IA model components is in the [notebooks](../../notebooks/) dirctory.


## Auxiliary Models

In addition to models for galaxy-halo alignments, this direcory contains models for the number and position of galaxies to assist in examining various alignment effects.  These include:

* `nfw_phase_space.py`
* `occupation_models.py`




