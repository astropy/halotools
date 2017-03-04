:orphan:

.. _subhalo_phase_space_model_tutorial:

************************************************************
Tutorial on Modeling HOD Satellites Using Subhalo Positions
************************************************************

This tutorial gives a detailed explanation of the
`~halotools.empirical_models.SubhaloPhaseSpace` class.

In any implementation of the HOD, ...

In the :ref:`general_comment_subhalo_phase_space` section


.. _general_comment_subhalo_phase_space:

General comment
================

As shown in `Nagai and Kravtsov (2005) <https://arxiv.org/abs/astro-ph/0408273/>`_ and subsequent papers, subhalos selected according to different subhalo properties exhibit different radial distributions. Thus, depending on which subhalos you choose to place your satellites in, you will get different radial distributions of satellites, and also different velocity distributions. Most of the freedom built into the `~halotools.empirical_models.SubhaloPhaseSpace` class is designed around allowing you to explore different choices for what we will loosely refer to as the "subhalo selection function".

.. code:: python

    from halotools.empirical_models import default_inherited_subhalo_props_dict

