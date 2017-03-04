.. _preloaded_models_overview:

*********************************************
Tutorial on models pre-built by Halotools
*********************************************

This section of the documentation reviews the pre-built composite models
provided by Halotools. For a quick reference, instead see :ref:`preloaded_models_list`.

All pre-built models can directly populate a simulation
with a mock catalog and make observational predictions that can be compared
to measurements made on a real galaxy sample. You need only choose the
pre-built model and simulation snapshot that is appropriate for your science application.
By following the links below, you can
find a step-by-step tutorial for how to get going with
any of the Halotools-provided pre-built models.

HOD-Style Models
========================
HOD-style models make predictions for galaxy samples selected
to be brighter or more massive than some threshold lower bound.
The fundamental quantity in HOD-style models is :math:`P(N_{\rm gal} | M_{\rm halo})`,
the probability that a halo of mass :math:`M_{\rm halo}` contains :math:`N_{\rm gal}`
galaxies in some sample. All HOD models are based on host halos only;
this distinguishes them from "subhalo-based models": in HOD-style models
there is no connection between subhalo and satellite abundance.
All the models in this section are built by the
`~halotools.empirical_models.PrebuiltHodModelFactory`


Traditional HOD models
--------------------------------------
The following models are some the most widely used HODs in the literature.
These are all designed to model either luminosity-threshold or stellar mass-threshold
galaxy samples.

.. toctree::
   :maxdepth: 1

   zheng07_composite_model
   leauthaud11_composite_model
   zu_mandelbaum15_composite_model

Traditional CLF models
-----------------------
The following model is described as a Conditional Luminosity Function model. However,
the model works equally well to model the Conditional Stellar Mass Function.

.. toctree::
   :maxdepth: 1

   cacciato09_composite_model


Decorated HOD models
--------------------------------------
Decorated HOD models relax the assumption that halo mass is the only
relevant halo property governing galaxy occupation statistics.
This class of model naturally inherits all of the flexibility
provided by the traditional HOD, while at the same time providing
further flexibility to study the co-evolution of galaxies and their halos.

.. toctree::
   :maxdepth: 1

   hearin15_composite_model

HOD models with color/SFR
--------------------------------------
While traditional HOD models have historically been applied to color-selected
samples, such models do not contain any parameters controlling the dependence
of color/SFR on halo properties. The HOD models in this section *do*
contain such parametric freedom, making these models much more useful to
constrain the physics of star formation and quenching.

.. toctree::
   :maxdepth: 1

   tinker13_composite_model
   zu_mandelbaum16_composite_model

Subhalo-based Models
====================================
Subhalo models presume a one-to-one correspondence between galaxies and (sub)halos.
Central galaxies reside at the center of host halos, satellite galaxies at the center of subhalos.
This assumption relaxes many of the "spherical cow" assumptions typical of HOD-style models,
though this is at the expense of fine-grained control over the galaxy distribution.
All the models in this section are built by the
`~halotools.empirical_models.PrebuiltSubhaloModelFactory`


Stellar-to-halo mass models (SMHM)
--------------------------------------
An SMHM model presumes that there is a one-to-one correspondence between
stellar mass and subhalo mass. The SMHM relation is given by some parameterized form
typically motivated by results based on abundance matching.

.. toctree::
   :maxdepth: 1

   behroozi10_composite_model


Abundance matching models
--------------------------------------
Abundance matching models do not parameterize the stellar-to-halo mass relation.
In abundance matching, the SMHM relation is defined by the condition that the
stellar mass function of the model must be exactly correct.

.. toctree::
   :maxdepth: 1

   abundance_matching_composite_model




