:orphan:

.. _nfw_profile_tutorial:

****************************************************
Tutorial on the NFWProfile and NFWPhaseSpace Models
****************************************************

This section of the documentation provides background material 
and detailed implementation notes on the functions and methods of the 
`~halotools.empirical_models.phase_space_models.profile_models.NFWProfile` 
and `~halotools.empirical_models.phase_space_models.NFWPhaseSpace` models.
Much of the functionality of these classes derives from the behavior defined in the 
`~halotools.empirical_models.phase_space_models.profile_models.AnalyticDensityProf` super-class. 
Here we only document 
the functionality and implementation that is unique to the 
`~halotools.empirical_models.phase_space_models.profile_models.NFWProfile` 
and `~halotools.empirical_models.phase_space_models.NFWPhaseSpace` models, 
and defer discussion of all super-class-derived behavior to the :ref:`profile_template_tutorial`. 

.. _nfw_phase_space_class_structure:

Class structure of the `~halotools.empirical_models.phase_space_models.NFWPhaseSpace` model
==========================================================================================================

The `~halotools.empirical_models.phase_space_models.NFWPhaseSpace` model is a container class 
for three independently-defined sets of behaviors: 

	1. Analytical descriptions of the distribution of points within the halo (`~halotools.empirical_models.phase_space_models.profile_models.NFWProfile`)
	2. Analytical descriptions of velocity dispersion of tracer particles orbiting within the halo (`~halotools.empirical_models.phase_space_models.velocity_models.NFWJeansVelocity`)
	3. Monte Carlo methods for generating random realizations of points in phase space (`~halotools.empirical_models.phase_space_models.MonteCarloGalProf`)

The `~halotools.empirical_models.phase_space_models.NFWPhaseSpace` class does not itself model any of the above functionality; each of the above three sets of behaviors are actually modeled in the indicated class, and `~halotools.empirical_models.phase_space_models.NFWPhaseSpace` uses multiple inheritance to compose these behaviors together into a composite model for the phase space distribution of points orbiting in virial equilibrium inside an NFW potential. In the three subsections below, we describe each of these three model components in turn. 

.. _nfw_spatial_profile_derivations:

Modeling the NFW Spatial Profile 
======================================



.. _nfw_dimensionless_mass_density_derivation: 

Derivation of the NFW dimensionless mass density 
--------------------------------------------------

As described in :ref:`halo_profile_definitions`, in the Halotools implementation of 
any analytic density profile the central mathematical quantity is the *dimensionless 
mass density,* defined as 

.. math::

	\tilde{\rho}_{\rm prof}(\tilde{r}) \equiv \rho_{\rm prof}(\tilde{r})/\rho_{\rm thresh}. 

Nearly all of the functionality of the 

.. math::

	\frac{c^{3}/3g(c)}{cx(1 + cx)^{2}}


Derivation of the NFW cumulative mass PDF 
------------------------------------------------

.. math::

	g(cx) / g(c)


Monte Carlo realizations of the NFW profile
------------------------------------------------

How are things computed in practice (lookup tables, etc.)


.. _nfw_jeans_velocity_profile_derivations:

Modeling the NFW Velocity Profile 
===========================================

How are things computed in practice (lookup tables, etc.)


.. _nfw_monte_carlo_derivations:

Monte Carlo realizations of the NFW profile
===========================================

How are things computed in practice (lookup tables, etc.)

