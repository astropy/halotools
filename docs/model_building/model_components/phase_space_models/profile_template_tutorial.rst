.. _profile_template_tutorial:

****************************************************
Tutorial on the AnalyticDensityProf Template Model
****************************************************

The purpose of the `~halotools.empirical_models.phase_space_models.profile_models.AnalyticDensityProf` 
class is to provide a template for any Halotools model of the spatial distribution 
of points within a halo. So in Halotools, any model for how either matter or galaxies 
are spatially distributed within their halos will subclass from the 
`~halotools.empirical_models.phase_space_models.profile_models.AnalyticDensityProf` class. This tutorial 
reviews the mathematics of halo profiles, and describes how the relevant equations 
are implemented in the code base. 

.. _halo_mass_definitions:

Halo Mass and Radius Definitions
===================================


Basic equations
-----------------------------------

The `~halotools.empirical_models.phase_space_models.profile_models.AnalyticDensityProf` class models 
a dark matter halo to be a spherically symmetric overdensity relative to some reference 
density, :math:`\rho_{\rm ref}(z)`. The reference density is typically either the critical 
energy density of the universe, :math:`\rho_{\rm crit}(z)`, or the mean matter density 
:math:`\rho_{\rm m}(z) = \Omega_{m}(z)\rho_{\rm crit}(z)`. In the spherical-overdensity 
definition of a halo, the spherical boundary of a halo is defined such that the region inside 
the spherical shell has a fixed density 

.. math::

	\rho_{\rm thresh}(z) \equiv \Delta_{\rm ref}(z)\rho_{\rm ref}(z)

The cosmological model determines :math:`\rho_{\rm ref}(z)`, and so the choice of a halo mass 
definition is determined by how one chooses :math:`\Delta_{\rm ref}(z)`. Typically, one chooses 
:math:`\Delta_{\rm ref}` to be some constant multiple of the reference density. In the conventional 
notation for this choice, :math:`\Delta_{\rm ref}(z) = 500c` refers to the case where 

.. math::

	\rho_{\rm thresh}(z) = 500\rho_{\rm crit}(z), 

and :math:`\Delta_{\rm ref}(z) = 200m` is shorthand for 

.. math::

	\rho_{\rm thresh}(z) = 200\rho_{\rm m}(z). 

The other common choice for the :math:`\Delta_{\rm ref}(z)` is :math:`\Delta_{\rm vir}(z)`, 
which defined by the solution of the gravitational collapse of a top-hat overdensity evolving in an 
expanding background. 

Once a choice is made for :math:`\Delta_{\rm ref}(z)`, the mass of a spherically symmetric halo is defined by:

.. math::

	M_{\Delta}(z) \equiv \frac{4\pi}{3}R_{\Delta}^{3}\Delta_{\rm ref}(z)\rho_{\rm ref}(z) 

This equation defines the relationship between the total mass of a halo :math:`M_{\Delta}` 
and the halo boundary :math:`R_{\Delta}`. 


Computing the relevant quantities
-----------------------------------

In Halotools, the reference densities are computed using the `~astropy.cosmology` sub-package, 
and the remaining quantities are computed in the 
`~halotools.empirical_models.phase_space_models.profile_models` sub-package, 
specifically the `~halotools.empirical_models.phase_space_models.profile_models.profile_helpers` module. 

============================================  ========================================================================================================= 
Quantity                                      Source Code                 
============================================  ========================================================================================================= 
:math:`\rho_{\rm thresh}(z)`                  `~halotools.empirical_models.phase_space_models.profile_models.profile_helpers.density_threshold`
:math:`\Delta_{\rm vir}(z)`                   `~halotools.empirical_models.phase_space_models.profile_models.profile_helpers.delta_vir`
:math:`M_{\Delta}(z)`                         `~halotools.empirical_models.phase_space_models.profile_models.profile_helpers.halo_radius_to_halo_mass`
:math:`R_{\Delta}(z)`                         `~halotools.empirical_models.phase_space_models.profile_models.profile_helpers.halo_mass_to_halo_radius`
:math:`\rho_{\rm crit}(z)`                    `~astropy.cosmology.FLRW.critical_density`
:math:`\Omega_{\rm m}(z)`                     `~astropy.cosmology.FLRW.Om`
============================================  =========================================================================================================

.. _halo_profile_definitions:

Spatial Profiles of Halos
===================================

Basic equations
-----------------------------------

For a given choice is made for :math:`\Delta_{\rm ref}(z)`, the mass of a spherically symmetric halo is 
is related to the spatial profile of the matter in its interior via:

.. math::

	M_{\Delta}(z) \equiv 4\pi\int_{0}^{R_{\Delta}}dr' r'^{2}\rho_{\rm prof}(r')

This equation defines the normalization of the halo profile :math:`\rho_{\rm prof}(r)`, which for any 
sub-class of `~halotools.empirical_models.phase_space_models.profile_models.AnalyticDensityProf` is 
computed with the 
`~halotools.empirical_models.phase_space_models.profile_models.AnalyticDensityProf.mass_density` method. 

For numerical stability, it is always preferable to work with order-unity quantities rather than astronomical numbers. So throughout the `~halotools.empirical_models.phase_space_models.profile_models` sub-package, most methods 
work with the *scaled radius*, :math:`\tilde{r}`, defined as:

.. math::

	\tilde{r} \equiv r/R_{\Delta}, 

and the `~halotools.empirical_models.phase_space_models.profile_models.AnalyticDensityProf.dimensionless_mass_density`, 
:math:`\tilde{\rho}_{\rm prof}`, defined as:

.. math::

	\tilde{\rho}_{\rm prof}(\tilde{r}) \equiv \rho_{\rm prof}(\tilde{r})/\rho_{\rm thresh}

In the implementation of `~halotools.empirical_models.phase_space_models.profile_models.AnalyticDensityProf`, 
for reasons of numerical stability a profile is actually defined by :math:`\tilde{\rho}_{\rm prof}(\tilde{r})`, 
and :math:`\rho_{\rm prof}(r)` is a derived quantity. 

In fact, in order to define a new 
profile model, one only need define a sub-class 
`~halotools.empirical_models.phase_space_models.profile_models.AnalyticDensityProf` and provide an 
implementation of the `~halotools.empirical_models.phase_space_models.profile_models.AnalyticDensityProf.dimensionless_mass_density` method, as *all* other profile quantities can be computed from this function. 

Convenience functions 
-----------------------

In addition to the `~halotools.empirical_models.phase_space_models.profile_models.AnalyticDensityProf.dimensionless_mass_density` method that defines the profile, instances of the 
`~halotools.empirical_models.phase_space_models.profile_models.AnalyticDensityProf` class 
have a number of other useful bound methods:

Enclosed mass
~~~~~~~~~~~~~~

The mass enclosed within a given radius is defined as:

.. math::

	M_{\Delta}(<r) \equiv 4\pi\int_{0}^{r}dr' r'^{2}\rho_{\rm prof}(r'), 

which can be computed via the 
`~halotools.empirical_models.phase_space_models.profile_models.AnalyticDensityProf.enclosed_mass` method. 

Cumulative mass PDF
~~~~~~~~~~~~~~~~~~~~

One particularly important quantity is the :math:`P_{\rm prof}(<\tilde{r})`, 
the cumulative probability of finding a randomly selected 
particle at a scaled-radius position less than :math:`\tilde{r}`:

.. math::

	P_{\rm prof}(<\tilde{r}) \equiv \frac{4\pi}{M_{\Delta}}\int_{0}^{\tilde{r}}d\tilde{r}' \tilde{r}'^{2}\rho_{\rm prof}(\tilde{r}').



Circular velocity 
~~~~~~~~~~~~~~~~~~

marf 

Maximum circular velocity 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

marf 






Computing the relevant quantities
-----------------------------------









