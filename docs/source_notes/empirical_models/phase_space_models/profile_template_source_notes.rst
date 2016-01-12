:orphan:

.. currentmodule:: halotools.empirical_models

.. _profile_template_tutorial:

****************************************************
Source code notes on `AnalyticDensityProf`
****************************************************

This section of the documentation provides background material and detailed implementation notes 
on the functions and methods of the primary base class used to model the spatial distribution 
of matter and galaxies within halos,  
`~halotools.empirical_models.AnalyticDensityProf`. 
This as an abstract base class and so it cannot itself be instantiated; only concrete 
sub-classes can be used to directly model the spatial profile of halos. 

The purpose of the `~halotools.empirical_models.AnalyticDensityProf` 
class is to provide a template for any Halotools model of the spatial distribution 
of points within a halo. So in Halotools, any analytical model for how either matter or galaxies 
are spatially distributed within their halos will subclass from the 
`~halotools.empirical_models.AnalyticDensityProf` class. 

This tutorial is organized as follows. The :ref:`halo_mass_definitions` section 
reviews how halo boundaries and masses are defined with respect to a cosmologically 
evolving reference density. The :ref:`halo_profile_definitions` section covers 
the mathematics of halo density profiles, including explicit derivations 
of the exact form of all equations are they are implemented in code. 
The tutorial concludes with the :ref:`analytic_density_prof_constructor` section by describing how the 
`__init__` constructor standardizes the attributes and behavior 
of the class to facilitate mock-making with a uniform syntax. 

.. _halo_mass_definitions:

Halo Mass and Radius Definitions
===================================


Basic equations
-----------------------------------

The `~halotools.empirical_models.AnalyticDensityProf` class models 
a dark matter halo to be a spherically symmetric overdensity relative to some reference 
density, :math:`\rho_{\rm ref}(z)`. The reference density is typically either the critical 
energy density of the universe, :math:`\rho_{\rm crit}(z)`, or the mean matter density 
:math:`\rho_{\rm m}(z) = \Omega_{m}(z)\rho_{\rm crit}(z)`. In the spherical-overdensity 
definition of a halo, the spherical boundary of a halo is defined such that the region inside 
the spherical shell has a fixed density 

.. math::

	\rho_{\rm thresh}(z) \equiv \Delta_{\rm ref}(z)\rho_{\rm ref}(z); 

the redshift-dependence of :math:`\rho_{\rm thresh}` reflects both the evolution of the reference 
density :math:`\rho_{\rm ref}` as well as any possible redshift-dependence in the scalar multiple 
:math:`\Delta_{\rm ref}`. 

The cosmological model determines :math:`\rho_{\rm ref}(z)`; the choice of a halo mass 
definition is determined by how one chooses :math:`\Delta_{\rm ref}(z)`. Typically, one chooses 
:math:`\Delta_{\rm ref}` to be some redshift-independent multiple of the reference density. In the conventional 
notation for this choice, :math:`\Delta_{\rm ref}(z) = 500c` refers to the case where 

.. math::

	\rho_{\rm thresh}(z) = 500\rho_{\rm crit}(z), 

and :math:`\Delta_{\rm ref}(z) = 200m` is shorthand for 

.. math::

	\rho_{\rm thresh}(z) = 200\rho_{\rm m}(z). 

The other common choice for :math:`\Delta_{\rm ref}(z)` is :math:`\Delta_{\rm vir}(z)`, 
which defined by the solution of the gravitational collapse of a top-hat overdensity evolving in an 
expanding background. 

Once a choice is made for :math:`\Delta_{\rm ref}(z)`, the mass of a spherically symmetric halo is defined by:

.. math::

	M_{\Delta}(z) \equiv \frac{4\pi}{3}R_{\Delta}^{3}\Delta_{\rm ref}(z)\rho_{\rm ref}(z) 

This equation defines the relationship between the total mass of a halo :math:`M_{\Delta}` 
and the halo boundary :math:`R_{\Delta}`. 


Computing the relevant quantities
-----------------------------------

In Halotools, the reference densities are computed using the `~astropy.cosmology` sub-package of Astropy, 
and the remaining quantities are computed in the 
`~halotools.empirical_models.profile_models` sub-package, 
specifically the `~halotools.empirical_models.profile_helpers` module. 

============================================  ========================================================================================================= 
Quantity                                      Source Code                 
============================================  ========================================================================================================= 
:math:`\rho_{\rm thresh}(z)`                  `~halotools.empirical_models.profile_helpers.density_threshold`
:math:`\Delta_{\rm vir}(z)`                   `~halotools.empirical_models.profile_helpers.delta_vir`
:math:`M_{\Delta}(z)`                         `~halotools.empirical_models.profile_helpers.halo_radius_to_halo_mass`
:math:`R_{\Delta}(z)`                         `~halotools.empirical_models.profile_helpers.halo_mass_to_halo_radius`
:math:`\rho_{\rm crit}(z)`                    `~astropy.cosmology.FLRW.critical_density`
:math:`\Omega_{\rm m}(z)`                     `~astropy.cosmology.FLRW.Om`
============================================  =========================================================================================================

.. _halo_profile_definitions:

Spatial Profiles of Halos
===================================

Basic equations
-----------------------------------

For a given choice of :math:`\Delta_{\rm ref}(z)`, the mass of a spherically symmetric halo is 
is related to the spatial profile of the matter in its interior via:

.. math::

	M_{\Delta}(z) \equiv 4\pi\int_{0}^{R_{\Delta}}dr' r'^{2}\rho_{\rm prof}(r')

This equation defines the normalization of the halo profile :math:`\rho_{\rm prof}(r)`, which for any 
sub-class of `~halotools.empirical_models.AnalyticDensityProf` is 
computed with the 
`~halotools.empirical_models.AnalyticDensityProf.mass_density` method. 

For numerical stability, it is always preferable to work with order-unity quantities rather than astronomical numbers. So throughout the `~halotools.empirical_models.profile_models` sub-package, most methods 
work with the *scaled radius*, :math:`\tilde{r}`, defined as:

.. math::

	\tilde{r} \equiv r/R_{\Delta}, 

and the `~halotools.empirical_models.AnalyticDensityProf.dimensionless_mass_density`, 
:math:`\tilde{\rho}_{\rm prof}`, defined as:

.. math::

	\tilde{\rho}_{\rm prof}(\tilde{r}) \equiv \rho_{\rm prof}(\tilde{r})/\rho_{\rm thresh}

In the implementation of `~halotools.empirical_models.AnalyticDensityProf`, 
for reasons of numerical stability a profile is actually defined by :math:`\tilde{\rho}_{\rm prof}(\tilde{r})`, 
and :math:`\rho_{\rm prof}(r)` is a derived quantity. 

In fact, in order to define a new 
profile model, one only need define a sub-class 
`~halotools.empirical_models.AnalyticDensityProf` and provide an 
implementation of the `~halotools.empirical_models.AnalyticDensityProf.dimensionless_mass_density` method, as *all* other profile quantities can be computed from this function. 

Convenience functions 
-----------------------

In addition to the `~halotools.empirical_models.AnalyticDensityProf.dimensionless_mass_density` method that defines the profile, instances of the 
`~halotools.empirical_models.AnalyticDensityProf` class 
have a number of other useful bound methods:

.. _computing_enclosed_mass:

Enclosed mass
~~~~~~~~~~~~~~

The mass enclosed within a given radius is defined as:

.. math::

	M_{\Delta}(<r) \equiv 4\pi\int_{0}^{r}dr' r'^{2}\rho_{\rm prof}(r'), 

which can be computed via the 
`~halotools.empirical_models.AnalyticDensityProf.enclosed_mass` method 
of the `~halotools.empirical_models.AnalyticDensityProf` class, 
or any of its sub-classes. 

.. _computing_cumulative_mass_PDF:

Cumulative mass PDF
~~~~~~~~~~~~~~~~~~~~

One particularly important quantity in making mocks is :math:`P_{\rm prof}(<\tilde{r})`, 
the cumulative probability of finding a randomly selected 
particle at a scaled-radius position less than :math:`\tilde{r}`:

.. math::

	P_{\rm prof}(<\tilde{r}) \equiv M_{\Delta}(<\tilde{r}) / M_{\Delta}.  

This function is computed by 
the `~halotools.empirical_models.AnalyticDensityProf.cumulative_mass_PDF` method 
of the `~halotools.empirical_models.AnalyticDensityProf` class. 
The :math:`P_{\rm prof}(<\tilde{r})` is used by 
`~halotools.empirical_models.MonteCarloGalProf` 
to help generate Monte Carlo realizations of halo density profiles. 

For reasons of numerical stability, in the Halotools implementation 
of the `~halotools.empirical_models.AnalyticDensityProf.enclosed_mass` method
the quantity :math:`M_{\Delta}(<r)` is computed as 
:math:`M_{\Delta}(<r) = P_{\rm prof}(<\tilde{r})M_{\Delta}`. 

.. _computing_virial_velocity:

Virial velocity 
~~~~~~~~~~~~~~~~~

A halo's *virial velocity* :math:`V_{\rm vir}` is defined as:

.. math::

	V^{2}_{\rm vir} \equiv GM_{\Delta}/R_{\Delta}

Intuitively, the virial velocity is the speed of a tracer particle on a 
circular orbit at a distance :math:`R_{\Delta}` from the center of a halo in virial equilibrium. 
You can compute :math:`V_{\rm vir}` via 
the `~halotools.empirical_models.AnalyticDensityProf.virial_velocity` method 
of the `~halotools.empirical_models.AnalyticDensityProf` class, 
or any of its subclasses. 

.. _computing_circular_velocity:

Circular velocity profile 
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The circular velocity profile, :math:`V_{\rm circ}(r)`, is defined as:

.. math::

	V^{2}_{\rm circ}(r) \equiv GM_{\Delta}(<r)/r, 

where *G* is Newton's constant. Intuitively, :math:`V_{\rm circ}(r)` is the speed of  
a tracer particle on a bound circular orbit at a distance *r* from the 
center of a virialized halo. You can compute :math:`V_{\rm circ}(r)` with  
the `~halotools.empirical_models.AnalyticDensityProf.circular_velocity` method 
of the `~halotools.empirical_models.AnalyticDensityProf` class, 
or any of its sub-classes. 

For reasons of numerical stability, when computing :math:`V_{\rm circ}(r)` 
it is useful to use the *dimensionless-circular velocity*, 
:math:`\tilde{V}_{\rm circ}(r)`, defined as 

.. math::

	\tilde{V}_{\rm circ}(r) \equiv V_{\rm circ}(r) / V_{\rm vir}, 

so that :math:`V_{\rm circ}(r) = \tilde{V}_{\rm circ}(r)V_{\rm vir}`.

In the actual Halotools implementation :math:`\tilde{V}_{\rm circ}(r)` is computed using 

.. math::

	\tilde{V}^{2}_{\rm circ}(\tilde{r}) = \frac{P_{\rm prof}(<\tilde{r})}{\tilde{r}}

To see that this alternative method of calculation is correct:

.. math:: 

	\tilde{V}_{\rm circ}(r) \equiv \frac{V_{\rm circ}(r)}{V_{\rm vir}} \\

	\tilde{V}_{\rm circ}(r) = \frac{GM_{\Delta}(<r)/r}{GM_{\Delta}/R_{\Delta}} \\

	\tilde{V}_{\rm circ}(r) = \frac{M_{\Delta}(<r)/M_{\Delta}}{r/R_{\Delta}}, 

where the second equality follows from the definition of :math:`V_{\rm circ}`. 
Since the numerator in the final expression is :math:`P_{\rm prof}(<r)` 
and the denominator is :math:`\tilde{r}`, we arrive at  

.. math::

	\tilde{V}^{2}_{\rm circ}(\tilde{r}) = \frac{P_{\rm prof}(<\tilde{r})}{\tilde{r}}

This is why in the source code for the 
`~halotools.empirical_models.AnalyticDensityProf.dimensionless_circular_velocity` method, the returned quantity is :math:`\sqrt{P_{\rm prof}(<\tilde{r})/\tilde{r}}`. Then the source code for the `~halotools.empirical_models.AnalyticDensityProf.circular_velocity` method simply multiplies the returned value of `~halotools.empirical_models.AnalyticDensityProf.dimensionless_circular_velocity` by :math:`V_{\rm vir}`. 

.. _computing_vmax:

Maximum circular velocity 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The maximum circular velocity :math:`V_{\rm max}` is defined as the maximum value attained by 
:math:`V_{\rm circ}(r)` over the entire profile of the halo. Halotools computes :math:`V_{\rm max}` 
by using Scipy's zero-finder `~scipy.optimize.minimize`. You can compute :math:`V_{\rm max}` 
using the `~halotools.empirical_models.AnalyticDensityProf.vmax` method of the `~halotools.empirical_models.AnalyticDensityProf` class, 
or any of its sub-classes. 


Computing the relevant quantities
-----------------------------------


============================================  ====================================================================================================================================================== 
Quantity                                      Source Code                 
============================================  ====================================================================================================================================================== 
:math:`\rho_{\rm prof}(r)`                    `~halotools.empirical_models.AnalyticDensityProf.mass_density`
:math:`\tilde{\rho}_{\rm prof}(\tilde{r})`                    `~halotools.empirical_models.AnalyticDensityProf.dimensionless_mass_density`
:math:`M_{\Delta}(<r)`                    	  `~halotools.empirical_models.AnalyticDensityProf.enclosed_mass`
:math:`P_{\rm prof}(<\tilde{r})`              `~halotools.empirical_models.AnalyticDensityProf.cumulative_mass_PDF`
:math:`V_{\rm vir}`                           `~halotools.empirical_models.AnalyticDensityProf.virial_velocity`
:math:`V_{\rm circ}(r)`                       `~halotools.empirical_models.AnalyticDensityProf.circular_velocity`
:math:`\tilde{V}_{\rm circ}(r)`               `~halotools.empirical_models.AnalyticDensityProf.dimensionless_circular_velocity`
============================================  ======================================================================================================================================================

.. currentmodule:: halotools.empirical_models

.. _analytic_density_prof_constructor:

Constructor of the `~AnalyticDensityProf` class 
=================================================

In this final section of the tutorial, we will look closely at the ``__init__`` constructor to see how it creates a standardized interface for the purpose of making mock catalogs with the Halotools factories. 

The `~AnalyticDensityProf` constructor has three required arguments: ``cosmology``, ``redshift`` and ``mdef``. Binding these attributes to the instance accomplishes several things:

1. When an instance of an `~AnalyticDensityProf` sub-class is incorporated into a composite model, these attributes will be compared against the corresponding attributes of other component models so that composite model consistency is ensured. 

.. currentmodule:: halotools.empirical_models

2. A fixed value for the ``density_threshold`` attribute can be bound to the instance that is consistent with the returned value of the `profile_helpers.density_threshold` function.

3. The string ``mdef`` is parsed with the `model_defaults.get_halo_boundary_key` function and the returned value is bound to the ``halo_boundary_key`` instance attribute. This guarantees that during mock-making, the appropriate column of the `~halotools.sim_manager.CachedHaloCatalog` halo_table will be automatically chosen for all methods of the `~AnalyticDensityProf` sub-class instance requiring this knowledge. 

4. Likewise, ``mdef`` is parsed with the `model_defaults.get_halo_mass_key` function and the returned value is bound to the ``prim_haloprop_key`` instance attribute. This guarantees that the `~AnalyticDensityProf` sub-class instance will use the halo_table column that is consistent with the input mass definition. 

At the conclusion of the constructor, a few empty sequences are bound to the instance. These are documented in the :ref:`param_dict_mechanism` and :ref:`prof_param_keys_mechanism`. 








