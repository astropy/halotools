:orphan:

.. _nfw_profile_tutorial:

****************************************************
Tutorial on the NFWProfile and NFWPhaseSpace Models
****************************************************

This section of the documentation provides background material 
and detailed implementation notes on the functions and methods of the 
`~halotools.empirical_models.phase_space_models.profile_models.NFWProfile` 
and `~halotools.empirical_models.phase_space_models.NFWPhaseSpace` models. 
We will start out with an overview of the class design before moving into 
the analytical derivations of the code implementation. 

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

The spatial profile of an NFW halo is modeled with the `~halotools.empirical_models.phase_space_models.profile_models.NFWProfile` class, which is itself a sub-class of:

	1. `~halotools.empirical_models.phase_space_models.profile_models.AnalyticDensityProf`
	2. `~halotools.empirical_models.phase_space_models.profile_models.ConcMass`

The `~halotools.empirical_models.phase_space_models.profile_models.AnalyticDensityProf` class governs most of the analytical expressions related to the NFW spatial profile; the `~halotools.empirical_models.phase_space_models.profile_models.ConcMass` class controls the mapping between dark matter halos and the NFW concentration associated to them. 

Most of the functionality of the `~halotools.empirical_models.phase_space_models.profile_models.NFWProfile` 
class derives from the behavior defined in the 
`~halotools.empirical_models.phase_space_models.profile_models.AnalyticDensityProf` super-class. 
Here we only document the functionality and implementation that is unique to the 
`~halotools.empirical_models.phase_space_models.profile_models.NFWProfile` model, 
and defer discussion of all super-class-derived behavior to the :ref:`profile_template_tutorial`. 

Halotools implementation of the NFW mass density 
--------------------------------------------------

Equation 3 of Navarro, Frenk and White (1995), arXiv:9508025, gives the original definition of the NFW profile:

.. math::

	\rho_{\rm NFW}(r) \equiv \rho_{\rm crit}\frac{\delta_{\rm c}}{r/r_{\rm s}(1 + r/r_{s})^{2}}, \\

	\delta_{\rm c} \equiv \frac{200}{3}\frac{c^{3}}{{\rm ln}(1+c) - c/(1+c)}

In the above equation, the factor of :math:`200\rho_{\rm crit}` reflects the **200m** halo mass definition convention adopted in NFW95, and :math:`c \equiv R_{200}/r_{s}` is the concentration parameter. In the notation adopted in :ref:`halo_mass_definitions`, :math:`200\times\rho_{\rm crit} \equiv \Delta_{\rm thresh}\times\rho_{\rm ref} \equiv \rho_{\rm thresh}`, so we can reformulate the above expression in the following more general form:

.. math::

	\rho_{\rm NFW}(r) = \rho_{\rm thresh}\frac{c^{3}/3g(c)}{r/r_{s}(1 + r/r_{s})^{2}}, \\

	g(c) \equiv {\rm ln}(1+c) - c/(1+c).

If we now substitute :math:`r/r_{s} = cr/R_{\Delta}` and define the *scaled radius* :math:`\tilde{r}\equiv r/R_{\Delta}` the above expression can be further rewritten as:

.. math::

	\rho_{\rm NFW}(\tilde{r})/\rho_{\rm thresh} \equiv \tilde{\rho}_{\rm NFW}(\tilde{r}) = \frac{c^{3}/3g(c)}{c\tilde{r}(1 + c\tilde{r})^{2}}. 

The above expression is the exact equation implemented in the `~halotools.empirical_models.phase_space_models.profile_models.NFWProfile.dimensionless_mass_density` method of the `~halotools.empirical_models.phase_space_models.profile_models.NFWProfile` class. The quantity :math:`\rho_{\rm thresh}` is calculated in Halotools using the `~halotools.empirical_models.phase_space_models.profile_models.profile_helpers.density_threshold` function, and :math:`g(c)` is computed using the `~halotools.empirical_models.phase_space_models.profile_models.NFWProfile.g` method of the `~halotools.empirical_models.phase_space_models.profile_models.NFWProfile` class. 

For any sub-class of `~halotools.empirical_models.phase_space_models.profile_models.AnalyticDensityProf`, 
once the `~halotools.empirical_models.phase_space_models.profile_models.NFWProfile.dimensionless_mass_density` method is defined, in principle all subsequent behavior is derived. In practice, if the associated integrals and derivatives can be computed analytically it is more efficient and numerically stable to implement the analytical results as over-rides of the super-class-defined methods. The subsections below derive the analytical equations used in all over-rides implemented in the `~halotools.empirical_models.phase_space_models.profile_models.NFWProfile` class. 

.. _nfw_cumulative_mass_pdf_derivation:

Derivation of the NFW cumulative mass PDF 
------------------------------------------------

The cumulative mass PDF, :math:`P_{\rm prof}(<\tilde{r})`, 
the cumulative probability of finding a randomly selected 
particle at a scaled-radius position less than :math:`\tilde{r}`, is defined as:

.. math::

	P_{\rm NFW}(<\tilde{r}) \equiv M_{\rm NFW}(<\tilde{r}) / M_{\Delta}.

In the above expression, 

.. math::

	M_{\rm NFW}(<\tilde{r}) \equiv 4\pi\rho_{\rm thresh}\int_{0}^{\tilde{r}}d\tilde{r}' \tilde{r}'^{2}\tilde{\rho}_{\rm NFW}(\tilde{r}') 

and 

.. math::

	M_{\Delta} \equiv 4\pi\rho_{\rm thresh}\int_{0}^{1}d\tilde{r}' \tilde{r}'^{2}\tilde{\rho}_{\rm NFW}(\tilde{r}'). 

Plugging in the definition of :math:`\tilde{\rho}_{\rm NFW}` and canceling the common pre-factors of 
:math:`4\pi\rho_{\rm thresh}c^{3}/3g(c)` gives:

.. math::

	P_{\rm NFW}(<\tilde{r}) = \frac{\int_{0}^{\tilde{r}}d\tilde{r}' \tilde{r}'^{2}1/c\tilde{r}'(1 + c\tilde{r}')^{2}}{\int_{0}^{1}d\tilde{r}' \tilde{r}'^{2}1/c\tilde{r}'(1 + c\tilde{r}')^{2}}

Now we change integration variables :math:`\tilde{r}'\rightarrow c\tilde{r}'=y`:

.. math::

	P_{\rm NFW}(<\tilde{r}) = \frac{\int_{0}^{c\tilde{r}}dy\frac{y}{(1 + y)^{2}}}{\int_{0}^{1}dy\frac{y}{(1 + y)^{2}}}

and use the definition of :math:`g(x) \equiv {\rm ln}(1+x) - x/(1+x) = \int_{0}^{x}dy\frac{y}{(1+y)^{2}}` to write the above expression as

.. math::

	P_{\rm NFW}(<\tilde{r}) = g(c\tilde{r}) / g(c)

The above equation is the exact expression used to calculate :math:`P_{\rm NFW}(<\tilde{r})` via the `~halotools.empirical_models.phase_space_models.profile_models.NFWProfile.cumulative_mass_PDF` function. 

.. _monte_carlo_nfw_spatial_profile:

Monte Carlo realizations of the NFW profile
------------------------------------------------

Halotools uses `Inverse Transform Sampling <https://en.wikipedia.org/wiki/Inverse_transform_sampling>`_, a standard Monte Carlo technique, to produce random realizations of halo profiles. The basic idea of this technique is to draw a random uniform number, *u*, and intrepret *u* as the probability :math:`u = P(<r)` of finding a  point tracing an NFW radial profile interior to position *r*. The mapping between *u* and *r* is already implemented via the `~halotools.empirical_models.phase_space_models.profile_models.NFWProfile.cumulative_mass_PDF` function, so we only need to use this function to provide the inverse mapping. This we do numerically by tabulating :math:`P_{\rm NFW}(<\tilde{r})` at a set of control points :math:`0<\tilde{r}<1` and then using the `scipy <http://www.scipy.org/>`_ function `~scipy.interpolate.InterpolatedUnivariateSpline`. This technique is used ubiquitously throughout the package, and the interpolation is actually implemented using the `~halotools.empirical_models.model_helpers.custom_spline` function, which is just a wrapper that customizes the edge case behavior of `~scipy.interpolate.InterpolatedUnivariateSpline`. 

The simplest place in the code base to see where Inverse Transform Sampling gives Monte Carlo realizations of the NFW profile is in the `~halotools.empirical_models.phase_space_models.profile_models.NFWProfile.mc_generate_nfw_radial_positions` source code. Here the implementation is basically straightforward. Because NFW profiles are power laws, the interpolation is more stable when it is done in log-space. 

.. _nfw_jeans_velocity_profile_derivations:

Modeling the NFW Velocity Profile 
===========================================

How are things computed in practice (lookup tables, etc.)


.. _nfw_monte_carlo_derivations:

Monte Carlo realizations of the NFW profile
===========================================

How are things computed in practice (lookup tables, etc.)

