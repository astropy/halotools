:orphan:

.. _nfw_profile_tutorial:

****************************************************
Tutorial on the NFWProfile and NFWPhaseSpace Models
****************************************************

.. currentmodule:: halotools.empirical_models.phase_space_models

This section of the documentation provides background material 
and detailed implementation notes on the functions and methods of the 
`~halotools.empirical_models.phase_space_models.profile_models.NFWProfile` 
and `~halotools.empirical_models.phase_space_models.NFWPhaseSpace` models. 
We will start in :ref:`nfw_phase_space_class_structure` with an overview 
of the orthogonal mix-in class design. The :ref:`nfw_spatial_profile_derivations` 
section covers the mathematics of NFW spatial profiles and their Monte Carlo realizations, 
including explicit derivations of the exact form of all equations are they are implemented in code. 
The section :ref:`nfw_jeans_velocity_profile_derivations` goes into the same level of detail but 
for the velocity profiles. The tutorial concludes in :ref:`nfw_monte_carlo_derivations`  
by describing how the `~NFWPhaseSpace` class can be used in both stand-alone fashion and 
as part of the Halotools mock-making framework to generate Monte Carlo realizations of 
points in NFW phase space. 

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

.. _nfw_dimensionless_mass_density:

Halotools implementation of the NFW mass density 
--------------------------------------------------
Equation 3 of Navarro, Frenk and White (1995), `arXiv:9508025 <http://arxiv.org/abs/astro-ph/9508025/>`_, gives the original definition of the NFW profile:

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

The `~halotools.empirical_models.phase_space_models.NFWPhaseSpace` model solves for the velocity profile of satellite galaxies by making the following assumptions: 

	1. satellites trace the same spatial profile as their underlying gravitational potential well,
	2. satellites are in virial equilibrium with their potential, and 
	3. satellite orbits are isotropic. 

In any such system, the first moment of the radial velocity distribution of the satellites 
is zero (there is no net infall or outflow), and the second moment :math:`\sigma^{2}_{r}(r)` 
can be calculated analytically by solving the Jeans equation, 
which under these assumptions takes the following form:

.. math::

	\sigma^{2}_{r}(r) = \frac{1}{\rho_{\rm sat}(r)}\int_{r}^{\infty}{\rm d}r\rho_{\rm sat}(r)\frac{{\rm d}\Phi(r)}{{\rm d}r},

In the above equation, :math:`\rho_{\rm sat}` is the number density profile of the satellite galaxies and :math:`\Phi` is the gravitational potential. For a justification of this simplification of the Jeans equation, see :ref:`jeans_equation_derivations`. 

The `~halotools.empirical_models.phase_space_models.NFWPhaseSpace` model assumes that :math:`\rho_{\rm sat} = \rho_{\rm NFW}`. So we can rewrite the above equation using the dimensionless quantities :math:`\tilde{r}\equiv r/R_{\Delta}` and :math:`\tilde{\rho}_{\rm NFW}(\tilde{r}) \equiv \rho_{\rm NFW}(\tilde{r})/\rho_{\rm thresh}` and canceling the common factors of :math:`\rho_{\rm thresh}`:

.. math::

	\sigma^{2}_{r}(\tilde{r}) = \frac{1}{\tilde{\rho}_{\rm NFW}(\tilde{r})}\int_{\tilde{r}}^{\infty}{\rm d}\tilde{r}\tilde{\rho}_{\rm NFW}(\tilde{r})\frac{{\rm d}\Phi(\tilde{r})}{{\rm d}\tilde{r}},

For any spherically symmetric gravitational potential, 

.. math::

	\Phi(x) = \frac{-GM_{\Delta}(<x)}{x} \\

	\Rightarrow \frac{{\rm d}\Phi(\tilde{r})}{{\rm d}\tilde{r}} = \frac{GM_{\Delta}(<\tilde{r})}{\tilde{r}^{2}} \equiv \frac{V_{\rm circ}^{2}(\tilde{r})}{\tilde{r}} = V_{\rm vir}^{2}\frac{P_{\rm NFW}(<\tilde{r})}{\tilde{r}^{2}}, 

where in the second-to-last equality we have used the definition of :math:`V^{2}_{\rm circ}`, and in the last equality we have used the derivation provided in the :ref:`computing_circular_velocity` section of the :ref:`profile_template_tutorial`. 

From the :ref:`nfw_cumulative_mass_pdf_derivation` we have that :math:`P_{\rm NFW}(<\tilde{r}) = g(c\tilde{r}) / g(c)`, and using the expression for :math:`\tilde{\rho}_{\rm NFW}` given in :ref:`nfw_dimensionless_mass_density`, we can plug in these expressions to the above equation: 

.. math::

	\sigma^{2}_{r}(\tilde{r}) = \frac{c\tilde{r}(1 + c\tilde{r})^{2}}{c^{3}/3g(c)}\int_{\tilde{r}}^{\infty}{\rm d}\tilde{r}\frac{c^{3}/3g(c)}{c\tilde{r}(1 + c\tilde{r})^{2}}V_{\rm vir}^{2}\frac{g(c\tilde{r})}{g(c)\tilde{r}^{2}}. 

Canceling common factors of :math:`c^{3}/3g(c)` and rearranging terms gives us:

.. math::

	\Rightarrow \sigma^{2}_{r}(\tilde{r}) = V_{\rm vir}^{2}\frac{c\tilde{r}(1 + c\tilde{r})^{2}}{g(c)}\int_{\tilde{r}}^{\infty}{\rm d}\tilde{r}\frac{g(c\tilde{r})}{c\tilde{r}^{3}(1 + c\tilde{r})^{2}}

Finally, we change integration variables :math:`\tilde{r}\rightarrow c\tilde{r}=y` to give:

.. math::

	\Rightarrow \sigma^{2}_{r}(\tilde{r}) = V_{\rm vir}^{2}\frac{c^{2}\tilde{r}(1 + c\tilde{r})^{2}}{g(c)}\int_{c\tilde{r}}^{\infty}{\rm d}y\frac{g(y)}{y^{3}(1 + y)^{2}}

Defining the *dimensionless radial velocity dispersion* :math:`\tilde{\sigma}_{r}\equiv\sigma_{r}/V_{\rm vir}`, the above equation is the exact expression used in the `~halotools.empirical_models.phase_space_models.velocity_models.NFWJeansVelocity.dimensionless_radial_velocity_dispersion` method of the 
`~halotools.empirical_models.phase_space_models.velocity_models.NFWJeansVelocity` class, which is where the velocity profile behavior of the `~halotools.empirical_models.phase_space_models.NFWPhaseSpace` class is defined. The above expression is also the same expression appearing in Eq. 24 of More et al. (2008), `arXiv:0807.4529 <http://arxiv.org/abs/0807.4529/>`_, with the only differences being of notation: :math:`g(c) \leftrightarrow \mu(c)` and :math:`c\tilde{r} \leftrightarrow r/r_{\rm s}`. 


.. _nfw_monte_carlo_derivations:

Monte Carlo realizations of the NFW profile
===========================================

The `~halotools.empirical_models.phase_space_models.NFWPhaseSpace` model can either be used as a stand-alone class to generate an arbitrary number of points in NFW phase space, or as part of a composite galaxy-halo model that generates full-scale mock galaxy catalogs. We document each of these options in turn. 

.. _stand_alone_mc_nfw_phase_space:

Stand-alone Monte Carlo realizations of NFW phase space 
---------------------------------------------------------
.. currentmodule:: halotools.empirical_models.phase_space_models

The `~halotools.empirical_models.phase_space_models.NFWPhaseSpace.mc_generate_nfw_phase_space_points` 
method can be used to create an Astropy `~astropy.table.Table` storing a collection of points in NFW phase space. 

>>> from halotools.empirical_models.phase_space_models import NFWPhaseSpace
>>> nfw = NFWPhaseSpace()
>>> data = nfw.mc_generate_nfw_phase_space_points(Ngals = 100, mass = 1e13, conc = 10) 

In the source code, the generation of these points happens in two steps. First, *x, y, z* points are drawn using the `~NFWPhaseSpace.mc_halo_centric_pos` method defined in the `~MonteCarloGalProf` orthogonal mix-in class. Following the same methodology described in :ref:`monte_carlo_nfw_spatial_profile`, the `~NFWPhaseSpace.mc_halo_centric_pos` method uses inverse transform sampling together with the `~NFWPhaseSpace.cumulative_mass_PDF` function to draw random realizations of dimensionless NFW profile radii; these dimensionless radii are then scaled according to the halo mass and radius definition selected by the keyword arguments passed to the `~NFWPhaseSpace` constructor. See the :ref:`monte_carlo_galprof_spatial_positions` section of the :ref:`monte_carlo_galprof_mixin_tutorial` for a detailed explanation of how this method works. 

Once dimensionless radial positions have been drawn, the `~NFWPhaseSpace.mc_generate_nfw_phase_space_points` method passes these positions to the `~MonteCarloGalProf.mc_radial_velocity` method of the `~MonteCarloGalProf` orthogonal mix-in class. This method works differently than the `~NFWPhaseSpace.mc_halo_centric_pos` method: the `~MonteCarloGalProf.mc_radial_velocity` method does *not* use inverse transform sampling. This is because radial velocity distributions are assumed to be Gaussian, and there is an optimized function `numpy.random.normal` in the Numpy code base for directly drawing from a Gaussian distribution. A Gaussian distribution is specified by its first two moments: the first moment is centered at the velocity of the host halo, and the second moment is calculated using the `~NFWPhaseSpace.dimensionless_radial_velocity_dispersion` method described in the previous section. 

In practice, for performance reasons the `~MonteCarloGalProf.mc_radial_velocity` method actually uses a lookup table tabulated from `~NFWPhaseSpace.dimensionless_radial_velocity_dispersion` rather than the actual `~NFWPhaseSpace.dimensionless_radial_velocity_dispersion` method itself. For further information concerning this detail, see the :ref:`monte_carlo_galprof_mixin_tutorial`. 

At this point, random positions and velocities have been drawn for the satellites and the `~NFWPhaseSpace.mc_generate_nfw_phase_space_points` method bundles these arrays into an Astropy `~astropy.table.Table` and returns the result. 



.. _making_mocks_nfw_phasespace_satellites:

Making mocks with NFWPhaseSpace satellites 
---------------------------------------------------------

As described in :ref:`generic_model_component_tutorial`, there are a small number of boilerplate lines of code that must go into the constructor of any class in order for the class instance to be integrated into the factory design pattern of Halotools composite models. In this final section of the tutorial, we will look closely at the `~NFWPhaseSpace` constructor to see how the analytical functions and Monte Carlo methods described above get incorporated into the Halotools framework. 

As `~NFWPhaseSpace` is primarily a container class for externally-defined behavior, the primary task of its constructor is to call the constructors of its super-classes. There are three super-classes whose behavior is being composed, `~NFWProfile`, `~NFWJeansVelocity` and `~MonteCarloGalProf`, whose roles we will now describe in turn. 

.. _nfw_profile_class_constructor:

Constructor of the `~NFWProfile` class 
------------------------------------------

The `~NFWProfile` class is a sub-class of `~AnalyticDensityProf`. Calling the constructor of this super-class is the first thing the `~NFWProfile` constructor does. See the :ref:`analytic_density_prof_constructor` section for what this accomplishes. 

The `~NFWProfile` class is also a sub-class of `~ConcMass`. Calling the constructor of `~ConcMass` is the next thing done by the `~NFWProfile` constructor. Any NFW profile is specified entirely by the mass and concentration of the halo, and there exist many different calibrated models for the mapping between halo mass and halo concentration. The `~ConcMass` class is responsible for providing access to such models. 

When using the `~NFWProfile` class in stand-alone fashion, all the analytical functions bound to the instance require that the halo concentration be supplied as an independent argument, and so the behavior inherited by the `~ConcMass` class is irrelevant in such cases. However, when the `~NFWProfile` class is used as a component model in the mock-making framework, the mapping between a halo catalog and the halo concentration must be provided. For such a use-case, the `~ConcMass` class provides the user with the ``direct_from_halo_catalog`` model option to simply use the concentration in the halo_table itself; the user also has the option to instead choose an analytical model connecting halo concentration to the halos in the halo_table. See the docstring of `~ConcMass` for further details about the available options. 

As described in the :ref:`prof_param_keys_mechanism`, one of the boilerplate standardizations of halo profile models is that all sub-classes of the `~profile_models.AnalyticalDensityProf` class must have a bound method with the same name as every element in the ``prof_param_keys`` list of strings bound to the instance. In the case of the `~NFWProfile`, there is only a single profile parameter: ``conc_NFWmodel``. Accordingly, there is a `NFWProfile.conc_NFWmodel` method; the behavior of this method derives entirely from the `ConcMass.compute_concentration` method.  


.. _nfw_jeans_velocity_class_constructor:

Constructor of the `~NFWJeansVelocity` class 
------------------------------------------------

This constructor currently has no functionality whatsoever. It is currently only acting as a placeholder for potential future options. 

.. _monte_carlo_galprof_class_constructor:

Constructor of the `~MonteCarloGalProf` class 
-------------------------------------------------






