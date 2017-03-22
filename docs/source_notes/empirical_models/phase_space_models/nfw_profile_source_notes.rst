:orphan:

.. currentmodule:: halotools.empirical_models

.. _nfw_profile_tutorial:

******************************************************************
Source code notes on `NFWProfile` and `NFWPhaseSpace`
******************************************************************


This section of the documentation provides background material
and detailed implementation notes on the functions and methods of the
`~halotools.empirical_models.NFWProfile`
and `~halotools.empirical_models.NFWPhaseSpace` models.

Outline
========

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

Class structure of the `~halotools.empirical_models.NFWPhaseSpace` model
==========================================================================================================

The `~halotools.empirical_models.NFWPhaseSpace` model is a container class
for two independently-defined sets of behaviors:

	1. Analytical descriptions of the spatial and velocity distribution of points within the halo
	2. Monte Carlo methods for generating random realizations of points in phase space

For the first set of behaviors, the spatial distributions are modeled by
`~halotools.empirical_models.NFWProfile`, while the `~halotools.empirical_models.NFWPhaseSpace` sub-class
controls the velocities. The second set of behaviors is controlled entirely by
`~halotools.empirical_models.MonteCarloGalProf`, regardless of the details of the profile model.


.. _nfw_spatial_profile_derivations:

Modeling the NFW Spatial Profile
======================================

The spatial profile of an NFW halo is modeled with the `~halotools.empirical_models.NFWProfile` class, which is itself a sub-class of `~halotools.empirical_models.AnalyticDensityProf`

Most of the functionality of the `~halotools.empirical_models.NFWProfile`
class derives from the behavior defined in the
`~halotools.empirical_models.AnalyticDensityProf` super-class.
Here we only document the functionality and implementation that is unique to the
`~halotools.empirical_models.NFWProfile` model,
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

The above expression is the exact equation implemented in the `~halotools.empirical_models.NFWProfile.dimensionless_mass_density` method of the `~halotools.empirical_models.NFWProfile` class. The quantity :math:`\rho_{\rm thresh}` is calculated in Halotools using the `~halotools.empirical_models.profile_helpers.density_threshold` function, and :math:`g(c)` is computed using the `~halotools.empirical_models.NFWProfile.g` method of the `~halotools.empirical_models.NFWProfile` class.

For any sub-class of `~halotools.empirical_models.AnalyticDensityProf`,
once the `~halotools.empirical_models.NFWProfile.dimensionless_mass_density` method is defined, in principle all subsequent behavior is derived. In practice, if the associated integrals and derivatives can be computed analytically it is more efficient and numerically stable to implement the analytical results as over-rides of the super-class-defined methods. The subsections below derive the analytical equations used in all over-rides implemented in the `~halotools.empirical_models.NFWProfile` class.

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

The above equation is the exact expression used to calculate :math:`P_{\rm NFW}(<\tilde{r})` via the `~halotools.empirical_models.NFWProfile.cumulative_mass_PDF` function.

.. _monte_carlo_nfw_spatial_profile:

Monte Carlo realizations of the NFW profile
------------------------------------------------

Halotools uses `Inverse Transform Sampling <https://en.wikipedia.org/wiki/Inverse_transform_sampling>`_, a standard Monte Carlo technique, to produce random realizations of halo profiles. The basic idea of this technique is to draw a random uniform number, *u*, and intrepret *u* as the probability :math:`u = P(<r)` of finding a  point tracing an NFW radial profile interior to position *r*. The mapping between *u* and *r* is already implemented via the `~halotools.empirical_models.NFWProfile.cumulative_mass_PDF` function, so we only need to use this function to provide the inverse mapping. This we do numerically by tabulating :math:`P_{\rm NFW}(<\tilde{r})` at a set of control points :math:`0<\tilde{r}<1` and then using the `scipy <http://www.scipy.org/>`_ function `~scipy.interpolate.InterpolatedUnivariateSpline`. This technique is used ubiquitously throughout the package, and the interpolation is actually implemented using the `~halotools.empirical_models.custom_spline` function, which is just a wrapper that customizes the edge case behavior of `~scipy.interpolate.InterpolatedUnivariateSpline`.

The simplest place in the code base to see where Inverse Transform Sampling gives Monte Carlo realizations of the NFW profile is in the `~halotools.empirical_models.NFWProfile.mc_generate_nfw_radial_positions` source code. Here the implementation is basically straightforward. Because NFW profiles are power laws, the interpolation is more stable when it is done in log-space.

.. _nfw_jeans_velocity_profile_derivations:

Modeling the NFW Velocity Profile
===========================================

The `~halotools.empirical_models.NFWPhaseSpace` model solves for the velocity profile of satellite galaxies by making the following assumptions:

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

The `~halotools.empirical_models.NFWPhaseSpace` model assumes that :math:`\rho_{\rm sat} = \rho_{\rm NFW}`. So we can rewrite the above equation using the dimensionless quantities :math:`\tilde{r}\equiv r/R_{\Delta}` and :math:`\tilde{\rho}_{\rm NFW}(\tilde{r}) \equiv \rho_{\rm NFW}(\tilde{r})/\rho_{\rm thresh}` and canceling the common factors of :math:`\rho_{\rm thresh}`:

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

Defining the *dimensionless radial velocity dispersion* :math:`\tilde{\sigma}_{r}\equiv\sigma_{r}/V_{\rm vir}`, the above equation is the exact expression used in the `~halotools.empirical_models.NFWPhaseSpace.dimensionless_radial_velocity_dispersion` method of the
`~halotools.empirical_models.NFWPhaseSpace` class. The above expression is also the same expression appearing in Eq. 24 of More et al. (2008), `arXiv:0807.4529 <http://arxiv.org/abs/0807.4529/>`_, with the only differences being of notation: :math:`g(c) \leftrightarrow \mu(c)` and :math:`c\tilde{r} \leftrightarrow r/r_{\rm s}`.


.. _nfw_monte_carlo_derivations:

Monte Carlo realizations of the NFW profile
===========================================

The `~halotools.empirical_models.NFWPhaseSpace` model can either be used as a stand-alone class to generate an arbitrary number of points in NFW phase space, or as part of a composite galaxy-halo model that generates full-scale mock galaxy catalogs. We document each of these options in turn.

.. _stand_alone_mc_nfw_phase_space:

Stand-alone Monte Carlo realizations of NFW phase space
---------------------------------------------------------

The `~halotools.empirical_models.NFWPhaseSpace.mc_generate_nfw_phase_space_points`
method can be used to create an Astropy `~astropy.table.Table` storing a collection of points in NFW phase space.

>>> from halotools.empirical_models import NFWPhaseSpace
>>> nfw = NFWPhaseSpace()
>>> data = nfw.mc_generate_nfw_phase_space_points(Ngals = 100, mass = 1e13, conc = 10)  # doctest: +SKIP

In the source code, the generation of these points happens in two steps. First, *x, y, z* points are drawn using the `~NFWPhaseSpace.mc_halo_centric_pos` method defined in the `~MonteCarloGalProf` orthogonal mix-in class. Following the same methodology described in :ref:`monte_carlo_nfw_spatial_profile`, the `~NFWPhaseSpace.mc_halo_centric_pos` method uses inverse transform sampling together with the `~NFWPhaseSpace.cumulative_mass_PDF` function to draw random realizations of dimensionless NFW profile radii; these dimensionless radii are then scaled according to the halo mass and radius definition selected by the keyword arguments passed to the `~NFWPhaseSpace` constructor. See the :ref:`monte_carlo_galprof_spatial_positions` section of the :ref:`monte_carlo_galprof_mixin_tutorial` for a detailed explanation of how this method works.

Once dimensionless radial positions have been drawn, the `~NFWPhaseSpace.mc_generate_nfw_phase_space_points` method passes these positions to the `~MonteCarloGalProf.mc_radial_velocity` method of the `~MonteCarloGalProf` orthogonal mix-in class. This method works differently than the `~NFWPhaseSpace.mc_halo_centric_pos` method: the `~MonteCarloGalProf.mc_radial_velocity` method does *not* use inverse transform sampling. This is because radial velocity distributions are assumed to be Gaussian, and there is an optimized function `numpy.random.normal` in the Numpy code base for directly drawing from a Gaussian distribution. A Gaussian distribution is specified by its first two moments: the first moment is centered at the velocity of the host halo, and the second moment is calculated using the `~NFWPhaseSpace.dimensionless_radial_velocity_dispersion` method described in the previous section.

In practice, for performance reasons the `~MonteCarloGalProf.mc_radial_velocity` method actually uses a lookup table tabulated from `~NFWPhaseSpace.dimensionless_radial_velocity_dispersion` rather than the actual `~NFWPhaseSpace.dimensionless_radial_velocity_dispersion` method itself. For further information concerning this detail, see the :ref:`monte_carlo_galprof_mixin_tutorial`.

At this point, random positions and velocities have been drawn for the satellites and the `~NFWPhaseSpace.mc_generate_nfw_phase_space_points` method bundles these arrays into an Astropy `~astropy.table.Table` and returns the result.



.. _making_mocks_nfw_phasespace_satellites:

Making mocks with NFWPhaseSpace satellites
---------------------------------------------------------

There are a small number of boilerplate lines of code that must go into the constructor of any class in order for the class instance to be integrated into the factory design pattern of Halotools composite models. In this final section of the tutorial, we will look closely at the `~NFWPhaseSpace` constructor to see how the analytical functions and Monte Carlo methods described above get incorporated into the Halotools framework.

As `~NFWPhaseSpace` is primarily a container class for externally-defined behavior, the primary task of its constructor is to call the constructors of its super-classes. There are two super-classes whose behavior is being composed, `~NFWProfile` and `~MonteCarloGalProf`, whose roles we will now describe in turn.

.. _nfw_profile_class_constructor:

Constructor of the `~NFWProfile` class
------------------------------------------

The `~NFWProfile` class is a sub-class of `~AnalyticDensityProf`. Calling the constructor of this super-class is the first thing the `~NFWProfile` constructor does. See the :ref:`analytic_density_prof_constructor` section for what this accomplishes.

Any NFW profile is specified entirely by mass and concentration, and there exist many different calibrated models for the mapping between halo mass and halo concentration. The ``conc_mass_model`` keyword argument allows you to control the behavior of this relation. This keyword allows you to select between different nicknames for conc-mass relations in Halotools, or alternatively you can pass in any callable function that accepts a ``table`` keyword argument and returns an array of floats the same length as the `~astropy.table.Table` object bound to the keyword.

Below we show a simple example of how to use a custom concentration-mass relation with `~NFWPhaseSpace`:

.. code:: python

	def custom_conc_mass(table):
		mass = table['halo_mvir']
		return np.zeros_like(mass) + 5.

	nfw_model = NFWPhaseSpace(conc_mass_model=custom_conc_mass)


When using the `~NFWProfile` class in stand-alone fashion, all the analytical functions bound to the instance require that the halo concentration be supplied as an independent argument, and so the behavior inherited by the ``conc_mass_model`` keyword argument is irrelevant in such cases. However, when the `~NFWProfile` class is used as a component model in the mock-making framework, the mapping between a halo catalog and the halo concentration must be provided. For such a use-case, the ``conc_mass_model`` keyword argument provides the user with the ``direct_from_halo_catalog`` model option to simply use the concentration in the halo_table itself.

As described in the :ref:`prof_param_keys_mechanism`, one of the boilerplate standardizations of halo profile models is that all sub-classes of the `~profile_models.AnalyticalDensityProf` class must have a bound method with the same name as every element in the ``prof_param_keys`` list of strings bound to the instance. In the case of the `~NFWProfile`, there is only a single profile parameter: ``conc_NFWmodel``. Accordingly, there is a `NFWProfile.conc_NFWmodel` method; the behavior of this method derives entirely from the model provided by the ``conc_mass_model`` keyword argument.


.. _monte_carlo_galprof_class_constructor:

Constructor of the `~MonteCarloGalProf` class
-------------------------------------------------

The final super-class constructor called is `MonteCarloGalProf.__init__`, which performs four functions:

1. A python dictionary called ``new_haloprop_func_dict`` is created and bound to the instance.

As described in :ref:`new_haloprop_func_dict_mechanism`, the purpose of ``new_haloprop_func_dict`` is to create a new column in the halo_table *before* mock-population begins, and to automatically guarantee that all galaxies in the galaxy_table created during mock-population will inherit whatever new halo property is created. The keys of ``new_haloprop_func_dict`` are the names of the to-be-inherited property, the values are function objects that operate on the original halo_table and return the value of the newly created halo property. If component models require access to a halo property that is not already in the halo_table and this quantity is expensive to calculate, this mechanism can save considerable runtime during mock-population as it can be computed in advance.

In the case of our `~NFWPhaseSpace` model, we calculate the ``conc_NFWmodel`` property with ``new_haloprop_func_dict``. The newly created halo_table key will be called ``conc_NFWmodel``, and the value bound to this key will be whatever result is returned by the `NFWPhaseSpace.conc_NFWmodel` function.

2. A `numpy.dtype` object called ``_galprop_dtypes_to_allocate`` is created and bound to the instance.

As described in :ref:`galprop_dtypes_to_allocate_mechanism`, the purpose of ``_galprop_dtypes_to_allocate`` is to inform the `~halotools.empirical_models.HodModelFactory` the name and data type of the galaxy attributes that will be created by the component model, so that the appropriate memory can be pre-allocated without any hard-coding in the `~halotools.empirical_models.HodModelFactory`. For the case of our `~NFWPhaseSpace` model, we require *x, y, z, vx, vy, vz*, and we also allocate *host_centric_distance* as this is an interesting physical characteristic of satellite galaxies upon which other properties defined elsewhere may depend.

3. Build lookup tables for the spatial and velocity profiles using `MonteCarloGalProf.setup_prof_lookup_tables`.

The purpose of these lookup tables is to improve performance of the Monte Carlo generation of mock galaxy spatial positions and velocities. See :ref:`monte_carlo_galprof_lookup_tables` for details.

4. A python list called ``_mock_generation_calling_sequence`` is created and bound to the instance.

This list determines which bound methods of `NFWPhaseSpace` will be called during mock-population, and in which order. The `NFWPhaseSpace` model only has a single such method, `~NFWPhaseSpace.assign_phase_space`, which itself simply calls the `~NFWPhaseSpace.mc_pos` and `~NFWPhaseSpace.mc_vel` methods in turn. See :ref:`mock_generation_calling_sequence_mechanism` for details.













