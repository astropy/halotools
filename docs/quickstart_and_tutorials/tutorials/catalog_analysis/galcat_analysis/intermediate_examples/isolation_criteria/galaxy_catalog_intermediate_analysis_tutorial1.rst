.. _galaxy_catalog_intermediate_analysis_tutorial1:

Galaxy Catalog Analysis Example: Identifying isolated galaxies, Part II
=======================================================================

In this tutorial, we'll start from a mock galaxy catalog and show how to
determine which galaxies are "isolated" according to a variety of
criteria. This tutorial demos more advanced usage of the isolation
criteria functionality. For example, for each galaxy in some sample, you
may wish to have an isolation criterion that depends on the stellar mass
of the galaxy in question. This tutorial gives several examples of how
to apply such a condition to a mock galaxy sample.

This is a companion tutorial to :ref:`galaxy_catalog_analysis_tutorial10`.
Be sure you have read and understand that tutorial before proceeding.

There is also an IPython Notebook in the following location that can be
used as a companion to the material in this section of the tutorial:


    **halotools/docs/notebooks/galcat_analysis/intermediate_examples/galaxy_catalog_intermediate_analysis_tutorial1.ipynb**

By following this tutorial together with this notebook,
you can play around with your own variations of the calculation
as you learn the basic syntax.

Generate a mock galaxy catalog
------------------------------

Let's start out by generating a mock galaxy catalog into an N-body
simulation in the usual way. Here we'll assume you have the
:math:``z=0`` rockstar halos for the bolshoi simulation, as this is the
default halo catalog.

.. code:: python

    from halotools.empirical_models import PrebuiltSubhaloModelFactory
    model = PrebuiltSubhaloModelFactory('smhm_binary_sfr')

    from halotools.sim_manager import CachedHaloCatalog
    halocat = CachedHaloCatalog(simname='bolshoi', redshift=0, halo_finder = 'rockstar')

    model.populate_mock(halocat)


Example 1: Variable isolation criteria in 3d
--------------------------------------------

For the first example we'll find "isolated" galaxies using a 3d search
criteria that depends on stellar mass in the following manner. From our
mock, we know the stellar mass of every galaxy. The stellar mass component of the underlying model
that generated the mock, `halotools.empirical_models.Behroozi10SmHm`,
has a `halotools.empirical_models.Behroozi10SmHm.mean_log_halo_mass`
method that provides a map from :math:`M_{\ast}^{\rm cen}` to
:math:`M_{\rm vir}^{\rm host}.` There is also a
`~halotools.empirical_models.halo_mass_to_halo_radius` function
in `~halotools.empirical_models` that provides a map from :math:`M_{\rm vir}^{\rm host}` to
:math:`R_{\rm vir}.` We will use these two functions together to draw a
sphere of radius :math:`R_{\rm vir}` around each galaxy; a galaxy will
be said to be isolated if there are no other galaxies in the sample
within this search radius. Of course, some of the galaxies in our sample
are not central galaxies, and there is also scatter between
:math:`M_{\ast}^{\rm cen}` and :math:`M_{\rm vir}^{\rm host},` and so
this criteria will not be perfect; since you know the true
central/satellite designation in the mock, you can always test this
assumption with a standard purity/completeness analysis.

Let's select a sample of galaxies with
:math:`M_{\ast}>10^{10}M_{\odot},` and focus on identifying isolated
galaxies in a specific stellar mass range of
:math:`10^{10.4}M_{\odot} < M_{\ast} < 10^{10.5}M_{\odot}:`

.. code:: python

    all_gal_mask = model.mock.galaxy_table['stellar_mass'] > 1e10
    all_gals = model.mock.galaxy_table[all_gal_mask]

    sample_mask = (all_gals['stellar_mass'] > 4e10) & (all_gals['stellar_mass'] < 5e10)
    sm_selected_gals = all_gals[sample_mask]

    logsm = np.log10(sm_selected_gals['stellar_mass'])
    presumed_mvir = 10**model.mean_log_halo_mass(logsm)

    from halotools.empirical_models import halo_mass_to_halo_radius

    presumed_rvir = halo_mass_to_halo_radius(presumed_mvir,
                    cosmology = model.mock.cosmology,
                    redshift = model.mock.redshift,
                    mdef = 'vir')

    from halotools.mock_observables import conditional_spherical_isolation

The calling signature of `~halotools.mock_observables.conditional_spherical_isolation`
accepts a multi-dimensional array storing the x, y, z positions of each point. You
can place your points into the appropriate form using
`numpy.vstack`, but below we'll demo how to use the
`~halotools.mock_observables.return_xyz_formatted_array` function for
this purpose, as this function provides additional convenient behavior
that we'll use later in the tutorial.

.. code:: python

    from halotools.mock_observables import return_xyz_formatted_array

    sample1_pos = return_xyz_formatted_array(sm_selected_gals['x'], sm_selected_gals['y'], sm_selected_gals['z'])
    sample2_pos = return_xyz_formatted_array(all_gals['x'], all_gals['y'], all_gals['z'])

    is_isolated = conditional_spherical_isolation(sample1_pos, sample2_pos, presumed_rvir, period = model.mock.Lbox)

The boolean-valued array ``is_isolated`` is equal to ``True`` for those
galaxies in ``sm_selected_gals`` with zero other galaxies located within
a sphere of their presumed value of :math:`R_{\rm vir}.`

Example 2: :math:`M_{\ast}` dependent isolation criteria
----------------------------------------------------------

In this next example we'll show how to apply an isolation criterion that
depends on the stellar mass of the galaxies. This calculation, as well
as the remaining ones in this tutorial, will make use of Halotools
marking functions. The way this works is as follows. Every galaxy in
both ``sample1`` and ``sample2`` are given a "mark"; arrays storing
these marks are passed to the `~halotools.mock_observables.conditional_spherical_isolation`
function together with the normal arrays storing galaxy positions.
Additionally, you must select a "condition function", :math:`f;` the
condition function :math:`f` acceps a mark :math:`m_{1}` from a galaxy
in ``sample1`` and a mark :math:`m_{2}` from a galaxy in ``sample2`` and
returns a boolean. For each galaxy in ``sample1``, the
`~halotools.mock_observables.conditional_spherical_isolation` function searches ``sample2`` for
spatially nearby neighbors. A galaxy in ``sample2`` will only be
considered as a candidate neighbor if it lies within the ``r_max`` value
of the point in ``sample1`` *and* if the marking function
:math:`f(m_{1}, m_{2})` returns ``True``.

For example, suppose we define the conditional function to be
:math:`f(m_{1}, m_{2}) = {\rm True}` if :math:`m_{1} < m_{2}` and
``False`` otherwise, and suppose that for our marks we passed in the
stellar mass :math:`M_{\ast}` of each galaxy. When evaluating whether
some galaxy in ``sample1`` is isolated, what this choice for the
conditional function would do is to ignore all those galaxies in
``sample2`` that are less massive than the ``sample1`` galaxy under
consideration. So with this choice, the adopted definition of isolation
is whether or not a *more massive galaxy* resides within some search
radius. Let's see how to apply this isolation criterion to the galaxy
samples defined above.

For simplicity, we'll select a fixed ``r_max``, but you are free to
apply variable values of ``r_max`` together with the conditional
function formalism.

.. code:: python

    r_max = 0.5

    marks1 = sm_selected_gals['stellar_mass']
    marks2 = all_gals['stellar_mass']

Now we select the value of ``cond_func`` for the conditional function described above.
See the docstring of `~halotools.mock_observables.conditional_spherical_isolation`
for the function <==> function ID correspondence.

    cond_func = 2

    is_isolated = conditional_spherical_isolation(sample1_pos, sample2_pos, r_max,
                        marks1=marks1, marks2=marks2, cond_func=cond_func, period = model.mock.Lbox)

Example 3: Alternative :math:`M_{\ast}` dependent isolation criteria
----------------------------------------------------------------------

In this example, we'll do a similar calculation to the one above, except
we'll make a slight variation to the definition of isolation: a galaxy
in ``sample1`` will be said to be isolated if there are no ``sample2``
galaxies more massive than :math:`M_{\ast}` + *0.5dex* within 1
Mpc/h of the galaxy in ``sample1``.

For this calculation, we'll need to use ``cond_func`` = 6, which is
defined as
:math:`f(m^{a}_{1}, m^{b}_{1}, m^{a}_{2}, m^{b}_{2}) = {\rm True}` if
:math:`m^{a}_{1} < m^{a}_{2} + m^{b}_{1},` and ``False`` otherwise. For
:math:`m^{a}_{i}` we will pass in :math:`\log_{10}(M_{\ast}),` and for
:math:`m^{b}_{i}` we pass in :math:`0.5.` We will bundle the marks into
a multi-dimensional Numpy array using the same `numpy.vstack` method we
used to bundle our spatial positions into a *Npts x 3* array.

.. code:: python

    marks1 = np.vstack([np.log10(sm_selected_gals['stellar_mass']), np.zeros(len(sm_selected_gals))+0.5]).T
    marks2 = np.vstack([np.log10(all_gals['stellar_mass']), np.zeros(len(all_gals))+0.5]).T

    cond_func = 6

    is_isolated = conditional_spherical_isolation(sample1_pos, sample2_pos, r_max,
                        marks1=marks1, marks2=marks2, cond_func=cond_func, period = model.mock.Lbox)

Example 4: :math:`M_{\ast}` dependent isolation criteria in redshift-space
----------------------------------------------------------------------------

We will conclude this tutorial by putting together all of the features
of the `~halotools.mock_observables.isolation_functions` sub-package into a single,
observationally realistic example. We will demonstrate how to apply the
following isolation criteria on a mock galaxy sample:

Around each ``sample1`` galaxy with stellar mass :math:`M_{\ast}`, we
will draw a cylinder of radius :math:`2R_{\rm vir}` and length
:math:`3V_{\rm vir},` where :math:`R_{\rm vir}` and :math:`V_{\rm vir}`
are the virial radius and velocity inferred from the underlying
stellar-to-halo mass relation. In order for a ``sample1`` galaxy to be
isolated, there must be no other ``sample2`` galaxies more massive than
*0.5* dex within this cylinder.

We already computed :math:`R_{\rm vir}` above;
we will compute :math:`V_{\rm vir}` using the
`~halotools.mock_observables.halo_mass_to_virial_velocity` function:

.. code:: python

    from halotools.empirical_models import halo_mass_to_virial_velocity

    presumed_vvir = halo_mass_to_virial_velocity(presumed_mvir,
                    cosmology = model.mock.cosmology,
                    redshift = model.mock.redshift,
                    mdef = 'vir')

The units of ``presumed_vvir`` are in km/s, so we must convert these to
units of length. Recall that *h=1* and that all Halotools length-units
are in Mpc/h.

.. code:: python

    H0 = 100.0 # Hubble constant in h=1 units of km/s/Mpc
    pi_max = 3*presumed_vvir/H0
    rp_max = 2*presumed_rvir

Our marks and ``cond_func`` are the same as before, repeated below for convenience:

.. code:: python

    marks1 = np.vstack([np.log10(sm_selected_gals['stellar_mass']), np.zeros(len(sm_selected_gals))+0.5]).T
    marks2 = np.vstack([np.log10(all_gals['stellar_mass']), np.zeros(len(all_gals))+0.5]).T

    cond_func = 6

    from halotools.mock_observables import conditional_cylindrical_isolation

    is_isolated = conditional_cylindrical_isolation(sample1_pos, sample2_pos, rp_max, pi_max,
                        marks1=marks1, marks2=marks2, cond_func=cond_func, period = model.mock.Lbox)
