:orphan:

.. _subhalo_phase_space_model_tutorial:

************************************************************
Tutorial on Modeling HOD Satellites Using Subhalo Positions
************************************************************

This tutorial gives a detailed explanation of the
`~halotools.empirical_models.SubhaloPhaseSpace` class.
This class can be used to model the distribution of HOD satellites within their halos
by placing the satellites onto the locations of subhalos.

In any implementation of the HOD, there is no direct connection between the number of dark matter subhalos in the halo and the number of satellite galaxies in the halo. Instead, the number of satellites is determined by some parameterized analytical model for :math:`P(N_{\rm sat} | x_{\rm halo}, y_{\rm halo}, \dots, z_{\rm halo}).` The distribution :math:`P(N_{\rm sat})` is commonly assumed to be a Poisson distribution with a first moment determined by halo mass alone, :math:`\langle N_{\rm sat} | M_{\rm vir} \rangle.`

Similarly, the spatial position and velocities of satellites within their halos must also be specified by the HOD. The two most common choices for the intra-halo distributions are

1. An NFW profile with concentration either determined by
    * some analytical mean relation :math:`\bar{c}(M_{\rm vir})`
    * the actual concentration of the simulated halo.
2. A randomly selected dark matter particle within the simulated halo

The `~halotools.empirical_models.SubhaloPhaseSpace` class in Halotools allows a third option: in this class, satellites reside at the centers of subhalos, an empirical modeling feature typically limited to abundance matching models. This tutorial covers the many options associated with this class.


.. _general_comment_subhalo_phase_space:

General comment about using subhalos for satellite positions
================================================================

As shown in `Nagai and Kravtsov (2005) <https://arxiv.org/abs/astro-ph/0408273/>`_ and subsequent papers, subhalos selected according to different subhalo properties exhibit different radial distributions. Thus, depending on which subhalos you choose to place your satellites in, you will get different radial distributions of satellites, and also different velocity distributions. Most of the freedom built into the `~halotools.empirical_models.SubhaloPhaseSpace` class is designed around allowing you to explore different choices for what we will loosely refer to as the "subhalo selection function".


Choosing which subhalos are viable options for satellites
================================================================

Most Halotools-provided halo catalogs come with subhalos in addition to host halos (see :ref:`rockstar_subhalo_nomenclature` for details). All the Halotools catalogs with ``version_name = halotools_v0p4`` were made using a cut on :math:`M_{\rm peak}` at 300 particles: any subhalo which never had greater than 300 particles in its entire history is thrown out of the catalog. You can always place additional cuts by manipulating the ``halo_table`` of a halo catalog prior to populating a mock.
When using the `~halotools.empirical_models.SubhaloPhaseSpace` class,
you should be aware of any cut that was placed on the halo catalog you are using to populate satellites.

When throwing out subhalos based on any criteria, you change the subhalo selection function and influence the profiles of your satellite galaxies. If your science target is sensitive to the phase space occupied by satellite galaxies, it is generally a good idea to test whether your results are sensitive to any cuts that may have been placed on the subhalo catalog you used.

Choosing which subhalos are preferentially populated
================================================================

In any HOD model, the number of satellites in a given halo is determined by a Monte Carlo realization of :math:`P(N_{\rm sat} | {\rm halo}),` which has no necessary connection to the number of subhalos in the halo. Thus for models using `~halotools.empirical_models.SubhaloPhaseSpace`, it will be possible for host halos to have empty subhalos. The `~halotools.empirical_models.SubhaloPhaseSpace` class gives you freedom to choose which subhalos in each halo are populated first. This is accomplished in a pre-processing phase of the mock population. Within each host, you have the option to sort subhalos according to one of their properties; once :math:`N_{\rm sat}` is determined by the halo occupation model, the first :math:`N_{\rm sat}` subhalos in the list will be selected to host the satellites.

The order in which subhalos are selected is determined by a combination of the ``intra_halo_sorting_key`` and ``reverse_intra_halo_order`` keyword arguments to the `~halotools.empirical_models.SubhaloPhaseSpace` class. The default option is to have ``intra_halo_sorting_key = halo_mpeak`` with ``reverse_intra_halo_order = True``; for this choice, subhalos with the *largest* peak masses will be preferentially selected to host satellite galaxies. If the ``reverse_intra_halo_order`` were instead ``False``, then subhalos with the *smallest* peak masses will be selected first. The ``intra_halo_sorting_key`` can be used with any column appearing in the halo catalog.

Choosing how to deal with not having enough subhalos
================================================================

Because :math:`N_{\rm sat}` and :math:`N_{\rm sub}` are disconnected, it is possible that in some halos, your HOD model will require more satellite galaxies than the number of subhalos in the halo. For halo catalogs with Bolshoi-like resolution and galaxy samples with SDSS-like number densities, this does not happen often, roughly 1-2% of the time. To deal with this situation, the `~halotools.empirical_models.SubhaloPhaseSpace` class randomly selects a subhalo in some other host of a similar mass, and uses the host-centric distance of that subhalo for the host-centric distance of the satellite.

In a little more detail, during the pre-processing phase of mock population, host halos are initially binned according to the ``binning_key`` column of the halo catalog; by default, this binning is done on the ``halo_mvir_host_halo`` property. The bins themselves are defined by the ``host_haloprop_bins`` argument. Care must be taken by the user to ensure that 1. no bins are empty, 2. the bins encompass the range of the property spanned by the host halos in the catalog, and 3. the bins are sufficiently narrow so that the true host mass-dependence of the radial profiles is captured. Halotools will raise an exception if an unacceptable choice is made.


Inheriting additional subhalo properties besides position and velocity
========================================================================

For each satellite galaxy, once its parent subhalo is selected, the position and velocity of that subhalo are assigned to the satellite. As an additional modeling feature, you may choose to inherit additional subhalo properties besides just the phase space coordinates. For example, you may wish to also inherit the subhalo formation time to implement an age matching-like model, or the subhalo spin for morphology modeling.

The additional properties you choose to inherit are specified by the ``inherited_subhalo_props_dict`` argument. Each key of the ``inherited_subhalo_props_dict`` dictionary gives the name of a column in the ``subhalo_table`` that you wish to inherit. The value bound to each key is a tuple of two strings. The first string specifies the name you would like to give the inherited property in the ``galaxy_table``. The second string specifies the data type of the column, e.g., 'f4' or 'i8'.

This is most easily understood by looking at a specific example. The default dictionary is as follows:

.. code:: python

    default_inherited_subhalo_props_dict = (
        {'halo_id': ('halo_id', 'i8'),
        'halo_x': ('x', 'f8'),
        'halo_y': ('y', 'f8'),
        'halo_z': ('z', 'f8'),
        'halo_vx': ('vx', 'f8'),
        'halo_vy': ('vy', 'f8'),
        'halo_vz': ('vz', 'f8'),
        'halo_mpeak': ('halo_mpeak', 'f8')})

You can import this dictionary directly from the `~halotools.empirical_models` sub-package:

>>> from halotools.empirical_models import default_inherited_subhalo_props_dict

This way, you can supplement default_inherited_subhalo_props_dict with whatever additional properties you are interested in, and pass the result to ``inherited_subhalo_props_dict``. For example, if you wanted to inherit the spin of your subhalos:

>>> from halotools.empirical_models import default_inherited_subhalo_props_dict
>>> default_inherited_subhalo_props_dict['halo_spin'] = ('halo_spin', 'f8')

A worked example
=================

Let's suppose we'd like to start with the ``leauthaud11`` HOD model as our baseline; this model assumes satellites are distributed according to an NFW profile. What we'll do next is show how to build alternative models that are identical in every respect to ``leauthaud11``, except that satellites are placed on subhalos instead of tracing an NFW profile.

.. code:: python

    from halotools.empirical_models import PrebuiltHodModelFactory
    orig_model = PrebuiltHodModelFactory('leauthaud11')

    from halotools.empirical_models import SubhaloPhaseSpace
    alt_profile_model = SubhaloPhaseSpace('satellites', np.logspace(10.5, 15.2, 15))

    from halotools.empirical_models import HodModelFactory

    new_model_dictionary = orig_model.model_dictionary
    new_model_dictionary['satellites_profile'] = alt_profile_model
    new_model = HodModelFactory(**new_model_dictionary)

The ``new_model`` can be used to populate mock catalogs just like any Halotools composite model:

.. code:: python

    from halotools.sim_manager import CachedHaloCatalog
    halocat = CachedHaloCatalog(simname='bolplanck')
    new_model.populate_mock(halocat, seed=43)
    orig_model.populate_mock(halocat, seed=43)

Since we called `~halotools.empirical_models.HodModelFactory.populate_mock` by passing the ``seed`` keyword, all Monte Carlo functions have deterministic behavior and so the occupation statistics of these two realizations are identical:

.. code:: python

    print(len(new_model.mock.galaxy_table), len(orig_model.mock.galaxy_table))
    (49098, 49098)

In addition to the usual ``halo_table`` column names, when using `~halotools.empirical_models.SubhaloPhaseSpace` there is an an additional column called ``real_subhalo``. For satellites that sit on a subhalo that is within their actual host halo, this column is ``True``. Otherwise, this column is ``False`` (n.b., a peculiarity of the  implementation is that the ``real_subhalo`` column is also ``False`` for centrals). For reference, in this particular realization, only ~1.3% of satellites have this column equal to ``True``.









