:orphan:

.. _subhalo_phase_space_model_tutorial:

************************************************************
Tutorial on Modeling HOD Satellites Using Subhalo Positions
************************************************************

This tutorial gives a detailed explanation of
`~halotools.empirical_models.SubhaloPhaseSpace`.
This class can be used to model the distribution of HOD satellites within their halos
by placing the satellites onto the locations of subhalos.

In any implementation of the HOD, there is no direct connection between the number of dark matter subhalos in the halo and the number of model satellites in the halo. Instead, the number of satellites is determined by some parameterized analytical model for :math:`P(N_{\rm sat} | x_{\rm halo}, y_{\rm halo}, \dots, z_{\rm halo}).` The distribution :math:`P(N_{\rm sat})` is commonly assumed to be a Poisson distribution with a first moment determined by halo mass alone, :math:`\langle N_{\rm sat} | M_{\rm vir} \rangle.`

Similarly, the spatial position of satellites within their halos must also be specified by the HOD. The two most common choices for the intra-halo distributions are

1. An NFW profile with concentration either determined by
    * some analytical mean relation :math:`\bar{c}(M_{\rm vir})`
    * the actual concentration determined by a fit to the profile of the simulation halo.
2. A randomly selected dark matter particle within the halo in the simulation

The `~halotools.empirical_models.SubhaloPhaseSpace` class in Halotools allows a third option: in this class, satellites reside at the centers of subhalos, an empirical modeling feature typically limited to abundance matching models. This tutorial covers the many options associated with this class.


.. _general_comment_subhalo_phase_space:

General comment about using subhalos for satellite positions
================================================================

As shown in `Nagai and Kravtsov (2005) <https://arxiv.org/abs/astro-ph/0408273/>`_ and subsequent papers, subhalos selected according to different subhalo properties exhibit different radial distributions. Thus, depending on which subhalos you choose to place your satellites in, you will get different radial distributions of satellites, and also different velocity distributions. Most of the freedom built into the `~halotools.empirical_models.SubhaloPhaseSpace` class is designed around allowing you to explore different choices for what we will loosely refer to as the "subhalo selection function".


Choosing which subhalos are viable options for satellites
================================================================

All Halotools-provided halo catalogs come with subhalos. All the Halotools catalogs with ``version_name = halotools_v0p4`` were made using a cut on :math:`M_{\rm peak}` at 300 particles: any subhalo which never had greater than 300 particles in its entire history is thrown out of the catalog. You can always place additional cuts by masking out the ``halo_table`` of a halo catalog prior to populating a mock.
When using the `~halotools.empirical_models.SubhaloPhaseSpace` class,
you should be aware of any cut that was placed on the halo catalog you are using to populate satellites.

When throwing out subhalos based on any criteria, you introduce a non-trivial effect on the profiles of your satellite galaxies. If your science target of interest is sensitive to the phase space occupied by satellite galaxies, it is generally a good idea to test whether your results are sensitive to any cuts that may have been placed on the subhalo catalog you used.

Choosing which subhalos are preferentially populated
================================================================

As for any HOD model, the number of satellites in a given halo is determined by a Monte Carlo realization of :math:`P(N_{\rm sat} | {\rm halo}),` which has no necessary connection to the number of subhalos in the halo. Thus for models using `~halotools.empirical_models.SubhaloPhaseSpace`, it will be possible for host halos to have empty subhalos. The `~halotools.empirical_models.SubhaloPhaseSpace` class gives you freedom to choose which subhalos in each halo are populated first. This is accomplished in a pre-processing phase of the mock population. Within each host, you have the option to sort subhalos according to one of their properties; once :math:`N_{\rm sat}` is determined by the halo occupation model, the first :math:`N_{\rm sat}` subhalos in the list will be selected to host the satellites.

The order in which subhalos are selected is determined by a combination of the ``intra_halo_sorting_key`` and ``reverse_intra_halo_order`` keyword arguments to the `~halotools.empirical_models.SubhaloPhaseSpace` class. The default option is to have ``intra_halo_sorting_key = halo_mpeak`` with ``reverse_intra_halo_order = True``; for this choice, subhalos with the *largest* peak masses will be preferentially selected to host satellite galaxies. If the ``reverse_intra_halo_order`` were instead ``False``, then subhalos with the *smallest* peak masses will be selected first. The ``intra_halo_sorting_key`` can be used with any column appearing in the halo catalog.

Choosing how to deal with not having enough subhalos
================================================================

Because :math:`N_{\rm sat}` and :math:`N_{\rm sub}` are disconnected, it is possible that in one or more halos, your HOD model will require more satellite galaxies than the number of subhalos in the halo. For typical halo catalogs and galaxy samples with SDSS-like number densities, this does not happen often, roughly 1-2% of the time. To deal with this situation, the `~halotools.empirical_models.SubhaloPhaseSpace` class randomly selects a subhalo in some other host of a similar mass, and uses the host-centric distance of that subhalo for the host-centric distance of the satellite.

In a little more detail, during the pre-processing phase of mock population, host halos are initially binned according to the ``binning_key`` column of the halo catalog; by default, this binning is done on the ``halo_mvir_host_halo`` property. The bins themselves are defined by the ``host_haloprop_bins`` argument. Care must be taken by the user to ensure that no bins are empty, and also that the bins are sufficiently narrow so that the true mass-dependence of the radial profiles is captured. Halotools will raise an exception if an unacceptable choice is made.














