.. _galaxy_catalog_analysis_tutorial6:

Galaxy Catalog Analysis Example: Mean infall velocity into cluster BCGs
==================================================================================================

In this example we'll show how to calculate the mean infall velocity of
galaxies towards the cluster BCGs.
In particular, we'll use the `~halotools.empirical_models.Behroozi10SmHm` model
to paint stellar masses onto subhalos, and then we'll select a
population of :math:`M_{\ast}/M_{\odot}>10^{11.75}` galaxies as our BCG sample,
and :math:`10^{10.75}<M_{\ast}/M_{\odot}<10^{11}` galaxies as the
population we'll use as tracers of the velocity field.

There is also an IPython Notebook in the following location that can be
used as a companion to the material in this section of the tutorial:


    **halotools/docs/notebooks/galcat_analysis/basic_examples/galaxy_catalog_analysis_tutorial6.ipynb**

By following this tutorial together with this notebook,
you can play around with your own variations of the calculation
as you learn the basic syntax.

Generate a mock galaxy catalog
------------------------------

Let's start out by generating a mock galaxy catalog into an N-body
simulation in the usual way. Here we'll assume you have the z=0 rockstar
halos for the multidark simulation, which we use to make sure we have enough BCGs.

.. code:: python

    from halotools.empirical_models import PrebuiltSubhaloModelFactory
    model = PrebuiltSubhaloModelFactory('smhm_binary_sfr')
    from halotools.sim_manager import CachedHaloCatalog
    halocat = CachedHaloCatalog(simname = 'multidark', redshift = 0, halo_finder = 'rockstar')
    model.populate_mock(halocat)

Our mock galaxies are stored in the ``galaxy_table`` of ``model.mock``
in the form of an Astropy Table.

Extract the position and velocity coordinates
---------------------------------------------
To calculate the mean radial velocity between two sets of points,
we need to know both their positions and velocities.
As described in :ref:`mock_obs_pos_formatting`,
functions in the `~halotools.mock_observables` package
such as `~halotools.mock_observables.mean_radial_velocity_vs_r` take array inputs in a
specific form: a (*Npts, 3)*-shape Numpy array. You can use the
`~halotools.mock_observables.return_xyz_formatted_array` convenience
function for this purpose, which we will do after first
selecting a tracer and a BCG population of mock galaxies.

.. code:: python

    from halotools.mock_observables import return_xyz_formatted_array

    cluster_central_mask = (model.mock.galaxy_table['stellar_mass'] > 10**11.5)
    cluster_centrals = model.mock.galaxy_table[cluster_central_mask]

    low_mass_tracers_mask = ((model.mock.galaxy_table['stellar_mass'] > 10**10) &
                             (model.mock.galaxy_table['stellar_mass'] < 10**10.5))
    low_mass_tracers = model.mock.galaxy_table[low_mass_tracers_mask]

    cluster_pos = return_xyz_formatted_array(cluster_centrals['x'],
        cluster_centrals['y'] ,cluster_centrals['z'])
    cluster_vel = return_xyz_formatted_array(cluster_centrals['vx'],
        cluster_centrals['vy'] ,cluster_centrals['vz'])

    low_mass_tracers_pos = return_xyz_formatted_array(low_mass_tracers['x'],
        low_mass_tracers['y'], low_mass_tracers['z'])
    low_mass_tracers_vel = return_xyz_formatted_array(low_mass_tracers['vx'],
        low_mass_tracers['vy'], low_mass_tracers['vz'])


Calculate :math:`<V_{\rm rad}>(r)`
----------------------------------

.. code:: python

    from halotools.mock_observables import mean_radial_velocity_vs_r

    rbins = np.logspace(-0.5, 1.25, 15)
    rbin_midpoints = (rbins[1:] + rbins[:-1])/2.

    vr_clusters = mean_radial_velocity_vs_r(cluster_pos, cluster_vel, rbins_absolute=rbins,
                        sample2=low_mass_tracers_pos, velocities2=low_mass_tracers_vel,
                        period = model.mock.Lbox, do_auto=False, do_cross=True)

Plot the result
~~~~~~~~~~~~~~~

.. code:: python

    fig, ax = plt.subplots(1, 1)

    __=ax.plot(rbin_midpoints, vr_clusters, color='k')
    xscale = ax.set_xscale('log')

    xlim = ax.set_xlim(xmin=0.5, xmax=20)

    xlabel = ax.set_xlabel(r'$r $  $\rm{[Mpc]}$', fontsize=15)
    ylabel = ax.set_ylabel(r'$\langle V_{\rm rad}\rangle$  $[{\rm km/s}]$', fontsize=15)
    title = ax.set_title('Radial infall velocity into cluster BCGs', fontsize=15)


.. image:: cluster_bcg_infall_velocity.png

As shown in the plot, as galaxies approach the neighborhood of a BCG,
on average they tend to fall towards it.
Spatial separations that are on the order of the halo radius of the BCG
correspond to the multi-stream region where the velocities of the
tracer galaxies start to virialize with the cluster halo. This is
why we see the upturn in the mean radial velocity on scales ~3 Mpc.


This tutorial continues with :ref:`galaxy_catalog_analysis_tutorial7`.






