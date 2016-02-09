:orphan:

.. _galaxy_catalog_analysis_tutorial2:

Example 2: Galaxy clustering in 3d
====================================

In this example, we'll show how to calculate the two-point clustering 
of a mock galaxy catalog, :math:`\xi_{\rm gg}(r)`. 
We'll also show how to compute cross-correlations between two different 
galaxy samples, and also the one-halo and two-halo decomposition 
:math:`\xi^{\rm 1h}_{\rm gg}(r)` and :math:`\xi^{\rm 2h}_{\rm gg}(r)`. 

Generate a mock galaxy catalog 
---------------------------------
Let's start out by generating a mock galaxy catalog into an N-body
simulation in the usual way. Here we'll assume you have the :math:`z=0`
rockstar halos for the bolshoi simulation, as this is the
default halo catalog. 

.. code:: python

    from halotools.empirical_models import PrebuiltSubhaloModelFactory
    model = PrebuiltSubhaloModelFactory('smhm_binary_sfr')
    model.populate_mock(simname = 'bolshoi', redshift = 0, halo_finder = 'rockstar')

Calculate two-point galaxy clustering :math:`\xi_{\rm gg}(r)`
-------------------------------------------------------------

The `~halotools.mock_observables` package takes array inputs in a 
specific form: a (*Npts, 3)*-shape Numpy array. There is a built-in 
method to all mock objects that returns the xyz-positions of your 
galaxies in the format used throughout `~halotools.mock_observables`:



Now we can calculate the clustering, as well as the one- and two-halo
decomposition

.. code:: python

    rbins = np.logspace(-1, 1.25, 15)
    rbin_centers = (rbins[1:] + rbins[:-1])/2.
    
    xi_all = mock_observables.tpcf(pos, rbins, 
                period = model.mock.Lbox, 
                num_threads = 'max')


The ``tpcf_one_two_halo_decomp`` function calculates the two-point
correlation function, decomposed into contributions from galaxies in the
same halo, and galaxies in different halos. In order to use this
function, we must provide an input array of host halo IDs that are equal
for galaxies occupying the same halo, and distinct for galaxies in
different halos. We'll use the ``halo_hostid`` column for this purpose,
and then select a galaxy sample with :math:`M_{\ast}>10^{10}M_{\odot}.`

.. code:: python

    add_halo_hostid(gals)
    
    sample_mask = model.mock.galaxy_table['stellar_mass'] > 1e10
    gals = model.mock.galaxy_table[sample_mask]
    pos = model.mock.xyz_positions(mask = sample_mask)

    xi_1h, xi_2h = mock_observables.tpcf_one_two_halo_decomp(pos,
                    gals['halo_hostid'].data, rbins, 
                    period = model.mock.Lbox, 
                    num_threads='max')

Plot the results
~~~~~~~~~~~~~~~~

.. code:: python

    plt.plot(rbin_centers, rbin_centers*rbin_centers*xi_all, 
             label='All galaxies')
    
    plt.plot(rbin_centers, rbin_centers*rbin_centers*xi_1h, 
             label = '1-halo term')
    plt.plot(rbin_centers, rbin_centers*rbin_centers*xi_2h, 
             label = '2-halo term')
    
    plt.xlim(xmin = 0.1, xmax = 20)
    plt.ylim(ymin = 1, ymax = 50)
    plt.loglog()
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(r'$r $  $\rm{[Mpc]}$', fontsize=25)
    plt.ylabel(r'$r^{2}\xi_{\rm gg}(r)$  $\rm{[Mpc^2]}$', fontsize=25)
    
    plt.legend(loc='best', fontsize=20)



.. parsed-literal::

    <matplotlib.legend.Legend at 0x116aced90>




.. image:: output_9_1.png







